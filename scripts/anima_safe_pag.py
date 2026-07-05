"""Anima Safe PAG — standalone Perturbed Attention Guidance for Forge Neo.

A SELF-CONTAINED extension script (like the PiD / Anima-Ref PoC), completely
independent of the SAM3 machinery. It does not import anything from
``sam3ext`` and touches no Forge core file.

What it does
------------
A softer variant of Perturbed Attention Guidance (PAG) for Anima / Cosmos /
Predict2-style DiT models, ported from the ComfyUI node
``iljung1106/comfyui-anima-safe-pag``. It perturbs the self-attention of the
later transformer blocks to build a deliberately *weak* prediction, then
steers the CFG result *away* from it — improving local structure, line
confidence and small details.

How it hooks into Forge Neo (no core edits)
-------------------------------------------
Forge Neo's real sampling path (``backend/sampling/sampling_function.py``)
does NOT invoke ``sampler_calc_cond_batch_function`` (the setter exists on the
patcher but the sampler calls ``calc_cond_uncond_batch`` directly). The two
hooks it DOES honour are:

  * ``model_options["model_function_wrapper"]`` — wraps ``model.apply_model``
    with ``{"input","timestep","c","cond_or_uncond"}`` (proven by the PoC).
  * ``post_cfg_function(args)`` — args carry ``denoised`` (= cfg_result),
    ``cond_denoised``, ``uncond_denoised``, ``sigma``, ``input`` …

So we:

  1. In the model-function wrapper, append a COPY of the *cond* rows to the
     batch and run a SINGLE ``apply_model`` (no separate forward → the extra
     PAG prediction rides along in the same kernel launch). Self-attention on
     the appended rows is perturbed (value-only lerp). We return the original
     rows untouched, so CFG proceeds exactly as normal, and stash the pag raw
     output.
  2. In ``post_cfg_function`` we recover the model's denoise scale ``c_out``
     empirically from ``(cond_denoised-uncond_denoised)`` vs the raw cond/uncond
     outputs, convert the pag prediction into denoised (x0) space *exactly*
     (parameterization-agnostic), and apply
     ``result = cfg_result + scale*(cond_denoised - pag_denoised)`` followed by
     the std-matching rescale.

Attention perturbation
----------------------
``backend/nn/anima.py`` computes self-attention as a bare
``scaled_dot_product_attention(q, k, v)`` module-global call inside
``block.self_attn`` (``TransformerBlock``). We wrap that module global so, for
the appended pag rows only, the attention output becomes
``lerp(sdpa(q,k,v), v, strength)`` — the value-only ("identity attention")
path. Value carries no RoPE, so this is rotary-safe. On grouped-query models
(``v`` head count != output head count) the shapes differ and we skip the
perturbation rather than corrupt the tensor.

Safety
------
Everything is wrapped in try/except and ALWAYS falls back to the normal
``apply_model`` / ``denoised`` on any error, so enabling this can never break a
generation — worst case it logs and renders normally.

⚠️ 실험 기능 — 정적으로는 Forge Neo `neo` 브랜치 소스(sampling_function.py /
patcher/base.py / nn/anima.py)와 일치하도록 작성했으나, 실제 Anima 체크포인트로
end-to-end 런타임 검증이 1회 필요합니다. 콘솔의 ``[AnimaSafePAG]`` 로그로 경로가
살아있는지 확인하세요.
"""
from __future__ import annotations

import sys
import traceback
from functools import partial

import gradio as gr

from modules import script_callbacks, scripts

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _log(msg: str) -> None:
    print(f"[AnimaSafePAG] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Shared runtime state (single-threaded per generation; the sampler drives one
# denoise step at a time — wrapper then post_cfg — so a plain dict is enough).
# ---------------------------------------------------------------------------

_STATE: dict = {
    "on": False,          # PAG active for the current generation
    "scale": 4.0,         # guidance strength
    "strength": 0.75,     # attention perturbation blend (normal ↔ value-only)
    "rescale": 0.20,      # std-matching rescale factor
    "targets": set(),     # block indices to perturb
    "start": 0.0,         # start percent of sampling
    "end": 0.7,           # end percent of sampling
    "total": 20,          # total steps this pass
    "step": 0,            # current step counter
    "b0": None,           # first index of the appended pag rows (or None)
    "active": 0,          # >0 while inside a targeted self_attn.forward
    "pag_raw": None,      # stashed pag model-output for this step
    "cond_raw": None,     # stashed normal cond model-output
    "uncond_raw": None,   # stashed normal uncond model-output
    "gqa_warned": False,  # emit the GQA-skip note only once
    "apg_autooff_rescale": True,  # skip PAG rescale while APG is on (toggleable)
}


# ---------------------------------------------------------------------------
# APG (Adaptive Projected Guidance) — a separate, model-agnostic post-CFG
# guidance variant. It REPLACES the CFG-combine result with a projected one so
# high guidance no longer oversaturates. Runs on any engine (not just Anima);
# unlike PAG it needs no attention patching, only the post_cfg args.
#
# With eta=1, norm_threshold=0, momentum=0 it reduces EXACTLY to standard CFG,
# so it is safe as a default-on-but-neutral toggle.
# ---------------------------------------------------------------------------

_APG: dict = {
    "on": False,
    "eta": 0.0,           # weight of the cond-parallel guidance component
    "norm_threshold": 15.0,  # clamp guidance L2 norm to this (0 = disabled)
    "momentum": 0.0,      # running-average coefficient across steps
    "avg": None,          # momentum buffer (per-generation)
    "last_sigma": None,   # to detect a new sampling pass (sigma goes up → reset)
}


# ---------------------------------------------------------------------------
# Attention perturbation — patch the anima module-global SDPA once (idempotent).
# ---------------------------------------------------------------------------


def _patched_sdpa(query, key, value, *args, **kwargs):
    """Drop-in for ``backend.nn.anima.scaled_dot_product_attention``.

    Runs the real attention, then — only while inside a *targeted* self_attn
    forward (``active>0``) and only for the appended pag rows (``b0``) — blends
    the output toward the raw value path (identity attention). This is the
    "soft PAG" perturbation.
    """
    orig = _STATE.get("_orig_sdpa")
    out = orig(query, key, value, *args, **kwargs)
    try:
        b0 = _STATE["b0"]
        if _STATE["active"] > 0 and b0 is not None and b0 < out.shape[0]:
            if value.shape == out.shape:
                out = out.clone()
                out[b0:] = torch.lerp(out[b0:], value[b0:], float(_STATE["strength"]))
            elif not _STATE["gqa_warned"]:
                _STATE["gqa_warned"] = True
                _log(
                    "self-attn value/out head layout differ (grouped-query?) — "
                    "skipping perturbation on this block; PAG will be weaker."
                )
    except Exception as e:  # never let the patch break sampling
        _log(f"sdpa perturb skipped: {type(e).__name__}: {e}")
    return out


def _make_selfattn_wrapper(idx: int, orig_forward):
    """Wrap a block's ``self_attn.forward`` to raise the ``active`` flag only
    for the configured target blocks, and only during the appended pag forward
    (``b0`` set)."""

    def _wrapped(*args, **kwargs):
        if _STATE["b0"] is not None and idx in _STATE["targets"]:
            _STATE["active"] += 1
            try:
                return orig_forward(*args, **kwargs)
            finally:
                _STATE["active"] -= 1
        return orig_forward(*args, **kwargs)

    return _wrapped


def _ensure_patched(diffusion_model) -> int:
    """Idempotently install the SDPA patch + per-block self_attn wrappers.
    Returns the number of transformer blocks (0 if the structure is unexpected).
    """
    try:
        from backend.nn import anima as anima_mod
    except Exception as e:
        _log(f"cannot import backend.nn.anima: {type(e).__name__}: {e}")
        return 0

    if not hasattr(anima_mod, "scaled_dot_product_attention"):
        _log("backend.nn.anima has no module-global scaled_dot_product_attention "
             "— cannot perturb attention on this build. PAG disabled.")
        return 0

    if not getattr(anima_mod, "_pag_sdpa_patched", False):
        _STATE["_orig_sdpa"] = anima_mod.scaled_dot_product_attention
        anima_mod.scaled_dot_product_attention = _patched_sdpa
        anima_mod._pag_sdpa_patched = True
        _log("patched backend.nn.anima.scaled_dot_product_attention ✅")

    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        _log("diffusion_model has no .blocks — unexpected Anima structure.")
        return 0

    wrapped = 0
    for idx, block in enumerate(blocks):
        sa = getattr(block, "self_attn", None)
        if sa is None:
            continue
        if getattr(sa, "_pag_wrapped", False):
            continue
        try:
            sa._pag_orig_forward = sa.forward
            sa.forward = _make_selfattn_wrapper(idx, sa._pag_orig_forward)
            sa._pag_wrapped = True
            wrapped += 1
        except Exception as e:
            _log(f"failed to wrap block {idx} self_attn: {type(e).__name__}: {e}")
    if wrapped:
        _log(f"wrapped {wrapped} self_attn block(s)")
    return len(blocks)


# ---------------------------------------------------------------------------
# Conditioning-batch helpers
# ---------------------------------------------------------------------------


def _extend_c(c: dict, idx_tensor, batch: int) -> dict:
    """Append the pag (cond) rows to every batched tensor in the conditioning
    dict. Non-tensor entries (e.g. ``transformer_options``) pass through."""
    out = {}
    for k, v in c.items():
        if k == "transformer_options":
            out[k] = v
            continue
        if torch.is_tensor(v) and v.shape[0] == batch:
            out[k] = torch.cat([v, v.index_select(0, idx_tensor)], dim=0)
        else:
            out[k] = v
    return out


def _percent_in_range() -> bool:
    total = max(1, int(_STATE["total"]))
    pct = _STATE["step"] / total
    return _STATE["start"] <= pct <= _STATE["end"]


# ---------------------------------------------------------------------------
# The two Forge-honoured hooks
# ---------------------------------------------------------------------------


def _model_wrapper(apply_model, w):
    """model_function_wrapper: run one enlarged ``apply_model`` that also
    produces the perturbed (pag) prediction for the cond rows."""
    x = w.get("input")
    ts = w.get("timestep")
    c = w.get("c") or {}
    cou = w.get("cond_or_uncond")

    _STATE["pag_raw"] = None
    if not _STATE["on"] or torch is None:
        return apply_model(x, ts, **c)

    _STATE["step"] += 1
    try:
        # Only the standard 2-group (cond + uncond) batching is supported;
        # anything else (no CFG, composition, unknown layout) → plain path.
        if cou is None or len(cou) != 2:
            return apply_model(x, ts, **c)
        if not _percent_in_range():
            return apply_model(x, ts, **c)
        # ControlNet etc. inject batched tensors we can't safely widen.
        if c.get("control") is not None:
            return apply_model(x, ts, **c)

        batch = x.shape[0]
        chunk = batch // len(cou)
        cond_idx, uncond_idx = [], []
        for i, marker in enumerate(cou):
            rng = list(range(i * chunk, (i + 1) * chunk))
            (cond_idx if int(marker) == 0 else uncond_idx).extend(rng)
        if not cond_idx or not uncond_idx:
            return apply_model(x, ts, **c)

        idx = torch.tensor(cond_idx, device=x.device, dtype=torch.long)
        x_ext = torch.cat([x, x.index_select(0, idx)], dim=0)
        ts_ext = torch.cat([ts, ts.index_select(0, idx)], dim=0)
        c_ext = _extend_c(c, idx, batch)

        _STATE["b0"] = batch
        try:
            out_ext = apply_model(x_ext, ts_ext, **c_ext)
        finally:
            _STATE["b0"] = None

        out = out_ext[:batch]
        uidx = torch.tensor(uncond_idx, device=x.device, dtype=torch.long)
        _STATE["pag_raw"] = out_ext[batch:].detach().float()
        _STATE["cond_raw"] = out.index_select(0, idx).detach().float()
        _STATE["uncond_raw"] = out.index_select(0, uidx).detach().float()
        return out
    except Exception as e:
        _STATE["b0"] = None
        _STATE["pag_raw"] = None
        _log(f"wrapper fallback → normal apply_model: {type(e).__name__}: {e}")
        return apply_model(x, ts, **c)


def _project(v0, v1):
    """Decompose ``v0`` into (parallel, orthogonal) components relative to the
    direction of ``v1``. Per-sample (reduce over every dim except batch)."""
    dims = list(range(1, v1.ndim))
    v1n = torch.nn.functional.normalize(v1, dim=dims)
    parallel = (v0 * v1n).sum(dim=dims, keepdim=True) * v1n
    orthogonal = v0 - parallel
    return parallel, orthogonal


def _apply_apg(args, cond_scale):
    """Adaptive Projected Guidance — returns a replacement CFG-combine result
    in denoised space. With eta=1, norm_threshold=0, momentum=0 this is exactly
    standard CFG. Falls back to None (→ caller keeps the original denoised) on
    any problem."""
    try:
        cond = args["cond_denoised"].float()
        uncond = args["uncond_denoised"].float()
        guidance = cond - uncond

        # Momentum: running average across steps; reset on a new pass (sigma up).
        mom = float(_APG["momentum"])
        if mom != 0.0:
            sigma = args.get("sigma")
            cur = float(sigma.flatten()[0]) if sigma is not None else None
            last = _APG["last_sigma"]
            avg = _APG["avg"]
            if (avg is None or avg.shape != guidance.shape
                    or (cur is not None and last is not None and cur > last + 1e-6)):
                avg = torch.zeros_like(guidance)
            avg = mom * avg + guidance
            _APG["avg"] = avg
            _APG["last_sigma"] = cur
            guidance = avg

        # Norm clamp: keep the guidance vector under a fixed L2 magnitude.
        nt = float(_APG["norm_threshold"])
        if nt > 0.0:
            dims = list(range(1, guidance.ndim))
            gnorm = guidance.norm(p=2, dim=dims, keepdim=True)
            scale = torch.clamp(nt / gnorm.clamp_min(1e-8), max=1.0)
            guidance = guidance * scale

        # Project onto cond, downweight the parallel (saturating) component.
        parallel, orthogonal = _project(guidance, cond)
        modified = orthogonal + float(_APG["eta"]) * parallel

        # uncond + w*modified  →  standard CFG when modified == (cond-uncond).
        return uncond + float(cond_scale) * modified
    except Exception as e:
        _log(f"APG skipped: {type(e).__name__}: {e}")
        return None


def _apply_pag(args, base):
    """Add the PAG guidance term onto ``base`` (denoised space), then rescale.
    Uses the empirically-recovered ``c_out`` so it holds for eps / v / flow
    parameterizations alike. Returns ``base`` unchanged on any problem."""
    cd = args["cond_denoised"].float()
    ud = args["uncond_denoised"].float()
    cond_raw = _STATE["cond_raw"]
    uncond_raw = _STATE["uncond_raw"]
    pag_raw = _STATE["pag_raw"]
    if cond_raw is None or uncond_raw is None:
        return base
    if cd.shape != cond_raw.shape or pag_raw.shape != cond_raw.shape:
        return base

    # (cd-ud) = c_out*(cond_raw-uncond_raw). Recover c_out by least squares.
    do = cond_raw - uncond_raw
    dd = cd - ud
    denom = (do * do).sum()
    if float(denom) <= 1e-8:
        return base
    c_out = (dd * do).sum() / denom

    # cond_denoised - pag_denoised = c_out*(cond_raw - pag_raw)
    guidance = float(_STATE["scale"]) * c_out * (cond_raw - pag_raw)
    result = base + guidance

    # Rescale — auto-skipped while APG is on (APG already governs magnitude),
    # unless the user turned that guard off.
    r = float(_STATE["rescale"])
    apg_governs = _APG["on"] and _STATE.get("apg_autooff_rescale", True)
    if r > 0 and not apg_governs:
        dims = list(range(1, result.ndim))
        std_c = cd.std(dim=dims, keepdim=True).clamp_min(1e-6)
        std_r = result.std(dim=dims, keepdim=True).clamp_min(1e-6)
        factor = r * (std_c / std_r) + (1.0 - r)
        result = result * factor
    return result


def _post_cfg(args):
    """post_cfg orchestrator. Order: APG recomputes the CFG-combine base (if on),
    then PAG guidance is layered on top (if on). Because PAG adds its term at
    post_cfg via a recovered ``c_out``, it composes on top of ANY base — standard
    CFG, APG, or a built-in like MaHiRo/RescaleCFG.
    """
    denoised = args["denoised"]
    if torch is None:
        return denoised
    try:
        result = denoised.float()

        if _APG["on"]:
            apg = _apply_apg(args, args.get("cond_scale", 1.0))
            if apg is not None:
                result = apg

        if _STATE["on"] and _STATE["pag_raw"] is not None:
            result = _apply_pag(args, result)

        return result.to(denoised.dtype)
    except Exception as e:
        _log(f"post_cfg fallback → unmodified cfg: {type(e).__name__}: {e}")
        return denoised
    finally:
        _STATE["pag_raw"] = None


# ---------------------------------------------------------------------------
# Block-index parsing
# ---------------------------------------------------------------------------


def _parse_blocks(spec: str, n: int) -> set:
    """Parse "18", "14-27", "14,16,18" → a set of valid indices. Empty spec →
    auto: the later half (node recommends >14 on a 28-block Anima)."""
    spec = (spec or "").strip()
    if not spec:
        return set(range(n // 2, n))
    out: set = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                out.update(range(int(a), int(b) + 1))
            except ValueError:
                pass
        else:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return {i for i in out if 0 <= i < n}


def _get_diffusion_model(unet):
    model = getattr(unet, "model", None)
    for attr in ("diffusion_model",):
        dm = getattr(model, attr, None)
        if dm is not None:
            return dm
    return None


# ---------------------------------------------------------------------------
# XYZ plot integration — lets users grid an ON/OFF (and param) comparison.
# The AxisOption apply functions mutate ``p`` per grid cell; the script's
# ``process_before_every_sampling`` then reads ``p._anima_safe_pag_xyz`` and
# overrides its own UI args from it (same pattern as the SAM3 script).
# ---------------------------------------------------------------------------


def _pag_xyz_set(p, x, xs, *, field: str):
    if not hasattr(p, "_anima_safe_pag_xyz"):
        p._anima_safe_pag_xyz = {}
    p._anima_safe_pag_xyz[field] = x


def _make_pag_xyz_axis() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return

    bool_choices = lambda: ["True", "False"]  # noqa: E731

    axis = [
        # The headline request: ON/OFF comparison grid.
        xyz_grid.AxisOption(
            "[Anima PAG] Enable", str,
            partial(_pag_xyz_set, field="enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Anima PAG] Scale", float, partial(_pag_xyz_set, field="scale")),
        xyz_grid.AxisOption(
            "[Anima PAG] Perturbation Strength", float,
            partial(_pag_xyz_set, field="strength"),
        ),
        xyz_grid.AxisOption(
            "[Anima PAG] Block Indices", str,
            partial(_pag_xyz_set, field="blocks"),
        ),
        xyz_grid.AxisOption("[Anima PAG] Start Percent", float, partial(_pag_xyz_set, field="start")),
        xyz_grid.AxisOption("[Anima PAG] End Percent", float, partial(_pag_xyz_set, field="end")),
        xyz_grid.AxisOption("[Anima PAG] Rescale", float, partial(_pag_xyz_set, field="rescale")),
        # APG axes (ON/OFF + the three knobs) for the same style of comparison.
        xyz_grid.AxisOption(
            "[Anima APG] Enable", str,
            partial(_pag_xyz_set, field="apg_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Anima APG] Eta", float, partial(_pag_xyz_set, field="apg_eta")),
        xyz_grid.AxisOption("[Anima APG] Norm Threshold", float, partial(_pag_xyz_set, field="apg_norm")),
        xyz_grid.AxisOption("[Anima APG] Momentum", float, partial(_pag_xyz_set, field="apg_momentum")),
    ]

    if not any(a.label.startswith(("[Anima PAG]", "[Anima APG]")) for a in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def _pag_on_before_ui() -> None:
    try:
        _make_pag_xyz_axis()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_pag_on_before_ui)


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


# ---------------------------------------------------------------------------
# The extension script
# ---------------------------------------------------------------------------


class AnimaSafePAG(scripts.Script):
    # Run late so any other script has set up the unet first (mirrors the PoC).
    sorting_priority = 98

    def title(self):
        return "Anima Safe PAG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima Safe PAG (Perturbed Attention Guidance)", open=False):
            gr.Markdown(
                "Anima/Cosmos/Predict2 계열 **DiT 전용** soft-PAG. 후반 self-attention을 "
                "부드럽게 흐린 *약한 예측*을 만들어 CFG 결과를 그 반대로 밀어 구조·디테일을 "
                "개선합니다. **Anima 엔진이 로드된 경우에만** 동작하며, 그 외 모델·오류 시 "
                "일반 생성으로 폴백합니다. PAG는 원리상 cond 예측을 한 번 더 계산하지만 "
                "여기서는 별도 forward 없이 **같은 배치에 접어** 오버헤드를 최소화합니다."
            )
            enabled = gr.Checkbox(
                label="Enable Anima Safe PAG",
                value=False,
                elem_id="anima_safe_pag_enable",
            )
            scale = gr.Slider(
                label="PAG scale (guidance strength)",
                minimum=0.0, maximum=15.0, step=0.1, value=4.0,
                elem_id="anima_safe_pag_scale",
            )
            strength = gr.Slider(
                label="Perturbation strength (normal ↔ value-only)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.75,
                elem_id="anima_safe_pag_strength",
            )
            block_indices = gr.Textbox(
                label="Block indices (빈칸=후반 절반 자동, 예: 18 / 14-27 / 14,16,18)",
                value="",
                elem_id="anima_safe_pag_blocks",
            )
            with gr.Row():
                start_percent = gr.Slider(
                    label="Start percent",
                    minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                    elem_id="anima_safe_pag_start",
                )
                end_percent = gr.Slider(
                    label="End percent",
                    minimum=0.0, maximum=1.0, step=0.01, value=0.7,
                    elem_id="anima_safe_pag_end",
                )
            rescale = gr.Slider(
                label="Rescale (대비/채도 과다 억제)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.20,
                elem_id="anima_safe_pag_rescale",
            )

            gr.Markdown("---\n### APG (Adaptive Projected Guidance)")
            gr.Markdown(
                "높은 CFG의 **과채도·번짐을 만드는 성분만 골라 억제**해 guidance를 세게 "
                "밀어도 자연스럽게 유지합니다. RescaleCFG의 상위호환이며 **추가 forward "
                "없이** CFG 합성 지점에서 계산만 바꿉니다. PAG와 독립이라 같이 켜도 됩니다 "
                "(APG가 베이스를 만들고 PAG가 그 위에 구조를 더함). Anima 외 모델에서도 동작."
            )
            apg_enabled = gr.Checkbox(
                label="Enable APG",
                value=False,
                elem_id="anima_safe_pag_apg_enable",
            )
            apg_autooff = gr.Checkbox(
                label="APG 켜지면 PAG rescale 자동 끄기 (이중 크기보정 방지)",
                value=True,
                elem_id="anima_safe_pag_apg_autooff",
            )
            with gr.Accordion("APG Advanced (세부값)", open=False):
                gr.Markdown(
                    "기본값이면 무난합니다. 파고들 때: **eta**=평행(과채도) 성분 비중 "
                    "(0=최대 억제, 1=표준 CFG로 환원), **norm threshold**=guidance 크기 "
                    "상한(0=off), **momentum**=스텝 간 running-average(음수 권장, 0=off). "
                    "guidance 세기는 메인 **CFG Scale** 슬라이더를 그대로 씁니다."
                )
                apg_eta = gr.Slider(
                    label="APG eta (평행 성분 비중 · 낮을수록 과채도↓)",
                    minimum=-10.0, maximum=10.0, step=0.05, value=0.0,
                    elem_id="anima_safe_pag_apg_eta",
                )
                apg_norm = gr.Slider(
                    label="APG norm threshold (guidance 크기 상한 · 0=off)",
                    minimum=0.0, maximum=50.0, step=0.5, value=15.0,
                    elem_id="anima_safe_pag_apg_norm",
                )
                apg_momentum = gr.Slider(
                    label="APG momentum (스텝 간 running-average · 음수 권장 · 0=off)",
                    minimum=-1.0, maximum=1.0, step=0.05, value=0.0,
                    elem_id="anima_safe_pag_apg_momentum",
                )
        return [
            enabled, scale, strength, block_indices, start_percent, end_percent, rescale,
            apg_enabled, apg_eta, apg_norm, apg_momentum, apg_autooff,
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if torch is None:
            return

        # XYZ-plot overrides (set per grid cell by the AxisOption apply fns).
        xyz = getattr(p, "_anima_safe_pag_xyz", {}) or {}

        def _xyz_num(key, cur):
            if key in xyz:
                try:
                    return float(xyz[key])
                except (TypeError, ValueError):
                    return cur
            return cur

        def _arg(i, default):
            return args[i] if len(args) > i else default

        # ---- PAG enable (checkbox, XYZ can override for ON/OFF grids) ----
        try:
            pag_enabled = bool(_arg(0, False))
        except Exception:
            pag_enabled = False
        if "enabled" in xyz:
            pag_enabled = _as_bool(xyz["enabled"], pag_enabled)

        # ---- APG enable ----
        try:
            apg_enabled = bool(_arg(7, False))
        except Exception:
            apg_enabled = False
        if "apg_enabled" in xyz:
            apg_enabled = _as_bool(xyz["apg_enabled"], apg_enabled)

        # Nothing to do → make sure neither leaks into this generation.
        if not pag_enabled and not apg_enabled:
            _STATE["on"] = False
            _APG["on"] = False
            return

        # ---- Read every knob (with XYZ overrides) ----
        try:
            scale = _xyz_num("scale", float(_arg(1, 4.0)))
            strength = _xyz_num("strength", float(_arg(2, 0.75)))
            block_spec = str(xyz["blocks"]) if "blocks" in xyz else str(_arg(3, ""))
            start = _xyz_num("start", float(_arg(4, 0.0)))
            end = _xyz_num("end", float(_arg(5, 0.7)))
            rescale = _xyz_num("rescale", float(_arg(6, 0.20)))
            apg_eta = _xyz_num("apg_eta", float(_arg(8, 0.0)))
            apg_norm = _xyz_num("apg_norm", float(_arg(9, 15.0)))
            apg_momentum = _xyz_num("apg_momentum", float(_arg(10, 0.0)))
            apg_autooff = _as_bool(_arg(11, True), True)
        except Exception as e:
            _STATE["on"] = False
            _APG["on"] = False
            _log(f"bad args, disabling: {type(e).__name__}: {e}")
            return

        sd_model = getattr(p, "sd_model", None)
        engine = type(sd_model).__name__ if sd_model is not None else "?"
        forge_objects = getattr(sd_model, "forge_objects", None)
        unet = getattr(forge_objects, "unet", None)
        if unet is None:
            _STATE["on"] = False
            _APG["on"] = False
            _log("no forge_objects.unet — cannot attach guidance.")
            return

        # ---- APG: model-agnostic, needs no patching. Reset momentum buffer. ----
        if apg_enabled:
            _APG.update(
                on=True, eta=apg_eta, norm_threshold=apg_norm,
                momentum=apg_momentum, avg=None, last_sigma=None,
            )
        else:
            _APG["on"] = False
        _STATE["apg_autooff_rescale"] = apg_autooff

        # ---- PAG: Anima-only, needs the attention patch + a diffusion_model. ----
        pag_ok = False
        targets: set = set()
        nblocks = 0
        if pag_enabled:
            if engine != "Anima":
                _log(f"engine={engine} (not 'Anima') — PAG skipped (APG still runs "
                     "if enabled).")
            else:
                dm = _get_diffusion_model(unet)
                if dm is None:
                    _log("could not resolve diffusion_model — PAG skipped.")
                else:
                    nblocks = _ensure_patched(dm)
                    if nblocks:
                        targets = _parse_blocks(block_spec, nblocks)
                        if targets:
                            pag_ok = True
                        else:
                            _log("no valid target blocks — PAG skipped.")

        if pag_ok:
            _STATE.update(
                on=True, scale=scale, strength=strength, rescale=rescale,
                targets=targets, start=min(start, end), end=max(start, end),
                total=int(getattr(p, "steps", 20) or 20),
                step=0, b0=None, active=0, pag_raw=None,
            )
        else:
            _STATE["on"] = False

        if not _STATE["on"] and not _APG["on"]:
            return

        # ---- Attach hooks (clone so Forge core / other gens are untouched) ----
        try:
            unet = unet.clone()
            if _STATE["on"]:
                unet.set_model_unet_function_wrapper(_model_wrapper)
            unet.set_model_sampler_post_cfg_function(_post_cfg)
            p.sd_model.forge_objects.unet = unet

            if not hasattr(p, "extra_generation_params"):
                p.extra_generation_params = {}
            if _STATE["on"]:
                p.extra_generation_params["Anima Safe PAG"] = (
                    f"scale={scale}, strength={strength}, blocks={sorted(targets)}, "
                    f"range={_STATE['start']:.2f}-{_STATE['end']:.2f}, rescale={rescale}"
                )
            if _APG["on"]:
                p.extra_generation_params["Anima APG"] = (
                    f"eta={apg_eta}, norm={apg_norm}, momentum={apg_momentum}"
                )
            _log(
                f"attached ✅ engine={engine} "
                f"PAG={'on' if _STATE['on'] else 'off'} "
                f"(blocks={sorted(targets)} scale={scale} strength={strength} "
                f"rescale={'auto-off' if (_APG['on'] and apg_autooff) else rescale}) "
                f"APG={'on' if _APG['on'] else 'off'} "
                f"(eta={apg_eta} norm={apg_norm} mom={apg_momentum})"
            )
        except Exception as e:
            _STATE["on"] = False
            _APG["on"] = False
            _log(f"failed to attach hooks: {type(e).__name__}: {e}")

    def postprocess(self, p, processed, *args):
        # Belt-and-suspenders: make sure guidance doesn't leak into a later
        # generation if forge_objects reuse ever changed.
        _STATE["on"] = False
        _APG["on"] = False
