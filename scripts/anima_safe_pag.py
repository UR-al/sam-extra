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

import gradio as gr

from modules import scripts

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


def _post_cfg(args):
    """post_cfg_function: steer the CFG result away from the pag prediction,
    then rescale. All maths in denoised (x0) space via empirically-recovered
    ``c_out`` so it holds for eps / v / flow-matching parameterizations alike.
    """
    denoised = args["denoised"]
    if _STATE["pag_raw"] is None or not _STATE["on"] or torch is None:
        return denoised
    try:
        cd = args["cond_denoised"].float()
        ud = args["uncond_denoised"].float()
        cond_raw = _STATE["cond_raw"]
        uncond_raw = _STATE["uncond_raw"]
        pag_raw = _STATE["pag_raw"]
        if cond_raw is None or uncond_raw is None:
            return denoised
        if cd.shape != cond_raw.shape or pag_raw.shape != cond_raw.shape:
            return denoised

        # denoised = c_skip*x + c_out*model_out (c_skip, c_out scalar per step).
        # → (cd-ud) = c_out*(cond_raw-uncond_raw). Recover c_out by least squares.
        do = cond_raw - uncond_raw
        dd = cd - ud
        denom = (do * do).sum()
        if float(denom) <= 1e-8:
            return denoised
        c_out = (dd * do).sum() / denom

        # cond_denoised - pag_denoised = c_out*(cond_raw - pag_raw)
        guidance = float(_STATE["scale"]) * c_out * (cond_raw - pag_raw)
        result = denoised.float() + guidance

        r = float(_STATE["rescale"])
        if r > 0:
            dims = list(range(1, result.ndim))
            std_c = cd.std(dim=dims, keepdim=True).clamp_min(1e-6)
            std_r = result.std(dim=dims, keepdim=True).clamp_min(1e-6)
            factor = r * (std_c / std_r) + (1.0 - r)
            result = result * factor
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
        return [enabled, scale, strength, block_indices, start_percent, end_percent, rescale]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if torch is None:
            return
        try:
            enabled = bool(args[0]) if len(args) > 0 else False
        except Exception:
            enabled = False

        if not enabled:
            _STATE["on"] = False
            return

        try:
            scale = float(args[1]); strength = float(args[2])
            block_spec = str(args[3]); start = float(args[4])
            end = float(args[5]); rescale = float(args[6])
        except Exception as e:
            _STATE["on"] = False
            _log(f"bad args, disabling: {type(e).__name__}: {e}")
            return

        sd_model = getattr(p, "sd_model", None)
        engine = type(sd_model).__name__ if sd_model is not None else "?"
        if engine != "Anima":
            _STATE["on"] = False
            _log(f"engine={engine} (not 'Anima') — Safe PAG only supports Anima/"
                 "Cosmos DiT. Skipping.")
            return

        forge_objects = getattr(sd_model, "forge_objects", None)
        unet = getattr(forge_objects, "unet", None)
        if unet is None:
            _STATE["on"] = False
            _log("no forge_objects.unet — cannot attach PAG.")
            return

        dm = _get_diffusion_model(unet)
        if dm is None:
            _STATE["on"] = False
            _log("could not resolve diffusion_model — skipping.")
            return

        nblocks = _ensure_patched(dm)
        if nblocks == 0:
            _STATE["on"] = False
            return

        targets = _parse_blocks(block_spec, nblocks)
        if not targets:
            _STATE["on"] = False
            _log("no valid target blocks — skipping.")
            return

        _STATE.update(
            on=True,
            scale=scale,
            strength=strength,
            rescale=rescale,
            targets=targets,
            start=min(start, end),
            end=max(start, end),
            total=int(getattr(p, "steps", 20) or 20),
            step=0,
            b0=None,
            active=0,
            pag_raw=None,
        )

        try:
            unet = unet.clone()
            unet.set_model_unet_function_wrapper(_model_wrapper)
            unet.set_model_sampler_post_cfg_function(_post_cfg)
            p.sd_model.forge_objects.unet = unet
            _log(
                f"attached ✅ engine=Anima blocks={nblocks} targets={sorted(targets)} "
                f"scale={scale} strength={strength} rescale={rescale} "
                f"range={_STATE['start']:.2f}-{_STATE['end']:.2f}"
            )
        except Exception as e:
            _STATE["on"] = False
            _log(f"failed to attach hooks: {type(e).__name__}: {e}")

    def postprocess(self, p, processed, *args):
        # Belt-and-suspenders: make sure PAG doesn't leak into a later non-PAG
        # generation if forge_objects reuse ever changed.
        _STATE["on"] = False
