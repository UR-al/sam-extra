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
Forge has used two Anima attention paths over time: older builds call a
module-global ``scaled_dot_product_attention`` while current Forge Neo builds
route ``SelfCrossAttention.torch_attention_op`` through ``attention_function``.
We patch whichever path the loaded build actually uses so, for the appended
PAG rows only, the pre-projection attention output becomes a blend toward the
raw value path ("identity attention"). Value carries no RoPE, so this is
rotary-safe. On grouped-query models (``v`` head count != output head count)
the shapes differ and we skip the perturbation rather than corrupt the tensor.

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
    "on": False,          # any perturbation guidance active this generation
    # Attention-perturbation method: "pag" (value-only) or "seg" (toward-mean),
    # mutually exclusive; None → no attention perturbation (SLG only).
    "attn_method": "pag",
    "attn_scale": 4.0,    # guidance strength for the attention-perturbation term
    "strength": 0.75,     # blend amount (pag: →value; seg: →uniform/mean)
    "attn_targets": set(),  # block indices for attention perturbation
    # SLG (Skip Layer Guidance): skip whole blocks on a separate weak prediction.
    "slg_on": False,
    "slg_scale": 3.0,
    "slg_targets": set(),  # block indices to skip
    "auto_decay": True,   # halve each scale when >1 perturbation is active (toggle)
    "rescale": 0.20,      # std-matching rescale factor
    "start": 0.0,         # start percent of sampling
    "end": 0.7,           # end percent of sampling
    "total": 20,          # total steps this pass
    "step": 0,            # current step counter
    # Row ranges of the appended weak predictions in the enlarged batch.
    "attn_b0": None, "attn_b1": None,   # attention-perturbed cond rows
    "slg_b0": None, "slg_b1": None,     # layer-skipped cond rows
    "any_b0": None,       # min appended index (marks "inside enlarged forward")
    "active": 0,          # >0 while inside a targeted self_attn.forward
    "attn_raw": None,     # stashed attention-perturbed model-output
    "slg_raw": None,      # stashed layer-skipped model-output
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
# Adaptive Guidance — skip the uncond (negative) forward in the later steps,
# where it barely affects the result, for a near-lossless speedup. Implemented
# in the model_function_wrapper by running cond-only and setting the uncond
# output equal to cond (→ CFG collapses to the cond prediction that step).
# Model-agnostic; shares the wrapper with perturbation guidance.
# ---------------------------------------------------------------------------

_ADG: dict = {
    "on": False,
    "start": 0.5,     # skip uncond once this fraction of steps has elapsed
    "interval": 0,    # 0 = skip every step after start; N>0 = still keep uncond every Nth step
}

# Original (unpatched) Anima attention functions. Current Forge Neo uses the
# class-level op; _ORIG_SDPA is retained for compatibility with older builds.
_ORIG_ANIMA_ATTN_OP = None
_ORIG_SDPA = None


# ---------------------------------------------------------------------------
# Attention perturbation — patch the active Anima attention path once.
# ---------------------------------------------------------------------------


def _patched_anima_attention_op(query, key, value, *args, **kwargs):
    """Drop-in for current Forge Neo's ``SelfCrossAttention.torch_attention_op``.

    The inputs are ``[B, seq, heads, dim]`` and the original output is
    ``[B, seq, heads*dim]``. Perturbing here keeps the operation before
    ``output_proj``, matching the older SDPA patch semantically.
    """
    out = _ORIG_ANIMA_ATTN_OP(query, key, value, *args, **kwargs)
    if _STATE["active"] <= 0 or _STATE["attn_b0"] is None:
        return out
    try:
        a0, a1 = _STATE["attn_b0"], _STATE["attn_b1"]
        method = _STATE["attn_method"]
        if not method or a1 > out.shape[0]:
            return out

        # attention_function returns merged heads. Convert value to the same
        # pre-projection layout before blending the appended weak rows.
        value_path = value.reshape(value.shape[0], value.shape[1], -1)
        if value_path.shape != out.shape:
            if not _STATE["gqa_warned"]:
                _STATE["gqa_warned"] = True
                _log("self-attn value/out head layout differ (grouped-query?)"
                     " — skipping attention perturbation on this block.")
            return out

        strength = float(_STATE["strength"])
        out = out.clone()
        if method == "pag":
            target = value_path[a0:a1]
        elif method == "seg":
            # Uniform attention is the sequence-wise mean of values.
            target = value_path[a0:a1].mean(dim=1, keepdim=True).expand_as(out[a0:a1])
        else:
            return out
        out[a0:a1] = torch.lerp(out[a0:a1], target, strength)
    except Exception as e:  # never let the patch break sampling
        _log(f"attention perturb skipped: {type(e).__name__}: {e}")
    return out


def _patched_sdpa(query, key, value, *args, **kwargs):
    """Drop-in for ``backend.nn.anima.scaled_dot_product_attention``.

    Runs the real attention, then — only while inside a *targeted* self_attn
    forward (``active>0``) and only for the appended attention-perturbed rows
    ``[attn_b0:attn_b1]`` — blends the output toward:
      * the raw value path  (PAG: identity attention → out≈value), or
      * the seq-uniform mean (SEG-approx: smoothed/uniform attention).
    """
    out = _ORIG_SDPA(query, key, value, *args, **kwargs)
    # Disabled fast-path: not inside a targeted perturbation forward → return
    # immediately (one bool check) without touching the rest of _STATE.
    if _STATE["active"] <= 0 or _STATE["attn_b0"] is None:
        return out
    try:
        a0, a1 = _STATE["attn_b0"], _STATE["attn_b1"]
        method = _STATE["attn_method"]
        if method and a1 <= out.shape[0]:
            strength = float(_STATE["strength"])
            if method == "pag":
                if value.shape == out.shape:
                    out = out.clone()
                    out[a0:a1] = torch.lerp(out[a0:a1], value[a0:a1], strength)
                elif not _STATE["gqa_warned"]:
                    _STATE["gqa_warned"] = True
                    _log("self-attn value/out head layout differ (grouped-query?)"
                         " — skipping PAG perturbation on this block.")
            elif method == "seg":
                # SEG-approx: blur attention toward uniform → mean of values over
                # the sequence axis (dim=2 for [B, heads, seq, dim]).
                seqdim = out.ndim - 2
                vmean = value.mean(dim=seqdim, keepdim=True)
                if vmean.shape[1] == out.shape[1] and vmean.shape[-1] == out.shape[-1]:
                    out = out.clone()
                    out[a0:a1] = torch.lerp(
                        out[a0:a1], vmean[a0:a1].expand_as(out[a0:a1]), strength
                    )
                elif not _STATE["gqa_warned"]:
                    _STATE["gqa_warned"] = True
                    _log("self-attn head layout differ — skipping SEG on this block.")
    except Exception as e:  # never let the patch break sampling
        _log(f"sdpa perturb skipped: {type(e).__name__}: {e}")
    return out


def _make_selfattn_wrapper(idx: int, orig_forward):
    """Wrap ``self_attn.forward`` to raise ``active`` only for the attention
    target blocks, and only during the appended enlarged forward (``any_b0``)."""

    def _wrapped(*args, **kwargs):
        if _STATE["any_b0"] is not None and idx in _STATE["attn_targets"]:
            _STATE["active"] += 1
            try:
                return orig_forward(*args, **kwargs)
            finally:
                _STATE["active"] -= 1
        return orig_forward(*args, **kwargs)

    return _wrapped


def _make_block_wrapper(idx: int, orig_forward):
    """Wrap a whole ``block.forward`` for SLG: for the layer-skipped rows
    ``[slg_b0:slg_b1]`` make the block a no-op (output = input), i.e. skip its
    contribution → a weak "implicit model" prediction."""

    def _wrapped(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        try:
            s0, s1 = _STATE["slg_b0"], _STATE["slg_b1"]
            if s0 is not None and idx in _STATE["slg_targets"] and torch.is_tensor(out):
                x_in = args[0] if args else kwargs.get("x")
                if (torch.is_tensor(x_in) and x_in.shape == out.shape
                        and s1 <= out.shape[0]):
                    out = out.clone()
                    out[s0:s1] = x_in[s0:s1]
        except Exception as e:
            _log(f"slg skip skipped: {type(e).__name__}: {e}")
        return out

    return _wrapped


def _ensure_patched(diffusion_model) -> int:
    """Install the active attention patch plus the per-block PAG/SLG wrappers.

    Current Forge Neo uses ``SelfCrossAttention.torch_attention_op``. The
    module-global SDPA fallback is only for older Forge builds.
    """
    try:
        from backend.nn import anima as anima_mod
    except Exception as e:
        _log(f"cannot import backend.nn.anima: {type(e).__name__}: {e}")
        return 0

    attn_cls = getattr(anima_mod, "SelfCrossAttention", None)
    if attn_cls is not None and hasattr(attn_cls, "torch_attention_op"):
        if not getattr(attn_cls, "_pag_attention_op_patched", False):
            global _ORIG_ANIMA_ATTN_OP
            _ORIG_ANIMA_ATTN_OP = attn_cls.torch_attention_op
            attn_cls.torch_attention_op = staticmethod(_patched_anima_attention_op)
            attn_cls._pag_attention_op_patched = True
            _log("patched backend.nn.anima.SelfCrossAttention.torch_attention_op ✅")
    elif hasattr(anima_mod, "scaled_dot_product_attention"):
        if not getattr(anima_mod, "_pag_sdpa_patched", False):
            global _ORIG_SDPA
            _ORIG_SDPA = anima_mod.scaled_dot_product_attention
            anima_mod.scaled_dot_product_attention = _patched_sdpa
            anima_mod._pag_sdpa_patched = True
            _log("patched backend.nn.anima.scaled_dot_product_attention ✅")
    else:
        _log("backend.nn.anima has no supported attention entry point — "
             "cannot perturb attention on this build.")
        return 0

    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        _log("diffusion_model has no .blocks — unexpected Anima structure.")
        return 0

    wrapped_sa = wrapped_bl = 0
    for idx, block in enumerate(blocks):
        # SLG: wrap the whole block forward.
        if not getattr(block, "_pag_block_wrapped", False):
            try:
                block._pag_orig_block_forward = block.forward
                block.forward = _make_block_wrapper(idx, block._pag_orig_block_forward)
                block._pag_block_wrapped = True
                wrapped_bl += 1
            except Exception as e:
                _log(f"failed to wrap block {idx} forward: {type(e).__name__}: {e}")
        # PAG/SEG: wrap the self_attn forward.
        sa = getattr(block, "self_attn", None)
        if sa is not None and not getattr(sa, "_pag_wrapped", False):
            try:
                sa._pag_orig_forward = sa.forward
                sa.forward = _make_selfattn_wrapper(idx, sa._pag_orig_forward)
                sa._pag_wrapped = True
                wrapped_sa += 1
            except Exception as e:
                _log(f"failed to wrap block {idx} self_attn: {type(e).__name__}: {e}")
    if wrapped_sa or wrapped_bl:
        _log(f"wrapped {wrapped_sa} self_attn + {wrapped_bl} block forward(s)")
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


def _select_c(c: dict, idx_tensor, batch: int) -> dict:
    """Keep only the ``idx_tensor`` rows of every batched tensor (for the
    Adaptive-Guidance cond-only forward). Non-tensor entries pass through."""
    out = {}
    for k, v in c.items():
        if k == "transformer_options":
            out[k] = v
            continue
        if torch.is_tensor(v) and v.shape[0] == batch:
            out[k] = v.index_select(0, idx_tensor)
        else:
            out[k] = v
    return out


def _pct_now() -> float:
    total = max(1, int(_STATE["total"]))
    return _STATE["step"] / total


def _percent_in_range() -> bool:
    pct = _pct_now()
    return _STATE["start"] <= pct <= _STATE["end"]


def _adg_should_skip() -> bool:
    """Adaptive Guidance: skip the uncond forward this step?"""
    if not _ADG["on"]:
        return False
    if _pct_now() < float(_ADG["start"]):
        return False
    iv = int(_ADG["interval"])
    if iv > 0 and (_STATE["step"] % iv == 0):
        return False  # periodically keep an uncond forward for safety
    return True


# ---------------------------------------------------------------------------
# The two Forge-honoured hooks
# ---------------------------------------------------------------------------


def _clear_markers():
    _STATE["any_b0"] = None
    _STATE["attn_b0"] = _STATE["attn_b1"] = None
    _STATE["slg_b0"] = _STATE["slg_b1"] = None


def _model_wrapper(apply_model, w):
    """model_function_wrapper: run ONE enlarged ``apply_model`` that also
    produces the weak predictions (attention-perturbed for PAG/SEG and/or
    layer-skipped for SLG) by appending copies of the cond rows to the batch.
    No separate forward — every weak prediction rides the same kernel launch.
    """
    x = w.get("input")
    ts = w.get("timestep")
    c = w.get("c") or {}
    cou = w.get("cond_or_uncond")

    _STATE["attn_raw"] = None
    _STATE["slg_raw"] = None
    if not (_STATE["on"] or _ADG["on"]) or torch is None:
        return apply_model(x, ts, **c)

    _STATE["step"] += 1
    try:
        if cou is None or len(cou) != 2:
            return apply_model(x, ts, **c)
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
        uidx = torch.tensor(uncond_idx, device=x.device, dtype=torch.long)
        n = len(cond_idx)

        # --- Adaptive Guidance: in the late steps, run cond-only and set the
        # uncond output = cond → CFG collapses to cond (guidance neutralized),
        # saving the uncond half of the batch. Takes precedence over the
        # (adds-work) perturbation path in those steps. ---
        if _adg_should_skip():
            x_c = x.index_select(0, idx)
            ts_c = ts.index_select(0, idx)
            c_c = _select_c(c, idx, batch)
            out_c = apply_model(x_c, ts_c, **c_c)
            out_full = torch.empty(
                (batch,) + tuple(out_c.shape[1:]),
                device=out_c.device, dtype=out_c.dtype,
            )
            out_full.index_copy_(0, idx, out_c)
            out_full.index_copy_(0, uidx, out_c)  # uncond := cond
            return out_full

        if not _STATE["on"] or not _percent_in_range():
            return apply_model(x, ts, **c)

        attn_on = bool(_STATE["attn_method"]) and float(_STATE["attn_scale"]) > 0 \
            and bool(_STATE["attn_targets"])
        slg_on = bool(_STATE["slg_on"]) and float(_STATE["slg_scale"]) > 0 \
            and bool(_STATE["slg_targets"])
        if not attn_on and not slg_on:
            return apply_model(x, ts, **c)

        # Lay out the appended weak-prediction rows.
        cursor = batch
        appended = []
        a0 = a1 = s0 = s1 = None
        if attn_on:
            a0, a1 = cursor, cursor + n
            appended.append(idx); cursor += n
        if slg_on:
            s0, s1 = cursor, cursor + n
            appended.append(idx); cursor += n

        app_idx = torch.cat(appended) if len(appended) > 1 else appended[0]
        x_ext = torch.cat([x, x.index_select(0, app_idx)], dim=0)
        ts_ext = torch.cat([ts, ts.index_select(0, app_idx)], dim=0)
        c_ext = _extend_c(c, app_idx, batch)

        _STATE["attn_b0"], _STATE["attn_b1"] = a0, a1
        _STATE["slg_b0"], _STATE["slg_b1"] = s0, s1
        _STATE["any_b0"] = batch
        try:
            out_ext = apply_model(x_ext, ts_ext, **c_ext)
        finally:
            _clear_markers()

        out = out_ext[:batch]
        if a0 is not None:
            _STATE["attn_raw"] = out_ext[a0:a1].detach().float()
        if s0 is not None:
            _STATE["slg_raw"] = out_ext[s0:s1].detach().float()
        _STATE["cond_raw"] = out.index_select(0, idx).detach().float()
        _STATE["uncond_raw"] = out.index_select(0, uidx).detach().float()
        return out
    except Exception as e:
        _clear_markers()
        _STATE["attn_raw"] = None
        _STATE["slg_raw"] = None
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


def _apply_perturbation(args, base):
    """Add the perturbation-guidance term(s) onto ``base`` (denoised space):
    ``scale·(cond_denoised − weak_denoised)`` for each active weak prediction
    (attention-perturbed for PAG/SEG, and/or layer-skipped for SLG). Uses the
    empirically-recovered ``c_out`` so it holds for eps/v/flow alike. When more
    than one is active AND ``auto_decay`` is on, each scale is divided by the
    active count. Returns ``base`` unchanged on any problem."""
    cd = args["cond_denoised"].float()
    ud = args["uncond_denoised"].float()
    cond_raw = _STATE["cond_raw"]
    uncond_raw = _STATE["uncond_raw"]
    attn_raw = _STATE["attn_raw"]
    slg_raw = _STATE["slg_raw"]
    if cond_raw is None or uncond_raw is None:
        return base
    if cd.shape != cond_raw.shape:
        return base

    # (cd-ud) = c_out*(cond_raw-uncond_raw). Recover c_out by least squares.
    do = cond_raw - uncond_raw
    dd = cd - ud
    denom = (do * do).sum()
    if float(denom) <= 1e-8:
        return base
    c_out = (dd * do).sum() / denom

    terms = []
    if attn_raw is not None and attn_raw.shape == cond_raw.shape:
        terms.append((float(_STATE["attn_scale"]), attn_raw))
    if slg_raw is not None and slg_raw.shape == cond_raw.shape:
        terms.append((float(_STATE["slg_scale"]), slg_raw))
    if not terms:
        return base

    decay = 1.0 / len(terms) if (_STATE["auto_decay"] and len(terms) > 1) else 1.0
    result = base
    for scale, weak in terms:
        # cond_denoised - weak_denoised = c_out*(cond_raw - weak_raw)
        result = result + (scale * decay) * c_out * (cond_raw - weak)

    # Rescale — auto-skipped while APG governs magnitude (toggleable).
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
    then perturbation guidance (PAG/SEG/SLG) is layered on top. Because the
    perturbation term is added via a recovered ``c_out``, it composes on top of
    ANY base — standard CFG, APG, or a built-in like MaHiRo/RescaleCFG.
    """
    denoised = args["denoised"]
    if torch is None:
        return denoised
    # Nothing to combine (e.g. only Adaptive Guidance active) → return as-is
    # without the float() round-trip / two latent-sized allocations per step.
    has_pert = _STATE["on"] and (_STATE["attn_raw"] is not None or _STATE["slg_raw"] is not None)
    if not _APG["on"] and not has_pert:
        _STATE["attn_raw"] = None
        _STATE["slg_raw"] = None
        return denoised
    try:
        result = denoised.float()

        if _APG["on"]:
            apg = _apply_apg(args, args.get("cond_scale", 1.0))
            if apg is not None:
                result = apg

        if _STATE["on"] and (_STATE["attn_raw"] is not None or _STATE["slg_raw"] is not None):
            result = _apply_perturbation(args, result)

        return result.to(denoised.dtype)
    except Exception as e:
        _log(f"post_cfg fallback → unmodified cfg: {type(e).__name__}: {e}")
        return denoised
    finally:
        _STATE["attn_raw"] = None
        _STATE["slg_raw"] = None


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
            "[Anima Pert] Enable", str,
            partial(_pag_xyz_set, field="enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] Attn Method", str,
            partial(_pag_xyz_set, field="method"),
            choices=lambda: ["PAG", "SEG", "None"],
        ),
        xyz_grid.AxisOption("[Anima Pert] Attn Scale", float, partial(_pag_xyz_set, field="scale")),
        xyz_grid.AxisOption(
            "[Anima Pert] Perturbation Strength", float,
            partial(_pag_xyz_set, field="strength"),
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] Attn Block Indices", str,
            partial(_pag_xyz_set, field="blocks"),
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] SLG Enable", str,
            partial(_pag_xyz_set, field="slg_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Anima Pert] SLG Scale", float, partial(_pag_xyz_set, field="slg_scale")),
        xyz_grid.AxisOption(
            "[Anima Pert] SLG Block Indices", str,
            partial(_pag_xyz_set, field="slg_blocks"),
        ),
        xyz_grid.AxisOption("[Anima Pert] Start Percent", float, partial(_pag_xyz_set, field="start")),
        xyz_grid.AxisOption("[Anima Pert] End Percent", float, partial(_pag_xyz_set, field="end")),
        xyz_grid.AxisOption("[Anima Pert] Rescale", float, partial(_pag_xyz_set, field="rescale")),
        # APG axes (ON/OFF + the three knobs) for the same style of comparison.
        xyz_grid.AxisOption(
            "[Anima APG] Enable", str,
            partial(_pag_xyz_set, field="apg_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Anima APG] Eta", float, partial(_pag_xyz_set, field="apg_eta")),
        xyz_grid.AxisOption("[Anima APG] Norm Threshold", float, partial(_pag_xyz_set, field="apg_norm")),
        xyz_grid.AxisOption("[Anima APG] Momentum", float, partial(_pag_xyz_set, field="apg_momentum")),
        # Adaptive Guidance (speed) axes.
        xyz_grid.AxisOption(
            "[Anima AdaptiveG] Enable", str,
            partial(_pag_xyz_set, field="adg_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Anima AdaptiveG] Skip After", float, partial(_pag_xyz_set, field="adg_start")),
        xyz_grid.AxisOption("[Anima AdaptiveG] Keep Every", float, partial(_pag_xyz_set, field="adg_interval")),
    ]

    if not any(a.label.startswith(("[Anima Pert]", "[Anima APG]", "[Anima AdaptiveG]"))
               for a in xyz_grid.axis_options):
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
    # sorting_priority governs BOTH the accordion position and the
    # process order (lower = higher up / earlier). 0 keeps this panel in the
    # SAM3 extension block right under the SAM3 accordion. We still clone from
    # the CURRENT forge_objects.unet, so other unet-patching scripts compose
    # regardless of order.
    sorting_priority = 0

    def title(self):
        return "Anima Perturbation Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima Perturbation Guidance (PAG / SEG / SLG)", open=False):
            gr.Markdown(
                "Anima/Cosmos/Predict2 계열 **DiT 전용** perturbation guidance. 후반 블록에 "
                "*약한 예측*을 만들어 CFG를 그 반대로 밀어 구조·디테일을 강화합니다. **Anima "
                "엔진에서만** 동작하고 그 외/오류 시 폴백합니다. 약한 예측은 별도 forward 없이 "
                "**같은 배치에 접어** 계산합니다.\n\n"
                "- **PAG**(value-only) ↔ **SEG**(uniform/blur)는 성격이 겹쳐 **택1**\n"
                "- **SLG**(레이어 스킵)는 PAG/SEG와 **병용 가능**"
            )
            enabled = gr.Checkbox(
                label="Enable Perturbation Guidance",
                value=False,
                elem_id="anima_safe_pag_enable",
            )
            attn_method = gr.Radio(
                label="Attention perturbation (PAG ↔ SEG 택1 · None=SLG만)",
                choices=["PAG", "SEG", "None"],
                value="PAG",
                elem_id="anima_safe_pag_method",
            )
            scale = gr.Slider(
                label="Attention guidance scale (PAG/SEG)",
                minimum=0.0, maximum=15.0, step=0.1, value=4.0,
                elem_id="anima_safe_pag_scale",
            )
            strength = gr.Slider(
                label="Perturbation strength (PAG: →value · SEG: →uniform)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.75,
                elem_id="anima_safe_pag_strength",
            )
            block_indices = gr.Textbox(
                label="Attention block indices (빈칸=후반 절반 자동, 예: 18 / 14-27)",
                value="",
                elem_id="anima_safe_pag_blocks",
            )

            gr.Markdown("**SLG (Skip Layer Guidance)** — PAG/SEG와 병용 가능")
            slg_on = gr.Checkbox(
                label="Enable SLG (skip layers)",
                value=False,
                elem_id="anima_safe_pag_slg_enable",
            )
            slg_scale = gr.Slider(
                label="SLG guidance scale",
                minimum=0.0, maximum=15.0, step=0.1, value=3.0,
                elem_id="anima_safe_pag_slg_scale",
            )
            slg_blocks = gr.Textbox(
                label="SLG skip block indices (빈칸=후반 절반 자동)",
                value="",
                elem_id="anima_safe_pag_slg_blocks",
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
            auto_decay = gr.Checkbox(
                label="여러 perturbation 동시 사용 시 각 scale 자동 감쇠 (÷활성 수)",
                value=True,
                elem_id="anima_safe_pag_auto_decay",
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

            gr.Markdown("---\n### Adaptive Guidance (속도)")
            gr.Markdown(
                "샘플링 **후반부에는 uncond(네거티브) 예측 기여가 미미**하므로, 지정 지점 "
                "이후 uncond forward를 **생략**해 그 스텝 배치를 절반으로 줄입니다(무손실에 "
                "가까운 속도↑). 추가 계산이 아니라 **빼는** 쪽이며 모든 모델에서 동작합니다. "
                "생략 스텝에서는 perturbation도 함께 쉬어 CFG가 cond로 붕괴됩니다."
            )
            adg_enabled = gr.Checkbox(
                label="Enable Adaptive Guidance (후반 uncond 생략)",
                value=False,
                elem_id="anima_safe_pag_adg_enable",
            )
            adg_start = gr.Slider(
                label="Skip after (이 지점 이후 uncond 생략)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.5,
                elem_id="anima_safe_pag_adg_start",
            )
            with gr.Accordion("Adaptive Guidance Advanced (세부값)", open=False):
                gr.Markdown(
                    "**keep every N** — 생략 구간에서도 N스텝마다 uncond를 한 번씩 유지해 "
                    "품질을 보수적으로 지킵니다(0=항상 생략, 값 클수록 더 자주 생략)."
                )
                adg_interval = gr.Slider(
                    label="Keep every N steps (0 = 항상 생략)",
                    minimum=0, maximum=10, step=1, value=0,
                    elem_id="anima_safe_pag_adg_interval",
                )
        return [
            enabled, attn_method, scale, strength, block_indices,
            slg_on, slg_scale, slg_blocks,
            start_percent, end_percent, rescale, auto_decay,
            apg_enabled, apg_eta, apg_norm, apg_momentum, apg_autooff,
            adg_enabled, adg_start, adg_interval,
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

        # ---- Perturbation enable (checkbox, XYZ can override for ON/OFF grids) ----
        try:
            pert_enabled = bool(_arg(0, False))
        except Exception:
            pert_enabled = False
        if "enabled" in xyz:
            pert_enabled = _as_bool(xyz["enabled"], pert_enabled)

        # ---- APG enable ----
        try:
            apg_enabled = bool(_arg(12, False))
        except Exception:
            apg_enabled = False
        if "apg_enabled" in xyz:
            apg_enabled = _as_bool(xyz["apg_enabled"], apg_enabled)

        # ---- Adaptive Guidance enable ----
        try:
            adg_enabled = bool(_arg(17, False))
        except Exception:
            adg_enabled = False
        if "adg_enabled" in xyz:
            adg_enabled = _as_bool(xyz["adg_enabled"], adg_enabled)

        if not pert_enabled and not apg_enabled and not adg_enabled:
            _STATE["on"] = False
            _APG["on"] = False
            _ADG["on"] = False
            return

        # ---- Read every knob (with XYZ overrides) ----
        try:
            method_str = str(xyz["method"] if "method" in xyz else _arg(1, "PAG")).strip().lower()
            attn_method = method_str if method_str in ("pag", "seg") else None
            scale = _xyz_num("scale", float(_arg(2, 4.0)))
            strength = _xyz_num("strength", float(_arg(3, 0.75)))
            block_spec = str(xyz["blocks"]) if "blocks" in xyz else str(_arg(4, ""))
            slg_on = _as_bool(xyz["slg_enabled"], _as_bool(_arg(5, False), False)) \
                if "slg_enabled" in xyz else _as_bool(_arg(5, False), False)
            slg_scale = _xyz_num("slg_scale", float(_arg(6, 3.0)))
            slg_block_spec = str(xyz["slg_blocks"]) if "slg_blocks" in xyz else str(_arg(7, ""))
            start = _xyz_num("start", float(_arg(8, 0.0)))
            end = _xyz_num("end", float(_arg(9, 0.7)))
            rescale = _xyz_num("rescale", float(_arg(10, 0.20)))
            auto_decay = _as_bool(_arg(11, True), True)
            apg_eta = _xyz_num("apg_eta", float(_arg(13, 0.0)))
            apg_norm = _xyz_num("apg_norm", float(_arg(14, 15.0)))
            apg_momentum = _xyz_num("apg_momentum", float(_arg(15, 0.0)))
            apg_autooff = _as_bool(_arg(16, True), True)
            adg_start = _xyz_num("adg_start", float(_arg(18, 0.5)))
            adg_interval = int(_xyz_num("adg_interval", float(_arg(19, 0))))
        except Exception as e:
            _STATE["on"] = False
            _APG["on"] = False
            _ADG["on"] = False
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

        # ---- Adaptive Guidance: model-agnostic, needs no patching. ----
        if adg_enabled:
            _ADG.update(on=True, start=adg_start, interval=max(0, adg_interval))
        else:
            _ADG["on"] = False

        # ---- Perturbation: Anima-only, needs the attention/block patches. ----
        pert_ok = False
        attn_targets: set = set()
        slg_targets: set = set()
        nblocks = 0
        if pert_enabled:
            if engine != "Anima":
                _log(f"engine={engine} (not 'Anima') — perturbation skipped (APG "
                     "still runs if enabled).")
            else:
                dm = _get_diffusion_model(unet)
                if dm is None:
                    _log("could not resolve diffusion_model — perturbation skipped.")
                else:
                    nblocks = _ensure_patched(dm)
                    if nblocks:
                        if attn_method and scale > 0:
                            attn_targets = _parse_blocks(block_spec, nblocks)
                        if slg_on and slg_scale > 0:
                            slg_targets = _parse_blocks(slg_block_spec, nblocks)
                        if (attn_method and attn_targets) or (slg_on and slg_targets):
                            pert_ok = True
                        else:
                            _log("no valid target blocks — perturbation skipped.")

        if pert_ok:
            _STATE.update(
                on=True, attn_method=(attn_method if attn_targets else None),
                attn_scale=scale, strength=strength, attn_targets=attn_targets,
                slg_on=bool(slg_on and slg_targets), slg_scale=slg_scale,
                slg_targets=slg_targets, auto_decay=auto_decay, rescale=rescale,
                start=min(start, end), end=max(start, end),
                active=0, attn_raw=None, slg_raw=None,
            )
            _clear_markers()
        else:
            _STATE["on"] = False

        if not _STATE["on"] and not _APG["on"] and not _ADG["on"]:
            return

        # Step counter / total drive both perturbation range and AG gating.
        _STATE["total"] = int(getattr(p, "steps", 20) or 20)
        _STATE["step"] = 0

        # ---- Attach hooks (clone so Forge core / other gens are untouched) ----
        try:
            unet = unet.clone()
            # The wrapper is needed for perturbation AND for Adaptive Guidance
            # (both manipulate the cond/uncond batch before apply_model).
            if _STATE["on"] or _ADG["on"]:
                unet.set_model_unet_function_wrapper(_model_wrapper)
            unet.set_model_sampler_post_cfg_function(_post_cfg)
            p.sd_model.forge_objects.unet = unet

            if not hasattr(p, "extra_generation_params"):
                p.extra_generation_params = {}
            if _STATE["on"]:
                parts = []
                if _STATE["attn_method"]:
                    parts.append(f"{_STATE['attn_method'].upper()} scale={scale} "
                                 f"strength={strength} blocks={sorted(attn_targets)}")
                if _STATE["slg_on"]:
                    parts.append(f"SLG scale={slg_scale} skip={sorted(slg_targets)}")
                p.extra_generation_params["Anima Perturbation Guidance"] = (
                    "; ".join(parts)
                    + f"; range={_STATE['start']:.2f}-{_STATE['end']:.2f}"
                    + f"; auto_decay={auto_decay}"
                )
            if _APG["on"]:
                p.extra_generation_params["Anima APG"] = (
                    f"eta={apg_eta}, norm={apg_norm}, momentum={apg_momentum}"
                )
            if _ADG["on"]:
                p.extra_generation_params["Anima Adaptive Guidance"] = (
                    f"skip_after={_ADG['start']:.2f}, keep_every={_ADG['interval']}"
                )
            _log(
                f"attached ✅ engine={engine} "
                f"pert={'on' if _STATE['on'] else 'off'} "
                f"(attn={_STATE['attn_method']}:{sorted(attn_targets)} scale={scale} "
                f"slg={_STATE['slg_on']}:{sorted(slg_targets)} scale={slg_scale} "
                f"auto_decay={auto_decay} "
                f"rescale={'auto-off' if (_APG['on'] and apg_autooff) else rescale}) "
                f"APG={'on' if _APG['on'] else 'off'} "
                f"(eta={apg_eta} norm={apg_norm} mom={apg_momentum}) "
                f"AdaptiveG={'on' if _ADG['on'] else 'off'} "
                f"(skip_after={_ADG['start']} keep_every={_ADG['interval']})"
            )
        except Exception as e:
            _STATE["on"] = False
            _APG["on"] = False
            _ADG["on"] = False
            _log(f"failed to attach hooks: {type(e).__name__}: {e}")

    def postprocess(self, p, processed, *args):
        # Belt-and-suspenders: make sure guidance doesn't leak into a later
        # generation if forge_objects reuse ever changed.
        _STATE["on"] = False
        _APG["on"] = False
        _ADG["on"] = False
        # Drop the stashed latent-sized tensors so they don't sit in VRAM
        # between generations (they'd otherwise linger until the next gen).
        _STATE["cond_raw"] = None
        _STATE["uncond_raw"] = None
        _STATE["attn_raw"] = None
        _STATE["slg_raw"] = None
        _APG["avg"] = None
