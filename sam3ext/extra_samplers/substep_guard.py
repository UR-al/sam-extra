"""When Euler (SMEA) Dy CFG++ runs without its sub-steps, and how the infotext and the console say so.

The Dy/SMEA sub-steps evaluate the model at another resolution (half, ×1.25). Some full-resolution
machinery of Forge Neo 2.29.2 cannot follow that, so on such a request the two samplers run their
plain CFG++ Euler steps (``euler_dy(substeps=False)``) instead of failing:

* **Spectrum Integrated** — Forge's built-in ``extensions-builtin/sd_forge_spectrum``. Its
  ``spectrum_unet_wrapper`` (``lib_spectrum/forecaster.py``, set as the UNet's
  ``model_function_wrapper`` in ``process_before_every_sampling``) counts model calls and keeps one
  output shape: a sub-step resets its buffers to the sub-step's resolution and the next *forecast*
  call returns that shape for a full-resolution input — a shape-mismatch error (Euler Dy for
  warm-up ≤ 6, Spectrum's default; Euler SMEA Dy for warm-up ≤ 4). Detected the way the SPEED port
  does (``sam3ext/speed/forge_host.py`` ``_spectrum_wrapper``): the wrapper's ``__qualname__``
  contains ``spectrum_unet_wrapper`` or its module is ``lib_spectrum…``.
* **Wan 2.2 I2V** — ``dynamic_args.concat_latent`` (``backend/diffusion_engine/wan.py``
  ``encode_first_stage`` → ``image_to_video``) is concatenated with the latent on the channel axis at
  full size whenever the DiT's ``in_dim`` exceeds its latent channels ``out_dim`` (``backend/nn/wan.py``).
* **PiD** — ``dynamic_args.lq_latent[0]``, the full-size low-quality input, conditions every forward
  (``backend/nn/pixeldit/pid.py``; ``dynamic_args.pid`` marks the model).
* ``unet.extra_concat_condition`` — set only by third-party extensions; Forge concatenates it for an
  inpainting model only while it has the latent's size (``backend/sampling/sampling_function.py``).

Checked when Forge initialises the sampler (``registry.ExtraKDiffusionSampler.initialize``), i.e. after
every script's ``process_before_every_sampling`` and after the model load / conditioning / image
encode that set these inputs. Each reason is logged once per session and recorded in the infotext as
``Extra Samplers status: dy sub-steps skipped (<reason> + <reason>)`` — not read back on paste.

``Extra Samplers status`` is shared: ``record_status_part`` keeps one text per part (the Dy part
above, the ER SDE noise-window part ``params.WINDOW_SKIPPED_STATUS``) and writes them joined by
``"; "`` in the order they first appeared, so a request with only Dy reasons keeps exactly the text
above.
"""

from __future__ import annotations

import importlib

import torch

__all__ = [
    "REASON_EXTRA_CONCAT",
    "REASON_PID",
    "REASON_SPECTRUM",
    "REASON_WAN",
    "STATUS_KEY",
    "blocking_reasons",
    "explain",
    "record_status",
    "record_status_part",
    "spectrum_active",
]

STATUS_KEY = "Extra Samplers status"
_STATUS_PREFIX = "dy sub-steps skipped"
_SKIPS_ATTR = "_sam_extra_substep_skips"   # the reasons already recorded for this request (all passes)
_PARTS_ATTR = "_sam_extra_status_parts"    # {part key: text} of the request's status, in first-seen order
_DY_PART = "dy"

REASON_SPECTRUM = "Spectrum"
REASON_WAN = "Wan I2V concat_latent"
REASON_PID = "PiD lq_latent"
REASON_EXTRA_CONCAT = "extra_concat_condition"

_EXPLANATION = {
    REASON_SPECTRUM: (
        "Forge's Spectrum Integrated is on - its forecaster keeps one output shape and would hand the "
        "sub-step's resolution back to a full-resolution step"
    ),
    REASON_WAN: "Wan 2.2 I2V concatenates its full-size concat_latent with the latent on every forward",
    REASON_PID: "PiD conditions every forward on its full-size lq_latent",
    REASON_EXTRA_CONCAT: "an extension set unet.extra_concat_condition (full-size concat conditioning)",
}


def explain(reason: str) -> str:
    return _EXPLANATION.get(reason, reason)


def _forge_unet(p):
    """The UNet patcher Forge samples with (``sampling_function`` reads ``sd_model.forge_objects.unet``)."""
    return getattr(getattr(getattr(p, "sd_model", None), "forge_objects", None), "unet", None)


def _forge_dynamic_args():
    """Forge's ``backend.args.dynamic_args`` (None outside Forge)."""
    try:
        return getattr(importlib.import_module("backend.args"), "dynamic_args", None)
    except Exception:
        return None


def spectrum_active(unet) -> bool:
    """Is Forge's Spectrum Integrated wrapper the UNet's ``model_function_wrapper``?"""
    options = getattr(unet, "model_options", None)
    wrapper = options.get("model_function_wrapper") if isinstance(options, dict) else None
    if wrapper is None:
        return False
    qualname = str(getattr(wrapper, "__qualname__", "") or getattr(wrapper, "__name__", ""))
    module = str(getattr(wrapper, "__module__", "") or "")
    return "spectrum_unet_wrapper" in qualname or module.startswith("lib_spectrum")


def _wan_concat(unet, dynamic_args) -> bool:
    if not torch.is_tensor(getattr(dynamic_args, "concat_latent", None)):
        return False
    model = getattr(getattr(unet, "model", None), "diffusion_model", None)
    in_dim, out_dim = getattr(model, "in_dim", None), getattr(model, "out_dim", None)
    if isinstance(in_dim, int) and isinstance(out_dim, int):
        return out_dim < in_dim   # T2V (in_dim == out_dim) never reads a leftover concat_latent
    return True


def _pid_lq(dynamic_args) -> bool:
    lq = getattr(dynamic_args, "lq_latent", None)
    first = lq[0] if isinstance(lq, (list, tuple)) and len(lq) > 0 else None
    return getattr(dynamic_args, "pid", False) is True or torch.is_tensor(first)


def blocking_reasons(p, dynamic_args=None) -> list:
    """Why the sub-steps cannot run on this request (empty: they can)."""
    unet = _forge_unet(p)
    dynamic_args = _forge_dynamic_args() if dynamic_args is None else dynamic_args
    reasons = []
    if spectrum_active(unet):
        reasons.append(REASON_SPECTRUM)
    if dynamic_args is not None:
        if _wan_concat(unet, dynamic_args):
            reasons.append(REASON_WAN)
        if _pid_lq(dynamic_args):
            reasons.append(REASON_PID)
    if getattr(unet, "extra_concat_condition", None) is not None:
        reasons.append(REASON_EXTRA_CONCAT)
    return reasons


def record_status(p, reasons) -> str | None:
    """Add ``reasons`` to the request's ``Extra Samplers status`` infotext (every pass of it)."""
    seen = list(getattr(p, _SKIPS_ATTR, None) or ())
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    if not seen:
        return None
    try:
        setattr(p, _SKIPS_ATTR, seen)
    except Exception:
        pass
    text = f"{_STATUS_PREFIX} ({' + '.join(seen)})"
    record_status_part(p, _DY_PART, text)
    return text


def record_status_part(p, key: str, text: str) -> str:
    """Set the status part ``key`` of the request to ``text`` (a part keeps its first position) and
    write ``Extra Samplers status`` as every part joined by ``"; "``; returns the whole value."""
    parts = getattr(p, _PARTS_ATTR, None)
    if not isinstance(parts, dict):
        parts = {}
    parts[key] = text
    try:
        setattr(p, _PARTS_ATTR, parts)
    except Exception:
        pass
    value = "; ".join(parts.values())
    params = getattr(p, "extra_generation_params", None)
    if isinstance(params, dict):
        params[STATUS_KEY] = value
    return value
