"""Small sigma helpers shared by the post-CFG detail stages (TSR, MG/HiGS, HiFlow, adaptive SMC).

Forge hands every post-CFG function ``args["sigma"]`` (the sigma the model was called with),
``args["input"]`` (the sampler's x_t) and ``args["model"]`` (the KModel, whose ``predictor`` knows
the parameterisation). Two facts the stages rely on:

* **Parameterisation.** Forge's ``AbstractPrediction`` (backend/modules/k_prediction.py) uses
  ``prediction_type == "const"`` for rectified-flow models (Anima, Flux, SD3: ``x_t = (1-σ)·x0 + σ·ε``)
  and the variance-exploding k-diffusion form ``x_t = x0 + σ·ε`` for ``epsilon``/``v_prediction``
  (SD1/SDXL). In both, the sampler's derivative is ``d = (x_t - x0) / σ`` (k-diffusion ``to_d``) —
  the flow velocity on a const model, ε on an eps model.
* **Detail Daemon.** ``scripts/anima_detail_daemon.py`` lowers the sigma handed to the model from
  ``on_cfg_denoiser``; the sampler itself still steps with its own sigma. Stages that work in the
  sampler's velocity space (MG/HiGS, adaptive SMC, HiFlow) use that unscaled sigma
  (``sam3ext.guidance.dave_gate.pre_dd_sigma``); TSR follows ComfyUI's node and uses the sigma the
  model saw.
"""

from __future__ import annotations

import math

from .dave_gate import ISCLOSE_ATOL, ISCLOSE_RTOL, pre_dd_sigma

__all__ = [
    "first_float",
    "is_flow_model",
    "half_log_snr",
    "sampler_sigma",
    "schedule_index",
    "sampling_schedule",
]


def first_float(value):
    """First element of a tensor sigma (or the number itself) as a Python float, else None."""
    if value is None:
        return None
    try:
        if hasattr(value, "flatten") and hasattr(value, "shape"):
            if value.numel() == 0:
                return None
            return float(value.flatten()[0].item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def is_flow_model(model) -> bool | None:
    """True for a rectified-flow (``const``) predictor, False for eps/v, None when unknown."""
    predictor = getattr(model, "predictor", None)
    kind = getattr(predictor, "prediction_type", None)
    if kind is None:
        return None
    return str(kind) == "const"


def half_log_snr(sigma: float, flow: bool) -> float:
    """``log(alpha/sigma)`` of the forward process at ``sigma``.

    Flow (``x_t = (1-σ)x0 + σε``): ``alpha = 1-σ`` → ``log((1-σ)/σ)``. Eps/v k-diffusion
    (``x_t = x0 + σε``): ``alpha = 1`` → ``-log σ`` (ComfyUI ``sigma_to_half_log_snr`` for
    CONST and for the discrete schedules). ``sigma`` must be in ``(0, 1)`` for flow, ``> 0`` else.
    """
    sigma = float(sigma)
    if flow:
        return math.log((1.0 - sigma) / sigma)
    return -math.log(sigma)


def sampler_sigma(args) -> float | None:
    """The sampler's own sigma for this post-CFG call (before Detail Daemon scaled it)."""
    current = args.get("sigma") if isinstance(args, dict) else None
    noted = pre_dd_sigma(current)
    if noted is not None:
        value = first_float(noted)
        if value is not None:
            return value
    return first_float(current)


def sampling_schedule(args):
    """Forge's ``sampling_sigmas`` for this run (whole list), or None."""
    if not isinstance(args, dict):
        return None
    options = args.get("model_options") or {}
    if not isinstance(options, dict):
        return None
    transformer = options.get("transformer_options") or {}
    if not isinstance(transformer, dict):
        return None
    return transformer.get("sampling_sigmas")


def schedule_index(schedule, sigma: float | None) -> int | None:
    """Index of the first schedule entry ``isclose`` to ``sigma`` (DAVE's rule), else None.

    None also when there is no schedule. A second-order sampler's midpoint evaluation is not on
    the schedule, so it returns None — the history stages use that to update their state once per
    sampler step.
    """
    if schedule is None or sigma is None:
        return None
    try:
        values = schedule.flatten().tolist() if hasattr(schedule, "flatten") else list(schedule)
    except (TypeError, RuntimeError):
        return None
    for index, value in enumerate(values):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if abs(value - sigma) <= ISCLOSE_ATOL + ISCLOSE_RTOL * abs(sigma):
            return index
    return None
