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
* **Which part of the list a pass walks.** Forge stores its whole sigma list in
  ``transformer_options['sampling_sigmas']``; txt2img's first pass walks all of it, img2img and the
  hires pass walk ``sigmas[steps - t_enc - 1:]``. ``forge_sampling_offset`` works that start out with
  Forge's own ``setup_img2img_steps`` (shared by Detail Daemon, DAVE's gate and Colorcraft).
* **A list re-published mid-run.** Anima SPEED patches the schedule of the run it segments (the
  transition sigma, or the whole re-spaced tail) and puts the patched copy into
  ``sampling_sigmas`` after each transition, then the original back. The copy is a new tensor, so
  a stage that keys its run on the list object (Colorcraft) would take it for another sampling run;
  ``mark_republished`` tags the copy with the list it stands in for and ``republished_from`` reads
  the tag back. Stages that read the current list on every call (Detail Daemon, DAVE, MG/HiGS,
  HiFlow) need nothing.
"""

from __future__ import annotations

import math

from .dave_gate import ISCLOSE_ATOL, ISCLOSE_RTOL, pre_dd_sigma

__all__ = [
    "first_float",
    "forge_sampling_offset",
    "is_flow_model",
    "is_img2img_request",
    "half_log_snr",
    "mark_republished",
    "republished_from",
    "sampler_publishes_sigmas",
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


def is_img2img_request(p) -> bool:
    """Is ``p`` a ``StableDiffusionProcessingImg2Img`` (or a subclass of it)?

    Checked by class name so the helper needs no ``modules.processing`` import."""
    return any(
        cls.__name__ == "StableDiffusionProcessingImg2Img"
        for cls in type(p).__mro__
    )


def forge_sampling_offset(p):
    """Index in Forge's ``sampling_sigmas`` of the first sigma this pass samples.

    txt2img's first pass walks the whole list (``KDiffusionSampler.sample``),
    so the offset is 0. img2img and the hires pass walk
    ``sigmas[steps - t_enc - 1:]`` (modules/sd_samplers_kdiffusion.py:145-148).
    Their ``steps, t_enc`` come from Forge's own ``setup_img2img_steps`` with
    the argument ``processing.py`` passes it: the hires pass uses
    ``hr_second_pass_steps or steps`` (:1552), img2img uses None (:1920).
    Returns None when the offset cannot be worked out; each caller says what
    it does then (Detail Daemon counts back like a run it did not set up,
    DAVE's gate uses the whole list, Colorcraft finds the run's first sigma).
    It is not counted back from the end with the sampler's step count,
    because some schedulers (Forge's ``ddim_scheduler``) return more than
    ``steps + 1`` sigmas.

    Moved here unchanged from scripts/anima_detail_daemon.py and
    scripts/anima_safe_pag.py, which had identical copies (2026-10-03)."""
    if getattr(p, "is_hr_pass", False):
        requested = getattr(p, "hr_second_pass_steps", 0) or getattr(p, "steps", None)
    elif is_img2img_request(p):
        requested = None
    else:
        return 0
    try:
        from modules import sd_samplers_common

        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, requested)
        return int(steps) - int(t_enc) - 1
    except Exception:
        return None


def sampler_publishes_sigmas(p) -> bool:
    """Does this pass's sampler set ``transformer_options['sampling_sigmas']``?

    Forge's k-diffusion samplers do, right before sampling
    (modules/sd_samplers_kdiffusion.py:192, 246). The CompVis timestep
    samplers (DDIM, PLMS — ``classic_ddim_eps_estimation``) never do,
    so a list found there was left by an earlier run. Same rule as
    ``_sampler_publishes_sigmas`` in scripts/anima_safe_pag.py."""
    denoiser = getattr(getattr(p, "sampler", None), "model_wrap_cfg", None)
    return not bool(getattr(denoiser, "classic_ddim_eps_estimation", False))


# The attribute ``mark_republished`` sets on a re-published copy (a Python attribute on the tensor).
REPUBLISHED_FROM = "_sam_extra_republished_from"


def mark_republished(sigmas, source):
    """Tag ``sigmas`` — ``source`` with some values patched, published in its place mid-run — as the
    same run's list. A copy of a copy points at the first list. Returns ``sigmas``."""
    original = republished_from(source)
    try:
        setattr(sigmas, REPUBLISHED_FROM, source if original is None else original)
    except (AttributeError, TypeError):
        pass
    return sigmas


def republished_from(sigmas):
    """The list ``sigmas`` was re-published in place of (``mark_republished``), else None."""
    return getattr(sigmas, REPUBLISHED_FROM, None)
