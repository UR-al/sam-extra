"""CFG Wavelet Mixing and Sliding-Mode Control guidance transforms."""

from __future__ import annotations

from typing import Any

import torch

from .haar import (
    haar_dwt2d,
    haar_idwt2d,
    pad_even,
    safe_compute_dtype,
    sigma_norm,
)


SMC_NORM_EPS = 1e-8
SMC_DELTA_FLOOR = 1e-8

# Keep these names and values aligned with namemechan/ComfyUI-DCW.  ``Auto``
# resolves to one of the concrete presets at generation start, after Forge has
# attached the real model to the processing object.
SMC_PRESETS: dict[str, tuple[float, float]] = {
    "SD1.5 / SD2": (5.0, 0.10),
    "SDXL": (5.0, 0.10),
    "SD3 / SD3.5": (6.0, 0.10),
    "Flux": (6.0, 0.70),
    "Qwen-Image": (6.0, 0.10),
    "Cosmos / Wan": (6.0, 0.20),
    "Custom": (6.0, 0.10),
}
SMC_PRESET_NAMES: tuple[str, ...] = (
    "Off",
    "Auto",
    *SMC_PRESETS.keys(),
)


def normalize_smc_preset(value: Any) -> str:
    """Return a public SMC preset name, defaulting unknown input to ``Off``."""
    text = str(value or "").strip().casefold()
    aliases = {name.casefold(): name for name in SMC_PRESET_NAMES}
    return aliases.get(text, "Off")


def _model_candidates(model: Any) -> list[Any]:
    """Collect the small Forge/Comfy model wrapper chain without recursion."""
    if model is None:
        return []
    result: list[Any] = []
    pending = [model]
    seen: set[int] = set()
    while pending and len(result) < 24:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(current)
        for name in (
            "forge_objects",
            "unet",
            "model",
            "diffusion_model",
            "model_config",
        ):
            child = getattr(current, name, None)
            if child is not None and id(child) not in seen:
                pending.append(child)
    return result


def detect_smc_preset(model: Any) -> str:
    """Match ComfyUI-DCW's ``Auto`` model-family detection in Forge wrappers."""
    candidates = _model_candidates(model)
    names = " ".join(type(item).__name__.casefold() for item in candidates)

    # Preserve upstream's class-name priority.  Forge's Anima engine exposes
    # the top-level class as ``Anima`` and its DiT as a Cosmos/Predict2 family.
    if "flux" in names:
        return "Flux"
    if any(token in names for token in ("cosmos", "predict2", "wan", "anima")):
        return "Cosmos / Wan"
    if any(token in names for token in ("sd3", "mmdit")):
        return "SD3 / SD3.5"
    if "sdxl" in names:
        return "SDXL"
    if "qwen" in names:
        return "Qwen-Image"

    model_types = " ".join(
        str(getattr(item, "model_type", "") or "").casefold()
        for item in candidates
    )
    if "flow" in model_types:
        return "SD3 / SD3.5"
    if any(token in model_types for token in ("v_pred", "v-pred", "vprediction")):
        return "SD1.5 / SD2"
    return "SD1.5 / SD2"


def resolve_smc_preset(
    preset: Any,
    model: Any = None,
    *,
    custom_lambda: float = 6.0,
    custom_k: float = 0.10,
) -> tuple[str, str, float, float]:
    """Resolve ``(requested, concrete, lambda, k)`` for an SMC selection."""
    requested = normalize_smc_preset(preset)
    if requested == "Off":
        return requested, "Off", 0.0, 0.0
    concrete = detect_smc_preset(model) if requested == "Auto" else requested
    if concrete == "Custom":
        return requested, concrete, float(custom_lambda), float(custom_k)
    lambda_value, k_value = SMC_PRESETS[concrete]
    return requested, concrete, lambda_value, k_value


def _finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def apply_smc_error(
    error: torch.Tensor,
    previous,
    lambda_value: float,
    k_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply unit-L2 switching control and return ``(corrected, new_state)``."""
    # Match ComfyUI-DCW's public contract: either zero disables SMC. Although
    # lambda=0 could define a derivative-only controller mathematically, the
    # reference node deliberately uses it as an off sentinel.
    if lambda_value == 0.0 or k_value == 0.0:
        return error, error.detach()

    working = _finite_or_zero(error.float())
    if not torch.is_tensor(previous) or previous.shape != working.shape:
        previous_working = working.detach()
    else:
        previous_working = _finite_or_zero(
            previous.to(device=working.device, dtype=working.dtype)
        )
    surface = _finite_or_zero(
        (working - previous_working) + float(lambda_value) * previous_working
    )
    reduce_dims = tuple(range(1, surface.ndim))
    norm = torch.linalg.vector_norm(
        surface, dim=reduce_dims, keepdim=True
    ).clamp_min(SMC_NORM_EPS)
    raw_delta = -float(k_value) * (surface / norm)
    delta_limit = (
        0.5 * working.abs().mean(dim=reduce_dims, keepdim=True)
    ).clamp_min(SMC_DELTA_FLOOR)
    delta = raw_delta.clamp(-delta_limit, delta_limit)
    corrected = _finite_or_zero(working + delta)
    return corrected.to(error.dtype), corrected.detach()


SMC_MODE_UNIT = "Unit-L2"
SMC_MODE_ADAPTIVE = "Adaptive sign"
SMC_MODE_NAMES: tuple[str, ...] = (SMC_MODE_UNIT, SMC_MODE_ADAPTIVE)
# sorryhyun/anima_lora docs/inference/smc_cfg.md + library/inference/args.py (MIT): alpha 0.2, lambda 5.
SMC_ADAPTIVE_ALPHA = 0.2
SMC_ADAPTIVE_LAMBDA = 5.0
SMC_ADAPTIVE_FLOOR = 1e-12


def normalize_smc_mode(value: Any) -> str:
    """Public SMC mode name; unknown input keeps the original unit-L2 controller."""
    text = str(value or "").strip().casefold()
    if text.startswith("adaptive"):
        return SMC_MODE_ADAPTIVE
    return SMC_MODE_UNIT


def apply_smc_adaptive(
    error: torch.Tensor,
    sigma: float | None,
    previous,
    alpha: float,
    lambda_value: float,
) -> tuple[torch.Tensor, Any]:
    """Adaptive-gain sliding-mode CFG, ``(corrected_error, new_state)`` in denoised (x0) space.

    sorryhyun's Anima form of CFG-Ctrl (arXiv 2603.03281) in velocity space
    (sorryhyun/anima_lora ``library/inference/corrections/smc_cfg.py`` and
    sorryhyun/ComfyUI-Spectrum-KSampler ``smc_cfg.py``, both MIT)::

        e_t = v_c − v_u,  s_t = (e_t − e_prev) + λ·e_prev,  k_t = α·mean|e_t|
        Δe  = −k_t·sign(s_t),   v̂ = v_u + w·(e_t + Δe),   e_prev ← e_t (uncorrected)

    With ``x0 = x_t − σ·v`` the same controller in x0 space is exact when the stored error is
    rescaled by ``σ_t/σ_prev`` (velocity errors at different sigmas carry a 1/σ factor)::

        e = x0_c − x0_u,  e_prev' = e_prev·σ_t/σ_prev,  s = (e − e_prev') + λ·e_prev'
        d = −α·mean|e|·sign(s),   corrected = e + d

    (the sign flips twice and ``mean|e_v|·σ = mean|e_x0|``). The mean runs over the whole
    tensor like the original (one gain for the batch); the state keeps the *uncorrected* error
    like sorryhyun (the paper and the official code keep the corrected one). The first call, a
    shape change or an unknown sigma start from ``e_prev = e``. ``alpha == 0`` is a no-op that still
    records the state.
    """
    working = _finite_or_zero(error.float())
    state = previous if isinstance(previous, dict) else None
    sigma_now = None if sigma is None else float(sigma)
    if (
        state is None
        or not torch.is_tensor(state.get("e"))
        or tuple(state["e"].shape) != tuple(working.shape)
        or sigma_now is None
        or not state.get("sigma")
        or sigma_now <= 0.0
    ):
        e_prev = working
    else:
        e_prev = _finite_or_zero(
            state["e"].to(device=working.device, dtype=working.dtype)
        ) * (sigma_now / float(state["sigma"]))
    new_state = {"e": working.detach(), "sigma": sigma_now}
    if float(alpha) == 0.0:
        return error, new_state
    surface = (working - e_prev) + float(lambda_value) * e_prev
    gain = float(alpha) * working.abs().mean().clamp_min(SMC_ADAPTIVE_FLOOR)
    corrected = _finite_or_zero(working - gain * torch.sign(surface))
    return corrected.to(error.dtype), new_state


def apply_cwm_error(
    error: torch.Tensor,
    sigma,
    effective_scale: float,
    alpha_low: float,
    alpha_high: float,
) -> torch.Tensor:
    """Scale CFG error by scheduled Haar bands."""
    scale = float(effective_scale)
    if alpha_low == 0.0 and alpha_high == 0.0:
        return error * scale

    original_dtype = error.dtype
    working = _finite_or_zero(
        error.to(dtype=safe_compute_dtype(original_dtype))
    )
    schedule = sigma_norm(sigma, working)
    low_scale = scale * (1.0 + float(alpha_low) * schedule)
    high_scale = scale * (1.0 + float(alpha_high) * (1.0 - schedule))
    # Negative products have no real geometric mean. Fall back to an
    # arithmetic midpoint rather than emitting NaNs for experimental values.
    product = low_scale * high_scale
    middle_scale = torch.where(
        product >= 0,
        product.clamp_min(0).sqrt(),
        (low_scale + high_scale) * 0.5,
    ) if torch.is_tensor(product) else (
        product**0.5 if product >= 0 else (low_scale + high_scale) * 0.5
    )

    padded, (height, width) = pad_even(working)
    ll, lh, hl, hh = haar_dwt2d(padded)
    result = haar_idwt2d(
        ll * low_scale,
        lh * middle_scale,
        hl * middle_scale,
        hh * high_scale,
    )[..., :height, :width]
    return result.to(original_dtype)


def compose_cfg(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    sigma,
    effective_scale: float,
    mode: str,
    alpha_low: float,
    alpha_high: float,
    smc_lambda: float,
    smc_k: float,
    smc_previous,
    *,
    smc_mode: str = SMC_MODE_UNIT,
    smc_sigma: float | None = None,
    smc_alpha: float = SMC_ADAPTIVE_ALPHA,
) -> tuple[torch.Tensor, Any]:
    """Compose standard/CWM/SMC/SMC+CWM in denoised space.

    ``smc_mode`` picks the SMC controller: the unit-L2 one (``smc_lambda``/``smc_k``, namemechan
    contract) or the adaptive sign one (``smc_lambda``/``smc_alpha`` at the sampler's
    ``smc_sigma`` — :func:`apply_smc_adaptive`)."""
    error = _finite_or_zero(cond.float() - uncond.float())
    uncond_working = _finite_or_zero(uncond.float())
    next_previous = smc_previous
    if mode in {"smc", "smc+cwm"}:
        if smc_mode == SMC_MODE_ADAPTIVE:
            error, next_previous = apply_smc_adaptive(
                error, smc_sigma, smc_previous, smc_alpha, smc_lambda
            )
        else:
            error, next_previous = apply_smc_error(
                error, smc_previous, smc_lambda, smc_k
            )
    if mode in {"cwm", "smc+cwm"}:
        guided_error = apply_cwm_error(
            error, sigma, effective_scale, alpha_low, alpha_high
        )
    else:
        guided_error = error * float(effective_scale)
    return _finite_or_zero(uncond_working + guided_error), next_previous
