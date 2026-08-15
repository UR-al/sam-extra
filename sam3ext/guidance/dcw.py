"""Differential Correction in Wavelet space (independent PyTorch rewrite)."""

from __future__ import annotations

import math

import torch

from .haar import (
    haar_dwt2d,
    haar_idwt2d,
    pad_even,
    safe_compute_dtype,
    sigma_norm,
)


def _channel_energy_weight(
    band: torch.Tensor,
    clamp_low: float = 0.25,
    clamp_high: float = 4.0,
) -> torch.Tensor:
    reduce_dims = tuple(range(2, band.ndim))
    energy = band.float().square().mean(dim=reduce_dims, keepdim=True)
    relative = energy / energy.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return relative.clamp(clamp_low, clamp_high).to(band.dtype)


def apply_dcw(
    denoised: torch.Tensor,
    x_t: torch.Tensor,
    sigma,
    lambda_low: float,
    lambda_high: float,
    *,
    rdc_tau: float = 0.0,
    rdc_alpha_ll: float = 0.03,
    rdc_alpha_hh: float = 0.0,
    rdc_state: dict | None = None,
) -> torch.Tensor:
    """Apply instantaneous DCW and optional cross-step RDC in one Haar pass.

    ``rdc_state`` belongs to one generation/pass.  The first step seeds a
    per-band EMA; later steps pull each corrected band back toward that EMA.
    A zero ``rdc_tau`` is an exact no-op and retains DCW's historical identity
    fast path.
    """
    tau = float(rdc_tau)
    alpha_ll = float(rdc_alpha_ll)
    alpha_hh = float(rdc_alpha_hh)
    if tau < 0.0 or alpha_ll < 0.0 or alpha_hh < 0.0:
        raise ValueError("RDC tau and alpha values must be non-negative")
    rdc_active = tau > 0.0 and rdc_state is not None
    if lambda_low == 0.0 and lambda_high == 0.0 and not rdc_active:
        return denoised
    if denoised.shape != x_t.shape or denoised.ndim not in (4, 5):
        raise ValueError(
            f"DCW requires matching 4-D/5-D tensors, got "
            f"{tuple(denoised.shape)} and {tuple(x_t.shape)}"
        )

    original_dtype = denoised.dtype
    compute_dtype = safe_compute_dtype(original_dtype)
    clean = denoised.to(dtype=compute_dtype)
    live = x_t.to(device=clean.device, dtype=compute_dtype)
    schedule = sigma_norm(sigma, clean)
    low_gain = float(lambda_low) * schedule
    high_gain = float(lambda_high) * (1.0 - schedule)
    middle_gain = (low_gain + high_gain) * 0.5

    clean_pad, (height, width) = pad_even(clean)
    live_pad, _ = pad_even(live)
    clean_bands = haar_dwt2d(clean_pad)
    live_bands = haar_dwt2d(live_pad)
    gains = (low_gain, middle_gain, middle_gain, high_gain)
    corrected = list(
        clean_band
        + gain * _channel_energy_weight(live_band) * (live_band - clean_band)
        for clean_band, live_band, gain in zip(clean_bands, live_bands, gains)
    )

    if rdc_active:
        schedule_value = (
            float(schedule.float().mean().item())
            if torch.is_tensor(schedule)
            else float(schedule)
        )
        previous_schedule = float(rdc_state.get("_s_prev", schedule_value))
        delta_schedule = abs(previous_schedule - schedule_value)
        beta = 1.0 - math.exp(-delta_schedule / max(tau, 1e-6))
        rdc_state["_s_prev"] = schedule_value

        middle_alpha = (alpha_ll + alpha_hh) * 0.5
        band_names = ("LL", "LH", "HL", "HH")
        alphas = (alpha_ll, middle_alpha, middle_alpha, alpha_hh)
        for index, (name, band, alpha) in enumerate(
            zip(band_names, corrected, alphas)
        ):
            if alpha == 0.0:
                continue
            detached = band.detach()
            previous_ema = rdc_state.get(name)
            if (
                not torch.is_tensor(previous_ema)
                or previous_ema.shape != detached.shape
                or previous_ema.device != detached.device
            ):
                rdc_state[name] = detached.to(dtype=compute_dtype).clone()
                continue
            previous_ema = previous_ema.to(
                device=detached.device,
                dtype=compute_dtype,
            )
            new_ema = (1.0 - beta) * previous_ema + beta * detached
            rdc_state[name] = new_ema.detach().clone()
            corrected[index] = band - alpha * (band - new_ema)

    result = haar_idwt2d(*corrected)[..., :height, :width]
    return result.to(dtype=original_dtype)
