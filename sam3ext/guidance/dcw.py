"""Differential Correction in Wavelet space (independent PyTorch rewrite)."""

from __future__ import annotations

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
) -> torch.Tensor:
    """Correct x0 bands toward the live latent, with an identity zero path."""
    if lambda_low == 0.0 and lambda_high == 0.0:
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
    corrected = tuple(
        clean_band
        + gain * _channel_energy_weight(live_band) * (live_band - clean_band)
        for clean_band, live_band, gain in zip(clean_bands, live_bands, gains)
    )
    result = haar_idwt2d(*corrected)[..., :height, :width]
    return result.to(dtype=original_dtype)
