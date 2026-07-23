"""CNS-inspired live-wavelet recoloring for an existing noise sample."""

from __future__ import annotations

import torch

from .haar import haar_dwt2d, haar_idwt2d, pad_even


def _energy(band: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, band.ndim))
    return band.float().square().mean(dim=dims, keepdim=True).clamp_min(1e-8)


def color_noise_wavelet(
    noise: torch.Tensor,
    x_t: torch.Tensor,
    strength: float = 1.0,
    gamma_power: float = 0.5,
    gamma_scale: float = 3.0,
) -> torch.Tensor:
    """Recolor ``noise`` while preserving its global standard deviation."""
    if strength == 0.0:
        return noise
    if noise.shape != x_t.shape or noise.ndim not in (4, 5):
        raise ValueError(
            f"CNS requires matching 4-D/5-D tensors, got "
            f"{tuple(noise.shape)} and {tuple(x_t.shape)}"
        )
    if gamma_scale <= 0.0:
        raise ValueError("gamma_scale must be positive")

    original_dtype = noise.dtype
    white = noise.float()
    live = x_t.to(device=white.device, dtype=torch.float32)
    live_pad, (height, width) = pad_even(live)
    live_bands = haar_dwt2d(live_pad)
    energies = tuple(_energy(band) for band in live_bands)
    total = sum(energies)
    deficits = tuple(
        (1.0 - (energy / total / float(gamma_scale)).clamp(0.0, 1.0))
        .clamp_min(1e-8)
        .pow(float(gamma_power))
        for energy in energies
    )
    rms = (
        sum(weight.square() for weight in deficits) / len(deficits)
    ).sqrt().clamp_min(1e-8)
    weights = tuple(weight / rms for weight in deficits)

    white_pad, _ = pad_even(white)
    colored = haar_idwt2d(
        *(band * weight for band, weight in zip(haar_dwt2d(white_pad), weights))
    )[..., :height, :width]
    colored = colored * (
        white.std().clamp_min(1e-8) / colored.std().clamp_min(1e-8)
    )
    amount = min(1.0, max(0.0, float(strength)))
    mixed = torch.lerp(white, colored, amount)
    # Partial interpolation can lower variance even though both endpoints have
    # the same standard deviation. Re-apply the original energy budget after
    # mixing so every non-zero strength preserves the seeded sample's scale.
    mixed = mixed * (
        white.std().clamp_min(1e-8) / mixed.std().clamp_min(1e-8)
    )
    return mixed.to(original_dtype)
