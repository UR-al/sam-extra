"""Temporal Score Rescaling (TSR) — a post-CFG rescale of the predicted score by SNR.

Paper: Xu et al., "Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow
Models", arXiv 2510.01184. Ported from ComfyUI ``comfy_extras/nodes_eps.py``
``TemporalScoreRescaling`` (GPL-3.0, like this extension):

    snr = exp(2·half_log_snr(σ))
    r   = (snr·v + 1) / (snr·v/k + 1),   v = tsr_sigma²
    α   = σ·exp(half_log_snr(σ))          (flow: 1-σ, eps/v k-diffusion: 1)
    x0' = lerp(x/α, x0, r)                (derived from x0' = (x - r·σ·ε̂)/α)

``r → 1`` at high noise (no change) and ``r → k`` as the noise vanishes, so ``k < 1`` keeps a
little more of the predicted noise late in sampling — the node's tooltip: "Lower k produces more
detailed results; higher k produces smoother results". ``tsr_sigma`` sets how early the rescale
takes effect (larger = earlier). ``k = 1`` is an exact no-op (the input tensor is returned).

Host changes: Forge has no ``model_sampling``; the parameterisation comes from the KModel
predictor (``sigmas.is_flow_model``) and the half-log-SNR is computed directly
(``sigmas.half_log_snr``). The node compares a scalar sigma; here every batch row gets its own
``r`` and ``α`` (Forge passes one sigma per row), and a row with ``σ <= 0`` — or ``σ >= 1`` on a
flow model, where ``α = 0`` — is left unchanged (``r = 1``), the node's own "no rescaling" cases.
"""

from __future__ import annotations

import math

import torch

from .sigmas import half_log_snr

__all__ = ["DEFAULT_K", "DEFAULT_SIGMA", "rescale_factors", "apply_tsr"]

# ComfyUI node defaults (tsr_k 0.95, tsr_sigma 1.0; both 0.01..100).
DEFAULT_K = 0.95
DEFAULT_SIGMA = 1.0


def rescale_factors(sigma: float, k: float, tsr_sigma: float, flow: bool) -> tuple[float, float]:
    """``(r, α)`` for one sigma — ``(1.0, 1.0)`` where TSR leaves the row alone."""
    sigma = float(sigma)
    if k == 1.0 or not math.isfinite(sigma) or sigma <= 0.0:
        return 1.0, 1.0
    if flow and sigma >= 1.0:
        return 1.0, 1.0
    hls = half_log_snr(sigma, flow)
    snr = math.exp(2.0 * hls)
    variance = float(tsr_sigma) ** 2
    if snr == 0.0:
        return 1.0, 1.0
    if math.isinf(snr):
        return float(k), 1.0
    r = (snr * variance + 1.0) / (snr * variance / float(k) + 1.0)
    alpha = sigma * math.exp(hls)
    return r, alpha


def apply_tsr(denoised: torch.Tensor, x: torch.Tensor, sigma, *, k: float, tsr_sigma: float,
              flow: bool) -> torch.Tensor:
    """TSR on a ``[B, ...]`` x0 prediction; ``sigma`` is a scalar or one value per row."""
    if float(k) == 1.0:
        return denoised
    if denoised.shape != x.shape:
        raise ValueError(f"TSR needs matching tensors, got {tuple(denoised.shape)} and {tuple(x.shape)}")
    batch = denoised.shape[0]
    if torch.is_tensor(sigma):
        values = sigma.detach().float().flatten().cpu().tolist()
    else:
        values = [float(sigma)]
    if len(values) == 1:
        values = values * batch
    if len(values) != batch:
        raise ValueError(f"TSR got {len(values)} sigmas for a batch of {batch}")

    factors = [rescale_factors(value, k, tsr_sigma, flow) for value in values]
    if all(r == 1.0 for r, _alpha in factors):
        return denoised
    view = (batch,) + (1,) * (denoised.ndim - 1)
    r = torch.tensor([f[0] for f in factors], dtype=torch.float32, device=denoised.device).view(view)
    alpha = torch.tensor([f[1] for f in factors], dtype=torch.float32, device=denoised.device).view(view)
    clean = denoised.float()
    scaled_input = x.to(device=clean.device, dtype=torch.float32) / alpha
    out = torch.lerp(scaled_input, clean, r)
    return out.to(denoised.dtype)
