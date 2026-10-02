"""HiFlow flow-aligned guidance for the hires pass, as a per-evaluation x0 replacement.

Paper: Bu et al., "HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned
Guidance", arXiv 2504.06232 (NeurIPS 2025). Origin code: Bujiazi/HiFlow
``flux_pipeline_hiflow.py`` / ``utils.py`` @ 31cc2b1 (Apache-2.0 — notice in THIRD_PARTY_NOTICES.md).
Modified by sam-extra, 2026-10-02 (Apache-2.0 §4(b)): the pipeline's velocity edit is rewritten as the
x0 replacement below and connected to Forge's hires fix; every change is listed in this docstring.
Where the paper and the released code differ, this follows the code:

* **reference** — the low-resolution pass's x0 prediction at the *same* noise level, bicubic-upsampled
  in latent space (``F.interpolate(..., mode="bicubic", align_corners=False)``);
* **direction alignment** — ``D = X + α·LPF(R − X)`` (paper Eq. 8; LPF linear), with the code's
  Butterworth mask ``1/(1 + (d²/D²)^4)`` on the centred 2-D FFT, ``d² = (2h/H−1)² + (2w/W−1)²``,
  cutoff D = 0.2 (the paper text says 0.4);
* **acceleration alignment** — the code edits the Euler velocity
  ``v̂ = v + β·(v_high_prev + u − u_prev − v)`` with ``u = (x − R)/σ`` and the *aligned* previous high
  velocity. Writing the returned x0 as ``O = x − σ·v̂`` makes x_t cancel, so it is exactly::

      O_0 = D_0
      O_k = D_k − β_k·[(D_k − R_k) − (σ_k/σ_{k−1})·(O_{k−1} − R_{k−1})]

  (any sampler that forms ``d = (x − O)/σ`` then gets the code's velocity);
* **weights** — ``α_k = A·(N−k)/N``, ``β_k = B·(N−k)/N`` over the N hires steps (code defaults
  A = 1.0, B = 0.5 for the 2K stage).

Host differences, all at the seam: the reference is looked up by the sampler's sigma and linearly
interpolated between recorded sigmas when the hires schedule is not a subset of the base one
(different steps, shift or denoise — ``sam3ext.guidance.trajectory``); initialization alignment is
Forge's own hires fix (upscale the base result, re-noise at the entry sigma), so it is not redone;
an evaluation off the hires schedule (a second-order midpoint) gets the direction term only and does
not touch the acceleration state; the model-side high-resolution tricks of the official runs (NTK
RoPE, proportional attention, swin padding) and the multi-stage cascade are not part of this stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "BUTTERWORTH_ORDER",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEFAULT_CUTOFF",
    "HiFlowState",
    "apply_hiflow",
    "butterworth_lowpass",
    "butterworth_mask",
    "resize_latent",
    "step_weight",
]

DEFAULT_ALPHA = 1.0     # run_hiflow.py alphas[0] (2K stage)
DEFAULT_BETA = 0.5      # run_hiflow.py betas[0]
DEFAULT_CUTOFF = 0.2    # run_hiflow.py filter_ratio (paper text: 0.4)
BUTTERWORTH_ORDER = 4   # utils.py butterworth_low_pass_filter n=4


@dataclass
class HiFlowState:
    """The previous hires step's ``(σ, O, R)`` for acceleration alignment."""

    prev_sigma: float | None = None
    prev_out: Any = None
    prev_ref: Any = None
    counters: dict = field(default_factory=lambda: {"direction": 0, "acceleration": 0})

    def reset(self) -> None:
        self.prev_sigma = None
        self.prev_out = None
        self.prev_ref = None


_MASK_CACHE: dict = {}


def butterworth_mask(height: int, width: int, cutoff: float, *, order: int = BUTTERWORTH_ORDER,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    """Centred Butterworth low-pass mask (origin ``utils.py``): ``1/(1 + (d²/D²)^n)``."""
    key = (int(height), int(width), float(cutoff), int(order), str(device), dtype)
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    h = torch.arange(height, dtype=torch.float64)
    w = torch.arange(width, dtype=torch.float64)
    d_square = (2.0 * h / height - 1.0).unsqueeze(1) ** 2 + (2.0 * w / width - 1.0).unsqueeze(0) ** 2
    cutoff = max(float(cutoff), 1e-6)
    mask = 1.0 / (1.0 + (d_square / cutoff ** 2) ** int(order))
    mask = mask.to(device=device, dtype=dtype)
    if len(_MASK_CACHE) > 16:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = mask
    return mask


def butterworth_lowpass(t: torch.Tensor, cutoff: float, *, order: int = BUTTERWORTH_ORDER) -> torch.Tensor:
    """Low band of ``t`` over its last two axes (fp32 FFT, like the origin)."""
    work = t.float()
    mask = butterworth_mask(work.shape[-2], work.shape[-1], cutoff, order=order, device=work.device)
    spectrum = torch.fft.fftshift(torch.fft.fft2(work, dim=(-2, -1)), dim=(-2, -1))
    low = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), dim=(-2, -1)).real
    return low


def resize_latent(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bicubic latent upsampling (``align_corners=False``) of a 4-D or 5-D latent to ``size`` (H, W)."""
    height, width = int(size[0]), int(size[1])
    work = t.float()
    if tuple(work.shape[-2:]) == (height, width):
        return work
    if work.ndim == 4:
        return F.interpolate(work, size=(height, width), mode="bicubic", align_corners=False)
    if work.ndim == 5:
        b, c, frames, h, w = work.shape
        flat = work.reshape(b, c * frames, h, w)
        out = F.interpolate(flat, size=(height, width), mode="bicubic", align_corners=False)
        return out.reshape(b, c, frames, height, width)
    raise ValueError(f"HiFlow expects a 4-D or 5-D latent, got {tuple(work.shape)}")


def step_weight(position: float | None, steps: int | None) -> float:
    """``(N − k)/N`` for (fractional) hires step ``k`` of ``N`` — 1.0 at the first step; 1.0 if unknown."""
    if position is None or not steps or steps <= 0:
        return 1.0
    return max(0.0, min(1.0, (float(steps) - float(position)) / float(steps)))


def apply_hiflow(denoised: torch.Tensor, sigma: float, reference: torch.Tensor, state: HiFlowState, *,
                 alpha: float, beta: float, cutoff: float, weight: float, step_start: bool) -> torch.Tensor:
    """One hires evaluation: direction alignment, then acceleration alignment on step starts.

    ``reference`` is R_k (already at the hires latent size), ``weight`` is ``(N−k)/N``.
    ``step_start`` False (a midpoint evaluation) applies the direction term only and leaves the
    acceleration state alone."""
    clean = denoised.float()
    ref = reference.to(device=clean.device, dtype=torch.float32)
    a = float(alpha) * float(weight)
    out = clean
    if a != 0.0:
        out = clean + a * butterworth_lowpass(ref - clean, cutoff)
        state.counters["direction"] += 1
    if not step_start:
        return out.to(denoised.dtype)
    b = float(beta) * float(weight)
    if (
        b != 0.0
        and torch.is_tensor(state.prev_out)
        and state.prev_out.shape == out.shape
        and state.prev_sigma
    ):
        prev_out = state.prev_out.to(device=out.device, dtype=torch.float32)
        prev_ref = state.prev_ref.to(device=out.device, dtype=torch.float32)
        ratio = float(sigma) / float(state.prev_sigma)
        out = out - b * ((out - ref) - ratio * (prev_out - prev_ref))
        state.counters["acceleration"] += 1
    state.prev_sigma = float(sigma)
    state.prev_out = out.detach()
    state.prev_ref = ref.detach()
    return out.to(denoised.dtype)
