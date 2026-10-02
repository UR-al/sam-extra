"""History-based guidance after CFG: Momentum Guidance (MG) and History-Guided Sampling (HiGS).

Both push the current guided prediction away from an average of the earlier steps, at no extra
model evaluation. They extrapolate along the same history, so turning both on double-counts it.

MG — "Momentum Guidance", arXiv 2602.20360 (flow models, Euler ODE). Paper notation
``X_t = t·X₁ + (1−t)·X₀`` (data at t = 1), velocity v, EMA m::

    Z_{i+1} = Z_i + Δt·[v_i + α(v_i − m_i)]        (Eq. 13)
    m_{i+1} = (1−β)·v_i + β·m_i,   m_{t₀} = v₀      (Eq. 12 — the EMA takes the *unmodified* v)

On a Forge flow model ``t = 1 − σ`` and ``v = (D − x)/σ``; returning
``D̃ = D + α·σ·(v − m)`` makes Forge's Euler step reproduce Eq. 13 exactly. The same formula on an
eps/v (k-diffusion) model is ε-momentum — the paper covers flow models only. ``normalize`` is the
paper's optional §8.2 rescale ``m ← (‖v‖₂/(‖m‖₂+ε))·m`` per sample (used for its SD3/FLUX tables).
No official code exists; pamparamm/sd-perturbed-attention ``mg_nodes.py`` (MIT) extrapolates x0
without the σ factor, so it is not this formula.

HiGS — "History-Guided Sampling for Plug-and-Play Enhancement of Diffusion Models",
arXiv 2509.22300 (ICLR 2026), x0 space, t ∈ [0, 1] with 1 = noise::

    g_k  = α·D_k + (1−α)·g_{k−1},  g = 0 before the first step   (zero-initialised, no bias fix)
    ΔD   = D_CFG − g
    ΔD^∥ = (⟨ΔD, D⟩/⟨D, D⟩)·D  (per sample, float64),  ΔD(η) = ΔD − ΔD^∥ + η·ΔD^∥
    w(t) = w_HiGS·√((t − t_min)/(t_max − t_min))  for t_min < t ≤ t_max, else 0
    H(R) = sigmoid(λ·(R − R_c)),  R = √(u² + v²), u = arange(H)/H, v = arange(W)/W  (orthonormal 2-D DCT)
    D_HiGS = D_CFG + w(t)·iDCT(H·DCT(ΔD(η)))

The first evaluation only seeds the history (``g = α·D₀``) and returns D unchanged; every later
evaluation computes its output first and then updates g with the unmodified D. Reimplemented from
the paper's equations and Algorithms 2–3 (its code is under arXiv's licence, not reused).
``t`` is σ on a flow model (the shifted schedule) and σ/(1+σ) on an eps/v model (the paper does not
say how it normalised SDXL's t; comfy-cfg-megapack's ``noise_level`` uses σ/(1+σ)).

Host rules shared by both (``step_role``): an evaluation whose sampler sigma is not on Forge's
``sampling_sigmas`` (a second-order midpoint) neither applies nor updates; a repeat of the last
sigma (Heun's corrector reused as the next predictor) applies without updating; a rising sigma
starts a new run (reset).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "HIGS_SHARPNESS",
    "HistoryState",
    "ROLE_NEW",
    "ROLE_REPEAT",
    "ROLE_SKIP",
    "apply_higs",
    "apply_mg",
    "dct_highpass",
    "higs_weight",
    "noise_level",
    "step_role",
]

ROLE_NEW = "new"        # apply, then update the history
ROLE_REPEAT = "repeat"  # apply with the current history, do not update it
ROLE_SKIP = "skip"      # leave the prediction and the history alone

# HiGS Appendix F / Algorithm 3: sharpness λ = 50, threshold R_c ≈ 0.05.
HIGS_SHARPNESS = 50.0
_EPS = 1e-12
_SIGMA_RTOL = 1e-4


@dataclass
class HistoryState:
    """One run's MG velocity EMA and HiGS prediction EMA (each None until seeded)."""

    mg_m: Any = None
    higs_g: Any = None
    last_sigma: float | None = None
    counters: dict = field(default_factory=lambda: {"mg": 0, "higs": 0})

    def reset(self) -> None:
        self.mg_m = None
        self.higs_g = None
        self.last_sigma = None


def noise_level(sigma: float, flow: bool) -> float:
    """t ∈ [0, 1] with 1 = noise: σ on a flow model, σ/(1+σ) on an eps/v model."""
    sigma = float(sigma)
    return sigma if flow else sigma / (1.0 + sigma)


def step_role(state: HistoryState, sigma: float | None, on_schedule: bool | None) -> str:
    """How this evaluation treats the history (see the module docstring); may reset ``state``."""
    if sigma is None or not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        return ROLE_SKIP
    if on_schedule is False:
        return ROLE_SKIP
    sigma = float(sigma)
    last = state.last_sigma
    if last is not None:
        if abs(sigma - last) <= _SIGMA_RTOL * max(abs(last), 1e-12):
            return ROLE_REPEAT
        if sigma > last:
            state.reset()   # sigma went back up: a new sampling run
    state.last_sigma = sigma
    return ROLE_NEW


def _per_sample_norm(t: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, t.ndim))
    return torch.linalg.vector_norm(t, dim=dims, keepdim=True)


def apply_mg(denoised: torch.Tensor, x: torch.Tensor, sigma: float, state: HistoryState, *,
             alpha: float, beta: float, normalize: bool, active: bool, role: str) -> torch.Tensor:
    """Momentum Guidance on one evaluation: ``D + α·σ·(v − m)``; updates ``state.mg_m`` on ``ROLE_NEW``.

    ``active`` is the window test (outside it the EMA keeps updating but nothing is applied).
    The first evaluation seeds ``m = v₀`` and returns ``denoised`` itself."""
    if role == ROLE_SKIP:
        return denoised
    sigma = float(sigma)
    clean = denoised.float()
    velocity = (clean - x.to(device=clean.device, dtype=torch.float32)) / sigma
    m = state.mg_m
    if not torch.is_tensor(m) or m.shape != velocity.shape:
        if role == ROLE_NEW:
            state.mg_m = velocity.detach()
        return denoised
    m = m.to(device=velocity.device, dtype=velocity.dtype)
    out = denoised
    if active and float(alpha) != 0.0:
        used = m
        if normalize:
            used = m * (_per_sample_norm(velocity) / (_per_sample_norm(m) + 1e-8))
        out = (clean + float(alpha) * sigma * (velocity - used)).to(denoised.dtype)
        state.counters["mg"] += 1
    if role == ROLE_NEW:
        state.mg_m = ((1.0 - float(beta)) * velocity + float(beta) * m).detach()
    return out


_DCT_CACHE: dict = {}


def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``C`` (``C @ x`` transforms the last axis of a column vector)."""
    key = (int(n), str(device), dtype)
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    matrix = torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n)) * math.sqrt(2.0 / n)
    matrix[0, :] = matrix[0, :] / math.sqrt(2.0)
    matrix = matrix.to(device=device, dtype=dtype)
    if len(_DCT_CACHE) > 16:
        _DCT_CACHE.clear()
    _DCT_CACHE[key] = matrix
    return matrix


def dct_highpass(t: torch.Tensor, cutoff: float, sharpness: float = HIGS_SHARPNESS) -> torch.Tensor:
    """``iDCT(H·DCT(t))`` over the last two axes, ``H = sigmoid(λ(R − R_c))`` (HiGS Eq. 8)."""
    height, width = int(t.shape[-2]), int(t.shape[-1])
    work = t.float()
    c_h = _dct_matrix(height, work.device, work.dtype)
    c_w = _dct_matrix(width, work.device, work.dtype)
    spectrum = c_h @ work @ c_w.transpose(0, 1)
    u = torch.arange(height, device=work.device, dtype=work.dtype) / height
    v = torch.arange(width, device=work.device, dtype=work.dtype) / width
    radius = torch.sqrt(u.unsqueeze(1) ** 2 + v.unsqueeze(0) ** 2)
    mask = torch.sigmoid(float(sharpness) * (radius - float(cutoff)))
    return c_h.transpose(0, 1) @ (spectrum * mask) @ c_w


def higs_weight(t: float, weight: float, t_min: float, t_max: float) -> float:
    """``w(t) = w_HiGS·√((t − t_min)/(t_max − t_min))`` for ``t_min < t ≤ t_max``, else 0 (Eq. 6)."""
    if not (t_min < t <= t_max) or t_max <= t_min:
        return 0.0
    return float(weight) * math.sqrt((t - t_min) / (t_max - t_min))


def apply_higs(denoised: torch.Tensor, sigma: float, state: HistoryState, *, weight: float,
               eta: float, alpha: float, cutoff: float, t_min: float, t_max: float,
               flow: bool, role: str) -> torch.Tensor:
    """HiGS on one evaluation; seeds/updates ``state.higs_g`` on ``ROLE_NEW``."""
    if role == ROLE_SKIP:
        return denoised
    clean = denoised.float()
    g = state.higs_g
    if not torch.is_tensor(g) or g.shape != clean.shape:
        if role == ROLE_NEW:
            state.higs_g = (float(alpha) * clean).detach()   # zero-initialised EMA after one update
        return denoised
    g = g.to(device=clean.device, dtype=clean.dtype)
    out = denoised
    w = higs_weight(noise_level(sigma, flow), weight, t_min, t_max)
    if w != 0.0:
        diff = (clean - g).double()
        reference = clean.double()
        dims = tuple(range(1, diff.ndim))
        dot = (diff * reference).sum(dim=dims, keepdim=True)
        norm = (reference * reference).sum(dim=dims, keepdim=True).clamp_min(_EPS)
        parallel = dot / norm * reference
        diff = (diff - parallel + float(eta) * parallel).float()
        out = (clean + w * dct_highpass(diff, cutoff)).to(denoised.dtype)
        state.counters["higs"] += 1
    if role == ROLE_NEW:
        state.higs_g = (float(alpha) * clean + (1.0 - float(alpha)) * g).detach()
    return out
