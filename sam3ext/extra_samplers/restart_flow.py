"""Restart (flow) — Restart sampling written from the paper, with the flow-model forward kernel.

Xu et al., "Restart Sampling for Improving Generative Processes", arXiv:2306.14878 (Algorithm 1): run a
deterministic ODE sampler (Heun) down the schedule; on reaching a low noise level ``t_min`` add forward noise
back up to ``t_max`` and solve the ODE down to ``t_min`` again, ``K`` times. The added noise contracts the
accumulated error while the ODE keeps the discretisation error small. This is sam-extra's own code. The
reference repository (Newbeeer/diffusion_restart_sampling) has no licence and was not opened; Forge's
built-in ``Restart`` (A1111's ``restart_sampler``, ``modules/sd_samplers_extra.py``) is only *executed* by a
test as the oracle of the ε case — nothing of it is copied.

Why a flow version. A1111's Restart adds VE noise ``x + ε·√(σ_max² − σ_min²)`` with ``t_max = 2``,
``t_min = 0.1`` in σ and rebuilds a Karras grid. On a flow model (``x = (1 − σ)·x0 + σ·n``: Anima, Flux, SD3)
that is wrong three ways: σ = 2 is beyond pure noise (σ ≤ 1), VE noise does not shrink the signal part, and
Karras ignores the model's shift. Here, with ``α = 1 − σ`` on flow (``prediction_type == "const"``) and
``α = 1`` otherwise, everything is placed in the ε-equivalent noise level ``s = σ/α`` (the VE σ of
``x/α``):

* **Window.** ``t_min``/``t_max`` are the paper's / A1111's ``s`` values ``RESTART_S_MIN = 0.1`` and
  ``RESTART_S_MAX = 2.0`` (on flow σ ∈ [0.0909, 0.667]). They are snapped to the scheduler's own grid:
  ``lo`` = the index 1 ≤ j ≤ n−1 whose ``s_j`` is closest to 0.1, ``hi`` = the index j < lo whose ``s_j`` is
  closest to 2 (σ = 1, where ``s`` is infinite, never wins); no restart when ``hi ≥ lo`` or ``s_hi ≤ s_lo``.
  The scheduler's σ are kept as they are (no Karras rebuild), so the shift is respected and σ never
  reaches 1.
* **Forward kernel.** From ``x`` at σ_lo to σ_hi: ``x ← (α_hi/α_lo)·x + α_hi·√(s_hi² − s_lo²)·ε·s_noise``,
  the exact transition of the flow forward process (it maps the model marginal ``N(α_lo·x0, σ_lo²)`` per
  x0 to ``N(α_hi·x0, σ_hi²)``); on ε/v models (α = 1) it is A1111's ``x + ε·s_noise·√(σ_hi² − σ_lo²)``.
* **Steps.** Every step is Forge's Heun step (``sample_heun`` at ``s_churn = 0``, Euler into σ = 0); the
  restart re-walks the grid's own steps ``hi … lo`` ``K`` times, ``K = RESTART_TIMES`` (1), or
  ``RESTART_TIMES_LONG`` (2) from ``RESTART_LONG_FROM`` (36) steps (A1111's rule). Below
  ``RESTART_MIN_STEPS`` (20) steps there is no restart and the sampler is exactly Forge's Heun.

Cost: Heun's two model calls per step plus ``K × (lo − hi)`` re-walked Heun steps — A1111's version keeps
the budget by shrinking its own Karras main grid, which a scheduler-preserving restart cannot do (28 steps
on Anima's shift-3 Simple-like grid: 10 re-walked steps, 75 model calls instead of 55).

Host bookkeeping. Noise is ``noise_sampler`` (default ``k_diffusion.sampling.default_noise_sampler(x)`` — Forge's
per-image ``TorchHijack`` stream, the one Forge's Restart draws from). The callback's ``i`` is the grid index
of the step being taken (the progress steps back during a restart and never passes 100 %; Forge's console
total bar counts the extra steps). Forge's CFG denoiser counts its calls in ``model.step`` for prompt
editing, Skip Early CFG / NGMS and the refiner switch; before each Heun step at grid index ``j`` this sets
``model.step = base + 2·j`` when it is an int, ``base`` being its value when the sampler was called — what
the main pass would have reached (a no-op there: Heun makes two calls per step), so a re-walked step is
scheduled as that step. Taking ``base`` from the counter itself also covers SPEED, which calls the sampler
once per resolution segment with the counter running on (``sam3ext/speed/runner.py``), without declaring
SPEED's ``sam_extra_step_offset`` (only the samplers with step-placed extras take it); SPEED plans the
restart per segment (a segment under 20 steps runs plain Heun).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .common import is_const, k_sampling, model_sampling

__all__ = [
    "RESTART_LONG_FROM",
    "RESTART_MIN_STEPS",
    "RESTART_S_MAX",
    "RESTART_S_MIN",
    "RESTART_TIMES",
    "RESTART_TIMES_LONG",
    "RestartPlan",
    "noise_level",
    "restart_kernel",
    "restart_plan",
    "sample_restart_flow",
]

# The restart window in the ε-equivalent noise level s = σ/α (the paper's and A1111's [0.1, 2] in VE σ).
RESTART_S_MIN = 0.1
RESTART_S_MAX = 2.0
RESTART_MIN_STEPS = 20       # fewer steps: no restart (A1111: ``steps >= 20``)
RESTART_TIMES = 1            # K
RESTART_TIMES_LONG = 2       # K from RESTART_LONG_FROM steps (A1111: ``steps >= 36`` → 2)
RESTART_LONG_FROM = 36


@dataclass(frozen=True)
class RestartPlan:
    """Restart after the step that reaches grid index ``lo``: jump to index ``hi`` and re-walk, ``times`` times."""

    lo: int
    hi: int
    times: int


def noise_level(sigma: float, flow: bool) -> float:
    """The ε-equivalent noise level ``s = σ/α`` (``α = 1 − σ`` on flow, 1 otherwise); infinite at α ≤ 0."""
    alpha = 1.0 - sigma if flow else 1.0
    return math.inf if alpha <= 0 else sigma / alpha


def restart_plan(sigmas, flow: bool) -> RestartPlan | None:
    """Where the restart goes on this grid (see the module docstring), or None (no restart)."""
    steps = len(sigmas) - 1
    if steps < RESTART_MIN_STEPS:
        return None
    levels = [noise_level(float(sigma), flow) for sigma in sigmas]
    lo = min(range(1, steps), key=lambda j: (abs(levels[j] - RESTART_S_MIN), j))
    candidates = [j for j in range(lo) if math.isfinite(levels[j])]
    if not candidates:
        return None
    hi = min(candidates, key=lambda j: (abs(levels[j] - RESTART_S_MAX), j))
    if hi >= lo or not levels[hi] > levels[lo]:
        return None
    times = RESTART_TIMES_LONG if steps >= RESTART_LONG_FROM else RESTART_TIMES
    return RestartPlan(lo=lo, hi=hi, times=times)


def restart_kernel(sigma_lo, sigma_hi, flow: bool) -> tuple:
    """``(scale, noise_std)`` of the forward jump σ_lo → σ_hi: ``x ← scale·x + noise_std·ε`` (before ``s_noise``)."""
    if not flow:
        return 1.0, (sigma_hi ** 2 - sigma_lo ** 2) ** 0.5
    alpha_lo, alpha_hi = 1 - sigma_lo, 1 - sigma_hi
    level_lo, level_hi = sigma_lo / alpha_lo, sigma_hi / alpha_hi
    return alpha_hi / alpha_lo, alpha_hi * (level_hi ** 2 - level_lo ** 2) ** 0.5


def _call_counter(model):
    """Forge's denoiser call counter (``CFGDenoiser.step``) when it is an int, else None."""
    step = getattr(model, "step", None)
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def _set_step(model, base, index: int) -> None:
    """The call counter at the start of the Heun step ``index`` of this call (two calls per step from ``base``)."""
    if base is not None:
        model.step = base + 2 * index


def _heun_step(model, x, sigmas, j: int, s_in, extra_args: dict, callback, to_d):
    """Forge's ``sample_heun`` step ``j`` at ``s_churn = 0`` (Euler into σ = 0)."""
    sigma_hat = sigmas[j]
    denoised = model(x, sigma_hat * s_in, **extra_args)
    d = to_d(x, sigma_hat, denoised)
    if callback is not None:
        callback({"x": x, "i": j, "sigma": sigmas[j], "sigma_hat": sigma_hat, "denoised": denoised})
    dt = sigmas[j + 1] - sigma_hat
    if sigmas[j + 1] == 0:
        x = x + d * dt
    else:
        x_2 = x + d * dt
        denoised_2 = model(x_2, sigmas[j + 1] * s_in, **extra_args)
        d_2 = to_d(x_2, sigmas[j + 1], denoised_2)
        d_prime = (d + d_2) / 2
        x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_restart_flow(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0,
                        noise_sampler=None):
    """Heun with Restart in the ε-equivalent noise level (module docstring)."""
    ks = k_sampling()
    extra_args = {} if extra_args is None else extra_args
    flow = is_const(model_sampling(model))
    noise_sampler = ks.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    base = _call_counter(model)
    plan = restart_plan(sigmas, flow)
    for i in ks.trange(len(sigmas) - 1, disable=disable):
        _set_step(model, base, i)
        x = _heun_step(model, x, sigmas, i, s_in, extra_args, callback, ks.to_d)
        if plan is None or i + 1 != plan.lo:
            continue
        sigma_lo, sigma_hi = sigmas[plan.lo], sigmas[plan.hi]
        for _ in range(plan.times):
            noise = noise_sampler(sigma_lo, sigma_hi)
            if flow:
                scale, std = restart_kernel(sigma_lo, sigma_hi, True)
                x = scale * x + std * noise * s_noise
            else:
                x = x + noise * s_noise * (sigma_hi ** 2 - sigma_lo ** 2) ** 0.5
            for j in range(plan.hi, plan.lo):
                _set_step(model, base, j)
                x = _heun_step(model, x, sigmas, j, s_in, extra_args, callback, ks.to_d)
    return x
