"""ER SDE (Reverse-time) and ER SDE (ODE): Forge's own ER-SDE solver with ComfyUI's other noise scalers.

Paper: Cui et al., "Elucidating the Solution Space of Extended Reverse-Time SDE for Diffusion
Models", arXiv:2309.06169. Every reverse-time SDE that keeps the model's marginals is picked out by
one noise scale function ``h(λ)`` (λ = σ/α); it enters the solver's step ratio
``r = h(λ_t)/h(λ_s)`` and the injected noise ``sqrt(λ_t² − λ_s²·r²)``.

Forge Neo's sampler list already has **ER SDE** — ``sample_er_sde`` in
``modules_forge/packages/k_diffusion/sampling.py`` (ComfyUI v0.3.75's code) with the paper's tuned
scaler ``h(λ) = λ·(e^(λ^0.3) + 10)`` and ``max_stage = 3``. These two entries call that same
function with the other two scalers of ComfyUI's ``SamplerER_SDE`` node:

* ``ER SDE (Reverse-time)``: ``h(λ) = λ^(η+1)`` — the classical reverse-time SDE at η = 1;
* ``ER SDE (ODE)``: ``h(λ) = λ`` — the noise term vanishes; the node also sets ``s_noise = 0`` so
  no noise is drawn at all.

Ported from ComfyUI ``comfy_extras/nodes_custom_sampler.py`` ``SamplerER_SDE``
(comfyanonymous/ComfyUI@36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508, lines 585-633; GPL-3.0, like this
extension). What is the node's, unchanged: the two scalers (``reverse_time_sde_noise_scaler``,
``ode_noise_scaler``), the rule ``solver_type == "ODE" or eta == 0`` → ODE with ``s_noise = 0``, and
the three keyword arguments handed to ``sample_er_sde`` (``s_noise``, ``noise_scaler``,
``max_stage``). Host changes (sam-extra, 2026-10-03): the node's ``ER-SDE`` choice is Forge's
built-in entry and is not repeated here; η and the stage come from the ``Extra Samplers`` accordion
(``params.py``) instead of node inputs and are clamped to the node's ranges; ``s_noise`` comes from
Forge's Settings → Sampler parameters (``Sigma noise``) for Reverse-time.
``tests/test_extra_samplers_er_sde_origin.py`` runs the node's verbatim code against these wrappers.
"""

from __future__ import annotations

import torch

from .common import k_sampling
from .params import DEFAULT_ETA, DEFAULT_MAX_STAGE, coerce_eta, coerce_max_stage

__all__ = [
    "ODE",
    "REVERSE_TIME",
    "er_sde_kwargs",
    "ode_noise_scaler",
    "reverse_time_sde_noise_scaler",
    "sample_er_sde_ode",
    "sample_er_sde_reverse_time",
]

# ComfyUI SamplerER_SDE ``solver_type`` names of the two scalers used here.
REVERSE_TIME = "Reverse-time SDE"
ODE = "ODE"


def reverse_time_sde_noise_scaler(eta: float):
    """``h(λ) = λ^(η+1)`` (SamplerER_SDE.execute, nodes_custom_sampler.py:609-610)."""

    def scaler(x: torch.Tensor) -> torch.Tensor:
        return x ** (eta + 1)

    return scaler


def ode_noise_scaler(x: torch.Tensor) -> torch.Tensor:
    """``h(λ) = λ`` (nodes_custom_sampler.py:612-613)."""
    return x


def er_sde_kwargs(solver_type: str, max_stage: int, eta: float, s_noise: float) -> dict:
    """The keyword arguments ``SamplerER_SDE.execute`` hands to ``sample_er_sde`` (lines 600-631)."""
    if solver_type not in (REVERSE_TIME, ODE):
        raise ValueError(f"unknown ER SDE solver type: {solver_type!r}")
    if solver_type == ODE or eta == 0:
        s_noise = 0.0
        solver_type = ODE
    noise_scaler = ode_noise_scaler if solver_type == ODE else reverse_time_sde_noise_scaler(eta)
    return {"s_noise": s_noise, "noise_scaler": noise_scaler, "max_stage": max_stage}


@torch.no_grad()
def sample_er_sde_reverse_time(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    s_noise=1.0, noise_sampler=None, max_stage=DEFAULT_MAX_STAGE, er_sde_eta=DEFAULT_ETA,
):
    """ER-SDE solver with ``h(λ) = λ^(η+1)``; η = 0 is the ODE (no noise drawn).

    ``er_sde_eta`` is deliberately not named ``eta``: Forge's ``Sampler.initialize`` passes its global
    ancestral eta (and writes ``Eta`` to the infotext) to every sampler with an ``eta`` parameter."""
    kwargs = er_sde_kwargs(REVERSE_TIME, coerce_max_stage(max_stage), coerce_eta(er_sde_eta), s_noise)
    return k_sampling().sample_er_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        noise_sampler=noise_sampler, **kwargs,
    )


@torch.no_grad()
def sample_er_sde_ode(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    noise_sampler=None, max_stage=DEFAULT_MAX_STAGE,
):
    """ER-SDE solver with ``h(λ) = λ`` and ``s_noise = 0`` — deterministic, never draws noise."""
    kwargs = er_sde_kwargs(ODE, coerce_max_stage(max_stage), 0.0, 0.0)
    return k_sampling().sample_er_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        noise_sampler=noise_sampler, **kwargs,
    )
