"""ER SDE (Reverse-time), ER SDE (ODE) and ER SDE (Tunable): Forge's own ER-SDE solver with ComfyUI's noise scalers.

Paper: Cui et al., "Elucidating the Solution Space of Extended Reverse-Time SDE for Diffusion
Models", arXiv:2309.06169. Every reverse-time SDE that keeps the model's marginals is picked out by
one noise scale function ``h(λ)`` (λ = σ/α); it enters the solver's step ratio
``r = h(λ_t)/h(λ_s)`` and the injected noise ``sqrt(λ_t² − λ_s²·r²)``.

Forge Neo's sampler list already has **ER SDE** — ``sample_er_sde`` in
``modules_forge/packages/k_diffusion/sampling.py`` (ComfyUI v0.3.75's code) with the paper's tuned
scaler ``h(λ) = λ·(e^(λ^0.3) + 10)`` and ``max_stage = 3``. These three entries call that same
function with the scalers of ComfyUI's ``SamplerER_SDE`` node:

* ``ER SDE (Reverse-time)``: ``h(λ) = λ^(η+1)`` — the classical reverse-time SDE at η = 1;
* ``ER SDE (ODE)``: ``h(λ) = λ`` — the noise term vanishes; the node also sets ``s_noise = 0`` so
  no noise is drawn at all;
* ``ER SDE (Tunable)``: ``h(λ) = λ·(e^(λ^0.3) + 10)^η`` — the node's ``ER-SDE`` solver type since
  ComfyUI 40e46c71 ("Extend ER-SDE noise scaler by scaling h(t)", PR #15428): the paper's scaler with
  its ``h²`` contribution scaled by η. At η = 1 it is Forge's built-in ``ER SDE`` bit for bit
  (``t ** 1.0`` is exact in torch), so the entry is a strict superset of the built-in one.

Reverse-time and Tunable can restrict their noise to a σ window (``er_sde_window``; sam-extra's own
code, the idea credited to pamparamm/ComfyUI-ppm): inside it the step uses the SDE scaler, outside the
ODE scaler, and no noise is drawn there. η = 0 (or the ODE entry) ignores the window.

Ported from ComfyUI ``comfy_extras/nodes_custom_sampler.py`` ``SamplerER_SDE``
(comfyanonymous/ComfyUI@36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508, lines 585-633, for the reverse-time and
ODE scalers; the ``ER-SDE`` scaler with η from comfyanonymous/ComfyUI@40e46c711025947f126cccaa1a692a14937a3096,
the same lines, byte-identical at both commits; GPL-3.0, like this extension). What is the node's,
unchanged: the three scalers (``er_sde_noise_scaler``, ``reverse_time_sde_noise_scaler``,
``ode_noise_scaler``), the rule ``solver_type == "ODE" or eta == 0`` → ODE with ``s_noise = 0``, and the
three keyword arguments handed to ``sample_er_sde`` (``s_noise``, ``noise_scaler``, ``max_stage``). Host
changes (sam-extra, 2026-10-03 and 2026-10-04): η and the stage come from the ``Extra Samplers``
accordion (``params.py``) instead of node inputs and are clamped to the node's ranges; ``s_noise`` comes
from Forge's Settings → Sampler parameters (``Sigma noise``) for Reverse-time and Tunable; the optional
noise window. ``tests/test_extra_samplers_er_sde_origin.py`` and
``tests/test_extra_samplers_er_sde_eta_origin.py`` run the node's verbatim code against these wrappers.
"""

from __future__ import annotations

import torch

from .common import k_sampling
from .er_sde_window import sample_er_sde_windowed
from .params import DEFAULT_ETA, DEFAULT_MAX_STAGE, coerce_eta, coerce_max_stage

__all__ = [
    "ER_SDE",
    "ODE",
    "REVERSE_TIME",
    "er_sde_kwargs",
    "er_sde_noise_scaler",
    "ode_noise_scaler",
    "reverse_time_sde_noise_scaler",
    "sample_er_sde_ode",
    "sample_er_sde_reverse_time",
    "sample_er_sde_tunable",
]

# ComfyUI SamplerER_SDE ``solver_type`` names of the three scalers.
ER_SDE = "ER-SDE"
REVERSE_TIME = "Reverse-time SDE"
ODE = "ODE"


def er_sde_noise_scaler(eta: float):
    """``h(λ) = λ·(e^(λ^0.3) + 10)^η`` (SamplerER_SDE.execute, nodes_custom_sampler.py:606-607 @ 40e46c71)."""

    def scaler(x: torch.Tensor) -> torch.Tensor:
        return x * ((x ** 0.3).exp() + 10.0) ** eta

    return scaler


def reverse_time_sde_noise_scaler(eta: float):
    """``h(λ) = λ^(η+1)`` (SamplerER_SDE.execute, nodes_custom_sampler.py:609-610)."""

    def scaler(x: torch.Tensor) -> torch.Tensor:
        return x ** (eta + 1)

    return scaler


def ode_noise_scaler(x: torch.Tensor) -> torch.Tensor:
    """``h(λ) = λ`` (nodes_custom_sampler.py:612-613)."""
    return x


_SDE_SCALERS = {ER_SDE: er_sde_noise_scaler, REVERSE_TIME: reverse_time_sde_noise_scaler}


def er_sde_kwargs(solver_type: str, max_stage: int, eta: float, s_noise: float) -> dict:
    """The keyword arguments ``SamplerER_SDE.execute`` hands to ``sample_er_sde`` (lines 600-631)."""
    if solver_type not in (ER_SDE, REVERSE_TIME, ODE):
        raise ValueError(f"unknown ER SDE solver type: {solver_type!r}")
    if solver_type == ODE or eta == 0:
        s_noise = 0.0
        solver_type = ODE
    noise_scaler = ode_noise_scaler if solver_type == ODE else _SDE_SCALERS[solver_type](eta)
    return {"s_noise": s_noise, "noise_scaler": noise_scaler, "max_stage": max_stage}


def _run(model, x, sigmas, extra_args, callback, disable, noise_sampler, kwargs: dict, window):
    """Forge's ``sample_er_sde`` with ``kwargs``; windowed when a window is given and the step is an SDE."""
    ks = k_sampling()
    if window is None or kwargs["noise_scaler"] is ode_noise_scaler:
        return ks.sample_er_sde(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
            noise_sampler=noise_sampler, **kwargs,
        )
    return sample_er_sde_windowed(
        ks, model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        s_noise=kwargs["s_noise"], noise_sampler=noise_sampler, sde_scaler=kwargs["noise_scaler"],
        ode_scaler=ode_noise_scaler, max_stage=kwargs["max_stage"], window=window,
    )


@torch.no_grad()
def sample_er_sde_reverse_time(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    s_noise=1.0, noise_sampler=None, max_stage=DEFAULT_MAX_STAGE, er_sde_eta=DEFAULT_ETA, er_sde_window=None,
):
    """ER-SDE solver with ``h(λ) = λ^(η+1)``; η = 0 is the ODE (no noise drawn).

    ``er_sde_eta`` is deliberately not named ``eta``: Forge's ``Sampler.initialize`` passes its global
    ancestral eta (and writes ``Eta`` to the infotext) to every sampler with an ``eta`` parameter.
    ``er_sde_window``: ``(start, end)`` sampling percentages of the noise window, or None (every step)."""
    kwargs = er_sde_kwargs(REVERSE_TIME, coerce_max_stage(max_stage), coerce_eta(er_sde_eta), s_noise)
    return _run(model, x, sigmas, extra_args, callback, disable, noise_sampler, kwargs, er_sde_window)


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


@torch.no_grad()
def sample_er_sde_tunable(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    s_noise=1.0, noise_sampler=None, max_stage=DEFAULT_MAX_STAGE, er_sde_eta=DEFAULT_ETA, er_sde_window=None,
):
    """ER-SDE solver with ``h(λ) = λ·(e^(λ^0.3) + 10)^η`` — Forge's built-in ``ER SDE`` at η = 1 without a
    window; η = 0 is the ODE (no noise drawn). Arguments as ``sample_er_sde_reverse_time``."""
    kwargs = er_sde_kwargs(ER_SDE, coerce_max_stage(max_stage), coerce_eta(er_sde_eta), s_noise)
    return _run(model, x, sigmas, extra_args, callback, disable, noise_sampler, kwargs, er_sde_window)
