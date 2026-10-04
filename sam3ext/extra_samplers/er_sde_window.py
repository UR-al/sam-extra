"""The ER SDE noise window: SDE noise only on the steps that land inside a σ window, ODE steps elsewhere.

sam-extra's own code (2026-10-04). The idea of restricting ER-SDE's noise to a part of the schedule is
pamparamm/ComfyUI-ppm's ``er_sde_scheduled`` (AGPL-3.0); nothing of ppm was opened or copied — this
module is written from the ER-SDE paper and Forge's own solver.

Why it keeps the marginals: every step of Forge's ``sample_er_sde`` (Cui et al., arXiv:2309.06169) is a
consistent step of *some* marginal-preserving reverse-time SDE, picked by the noise scale function
``h(λ)`` used inside that step (its ratio ``r = h(λ_t)/h(λ_s)``, its two stage integrals and its noise
``sqrt(λ_t² − λ_s²·r²)``). Choosing ``h`` per step — the SDE scaler on the steps inside the window, the
ODE scaler ``h(λ) = λ`` (whose noise term is exactly zero) outside — therefore still follows the model's
marginals; outside the window no noise is drawn at all.

The window is given in sampling percentages of the model's own schedule (``start``, ``end`` in 0..1)
and turned into σ with the predictor's ``percent_to_sigma`` (Forge's ``Prediction`` /
``PredictionDiscreteFlow`` / ``PredictionFlux``, the same formulas as ComfyUI's model sampling): a step
``i`` is inside when ``percent_to_sigma(end) ≤ σ_{i+1} ≤ percent_to_sigma(start)`` — the σ the step lands
on, the σ its noise is drawn for. ``start ≥ end`` is an empty window (no step gets noise).

How, without copying the solver loop (one call of Forge's ``sample_er_sde`` with three per-call wrappers
that share the step index):

* a **callback wrapper** — ``sample_er_sde`` calls ``callback({"i": i, …})`` right after the model
  evaluation of step ``i`` and before its first use of ``noise_scaler``; the wrapper records ``i`` and
  forwards the same dictionary to the real callback (if any). It is always passed, so the index is
  always known (``tests/test_extra_samplers_er_sde_window.py`` pins that order in Forge's source);
* a **step-aware scaler** — the SDE scaler for a step inside the window, ``h(λ) = λ`` otherwise, for ``r``
  and both stage integrals of that step;
* a **gated noise sampler** — draws from the real sampler (Forge's per-image stream by default) only for
  a step inside the window and returns zeros otherwise, without drawing (``x + α·0·…`` leaves ``x``
  bit-identical; the ODE step's ``sqrt(…).nan_to_num()`` factor is finite).

ComfyUI calls ``callback`` before the scaler in its own identical ``sample_er_sde`` too, so the desktop
app's node pack can wrap ComfyUI's function the same way.
"""

from __future__ import annotations

import torch

__all__ = [
    "WindowedErSde",
    "in_window_steps",
    "sample_er_sde_windowed",
    "window_bounds",
]


def window_bounds(sampling, start: float, end: float) -> tuple:
    """``(sigma_hi, sigma_lo)`` of the window ``start..end`` (sampling percentages) on ``sampling``.

    Raises ``ValueError`` when the predictor has no ``percent_to_sigma``."""
    percent_to_sigma = getattr(sampling, "percent_to_sigma", None)
    if not callable(percent_to_sigma):
        raise ValueError("the ER SDE noise window needs the model's percent_to_sigma, which this predictor lacks")
    return float(percent_to_sigma(float(start))), float(percent_to_sigma(float(end)))


def in_window_steps(sigmas, start: float, end: float, sigma_hi: float, sigma_lo: float) -> list:
    """For each step ``i`` (``σ_i → σ_{i+1}``): does it land inside the window? ``start ≥ end`` → none."""
    steps = len(sigmas) - 1
    if float(start) >= float(end):
        return [False] * steps
    return [sigma_lo <= float(sigmas[i + 1]) <= sigma_hi for i in range(steps)]


class WindowedErSde:
    """The three wrappers of one windowed ``sample_er_sde`` call (see the module docstring)."""

    def __init__(self, inside: list, sde_scaler, ode_scaler, noise_sampler, like: torch.Tensor, callback=None):
        self.inside = list(inside)
        self.sde_scaler = sde_scaler
        self.ode_scaler = ode_scaler
        self.real_noise = noise_sampler
        self.like = like
        self.real_callback = callback
        self.step = None
        self.draws = 0

    def _inside(self) -> bool:
        if self.step is None:
            # Forge/ComfyUI call the callback before the scaler; if that ever changes, fail loudly
            # instead of silently using another step's choice.
            raise RuntimeError("ER SDE noise window: the scaler ran before the step's callback")
        return self.inside[self.step]

    def callback(self, info: dict) -> None:
        self.step = int(info["i"])
        if self.real_callback is not None:
            self.real_callback(info)

    def scaler(self, x: torch.Tensor) -> torch.Tensor:
        return self.sde_scaler(x) if self._inside() else self.ode_scaler(x)

    def noise_sampler(self, sigma, sigma_next) -> torch.Tensor:
        if self._inside():
            self.draws += 1
            return self.real_noise(sigma, sigma_next)
        return torch.zeros_like(self.like)


def sample_er_sde_windowed(
    ks, model, x, sigmas, *, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None,
    sde_scaler, ode_scaler, max_stage, window,
):
    """Forge's ``ks.sample_er_sde`` with the SDE scaler on the window's steps and the ODE scaler elsewhere.

    ``window`` is ``(start, end)`` in sampling percentages; the bounds come from
    ``model.inner_model.predictor.percent_to_sigma`` (``ValueError`` without it). ``noise_sampler`` None →
    Forge's ``default_noise_sampler(x)`` (the per-image stream ``sample_er_sde`` would build itself)."""
    start, end = float(window[0]), float(window[1])
    sigma_hi, sigma_lo = window_bounds(model.inner_model.predictor, start, end)
    inside = in_window_steps(sigmas, start, end, sigma_hi, sigma_lo)
    real_noise = ks.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    wrappers = WindowedErSde(inside, sde_scaler, ode_scaler, real_noise, x, callback=callback)
    return ks.sample_er_sde(
        model, x, sigmas, extra_args=extra_args, callback=wrappers.callback, disable=disable,
        s_noise=s_noise, noise_sampler=wrappers.noise_sampler, noise_scaler=wrappers.scaler, max_stage=max_stage,
    )
