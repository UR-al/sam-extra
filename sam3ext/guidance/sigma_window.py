"""Sigma window gate, ported from the Anima Safe PAG ComfyUI node.

origin: iljung1106/comfyui-anima-safe-pag@905b0107d1f924fc6acbcac3b6a879b566ff671c:__init__.py:9-34
(MIT License, Copyright (c) 2026 — full notice in THIRD_PARTY_NOTICES.md)

The node turns its ``start_percent``/``end_percent`` into a sigma window once,
when it patches the model, with the model's own ``percent_to_sigma``; every
model call is then tested against the *current sigma*, both ends inclusive.
That is what makes the window independent of the step count, the scheduler
and img2img denoise — a step-fraction gate is not.

Behaviour kept from the original:

* both percents are clamped to ``[0, 1]`` and swapped when reversed
  (:26-29), before ``percent_to_sigma`` is called (:31-33);
* the active test swaps reversed sigma bounds and includes both ends
  (:15-22), so ``sigma_end <= sigma <= sigma_start``;
* a tensor sigma is read from its first element (:9-12).

The only change is the host seam: the original asks a ComfyUI model for
``model_sampling``; here the caller passes the ``percent_to_sigma`` callable
(ComfyUI ``model_sampling.percent_to_sigma``, Forge ``KModel.predictor``). The
module is pure Python — no torch import — so gates can be tested without it.
"""

from __future__ import annotations

from typing import Callable, Tuple

__all__ = [
    "clamp_percent_range",
    "percent_range_to_sigmas",
    "sigma_active",
    "sigma_to_float",
]


def sigma_to_float(sigma) -> float:
    """First element of a tensor/array sigma, else ``float(sigma)`` (origin :9-12)."""
    if hasattr(sigma, "flatten") and hasattr(sigma, "shape"):
        return float(sigma.flatten()[0].item())
    return float(sigma)


def clamp_percent_range(start_percent, end_percent) -> Tuple[float, float]:
    """Clamp both percents to ``[0, 1]`` and put them in order (origin :26-29)."""
    start_percent = max(0.0, min(1.0, float(start_percent)))
    end_percent = max(0.0, min(1.0, float(end_percent)))
    if start_percent > end_percent:
        start_percent, end_percent = end_percent, start_percent
    return start_percent, end_percent


def percent_range_to_sigmas(
    percent_to_sigma: Callable[[float], float], start_percent, end_percent,
):
    """``(sigma_start, sigma_end, start_percent, end_percent)`` (origin :25-34).

    ``sigma_start`` belongs to the (smaller) start percent, so it is the upper
    bound of the window on a descending schedule. The sigmas are returned as
    ``percent_to_sigma`` gives them, like the original.
    """
    start_percent, end_percent = clamp_percent_range(start_percent, end_percent)
    sigma_start = percent_to_sigma(start_percent)
    sigma_end = percent_to_sigma(end_percent)
    return sigma_start, sigma_end, start_percent, end_percent


def sigma_active(sigma, sigma_start, sigma_end) -> bool:
    """``sigma_end <= sigma <= sigma_start``, bounds swapped if reversed (origin :15-22)."""
    sigma_start = sigma_to_float(sigma_start)
    sigma_end = sigma_to_float(sigma_end)
    if sigma_start < sigma_end:
        sigma_start, sigma_end = sigma_end, sigma_start

    value = sigma_to_float(sigma)
    return sigma_end <= value <= sigma_start
