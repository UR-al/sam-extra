# Colorcraft (sam-extra) — per-step strength schedule and its lookup.
#
# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf
#   * make_schedule        lib_colorcraft/schedule.py:3-39   — unchanged
#   * sigma_to_value       nodes.py:35-59                    — unchanged (upstream adapted it from
#                          Jonseed/ComfyUI-Detail-Daemon, MIT)
#   * forge_step_position  scripts/colorcraft.py:930-935     — the Forge script's ``denoiser_callback``
#                          step counter as a pure function (see its docstring)
#
# MIT License
#
# Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes by sam-extra (2026-10-03): the two upstream functions are copied unchanged (their bodies
# are compared with the pinned upstream copy in tests/test_colorcraft_origin.py). The step counter
# is new code that reproduces the upstream Forge script's arithmetic for samplers that publish no
# sigma list (Forge's CompVis DDIM/PLMS); see hook.py.
"""Per-step strength schedule (``make_schedule``) and where a model call sits on it.

The ComfyUI node looks the schedule up by the sigma each model evaluation ran at
(``sigma_to_value``) against the sigma list its sampler was handed. The Forge hook does the same
with the part of Forge's ``sampling_sigmas`` that the pass actually walks
(``sam3ext.guidance.sigmas.forge_sampling_offset``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_schedule", "sigma_to_value", "forge_step_position"]


# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a:lib_colorcraft/schedule.py:3-39 (unchanged)
def make_schedule(steps, start, end, bias, amount, exponent, start_off, end_off, smooth):
    """Builds a length-`steps` array of per-step values, ramping from
    `start_off` up to `amount` and back down to `end_off` between `start`
    and `end`. Snapped to the nearest step, so it can come out slightly
    asymmetric at low step counts."""
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)
    start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    if exponent >= 0:
        start_values = start_values ** exponent
    else:
        start_values = 1 - (1 - start_values) ** abs(1 / exponent)
    if start_values.any():
        start_values *= (amount - start_off)
        start_values += start_off

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    if exponent >= 0:
        end_values = end_values ** exponent
    else:
        end_values = 1 - (1 - end_values) ** abs(1 / exponent)
    if end_values.any():
        end_values *= (amount - end_off)
        end_values += end_off

    multipliers[start_idx:mid_idx + 1] = start_values
    multipliers[mid_idx:end_idx + 1] = end_values
    multipliers[:start_idx] = start_off
    multipliers[end_idx + 1:] = end_off
    return multipliers


# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a:nodes.py:35-59 (unchanged)
def sigma_to_value(sigma, sigmas, schedule):
    """Maps the sigma a model eval actually ran at to a value from the
    discrete per-step schedule array. Adapted from Jonseed's ComfyUI
    port of Detail Daemon: https://github.com/Jonseed/ComfyUI-Detail-Daemon"""
    real_sigmas = sigmas[:-1]
    n = len(schedule)
    if n < 2 or len(real_sigmas) < 2 or sigma <= 0:
        return float(schedule[0]) if n else 0.0

    deltas = (real_sigmas - sigma).abs()
    idx = int(deltas.argmin())

    if (
        (idx == 0 and sigma >= real_sigmas[0])
        or (idx == n - 1 and sigma <= real_sigmas[-1])
        or deltas[idx] == 0
    ):
        return float(schedule[idx])

    idx_lo, idx_hi = (idx, idx - 1) if sigma > real_sigmas[idx] else (idx + 1, idx)
    sig_lo, sig_hi = real_sigmas[idx_lo], real_sigmas[idx_hi]
    if sig_hi == sig_lo:
        return float(schedule[idx_lo])
    ratio = float(max(0.0, min(1.0, (sigma - sig_lo) / (sig_hi - sig_lo))))
    return float(schedule[idx_lo] + ratio * (schedule[idx_hi] - schedule[idx_lo]))


def forge_step_position(sampling_step, total_sampling_steps, denoiser_step, denoiser_total_steps,
                        denoiser_steps):
    """``(current_step, actual_steps)`` the way upstream's Forge script counts a model call.

    origin: muerrilla/ComfyUI-Colorcraft@d28ac6a:scripts/colorcraft.py:930-935 — ``denoiser_callback``::

        step = max(params.sampling_step, params.denoiser.step)
        steps = max(params.total_sampling_steps, params.denoiser.total_steps)
        actual_steps = steps - max(steps // params.denoiser.steps - 1, 0)
        self.current_step = min(step, actual_steps - 1)

    Only used for samplers that publish no sigma list (Forge's CompVis DDIM/PLMS,
    ``classic_ddim_eps_estimation``) — everything else is looked up by sigma like the ComfyUI node.
    Returns ``(None, None)`` when the counts are unusable (no steps yet, a zero divisor).
    """
    try:
        step = max(int(sampling_step or 0), int(denoiser_step or 0))
        steps = max(int(total_sampling_steps or 0), int(denoiser_total_steps or 0))
        divisor = int(denoiser_steps or 0)
    except (TypeError, ValueError):
        return None, None
    if steps <= 0 or divisor <= 0:
        return None, None
    actual_steps = steps - max(steps // divisor - 1, 0)
    if actual_steps <= 0:
        return None, None
    return min(step, actual_steps - 1), actual_steps
