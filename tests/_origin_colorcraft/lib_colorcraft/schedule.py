# Verbatim copy of muerrilla/ComfyUI-Colorcraft lib_colorcraft/schedule.py for the Colorcraft origin-parity tests
# (tests/test_colorcraft_origin.py). Test oracle only: the extension never imports it.
#
# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:lib_colorcraft/schedule.py
# https://github.com/muerrilla/ComfyUI-Colorcraft — MIT License, Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla).
# The MIT notice is tests/_origin_colorcraft/LICENSE (muerrilla/ComfyUI-Colorcraft's LICENSE at that commit, verbatim).
# Only this header was added; everything after the marker line is the file unchanged
# (tests/test_colorcraft_origin.py pins its SHA-256, so an accidental edit fails).
# ---- upstream lib_colorcraft/schedule.py below (verbatim) ----
import numpy as np

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
