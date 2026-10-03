"""Sigma lists for the ``custom`` scheduler: strict parsing and Forge's log-linear interpolation.

Accepted text: numbers separated by commas and/or whitespace, optionally inside one pair of
``[]`` or ``()`` — e.g. ``[1.0, 0.6, 0.25, 0.1, 0.0]``. Each item must be a plain decimal literal
(``1``, ``0.25``, ``.5``, ``1e-3``); ``nan``, ``inf``, hex and anything else are rejected, and so are
negative values. Nothing is evaluated — every item is matched against a regular expression and
converted with ``float()``. At most ``MAX_VALUES`` numbers and ``MAX_TEXT_LENGTH`` characters.

How a list becomes ``n`` sigmas (plus the final 0 every Forge scheduler ends with):

* a list that starts with exactly 1.0 and ends with exactly 0.0 is *normalised*: each value v is
  scaled to ``sigma_min + v * (sigma_max - sigma_min)`` (1.0 → sigma_max, 0.0 → sigma_min);
* any other list is taken as sigmas as they are; a trailing 0.0 is read as the final zero and
  dropped before interpolation (it is appended again at the end);
* the remaining values must all be above 0 and are interpolated to ``n`` values with Forge's
  ``modules.sd_schedulers._loglinear_interp`` (the log-linear interpolation Forge uses for Align
  Your Steps) when their count differs from ``n`` — called at run time, not copied. A single step
  (``n == 1``) takes the first value: that interpolation would give the last one.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Sequence

MAX_TEXT_LENGTH = 20000
MAX_VALUES = 1000

# ASCII digits only (``\d`` would also take other scripts' digits, which ``float()`` accepts).
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_SEPARATORS = re.compile(r"[\s,]+")


class SigmaListError(ValueError):
    """The sigma list text cannot be used."""


def parse_sigma_list(text) -> list[float]:
    """Numbers of a sigma list text. Raises ``SigmaListError`` for anything that is not a list of
    at least two finite, non-negative numbers."""
    if not isinstance(text, str):
        raise SigmaListError("the sigma list must be text")
    body = text.strip()
    if len(body) > MAX_TEXT_LENGTH:
        raise SigmaListError(f"the sigma list is longer than {MAX_TEXT_LENGTH} characters")
    if body and body[0] in "[(":
        closing = "]" if body[0] == "[" else ")"
        if not body.endswith(closing):
            raise SigmaListError(f"the sigma list opens with {body[0]!r} but does not end with {closing!r}")
        body = body[1:-1]
    items = [item for item in _SEPARATORS.split(body.strip()) if item]
    if len(items) < 2:
        raise SigmaListError("a sigma list needs at least two numbers")
    if len(items) > MAX_VALUES:
        raise SigmaListError(f"a sigma list can have at most {MAX_VALUES} numbers")
    values = []
    for item in items:
        if not _NUMBER.fullmatch(item):
            raise SigmaListError(f"{item[:40]!r} is not a plain number")
        try:
            value = float(item)
        except (ValueError, OverflowError):
            raise SigmaListError(f"{item[:40]!r} is not a usable number") from None
        if not math.isfinite(value):
            raise SigmaListError(f"{item[:40]!r} is not finite")
        if value < 0.0:
            raise SigmaListError(f"{item[:40]!r} is negative; sigmas cannot be below 0")
        values.append(value)
    return values


def is_normalized(values: Sequence[float]) -> bool:
    return len(values) >= 2 and values[0] == 1.0 and values[-1] == 0.0


def forge_loglinear_interp() -> Callable:
    """Forge's ``modules.sd_schedulers._loglinear_interp``; ``SigmaListError`` outside Forge."""
    try:
        from modules import sd_schedulers

        interp = sd_schedulers._loglinear_interp
    except (ImportError, AttributeError) as exc:
        raise SigmaListError(
            "Forge's log-linear interpolation (modules.sd_schedulers._loglinear_interp) is not available: "
            f"{type(exc).__name__}"
        ) from None
    if not callable(interp):
        raise SigmaListError("Forge's modules.sd_schedulers._loglinear_interp is not callable")
    return interp


def sigmas_from_list(text, steps: int, sigma_min: float, sigma_max: float,
                     *, interp: Callable | None = None) -> list[float]:
    """``steps`` sigmas (without the final 0) for the sigma list ``text``.

    ``interp(values, count)`` is the log-linear interpolation; it defaults to Forge's own.
    """
    values = parse_sigma_list(text)
    if is_normalized(values):
        values = [float(sigma_min) + v * (float(sigma_max) - float(sigma_min)) for v in values]
    elif values[-1] == 0.0:
        values = values[:-1]   # the final zero; appended again after interpolation
    for index, value in enumerate(values):
        if not value > 0.0:
            raise SigmaListError(
                f"sigma #{index + 1} is {value:g}; sigmas before the end must be above 0 (a list that starts "
                "with 1.0 and ends with 0.0 is scaled between sigma_max and sigma_min instead)"
            )
    if steps <= 0:
        return []
    if steps == 1:
        # Forge's interpolation puts a single position at the list's last (smallest) value, so a
        # one-step generation would start from sigma_min. One step starts at the first value, like
        # the expression mode (x = 0) and Forge's Karras/Exponential.
        return [float(values[0])]
    if len(values) == steps:
        return [float(v) for v in values]
    if interp is None:
        interp = forge_loglinear_interp()
    result = [float(v) for v in interp(values, steps)]
    if len(result) != steps or not all(math.isfinite(v) and v > 0.0 for v in result):
        raise SigmaListError("the interpolated sigma list is not usable")
    return result
