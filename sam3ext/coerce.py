"""Defensive numeric coercion shared by the Refine and Anima UI handlers.

Gradio sliders return floats and dropdowns return their selected string, but
widget-order quirks or browser-side surprises can put an empty textbox value
where a number was expected. Rather than raise (which surfaces as a silently
broken button), fall back to the caller-supplied default.
"""

from __future__ import annotations

from typing import Any


def as_float(value: Any, default: float) -> float:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int) -> int:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return default
    try:
        return int(float(value))  # float-first so "12.0" works
    except (TypeError, ValueError):
        return default
