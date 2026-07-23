"""Shared opt-in switch for noisy Guidance verification logs.

This module intentionally lives outside ``sam3ext`` so the independent
Guidance scripts do not import or initialize SAM3 processing code.
"""

from __future__ import annotations


_enabled = False


def set_guidance_diagnostics(enabled: bool) -> None:
    global _enabled
    _enabled = bool(enabled)


def guidance_diagnostics_enabled() -> bool:
    return bool(_enabled)
