"""Per-generation values of the Extra Schedulers accordion and the state the schedulers read.

Forge calls a scheduler as ``scheduler.function(n=..., sigma_min=..., sigma_max=..., device=...)``
(modules/sd_samplers_kdiffusion.py ``KDiffusionSampler.get_sigmas``) without the processing
object, so a scheduler cannot look at ``p``. The always-on script (``scripts/anima_extra_schedulers.py``)
therefore stores the generation's values here in its ``process`` hook — before any sampling of
that generation — and the schedulers read them back with ``active()``. Forge runs one generation
at a time, and every generation (UI, API and XYZ cells alike) calls ``process`` with its own
arguments, so the state always belongs to the running generation. It is deliberately not reset in
``postprocess``: sampling started from other scripts' ``postprocess_image`` (ADetailer's inner
img2img, img2img-hires-fix) still belongs to the same generation and must see the same values.

Laplace ranges and defaults are those of ComfyUI's ``LaplaceScheduler`` node (mu -10..10 default
0, beta 0..10 default 0.5 — comfy_extras/nodes_custom_sampler.py:119-123 at 36c0b0a6, checked by
tests/test_extra_schedulers_origin.py).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace

MODE_EXPRESSION = "expression"
MODE_SIGMAS = "sigmas"
MODES = (MODE_EXPRESSION, MODE_SIGMAS)

# x = s / (n - 1): M at the first step, m at the last — the same curve as the Exponential
# schedule (k-diffusion get_sigmas_exponential).
DEFAULT_EXPRESSION = "M * (m / M) ** x"
# The example list of the custom scheduler's description: it starts at 1.0 and ends at 0.0, so it
# is scaled between sigma_max and sigma_min.
DEFAULT_SIGMAS = "[1.0, 0.6, 0.25, 0.1, 0.0]"

LAPLACE_MU_MIN, LAPLACE_MU_MAX, LAPLACE_MU_DEFAULT = -10.0, 10.0, 0.0
LAPLACE_BETA_MIN, LAPLACE_BETA_MAX, LAPLACE_BETA_DEFAULT = 0.0, 10.0, 0.5


@dataclass(frozen=True)
class ExtraSchedulerSettings:
    custom_mode: str = MODE_EXPRESSION
    custom_expression: str = DEFAULT_EXPRESSION
    custom_sigmas: str = DEFAULT_SIGMAS
    laplace_mu: float = LAPLACE_MU_DEFAULT
    laplace_beta: float = LAPLACE_BETA_DEFAULT


DEFAULTS = ExtraSchedulerSettings()


# ── coercion (UI, API, infotext and XYZ values all pass through these) ──

def coerce_float(value, default: float, low: float, high: float) -> float:
    """``value`` as a float clamped to [low, high]; ``default`` for anything that is not a finite number."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(high, max(low, number))


def coerce_laplace_mu(value) -> float:
    return coerce_float(value, LAPLACE_MU_DEFAULT, LAPLACE_MU_MIN, LAPLACE_MU_MAX)


def coerce_laplace_beta(value) -> float:
    return coerce_float(value, LAPLACE_BETA_DEFAULT, LAPLACE_BETA_MIN, LAPLACE_BETA_MAX)


def coerce_mode(value) -> str:
    """``expression``/``sigmas`` from the stored value or a UI label ("Expression", "Sigma list")."""
    text = str(value or "").strip().lower()
    if text in ("sigmas", "sigma list", "sigma_list", "list", "sigma"):
        return MODE_SIGMAS
    return MODE_EXPRESSION


def coerce_text(value, default: str) -> str:
    """The text as given (stripped). It is not shortened here: a text that is too long is rejected
    by the parser when the custom scheduler runs, instead of being cut into a different schedule."""
    if value is None:
        return default
    return str(value).strip()


def coerce(raw: dict) -> ExtraSchedulerSettings:
    """Settings from a dict of possibly missing or malformed values (never raises)."""
    raw = dict(raw or {})
    return ExtraSchedulerSettings(
        custom_mode=coerce_mode(raw.get("custom_mode", MODE_EXPRESSION)),
        custom_expression=coerce_text(raw.get("custom_expression"), DEFAULT_EXPRESSION),
        custom_sigmas=coerce_text(raw.get("custom_sigmas"), DEFAULT_SIGMAS),
        laplace_mu=coerce_laplace_mu(raw.get("laplace_mu", LAPLACE_MU_DEFAULT)),
        laplace_beta=coerce_laplace_beta(raw.get("laplace_beta", LAPLACE_BETA_DEFAULT)),
    )


_COERCERS = {
    "custom_mode": coerce_mode,
    "custom_expression": lambda value: coerce_text(value, DEFAULT_EXPRESSION),
    "custom_sigmas": lambda value: coerce_text(value, DEFAULT_SIGMAS),
    "laplace_mu": coerce_laplace_mu,
    "laplace_beta": coerce_laplace_beta,
}


def with_overrides(settings: ExtraSchedulerSettings, overrides) -> ExtraSchedulerSettings:
    """Apply XYZ-plot overrides (field name -> raw value) through the same coercion; unknown
    fields are ignored."""
    if not isinstance(overrides, dict):
        return settings
    changes = {field: _COERCERS[field](value) for field, value in overrides.items() if field in _COERCERS}
    return replace(settings, **changes) if changes else settings


# ── the running generation's values ──

_LOCK = threading.Lock()
_ACTIVE = DEFAULTS


def set_active(settings: ExtraSchedulerSettings) -> None:
    global _ACTIVE
    if not isinstance(settings, ExtraSchedulerSettings):
        settings = DEFAULTS
    with _LOCK:
        _ACTIVE = settings


def active() -> ExtraSchedulerSettings:
    with _LOCK:
        return _ACTIVE


def reset() -> None:
    set_active(DEFAULTS)
