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

Flow Cosmos rho7's three values (ρ, and the VE-equivalent noise range σ̃ max / σ̃ min the ρ-ramp is laid on before
the flow map t = σ̃/(1 + σ̃)) default to Cosmos-Predict2's ρ 7 on 0.002 … 80 — ``schedulers.FLOW_COSMOS_RHO`` /
``FLOW_COSMOS_SIGMA_MAX`` / ``FLOW_COSMOS_SIGMA_MIN``, the same numbers (tests/test_extra_schedulers_schedulers.py
checks they agree), so the defaults give exactly Cosmos-Predict2's list (= the GPU A/B custom expression). The slider
ranges are this extension's choice: ρ 1 … 15 (step 0.1), σ̃ max 1 … 1000 (step 0.1), σ̃ min 0.0001 … 1 (step 0.0001).
Like Laplace's, a value that is not a finite number (also an integer too large for a float) becomes the default and one
outside its range is clamped to it; σ̃ min must also stay below σ̃ max (both can only meet at 1), which the scheduler
checks when it runs.
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

# Flow Cosmos rho7 (D19): (min, max, default, slider step) of ρ, σ̃ max and σ̃ min.
FLOW_COSMOS_RHO_MIN, FLOW_COSMOS_RHO_MAX, FLOW_COSMOS_RHO_DEFAULT, FLOW_COSMOS_RHO_STEP = 1.0, 15.0, 7.0, 0.1
FLOW_COSMOS_SIGMA_MAX_MIN, FLOW_COSMOS_SIGMA_MAX_MAX, FLOW_COSMOS_SIGMA_MAX_DEFAULT, FLOW_COSMOS_SIGMA_MAX_STEP = (
    1.0, 1000.0, 80.0, 0.1)
FLOW_COSMOS_SIGMA_MIN_MIN, FLOW_COSMOS_SIGMA_MIN_MAX, FLOW_COSMOS_SIGMA_MIN_DEFAULT, FLOW_COSMOS_SIGMA_MIN_STEP = (
    0.0001, 1.0, 0.002, 0.0001)


@dataclass(frozen=True)
class ExtraSchedulerSettings:
    custom_mode: str = MODE_EXPRESSION
    custom_expression: str = DEFAULT_EXPRESSION
    custom_sigmas: str = DEFAULT_SIGMAS
    laplace_mu: float = LAPLACE_MU_DEFAULT
    laplace_beta: float = LAPLACE_BETA_DEFAULT
    flow_cosmos_rho: float = FLOW_COSMOS_RHO_DEFAULT
    flow_cosmos_sigma_max: float = FLOW_COSMOS_SIGMA_MAX_DEFAULT
    flow_cosmos_sigma_min: float = FLOW_COSMOS_SIGMA_MIN_DEFAULT


DEFAULTS = ExtraSchedulerSettings()


# ── coercion (UI, API, infotext and XYZ values all pass through these) ──

def coerce_float(value, default: float, low: float, high: float) -> float:
    """``value`` as a float clamped to [low, high]; ``default`` for anything that is not a finite number.

    An integer too large for a float (an API request's JSON integer past ~1.8e308, where ``float()`` raises
    OverflowError) counts as not finite, like ``1e400`` or the same digits as text (both inf): it reads as the default
    instead of failing the whole request's coercion (SPEED's settings read it the same way)."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return min(high, max(low, number))


def coerce_laplace_mu(value) -> float:
    return coerce_float(value, LAPLACE_MU_DEFAULT, LAPLACE_MU_MIN, LAPLACE_MU_MAX)


def coerce_laplace_beta(value) -> float:
    return coerce_float(value, LAPLACE_BETA_DEFAULT, LAPLACE_BETA_MIN, LAPLACE_BETA_MAX)


def coerce_flow_cosmos_rho(value) -> float:
    return coerce_float(value, FLOW_COSMOS_RHO_DEFAULT, FLOW_COSMOS_RHO_MIN, FLOW_COSMOS_RHO_MAX)


def coerce_flow_cosmos_sigma_max(value) -> float:
    return coerce_float(value, FLOW_COSMOS_SIGMA_MAX_DEFAULT, FLOW_COSMOS_SIGMA_MAX_MIN, FLOW_COSMOS_SIGMA_MAX_MAX)


def coerce_flow_cosmos_sigma_min(value) -> float:
    return coerce_float(value, FLOW_COSMOS_SIGMA_MIN_DEFAULT, FLOW_COSMOS_SIGMA_MIN_MIN, FLOW_COSMOS_SIGMA_MIN_MAX)


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
        flow_cosmos_rho=coerce_flow_cosmos_rho(raw.get("flow_cosmos_rho", FLOW_COSMOS_RHO_DEFAULT)),
        flow_cosmos_sigma_max=coerce_flow_cosmos_sigma_max(
            raw.get("flow_cosmos_sigma_max", FLOW_COSMOS_SIGMA_MAX_DEFAULT)),
        flow_cosmos_sigma_min=coerce_flow_cosmos_sigma_min(
            raw.get("flow_cosmos_sigma_min", FLOW_COSMOS_SIGMA_MIN_DEFAULT)),
    )


_COERCERS = {
    "custom_mode": coerce_mode,
    "custom_expression": lambda value: coerce_text(value, DEFAULT_EXPRESSION),
    "custom_sigmas": lambda value: coerce_text(value, DEFAULT_SIGMAS),
    "laplace_mu": coerce_laplace_mu,
    "laplace_beta": coerce_laplace_beta,
    "flow_cosmos_rho": coerce_flow_cosmos_rho,
    "flow_cosmos_sigma_max": coerce_flow_cosmos_sigma_max,
    "flow_cosmos_sigma_min": coerce_flow_cosmos_sigma_min,
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
