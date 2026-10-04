"""Extra entries for Forge's "Schedule type" list.

Every function has Forge's scheduler signature — Forge calls
``scheduler.function(n=steps, sigma_min=..., sigma_max=..., device=devices.cpu)`` (plus ``rho`` for
a scheduler with a ``default_rho`` and ``inner_model`` for one with ``need_inner_model``) in
``KDiffusionSampler.get_sigmas`` — and returns ``n`` sigmas
followed by the final 0, as a float32 tensor, like Forge's own schedulers. ``sigma_min`` /
``sigma_max`` are the model's (or the Settings overrides "sigma min"/"sigma max" Forge applies
before the call). With p = i / (n - 1) for step i = 0 … n - 1 (p = 0 when n == 1):

* **Cosine** — ``σmin + ½(σmax − σmin)(1 − cos(π(1 − √p)))``: σmax at p = 0, σmin at p = 1, a slow
  first drop.
* **CosineExponential blend** — ``(1 − p)·Cosine + p·Exponential``, where Exponential is
  k-diffusion's ``get_sigmas_exponential`` (``exp(linspace(log σmax, log σmin, n))``): starts as
  Cosine and ends with the exponential's long tail.
* **Phi** — ``σmin + (σmax − σmin)(1 − p)^(φ²)``, φ the golden ratio. The idea is Extraltodeus's
  "Golden Scheduler" (github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler, credited by the
  README of aoleg/Neo_ExtraSchedulers); credit only — that repository's code was not read.
* **Laplace** — ComfyUI's ``get_sigmas_laplace`` (below, GPL-3.0) with the accordion's μ/β, the
  schedule of Hang et al., "Improved Noise Schedule for Diffusion Training", arXiv:2407.03297. Where
  ComfyUI's clamp to [σmin, σmax] would repeat a sigma — on a flow model (σmax = 1) the default μ 0
  holds the first half of the steps at 1 — the n steps are spread evenly over the part of the curve
  inside that range instead (``_laplace_in_range``); a schedule without a repeat is ComfyUI's, bit for bit.
* **Karras Dynamic** — the Karras/EDM ramp (Karras et al., arXiv:2206.00364)
  ``(σmax^(1/ρᵢ) + p(σmin^(1/ρᵢ) − σmax^(1/ρᵢ)))^ρᵢ`` with a per-step exponent
  ``ρᵢ = ρ + 2cos(2πi/n)``, ρ = 7 (Forge's Settings "rho" replaces it like for Karras; ρ must be
  above 2 so every ρᵢ stays positive). Evaluated as ``σmax((1 − p) + p(σmin/σmax)^(1/ρᵢ))^ρᵢ``, the
  same value without overflow when ρᵢ is small, and exactly σmin at p = 1. A small ρᵢ lifts its step
  towards σmax, so below ρ ≈ 4 the sigmas rise mid-run for most step counts; such a schedule stops
  the generation with ``KarrasDynamicError`` ("raise rho to about 4 or more") instead of being used.
  The origin of this variant is unverified: it is known only from the README of an unlicensed
  extension, which credits "yoinked-h"; no code of it was read or used.
* **custom** — the accordion's safe expression (``expression.py``) or sigma list
  (``sigma_list.py``).
* **React Cosinusoidal DynSF** — ``σmax·((σmin + (σmax − σmin)·cos(πp/2)) / σmax)^(f·p)`` with the factor
  f = 2.15 (``REACT_DYNSF_FACTOR``): σmax at p = 0, σmax·(σmin/σmax)^f at p = 1. The formula is yoinked-h's,
  added to Panchovix/stable-diffusion-webui-reForge by commit 25a5fb38f (``ldm_patched/modules/samplers.py``,
  AGPL-3.0); it is written here from the published one-line formula — no reForge code was copied. reForge
  exposes f as a setting (default 2.15) and appends the final 0 in its ``ldm_patched`` variant, as here. A σ-range
  formula (the model's shift is ignored) that keeps many steps at high σ: on Anima (shift 3) with 28 steps 17 of
  them lie above σ 0.5 (median σ 0.720), 4 below 0.05, and the last σ before 0 is about 3.7e-6 — a nearly wasted
  step. Other factors: the custom expression ``M*((m+(M-m)*cos(x*pi/2))/M)**(f*x)`` (2.15 gives the same values).
  In float32 the first two σ become equal from 448 steps on Anima's range (489 on SDXL's 0.0291675…14.614642),
  which the flat-step rule below refuses.
* **Flow Cosmos rho7** — on a flow model, Karras et al.'s ρ-ramp laid on a VE-equivalent noise range σ̃ and mapped to
  rectified-flow time: ``s = (σ̃max^(1/ρ) + p(σ̃min^(1/ρ) − σ̃max^(1/ρ)))^ρ``, ``t = s / (1 + s)``. ρ, σ̃max and σ̃min
  are the accordion's Flow Cosmos values (``settings.active()``, read when the scheduler runs, like Laplace's μ/β);
  their defaults ρ 7 on σ̃ 0.002 … 80 are Cosmos-Predict2's schedule (Anima's parent model; ComfyUI's
  ``supported_models`` gives Cosmos the same EDM range): 80/81 ≈ 0.98765 at p = 0 and ≈ 0.001996 at p = 1. The
  model's own sigma range (and the Settings overrides) is not used — the list depends only on n and those three
  values. Scaling the σ̃ range by k is the flow time shift k (t' = k·t / (1 + (k − 1)·t) ⇔ σ̃' = k·σ̃, every s_i
  scales by k), so Anima's shift 3 corresponds to σ̃ 0.006 … 240 — a fact of the maths, not a quality claim. The
  name ``flow_cosmos_rho7`` (kept whatever ρ is) and the idea are KeithZ117/Comfyui-anima-sampler's (commit effba3c5,
  MIT: its ``build_flow_cosmos_rho_sigmas``); it is written here from the formula — none of that code was copied.
  The operations are those of the custom expression
  ``(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7/(1+(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7)`` (with the three
  values in place of 80, 0.002 and 7), so with the defaults the two are equal bit for bit. On an eps/v model (SD,
  SDXL) the flow time map has no meaning: there it is Forge's own Karras — ``k_diffusion.sampling.get_sigmas_karras``
  (k-diffusion's function as Forge ships it), called at run time with the accordion's ρ, not copied — on the
  σmin/σmax Forge passes (the model's range, or the Settings "sigma min" / "sigma max" overrides); the σ̃ range is
  not used there. One console line per process. The scheduler has no ``default_rho``, so neither Settings → rho nor
  a pasted "Schedule rho" reaches it: with the default ρ 7 the list equals what Forge's Karras gives only while that
  rho is 0 or 7. ρ must be finite and above 0, and on a flow model 0 < σ̃min < σ̃max, both finite — otherwise
  ``ExtraSchedulerError`` (the accordion's sliders keep ρ in 1 … 15, σ̃max in 1 … 1000, σ̃min in 0.0001 … 1, so only
  σ̃min = σ̃max = 1 gets there from the UI or the API). Like the custom scheduler's, every sigma before the final 0 must
  be a finite float32 of at least the smallest normal float32, and an overflow stops it too — on a flow model only values
  past the sliders get there (a tiny ρ or σ̃min cancels the ramp's end to 0), on an eps/v model also a very small
  Settings "sigma min" (Forge's float32 Karras then ends on 0 or a subnormal, e.g. ρ 1 with 1e-7 or ρ 7 with 1e-40),
  which Forge's samplers would divide by. Which branch is decided by the predictor
  Forge hands a ``need_inner_model`` scheduler (``sam3ext.guidance.sigmas.is_flow_model``:
  ``prediction_type == "const"``); without one it stops with ``ExtraSchedulerError``. With the defaults, two
  neighbouring flow times first become equal in float32 at 1,119,764 steps, which the flat-step rule below refuses
  (a narrow σ̃ range gets there much sooner; the message then names the three values).
* **Flow Cosmos Dynamic** — the flow-model version of Karras Dynamic, this extension's own composition of the two
  entries above: on a flow model, Karras Dynamic's ramp with its per-step exponent ``ρᵢ = ρ + 2cos(2πi/n)`` (the
  same float operations — ``_karras_dynamic_values``: the ratio form ``σ̃max((1 − p) + p(σ̃min/σ̃max)^(1/ρᵢ))^ρᵢ`` and
  exactly σ̃min at p = 1) laid on Flow Cosmos rho7's VE-equivalent range and mapped to flow time ``t = s / (1 + s)``.
  ρ, σ̃max and σ̃min are the same three accordion values as Flow Cosmos rho7's (defaults ρ 7 on σ̃ 0.002 … 80; no
  values of its own); the model's sigma range is not used. On an eps/v model it is Karras Dynamic itself on the
  σmin/σmax Forge passes, with the accordion's ρ instead of Settings → rho (one console line per process). Karras
  Dynamic's rules hold on both branches: ρ must be above 2 and the float32 list must fall at every step, else
  ``KarrasDynamicError`` (below ρ ≈ 4 it rises mid-run for most step counts — on σ̃ 0.002 … 80 every ρ up to 3.8
  does for some n ≤ 300, from 3.9 none); on a flow model the σ̃-range, float32-tiny and flat-step checks are Flow
  Cosmos rho7's, while an eps/v model gets Karras Dynamic's checks only (so a very small Settings "sigma min" that
  makes Karras Dynamic end on 0 or a subnormal runs, as with Karras Dynamic itself). Since ρᵢ = ρ only where
  cos(2πi/n) = 0, it equals Flow Cosmos rho7 (the same ρ and range) at the first
  step (σ̃max), the last one (σ̃min) and — when 4 divides n — steps n/4 and 3n/4 (the same value in another operation
  order: equal there in float32); elsewhere its σ̃ is lower where ρᵢ > ρ (the first and last quarter) and higher in
  the middle half, by the power-mean inequality. With the defaults it falls in float32 for every n up to 3000; two
  neighbouring flow times first become equal at 1,277,194 steps (steps 476/477, t ≈ 0.987626, near the start where ρᵢ
  ≈ ρ + 2; found by bisection between 1.2 and 1.35 million, with 1,277,193 and the 100 counts below it falling), which
  the flat-step rule refuses. Karras Dynamic's origin is unverified (see that entry); the flow version is not from any
  other project.

The closed forms are evaluated in float64 and stored as float32. Laplace keeps ComfyUI's float32
torch arithmetic so its values match the node bit for bit; Flow Cosmos rho7 on an eps/v model returns the
float32 tensor of Forge's Karras function itself, Flow Cosmos Dynamic there Karras Dynamic's.

No step may keep sigma where it was: a schedule with the same sigma twice in a row before the final 0
stops the generation with an ``ExtraSchedulerError``. Such a zero-length step is wasted with Euler and
makes multistep samplers such as Res Multistep or DPM++ 2M divide by zero — NaN, a black image
(measured with Laplace on Anima 3.8B). Karras Dynamic also refuses a rising step; a custom list may rise.

The Cosine, CosineExponential blend and Phi formulas are public math, written from their
descriptions; no code of the unlicensed extensions aoleg/Neo_ExtraSchedulers and
DenOfEquity/webUI_ExtraSchedulers was read or used.
"""
from __future__ import annotations

import importlib
import math
import sys
from typing import Callable, Iterable

import torch

from ..guidance.sigmas import is_flow_model
from . import expression as expr
from . import settings as es_settings
from .sigma_list import SigmaListError, sigmas_from_list

GOLDEN_RATIO = expr.GOLDEN_RATIO
KARRAS_DYNAMIC_RHO = 7.0
# React Cosinusoidal DynSF's exponent factor: reForge's default (its setting
# "reforge_react_cosinusoidal_dynsf_factor", 0.1-10); fixed here — other factors through the custom expression.
REACT_DYNSF_FACTOR = 2.15
# Flow Cosmos rho7: Cosmos-Predict2's EDM noise range and Karras's rho (module-level literals — the desktop app
# pins them by AST) — the defaults of the accordion's Flow Cosmos values (settings.FLOW_COSMOS_*_DEFAULT, the same
# numbers). On an eps/v model Forge's Karras gets the accordion's rho, on the sigma range Forge passes.
FLOW_COSMOS_SIGMA_MAX = 80.0
FLOW_COSMOS_SIGMA_MIN = 0.002
FLOW_COSMOS_RHO = 7.0
# Printed once per process the first time Flow Cosmos rho7 runs on a model that is not a flow model, formatted with
# the rho of that run (with the default 7: "... using Karras rho 7 on the model's own sigma range").
FLOW_COSMOS_NOT_FLOW_LOG = ("[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho {rho:g} on the "
                            "model's own sigma range")
# The same for Flow Cosmos Dynamic (its own once-per-process flag): on an eps/v model it runs Karras Dynamic.
FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG = ("[Extra Schedulers] Flow Cosmos Dynamic: not a flow model - using Karras Dynamic "
                                    "rho {rho:g} on the model's own sigma range")
# Where Forge's rho comes from (modules/shared_options.py "rho", infotext "Schedule rho"): the Settings
# page lists the Sampler Parameters only with --adv-samplers, and pasting an image that has
# "Schedule rho" sets it as an override. Karras shares it.
RHO_SOURCE = ("Settings → Sampler Parameters → rho (shared with Karras; listed with --adv-samplers), "
              "or a 'Schedule rho' override from pasted infotext")


class ExtraSchedulerError(ValueError):
    """A scheduler of this extension cannot make a usable schedule; the message is shown in Forge's UI
    (it stops the generation — nothing falls back to another schedule)."""


class CustomSchedulerError(ExtraSchedulerError):
    """The custom scheduler's expression or sigma list cannot be used (shown in Forge's UI)."""


class KarrasDynamicError(ExtraSchedulerError):
    """Karras Dynamic cannot make a falling schedule with this rho (shown in Forge's UI)."""


def _positions(n: int) -> list[float]:
    """p = i / (n - 1) for i = 0 … n - 1 (just [0.0] for a single step)."""
    if n <= 1:
        return [0.0] * max(n, 0)
    return [i / (n - 1) for i in range(n)]


def _with_final_zero(values: Iterable[float], device) -> torch.Tensor:
    return torch.tensor([*map(float, values), 0.0], dtype=torch.float32, device=device)


ZERO_LENGTH_STEP = ("a step that keeps sigma where it was is wasted with Euler and makes multistep samplers such as "
                    "Res Multistep divide by zero (NaN, a black image)")


def _first_flat_step(sigmas: torch.Tensor) -> int | None:
    """Index i of the first sigma equal to the next one, among the sigmas before the final 0."""
    head = sigmas[:-1]
    flat = torch.nonzero(head[1:] == head[:-1])
    return int(flat[0]) if flat.numel() else None


def _check_no_flat_step(name: str, sigmas: torch.Tensor, sigma_min: float, sigma_max: float, *,
                        source: str | None = None) -> None:
    """``source`` replaces the note on where the range comes from (default: Forge's sigma_min / sigma_max and the
    Settings that override them) for a list that does not come from that range (Flow Cosmos rho7 on a flow model)."""
    step = _first_flat_step(sigmas)
    if step is not None:
        if source is None:
            source = (f"sigma_min {sigma_min:g}, sigma_max {sigma_max:g} — Settings → Sampler Parameters → sigma min / "
                      f"sigma max")
        raise ExtraSchedulerError(
            f"Extra Schedulers: {name} gives sigma {float(sigmas[step]):.6g} at steps {step} and {step + 1} "
            f"({source}); {ZERO_LENGTH_STEP}"
        )


def _cosine_value(p: float, sigma_min: float, sigma_max: float) -> float:
    return sigma_min + 0.5 * (sigma_max - sigma_min) * (1.0 - math.cos(math.pi * (1.0 - math.sqrt(p))))


def _exponential_value(p: float, sigma_min: float, sigma_max: float) -> float:
    # exp(linspace(log σmax, log σmin, n))[i] — k-diffusion get_sigmas_exponential
    log_max = math.log(sigma_max)
    return math.exp(log_max + (math.log(sigma_min) - log_max) * p)


def _check_positive(name: str, sigma_min: float, sigma_max: float) -> None:
    if not (sigma_min > 0.0 and sigma_max > 0.0):
        raise ExtraSchedulerError(
            f"Extra Schedulers: {name} needs sigma_min and sigma_max above 0 (got {sigma_min:g} and {sigma_max:g})"
        )


def cosine(n, sigma_min, sigma_max, device="cpu"):
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    sigmas = _with_final_zero((_cosine_value(p, sigma_min, sigma_max) for p in _positions(int(n))), device)
    _check_no_flat_step("Cosine", sigmas, sigma_min, sigma_max)
    return sigmas


def cosine_exponential_blend(n, sigma_min, sigma_max, device="cpu"):
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    _check_positive("CosineExponential blend", sigma_min, sigma_max)
    values = (
        (1.0 - p) * _cosine_value(p, sigma_min, sigma_max) + p * _exponential_value(p, sigma_min, sigma_max)
        for p in _positions(int(n))
    )
    sigmas = _with_final_zero(values, device)
    _check_no_flat_step("CosineExponential blend", sigmas, sigma_min, sigma_max)
    return sigmas


def phi(n, sigma_min, sigma_max, device="cpu"):
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    exponent = GOLDEN_RATIO ** 2
    values = (sigma_min + (sigma_max - sigma_min) * (1.0 - p) ** exponent for p in _positions(int(n)))
    sigmas = _with_final_zero(values, device)
    _check_no_flat_step("Phi", sigmas, sigma_min, sigma_max)
    return sigmas


def _check_karras_dynamic_rho(rho: float, name: str, rho_source: str) -> None:
    if not rho > 2.0:
        # The per-step exponent rho + 2cos(2πi/n) reaches rho - 2; at or below 0 the ramp has no meaning.
        raise KarrasDynamicError(
            f"Extra Schedulers: {name} needs rho above 2 (its per-step exponent goes down to rho - 2), "
            f"got {rho:g}. Set rho to about 4 or more: {rho_source}."
        )


def _karras_dynamic_values(n: int, sigma_min: float, sigma_max: float, rho: float) -> tuple[list[float], int | None]:
    """Karras Dynamic's float64 ramp from ``sigma_max`` to ``sigma_min`` (module docstring), and the first step whose
    value is not finite (None when every one is; the list then stops before it). Shared by Karras Dynamic (Forge's σ
    range) and Flow Cosmos Dynamic (the σ̃ range, before the flow map)."""
    ratio = sigma_min / sigma_max
    values = []
    for i, p in enumerate(_positions(n)):
        if p >= 1.0:
            # The ramp ends at σmin whatever the exponent. Computed, (σmin/σmax)^(1/ρᵢ) underflows to 0
            # when ρᵢ is tiny (2 steps with rho just above 2: ρ₁ = rho − 2), which would give sigma 0.
            values.append(sigma_min)
            continue
        rho_i = rho + 2.0 * math.cos(2.0 * math.pi * i / n)
        # (σmax^(1/ρ) + p(σmin^(1/ρ) − σmax^(1/ρ)))^ρ written as σmax((1 − p) + p(σmin/σmax)^(1/ρ))^ρ — the
        # same value, without overflowing σmax^(1/ρ) when ρ is just above its minimum.
        try:
            value = sigma_max * ((1.0 - p) + p * ratio ** (1.0 / rho_i)) ** rho_i
        except OverflowError:
            value = math.inf
        if not math.isfinite(value):
            return values, i
        values.append(value)
    return values, None


def _karras_dynamic(n: int, sigma_min: float, sigma_max: float, rho: float, device, *, name: str,
                    rho_source: str) -> torch.Tensor:
    """Karras Dynamic on Forge's σ range; ``name`` and ``rho_source`` (where ρ comes from) go into the messages."""
    _check_positive(name, sigma_min, sigma_max)
    _check_karras_dynamic_rho(rho, name, rho_source)
    values, bad = _karras_dynamic_values(n, sigma_min, sigma_max, rho)
    if bad is not None:
        raise KarrasDynamicError(
            f"Extra Schedulers: {name} gave no finite sigma at step {bad} (rho {rho:g}, "
            f"sigma_min {sigma_min:g}, sigma_max {sigma_max:g})"
        )
    sigmas = _with_final_zero(values, device)
    _check_karras_dynamic_falls(sigmas, n, rho, sigma_min, sigma_max, name=name, rho_source=rho_source)
    return sigmas


def karras_dynamic(n, sigma_min, sigma_max, rho=KARRAS_DYNAMIC_RHO, device="cpu"):
    sigma_min, sigma_max, rho, n = float(sigma_min), float(sigma_max), float(rho), int(n)
    return _karras_dynamic(n, sigma_min, sigma_max, rho, device, name="Karras Dynamic", rho_source=RHO_SOURCE)


def react_cosinusoidal_dynsf(n, sigma_min, sigma_max, device="cpu"):
    """React Cosinusoidal DynSF: ``σmax·((σmin + (σmax − σmin)·cos(πp/2)) / σmax)^(f·p)``, f = ``REACT_DYNSF_FACTOR``.

    Written from the one-line formula (module docstring); float64 maths, float32 result, final 0 appended."""
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    _check_positive("React Cosinusoidal DynSF", sigma_min, sigma_max)
    values = (
        sigma_max * ((sigma_min + (sigma_max - sigma_min) * math.cos(math.pi * p / 2.0)) / sigma_max)
        ** (REACT_DYNSF_FACTOR * p)
        for p in _positions(int(n))
    )
    sigmas = _with_final_zero(values, device)
    _check_no_flat_step("React Cosinusoidal DynSF", sigmas, sigma_min, sigma_max)
    return sigmas


_flow_cosmos_not_flow_logged = False
_flow_cosmos_dynamic_not_flow_logged = False


def _flow_cosmos_times(n: int, rho: float = FLOW_COSMOS_RHO, ve_sigma_min: float = FLOW_COSMOS_SIGMA_MIN,
                       ve_sigma_max: float = FLOW_COSMOS_SIGMA_MAX) -> list[float]:
    """Flow times t = s / (1 + s) of the ρ-ramp s over the VE-equivalent range [ve_sigma_min, ve_sigma_max], float64
    (defaults: Cosmos-Predict2's ρ 7 on 0.002 … 80).

    Operation for operation the custom expression in the module docstring (x = i / (n − 1), 0 when n == 1), so
    the two give the same float64 values."""
    root_max = ve_sigma_max ** (1.0 / rho)
    root_min = ve_sigma_min ** (1.0 / rho)
    times = []
    for p in _positions(n):
        s = (root_max + p * (root_min - root_max)) ** rho
        times.append(s / (1.0 + s))
    return times


def forge_get_sigmas_karras() -> Callable:
    """Forge's "Karras": ``k_diffusion.sampling.get_sigmas_karras`` (``modules_forge/packages/k_diffusion/
    sampling.py``, k-diffusion's function as Forge ships it), resolved at call time and called, not copied — like
    ``sigma_list.forge_loglinear_interp``. ``ExtraSchedulerError`` outside Forge."""
    try:
        function = importlib.import_module("k_diffusion.sampling").get_sigmas_karras
    except (ImportError, AttributeError) as exc:
        raise ExtraSchedulerError(
            "Extra Schedulers: Flow Cosmos rho7 on an eps/v model uses Forge's Karras "
            f"(k_diffusion.sampling.get_sigmas_karras), which is not available: {type(exc).__name__}"
        ) from None
    if not callable(function):
        raise ExtraSchedulerError("Extra Schedulers: Forge's k_diffusion.sampling.get_sigmas_karras is not callable")
    return function


def _log_not_flow_once(rho: float) -> None:
    global _flow_cosmos_not_flow_logged
    if not _flow_cosmos_not_flow_logged:
        _flow_cosmos_not_flow_logged = True
        print(FLOW_COSMOS_NOT_FLOW_LOG.format(rho=rho), file=sys.stderr)


# Where Flow Cosmos rho7's own values come from (an API request, an XYZ axis or pasted infotext set the same fields).
FLOW_COSMOS_VALUES_SOURCE = "Extra Schedulers accordion → Flow Cosmos rho / sigma max / sigma min"


def _check_flow_cosmos_rho(rho: float) -> None:
    if not (math.isfinite(rho) and rho > 0.0):
        raise ExtraSchedulerError(
            f"Extra Schedulers: Flow Cosmos rho7 needs a finite rho above 0, got {rho:g} ({FLOW_COSMOS_VALUES_SOURCE})"
        )


def _check_flow_cosmos_range(ve_sigma_min: float, ve_sigma_max: float, name: str = "Flow Cosmos rho7") -> None:
    if not (math.isfinite(ve_sigma_min) and math.isfinite(ve_sigma_max) and 0.0 < ve_sigma_min < ve_sigma_max):
        raise ExtraSchedulerError(
            f"Extra Schedulers: {name} needs 0 < sigma~ min < sigma~ max, both finite, got sigma~ min "
            f"{ve_sigma_min:g} and sigma~ max {ve_sigma_max:g} ({FLOW_COSMOS_VALUES_SOURCE})"
        )


def _check_flow_cosmos_usable(sigmas: torch.Tensor, rho: float, where: str, name: str = "Flow Cosmos rho7") -> None:
    """Every sigma before the final 0 a finite float32 of at least ``_FLOAT32_TINY`` (the custom scheduler's rule).
    ``where`` names the range the list was laid on (and, on eps/v, the Settings that set it). Values past the sliders get
    here — a tiny ρ (both branches) or a σ̃ min far below float64's resolution of σ̃ max^(1/ρ) (flow) makes the ramp's
    subtraction cancel to 0 at p = 1, which the flat-step rule does not see when only the last sigma is 0, and a tiny ρ
    can make Forge's float32 Karras overflow to NaN — and, within the sliders, an eps/v model with a very small Settings
    "sigma min", where Forge's own float32 Karras ends on 0 (ρ 1 from about 1e-7) or on a subnormal (ρ 7 from about
    1e-38)."""
    head = sigmas[:-1]
    bad = torch.nonzero(~(torch.isfinite(head) & (head >= _FLOAT32_TINY)))
    if bad.numel():
        step = int(bad[0])
        raise ExtraSchedulerError(
            f"Extra Schedulers: {name} gives sigma {float(head[step]):.6g} at step {step} ({where}; rho "
            f"{rho:g} — {FLOW_COSMOS_VALUES_SOURCE}), not usable as one of Forge's float32 sigmas (the smallest usable "
            f"value is {_FLOAT32_TINY:.4g})"
        )


def flow_cosmos_rho7(n, sigma_min, sigma_max, inner_model=None, device="cpu"):
    """Flow Cosmos rho7 (module docstring): the ρ-ramp on the VE-equivalent range σ̃ in flow time on a flow model,
    otherwise Forge's own Karras (called at run time) with ρ on the sigma range Forge passes (the model's or the
    Settings overrides). ρ, σ̃ max and σ̃ min are the generation's accordion values (``settings.active()``), by
    default Cosmos-Predict2's ρ 7 on 0.002 … 80.

    ``inner_model`` is what Forge passes a ``need_inner_model`` scheduler (``KDiffusionSampler.model_wrap``, whose
    ``predictor`` tells a flow model from an eps/v one)."""
    current = es_settings.active()
    sigma_min, sigma_max, n = float(sigma_min), float(sigma_max), int(n)
    rho = float(current.flow_cosmos_rho)
    _check_positive("Flow Cosmos rho7", sigma_min, sigma_max)
    flow = is_flow_model(inner_model)
    if flow is None:
        raise ExtraSchedulerError(
            "Extra Schedulers: Flow Cosmos rho7 needs Forge's model (inner_model, whose predictor has a "
            "prediction_type) to tell a flow model from an eps/v one; none was given"
        )
    _check_flow_cosmos_rho(rho)
    if flow:
        ve_sigma_max, ve_sigma_min = float(current.flow_cosmos_sigma_max), float(current.flow_cosmos_sigma_min)
        _check_flow_cosmos_range(ve_sigma_min, ve_sigma_max)
        try:
            times = _flow_cosmos_times(n, rho, ve_sigma_min, ve_sigma_max)
        except OverflowError:
            raise ExtraSchedulerError(
                f"Extra Schedulers: Flow Cosmos rho7 with rho {rho:g} on sigma~ {ve_sigma_min:g} … {ve_sigma_max:g} "
                f"overflows ({FLOW_COSMOS_VALUES_SOURCE})"
            ) from None
        sigmas = _with_final_zero(times, device)
        _check_flow_cosmos_usable(sigmas, rho, f"sigma~ {ve_sigma_min:g} … {ve_sigma_max:g}")
        _check_no_flat_step("Flow Cosmos rho7", sigmas, sigma_min, sigma_max, source=(
            f"rho {rho:g}, sigma~ {ve_sigma_min:g} … {ve_sigma_max:g} — {FLOW_COSMOS_VALUES_SOURCE}; "
            "the model's sigma range is not used on a flow model"))
    else:
        karras = forge_get_sigmas_karras()
        _log_not_flow_once(rho)
        try:
            sigmas = karras(n, sigma_min, sigma_max, rho=rho, device=device)
        except OverflowError:   # sigma ** (1 / rho) for a ρ past the sliders, e.g. 14.6 ** 1000
            raise ExtraSchedulerError(
                f"Extra Schedulers: Flow Cosmos rho7 with rho {rho:g} on sigma {sigma_min:g} … {sigma_max:g} overflows "
                f"in Forge's Karras ({FLOW_COSMOS_VALUES_SOURCE})"
            ) from None
        _check_flow_cosmos_usable(sigmas, rho, f"Forge's Karras on sigma {sigma_min:g} … {sigma_max:g} — the model's "
                                               "range or Settings → Sampler Parameters → sigma min / sigma max")
        _check_no_flat_step("Flow Cosmos rho7", sigmas, sigma_min, sigma_max)
    return sigmas


FLOW_COSMOS_DYNAMIC = "Flow Cosmos Dynamic"   # the scheduler's name in its messages (registry.LABEL_FLOW_COSMOS_DYNAMIC)


def _flow_cosmos_dynamic_times(n: int, rho: float = FLOW_COSMOS_RHO, ve_sigma_min: float = FLOW_COSMOS_SIGMA_MIN,
                               ve_sigma_max: float = FLOW_COSMOS_SIGMA_MAX) -> tuple[list[float], int | None]:
    """Flow times t = s / (1 + s) of Karras Dynamic's ramp s over the VE-equivalent range [ve_sigma_min, ve_sigma_max],
    float64 (``_karras_dynamic_values`` — the same operations as Karras Dynamic), and the first step whose s is not
    finite (None when every one is)."""
    values, bad = _karras_dynamic_values(n, ve_sigma_min, ve_sigma_max, rho)
    return [s / (1.0 + s) for s in values], bad


def _log_dynamic_not_flow_once(rho: float) -> None:
    global _flow_cosmos_dynamic_not_flow_logged
    if not _flow_cosmos_dynamic_not_flow_logged:
        _flow_cosmos_dynamic_not_flow_logged = True
        print(FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG.format(rho=rho), file=sys.stderr)


def flow_cosmos_dynamic(n, sigma_min, sigma_max, inner_model=None, device="cpu"):
    """Flow Cosmos Dynamic (module docstring): Karras Dynamic's per-step-ρ ramp on the VE-equivalent range σ̃ in flow
    time on a flow model, otherwise Karras Dynamic itself on the sigma range Forge passes (the model's or the Settings
    overrides). ρ, σ̃ max and σ̃ min are Flow Cosmos rho7's accordion values (``settings.active()``), by default ρ 7 on
    0.002 … 80; Settings → rho does not reach it.

    ``inner_model`` is what Forge passes a ``need_inner_model`` scheduler (``KDiffusionSampler.model_wrap``, whose
    ``predictor`` tells a flow model from an eps/v one)."""
    current = es_settings.active()
    sigma_min, sigma_max, n = float(sigma_min), float(sigma_max), int(n)
    rho = float(current.flow_cosmos_rho)
    name = FLOW_COSMOS_DYNAMIC
    _check_positive(name, sigma_min, sigma_max)
    flow = is_flow_model(inner_model)
    if flow is None:
        raise ExtraSchedulerError(
            f"Extra Schedulers: {name} needs Forge's model (inner_model, whose predictor has a prediction_type) to "
            "tell a flow model from an eps/v one; none was given"
        )
    _check_karras_dynamic_rho(rho, name, FLOW_COSMOS_VALUES_SOURCE)
    if flow:
        ve_sigma_max, ve_sigma_min = float(current.flow_cosmos_sigma_max), float(current.flow_cosmos_sigma_min)
        _check_flow_cosmos_range(ve_sigma_min, ve_sigma_max, name)
        times, bad = _flow_cosmos_dynamic_times(n, rho, ve_sigma_min, ve_sigma_max)
        if bad is not None:
            raise KarrasDynamicError(
                f"Extra Schedulers: {name} gave no finite sigma~ at step {bad} (rho {rho:g}, sigma~ "
                f"{ve_sigma_min:g} … {ve_sigma_max:g} — {FLOW_COSMOS_VALUES_SOURCE})"
            )
        sigmas = _with_final_zero(times, device)
        _check_flow_cosmos_usable(sigmas, rho, f"sigma~ {ve_sigma_min:g} … {ve_sigma_max:g}", name)
        _check_no_flat_step(name, sigmas, sigma_min, sigma_max, source=(
            f"rho {rho:g}, sigma~ {ve_sigma_min:g} … {ve_sigma_max:g} — {FLOW_COSMOS_VALUES_SOURCE}; "
            "the model's sigma range is not used on a flow model"))
        _check_karras_dynamic_falls(sigmas, n, rho, ve_sigma_min, ve_sigma_max, name=name,
                                    rho_source=FLOW_COSMOS_VALUES_SOURCE)
    else:
        _log_dynamic_not_flow_once(rho)
        # exactly Karras Dynamic (D20): no rule of Flow Cosmos rho7's added — a very small Settings sigma min that
        # makes it end on 0 or a subnormal runs here as it does with Karras Dynamic
        sigmas = _karras_dynamic(n, sigma_min, sigma_max, rho, device, name=name, rho_source=FLOW_COSMOS_VALUES_SOURCE)
    return sigmas


def _check_karras_dynamic_falls(sigmas: torch.Tensor, n: int, rho: float, sigma_min: float, sigma_max: float, *,
                                name: str = "Karras Dynamic", rho_source: str = RHO_SOURCE) -> None:
    """Raise unless every sigma before the final 0 is below the one before it.

    rho > 2 only keeps every ρᵢ positive. A small ρᵢ pulls its step up towards σmax, so below
    rho ≈ 4 the schedule rises mid-run for most step counts (SDXL from rho ≈ 3.5, flow sigmas from
    ≈ 3.6, a σmax of 4500 from ≈ 3.85). From rho 4 no step was even flat in a sweep of 1–300 steps
    (rho 4–30 on those ranges and a narrow 0.5–0.75 one). Checked on the float32 values the sampler gets.
    """
    head = sigmas[:-1]
    not_falling = torch.nonzero(head[1:] >= head[:-1])
    if not_falling.numel() == 0:
        return
    step = int(not_falling[0]) + 1
    before, after = float(head[step - 1]), float(head[step])
    if sigma_min >= sigma_max:
        relation = "a larger" if sigma_min > sigma_max else "the same"
        raise KarrasDynamicError(
            f"Extra Schedulers: {name} cannot fall from sigma_max {sigma_max:g} to {relation} sigma_min "
            f"{sigma_min:g} (Settings → Sampler Parameters → sigma min / sigma max)"
        )
    if after == before:
        change = f"sigma stays at {before:.6g} at step {step} ({ZERO_LENGTH_STEP})"
    else:
        change = f"sigma rises from {before:.4g} to {after:.4g} at step {step}"
    raise KarrasDynamicError(
        f"Extra Schedulers: {name} with rho {rho:g} does not fall over {n} steps — {change}. "
        f"Raise rho to about 4 or more: {rho_source}."
    )


# ---------------------------------------------------------------------------
# Laplace — ComfyUI (GPL-3.0, the same license as this extension).
# origin: comfyanonymous/ComfyUI@36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508:comfy/k_diffusion/sampling.py:52-59,
# copied unchanged (tests/test_extra_schedulers_origin.py compares it with the verbatim copy in
# tests/_origin_comfyui_laplace.py). Unlike Forge's schedulers it returns n sigmas without a final
# 0 — ComfyUI's LaplaceScheduler node hands those n sigmas to SamplerCustom, which then stops at
# sigma_min. ``laplace`` below appends the 0 like every Forge scheduler.
# ---------------------------------------------------------------------------
def get_sigmas_laplace(n, sigma_min, sigma_max, mu=0., beta=0.5, device='cpu'):
    """Constructs the noise schedule proposed by Tiankai et al. (2024). """
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return sigmas


# The epsilon inside get_sigmas_laplace ("avoid log(0)"); _laplace_position inverts that same curve.
LAPLACE_EPSILON = 1e-5


def _laplace_position(sigma: float, mu: float, beta: float) -> float:
    """Where on ComfyUI's x axis (0 … 1) the Laplace curve passes ``sigma``, clamped to [0, 1] (beta > 0).

    The curve's log is mu - beta·log(2x + ε) below x = 0.5 and mu + beta·log(2 − 2x + ε) above it. Both
    exponents below are at most 0, so nothing overflows.
    """
    if not sigma > 0.0:
        return 1.0                          # the curve stays above 0 to its end
    target = math.log(sigma)
    if target >= mu:
        x = (math.exp((mu - target) / beta) - LAPLACE_EPSILON) / 2.0
    else:
        x = 1.0 - (math.exp((target - mu) / beta) - LAPLACE_EPSILON) / 2.0
    return min(max(x, 0.0), 1.0)


def _laplace_in_range(n: int, sigma_min: float, sigma_max: float, mu: float, beta: float, device) -> torch.Tensor:
    """ComfyUI's formula at n evenly spaced x over the part of the curve inside [sigma_min, sigma_max].

    get_sigmas_laplace spaces x over 0 … 1 and clamps; on a flow model (sigma_max 1) the default μ 0 puts
    e^μ at sigma_max, so the first half of the steps all get 1 — zero-length steps (Res Multistep returned
    a black image on Anima 3.8B). Here the same curve is sampled only where it is inside the range: there
    every x gives its own sigma.
    """
    start = _laplace_position(sigma_max, mu, beta)
    end = _laplace_position(sigma_min, mu, beta)
    x = torch.linspace(start, end, n, device=device)
    lmb = mu - beta * torch.sign(0.5 - x) * torch.log(1 - 2 * torch.abs(0.5 - x) + LAPLACE_EPSILON)
    return torch.clamp(torch.exp(lmb), min=sigma_min, max=sigma_max)


def laplace(n, sigma_min, sigma_max, device="cpu"):
    current = es_settings.active()
    n, sigma_min, sigma_max = int(n), float(sigma_min), float(sigma_max)
    mu, beta = float(current.laplace_mu), float(current.laplace_beta)
    sigmas = get_sigmas_laplace(n, sigma_min, sigma_max, mu=mu, beta=beta, device=device)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    if _first_flat_step(sigmas) is not None and beta > 0.0:
        spread = _laplace_in_range(n, sigma_min, sigma_max, mu, beta, device)
        sigmas = torch.cat([spread, spread.new_zeros([1])])
    step = _first_flat_step(sigmas)
    if step is not None:
        # beta 0 (one sigma, e^μ), a curve wholly above sigma_max or below sigma_min, or more steps than
        # float32 can tell apart on the part inside the range.
        top, bottom = math.exp(mu - beta * math.log(LAPLACE_EPSILON)), math.exp(mu + beta * math.log(LAPLACE_EPSILON))
        raise ExtraSchedulerError(
            f"Extra Schedulers: Laplace with mu {mu:g} and beta {beta:g} gives sigma {float(sigmas[step]):.6g} at "
            f"steps {step} and {step + 1} — its curve runs from {top:.4g} down to {bottom:.4g} (e^mu = "
            f"{math.exp(mu):.4g}) and too little of it lies between sigma_min {sigma_min:g} and sigma_max "
            f"{sigma_max:g} for {n} steps; {ZERO_LENGTH_STEP}. Raise beta, or move mu so that e^mu lies in that range."
        )
    return sigmas


# ---------------------------------------------------------------------------
# custom
# ---------------------------------------------------------------------------
def sigmas_from_expression(text, steps: int, sigma_min: float, sigma_max: float) -> list[float]:
    """``steps`` sigmas (without the final 0): the expression evaluated at each step."""
    compiled = expr.compile_expression(text)
    values = []
    for step in range(int(steps)):
        value = compiled.evaluate(**expr.step_variables(step, int(steps), sigma_min, sigma_max))
        if not value > 0.0:
            raise expr.ExpressionError(
                f"step {step} gives sigma {value:g}; every step's sigma must be above 0 — the final 0 is "
                "added after the last step (end at m, e.g. m + (M - m) * (1 - x) ** 3)"
            )
        values.append(value)
    return values


_FLOAT32_MAX = float(torch.finfo(torch.float32).max)
# The smallest normal float32. A sigma below it is stored as 0 or as a subnormal, and Forge's
# samplers divide by sigma (Euler's d = (x - denoised) / sigma), so a value that is "above 0" in
# float64 would still give inf/NaN mid-schedule.
_FLOAT32_TINY = float(torch.finfo(torch.float32).tiny)


def custom(n, sigma_min, sigma_max, device="cpu"):
    current = es_settings.active()
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    what = "sigma list" if current.custom_mode == es_settings.MODE_SIGMAS else "expression"
    try:
        if current.custom_mode == es_settings.MODE_SIGMAS:
            values = sigmas_from_list(current.custom_sigmas, int(n), sigma_min, sigma_max)
        else:
            values = sigmas_from_expression(current.custom_expression, int(n), sigma_min, sigma_max)
    except (expr.ExpressionError, SigmaListError) as exc:
        raise CustomSchedulerError(f"Extra Schedulers: the custom scheduler's {what} cannot be used: {exc}") from None
    for step, value in enumerate(values):
        if value > _FLOAT32_MAX:   # Forge's sigmas are float32; this would become inf
            raise CustomSchedulerError(
                f"Extra Schedulers: the custom scheduler's {what} gives sigma {value:g} at step {step}, "
                "too large for Forge's float32 sigmas"
            )
        if value < _FLOAT32_TINY:   # ... and this would become 0 (or a subnormal) before the last step
            raise CustomSchedulerError(
                f"Extra Schedulers: the custom scheduler's {what} gives sigma {value:g} at step {step}, "
                f"too small for Forge's float32 sigmas (the smallest usable value is {_FLOAT32_TINY:.4g})"
            )
    sigmas = _with_final_zero(values, device)
    step = _first_flat_step(sigmas)   # on the float32 values: two close float64 values can become one
    if step is not None:
        raise CustomSchedulerError(
            f"Extra Schedulers: the custom scheduler's {what} gives sigma {float(sigmas[step]):.6g} at steps "
            f"{step} and {step + 1}; {ZERO_LENGTH_STEP}"
        )
    return sigmas
