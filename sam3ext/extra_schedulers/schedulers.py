"""Extra entries for Forge's "Schedule type" list.

Every function has Forge's scheduler signature — Forge calls
``scheduler.function(n=steps, sigma_min=..., sigma_max=..., device=devices.cpu)`` (plus ``rho`` for
a scheduler with a ``default_rho``) in ``KDiffusionSampler.get_sigmas`` — and returns ``n`` sigmas
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

The closed forms are evaluated in float64 and stored as float32. Laplace keeps ComfyUI's float32
torch arithmetic so its values match the node bit for bit.

No step may keep sigma where it was: a schedule with the same sigma twice in a row before the final 0
stops the generation with an ``ExtraSchedulerError``. Such a zero-length step is wasted with Euler and
makes multistep samplers such as Res Multistep or DPM++ 2M divide by zero — NaN, a black image
(measured with Laplace on Anima 3.8B). Karras Dynamic also refuses a rising step; a custom list may rise.

The Cosine, CosineExponential blend and Phi formulas are public math, written from their
descriptions; no code of the unlicensed extensions aoleg/Neo_ExtraSchedulers and
DenOfEquity/webUI_ExtraSchedulers was read or used.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from . import expression as expr
from . import settings as es_settings
from .sigma_list import SigmaListError, sigmas_from_list

GOLDEN_RATIO = expr.GOLDEN_RATIO
KARRAS_DYNAMIC_RHO = 7.0
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


def _check_no_flat_step(name: str, sigmas: torch.Tensor, sigma_min: float, sigma_max: float) -> None:
    step = _first_flat_step(sigmas)
    if step is not None:
        raise ExtraSchedulerError(
            f"Extra Schedulers: {name} gives sigma {float(sigmas[step]):.6g} at steps {step} and {step + 1} "
            f"(sigma_min {sigma_min:g}, sigma_max {sigma_max:g} — Settings → Sampler Parameters → sigma min / "
            f"sigma max); {ZERO_LENGTH_STEP}"
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


def karras_dynamic(n, sigma_min, sigma_max, rho=KARRAS_DYNAMIC_RHO, device="cpu"):
    sigma_min, sigma_max, rho, n = float(sigma_min), float(sigma_max), float(rho), int(n)
    _check_positive("Karras Dynamic", sigma_min, sigma_max)
    if not rho > 2.0:
        # The per-step exponent rho + 2cos(2πi/n) reaches rho - 2; at or below 0 the ramp has no meaning.
        raise KarrasDynamicError(
            f"Extra Schedulers: Karras Dynamic needs rho above 2 (its per-step exponent goes down to rho - 2), "
            f"got {rho:g}. Set rho to about 4 or more: {RHO_SOURCE}."
        )
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
            raise KarrasDynamicError(
                f"Extra Schedulers: Karras Dynamic gave no finite sigma at step {i} (rho {rho:g}, "
                f"sigma_min {sigma_min:g}, sigma_max {sigma_max:g})"
            )
        values.append(value)
    sigmas = _with_final_zero(values, device)
    _check_karras_dynamic_falls(sigmas, n, rho, sigma_min, sigma_max)
    return sigmas


def _check_karras_dynamic_falls(sigmas: torch.Tensor, n: int, rho: float, sigma_min: float, sigma_max: float) -> None:
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
            f"Extra Schedulers: Karras Dynamic cannot fall from sigma_max {sigma_max:g} to {relation} sigma_min "
            f"{sigma_min:g} (Settings → Sampler Parameters → sigma min / sigma max)"
        )
    if after == before:
        change = f"sigma stays at {before:.6g} at step {step} ({ZERO_LENGTH_STEP})"
    else:
        change = f"sigma rises from {before:.4g} to {after:.4g} at step {step}"
    raise KarrasDynamicError(
        f"Extra Schedulers: Karras Dynamic with rho {rho:g} does not fall over {n} steps — {change}. "
        f"Raise rho to about 4 or more: {RHO_SOURCE}."
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
