# Derived from (MIT License, notices below):
#   howardhx/speed@ca7801c9bdffe681742e9592345bcf4885959be5 utils.py:18-77, 252-309
#     (power_spectrum, activation_time, delta_optimal_transitions, find_first_step_below,
#     validate_scales) and comfyui/speed_sampler.py:44-70, 155-197 (presets, parsing,
#     _resolve_transitions, _resolve_manual) — Copyright (c) 2026 Howard Xiao
#   aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 speed_core.py:210-529 (presets with
#     ref_latent/ref_shift, MAX_COARSE_FRACTION, adaptive delta, sigma_divisor / neo_shift,
#     _resolve_transitions, _resolve_manual, _parse_scales, _parse_sigmas) and spectral_utils.py:15-127
#     (equivalent_delta, reference_coarse_fraction) — Copyright (c) 2026 A. Izzuddin Al Faruq
#     (the presets, adaptive delta and neo_shift in that fork are Oleg Afonin's work)
#   sorryhyun/ComfyUI-Spectrum-KSampler@b46a364aec3b161b889c9cc26cd976a49eb537ae spd.py:62-104,
#     138-198 (resolve_spd_schedule, the respace loop's sigma arithmetic) — Copyright (c) 2026 sorryhyun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Modified by sam-extra, 2026-10-03 (part of a GPL-3.0-only extension; the MIT notices stay):
#   * aoleg's ``_resolve_transitions`` is split into ``delta_threshold_plan`` (sigma* + adaptive
#     pinning + divisor + step caps) and ``static_transitions`` (the step lookup), so the same
#     thresholds also drive the respace plan; the arithmetic and the caps are unchanged;
#   * a threshold the divisor pushes to >= 1 gets no reference-share cap (aoleg raises inside
#     ``reference_coarse_fraction`` there and the sampling run fails);
#   * ``respace_transitions`` replays sorryhyun's loop on the sigmas alone (the re-spacing depends
#     on nothing else), so the whole plan is known before the first model call;
#   * both plans drop stages whose transition fires at step 0 instead of expanding before any
#     coarse step (upstream's img2img/hires defect: the init latent lost its high frequencies and
#     its noise level rose); a plan with no coarse step left is "no coarse steps";
#   * a plan whose last transition is never reached is refused (upstream returns a smaller
#     latent than requested);
#   * the coarse grids are part of the plan: ``round(s * H)`` (official) or sorryhyun's even snap,
#     except that respace's last stage is always the exact full grid (sorryhyun pads odd latents
#     to even before sampling, Forge does not) — so respace is a geometric match of sorryhyun's SPD
#     only, and odd latents get other grids than there;
#   * dwt is refused up front when a transition that runs has r = s_next / s_i != 2 or does not
#     double the grid (official raises the same at the expansion, after the coarse steps ran);
#   * every ``Transition`` carries the sigma its expansion runs at and the aligned sigma, replayed
#     like the run (respace: after the earlier re-spacings), for the console plan and the status.
"""SPEED schedule maths: power-spectrum presets, transition sigmas and the two plans.

``transition`` mode is the official/aoleg plan: the coarse segment ends at the first step whose
sigma is <= sigma*, the expansion patches only that step's sigma to the aligned time
``sigma * kappa`` and the rest of the schedule is kept. ``respace`` mode is sorryhyun's: at the same
kind of hand-off every remaining sigma is multiplied by ``sigma_aligned / sigma`` (Sec. 4.3) and the
coarse grids are snapped to even sizes (the DiT patch).

sigma* comes from the latent power spectrum ``P(w) = A w^-beta`` at the coarse grid's Nyquist
frequency (paper Eq. 9/10, ``delta`` = noise-dominated tolerance), optionally pinned to the 1024 px
reference (aoleg's adaptive delta), divided by the ``neo_shift`` divisor, or given by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from .spectral import align_timestep, respace_alignment, snap

__all__ = [
    "CUSTOM_DEFAULT_A",
    "CUSTOM_DEFAULT_BETA",
    "MAX_COARSE_FRACTION",
    "MODES",
    "PRESETS",
    "PRESET_NAMES",
    "REFERENCE_LATENT",
    "RESPACE_PATCH",
    "THRESHOLDS",
    "TRANSFORMS",
    "Plan",
    "PlanError",
    "ThresholdPlan",
    "Transition",
    "activation_time",
    "build_plan",
    "delta_optimal_transitions",
    "delta_threshold_plan",
    "equivalent_delta",
    "parse_scales",
    "parse_sigmas",
    "power_spectrum",
    "preset_params",
    "reference_coarse_fraction",
    "respace_transitions",
    "stage_grids",
    "static_transitions",
    "validate_scales",
]

MODES = ("transition", "respace")
THRESHOLDS = ("neo_shift", "delta_optimal", "manual")
TRANSFORMS = ("dct", "dwt", "fft")

# aoleg speed_core.py:213 — latent short side of the 1024x1024 reference every sigma* is quoted at.
REFERENCE_LATENT = 128
# aoleg speed_core.py:222 — the coarse share may not exceed this (adaptive, presets without ref_shift).
MAX_COARSE_FRACTION = 0.8
# sorryhyun spd.py:56 — Anima DiT spatial patch; respace snaps its coarse grids to this multiple.
RESPACE_PATCH = 2
# official configs.yaml / comfyui speed_sampler.py:45 — the published Flux fit; aoleg's "custom" defaults.
CUSTOM_DEFAULT_A = 203.615097
CUSTOM_DEFAULT_BETA = 1.915461

# aoleg speed_core.py:258-267, values unchanged. sigma* at 1024x1024, scales 0.5,1.0, delta 0.01:
# flux 0.9680 (measured), flux2 0.9680 (measured, 64 latent), wan21 0.6941 (published fit),
# krea-2 0.9000, krea-2-raw 0.8300, z-image 0.9400, anima 0.9600 (spectrum-derived, untested).
# ``ref_shift`` turns on the adaptive reference-share cap (raw multiplier for discrete-flow models
# such as Anima, exp(mu) for Flux-style ones); measured presets carry none.
PRESETS: dict[str, dict | None] = {
    "flux": {"A": 1963.48906, "beta": 1.915461, "ref_latent": 128},
    "wan21": {"A": 219.484718, "beta": 2.422687, "ref_latent": 128, "ref_shift": 8.0},
    "krea-2": {"A": 2357.960557, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.158193},
    "krea-2-raw": {"A": 887.402358, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.158193},
    "z-image": {"A": 877.001476, "beta": 1.915461, "ref_latent": 128, "ref_shift": 3.0},
    "flux2": {"A": 520.495848, "beta": 1.915461, "ref_latent": 64},
    "anima": {"A": 8664.998524, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.0},
    "custom": None,
}
# UI order: this extension's model first.
PRESET_NAMES = ("anima", "flux", "flux2", "krea-2", "krea-2-raw", "z-image", "wan21", "custom")


class PlanError(ValueError):
    """The settings cannot produce a SPEED run for this pass (``reason`` is user-facing)."""

    def __init__(self, reason: str, *, kind: str = "skip"):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind  # "skip" (nothing to do) or "invalid" (settings error)


# --------------------------------------------------------------------------------------------
# Power spectrum and transition times (official utils.py, aoleg spectral_utils.py)
# --------------------------------------------------------------------------------------------


def power_spectrum(omega: float, A: float, beta: float) -> float:
    """Radial power law ``P(omega) = A * |omega|^-beta``."""
    return A * abs(omega) ** (-beta)


def activation_time(P_omega: float, delta: float) -> float:
    """Activation time of one radial frequency, paper Eq. 9."""
    if delta >= 1.0:
        raise ValueError(f"delta={delta} >= 1, but we assume the error threshold is < 1.")
    return 1.0 / (1.0 + math.sqrt(delta / (P_omega * (1.0 + P_omega - delta))))


def equivalent_delta(t_star: float, A: float, beta: float, omega: float) -> float:
    """The ``delta`` that puts ``activation_time`` at ``t_star`` for ``omega`` (aoleg)."""
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t_star must be in (0, 1); got {t_star}")
    P = power_spectrum(omega, A, beta)
    u = ((1.0 - t_star) / t_star) ** 2
    return u * P * (1.0 + P) / (1.0 + u * P)


def reference_coarse_fraction(t_star: float, shift: float) -> float:
    """Share of a uniform-in-t shifted schedule above ``t_star`` (aoleg)."""
    if shift <= 0.0:
        raise ValueError(f"shift must be positive; got {shift}")
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t_star must be in (0, 1); got {t_star}")
    T = t_star / (shift * (1.0 - t_star) + t_star)
    return 1.0 - T


def validate_scales(scales: Sequence[float]) -> None:
    """Strictly increasing scales in (0, 1] ending at 1.0 (official)."""
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty; supply at least one value.")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")
    if abs(scales[-1] - 1.0) > 1e-6:
        raise ValueError(f"last scale must equal 1.0 (full resolution); got {scales[-1]}")
    for a, b in zip(scales[:-1], scales[1:]):
        if not (a < b):
            raise ValueError(f"scales must be strictly increasing; got {list(scales)}")


def delta_optimal_transitions(
    scales: Sequence[float], delta: float, A: float, beta: float, H: int, W: int,
) -> list[float]:
    """Transition times for adjacent scales, paper Eq. 10 (``omega_max = min(H, W) / 2``)."""
    validate_scales(scales)
    omega_max = min(H, W) / 2.0
    transitions: list[float] = []
    for i in range(len(scales) - 1):
        omega_i = scales[i] * omega_max
        transitions.append(activation_time(power_spectrum(omega_i, A, beta), delta))
    return transitions


# --------------------------------------------------------------------------------------------
# Parsing (official/aoleg semantics)
# --------------------------------------------------------------------------------------------


def _numbers(text) -> list[float]:
    if isinstance(text, (list, tuple)):
        items: Iterable = text
    else:
        items = str(text if text is not None else "").replace(";", ",").split(",")
    out = []
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        if "/" in token:
            num, den = token.split("/", 1)
            try:
                value = float(num) / float(den)
            except ZeroDivisionError:   # a ValueError, so the callers report it as a settings error
                raise ValueError(f"division by zero: {token!r}") from None
        else:
            value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"not a finite number: {token!r}")
        out.append(value)
    return out


def parse_scales(text) -> tuple[float, ...]:
    """Comma-separated scales ending at 1.0 (``1/2`` fractions accepted, like the official CLI)."""
    try:
        values = _numbers(text)
    except ValueError as exc:
        raise ValueError(f"scales: {exc}") from None
    validate_scales(values)
    return tuple(values)


def parse_sigmas(text, *, strictly_decreasing: bool = True) -> tuple[float, ...]:
    """Manual transition sigmas, each in (0, 1) — flow-matching sigmas only (official/sorryhyun).

    The official node also requires them strictly decreasing; sorryhyun's respace loop does not.
    """
    try:
        values = _numbers(text)
    except ValueError as exc:
        raise ValueError(f"manual sigmas: {exc}") from None
    if not values:
        raise ValueError("manual sigmas: empty")
    if any(not (0.0 < v < 1.0) for v in values):
        raise ValueError(f"every manual sigma must be in (0, 1); got {values}")
    if strictly_decreasing:
        for a, b in zip(values[:-1], values[1:]):
            if not (a > b):
                raise ValueError(f"manual sigmas must be strictly decreasing; got {values}")
    return tuple(values)


def preset_params(name: str, A: float = CUSTOM_DEFAULT_A, beta: float = CUSTOM_DEFAULT_BETA):
    """``(A, beta, ref_latent, ref_shift)``; an unknown name or ``custom`` uses ``A``/``beta``."""
    preset = PRESETS.get(str(name))
    if preset is None:
        return float(A), float(beta), REFERENCE_LATENT, None
    return preset["A"], preset["beta"], int(preset["ref_latent"]), preset.get("ref_shift")


# --------------------------------------------------------------------------------------------
# Thresholds and the two plans
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdPlan:
    values: tuple[float, ...]            # sigma* per transition (after pinning / divisor)
    caps: tuple[int | None, ...]         # adaptive step cap per transition (None = no cap)
    notes: tuple[str, ...] = ()


def delta_threshold_plan(
    n_steps: int,
    scales: Sequence[float],
    delta: float,
    A: float,
    beta: float,
    H: int,
    W: int,
    *,
    adaptive: bool = False,
    ref_latent: int = REFERENCE_LATENT,
    ref_shift: float | None = None,
    sigma_divisor: float = 1.0,
) -> ThresholdPlan:
    """aoleg ``_resolve_transitions`` without the step lookup (speed_core.py:438-501)."""
    notes: list[str] = []
    if len(scales) < 2:
        return ThresholdPlan((), (), ())
    t_stars = delta_optimal_transitions(scales, delta, A, beta, H, W)
    if adaptive:
        t_ref = delta_optimal_transitions(scales, delta, A, beta, ref_latent, ref_latent)
        pinned = [max(actual, reference) for actual, reference in zip(t_stars, t_ref)]
        if any(abs(a - b) > 1e-9 for a, b in zip(t_stars, pinned)):
            omega_max = min(H, W) / 2.0
            try:
                eq = f"{equivalent_delta(pinned[0], A, beta, scales[0] * omega_max):.6g}"
            except ValueError:
                eq = "n/a"
            notes.append(
                f"adaptive delta: latent {H}x{W} vs reference {ref_latent}x{ref_latent}; sigma* "
                + ", ".join(f"{a:.4f}->{b:.4f}" for a, b in zip(t_stars, pinned))
                + f" (delta {delta:g} -> {eq} equivalent at this resolution)"
            )
        t_stars = pinned
    if sigma_divisor != 1.0:
        divided = [t / sigma_divisor for t in t_stars]
        notes.append(
            f"neo_shift: dividing sigma* by {sigma_divisor:g} ("
            + ", ".join(f"{a:.4f}->{b:.4f}" for a, b in zip(t_stars, divided)) + ")"
        )
        t_stars = divided
    caps: list[int | None] = []
    n_transitions = len(scales) - 1
    for i, t_thr in enumerate(t_stars):
        if not adaptive:
            caps.append(None)
        elif ref_shift:
            if 0.0 < t_thr < 1.0:
                caps.append(math.ceil(n_steps * reference_coarse_fraction(t_thr, ref_shift)))
            else:
                caps.append(None)
        else:
            caps.append(int(n_steps * MAX_COARSE_FRACTION * (i + 1) / n_transitions))
    return ThresholdPlan(tuple(float(t) for t in t_stars), tuple(caps), tuple(notes))


def static_transitions(
    sigmas, scales: Sequence[float], values: Sequence[float], caps: Sequence[int | None] | None = None,
) -> list[tuple[int, float, float]]:
    """``(step_idx, s_i, s_next)`` per transition on the unpatched schedule (official/aoleg lookup).

    The first step whose sigma is <= the threshold (never the final sigma), held at the adaptive cap,
    stopping at the first transition that does not fit — aoleg speed_core.py:468-505 and
    ``_resolve_manual`` (:508-529).
    """
    n_steps = len(sigmas) - 1
    caps = list(caps) if caps is not None else [None] * len(values)
    out: list[tuple[int, float, float]] = []
    for i, (s_old, s_new, t_thr) in enumerate(zip(scales[:-1], scales[1:], values)):
        step_idx = next((j for j in range(n_steps) if float(sigmas[j]) <= t_thr), n_steps)
        cap = caps[i] if i < len(caps) else None
        if cap is not None and step_idx > cap:
            step_idx = cap
        if step_idx >= n_steps:
            break
        out.append((step_idx, s_old, s_new))
    return out


def _fires(sig, i: int, stage: int, values, caps) -> bool:
    cap = caps[stage]
    return float(sig[i]) <= values[stage] or (cap is not None and i >= cap)


def respace_transitions(sigmas, scales: Sequence[float], values: Sequence[float], caps=None):
    """Replay sorryhyun's respace loop (spd.py:144-198) on the sigmas alone.

    Returns ``(dropped, events, final_sigmas)``: ``dropped`` leading stages whose hand-off already
    fires at step 0 (skipped, not expanded), ``events`` as ``(step, k, sigma, sigma_aligned)`` with
    ``k`` the transition index and ``sigma_aligned`` as stored in the float32 schedule, and the
    re-spaced float32 schedule after the last event.
    """
    caps = list(caps) if caps is not None else [None] * len(values)
    sig = sigmas.detach().clone().float() if torch.is_tensor(sigmas) else torch.tensor(list(sigmas), dtype=torch.float32)
    n = len(sig) - 1
    stage = 0
    while stage < len(values) and n >= 1 and _fires(sig, 0, stage, values, caps):
        stage += 1
    dropped = stage
    events = []
    for i in range(n):
        while stage < len(values) and _fires(sig, i, stage, values, caps):
            old = float(sig[i])
            r = scales[stage + 1] / scales[stage]
            new, _ = respace_alignment(old, r)
            if old > 0.0 and new != old:
                sig[i + 1:] = new * (sig[i + 1:] / old)
            sig[i] = new
            events.append((i, stage, old, float(sig[i])))
            stage += 1
    return dropped, events, sig


def _transition_hand_offs(sigmas, scales: Sequence[float], steps) -> list[tuple[float, float]]:
    """``(sigma, sigma_aligned)`` per hand-off of the transition plan, replayed on a schedule copy the
    way the run patches it (official/aoleg: only the transition step's sigma becomes ``sigma * kappa``),
    so two hand-offs at the same step see the first one's aligned sigma, like the run."""
    sig = sigmas.detach().clone() if torch.is_tensor(sigmas) else torch.tensor(list(sigmas), dtype=torch.float32)
    out = []
    for step, k in steps:
        t = float(sig[step])
        sig[step] = float(align_timestep(t, float(scales[k + 1]) / float(scales[k])))
        out.append((t, float(sig[step])))
    return out


@dataclass(frozen=True)
class Transition:
    """One expansion: at schedule index ``step``, ``scale_from`` -> ``scale_to`` (ladder index ``k``).

    ``sigma`` is the schedule value the expansion runs at and ``aligned`` the one the next segment
    starts from, both as the run will see them (respace mode: after the earlier re-spacings)."""

    step: int
    k: int
    scale_from: float
    scale_to: float
    grid_from: tuple[int, int]
    grid_to: tuple[int, int]
    sigma: float = math.nan
    aligned: float = math.nan


@dataclass
class Plan:
    mode: str
    transform: str
    full_grid: tuple[int, int]
    first_scale: float
    first_grid: tuple[int, int]
    transitions: list[Transition]
    thresholds: tuple[float, ...]
    n_steps: int
    notes: list[str] = field(default_factory=list)

    @property
    def coarse_steps(self) -> int:
        """Sampler steps that run below the full grid."""
        return self.transitions[-1].step if self.transitions else 0


def stage_grids(mode: str, scales: Sequence[float], H: int, W: int) -> list[tuple[int, int]]:
    """Grid of every stage in ``scales`` (the stages that actually run, first to full)."""
    grids: list[tuple[int, int]] = []
    if mode == "respace":
        for index, s in enumerate(scales):
            # Same tolerance as validate_scales: a last scale it accepts as 1.0 (e.g. 0.9999999) is the
            # full grid, not an even snap of it (an odd latent would otherwise end one row short).
            if abs(s - 1.0) <= 1e-6:
                grids.append((H, W))
            elif index == 0:
                grids.append((min(snap(H * s, RESPACE_PATCH), H), min(snap(W * s, RESPACE_PATCH), W)))
            else:
                prev_h, prev_w = grids[-1]
                grids.append((max(snap(H * s, RESPACE_PATCH), prev_h), max(snap(W * s, RESPACE_PATCH), prev_w)))
    else:
        for s in scales:
            grids.append((int(round(s * H)), int(round(s * W))))
    return grids


def build_plan(
    *,
    mode: str,
    transform: str,
    sigmas,
    full_grid: tuple[int, int],
    scales: Sequence[float],
    threshold: str,
    delta: float = 0.01,
    A: float = CUSTOM_DEFAULT_A,
    beta: float = CUSTOM_DEFAULT_BETA,
    ref_latent: int = REFERENCE_LATENT,
    ref_shift: float | None = None,
    adaptive: bool = True,
    sigma_divisor: float = 1.0,
    manual_sigmas: Sequence[float] = (),
) -> Plan:
    """The pass plan, or :class:`PlanError` when nothing would run below full resolution."""
    if mode not in MODES:
        raise PlanError(f"unknown mode {mode!r}", kind="invalid")
    if transform not in TRANSFORMS:
        raise PlanError(f"unknown transform {transform!r}", kind="invalid")
    if threshold not in THRESHOLDS:
        raise PlanError(f"unknown threshold source {threshold!r}", kind="invalid")
    try:
        validate_scales(scales)
    except ValueError as exc:
        raise PlanError(str(exc), kind="invalid") from None
    H, W = int(full_grid[0]), int(full_grid[1])
    n_steps = len(sigmas) - 1
    if n_steps < 1:
        raise PlanError("empty sigma schedule")
    n_transitions = len(scales) - 1
    if n_transitions == 0:
        raise PlanError("scales has no coarse stage (only 1.0)")

    notes: list[str] = []
    if threshold == "manual":
        values = [float(v) for v in manual_sigmas]
        if mode == "respace" and len(values) == 1 and n_transitions > 1:
            values = values * n_transitions      # sorryhyun resolve_spd_schedule: one sigma per hand-off
        if len(values) != n_transitions:
            raise PlanError(
                f"manual sigmas has length {len(values)}, expected {n_transitions} "
                "(one threshold per transition in scales)", kind="invalid",
            )
        if any(not (0.0 < v < 1.0) for v in values):
            raise PlanError(f"every manual sigma must be in (0, 1); got {values}", kind="invalid")
        if mode == "transition" and any(not (a > b) for a, b in zip(values[:-1], values[1:])):
            raise PlanError(f"manual sigmas must be strictly decreasing; got {values}", kind="invalid")
        caps: list[int | None] = [None] * n_transitions
    else:
        if not (0.0 < float(delta) < 1.0):
            raise PlanError(f"delta must be in (0, 1); got {delta}", kind="invalid")
        if not (float(sigma_divisor) > 0.0):
            raise PlanError(f"sigma divisor must be positive; got {sigma_divisor}", kind="invalid")
        # custom spectrum: A <= 0 has no activation time (math domain / division by zero) and a NaN A or
        # beta would hand the adaptive cap a NaN sigma* that silently runs — refuse both as settings errors
        if not (math.isfinite(float(A)) and float(A) > 0.0):
            raise PlanError(f"spectrum A must be a positive number; got {A}", kind="invalid")
        if not math.isfinite(float(beta)):
            raise PlanError(f"spectrum beta must be a finite number; got {beta}", kind="invalid")
        divisor = float(sigma_divisor) if threshold == "neo_shift" else 1.0
        try:
            tp = delta_threshold_plan(
                n_steps, scales, float(delta), float(A), float(beta), H, W,
                adaptive=bool(adaptive), ref_latent=int(ref_latent), ref_shift=ref_shift, sigma_divisor=divisor,
            )
        except (ArithmeticError, ValueError) as exc:   # e.g. P(omega) underflows to 0 for a huge beta
            raise PlanError(
                f"spectrum A {float(A):g} / beta {float(beta):g} gives no transition time ({exc})", kind="invalid",
            ) from None
        values, caps = list(tp.values), list(tp.caps)
        notes.extend(tp.notes)

    if mode == "transition":
        found = static_transitions(sigmas, scales, values, caps)
        dropped = 0
        while dropped < len(found) and found[dropped][0] == 0:
            dropped += 1
        if len(found) < n_transitions:
            missing = len(found)
            raise PlanError(
                f"the {scales[missing]:g}->{scales[missing + 1]:g} transition (sigma* {values[missing]:.4f}) "
                "is never reached in this schedule — the output would stay below full size"
            )
        steps = [(found[k][0], k) for k in range(dropped, n_transitions)]
        hand_offs = _transition_hand_offs(sigmas, scales, steps)
    else:
        dropped, events, _ = respace_transitions(sigmas, scales, values, caps)
        if dropped + len(events) < n_transitions:
            missing = dropped + len(events)
            raise PlanError(
                f"the {scales[missing]:g}->{scales[missing + 1]:g} hand-off (sigma {values[missing]:.4f}) "
                "is never reached in this schedule — the output would stay below full size"
            )
        steps = [(step, k) for step, k, _old, _new in events]
        hand_offs = [(old, new) for _step, _k, old, new in events]

    if not steps:
        raise PlanError(
            f"no coarse steps: the pass starts at sigma {float(sigmas[0]):.4f}, at or below the first "
            f"transition sigma {values[dropped - 1] if dropped else values[0]:.4f}"
        )
    if transform == "dwt":
        # Official utils.spectral_expand_and_align / aoleg _expand_and_align_torch: the Haar synthesis
        # is an exact 2x step, so every transition that runs needs r = s_next / s_i = 2.
        for _step, k in steps:
            r = float(scales[k + 1]) / float(scales[k])
            if abs(r - 2.0) > 1e-6:
                raise PlanError(
                    f"dwt needs r = 2 between consecutive scales; got {scales[k]:g}->{scales[k + 1]:g} "
                    f"(r = {r:.4f}) - use dct or fft for non-dyadic scales", kind="invalid",
                )
    used_scales = list(scales[dropped:])
    grids = stage_grids(mode, used_scales, H, W)
    if any(h < 1 or w < 1 for h, w in grids):
        raise PlanError(f"a coarse grid is empty for scales {list(scales)} at latent {H}x{W}", kind="invalid")
    if grids[-1] != (H, W):
        raise PlanError("internal: the last stage is not the full grid", kind="invalid")
    for (h0, w0), (h1, w1) in zip(grids[:-1], grids[1:]):
        if transform == "dwt" and (h1 != 2 * h0 or w1 != 2 * w0):
            raise PlanError(
                f"dwt needs every stage to double the grid exactly; got {(h0, w0)} -> {(h1, w1)} "
                "(use dct or fft for this ladder or latent size)", kind="invalid",
            )
        if h1 < h0 or w1 < w0:
            raise PlanError(f"grid shrinks between stages: {(h0, w0)} -> {(h1, w1)}", kind="invalid")
    if dropped:
        notes.append(
            f"{dropped} leading stage(s) skipped: their transition fires at step 0 "
            f"(pass starts at sigma {float(sigmas[0]):.4f})"
        )
    transitions = []
    for position, (step, k) in enumerate(steps):
        sigma, aligned = hand_offs[position]
        transitions.append(Transition(
            step=int(step), k=int(k), scale_from=float(scales[k]), scale_to=float(scales[k + 1]),
            grid_from=grids[position], grid_to=grids[position + 1], sigma=float(sigma), aligned=float(aligned),
        ))
    return Plan(
        mode=mode, transform=transform, full_grid=(H, W), first_scale=float(used_scales[0]),
        first_grid=grids[0], transitions=transitions, thresholds=tuple(values), n_steps=n_steps,
        notes=notes,
    )
