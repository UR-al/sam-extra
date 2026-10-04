"""Per-generation settings of the ER SDE entries: values, ranges, infotext and paste parsing.

The ``Extra Samplers`` accordion (``scripts/anima_extra_samplers.py``) holds five values that only the
three ER SDE entries of this extension read (Forge's built-in ``ER SDE`` reads none of them):

* ``ER SDE max stage`` — the solver order of ``sample_er_sde`` (1 = exponential Euler, 2 and 3 add
  the multistep terms). ComfyUI's ``SamplerER_SDE`` input: integer 1..3, default 3. All three entries.
* ``ER SDE eta`` — η of the noise scaler: ``h(λ) = λ^(η+1)`` for ``ER SDE (Reverse-time)``,
  ``h(λ) = λ·(e^(λ^0.3) + 10)^η`` for ``ER SDE (Tunable)`` (ComfyUI's ``ER-SDE`` solver type, which is
  Forge's built-in ``ER SDE`` at η = 1); ``ER SDE (ODE)`` has none. ComfyUI's input: 0..10, step 0.01,
  default 1.0; η = 0 is the ODE.
* ``ER SDE noise window`` (checkbox) with ``ER SDE noise start`` / ``ER SDE noise end`` — sampling
  percentages 0..1 of the model's own schedule (default 0.2-0.8). When the box is on, Reverse-time and
  Tunable inject noise only on the steps whose target σ lies in ``[percent_to_sigma(end),
  percent_to_sigma(start)]``; the other steps are ER-SDE steps with the ODE scaler and draw nothing
  (``er_sde_window``). Start ≥ end is an empty window (no step gets noise), not swapped.

The script stores the values on the request (``p.sam_extra_samplers``) in ``process``; the sampler
reads them when Forge initialises it (``registry.ExtraKDiffusionSampler.initialize``), so the base and
the hires pass of one request use the same values. A request that never ran the script — ADetailer's
inner img2img filters always-on scripts — falls back to the values of the last request the script
saw, then to the defaults. API: ``alwayson_scripts["Extra Samplers"]["args"] = [max_stage, eta,
noise_window, noise_start, noise_end]`` — positional, every one optional (``[]``, ``[3]`` and
``[3, 1.0]`` of older requests run exactly as before: window off), extra ones ignored.

Infotext: ``ER SDE max stage`` is written when it is not 3 (the key and the rule of
aoleg/Neo_ExtraSchedulers, so pasting either extension's infotext restores the stage).
``ER SDE eta`` is written when it is not 1.0. That extension takes η from Forge's global ``Eta``
setting instead, so pasting an infotext of ``Sampler: ER SDE (Reverse-time)`` (or ``Hires sampler``)
that has ``Eta`` but no ``ER SDE eta`` reads η from ``Eta`` (Reverse-time only: aoleg has no Tunable).
This extension does not write ``Eta``: Forge writes that key for the ancestral samplers' own eta, which
a hires pass can need at the same time. Because of that paste rule, ``ER SDE eta`` is also written at
1.0 when the request's Forge η is not 1.0 (Forge then writes ``Eta`` for any pass whose sampler takes
``eta`` — Euler a, DPM++ 4M SDE …), so this extension's own infotexts always paste back their η.
``ER SDE noise window`` (``start-end``, e.g. ``0.2-0.8``) is written when the window is on and
Reverse-time or Tunable runs with η > 0; pasting it turns the checkbox on and restores start/end, an
infotext without it (or with an unreadable value) turns the box off and resets start/end to 0.2/0.8.

This module has no Forge or gradio dependency (the tests import it directly).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "ARG_NAMES",
    "DEFAULT_ETA",
    "DEFAULT_MAX_STAGE",
    "DEFAULT_NOISE_END",
    "DEFAULT_NOISE_START",
    "DEFAULT_NOISE_WINDOW",
    "ETA_MAX",
    "ETA_MIN",
    "ETA_STEP",
    "KEY_ETA",
    "KEY_MAX_STAGE",
    "KEY_NOISE_WINDOW",
    "LABEL_ER_SDE_ODE",
    "LABEL_ER_SDE_REVERSE_TIME",
    "LABEL_ER_SDE_TUNABLE",
    "MAX_STAGE_MAX",
    "MAX_STAGE_MIN",
    "NOISE_PERCENT_MAX",
    "NOISE_PERCENT_MIN",
    "NOISE_PERCENT_STEP",
    "NOISE_WINDOW_OFF",
    "P_ATTR",
    "WINDOW_SKIPPED_STATUS",
    "XYZ_ATTR",
    "ErSdeSettings",
    "active",
    "apply_to",
    "coerce_eta",
    "coerce_max_stage",
    "coerce_noise_percent",
    "coerce_noise_window",
    "format_noise_window",
    "parse_noise_window",
    "paste_eta",
    "paste_max_stage",
    "paste_noise_end",
    "paste_noise_start",
    "paste_noise_window",
    "record_infotext",
    "reset_active",
    "resolve",
    "settings_from_args",
]

LABEL_ER_SDE_REVERSE_TIME = "ER SDE (Reverse-time)"
LABEL_ER_SDE_ODE = "ER SDE (ODE)"
LABEL_ER_SDE_TUNABLE = "ER SDE (Tunable)"

# The script's positional arguments, in order (API ``alwayson_scripts["Extra Samplers"]["args"]``).
ARG_NAMES = ("max_stage", "eta", "noise_window", "noise_start", "noise_end")

# ComfyUI SamplerER_SDE inputs (comfy_extras/nodes_custom_sampler.py:593-594 @ 40e46c71):
#   max_stage: Int, default 3, min 1, max 3
#   eta:       Float, default 1.0, min 0.0, max 10.0, step 0.01
DEFAULT_MAX_STAGE = 3
MAX_STAGE_MIN = 1
MAX_STAGE_MAX = 3
DEFAULT_ETA = 1.0
ETA_MIN = 0.0
ETA_MAX = 10.0
ETA_STEP = 0.01

# The noise window (sam-extra): sampling percentages of the model's schedule, off by default.
DEFAULT_NOISE_WINDOW = False
DEFAULT_NOISE_START = 0.2
DEFAULT_NOISE_END = 0.8
NOISE_PERCENT_MIN = 0.0
NOISE_PERCENT_MAX = 1.0
NOISE_PERCENT_STEP = 0.01

KEY_MAX_STAGE = "ER SDE max stage"
KEY_ETA = "ER SDE eta"
KEY_NOISE_WINDOW = "ER SDE noise window"
NOISE_WINDOW_OFF = "off"   # the XYZ value of a cell without the window
# The ``Extra Samplers status`` part written when a request asked for the window but the model's
# predictor has no ``percent_to_sigma`` (the sampler then runs without the window).
WINDOW_SKIPPED_STATUS = "noise window skipped (no percent_to_sigma)"
# Forge's ancestral eta key (modules/shared_options.py ``eta_ancestral``, infotext "Eta"); read on
# paste for infotexts of aoleg/Neo_ExtraSchedulers, whose Reverse-time sampler uses that setting.
_FORGE_ETA_KEY = "Eta"
_ETA_LABELS = (LABEL_ER_SDE_REVERSE_TIME, LABEL_ER_SDE_TUNABLE)   # the entries η (and the window) apply to
_ER_SDE_LABELS = (LABEL_ER_SDE_REVERSE_TIME, LABEL_ER_SDE_ODE, LABEL_ER_SDE_TUNABLE)

P_ATTR = "sam_extra_samplers"
XYZ_ATTR = "_sam_extra_samplers_xyz"

_TRUE_TEXTS = ("true", "on", "yes", "1")
# A non-negative decimal number (an exponent is allowed: ``format_noise_window`` writes ``1e-06``).
_PERCENT_TEXT = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_WINDOW_RE = re.compile(rf"\s*({_PERCENT_TEXT})\s*-\s*({_PERCENT_TEXT})\s*")


def coerce_max_stage(value, default: int = DEFAULT_MAX_STAGE) -> int:
    """An integer stage in 1..3; anything that is not a finite number gives ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return int(min(MAX_STAGE_MAX, max(MAX_STAGE_MIN, int(round(number)))))


def coerce_eta(value, default: float = DEFAULT_ETA) -> float:
    """A finite η in 0..10; anything else gives ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return float(min(ETA_MAX, max(ETA_MIN, number)))


def coerce_noise_window(value) -> bool:
    """The checkbox: ``True`` for ``True``, the numbers 1 / 1.0 and the texts true/on/yes/1 (any case),
    ``False`` for everything else (other numbers too — an API caller's ``2`` is not "on")."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_TEXTS
    return False


def coerce_noise_percent(value, default: float) -> float:
    """A sampling percentage clamped to 0..1; a bool (not a number here) or anything that is not a
    finite number gives ``default``."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return float(min(NOISE_PERCENT_MAX, max(NOISE_PERCENT_MIN, number)))


@dataclass(frozen=True)
class ErSdeSettings:
    max_stage: int = DEFAULT_MAX_STAGE
    eta: float = DEFAULT_ETA
    noise_window: bool = DEFAULT_NOISE_WINDOW
    noise_start: float = DEFAULT_NOISE_START
    noise_end: float = DEFAULT_NOISE_END

    def as_dict(self) -> dict:
        return {
            "max_stage": self.max_stage, "eta": self.eta, "noise_window": self.noise_window,
            "noise_start": self.noise_start, "noise_end": self.noise_end,
        }


_ACTIVE: ErSdeSettings | None = None


def active() -> ErSdeSettings | None:
    """The settings of the last request the script prepared (None before the first one)."""
    return _ACTIVE


def reset_active() -> None:
    global _ACTIVE
    _ACTIVE = None


def settings_from_args(args, xyz=None) -> ErSdeSettings:
    """Script arguments ``[max_stage, eta, noise_window, noise_start, noise_end]`` (missing ones default,
    extra ones ignored), then XYZ overrides (``noise_window`` is an ``(enabled, start, end)`` triple)."""
    args = list(args or ())

    def arg(index, default):
        return args[index] if len(args) > index else default

    max_stage = coerce_max_stage(arg(0, DEFAULT_MAX_STAGE))
    eta = coerce_eta(arg(1, DEFAULT_ETA))
    noise_window = coerce_noise_window(arg(2, DEFAULT_NOISE_WINDOW))
    noise_start = coerce_noise_percent(arg(3, DEFAULT_NOISE_START), DEFAULT_NOISE_START)
    noise_end = coerce_noise_percent(arg(4, DEFAULT_NOISE_END), DEFAULT_NOISE_END)
    if isinstance(xyz, dict):
        if "max_stage" in xyz:
            max_stage = coerce_max_stage(xyz["max_stage"], max_stage)
        if "eta" in xyz:
            eta = coerce_eta(xyz["eta"], eta)
        window = xyz.get("noise_window")
        if isinstance(window, (tuple, list)) and len(window) == 3:
            noise_window = coerce_noise_window(window[0])
            noise_start = coerce_noise_percent(window[1], noise_start)
            noise_end = coerce_noise_percent(window[2], noise_end)
    return ErSdeSettings(max_stage=max_stage, eta=eta, noise_window=noise_window,
                         noise_start=noise_start, noise_end=noise_end)


def apply_to(p, settings: ErSdeSettings) -> None:
    """Store ``settings`` on the request and remember them for requests without the script."""
    global _ACTIVE
    setattr(p, P_ATTR, settings)
    _ACTIVE = settings


def resolve(p) -> ErSdeSettings:
    """The request's settings, else the last request's, else the defaults."""
    settings = getattr(p, P_ATTR, None)
    if isinstance(settings, ErSdeSettings):
        return settings
    if isinstance(settings, dict):  # tolerate a plain dict set by an API caller
        return ErSdeSettings(
            max_stage=coerce_max_stage(settings.get("max_stage", DEFAULT_MAX_STAGE)),
            eta=coerce_eta(settings.get("eta", DEFAULT_ETA)),
            noise_window=coerce_noise_window(settings.get("noise_window", DEFAULT_NOISE_WINDOW)),
            noise_start=coerce_noise_percent(settings.get("noise_start", DEFAULT_NOISE_START), DEFAULT_NOISE_START),
            noise_end=coerce_noise_percent(settings.get("noise_end", DEFAULT_NOISE_END), DEFAULT_NOISE_END),
        )
    if _ACTIVE is not None:
        return _ACTIVE
    return ErSdeSettings()


def format_noise_window(start: float, end: float) -> str:
    """The infotext/XYZ text of a window: ``"0.2-0.8"`` (each bound rounded to 6 decimals, ``:g``)."""
    return f"{round(float(start), 6):g}-{round(float(end), 6):g}"


def parse_noise_window(text) -> tuple | None:
    """``"off"`` → ``(False, 0.2, 0.8)``; ``"a-b"`` with 0 ≤ a, b ≤ 1 → ``(True, a, b)`` (spaces around the
    numbers and the dash are allowed; a ≥ b is a valid, empty window); anything else → None."""
    value = str(text if text is not None else "").strip()
    if value.lower() == NOISE_WINDOW_OFF:
        return False, DEFAULT_NOISE_START, DEFAULT_NOISE_END
    match = _WINDOW_RE.fullmatch(value)
    if match is None:
        return None
    start, end = float(match.group(1)), float(match.group(2))
    for bound in (start, end):
        if not (math.isfinite(bound) and NOISE_PERCENT_MIN <= bound <= NOISE_PERCENT_MAX):
            return None
    return True, start, end


def _forge_eta_in_infotext(params: dict, forge_eta) -> bool:
    """Can this request's infotext carry Forge's ``Eta`` (which ``paste_eta`` would read as our η)?

    ``forge_eta`` is Forge's resolved ancestral η of the request (``Sampler.eta``: ``p.eta``, else the
    ``eta_ancestral`` setting). ``Sampler.initialize`` writes it as ``Eta`` for every pass whose
    sampler takes ``eta`` when it is not 1.0 — possibly a later hires pass, so the key itself may not
    be there yet."""
    if _FORGE_ETA_KEY in params:
        return True
    try:
        value = float(forge_eta)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value != 1.0


def record_infotext(p, label: str, settings: ErSdeSettings, forge_eta=None, window_skipped: bool = False) -> None:
    """Write the non-default values the sampler ``label`` uses into ``p.extra_generation_params``.

    η is rounded to 6 decimals only for the text (an XYZ range such as ``0.1-0.5 [5]`` yields
    ``0.30000000000000004``); the sampler itself runs with the exact value. ``ER SDE eta`` is also
    written at 1.0 when the infotext can carry Forge's own ``Eta`` (``forge_eta``, see the module
    docstring), so ``paste_eta`` never takes that ancestral η for ours. ``ER SDE noise window`` is
    written when the window is on, the sampler takes it (Reverse-time, Tunable) and η > 0 — unless the
    registry had to drop it (``window_skipped``: the model has no ``percent_to_sigma``)."""
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        return
    if label not in _ER_SDE_LABELS:
        return
    if settings.max_stage != DEFAULT_MAX_STAGE:
        params[KEY_MAX_STAGE] = settings.max_stage
    if label not in _ETA_LABELS:
        return
    if settings.eta != DEFAULT_ETA or _forge_eta_in_infotext(params, forge_eta):
        params[KEY_ETA] = round(settings.eta, 6)
    if settings.noise_window and settings.eta > 0 and not window_skipped:
        params[KEY_NOISE_WINDOW] = format_noise_window(settings.noise_start, settings.noise_end)


def _names_sampler(value, label: str) -> bool:
    text = str(value or "").strip()
    return text == label or text.startswith(label + " ")


def paste_max_stage(params: dict) -> int:
    """Paste field: the stage of an infotext, 3 when it has none (3 is never written)."""
    return coerce_max_stage(params.get(KEY_MAX_STAGE, DEFAULT_MAX_STAGE))


def paste_eta(params: dict) -> float:
    """Paste field: ``ER SDE eta``; else ``Eta`` of a Reverse-time infotext; else 1.0."""
    if KEY_ETA in params:
        return coerce_eta(params.get(KEY_ETA))
    reverse_time = any(
        _names_sampler(params.get(key), LABEL_ER_SDE_REVERSE_TIME) for key in ("Sampler", "Hires sampler")
    )
    if reverse_time and _FORGE_ETA_KEY in params:
        return coerce_eta(params.get(_FORGE_ETA_KEY))
    return DEFAULT_ETA


def _pasted_window(params: dict) -> tuple:
    parsed = parse_noise_window(params.get(KEY_NOISE_WINDOW)) if KEY_NOISE_WINDOW in params else None
    if parsed is None:
        return DEFAULT_NOISE_WINDOW, DEFAULT_NOISE_START, DEFAULT_NOISE_END
    return parsed


def paste_noise_window(params: dict) -> bool:
    """Paste field: the checkbox — on when the infotext has a readable ``ER SDE noise window``."""
    return _pasted_window(params)[0]


def paste_noise_start(params: dict) -> float:
    """Paste field: the window's start (0.2 when the infotext has no readable window)."""
    return _pasted_window(params)[1]


def paste_noise_end(params: dict) -> float:
    """Paste field: the window's end (0.8 when the infotext has no readable window)."""
    return _pasted_window(params)[2]
