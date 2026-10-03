"""Per-generation settings of the ER SDE pair: values, ranges, infotext and paste parsing.

The ``Extra Samplers`` accordion (``scripts/anima_extra_samplers.py``) holds two values that only the two
ER SDE entries read:

* ``ER SDE max stage`` — the solver order of ``sample_er_sde`` (1 = exponential Euler, 2 and 3 add
  the multistep terms). ComfyUI's ``SamplerER_SDE`` input: integer 1..3, default 3.
* ``ER SDE eta`` — η of the reverse-time noise scaler ``h(λ) = λ^(η+1)`` (``ER SDE (Reverse-time)``
  only; ``ER SDE (ODE)`` has none). ComfyUI's input: 0..10, step 0.01, default 1.0; η = 0 is the ODE.

The script stores the values on the request (``p.sam_extra_samplers``) in ``process``; the sampler
reads them when Forge initialises it (``registry.ExtraKDiffusionSampler.initialize``), so the base and
the hires pass of one request use the same values. A request that never ran the script — ADetailer's
inner img2img filters always-on scripts — falls back to the values of the last request the script
saw, then to the defaults.

Infotext: ``ER SDE max stage`` is written when it is not 3 (the key and the rule of
aoleg/Neo_ExtraSchedulers, so pasting either extension's infotext restores the stage).
``ER SDE eta`` is written when it is not 1.0. That extension takes η from Forge's global ``Eta``
setting instead, so pasting an infotext of ``Sampler: ER SDE (Reverse-time)`` (or ``Hires sampler``)
that has ``Eta`` but no ``ER SDE eta`` reads η from ``Eta``. This extension does not write ``Eta``:
Forge writes that key for the ancestral samplers' own eta, which a hires pass can need at the same
time. Because of that paste rule, ``ER SDE eta`` is also written at 1.0 when the request's Forge η
is not 1.0 (Forge then writes ``Eta`` for any pass whose sampler takes ``eta`` — Euler a, DPM++ 4M
SDE …), so this extension's own infotexts always paste back their η.

This module has no Forge or gradio dependency (the tests import it directly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DEFAULT_ETA",
    "DEFAULT_MAX_STAGE",
    "ETA_MAX",
    "ETA_MIN",
    "ETA_STEP",
    "KEY_ETA",
    "KEY_MAX_STAGE",
    "LABEL_ER_SDE_ODE",
    "LABEL_ER_SDE_REVERSE_TIME",
    "MAX_STAGE_MAX",
    "MAX_STAGE_MIN",
    "P_ATTR",
    "XYZ_ATTR",
    "ErSdeSettings",
    "active",
    "apply_to",
    "coerce_eta",
    "coerce_max_stage",
    "paste_eta",
    "paste_max_stage",
    "record_infotext",
    "reset_active",
    "resolve",
    "settings_from_args",
]

LABEL_ER_SDE_REVERSE_TIME = "ER SDE (Reverse-time)"
LABEL_ER_SDE_ODE = "ER SDE (ODE)"

# ComfyUI SamplerER_SDE inputs (comfy_extras/nodes_custom_sampler.py:593-594 @ 36c0b0a6):
#   max_stage: Int, default 3, min 1, max 3
#   eta:       Float, default 1.0, min 0.0, max 10.0, step 0.01
DEFAULT_MAX_STAGE = 3
MAX_STAGE_MIN = 1
MAX_STAGE_MAX = 3
DEFAULT_ETA = 1.0
ETA_MIN = 0.0
ETA_MAX = 10.0
ETA_STEP = 0.01

KEY_MAX_STAGE = "ER SDE max stage"
KEY_ETA = "ER SDE eta"
# Forge's ancestral eta key (modules/shared_options.py ``eta_ancestral``, infotext "Eta"); read on
# paste for infotexts of aoleg/Neo_ExtraSchedulers, whose Reverse-time sampler uses that setting.
_FORGE_ETA_KEY = "Eta"

P_ATTR = "sam_extra_samplers"
XYZ_ATTR = "_sam_extra_samplers_xyz"


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


@dataclass(frozen=True)
class ErSdeSettings:
    max_stage: int = DEFAULT_MAX_STAGE
    eta: float = DEFAULT_ETA

    def as_dict(self) -> dict:
        return {"max_stage": self.max_stage, "eta": self.eta}


_ACTIVE: ErSdeSettings | None = None


def active() -> ErSdeSettings | None:
    """The settings of the last request the script prepared (None before the first one)."""
    return _ACTIVE


def reset_active() -> None:
    global _ACTIVE
    _ACTIVE = None


def settings_from_args(args, xyz=None) -> ErSdeSettings:
    """Script arguments ``[max_stage, eta]`` (missing ones default), then XYZ overrides."""
    args = list(args or ())
    max_stage = coerce_max_stage(args[0] if len(args) > 0 else DEFAULT_MAX_STAGE)
    eta = coerce_eta(args[1] if len(args) > 1 else DEFAULT_ETA)
    if isinstance(xyz, dict):
        if "max_stage" in xyz:
            max_stage = coerce_max_stage(xyz["max_stage"], max_stage)
        if "eta" in xyz:
            eta = coerce_eta(xyz["eta"], eta)
    return ErSdeSettings(max_stage=max_stage, eta=eta)


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
        )
    if _ACTIVE is not None:
        return _ACTIVE
    return ErSdeSettings()


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


def record_infotext(p, label: str, settings: ErSdeSettings, forge_eta=None) -> None:
    """Write the non-default values the sampler ``label`` uses into ``p.extra_generation_params``.

    η is rounded to 6 decimals only for the text (an XYZ range such as ``0.1-0.5 [5]`` yields
    ``0.30000000000000004``); the sampler itself runs with the exact value. ``ER SDE eta`` is also
    written at 1.0 when the infotext can carry Forge's own ``Eta`` (``forge_eta``, see the module
    docstring), so ``paste_eta`` never takes that ancestral η for ours."""
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        return
    if label not in (LABEL_ER_SDE_REVERSE_TIME, LABEL_ER_SDE_ODE):
        return
    if settings.max_stage != DEFAULT_MAX_STAGE:
        params[KEY_MAX_STAGE] = settings.max_stage
    if label == LABEL_ER_SDE_REVERSE_TIME and (
        settings.eta != DEFAULT_ETA or _forge_eta_in_infotext(params, forge_eta)
    ):
        params[KEY_ETA] = round(settings.eta, 6)


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
