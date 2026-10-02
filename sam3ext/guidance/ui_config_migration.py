"""One-time ``ui-config.json`` migration for the PAG scale, DCW(+a) and CNS sliders of the Anima Guidance suite,
the Skimmed CFG ``Flip at`` slider and the Tile-Repair panel's negative prompt.

Forge's ``UiLoadsave`` (``modules/ui_loadsave.py:37-110``) reapplies every saved slider
``value``/``minimum``/``maximum``/``step`` keyed by
``customscript/anima_safe_pag.py/<tab>/<label>/<field>``. The DCW(+a) parity change moved
five sliders to the upstream ranges and defaults
(origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:636-667 lambda_l/lambda_h,
:675-704 alpha_l/alpha_h, :757-765 rdc_tau default 0.0) and dropped the separate RDC toggle,
but kept the labels. Without this migration an install that already has a ui-config keeps:

* the old RDC tau 0.15 while the saved ``Enable RDC = false`` is no longer applied (the
  hidden switch is ``do_not_save_to_config``), so ticking Enable DCW silently turns RDC on;
* the old slider bounds (``DCW lambda high`` ±0.5 step .005, CWM alpha max 1.0) and the old
  2× defaults (DCW 0.10/0.02, CWM 0.30/0.15).

Rules (user decision: new defaults = the originals, values a user set stay):

* PAG — the PAG parity change raised the ``Attn Scale`` maximum 15 -> 100 under the same label
  (origin: iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:201). A saved maximum 15
  gets the slider's saved bounds dropped; the saved scale stays (its default 4.0 did not change).
* RDC — only when the legacy ``Enable RDC/value`` key exists: a saved ``false`` beside the old
  default tau 0.15 sets the tau to 0.0 (RDC was off and the tau untouched; upstream's tau 0 is
  off). A tau the user set stays even beside a saved ``false`` (the change log says RDC now
  runs whenever Enable DCW is on), and ``true`` keeps the tau. Both legacy ``Enable RDC`` keys
  are removed, which also makes this step one-time.
* DCW — only when ``DCW lambda high`` still has a legacy bound (min -0.5, max 0.5 or step
  .005): its saved bounds are dropped (Forge then saves the new ones), a saved value equal to
  the old default 0.02 becomes 0.01 and anything outside ±0.3 is clamped like the server does;
  ``DCW lambda low`` equal to its old default 0.10 becomes 0.05 (its bounds did not change).
* CWM — per slider, only when the saved maximum is the legacy 1.0: bounds dropped, a saved
  value equal to the old default (low 0.30, high 0.15) becomes 0.0.
* CNS (origin: namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:391-439
  INPUT_TYPES) — ``CNS strength`` with the legacy step .01 and ``CNS gamma power`` with the
  legacy minimum .05 get their saved bounds dropped (a gamma power below the new minimum .1 is
  raised to it like the server clamp). ``CNS gamma scale`` changed its label (the old
  ``… (Anima 시작값 3.0)`` keys would never apply again): every legacy-label key is removed, and
  a saved value other than the old default 3.0 moves to the new label (clamped to [.1, 25]),
  so only the untouched old default falls back to upstream's 2.0.
* Skimmed CFG (``customscript/anima_skimmed_cfg.py/…``) — the ``Flip at`` slider kept its label
  but its step went .05 -> .01 (the vendored original, Extraltodeus/Skimmed_CFG@d8300583). A saved
  legacy step .05 gets the slider's saved bounds dropped; the saved value stays.
* Tile-Repair panel (``sam3ext/ui_anima.py``, not a script: keys are ``txt2img/<label>/<field>``)
  — only when the keys of the replaced ``SAM3 Anima Width``/``Height`` sliders are still there
  (the short-side slider took their place, so Forge never reads them again): they are removed,
  and ``SAM3 Anima Negative`` still at the old default ``blurry, low quality`` becomes sd-scripts'
  ``""`` (anima_minimal_inference.py ``--negative_prompt``). A negative the user typed stays.

Every rule is keyed on a legacy marker that the migration itself removes, so a value the user
saves afterwards is never touched again. Pure Python, no Gradio or Forge import.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Iterable, List, MutableMapping

__all__ = [
    "CNS_GAMMA_SCALE_LABEL",
    "PAG_SCALE_LABEL",
    "SCRIPT_FILE",
    "SKIMMED_FLIP_LABEL",
    "SKIMMED_SCRIPT_FILE",
    "TILE_REPAIR_NEGATIVE_LABEL",
    "TABS",
    "migrate_ui_settings",
    "migrate_ui_config_file",
]

SCRIPT_FILE = "anima_safe_pag.py"
TABS = ("txt2img", "img2img")

# The script's slider label (imported by scripts/anima_safe_pag.py); unchanged by the
# scale-100 parity change, so the old saved bounds would apply to it again.
PAG_SCALE_LABEL = "Attn Scale — PAG / SEG guidance scale (cond−weak 배율)"
# Pre-parity maximum (forge_sam3_extension@861ac02:scripts/anima_safe_pag.py:2702-2707).
_PAG_SCALE_LEGACY_MAXIMUM = 15.0

RDC_SWITCH_LABEL = "Enable RDC"
RDC_TAU_LABEL = "RDC tau (EMA 기억 구간)"
DCW_LOW_LABEL = "DCW lambda low"
DCW_HIGH_LABEL = "DCW lambda high"
CWM_LOW_LABEL = "CWM alpha low (초반 저주파 CFG)"
CWM_HIGH_LABEL = "CWM alpha high (후반 고주파 CFG)"

_RANGE_FIELDS = ("minimum", "maximum", "step")

# Pre-parity values of this extension (forge_sam3_extension@3522928:scripts/anima_safe_pag.py
# :3050-3060 DCW, :3064-3076 RDC, :3112-3122 CWM) -> upstream values.
_DCW_HIGH_LEGACY_BOUNDS = {"minimum": -0.5, "maximum": 0.5, "step": 0.005}
_RDC_TAU_OLD_DEFAULT = 0.15
_DCW_HIGH_LIMIT = 0.3  # upstream lambda_h min/max (dcw_node.py:650-667)
_DCW_OLD_TO_NEW = {DCW_LOW_LABEL: (0.10, 0.05), DCW_HIGH_LABEL: (0.02, 0.01)}
_CWM_LEGACY_MAXIMUM = 1.0
_CWM_OLD_TO_NEW = {CWM_LOW_LABEL: (0.30, 0.0), CWM_HIGH_LABEL: (0.15, 0.0)}

CNS_STRENGTH_LABEL = "CNS strength"
CNS_GAMMA_POWER_LABEL = "CNS gamma power"
# The script's slider label (imported by scripts/anima_safe_pag.py): upstream default 2.0,
# 3.0 is upstream README's "Flux / Anima + euler_ancestral_cfg_pp" row.
CNS_GAMMA_SCALE_LABEL = "CNS gamma scale (기본 2.0 · Anima+cfg_pp 권장 3.0)"
# Pre-parity CNS sliders (forge_sam3_extension@3522928:scripts/anima_safe_pag.py:3253-3270).
CNS_GAMMA_SCALE_LEGACY_LABEL = "CNS gamma scale (Anima 시작값 3.0)"
_CNS_STRENGTH_LEGACY_STEP = 0.01
_CNS_GAMMA_POWER_LEGACY_MINIMUM = 0.05
_CNS_GAMMA_POWER_MINIMUM = 0.1  # upstream gamma_power min (cns_sampler_patch.py:411)
_CNS_GAMMA_SCALE_OLD_DEFAULT = 3.0
_CNS_GAMMA_SCALE_BOUNDS = (0.1, 25.0)  # upstream gamma_scale min/max (:425-426)
_SLIDER_FIELDS = ("visible", "value") + _RANGE_FIELDS

# Skimmed CFG's own script (scripts/anima_skimmed_cfg.py); same label before and after the
# parity change, legacy step from forge_sam3_extension@4d0028e:scripts/anima_skimmed_cfg.py:399-403.
SKIMMED_SCRIPT_FILE = "anima_skimmed_cfg.py"
SKIMMED_FLIP_LABEL = "Flip at (%) · 0 = 사용 안 함"
_SKIMMED_FLIP_LEGACY_STEP = 0.05

# The Tile-Repair panel sits on the txt2img tab. Pre-parity values from
# forge_sam3_extension@4d0028e:sam3ext/ui_anima.py (negative default, Width/Height sliders).
TILE_REPAIR_TAB = "txt2img"
TILE_REPAIR_NEGATIVE_LABEL = "SAM3 Anima Negative"
_TILE_REPAIR_NEGATIVE_OLD_DEFAULT = "blurry, low quality"
_TILE_REPAIR_LEGACY_LABELS = ("SAM3 Anima Width", "SAM3 Anima Height")


def _key(tab: str, label: str, field: str, script_file: str) -> str:
    """The key ``UiLoadsave.add_component`` builds for a script control."""
    return f"customscript/{script_file}/{tab}/{label}/{field}"


def _panel_key(tab: str, label: str, field: str) -> str:
    """The key ``UiLoadsave.add_block`` builds for a control outside a script."""
    return f"{tab}/{label}/{field}"


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _same(value, reference: float) -> bool:
    number = _number(value)
    return number is not None and math.isclose(number, reference, rel_tol=0.0, abs_tol=1e-9)


def _replace_old_default(settings, key, old, new, changes, where) -> None:
    if key in settings and _same(settings[key], old):
        settings[key] = new
        changes.append(f"{where}: old default {old:g} -> upstream default {new:g}")


def _drop_bounds(settings, tab, label, script_file, changes) -> None:
    dropped = [
        field for field in _RANGE_FIELDS
        if settings.pop(_key(tab, label, field, script_file), None) is not None
    ]
    if dropped:
        changes.append(f"{tab}/{label}: dropped saved {'/'.join(dropped)} (new upstream range)")


def _migrate_rdc(settings, tab, script_file, changes) -> None:
    switch_key = _key(tab, RDC_SWITCH_LABEL, "value", script_file)
    if switch_key not in settings:
        return
    switch = settings.pop(switch_key)
    settings.pop(_key(tab, RDC_SWITCH_LABEL, "visible", script_file), None)
    tau_key = _key(tab, RDC_TAU_LABEL, "value", script_file)
    tau = _number(settings.get(tau_key))
    if switch is not True and tau is not None and not _same(tau, 0.0):
        where = f"{tab}/{RDC_TAU_LABEL}"
        if _same(tau, _RDC_TAU_OLD_DEFAULT):
            settings[tau_key] = 0.0
            changes.append(
                f"{where}: old default {tau:g} -> upstream default 0 "
                "(RDC off; the old Enable RDC switch was saved off)"
            )
        else:
            changes.append(
                f"{where}: kept the user-set {tau:g} although the old Enable RDC switch was "
                "saved off; RDC now runs whenever Enable DCW is on (set tau 0 to keep it off)"
            )
    changes.append(f"{tab}/{RDC_SWITCH_LABEL}: removed the legacy saved switch")


def _migrate_pag(settings, tab, script_file, changes) -> None:
    if _same(
        settings.get(_key(tab, PAG_SCALE_LABEL, "maximum", script_file)),
        _PAG_SCALE_LEGACY_MAXIMUM,
    ):
        _drop_bounds(settings, tab, PAG_SCALE_LABEL, script_file, changes)


def _migrate_dcw(settings, tab, script_file, changes) -> None:
    legacy = any(
        _same(settings.get(_key(tab, DCW_HIGH_LABEL, field, script_file)), old)
        for field, old in _DCW_HIGH_LEGACY_BOUNDS.items()
    )
    if not legacy:
        return
    _drop_bounds(settings, tab, DCW_HIGH_LABEL, script_file, changes)
    for label, (old, new) in _DCW_OLD_TO_NEW.items():
        _replace_old_default(
            settings, _key(tab, label, "value", script_file), old, new, changes,
            f"{tab}/{label}",
        )
    high_key = _key(tab, DCW_HIGH_LABEL, "value", script_file)
    high = _number(settings.get(high_key))
    if high is not None and abs(high) > _DCW_HIGH_LIMIT:
        clamped = math.copysign(_DCW_HIGH_LIMIT, high)
        settings[high_key] = clamped
        changes.append(f"{tab}/{DCW_HIGH_LABEL}: {high:g} -> {clamped:g} (upstream range ±0.3)")


def _migrate_cwm(settings, tab, script_file, changes) -> None:
    for label, (old, new) in _CWM_OLD_TO_NEW.items():
        if not _same(settings.get(_key(tab, label, "maximum", script_file)), _CWM_LEGACY_MAXIMUM):
            continue
        _drop_bounds(settings, tab, label, script_file, changes)
        _replace_old_default(
            settings, _key(tab, label, "value", script_file), old, new, changes,
            f"{tab}/{label}",
        )


def _migrate_cns(settings, tab, script_file, changes) -> None:
    if _same(
        settings.get(_key(tab, CNS_STRENGTH_LABEL, "step", script_file)),
        _CNS_STRENGTH_LEGACY_STEP,
    ):
        _drop_bounds(settings, tab, CNS_STRENGTH_LABEL, script_file, changes)

    if _same(
        settings.get(_key(tab, CNS_GAMMA_POWER_LABEL, "minimum", script_file)),
        _CNS_GAMMA_POWER_LEGACY_MINIMUM,
    ):
        _drop_bounds(settings, tab, CNS_GAMMA_POWER_LABEL, script_file, changes)
        power_key = _key(tab, CNS_GAMMA_POWER_LABEL, "value", script_file)
        power = _number(settings.get(power_key))
        if power is not None and power < _CNS_GAMMA_POWER_MINIMUM:
            settings[power_key] = _CNS_GAMMA_POWER_MINIMUM
            changes.append(
                f"{tab}/{CNS_GAMMA_POWER_LABEL}: {power:g} -> {_CNS_GAMMA_POWER_MINIMUM:g} "
                "(upstream minimum)"
            )

    legacy = {}
    for field in _SLIDER_FIELDS:
        key = _key(tab, CNS_GAMMA_SCALE_LEGACY_LABEL, field, script_file)
        if key in settings:
            legacy[field] = settings.pop(key)
    if not legacy:
        return
    where = f"{tab}/{CNS_GAMMA_SCALE_LEGACY_LABEL}"
    changes.append(f"{where}: removed the legacy-label keys ({'/'.join(legacy)})")
    value = _number(legacy.get("value"))
    new_key = _key(tab, CNS_GAMMA_SCALE_LABEL, "value", script_file)
    if value is None or new_key in settings:
        return
    if _same(value, _CNS_GAMMA_SCALE_OLD_DEFAULT):
        changes.append(f"{where}: old default {value:g} -> upstream default 2 (new label)")
        return
    low, high = _CNS_GAMMA_SCALE_BOUNDS
    carried = min(high, max(low, value))
    settings[new_key] = carried
    changes.append(f"{where}: saved {value:g} -> {CNS_GAMMA_SCALE_LABEL} = {carried:g}")


def _migrate_skimmed(settings, tab, changes) -> None:
    if _same(
        settings.get(_key(tab, SKIMMED_FLIP_LABEL, "step", SKIMMED_SCRIPT_FILE)),
        _SKIMMED_FLIP_LEGACY_STEP,
    ):
        _drop_bounds(settings, tab, SKIMMED_FLIP_LABEL, SKIMMED_SCRIPT_FILE, changes)


def _migrate_tile_repair(settings, changes) -> None:
    tab = TILE_REPAIR_TAB
    legacy = [
        _panel_key(tab, label, field)
        for label in _TILE_REPAIR_LEGACY_LABELS
        for field in _SLIDER_FIELDS
        if _panel_key(tab, label, field) in settings
    ]
    if not legacy:
        return
    for key in legacy:
        del settings[key]
    changes.append(
        f"{tab}/SAM3 Anima Width·Height: removed the keys of the replaced sliders "
        "(the short-side slider keeps the source aspect ratio)"
    )
    negative_key = _panel_key(tab, TILE_REPAIR_NEGATIVE_LABEL, "value")
    if settings.get(negative_key) == _TILE_REPAIR_NEGATIVE_OLD_DEFAULT:
        settings[negative_key] = ""
        changes.append(
            f"{tab}/{TILE_REPAIR_NEGATIVE_LABEL}: old default "
            f"{_TILE_REPAIR_NEGATIVE_OLD_DEFAULT!r} -> sd-scripts default ''"
        )


def migrate_ui_settings(
    settings: MutableMapping,
    *,
    script_file: str = SCRIPT_FILE,
    tabs: Iterable[str] = TABS,
) -> List[str]:
    """Migrate a loaded ui-config mapping in place; returns one line per change."""
    changes: List[str] = []
    tabs = tuple(tabs)
    for tab in tabs:
        _migrate_pag(settings, tab, script_file, changes)
        _migrate_rdc(settings, tab, script_file, changes)
        _migrate_dcw(settings, tab, script_file, changes)
        _migrate_cwm(settings, tab, script_file, changes)
        _migrate_cns(settings, tab, script_file, changes)
        _migrate_skimmed(settings, tab, changes)
    if TILE_REPAIR_TAB in tabs:
        _migrate_tile_repair(settings, changes)
    return changes


def migrate_ui_config_file(path, *, script_file: str = SCRIPT_FILE) -> List[str]:
    """Migrate ``ui-config.json`` on disk before Forge's ``UiLoadsave`` reads it.

    Run from ``on_before_ui`` (``webui.py`` calls it before ``ui.create_ui()``, which builds
    ``UiLoadsave`` at ``modules/ui.py:866``). A missing or unreadable file is left alone.
    When something changes, the original is first copied next to it
    (``<name>.bak-anima-guidance-<timestamp>``) and the new file is written atomically in
    Forge's own format (``ui_loadsave.write_to_file``: UTF-8 without BOM, indent 4,
    text-mode newlines)."""
    path = os.fspath(path)
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
        settings = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(settings, dict):
        return []
    changes = migrate_ui_settings(settings, script_file=script_file)
    if not changes:
        return []
    backup = f"{path}.bak-anima-guidance-{time.strftime('%Y%m%d-%H%M%S')}"
    with open(backup, "xb") as handle:
        handle.write(raw)
    temporary = f"{path}.tmp-anima-guidance"
    # Text mode like Forge's write_to_file, so the platform newline matches its own dumps.
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=4, ensure_ascii=False)
    os.replace(temporary, path)
    return changes + [f"backup: {backup}"]
