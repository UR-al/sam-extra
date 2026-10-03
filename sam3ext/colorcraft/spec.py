# Colorcraft (sam-extra) — the panel's argument model: field tables, flat script args <-> modifier
# chain and mask specs, the infotext format, and readers for upstream's and the fork's infotext.
#
# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:scripts/colorcraft.py
#   * the stack shape: 10 modifier tabs (I-X), 10 mask leaves (M1-M10), 5 combos (C1-C5) — :40-48
#   * slider ranges/defaults of every control that upstream's Forge panel has — :445-628
#   * MODIFIER_FIELDS/LEAF_FIELDS/COMBO_FIELDS order — :67-80
#   * build_leaf_spec / build_all_mask_specs semantics (blur node around a leaf or combo, a modifier
#     on an incomplete combo runs unmasked) — :266-322
#   * compute_mask_activeness (which masks/combos are "in use") — :142-189
#   * the "Colorcraft" infotext format MOD(...)MASK(...)COMBO(...) read by ``parse_upstream`` — :332-396
# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a:nodes.py — the node kinds and which parameters each
#   node's chain entry carries (ColorcraftBasic … ColorcraftShift, :73-390); the dev overrides and the
#   more_colors / color_shift gates of ColorcraftAdvanced (:131-212) with their node ranges.
# origin: aoleg/ComfyUI-Colorcraft@f00066c63c9d8f96abc119cada3b51b36688fcac:lib_colorcraft/spec.py, params.py
#   * the fork's "Colorcraft" infotext ("v1;name=value;…", defaults omitted) read by ``parse_fork``,
#     with the fork's parameter names, defaults and its mask-preview chain (spec.py:45-117, params.py:91-203)
#
# MIT License
#
# Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)
# (The fork aoleg/ComfyUI-Colorcraft — Oleg Afonin's Forge Neo port — is under the same MIT License;
#  its LICENSE carries the upstream notice with the earlier spelling "Sahand Ahmadiantehrani".)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes by sam-extra (2026-10-03) — this file is new code built on the sources above:
#   * Every modifier tab picks the ComfyUI node it stands for ("Type": Advanced, Basic, Luma, Chroma,
#     Chroma Plus, Punch, Shift) and runs exactly that node's math and parameter set. Upstream's Forge
#     tab is the Advanced node with Chroma Plus and Color Shift always applied; here those two keep the
#     node's gates (``more_colors``, ``color_shift``) and start ON, so the default behaves like
#     upstream's Forge tab. The node's dev overrides (recenter, max chroma, chroma plane) are offered
#     too; as in the node, each override's own checkbox is the gate.
#   * Upstream's per-tab "Hires. Pass" checkbox (base only / hires only) becomes Pass = Base / Hires /
#     Both.
#   * Modifier I starts Active (upstream: every tab inactive) so ticking Enable and moving a slider works.
#   * "Advanced Schedule" off resets bias, exponent and both offsets (upstream's Forge script; the
#     ComfyUI node keeps a hidden bias).
#   * Infotext: this extension writes its own key ``SAM Extra Colorcraft`` ("v1;…", defaults omitted, only
#     what the modifier's type reads) and reads upstream's and the fork's ``Colorcraft`` key on paste.
#   * API: the first script argument may carry the whole set-up in one value (that infotext string, or
#     a dict of argument paths) instead of 579 positional values (``is_compact_arg``).
"""Flat script arguments <-> the Colorcraft modifier chain, mask specs and infotext."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .engine import KINDS, needs_basis
from .masking import MASK_AXIS_OPTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODIFIER_COUNT = 10
MASK_COUNT = 10
COMBO_COUNT = 5
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
MODIFIER_TAGS = ROMAN[:MODIFIER_COUNT]
MASK_TAGS = [f"M{i + 1}" for i in range(MASK_COUNT)]
COMBO_TAGS = [f"C{i + 1}" for i in range(COMBO_COUNT)]

KIND_LABELS = {
    "advanced": "Advanced",
    "basic": "Basic",
    "luma": "Luma",
    "chroma": "Chroma",
    "chroma_plus": "Chroma Plus",
    "punch": "Punch",
    "shift": "Shift",
}
KIND_CHOICES = [KIND_LABELS[kind] for kind in KINDS]
KIND_IDS = {label: kind for kind, label in KIND_LABELS.items()}
PASS_BASE, PASS_HIRES, PASS_BOTH = "Base", "Hires", "Both"
PASS_CHOICES = [PASS_BASE, PASS_HIRES, PASS_BOTH]
MASK_NONE = "none"
MASK_CHOICES = [MASK_NONE] + MASK_TAGS + COMBO_TAGS
MASK_MODE_OPTIONS = ["highs", "lows", "split", "range", "protect range"]
COMBO_OP_OPTIONS = ["and", "or", "subtract", "xor"]
COLOR_SHIFT_MODE_OPTIONS = ["default", "legacy"]
CHROMA_PLANE_OPTIONS = ["temp_tint", "lab"]

INFOTEXT_KEY = "SAM Extra Colorcraft"
STATUS_KEY = "SAM Extra Colorcraft status"
FOREIGN_KEY = "Colorcraft"          # upstream muerrilla and the aoleg fork both write this key
VERSION = "v1"

FLOAT, BOOL, CHOICE = "float", "bool", "choice"


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    default: Any
    label: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple = ()
    group: str = ""
    # Where the range/default comes from: "forge" = upstream's Forge panel widget
    # (scripts/colorcraft.py), "node" = the ComfyUI node's INPUT_TYPES, "sam-extra" = this extension.
    origin: str = "forge"


def _float(name, default, label, minimum, maximum, step=0.01, group="", origin="forge"):
    return Field(name, FLOAT, default, label, minimum, maximum, step, (), group, origin)


def _bool(name, default, label, group="", origin="forge"):
    return Field(name, BOOL, default, label, group=group, origin=origin)


def _choice(name, default, label, choices, group="", origin="forge"):
    return Field(name, CHOICE, default, label, choices=tuple(choices), group=group, origin=origin)


# Per modifier tab, in argument order. Ranges: upstream's Forge panel (scripts/colorcraft.py:445-503)
# where it has the control, the ComfyUI node otherwise (dev overrides, the two gates).
MODIFIER_FIELDS: tuple[Field, ...] = (
    _bool("active", False, "Active", "head"),
    _choice("kind", "Advanced", "Type", KIND_CHOICES, "head", origin="sam-extra"),
    _choice("pass", PASS_BASE, "Pass", PASS_CHOICES, "head", origin="sam-extra"),
    _choice("mask", MASK_NONE, "Mask ➜", MASK_CHOICES, "head"),
    _float("strength", 1.0, "Strength", -1.0, 1.0, group="schedule"),
    _float("start", 0.5, "Start", 0.0, 1.0, group="schedule"),
    _float("end", 0.75, "End", 0.0, 1.0, group="schedule"),
    _bool("advanced", False, "Advanced Schedule", "shaping"),
    _float("exponent", 0.0, "Exponent", 0.0, 3.0, group="shaping"),
    _float("bias", 0.5, "Bias", 0.0, 1.0, group="shaping"),
    _float("start_off", 0.0, "Start Offset", -1.0, 1.0, group="shaping"),
    _float("end_off", 0.0, "End Offset", -1.0, 1.0, group="shaping"),
    _bool("smooth", True, "Smooth", "shaping"),
    _float("exposure", 0.0, "Exposure", -4.0, 4.0, group="luma"),
    _float("tone_compression", 0.0, "Tone Compression", -1.0, 1.0, group="luma"),
    _float("contrast", 0.0, "Contrast", -4.0, 4.0, group="contrast"),
    _float("clarity", 0.0, "Clarity", -4.0, 4.0, group="detail"),
    _float("sharpness", 0.0, "Sharpness", -4.0, 4.0, group="detail"),
    _float("temperature", 0.0, "Temperature", -4.0, 4.0, group="chroma"),
    _float("tint", 0.0, "Tint", -4.0, 4.0, group="chroma"),
    _float("vibrance", 0.0, "Vibrance", -1.0, 1.0, group="chroma"),
    _float("saturation", 0.0, "Saturation", -1.0, 1.0, group="chroma"),
    _float("chroma_contrast", 0.0, "Chroma Contrast", -4.0, 4.0, group="chroma"),
    _float("chroma_center", 0.0, "Chroma Center", -1.0, 1.0, group="chroma"),
    _bool("more_colors", True, "Apply Chroma Plus", "chroma_plus", origin="node"),
    _float("temp_plus_tint", 0.0, "Temp+Tint", -4.0, 4.0, group="chroma_plus"),
    _float("temp_minus_tint", 0.0, "Temp-Tint", -4.0, 4.0, group="chroma_plus"),
    _float("lab_a", 0.0, "Lab A", -4.0, 4.0, group="chroma_plus"),
    _float("lab_b", 0.0, "Lab B", -4.0, 4.0, group="chroma_plus"),
    _float("lab_a_plus_b", 0.0, "Lab A+B", -4.0, 4.0, group="chroma_plus"),
    _float("lab_a_minus_b", 0.0, "Lab A-B", -4.0, 4.0, group="chroma_plus"),
    _bool("color_shift", True, "Apply Color Shift", "color_shift", origin="node"),
    _float("color_shift_amount", 0.0, "Amount", -4.0, 4.0, group="color_shift"),
    _choice("color_shift_mode", "default", "Mode", COLOR_SHIFT_MODE_OPTIONS, "color_shift"),
    _float("color_shift_red", 0.0, "Red", -2.0, 2.0, group="color_shift"),
    _float("color_shift_green", 0.0, "Green", -2.0, 2.0, group="color_shift"),
    _float("color_shift_blue", 0.0, "Blue", -2.0, 2.0, group="color_shift"),
    _float("color_shift_brightness", 0.0, "Brightness", -2.0, 2.0, group="color_shift"),
    _bool("recenter_override", False, "Override Recenter", "dev", origin="node"),
    _float("recenter", 0.5, "Recenter", 0.0, 1.0, group="dev", origin="node"),
    _bool("max_chroma_override", False, "Override Max Chroma", "dev", origin="node"),
    _float("max_chroma", 2.5, "Max Chroma", 0.0, 10.0, group="dev", origin="node"),
    _bool("chroma_plane_override", False, "Override Chroma Plane", "dev", origin="node"),
    _choice("chroma_plane", "temp_tint", "Chroma Plane", CHROMA_PLANE_OPTIONS, "dev", origin="node"),
)

# Per mask leaf (upstream LEAF_FIELDS order, scripts/colorcraft.py:79; widgets :556-578).
LEAF_FIELDS: tuple[Field, ...] = (
    _choice("mask_axis", MASK_AXIS_OPTIONS[0], "Mask Axis", MASK_AXIS_OPTIONS),
    _choice("mask_mode", "highs", "Mask Mode", MASK_MODE_OPTIONS),
    _float("mask_strength", 1.0, "Strength", 0.0, 1.0),
    _float("mask_width", 0.0, "Width", 0.0, 2.0),
    _float("mask_center", 0.0, "Center", -1.0, 1.0),
    _float("mask_hardness", 1.0, "Hardness", 0.0, 10.0, 0.1),
    _float("blur", 0.0, "Blur", 0.0, 64.0, 0.1),
    _float("spread", 0.0, "Spread", -3.0, 3.0),
    _float("contrast", 0.0, "Contrast", -10.0, 10.0),
    _bool("normalize", False, "Normalize"),
)


def combo_ref_choices(index):
    """Mask A/B choices of combo ``index`` (0-based): leaves and earlier combos only (upstream :604)."""
    return [MASK_NONE] + MASK_TAGS + COMBO_TAGS[:index]


# Per combo (upstream COMBO_FIELDS order, :80; widgets :604-612). Mask A/B choices depend on the combo.
COMBO_FIELDS: tuple[Field, ...] = (
    _choice("mask_a", MASK_NONE, "Mask A", combo_ref_choices(COMBO_COUNT)),
    _choice("mask_b", MASK_NONE, "Mask B", combo_ref_choices(COMBO_COUNT)),
    _choice("operation", "and", "Operation", COMBO_OP_OPTIONS),
    _float("blur", 0.0, "Blur", 0.0, 64.0, 0.1),
    _float("spread", 0.0, "Spread", -3.0, 3.0),
    _float("contrast", 0.0, "Contrast", -10.0, 10.0),
    _bool("normalize", False, "Normalize"),
)

GLOBAL_FIELDS: tuple[Field, ...] = (
    _bool("enabled", False, "Enable Colorcraft", origin="sam-extra"),
    _bool("masking", False, "Enable masking", origin="forge"),
)

# The Debug panel's two generation-time values, appended after the combos (upstream :614-622; its
# axis/mask/colour selections are panel-only and are not script arguments here).
DEBUG_FIELDS: tuple[Field, ...] = (
    _bool("debug", False, "Capture debug latent", origin="sam-extra"),
    _float("debug_step", 5, "Debug Step", 0, 50, 1),
)

MODIFIER_BY_NAME = {f.name: f for f in MODIFIER_FIELDS}
LEAF_BY_NAME = {f.name: f for f in LEAF_FIELDS}
COMBO_BY_NAME = {f.name: f for f in COMBO_FIELDS}

N_GLOBAL = len(GLOBAL_FIELDS)
N_MODIFIER = len(MODIFIER_FIELDS)
N_LEAF = len(LEAF_FIELDS)
N_COMBO = len(COMBO_FIELDS)
MODIFIER_OFFSET = N_GLOBAL
LEAF_OFFSET = MODIFIER_OFFSET + N_MODIFIER * MODIFIER_COUNT
COMBO_OFFSET = LEAF_OFFSET + N_LEAF * MASK_COUNT
DEBUG_OFFSET = COMBO_OFFSET + N_COMBO * COMBO_COUNT
ARG_COUNT = DEBUG_OFFSET + len(DEBUG_FIELDS)

# Which parameters each node's chain entry carries (``make`` of each ComfyUI node class; the panel's
# colour-shift controls are named color_shift_* where the nodes use mode/red/green/blue/brightness).
_SHIFT_KEYS = (("color_shift_amount", "color_shift_amount"), ("mode", "color_shift_mode"),
               ("red", "color_shift_red"), ("green", "color_shift_green"),
               ("blue", "color_shift_blue"), ("brightness", "color_shift_brightness"))
_CHROMA_PLUS_KEYS = ("temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b")
_KIND_PARAMS = {
    "basic": (("contrast", "contrast"),) + _SHIFT_KEYS,
    "advanced": tuple((name, name) for name in (
        "exposure", "tone_compression", "contrast", "clarity", "sharpness",
        "temperature", "tint", "vibrance", "saturation", "chroma_contrast", "chroma_center",
        "more_colors") + _CHROMA_PLUS_KEYS + ("color_shift",)) + _SHIFT_KEYS + tuple(
        (name, name) for name in ("recenter_override", "recenter", "max_chroma_override",
                                  "max_chroma", "chroma_plane_override", "chroma_plane")),
    "luma": (("exposure", "exposure"), ("tone_compression", "tone_compression")),
    "chroma": tuple((name, name) for name in (
        "temperature", "tint", "vibrance", "saturation", "chroma_contrast", "chroma_center")),
    "chroma_plus": tuple((name, name) for name in _CHROMA_PLUS_KEYS),
    "punch": (("contrast", "contrast"), ("clarity", "clarity"), ("sharpness", "sharpness")),
    "shift": _SHIFT_KEYS,
}

# Panel groups each Type shows (ui.py) — the node's own widgets.
KIND_GROUPS = {
    "advanced": ("luma", "contrast", "detail", "chroma", "chroma_plus", "color_shift", "dev"),
    "basic": ("contrast", "color_shift"),
    "luma": ("luma",),
    "chroma": ("chroma",),
    "chroma_plus": ("chroma_plus",),
    "punch": ("contrast", "detail"),
    "shift": ("color_shift",),
}
# The two Advanced-node gates; the other nodes have no such switch.
KIND_GATES = {"advanced": ("more_colors", "color_shift")}
# ColorcraftBasic has no masking input.
KIND_TAKES_MASK = {kind: kind != "basic" for kind in KINDS}

_SCHEDULE_FIELDS = ("strength", "start", "end", "bias", "exponent", "start_off", "end_off", "smooth")
_SHAPING_FIELDS = ("exponent", "bias", "start_off", "end_off", "smooth")

# XYZ axes act on modifier I (and switch it on); "enabled" is the panel's master switch.
XYZ_MODIFIER_FIELDS = (
    "strength", "start", "end", "exposure", "tone_compression", "contrast", "clarity", "sharpness",
    "temperature", "tint", "vibrance", "saturation", "chroma_contrast",
)
XYZ_ATTR = "_sam3_colorcraft_xyz"


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


def _finite(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _truthy(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", ""):
        return False
    return default


def _choice_value(field_, value, choices=None):
    choices = list(choices if choices is not None else field_.choices)
    if value in choices:
        return value
    text = str(value).strip() if value is not None else ""
    lowered = {str(c).lower(): c for c in choices}
    if text.lower() in lowered:
        return lowered[text.lower()]
    if field_.name == "kind":
        # node ids (``chroma_plus``) are accepted too
        kind = text.lower().replace(" ", "_")
        if kind in KIND_LABELS:
            return KIND_LABELS[kind]
    return field_.default


def coerce(field_, value, choices=None):
    """``value`` as ``field_``'s type; anything unusable becomes the field's default."""
    if field_.kind == FLOAT:
        return _finite(value, field_.default)
    if field_.kind == BOOL:
        return _truthy(value, field_.default)
    return _choice_value(field_, value, choices)


def modifier_default(index, name):
    """Defaults of tab ``index`` — modifier I starts Active."""
    if name == "active":
        return index == 0
    return MODIFIER_BY_NAME[name].default


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    enabled: bool = False
    masking: bool = False
    modifiers: list = field(default_factory=list)   # MODIFIER_COUNT dicts, MODIFIER_FIELDS names
    leaves: list = field(default_factory=list)      # MASK_COUNT dicts, LEAF_FIELDS names
    combos: list = field(default_factory=list)      # COMBO_COUNT dicts, COMBO_FIELDS names
    debug: bool = False                             # Debug panel: capture the pre-edit x0 …
    debug_step: float = 5                           # … at this step

    def copy(self):
        return Config(self.enabled, self.masking, [dict(m) for m in self.modifiers],
                      [dict(m) for m in self.leaves], [dict(c) for c in self.combos], self.debug,
                      self.debug_step)


def default_config():
    return Config(
        enabled=False,
        masking=False,
        modifiers=[{f.name: modifier_default(i, f.name) for f in MODIFIER_FIELDS} for i in range(MODIFIER_COUNT)],
        leaves=[{f.name: f.default for f in LEAF_FIELDS} for _ in range(MASK_COUNT)],
        combos=[{f.name: f.default for f in COMBO_FIELDS} for _ in range(COMBO_COUNT)],
    )


def blank_config():
    """Every tab inactive — the starting point when an infotext says which tabs were active."""
    config = default_config()
    for modifier in config.modifiers:
        modifier["active"] = False
    return config


def arg_names():
    """``"enabled"``, ``"masking"``, ``"I.active"`` … ``"C5.normalize"``, ``"debug"``, ``"debug_step"``."""
    names = [f.name for f in GLOBAL_FIELDS]
    for tag in MODIFIER_TAGS:
        names += [f"{tag}.{f.name}" for f in MODIFIER_FIELDS]
    for tag in MASK_TAGS:
        names += [f"{tag}.{f.name}" for f in LEAF_FIELDS]
    for tag in COMBO_TAGS:
        names += [f"{tag}.{f.name}" for f in COMBO_FIELDS]
    names += [f.name for f in DEBUG_FIELDS]
    return names


# API convenience (sam-extra): instead of 579 positional values an API caller may put ONE value in the
# first argument (the Enable checkbox's slot, which the panel always fills with a bool):
#   * this extension's infotext value, ``"v1;mods=I;I.exposure=0.3"`` (what ``to_infotext`` writes), or
#   * a dict of ``arg_names()`` paths, ``{"I.exposure": 0.3, "M1.mask_axis": "hue", ...}`` — missing
#     paths keep the panel defaults (modifier I Active), unknown ones are ignored, values are coerced like
#     positional ones, and ``"enabled"`` defaults to True (sending the dict asks for Colorcraft).
# The remaining positional arguments are then ignored. Forge's API fills them with the panel defaults.
_COMPACT_TEXT_RE = re.compile(r"v1\s*;\s*mods\s*=")


def is_compact_arg(value):
    """Is ``value`` the API's one-value form (an infotext string or a dict of argument paths)?"""
    if isinstance(value, dict):
        return True
    return isinstance(value, str) and _COMPACT_TEXT_RE.match(value.strip()) is not None


def config_from_mapping(mapping):
    """``{"enabled": True, "I.exposure": 0.3, …}`` (``arg_names()`` paths) as a ``Config``."""
    config = default_config()
    config.enabled = _truthy(mapping.get("enabled", True), True)
    config.masking = _truthy(mapping.get("masking", False), False)
    for key, value in mapping.items():
        if value is None:
            continue
        tag, _, name = str(key).partition(".")
        if tag in MODIFIER_TAGS and name in MODIFIER_BY_NAME:
            config.modifiers[MODIFIER_TAGS.index(tag)][name] = coerce(MODIFIER_BY_NAME[name], value)
        elif tag in MASK_TAGS and name in LEAF_BY_NAME:
            config.leaves[MASK_TAGS.index(tag)][name] = coerce(LEAF_BY_NAME[name], value)
        elif tag in COMBO_TAGS and name in COMBO_BY_NAME:
            index = COMBO_TAGS.index(tag)
            choices = combo_ref_choices(index) if name in ("mask_a", "mask_b") else None
            config.combos[index][name] = coerce(COMBO_BY_NAME[name], value, choices)
    config.debug = coerce(DEBUG_FIELDS[0], mapping.get("debug", DEBUG_FIELDS[0].default))
    config.debug_step = coerce(DEBUG_FIELDS[1], mapping.get("debug_step", DEBUG_FIELDS[1].default))
    return config


def args_enabled(args):
    """Does the first script argument switch Colorcraft on? Reads nothing else (the hook's fast path)."""
    if not args:
        return False
    first = args[0]
    if isinstance(first, dict):
        return _truthy(first.get("enabled", True), True)
    if is_compact_arg(first):
        return True
    return bool(coerce(GLOBAL_FIELDS[0], first))


def config_from_args(args):
    """The script's positional arguments (``ui()`` order) as a ``Config``. Missing ones are defaults.

    A compact first argument (``is_compact_arg``) replaces all of them."""
    args = list(args or ())
    if args and isinstance(args[0], dict):
        return config_from_mapping(args[0])
    if args and is_compact_arg(args[0]):
        return parse_own(args[0])
    config = default_config()

    def arg(index, default):
        return args[index] if index < len(args) else default

    config.enabled = _truthy(arg(0, False), False)
    config.masking = _truthy(arg(1, False), False)
    for i in range(MODIFIER_COUNT):
        base = MODIFIER_OFFSET + i * N_MODIFIER
        for j, f in enumerate(MODIFIER_FIELDS):
            default = modifier_default(i, f.name)
            value = arg(base + j, default)
            config.modifiers[i][f.name] = default if value is None else coerce(f, value)
    for i in range(MASK_COUNT):
        base = LEAF_OFFSET + i * N_LEAF
        for j, f in enumerate(LEAF_FIELDS):
            config.leaves[i][f.name] = coerce(f, arg(base + j, f.default))
    for i in range(COMBO_COUNT):
        base = COMBO_OFFSET + i * N_COMBO
        for j, f in enumerate(COMBO_FIELDS):
            choices = combo_ref_choices(i) if f.name in ("mask_a", "mask_b") else None
            config.combos[i][f.name] = coerce(f, arg(base + j, f.default), choices)
    config.debug = coerce(DEBUG_FIELDS[0], arg(DEBUG_OFFSET, DEBUG_FIELDS[0].default))
    config.debug_step = coerce(DEBUG_FIELDS[1], arg(DEBUG_OFFSET + 1, DEBUG_FIELDS[1].default))
    return config


def config_to_args(config):
    """Inverse of ``config_from_args`` (tests, XYZ and paste helpers)."""
    args = [bool(config.enabled), bool(config.masking)]
    for modifier in config.modifiers:
        args += [modifier[f.name] for f in MODIFIER_FIELDS]
    for leaf in config.leaves:
        args += [leaf[f.name] for f in LEAF_FIELDS]
    for combo in config.combos:
        args += [combo[f.name] for f in COMBO_FIELDS]
    args += [bool(config.debug), config.debug_step]
    return args


def apply_xyz(config, xyz):
    """XYZ plot overrides (``p._sam3_colorcraft_xyz``): the master switch and modifier I's values.

    Any modifier-I value also switches modifier I on — an axis that sweeps an amount is meant to show it."""
    if not xyz:
        return config
    config = config.copy()
    if "enabled" in xyz:
        config.enabled = _truthy(xyz["enabled"], config.enabled)
    touched = False
    for name in XYZ_MODIFIER_FIELDS:
        if name in xyz:
            config.modifiers[0][name] = coerce(MODIFIER_BY_NAME[name], xyz[name])
            touched = True
    if touched:
        config.modifiers[0]["active"] = True
    return config


# ---------------------------------------------------------------------------
# Modifier chain and mask specs
# ---------------------------------------------------------------------------


def kind_of(modifier):
    return KIND_IDS.get(modifier.get("kind"), "advanced")


def runs_on_pass(modifier, is_hires):
    """Pass = Base runs on the first pass (and img2img), Hires on the hires pass, Both on either."""
    which = modifier.get("pass", PASS_BASE)
    if which == PASS_BOTH:
        return True
    return (which == PASS_HIRES) == bool(is_hires)


def schedule_of(modifier):
    """The node's schedule dict. Shaping off resets bias/exponent/offsets (upstream Forge :738-739)."""
    sched = {name: modifier[name] for name in _SCHEDULE_FIELDS}
    if not modifier.get("advanced"):
        sched["bias"], sched["exponent"], sched["start_off"], sched["end_off"] = 0.5, 0.0, 0.0, 0.0
    return sched


def params_of(modifier):
    """Exactly the parameters the modifier's node puts into its chain entry, under the node's names."""
    return {node_name: modifier[panel_name] for node_name, panel_name in _KIND_PARAMS[kind_of(modifier)]}


def schedule_is_zero(sched):
    """``make_schedule`` with amount 0 and both offsets 0 is zero at every step."""
    return sched["strength"] == 0 and sched["start_off"] == 0 and sched["end_off"] == 0


def params_are_noop(kind, p):
    """True when every edit of this entry is at 0 — the node would hand back its input."""
    if kind == "basic":
        return p["contrast"] == 0 and p["color_shift_amount"] == 0
    if kind == "advanced":
        amounts = ["exposure", "tone_compression", "contrast", "clarity", "sharpness", "temperature",
                   "tint", "vibrance", "saturation", "chroma_contrast"]
        if p["more_colors"]:
            amounts += list(_CHROMA_PLUS_KEYS)
        if p["color_shift"]:
            amounts.append("color_shift_amount")
        return all(p[name] == 0 for name in amounts)
    if kind == "luma":
        return p["exposure"] == 0 and p["tone_compression"] == 0
    if kind == "chroma":
        return all(p[name] == 0 for name in ("temperature", "tint", "vibrance", "saturation", "chroma_contrast"))
    if kind == "chroma_plus":
        return all(p[name] == 0 for name in _CHROMA_PLUS_KEYS)
    if kind == "punch":
        return all(p[name] == 0 for name in ("contrast", "clarity", "sharpness"))
    if kind == "shift":
        return p["color_shift_amount"] == 0
    return True


def _leaf_shaping_is_default(values):
    return values["blur"] == 0 and values["spread"] == 0 and values["contrast"] == 0 and not values["normalize"]


# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a:scripts/colorcraft.py:266-273 (build_leaf_spec)
def build_leaf_spec(leaf):
    spec = {
        "mask_axis": leaf["mask_axis"], "mask_mode": leaf["mask_mode"], "mask_center": leaf["mask_center"],
        "mask_hardness": leaf["mask_hardness"], "mask_width": leaf["mask_width"],
        "mask_strength": leaf["mask_strength"],
    }
    if not _leaf_shaping_is_default(leaf):
        return {"blur": leaf["blur"], "spread": leaf["spread"], "contrast": leaf["contrast"],
                "normalize": leaf["normalize"], "a": spec}
    return spec


def build_mask_specs(config):
    """``(leaf_specs, combo_specs)`` keyed by tag; an incomplete combo is None (upstream :276-312).

    A combo can only reference leaves and earlier combos, so one pass in order resolves every reference."""
    leaf_specs = {MASK_TAGS[i]: build_leaf_spec(leaf) for i, leaf in enumerate(config.leaves)}
    combo_specs = {}

    def resolve(ref):
        if ref == MASK_NONE:
            return None
        if ref in combo_specs:
            return combo_specs[ref]
        return leaf_specs.get(ref)

    for i, combo in enumerate(config.combos):
        a_spec = resolve(combo["mask_a"])
        b_spec = resolve(combo["mask_b"])
        if a_spec is None or b_spec is None:
            combo_specs[COMBO_TAGS[i]] = None
            continue
        inner = {"operation": combo["operation"], "a": a_spec, "b": b_spec}
        if not _leaf_shaping_is_default(combo):
            combo_specs[COMBO_TAGS[i]] = {"blur": combo["blur"], "spread": combo["spread"],
                                          "contrast": combo["contrast"], "normalize": combo["normalize"],
                                          "a": inner}
        else:
            combo_specs[COMBO_TAGS[i]] = inner
    return leaf_specs, combo_specs


def mask_spec_for(config, modifier, specs=None):
    """The mask spec gating this modifier, or None (masking off, none selected, incomplete combo)."""
    if not config.masking or not KIND_TAKES_MASK[kind_of(modifier)]:
        return None
    selection = modifier.get("mask", MASK_NONE)
    if selection == MASK_NONE:
        return None
    leaf_specs, combo_specs = specs if specs is not None else build_mask_specs(config)
    if selection in leaf_specs:
        return leaf_specs[selection]
    return combo_specs.get(selection)


def active_masks(config):
    """``(leaf tags, combo tags)`` the active modifiers' selections reach (upstream :142-189)."""
    leaves, combos = [], []
    if not config.masking:
        return leaves, combos
    pending = []
    for modifier in config.modifiers:
        if modifier["active"] and KIND_TAKES_MASK[kind_of(modifier)] and modifier["mask"] != MASK_NONE:
            pending.append(modifier["mask"])
    seen = set()
    while pending:
        tag = pending.pop()
        if tag in seen or tag == MASK_NONE:
            continue
        seen.add(tag)
        if tag in COMBO_TAGS:
            combo = config.combos[COMBO_TAGS.index(tag)]
            pending += [combo["mask_a"], combo["mask_b"]]
    leaves = [tag for tag in MASK_TAGS if tag in seen]
    combos = [tag for tag in COMBO_TAGS if tag in seen]
    return leaves, combos


@dataclass
class PassChain:
    """The chain entries one sampling pass runs, plus what the status line reports about them."""
    entries: list
    tags: list
    unresolved_masks: list
    needs_basis: bool


def build_chain(config, is_hires):
    """Chain entries (``{"kind", "params", "schedule", "mask"}`` like the nodes build) for one pass.

    Inactive tabs, tabs for the other pass and entries that cannot change anything are left out."""
    specs = build_mask_specs(config)
    entries, tags, unresolved = [], [], []
    if not config.enabled:
        return PassChain(entries, tags, unresolved, False)
    for i, modifier in enumerate(config.modifiers):
        if not modifier["active"] or not runs_on_pass(modifier, is_hires):
            continue
        kind = kind_of(modifier)
        sched = schedule_of(modifier)
        params = params_of(modifier)
        if schedule_is_zero(sched) or params_are_noop(kind, params):
            continue
        mask = mask_spec_for(config, modifier, specs)
        if (mask is None and config.masking and KIND_TAKES_MASK[kind]
                and modifier["mask"] in COMBO_TAGS):
            unresolved.append(f"{MODIFIER_TAGS[i]}->{modifier['mask']}")
        entries.append({"kind": kind, "params": params, "schedule": sched, "mask": mask})
        tags.append(MODIFIER_TAGS[i])
    return PassChain(entries, tags, unresolved, needs_basis(entries))


# ---------------------------------------------------------------------------
# Infotext — this extension's own key
# ---------------------------------------------------------------------------


def _fmt(field_, value):
    if field_.kind == BOOL:
        return "1" if value else "0"
    if field_.kind == FLOAT:
        # Ten significant digits: every value a slider, a typed number field (unclamped, see
        # javascript/colorcraft_sliders.js), XYZ or the API realistically gives comes back exactly on
        # paste, while binary noise (0.30000000000000004) still prints as 0.3. ``round(v, 6):g`` lost
        # digits past the sixth significant one (123.456789 -> 123.457) and turned 1e-7 into 0.
        return f"{float(value):.10g}"
    return str(value)


def _relevant_modifier_fields(modifier):
    """The controls of the modifier's own Type — the ones worth writing and restoring.

    Head (type, pass, mask when the type takes one), the whole schedule and every parameter of the
    Type's node, gated groups included (a pasted look restores the tab as it was). Controls of the
    other Types stay out: the tab hides them and they cannot change anything."""
    kind = kind_of(modifier)
    names = ["kind", "pass"]
    if KIND_TAKES_MASK[kind]:
        names.append("mask")
    names += ["strength", "start", "end", "advanced", *_SHAPING_FIELDS]
    for _, panel_name in _KIND_PARAMS[kind]:
        if panel_name not in names:
            names.append(panel_name)
    return names


def to_infotext(config):
    """``v1;mods=I,III;masking=1;masks=M1;combos=;I.exposure=0.3;…`` — non-default values only.

    Only active tabs, only the controls of each tab's Type, and only the masks/combos they reach."""
    active = [i for i, m in enumerate(config.modifiers) if m["active"]]
    parts = [VERSION, "mods=" + ",".join(MODIFIER_TAGS[i] for i in active)]
    leaves, combos = active_masks(config)
    if config.masking:
        parts.append("masking=1")
        parts.append("masks=" + ",".join(leaves))
        parts.append("combos=" + ",".join(combos))
    for i in active:
        modifier = config.modifiers[i]
        for name in _relevant_modifier_fields(modifier):
            f = MODIFIER_BY_NAME[name]
            if modifier[name] != f.default:
                parts.append(f"{MODIFIER_TAGS[i]}.{name}={_fmt(f, modifier[name])}")
    for tag in leaves:
        leaf = config.leaves[MASK_TAGS.index(tag)]
        for f in LEAF_FIELDS:
            if leaf[f.name] != f.default:
                parts.append(f"{tag}.{f.name}={_fmt(f, leaf[f.name])}")
    for tag in combos:
        combo = config.combos[COMBO_TAGS.index(tag)]
        for f in COMBO_FIELDS:
            if combo[f.name] != f.default:
                parts.append(f"{tag}.{f.name}={_fmt(f, combo[f.name])}")
    return ";".join(parts)


def _split_tags(text, allowed):
    return [tag for tag in (t.strip() for t in str(text).split(",")) if tag in allowed]


def parse_own(text):
    """Our infotext value -> ``Config`` (enabled). Unknown keys and bad values fall back to defaults."""
    config = blank_config()
    config.enabled = True
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, raw = chunk.partition("=")
        key, raw = key.strip(), raw.strip()
        if key == "mods":
            for tag in _split_tags(raw, MODIFIER_TAGS):
                config.modifiers[MODIFIER_TAGS.index(tag)]["active"] = True
            continue
        if key == "masking":
            config.masking = _truthy(raw, False)
            continue
        if key in ("masks", "combos"):
            continue
        tag, _, name = key.partition(".")
        if tag in MODIFIER_TAGS and name in MODIFIER_BY_NAME and name != "active":
            config.modifiers[MODIFIER_TAGS.index(tag)][name] = coerce(MODIFIER_BY_NAME[name], raw)
        elif tag in MASK_TAGS and name in LEAF_BY_NAME:
            config.leaves[MASK_TAGS.index(tag)][name] = coerce(LEAF_BY_NAME[name], raw)
        elif tag in COMBO_TAGS and name in COMBO_BY_NAME:
            index = COMBO_TAGS.index(tag)
            choices = combo_ref_choices(index) if name in ("mask_a", "mask_b") else None
            config.combos[index][name] = coerce(COMBO_BY_NAME[name], raw, choices)
    return config


# ---------------------------------------------------------------------------
# Infotext — upstream's Forge script ("Colorcraft: MOD(...)MASK(...)COMBO(...)")
# ---------------------------------------------------------------------------

# origin: scripts/colorcraft.py:67-83 (the order upstream writes the values in)
UPSTREAM_MODIFIER_FIELDS = [
    "active", "mask", "strength", "start", "end", "advanced", "exponent", "bias", "start_off", "end_off", "smooth",
    "exposure", "tone_compression",
    "contrast", "clarity", "sharpness",
    "temperature", "tint",
    "vibrance", "saturation",
    "chroma_contrast", "chroma_center",
    "temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b",
    "color_shift_amount", "color_shift_mode", "color_shift_red", "color_shift_green", "color_shift_blue", "color_shift_brightness",
    "hires",
]
UPSTREAM_MODIFIER_VALUE_FIELDS = [f for f in UPSTREAM_MODIFIER_FIELDS if f != "active"]
UPSTREAM_LEAF_FIELDS = ["mask_axis", "mask_mode", "mask_strength", "mask_width", "mask_center", "mask_hardness", "blur", "spread", "contrast", "normalize"]
UPSTREAM_COMBO_FIELDS = ["mask_a", "mask_b", "operation", "blur", "spread", "contrast", "normalize"]
_UPSTREAM_BOOL = {"advanced", "smooth", "normalize", "hires"}
_UPSTREAM_STR = {"color_shift_mode", "mask_axis", "mask_mode", "mask", "mask_a", "mask_b", "operation"}


def _upstream_cast(name, raw):
    """origin: scripts/colorcraft.py:332-337 (``_cast_field``); a bad number becomes None."""
    if name in _UPSTREAM_BOOL:
        return raw == "True"
    if name in _UPSTREAM_STR:
        return raw
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _upstream_entries(raw, value_fields):
    """origin: scripts/colorcraft.py:348-367 (``parse_entries``) — a length mismatch keeps what lines up."""
    result = {}
    if not raw:
        return result
    for entry in raw.split(";"):
        if ":" not in entry:
            continue
        tag, values_str = entry.split(":", 1)
        values = values_str.split(",")
        result[tag.strip()] = {f: _upstream_cast(f, v) for f, v in zip(value_fields, values)}
    return result


def _upstream_section(raw, label):
    """origin: scripts/colorcraft.py:378-380 (``extract_infotext_section``)."""
    match = re.search(rf"{re.escape(label)}\((.*?)\)", raw)
    return match.group(1) if match else ""


def is_upstream_text(text):
    return isinstance(text, str) and "MOD(" in text


def upstream_dict(text):
    """The dict upstream's own ``on_infotext_pasted`` turns the string into (scripts/colorcraft.py:383-396)."""
    modifiers = _upstream_entries(_upstream_section(text, "MOD"), UPSTREAM_MODIFIER_VALUE_FIELDS)
    for entry in modifiers.values():
        entry["active_flag"] = True
    return {
        "modifiers": modifiers,
        "masks": _upstream_entries(_upstream_section(text, "MASK"), UPSTREAM_LEAF_FIELDS),
        "combos": _upstream_entries(_upstream_section(text, "COMBO"), UPSTREAM_COMBO_FIELDS),
    }


def parse_upstream(data):
    """Upstream's Forge infotext (the string, or the dict its paste callback already made) -> ``Config``.

    Every upstream tab is the Advanced node with Chroma Plus and Color Shift always applied and no dev
    overrides; its "Hires. Pass" box is Pass Hires/Base. Masking is on when masks or combos were written
    (upstream restores its Masking toggle the same way)."""
    if isinstance(data, str):
        data = upstream_dict(data)
    config = blank_config()
    config.enabled = True
    modifiers = data.get("modifiers") or {}
    masks = data.get("masks") or {}
    combos = data.get("combos") or {}
    for tag, values in modifiers.items():
        if tag not in MODIFIER_TAGS:
            continue
        modifier = config.modifiers[MODIFIER_TAGS.index(tag)]
        modifier["active"] = True
        modifier["kind"] = KIND_LABELS["advanced"]
        modifier["more_colors"] = True
        modifier["color_shift"] = True
        for name, value in values.items():
            if value is None or name in ("active_flag", "active"):
                continue
            if name == "hires":
                modifier["pass"] = PASS_HIRES if _truthy(value) else PASS_BASE
            elif name in MODIFIER_BY_NAME:
                modifier[name] = coerce(MODIFIER_BY_NAME[name], value)
    for tag, values in masks.items():
        if tag in MASK_TAGS:
            leaf = config.leaves[MASK_TAGS.index(tag)]
            for name, value in values.items():
                if value is not None and name in LEAF_BY_NAME:
                    leaf[name] = coerce(LEAF_BY_NAME[name], value)
    for tag, values in combos.items():
        if tag in COMBO_TAGS:
            index = COMBO_TAGS.index(tag)
            combo = config.combos[index]
            for name, value in values.items():
                if value is not None and name in COMBO_BY_NAME:
                    choices = combo_ref_choices(index) if name in ("mask_a", "mask_b") else None
                    combo[name] = coerce(COMBO_BY_NAME[name], value, choices)
    config.masking = bool(masks) or bool(combos)
    return config


# ---------------------------------------------------------------------------
# Infotext — the aoleg fork ("Colorcraft: v1;name=value;…")
# ---------------------------------------------------------------------------

# origin: aoleg/ComfyUI-Colorcraft@f00066c:lib_colorcraft/params.py:91-203 — name: (type, default).
FORK_PARAMS = {
    "strength": (FLOAT, 1.0), "start": (FLOAT, 0.5), "end": (FLOAT, 0.75),
    "advanced": (BOOL, False), "bias": (FLOAT, 0.5), "smooth": (BOOL, True), "exponent": (FLOAT, 0.0),
    "start_off": (FLOAT, 0.0), "end_off": (FLOAT, 0.0),
    "exposure": (FLOAT, 0.0), "tone_compression": (FLOAT, 0.0),
    "contrast": (FLOAT, 0.0), "clarity": (FLOAT, 0.0), "sharpness": (FLOAT, 0.0),
    "temperature": (FLOAT, 0.0), "tint": (FLOAT, 0.0), "vibrance": (FLOAT, 0.0), "saturation": (FLOAT, 0.0),
    "chroma_contrast": (FLOAT, 0.0), "chroma_center": (FLOAT, 0.5),
    "more_colors": (BOOL, False), "temp_plus_tint": (FLOAT, 0.0), "temp_minus_tint": (FLOAT, 0.0),
    "lab_a": (FLOAT, 0.0), "lab_b": (FLOAT, 0.0), "lab_a_plus_b": (FLOAT, 0.0), "lab_a_minus_b": (FLOAT, 0.0),
    "color_shift": (BOOL, False), "color_shift_amount": (FLOAT, 0.0),
    "mode": (CHOICE, "default"), "red": (FLOAT, 0.0), "green": (FLOAT, 0.0), "blue": (FLOAT, 0.0),
    "brightness": (FLOAT, 0.0),
    "masking": (BOOL, False),
    "mask_mode": (CHOICE, "highs"), "mask_axis": (CHOICE, "exposure"), "mask_center": (FLOAT, 0.0),
    "mask_hardness": (FLOAT, 1.0), "mask_strength": (FLOAT, 1.0), "mask_width": (FLOAT, 0.0),
    "mask_combine": (BOOL, False), "mask_operation": (CHOICE, "and"),
    "mask_b_mode": (CHOICE, "highs"), "mask_b_axis": (CHOICE, "exposure"), "mask_b_center": (FLOAT, 0.0),
    "mask_b_hardness": (FLOAT, 1.0), "mask_b_strength": (FLOAT, 1.0), "mask_b_width": (FLOAT, 0.0),
    "mask_blur_radius": (FLOAT, 0.0), "mask_spread": (FLOAT, 0.0),
    "mask_preview": (BOOL, False), "mask_preview_color": (CHOICE, "red"),
    "dev": (BOOL, False), "recenter_override": (BOOL, False), "recenter": (FLOAT, 0.5),
    "max_chroma_override": (BOOL, False), "max_chroma": (FLOAT, 2.5),
    "chroma_plane_override": (BOOL, False), "chroma_plane": (CHOICE, "temp_tint"),
    "apply_to_hr": (BOOL, True), "debug": (BOOL, False),
}
_FORK_CHOICES = {
    "mode": COLOR_SHIFT_MODE_OPTIONS, "mask_mode": MASK_MODE_OPTIONS, "mask_b_mode": MASK_MODE_OPTIONS,
    # the fork's MASK_AXIS_OPTIONS (core.py:98-101) has no clarity/sharpness
    "mask_axis": MASK_AXIS_OPTIONS[:11], "mask_b_axis": MASK_AXIS_OPTIONS[:11],
    "mask_operation": COMBO_OP_OPTIONS, "mask_preview_color": ["red", "green", "blue", "white", "black"],
    "chroma_plane": CHROMA_PLANE_OPTIONS,
}
# origin: aoleg/ComfyUI-Colorcraft@f00066c:lib_colorcraft/spec.py:45-51 (PREVIEW_COLORS)
FORK_PREVIEW_COLORS = {
    "red": (0.5, -0.5, -0.5),
    "green": (-0.5, 0.5, -0.5),
    "blue": (-0.5, -0.5, 0.5),
    "white": (0.5, 0.5, 0.5),
    "black": (-0.5, -0.5, -0.5),
}


def is_fork_text(text):
    return isinstance(text, str) and not is_upstream_text(text) and text.strip().startswith("v1")


def fork_values(text):
    """origin: fork spec.py:177-198 (``from_infotext``) — complete dict, the fork's defaults for absent keys."""
    values = {name: default for name, (_, default) in FORK_PARAMS.items()}
    for chunk in str(text or "").split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, _, raw = chunk.partition("=")
        name = name.strip()
        if name not in FORK_PARAMS:
            continue
        kind, default = FORK_PARAMS[name]
        if kind == BOOL:
            parsed = raw.strip() in ("1", "true", "True", "yes")
        elif kind == FLOAT:
            parsed = _finite(raw, default)
        else:
            parsed = raw
            if parsed not in _FORK_CHOICES.get(name, ()):
                parsed = default
        values[name] = parsed
    return values


def parse_fork(text):
    """The fork's infotext -> ``Config``: its single Advanced modifier on tab I, its masks on M1/M2/C1.

    * Shaping off keeps the fork's bias (the ComfyUI node does); here shaping off resets bias, so a
      non-default bias turns shaping on with exponent/offsets 0 — the same schedule.
    * The fork's Chroma Center is a fraction of max chroma (0..1, older upstream math); upstream now
      maps -1..1 onto 0..max chroma, so the same pivot is ``2·c − 1``.
    * The fork's vibrance used the older upstream curve; the value is kept, the current math applies.
    * Mask preview becomes the Shift entry the fork built for it (legacy colour shift to the preview
      colour on the last step, gated by the mask) — upstream dropped that node for its Debug panel.
    * apply_to_hr on (the fork's default) is Pass Both, off is Base."""
    v = fork_values(text)
    config = blank_config()
    config.enabled = True
    modifier = config.modifiers[0]
    modifier["active"] = True
    modifier["pass"] = PASS_BOTH if v["apply_to_hr"] else PASS_BASE
    modifier["kind"] = KIND_LABELS["advanced"]
    for name in ("strength", "start", "end", "smooth", "exposure", "tone_compression", "contrast", "clarity",
                 "sharpness", "temperature", "tint", "vibrance", "saturation", "chroma_contrast",
                 "more_colors", "temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b",
                 "lab_a_minus_b", "color_shift", "color_shift_amount", "recenter_override", "recenter",
                 "max_chroma_override", "max_chroma", "chroma_plane_override", "chroma_plane"):
        modifier[name] = coerce(MODIFIER_BY_NAME[name], v[name])
    modifier["chroma_center"] = coerce(MODIFIER_BY_NAME["chroma_center"], 2.0 * v["chroma_center"] - 1.0)
    for ours, theirs in (("color_shift_mode", "mode"), ("color_shift_red", "red"),
                         ("color_shift_green", "green"), ("color_shift_blue", "blue"),
                         ("color_shift_brightness", "brightness")):
        modifier[ours] = coerce(MODIFIER_BY_NAME[ours], v[theirs])
    if v["advanced"]:
        modifier["advanced"] = True
        for name in ("bias", "exponent", "start_off", "end_off"):
            modifier[name] = coerce(MODIFIER_BY_NAME[name], v[name])
    elif v["bias"] != 0.5:
        modifier["advanced"] = True
        modifier["bias"] = coerce(MODIFIER_BY_NAME["bias"], v["bias"])
        modifier["exponent"], modifier["start_off"], modifier["end_off"] = 0.0, 0.0, 0.0

    if v["masking"]:
        config.masking = True
        leaf_a = config.leaves[0]
        for ours, theirs in (("mask_mode", "mask_mode"), ("mask_axis", "mask_axis"),
                             ("mask_center", "mask_center"), ("mask_hardness", "mask_hardness"),
                             ("mask_strength", "mask_strength"), ("mask_width", "mask_width")):
            leaf_a[ours] = coerce(LEAF_BY_NAME[ours], v[theirs])
        target = leaf_a
        modifier["mask"] = MASK_TAGS[0]
        if v["mask_combine"]:
            leaf_b = config.leaves[1]
            for ours, theirs in (("mask_mode", "mask_b_mode"), ("mask_axis", "mask_b_axis"),
                                 ("mask_center", "mask_b_center"), ("mask_hardness", "mask_b_hardness"),
                                 ("mask_strength", "mask_b_strength"), ("mask_width", "mask_b_width")):
                leaf_b[ours] = coerce(LEAF_BY_NAME[ours], v[theirs])
            combo = config.combos[0]
            combo["mask_a"], combo["mask_b"] = MASK_TAGS[0], MASK_TAGS[1]
            combo["operation"] = coerce(COMBO_BY_NAME["operation"], v["mask_operation"])
            target = combo
            modifier["mask"] = COMBO_TAGS[0]
        if v["mask_blur_radius"] > 0 or v["mask_spread"] != 0:
            target["blur"] = coerce(LEAF_BY_NAME["blur"], v["mask_blur_radius"])
            target["spread"] = coerce(LEAF_BY_NAME["spread"], v["mask_spread"])
        if v["mask_preview"]:
            red, green, blue = FORK_PREVIEW_COLORS[v["mask_preview_color"]]
            modifier.update({
                "kind": KIND_LABELS["shift"], "color_shift_amount": 1.0, "color_shift_mode": "legacy",
                "color_shift_red": red, "color_shift_green": green, "color_shift_blue": blue,
                "color_shift_brightness": 0.0, "strength": 1.0, "start": 1.0, "end": 1.0,
                "advanced": False, "bias": 0.5, "exponent": 0.0, "start_off": 0.0, "end_off": 0.0,
                "smooth": True,
            })
    return config


# ---------------------------------------------------------------------------
# Paste
# ---------------------------------------------------------------------------

_DECODE_CACHE: dict = {}
_DECODE_CACHE_LIMIT = 16


def decode(params):
    """The ``Config`` an infotext's parameters describe, or None when there is no Colorcraft key.

    Our key wins; otherwise upstream's or the fork's ``Colorcraft`` value (a string, or the dict upstream's
    own paste callback turns it into when that extension is installed too). Memoised per value, since
    every one of the panel's fields asks. A value that cannot be read (e.g. a dict of another shape left by
    some other extension's paste callback) is treated like an unknown value: None, remembered, never an
    exception — Forge would otherwise log one traceback per panel field (~580) on every paste."""
    if not isinstance(params, dict):
        return None
    if INFOTEXT_KEY in params:
        raw, reader = params[INFOTEXT_KEY], "own"
    elif FOREIGN_KEY in params:
        raw, reader = params[FOREIGN_KEY], "foreign"
    else:
        return None
    if isinstance(raw, dict):
        cached = _DECODE_CACHE.get(("dict", id(raw)))
        if cached is not None and cached[0] is raw:
            return cached[1]
        try:
            config = parse_upstream(raw)
        except Exception:
            config = None
        _remember(("dict", id(raw)), (raw, config))
        return config
    key = (reader, str(raw))
    cached = _DECODE_CACHE.get(key)
    if cached is not None:
        return cached[1]
    text = str(raw)
    try:
        if reader == "own":
            config = parse_own(text)
        elif is_upstream_text(text):
            config = parse_upstream(text)
        elif is_fork_text(text):
            config = parse_fork(text)
        else:
            config = None
    except Exception:
        config = None
    _remember(key, (raw, config))
    return config


def _remember(key, value):
    if len(_DECODE_CACHE) >= _DECODE_CACHE_LIMIT:
        _DECODE_CACHE.pop(next(iter(_DECODE_CACHE)))
    _DECODE_CACHE[key] = value


def paste_value(params, path):
    """The value of one script argument (``arg_names()`` path) for a pasted infotext, None = leave it."""
    config = decode(params)
    if path == "enabled":
        return config is not None and bool(config.enabled)
    if config is None:
        return None
    if path == "masking":
        return bool(config.masking)
    if path in ("debug", "debug_step"):
        return None                      # a tool, not part of the look: pasting leaves it alone
    tag, _, name = path.partition(".")
    if tag in MODIFIER_TAGS:
        return config.modifiers[MODIFIER_TAGS.index(tag)][name]
    if tag in MASK_TAGS:
        return config.leaves[MASK_TAGS.index(tag)][name]
    if tag in COMBO_TAGS:
        return config.combos[COMBO_TAGS.index(tag)][name]
    return None


__all__ = [
    "ARG_COUNT", "COMBO_COUNT", "COMBO_FIELDS", "COMBO_TAGS", "Config", "DEBUG_FIELDS", "FOREIGN_KEY",
    "GLOBAL_FIELDS",
    "INFOTEXT_KEY", "KIND_CHOICES", "KIND_GATES", "KIND_GROUPS", "KIND_IDS", "KIND_LABELS", "LEAF_FIELDS",
    "MASK_COUNT", "MASK_TAGS", "MODIFIER_COUNT", "MODIFIER_FIELDS", "MODIFIER_TAGS",
    "PASS_CHOICES", "PassChain", "STATUS_KEY", "XYZ_ATTR", "XYZ_MODIFIER_FIELDS", "apply_xyz", "arg_names",
    "args_enabled", "build_chain", "build_mask_specs", "config_from_args", "config_from_mapping",
    "config_to_args", "decode", "default_config", "is_compact_arg", "parse_fork", "parse_own", "parse_upstream",
    "paste_value", "to_infotext",
]
