# Colorcraft (sam-extra) — the shared editor's script arguments: hidden state, editor overlay, paste, labels.
#
# New sam-extra code (2026-10-03) with no upstream lines: it only serialises spec.py's field tables (whose
# ranges, defaults and stack shape come from muerrilla/ComfyUI-Colorcraft, see spec.py) into the panel's
# hidden state, the editor overlay, the paste values, the selector labels and javascript/colorcraft_schema.js.
"""The shared-editor panel's 67 script arguments <-> ``spec.Config`` (no Gradio here).

Arguments (``ARG_NAMES``, per tab)::

    0  enabled     Checkbox (sam3-on). Also the API's compact slot: infotext string, dict, or v0.31.0's list
    1  masking     Checkbox
    2  state       hidden Textbox: "" (= every default) or JSON {"v": 1, "rev": "<id>", "<path>": value, ...}
    3  debug       Checkbox (Debug capture; a tool: paste leaves it alone)
    4  debug_step  Slider
    5  ref         hidden Textbox "<modifier tag>|<mask tag>|<rev>": which nodes the editors show
    6..49          modifier editor, MODIFIER_FIELDS order
    50..59         leaf editor, LEAF_FIELDS order
    60..66         combo editor, COMBO_FIELDS order

A generation reads the state, then lays the editors' live values over the modifier and mask that ``ref`` names
-- only when ``ref``'s rev equals the state's rev. The browser writes the state only when the editors switch
node or a Reset runs (commit + load in one js-only update), so the visible node always reaches the generation
through its own controls, exactly like v0.31.0's per-control arguments.
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from . import spec

STATE_VERSION = 1
INITIAL_REV = "0"
DEFAULT_STATE = ""
DEFAULT_REF = f"{spec.MODIFIER_TAGS[0]}|{spec.MASK_TAGS[0]}|{INITIAL_REV}"

MOD_NAMES = tuple(f.name for f in spec.MODIFIER_FIELDS)
LEAF_NAMES = tuple(f.name for f in spec.LEAF_FIELDS)
COMBO_NAMES = tuple(f.name for f in spec.COMBO_FIELDS)
EDITOR_NAMES = (tuple(f"mod.{n}" for n in MOD_NAMES) + tuple(f"leaf.{n}" for n in LEAF_NAMES)
                + tuple(f"combo.{n}" for n in COMBO_NAMES))
HEAD_NAMES = ("enabled", "masking", "state", "debug", "debug_step", "ref")
ARG_NAMES = HEAD_NAMES + EDITOR_NAMES
ARG_COUNT = len(ARG_NAMES)                       # 67
STATE_INDEX, REF_INDEX, EDITOR_OFFSET = 2, 5, len(HEAD_NAMES)
EDITOR_COUNT = len(EDITOR_NAMES)                 # 61

NODE_TAGS = tuple(spec.MODIFIER_TAGS + spec.MASK_TAGS + spec.COMBO_TAGS)
_GLOBAL_PATHS = {f.name for f in spec.GLOBAL_FIELDS + spec.DEBUG_FIELDS}
STATE_PATHS = tuple(p for p in spec.arg_names() if p not in _GLOBAL_PATHS)     # 575, canonical order
_STATE_PATH_SET = frozenset(STATE_PATHS)

# v0.31.0's reader: the compact first argument, the wrapped list and direct 579-value callers.
_V031_CONFIG_FROM_ARGS = spec.config_from_args
_V031_ARGS_ENABLED = spec.args_enabled


class RefusedArgs(ValueError):
    """Arguments this reader will not guess at; the hook writes a ``not applied: …`` status instead."""


class LegacyPositionalArgs(RefusedArgs):
    """v0.31.0's 579 positional values sent through Forge's API: Forge kept only the first 67."""


class UnreadableState(RefusedArgs):
    """The state slot holds something that is not a state."""


LEGACY_NOTE = ("v0.31.0 positional arguments (579 values) are no longer read - send the infotext string or a "
               "dict of argument paths as the first argument, or the old list wrapped in one list")


def new_rev() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def node_fields(tag):
    if tag in spec.MODIFIER_TAGS:
        return spec.MODIFIER_FIELDS
    if tag in spec.MASK_TAGS:
        return spec.LEAF_FIELDS
    return spec.COMBO_FIELDS


def node_values(config, tag):
    if tag in spec.MODIFIER_TAGS:
        return config.modifiers[spec.MODIFIER_TAGS.index(tag)]
    if tag in spec.MASK_TAGS:
        return config.leaves[spec.MASK_TAGS.index(tag)]
    return config.combos[spec.COMBO_TAGS.index(tag)]


def node_default(tag, f):
    if tag in spec.MODIFIER_TAGS:
        return spec.modifier_default(spec.MODIFIER_TAGS.index(tag), f.name)
    return f.default


def _same(a, b):
    # repr + type: -0.0 stays apart from 0.0 and True from 1.0, so a value comes back bit for bit
    return type(a) is type(b) and repr(a) == repr(b)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def dump_state(config, rev) -> str:
    """The state for ``config``: every modifier/leaf/combo value that differs from its default, plus v and rev.

    Enable, masking and the Debug values are arguments of their own and are not stored."""
    out = {"v": STATE_VERSION, "rev": str(rev)}
    for tag in NODE_TAGS:
        values = node_values(config, tag)
        for f in node_fields(tag):
            value = values[f.name]
            if isinstance(value, float) and not math.isfinite(value):
                continue                                  # coerce() reads a missing value as the default
            if not _same(value, node_default(tag, f)):
                out[f"{tag}.{f.name}"] = value
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def is_state_value(value) -> bool:
    return value is None or value == "" or isinstance(value, dict) or (
        isinstance(value, str) and value.lstrip().startswith("{"))


def _rev(value):
    """A writer's rev (``new_rev`` / the browser's: a non-empty string without ``|``), else None. The initial rev
    belongs to the empty state only: a non-empty state carrying "0" is an outside one (Forge fills ``ref`` with
    "I|M1|0", which must not lay modifier I's and M1's default editors over it). colorcraft_editor.js reads the
    same way."""
    if not isinstance(value, str) or not value or "|" in value or value == INITIAL_REV:
        return None
    return value


def parse_state(raw):
    """``(paths, rev)``; "" / None is the default state with rev "0". ``rev`` is None when the state has none
    (or not a writer's, see ``_rev``): such a state is read on its own, never overlaid by the editors."""
    if raw is None or raw == "":
        return {}, INITIAL_REV
    if isinstance(raw, dict):
        data = dict(raw)
    elif isinstance(raw, str) and raw.lstrip().startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise UnreadableState(f"state is not valid JSON ({exc})") from None
        if not isinstance(data, dict):
            raise UnreadableState("state is not a JSON object")
    else:
        raise UnreadableState(f"state slot holds {type(raw).__name__}")
    version = data.pop("v", STATE_VERSION)
    if isinstance(version, bool) or version != STATE_VERSION:       # JSON true is not 1 (as in the browser)
        raise UnreadableState(f"state version {version!r} is not {STATE_VERSION}")
    rev = _rev(data.pop("rev", None))
    paths = {k: v for k, v in data.items() if k in _STATE_PATH_SET}
    return paths, rev


def config_from_state(enabled, masking, state, debug=None, debug_step=None):
    """The ``Config`` the state describes; the four head values coerced exactly like ``spec.config_from_args``."""
    paths, _ = parse_state(state)
    config = spec.config_from_mapping(paths)
    config.enabled = spec._truthy(enabled, False)
    config.masking = spec._truthy(masking, False)
    config.debug = spec.coerce(spec.DEBUG_FIELDS[0], spec.DEBUG_FIELDS[0].default if debug is None else debug)
    config.debug_step = spec.coerce(spec.DEBUG_FIELDS[1],
                                    spec.DEBUG_FIELDS[1].default if debug_step is None else debug_step)
    return config


# ---------------------------------------------------------------------------
# Editors
# ---------------------------------------------------------------------------


def parse_ref(ref):
    """``(modifier tag, mask tag, rev)`` or None when ``ref`` does not name two nodes and a rev."""
    parts = str(ref or "").split("|")
    if len(parts) != 3:
        return None
    mod, mask, rev = parts
    if mod not in spec.MODIFIER_TAGS or mask not in spec.MASK_TAGS + spec.COMBO_TAGS or not rev:
        return None
    return mod, mask, rev


def make_ref(mod, mask, rev) -> str:
    return f"{mod}|{mask}|{rev}"


def split_editors(values):
    values = list(values)
    a, b = len(MOD_NAMES), len(MOD_NAMES) + len(LEAF_NAMES)
    return values[:a], values[a:b], values[b:b + len(COMBO_NAMES)]


def apply_editors(config, mod, mask, values):
    """Lay the editors' values over modifier ``mod`` and mask node ``mask`` (coerced like positional args)."""
    mod_vals, leaf_vals, combo_vals = split_editors(values)
    index = spec.MODIFIER_TAGS.index(mod)
    modifier = config.modifiers[index]
    for f, value in zip(spec.MODIFIER_FIELDS, mod_vals):
        default = spec.modifier_default(index, f.name)
        modifier[f.name] = default if value is None else spec.coerce(f, value)
    if mask in spec.MASK_TAGS:
        leaf = config.leaves[spec.MASK_TAGS.index(mask)]
        for f, value in zip(spec.LEAF_FIELDS, leaf_vals):
            leaf[f.name] = spec.coerce(f, value)
    else:
        ci = spec.COMBO_TAGS.index(mask)
        combo = config.combos[ci]
        for f, value in zip(spec.COMBO_FIELDS, combo_vals):
            choices = spec.combo_ref_choices(ci) if f.name in ("mask_a", "mask_b") else None
            combo[f.name] = spec.coerce(f, value, choices)
    return config


def editor_values(config, mod, mask, previous=None):
    """The 61 editor values that show ``mod`` and ``mask``; the mask editor not in use keeps ``previous``."""
    prev_mod, prev_leaf, prev_combo = split_editors(previous) if previous is not None else (
        None, [f.default for f in spec.LEAF_FIELDS], [f.default for f in spec.COMBO_FIELDS])
    modifier = node_values(config, mod)
    out = [modifier[n] for n in MOD_NAMES]
    if mask in spec.MASK_TAGS:
        leaf = node_values(config, mask)
        return out + [leaf[n] for n in LEAF_NAMES] + list(prev_combo)
    combo = node_values(config, mask)
    return out + list(prev_leaf) + [combo[n] for n in COMBO_NAMES]


# ---------------------------------------------------------------------------
# What a generation reads
# ---------------------------------------------------------------------------


def args_enabled(args):
    """``spec.args_enabled`` plus the wrapped v0.31.0 list. Reads nothing past the first slot."""
    if args and isinstance(args[0], (list, tuple)):
        return _V031_ARGS_ENABLED(list(args[0]))
    return _V031_ARGS_ENABLED(args)


def _lower(values):
    return frozenset(str(v).lower() for v in values)


# v0.31.0's slots 3-5 are I.kind, I.pass and I.mask, spelled as its coerce() accepts them (any case; Type also by
# node id). The panel's slots 3-5 are debug (bool), debug_step (number) and ref ("<mod>|<mask>|<rev>"): never one.
_V031_HEAD_CHOICES = ((3, _lower(spec.KIND_CHOICES) | _lower(spec.KIND_IDS.values())),
                      (4, _lower(spec.PASS_CHOICES)), (5, _lower(spec.MASK_CHOICES)))


def _v031_positional(args):
    """Does this look like v0.31.0's positional list (cut to 67 by Forge)? Slot 2 (I.active) alone cannot tell:
    v0.31.0 read None there as the default, and None is also the panel's empty state."""
    state = args[STATE_INDEX] if STATE_INDEX < len(args) else None
    if isinstance(state, (bool, int, float)):
        return True
    return any(i < len(args) and isinstance(args[i], str) and args[i].strip().lower() in choices
               for i, choices in _V031_HEAD_CHOICES)


def config_from_script_args(args):
    """``hook.process``'s reader: compact first argument, the wrapped old list, or the 67-argument panel layout."""
    args = list(args or ())
    if args and spec.is_compact_arg(args[0]):
        return _V031_CONFIG_FROM_ARGS(args)
    if args and isinstance(args[0], (list, tuple)):
        return _V031_CONFIG_FROM_ARGS(list(args[0]))
    if len(args) == spec.ARG_COUNT:
        # A Python caller handing v0.31.0's 579 values straight to the hook (tests, scripts). Forge never
        # passes 579 values to this script: it slices exactly ARG_COUNT (67) out of the request.
        return _V031_CONFIG_FROM_ARGS(args)

    def arg(i, default=None):
        return args[i] if i < len(args) else default

    state = arg(STATE_INDEX)
    if _v031_positional(args):
        raise LegacyPositionalArgs(LEGACY_NOTE)
    if not is_state_value(state):
        raise UnreadableState(f"state slot holds {type(state).__name__}")
    config = config_from_state(arg(0, False), arg(1, False), state, arg(3), arg(4))
    if len(args) >= ARG_COUNT:
        ref = parse_ref(arg(REF_INDEX))
        _, rev = parse_state(state)
        if ref is not None and rev is not None and ref[2] == rev:
            apply_editors(config, ref[0], ref[1], args[EDITOR_OFFSET:EDITOR_OFFSET + EDITOR_COUNT])
    return config


def args_from_config(config, mod=None, mask=None, rev="r0"):
    """What the panel sends for ``config`` with the editors on ``mod``/``mask`` (tests, the API docs)."""
    mod = mod or spec.MODIFIER_TAGS[0]
    mask = mask or spec.MASK_TAGS[0]
    return ([bool(config.enabled), bool(config.masking), dump_state(config, rev), bool(config.debug),
             config.debug_step, make_ref(mod, mask, rev)] + editor_values(config, mod, mask))


def default_args():
    """``init_default_script_args``: the ``value`` of every component ``ui()`` returns."""
    config = spec.default_config()
    return ([False, False, DEFAULT_STATE, spec.DEBUG_FIELDS[0].default, spec.DEBUG_FIELDS[1].default, DEFAULT_REF]
            + editor_values(config, spec.MODIFIER_TAGS[0], spec.MASK_TAGS[0]))


def mask_values(config):
    """``debug_panel.render_for_panel``'s ``mask_values``: every leaf then every combo value, v0.31.0 order."""
    return spec.config_to_args(config)[spec.LEAF_OFFSET:spec.DEBUG_OFFSET]


# ---------------------------------------------------------------------------
# Paste (Forge infotext_fields; also the API's ``infotext`` field)
# ---------------------------------------------------------------------------

_PASTE_CACHE: dict = {}


def paste_target(config):
    """Where the editors land after a paste: the first Active modifier (else I) and the first leaf in use (else M1).

    Never a combo: a combo's Mask A/B choices depend on the combo, and a script argument's paste value must be a
    plain value (the API ``infotext`` path turns anything else into its ``str()``)."""
    mod = next((spec.MODIFIER_TAGS[i] for i, m in enumerate(config.modifiers) if m["active"]), spec.MODIFIER_TAGS[0])
    leaves, _ = spec.active_masks(config)
    return mod, (list(leaves) + [spec.MASK_TAGS[0]])[0]


def pasted(params):
    """``(config, state, mod, mask, rev)`` for an infotext with a Colorcraft key, else None.

    Memoised per params dict: every paste field asks, and the rev must be one per paste."""
    config = spec.decode(params)
    if config is None:
        return None
    hit = _PASTE_CACHE.get(id(params))
    if hit is not None and hit[0] is params:
        return hit[1]
    mod, mask = paste_target(config)
    rev = new_rev()
    result = (config, dump_state(config, rev), mod, mask, rev)
    _PASTE_CACHE.clear()
    _PASTE_CACHE[id(params)] = (params, result)
    return result


def paste_enabled(params):
    return spec.paste_value(params, "enabled")


def paste_masking(params):
    return spec.paste_value(params, "masking")


def paste_state(params):
    hit = pasted(params)
    return None if hit is None else hit[1]


def paste_ref(params):
    hit = pasted(params)
    return None if hit is None else make_ref(hit[2], hit[3], hit[4])


def paste_editor(params, name):
    """``name`` = "mod.<field>" or "leaf.<field>" (combo editors are never pasted: see ``paste_target``)."""
    hit = pasted(params)
    if hit is None:
        return None
    config, _, mod, mask, _ = hit
    kind, _, field = name.partition(".")
    return node_values(config, mod if kind == "mod" else mask)[field]


# ---------------------------------------------------------------------------
# At a glance (first paint and paste; colorcraft_editor.js renders the same text in the browser)
# ---------------------------------------------------------------------------

MARK_ON, MARK_EDITED = "●", "○"
INCOMPLETE = "(미완성)"          # after "➜ C1" in the summary: the combo does not resolve, the modifier runs unmasked


def _edited(config, tag):
    values = node_values(config, tag)
    return any(values[f.name] != node_default(tag, f) for f in node_fields(tag) if f.name != "active")


def _escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#x27;"))


def _used_masks(config):
    """The leaves and combos the active modifiers really mask with: ``spec.active_masks`` without the combos
    ``spec.build_mask_specs`` cannot resolve (A or B empty, or naming an incomplete combo) -- a modifier that
    picks one runs unmasked. ``(used tags, unresolved combo tags)``."""
    _, combo_specs = spec.build_mask_specs(config)
    unresolved = {tag for tag, s in combo_specs.items() if s is None}
    used = set()
    if not config.masking:
        return used, unresolved
    pending = [m["mask"] for m in config.modifiers
               if m["active"] and spec.KIND_TAKES_MASK[spec.kind_of(m)] and m["mask"] != spec.MASK_NONE]
    while pending:
        tag = pending.pop()
        if tag == spec.MASK_NONE or tag in used or tag in unresolved:
            continue
        used.add(tag)
        if tag in spec.COMBO_TAGS:
            combo = config.combos[spec.COMBO_TAGS.index(tag)]
            pending += [combo["mask_a"], combo["mask_b"]]
    return used, unresolved


def overview(config):
    """``(modifier choices, mask choices, summary html)``; choices are (label, value) pairs."""
    used, unresolved = _used_masks(config)
    mod_choices, active = [], []
    for i, tag in enumerate(spec.MODIFIER_TAGS):
        m = config.modifiers[i]
        if m["active"]:
            mark = MARK_ON
            text = f"{tag} {m['kind']}"
            if config.masking and spec.KIND_TAKES_MASK[spec.kind_of(m)] and m["mask"] != spec.MASK_NONE:
                text += f" ➜ {m['mask']}" + (INCOMPLETE if m["mask"] in unresolved else "")
            if m["pass"] != spec.PASS_BASE:
                text += f" ({m['pass']})"
            active.append(text)
        else:
            mark = MARK_EDITED if _edited(config, tag) else ""
        mod_choices.append((f"{tag} {mark}" if mark else tag, tag))
    mask_choices = []
    for tag in spec.MASK_TAGS + spec.COMBO_TAGS:
        mark = MARK_ON if tag in used else (MARK_EDITED if _edited(config, tag) else "")
        mask_choices.append((f"{tag} {mark}" if mark else tag, tag))
    text = ("켠 수정자: " + ", ".join(active)) if active else "켠 수정자 없음"
    if config.masking:
        text += " · 쓰는 마스크: " + (", ".join(t for t in NODE_TAGS if t in used) if used else "없음")
    if not config.enabled:
        text = "꺼짐 — " + text
    return mod_choices, mask_choices, f'<div class="samextra-cc-summary">{_escape(text)}</div>'


# ---------------------------------------------------------------------------
# Schema for javascript/colorcraft_editor.js (generated into javascript/colorcraft_schema.js)
# ---------------------------------------------------------------------------

GROUP_ORDER = ("luma", "contrast", "detail", "chroma", "chroma_plus", "color_shift", "dev")


def _field_json(f):
    return {"name": f.name, "default": f.default}


def schema():
    return {
        "version": STATE_VERSION,
        "initialRev": INITIAL_REV,
        "modifierTags": list(spec.MODIFIER_TAGS),
        "maskTags": list(spec.MASK_TAGS),
        "comboTags": list(spec.COMBO_TAGS),
        "maskNone": spec.MASK_NONE,
        "passBase": spec.PASS_BASE,
        "modifierFields": [_field_json(f) for f in spec.MODIFIER_FIELDS],
        "activeDefaults": [spec.modifier_default(i, "active") for i in range(spec.MODIFIER_COUNT)],
        "leafFields": [_field_json(f) for f in spec.LEAF_FIELDS],
        "comboFields": [_field_json(f) for f in spec.COMBO_FIELDS],
        "comboRefChoices": [spec.combo_ref_choices(i) for i in range(spec.COMBO_COUNT)],
        "groupOrder": list(GROUP_ORDER),
        "kindIds": dict(spec.KIND_IDS),
        "kindGroups": {k: list(v) for k, v in spec.KIND_GROUPS.items()},
        "kindGates": {k: list(v) for k, v in spec.KIND_GATES.items()},
        "kindTakesMask": dict(spec.KIND_TAKES_MASK),
    }


def schema_js() -> str:
    body = json.dumps(schema(), ensure_ascii=False, separators=(",", ":"))
    return ("// Generated from sam3ext/colorcraft/spec.py by sam3ext/colorcraft/panel_state.py — do not edit "
            "(tests/test_colorcraft_panel_state.py compares it).\n"
            f"window.samextraColorcraftSchema = {body};\n")


_SCHEMA_JS_PATH = Path(__file__).resolve().parents[2] / "javascript" / "colorcraft_schema.js"


def write_schema_js(path=None):
    """Write ``schema_js()`` as UTF-8 bytes with LF (no BOM); by default to ``javascript/colorcraft_schema.js``.

    From the extension root::

        venv\\Scripts\\python.exe -c "from sam3ext.colorcraft import panel_state; panel_state.write_schema_js()"
    """
    target = Path(path) if path is not None else _SCHEMA_JS_PATH
    target.write_bytes(schema_js().encode("utf-8"))
    return target
