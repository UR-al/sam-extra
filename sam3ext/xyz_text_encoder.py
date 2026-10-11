"""XYZ axis ``[Modules] Text Encoder`` — pure logic (no Forge, no torch, no Gradio).

Forge Neo (2.30) keeps VAEs and text encoders in ONE list: the "VAE / Text Encoder" dropdown, saved as
``opts.forge_additional_modules`` (full paths). It tells them apart only while loading the model —
``backend/loader.py`` ``replace_state_dict`` looks at each file's keys and shapes and writes a text encoder into one
slot of the checkpoint (``text_encoders.<slot>.``); the family's ``clip_target`` then reads the slot(s) it uses
(Anima ``qwen3_06b``, Krea2 ``qwen3vl_4b``, Flux ``clip_l`` + ``t5xxl`` …). A file that matches no branch is ignored.

The stock XYZ ``VAE`` axis hot-swaps VAE weights; there is no text-encoder axis. This module decides, from the
safetensors HEADER only (no tensor is read), which slot a file fills - ``te_slots`` mirrors replace_state_dict's
branches in their order (``tests/test_xyz_text_encoder_origin.py`` runs Forge's own function on the same headers) - and
builds one cell's module list: the current list with the entry of the picked file's slot replaced (``compose``). The
WebUI seam (``scripts/modules_xyz_text_encoder.py``) hands that list to Forge through ``p.override_settings``, the
path the API and Forge's built-in PiD script use, so Forge reloads the model exactly as if the file had been chosen in
the dropdown, writes the real list into the image's ``Module N`` infotext, and puts the user's selection back after
every cell (``process_images``' ``finally``) - config.json and the dropdown are never touched.

Values: ``Use Forge current`` (no change), a module file (basename, the dropdown's relative path, or the name without
extension), or ``a + b`` (text mode) for families with two encoders (Flux CLIP-L + T5).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .anima38.files import ARCHITECTURE as _ANIMA38_ADAPTER, QWEN35_MARKERS, V2_ARCHITECTURE as _ANIMA38_V2
# The stdlib-only copy of the header reader (sam3ext/anima_core.py has the same function but imports torch).
from .vae_degrid_models import read_safetensors_header

AXIS_LABEL = "[Modules] Text Encoder"
LABEL = "Text Encoder"
CURRENT = "Use Forge current"
# Checkpoint (1.0) stays the outer loop, then this axis, then the stock VAE axis (0.7): a text-encoder change reloads
# the model like a checkpoint change does.
COST = 0.9
# Set on the cell's copy of ``p`` by the axis (``{"value": …, "modules": [...]}``) and read by the hook's before_process.
CELL_ATTR = "_sam_extra_xyz_text_encoder"
MODULES_KEY = "forge_additional_modules"
CHECKPOINT_KEY = "sd_model_checkpoint"
VAE_KEY = "sd_vae"
SAFETENSORS_EXTENSIONS = (".safetensors", ".sft")

SLOT_NAMES = {
    "clip_l": "CLIP-L",
    "clip_g": "CLIP-G",
    "t5xxl": "T5-XXL",
    "umt5xxl": "UMT5-XXL",
    "gemma2_2b": "Gemma 2 2B",
    "qwen3vl_4b": "Qwen3-VL 4B",
    "qwen3vl_8b": "Qwen3-VL 8B",
    "qwen25_7b": "Qwen2.5-VL 7B",
    "qwen3_06b": "Qwen3 0.6B",
    "qwen3_4b": "Qwen3 4B",
    "qwen3_8b": "Qwen3 8B",
    "ministral3_3b": "Ministral 3 3B",
}


class AxisValueError(ValueError):
    """A value the axis cannot use; raised from ``confirm`` before the first cell (like the stock "Unknown Checkpoint")."""


class ForgeLoadError(Exception):
    """Forge's replace_state_dict would raise on this file (a failed ``assert`` or a missing key it reads)."""


# ---------------------------------------------------------------------------------------------------------------------
# Which slot Forge puts a file in — backend/loader.py replace_state_dict (Forge Neo 2.30), branch for branch
# ---------------------------------------------------------------------------------------------------------------------

# CLIP-G / CLIP-L: Forge tries these keys and accepts the first whose shape[0] is 1280 / 768. (Where the weights then
# go depends on the checkpoint — nowhere for a family without CLIP — but the file is a CLIP-G / CLIP-L encoder.)
CLIP_G_KEYS = (
    "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias",
    "text_encoders.clip_g.transformer.text_model.encoder.layers.0.layer_norm1.bias",
    "text_model.encoder.layers.0.layer_norm1.bias",
    "transformer.resblocks.0.ln_1.bias",
)
CLIP_L_KEYS = (
    "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias",
    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias",
    "text_encoders.clip_l.transformer.text_model.encoder.layers.0.layer_norm1.bias",
    "text_model.encoder.layers.0.layer_norm1.bias",
    "transformer.resblocks.0.ln_1.bias",
)
T5_KEY = "encoder.block.0.layer.0.SelfAttention.k.weight"
T5_QWEIGHT_KEY = "encoder.block.0.layer.0.SelfAttention.k.qweight"
T5_SHARED_KEY = "shared.weight"
UMT5_VOCAB = 256384
GEMMA2_KEY = "model.layers.0.post_feedforward_layernorm.weight"
Q_NORM_KEY = "model.layers.0.self_attn.q_norm.weight"
QWEN3VL_KEY = "model.visual.deepstack_merger_list.0.norm.weight"
QWEN3VL_MERGER_KEY = "model.visual.merger.linear_fc2.weight"
QWEN25_KEY = "model.layers.0.self_attn.k_proj.bias"
POST_ATTN_KEY = "model.layers.0.post_attention_layernorm.weight"
QWEN25_VISUAL_KEY = "visual.blocks.0.attn.proj.weight"
# replace_state_dict's VAE branches (flux 2, diffusers, sd/sdxl/wan, "format 0") - only to say "this is a VAE".
VAE_KEYS = (
    "decoder.post_quant_conv.weight",
    "decoder.up_blocks.0.resnets.0.norm1.weight",
    "decoder.conv_in.weight",
    "decoder.middle.0.residual.0.gamma",
    "first_stage_model.decoder.conv_in.weight",
)


def _dim0(header: dict, key: str):
    entry = header.get(key)
    shape = entry.get("shape") if isinstance(entry, dict) else None
    return shape[0] if isinstance(shape, (list, tuple)) and shape else None


def te_slots(header: dict) -> frozenset:
    """The text-encoder slots replace_state_dict would write this file into (empty: none — Forge ignores it).

    ``header`` is a safetensors header (``{key: {"dtype", "shape", "data_offsets"}, "__metadata__": …}``). The order and
    the ``if`` / ``elif`` structure are Forge's: a Qwen3-VL file that also has the Qwen3 keys is Qwen3-VL. Raises
    ``ForgeLoadError`` where Forge itself would fail (its asserts, or a key it reads that is missing)."""
    slots = set()
    if any(key in header and _dim0(header, key) == 1280 for key in CLIP_G_KEYS):
        slots.add("clip_g")
    if any(key in header and _dim0(header, key) == 768 for key in CLIP_L_KEYS):
        slots.add("clip_l")

    if T5_KEY in header:
        if T5_SHARED_KEY not in header:
            raise ForgeLoadError(f"a T5 encoder without {T5_SHARED_KEY}")
        slots.add("umt5xxl" if _dim0(header, T5_SHARED_KEY) == UMT5_VOCAB else "t5xxl")
    elif T5_QWEIGHT_KEY in header:
        slots.add("t5xxl")

    if GEMMA2_KEY in header:
        if Q_NORM_KEY in header:
            raise ForgeLoadError("Gemma-2 keys together with q_norm")
        slots.add("gemma2_2b")
    elif QWEN3VL_KEY in header:
        if QWEN3VL_MERGER_KEY not in header:
            raise ForgeLoadError(f"a Qwen3-VL encoder without {QWEN3VL_MERGER_KEY}")
        slots.add("qwen3vl_4b" if _dim0(header, QWEN3VL_MERGER_KEY) == 2560 else "qwen3vl_8b")
    elif QWEN25_KEY in header:
        if _dim0(header, QWEN25_KEY) != 512:
            raise ForgeLoadError("a Qwen2.5 k_proj bias that is not 512 wide")
        slots.add("qwen25_7b")
    elif POST_ATTN_KEY in header and Q_NORM_KEY in header:
        width = _dim0(header, POST_ATTN_KEY)
        slots.add("qwen3_06b" if width == 1024 else ("qwen3_4b" if width == 2560 else "qwen3_8b"))
    elif POST_ATTN_KEY in header:
        if _dim0(header, POST_ATTN_KEY) != 3072:
            raise ForgeLoadError("a Ministral-3 layer norm that is not 3072 wide")
        slots.add("ministral3_3b")

    if QWEN25_VISUAL_KEY in header:
        slots.add("qwen25_7b")
    return frozenset(slots)


# ---------------------------------------------------------------------------------------------------------------------
# What a module file is
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Kind:
    """``kind``: ``te`` (one slot - usable), ``multi`` (several slots), ``vae``, ``qwen35`` (Anima 3.8B's Qwen3.5,
    which Forge ignores), ``adapter`` (Anima 3.8B adapter), ``invalid`` (Forge would fail), ``unknown`` (no branch
    matches), ``unreadable`` (no safetensors header), ``opaque`` (GGUF / pickle: not read without loading it;
    ``in_te_dir`` says whether it sits in a text-encoder folder)."""

    kind: str
    slots: tuple = ()
    detail: str = ""
    in_te_dir: bool = False

    @property
    def slot(self):
        return self.slots[0] if self.kind == "te" else None

    @property
    def is_te_entry(self) -> bool:
        """Counts as a text encoder of the current selection (an opaque file only in a text-encoder folder)."""
        return self.kind in ("te", "multi") or (self.kind == "opaque" and self.in_te_dir)


def is_qwen35_header(name: str, header: dict) -> bool:
    """Anima 3.8B's Qwen3.5-4B file (sam3ext/anima38/files.py ``is_unused_qwen35_module``): the marker in the name and
    no ``model.`` key - replace_state_dict matches nothing and sam-extra's loader filter drops it from the list."""
    lowered = name.lower()
    if not any(marker in lowered for marker in QWEN35_MARKERS):
        return False
    return not any(isinstance(key, str) and key.startswith("model.") for key in header if key != "__metadata__")


def classify_header(name: str, header) -> Kind:
    if not isinstance(header, dict):
        return Kind("unreadable")
    metadata = header.get("__metadata__")
    if isinstance(metadata, dict) and metadata.get("architecture") in (_ANIMA38_ADAPTER, _ANIMA38_V2):
        return Kind("adapter")
    if is_qwen35_header(name, header):
        return Kind("qwen35")
    try:
        slots = te_slots(header)
    except ForgeLoadError as exc:
        return Kind("invalid", detail=str(exc))
    if len(slots) == 1:
        return Kind("te", tuple(slots))
    if slots:
        return Kind("multi", tuple(sorted(slots)))
    if any(key in header for key in VAE_KEYS):
        return Kind("vae")
    return Kind("unknown")


_CACHE: dict = {}


def _under(path: str, roots) -> bool:
    try:
        target = os.path.normcase(os.path.abspath(path))
        for root in roots or ():
            base = os.path.normcase(os.path.abspath(root))
            if os.path.commonpath([target, base]) == base:
                return True
    except (ValueError, TypeError, OSError):
        pass
    return False


def classify_file(path: str, *, te_roots=()) -> Kind:
    """``Kind`` of a module file; safetensors headers are cached by (path, size, mtime_ns)."""
    path = str(path)
    if not path.lower().endswith(SAFETENSORS_EXTENSIONS):
        return Kind("opaque", detail=os.path.splitext(path)[1].lower(), in_te_dir=_under(path, te_roots))
    try:
        stat = os.stat(path)
    except OSError:
        return Kind("unreadable")
    key = (os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns)
    kind = _CACHE.get(key)
    if kind is None:
        kind = classify_header(os.path.basename(path), read_safetensors_header(path))
        _CACHE[key] = kind
    return kind


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------------------------------------------------

def _basename(entry) -> str:
    return re.split(r"[\\/]", str(entry))[-1]


def stem(name) -> str:
    """The name Forge writes into ``Module N`` (file name without folder and extension)."""
    return os.path.splitext(_basename(name))[0]


def slot_name(slot) -> str:
    return SLOT_NAMES.get(slot, str(slot))


def is_current(value) -> bool:
    return str(value).strip().casefold() in (CURRENT.casefold(), "current")


def resolve(value, module_list) -> str:
    """The ``module_list`` key (a file name) for one value: the name, the dropdown's relative path (or a full path) —
    Forge itself matches modules by file name —, the same ignoring case, or the name without extension."""
    text = str(value).strip().strip('"').strip()
    if not text:
        raise AxisValueError(f'{AXIS_LABEL}: empty value')
    if text in module_list:
        return text
    base = _basename(text)
    if base in module_list:
        return base
    lowered = {name.casefold(): name for name in module_list}
    if base.casefold() in lowered:
        return lowered[base.casefold()]
    by_stem = [name for name in module_list if os.path.splitext(name)[0].casefold() == base.casefold()]
    if len(by_stem) == 1:
        return by_stem[0]
    if by_stem:
        raise AxisValueError(f'{AXIS_LABEL}: "{text}" matches several files ({", ".join(sorted(by_stem))}) - '
                             'write the name with its extension')
    raise AxisValueError(f'{AXIS_LABEL}: unknown file "{text}" - it is not in Forge\'s VAE / Text Encoder list '
                         '(put it in models/text_encoder and press the refresh button next to that list)')


def parse_value(value, module_list) -> tuple:
    """``()`` for "Use Forge current", else the file names (``module_list`` keys) of the value.

    ``a + b`` is split only when the whole text is not a file name itself (file names may contain "+")."""
    if is_current(value):
        return ()
    text = str(value).strip()
    try:
        return (resolve(text, module_list),)
    except AxisValueError:
        if "+" not in text:
            raise
    parts = [part.strip() for part in text.split("+")]
    if any(not part for part in parts):
        raise AxisValueError(f'{AXIS_LABEL}: "{text}" - write "a + b" with a file on each side of "+"')
    names = tuple(resolve(part, module_list) for part in parts)
    if len(set(names)) != len(names):
        raise AxisValueError(f'{AXIS_LABEL}: "{text}" names the same file twice')
    return names


def normalise(entries, module_list) -> list:
    """``entries`` (full paths or file names) the way Forge's ``modules_change`` keeps them: through ``module_list``
    by file name (unknown ones dropped), sorted; here also without duplicates."""
    out = []
    for entry in entries or ():
        name = _basename(entry)
        if name in module_list and module_list[name] not in out:
            out.append(module_list[name])
    return sorted(out)


def effective_base(override_settings, opts_modules) -> list:
    """The selection a cell starts from: an API caller's ``override_settings["forge_additional_modules"]`` if it set one,
    else the dropdown's (``opts.forge_additional_modules``)."""
    value = (override_settings or {}).get(MODULES_KEY)
    if isinstance(value, (list, tuple)):
        return list(value)
    return list(opts_modules or ())


def _not_a_te_message(name: str, kind: Kind) -> str:
    head = f'{AXIS_LABEL}: "{name}"'
    if kind.kind == "qwen35":
        return (f"{head} is the Qwen3.5 encoder of Anima 3.8B, which finds it by itself in models/text_encoder; "
                "Forge ignores it in the VAE / Text Encoder list, so the axis cannot swap it")
    if kind.kind == "adapter":
        return f"{head} is an Anima 3.8B adapter, not a text encoder - choose it in the Anima 3.8B accordion"
    if kind.kind == "vae":
        return f"{head} is a VAE, not a text encoder - use the VAE axis"
    if kind.kind == "multi":
        slots = ", ".join(slot_name(slot) for slot in kind.slots)
        return f"{head} is not a text encoder Forge can load as one (it fills several slots: {slots})"
    if kind.kind == "invalid":
        return f"{head} is not a text encoder Forge can load ({kind.detail})"
    if kind.kind == "unreadable":
        return f"{head} is not a text encoder Forge can load (its safetensors header cannot be read)"
    if kind.kind == "opaque":
        return (f"{head} is not a text encoder Forge can load as far as the axis can tell "
                "(a GGUF / pickle file outside a text-encoder folder)")
    return f"{head} is not a text encoder Forge can load (no text-encoder key Forge recognises)"


def _describe(kinds) -> str:
    kinds = list(kinds)
    slots = sorted({slot for kind in kinds for slot in kind.slots if kind.kind in ("te", "multi")})
    names = [slot_name(slot) for slot in slots]
    if any(kind.kind == "opaque" and kind.in_te_dir for kind in kinds):
        names.append("a GGUF encoder")
    return ", ".join(names) if names else "no text encoder"


def compose(base, picks, *, module_list, classify) -> list:
    """The module list of one cell: ``base`` (normalised) with the entries of each pick's slot replaced by the pick.

    * entries that are not text encoders (VAEs, Qwen3.5, …) stay;
    * every entry of the pick's slot is replaced; with none, the pick is added when the selection has no text encoder
      at all (as when it is chosen in the dropdown: the checkpoint's own encoder is replaced), and is an error otherwise
      (the model would load it and never use it);
    * a GGUF / pickle encoder (its slot is not read) only replaces the selection's one text encoder, and only alone;
    * two picks of one slot are an error.

    ``classify(path) -> Kind``. Returns full paths, sorted (``modules_change``'s normalisation). Raises AxisValueError."""
    kept = normalise(base, module_list)
    kinds = {path: classify(path) for path in kept}
    base_te = [path for path in kept if kinds[path].is_te_entry]
    result = list(kept)
    picked_slots = {}
    pick_paths = []
    for name in picks:
        path = module_list[name]
        kind = classify(path)
        if kind.kind == "te":
            slot = kind.slot
            if slot in picked_slots:
                raise AxisValueError(f'{AXIS_LABEL}: "{picked_slots[slot]}" and "{name}" both fill the {slot_name(slot)} '
                                     'slot - pick one of them per value')
            picked_slots[slot] = name
        elif not kind.is_te_entry or kind.kind == "multi":
            raise AxisValueError(_not_a_te_message(name, kind))
        pick_paths.append((name, path, kind))

    for name, path, kind in pick_paths:
        if kind.kind == "opaque":
            if len(pick_paths) > 1:
                raise AxisValueError(f'{AXIS_LABEL}: "{name}" is a {kind.detail or "GGUF"} file whose slot cannot be read '
                                     'without loading it - it can only be used alone')
            if len(base_te) != 1:
                raise AxisValueError(
                    f'{AXIS_LABEL}: "{name}" is a {kind.detail or "GGUF"} file whose slot cannot be read without loading '
                    f'it - it can only replace the one text encoder of the current VAE / Text Encoder selection, '
                    f'which has {len(base_te)}')
            result = [entry for entry in result if entry != base_te[0]] + [path]
            continue
        slot = kind.slot
        partial = [entry for entry in kept if kinds[entry].kind == "multi" and slot in kinds[entry].slots]
        if partial:
            raise AxisValueError(f'{AXIS_LABEL}: "{name}" would replace only part of "{_basename(partial[0])}", which '
                                 f'fills {", ".join(slot_name(s) for s in kinds[partial[0]].slots)} together')
        same = [entry for entry in kept if kinds[entry].kind == "te" and kinds[entry].slot == slot]
        if same:
            result = [entry for entry in result if entry not in same] + [path]
        elif not base_te:
            result.append(path)
        elif len(base_te) == 1 and kinds[base_te[0]].kind == "opaque" and len(pick_paths) == 1:
            # the selection's one encoder is a GGUF whose slot is not read: the pick takes its place
            result = [entry for entry in result if entry != base_te[0]] + [path]
        else:
            raise AxisValueError(
                f'{AXIS_LABEL}: "{name}" is a {slot_name(slot)} encoder but the current VAE / Text Encoder selection '
                f'has none (it has: {_describe(kinds[entry] for entry in base_te)}) - the model would load it and not '
                f'use it. To compare {slot_name(slot)} encoders, add one to that list first')
    return sorted(dict.fromkeys(result))


def plan(value, *, base, module_list, classify):
    """The cell's module list for ``value`` (``None`` for "Use Forge current"). Raises AxisValueError."""
    picks = parse_value(value, module_list)
    if not picks:
        return None
    return compose(base, picks, module_list=module_list, classify=classify)


# ---------------------------------------------------------------------------------------------------------------------
# Dropdown choices, cell labels
# ---------------------------------------------------------------------------------------------------------------------

def choices(module_list, module_map, current, classify, *, use_short=True) -> list:
    """"Use Forge current", then the text encoders of a slot the current selection has (every single-slot encoder when
    it has none), shown as the dropdown shows them (file name, or relative path). GGUF / pickle files are not listed
    (their slot is not read) but can be typed in text mode."""
    encoders = [kind for kind in map(classify, normalise(current, module_list)) if kind.is_te_entry]
    slots = {slot for kind in encoders for slot in kind.slots}
    has_encoder = bool(encoders)
    found = []
    for name, path in module_list.items():
        kind = classify(path)
        if kind.kind != "te" or (has_encoder and kind.slot not in slots):
            continue
        found.append(name if use_short else (module_map or {}).get(name, name))
    return [CURRENT, *sorted(found, key=str.casefold)]


def format_label(value, *, module_list, current, classify) -> str:
    """``Text Encoder: <name>`` (``a + b``), or ``Text Encoder: current (<the selection's encoders>)``."""
    try:
        picks = parse_value(value, module_list)
    except AxisValueError:
        return f"{LABEL}: {value}"
    if picks:
        return f"{LABEL}: " + " + ".join(stem(name) for name in picks)
    encoders = [stem(path) for path in normalise(current, module_list) if classify(path).is_te_entry]
    return f"{LABEL}: current ({' + '.join(encoders)})" if encoders else f"{LABEL}: current"


# ---------------------------------------------------------------------------------------------------------------------
# One cell's override_settings
# ---------------------------------------------------------------------------------------------------------------------

def checkpoint_to_carry(loading_parameters, selected):
    """The checkpoint to keep for the cell when another axis chose it (the stock Checkpoint axis writes
    ``forge_loading_parameters`` and puts ``opts.sd_model_checkpoint`` back, which this cell's settings refresh would
    otherwise undo): its name when it is not the dropdown's checkpoint, else ``None``."""
    info = (loading_parameters or {}).get("checkpoint_info")
    if info is None or selected is None:
        return None
    if getattr(info, "filename", None) == getattr(selected, "filename", None):
        return None
    return getattr(info, "name", None) or getattr(info, "title", None)


def cell_overrides(override_settings, modules, base_modules, carry_checkpoint=None):
    """A NEW ``override_settings`` dict for the cell, or ``None`` when ``modules`` is the selection already.

    New because xyz_grid gives every cell a shallow copy of ``p``: the dict is the parent's, shared by every cell (and
    the stock Checkpoint axis reads ``sd_model_checkpoint`` from it). An API caller's checkpoint is kept."""
    if list(modules) == list(base_modules):
        return None
    settings = dict(override_settings or {})
    settings[MODULES_KEY] = list(modules)
    if carry_checkpoint and CHECKPOINT_KEY not in (override_settings or {}):
        settings[CHECKPOINT_KEY] = carry_checkpoint
    return settings
