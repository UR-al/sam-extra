"""XYZ ``VAE`` axis — when Forge's in-place VAE swap cannot render a cell (pure logic: no Forge, no torch, no Gradio).

Forge's stock X/Y/Z "VAE" axis (scripts/xyz_grid.py ``apply_vae``) puts the file into ``p.override_settings["sd_vae"]``.
``process_images`` loads the cell's model and then swaps the VAE weights in place: ``modules/sd_vae.py``
``reload_vae_weights`` -> ``_load_vae_dict`` -> ``first_stage_model.load_state_dict(file)`` - strict, into the VAE the
model was BUILT with. Forge picks that VAE's class while it builds the model, from the VAE file of the "VAE / Text
Encoder" selection (``backend/loader.py``: for Wan / Qwen-Image / Anima a file with ``post_quant_conv.weight`` gets the
2D ``Qwen2DVAE``, any other the 3D ``WanVAE``). A file of the other family - or any file whose tensors that VAE does
not have exactly - fails the strict load, ``process_images`` raises and xyz_grid draws an empty (black) cell without
sampling. Choosing the file in the dropdown works because Forge then builds the model with it.

This module reads VAE files' headers (``xyz_text_encoder.read_header``, the [Modules] Text Encoder axis's safetensors /
GGUF reader and cache; whether a file is a VAE at all is that axis's ``classify_file``) and answers, per cell:

* ``strict_shapes`` / ``swappable`` - whether the strict load succeeds: the file's tensor names and shapes, without the
  keys ``_load_vae_dict`` drops, are exactly the model's VAE's (names and shapes are all ``load_state_dict`` checks);
* ``predicted_shapes`` - the model's VAE before Forge has built it: the last VAE file of the module list (Forge loads
  the list in order and each VAE file replaces the VAE) or the checkpoint's own; ``None`` when headers cannot tell
  (a pickle file, or a format ``replace_state_dict`` converts first);
* ``vae_family`` / ``describe`` - the file's family (``3d`` Wan / Qwen-Image, ``2d`` Qwen2D, ``kl`` AutoencoderKL) for
  messages and tests;
* ``replace_vae`` - the cell's module list with the VAE replaced: what the dropdown holds when the file is chosen there.

``sam3ext/xyz_vae_cell.py`` makes the call for a cell; a cell Forge cannot swap is loaded like the [Modules] Text
Encoder axis's cells (``sam3ext/xyz_text_encoder_cell.py``).

Forge Neo 2.31 changed the axis: ``apply_vae`` writes the cell's whole module list into
``p.override_settings["forge_additional_modules"]`` (the dropdown's list without the axis's own values, plus the value)
and the model is loaded with it - every family renders, nothing is swapped in place, and nothing writes ``sd_vae`` any
more (that is the new ``VAE ("state_dict" only)`` axis, 2.30's swap, not replaced). A VAE of the selection that is not
one of the values stays in that list, and the loader takes the last VAE of the sorted list; ``axis_cell_modules`` is
the list with exactly the value as the VAE, which the replaced axis writes instead.
"""
from __future__ import annotations

import os
import re

from . import xyz_text_encoder as xte

AXIS_LABEL = "VAE"                 # Forge's axis, replaced in place (scripts/modules_xyz_text_encoder.py)
OVERRIDE_KEY = "sd_vae"            # where xyz_grid's apply_vae puts the file (Forge 2.30; 2.31: its state_dict axis)
# Set on the cell's copy of ``p`` by the replaced axis (``{"value": x}``); the hook marks it ``SEEN`` when it has
# decided the cell, so a pass inside the cell (a copy of the cell's ``p``) is not decided again.
CELL_ATTR = "_sam_extra_xyz_vae"
SEEN = "seen"

# modules/sd_vae.py _load_vae_dict: keys starting with "loss" and vae_ignore_keys are not loaded
LOSS_PREFIX = "loss"
IGNORED_KEYS = frozenset({"model_ema.decay", "model_ema.num_updates"})

FAMILY_3D = "3d"
FAMILY_2D = "2d"
FAMILY_KL = "kl"
# Wan / Qwen-Image VAEs norm with RMS "gamma" tensors (3D: decoder.middle.0.residual.0.gamma; the 2D Qwen2D export:
# decoder.mid_block.resnets.0.norm1.gamma ...). backend/loader.py builds Qwen2DVAE when post_quant_conv.weight is there.
WAN_KEY = "decoder.middle.0.residual.0.gamma"
POST_QUANT_KEY = "post_quant_conv.weight"
GAMMA_SUFFIX = ".gamma"
# backend/loader.py replace_state_dict changes these formats before the model sees them (flux 2: the
# decoder.post_quant_conv prefix, diffusers: convert_vae_state_dict, "format 0": first_stage_model. keys) - the model's
# VAE then has other names than the file
CONVERTED_KEYS = ("decoder.post_quant_conv.weight", "decoder.up_blocks.0.resnets.0.norm1.weight",
                  "first_stage_model.decoder.conv_in.weight")
KL_KEYS = ("decoder.conv_in.weight", *CONVERTED_KEYS)
# The checkpoint's own VAE: huggingface_guess's vae_key_prefix ("first_stage_model." for SD1 / SDXL, "vae." for the
# rest; Chroma renames first_stage_model. to vae.). preprocess_state_dict prefixes every key of a file without a UNet
# key, so such a file has no VAE.
CHECKPOINT_VAE_PREFIXES = ("first_stage_model.", "vae.")
UNET_PREFIXES = xte.UNET_PREFIXES

FAMILY_NAMES = {FAMILY_3D: "3D Wan / Qwen-Image VAE", FAMILY_2D: "2D Qwen2D VAE", FAMILY_KL: "AutoencoderKL VAE"}
# the tensor whose first dimension is the latent channel count, per family (dimension index)
CHANNEL_KEYS = {FAMILY_3D: ("conv2.weight", 0), FAMILY_2D: (POST_QUANT_KEY, 0), FAMILY_KL: ("decoder.conv_in.weight", 1)}


def _keys(source) -> set:
    return {key for key in (source or ()) if isinstance(key, str) and key != "__metadata__"}


def strict_shapes(header) -> dict | None:
    """``{name: shape tuple}`` the strict load compares, from a header (``{name: {"shape": [...]}}``, as
    ``xyz_text_encoder.read_header`` gives) - without the keys ``_load_vae_dict`` drops. ``None``: not a header, or an
    entry without a shape."""
    if not isinstance(header, dict):
        return None
    out = {}
    for key, value in header.items():
        if key == "__metadata__" or not isinstance(key, str):
            continue
        if key[0:4] == LOSS_PREFIX or key in IGNORED_KEYS:
            continue
        shape = value.get("shape") if isinstance(value, dict) else None
        if not isinstance(shape, (list, tuple)):
            return None
        try:
            out[key] = tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            return None
    return out


def module_shapes(state_dict) -> dict:
    """``{name: shape tuple}`` of a built model's VAE (``first_stage_model.state_dict()``: tensors with ``.shape``)."""
    return {key: tuple(int(dim) for dim in value.shape) for key, value in state_dict.items()}


def swappable(model_shapes, file_shapes) -> bool:
    """Whether ``first_stage_model.load_state_dict(file)`` succeeds (strict): the same names with the same shapes."""
    return model_shapes is not None and file_shapes is not None and dict(model_shapes) == dict(file_shapes)


def vae_family(source) -> str | None:
    """``3d`` / ``2d`` / ``kl`` / ``None`` (not a VAE layout this knows) from a header or a shapes dict."""
    keys = _keys(source)
    if WAN_KEY in keys or any(key.endswith(GAMMA_SUFFIX) for key in keys):
        return FAMILY_2D if POST_QUANT_KEY in keys else FAMILY_3D
    if any(key in keys for key in KL_KEYS):
        return FAMILY_KL
    return None


def latent_channels(source):
    family = vae_family(source)
    if family is None or not isinstance(source, dict):
        return None
    key, index = CHANNEL_KEYS[family]
    entry = source.get(key)
    shape = entry.get("shape") if isinstance(entry, dict) else entry
    if isinstance(shape, (list, tuple)) and len(shape) > index:
        return shape[index]
    return None


def describe(source) -> str:
    """``2D Qwen2D VAE (16 ch)`` …, for messages."""
    family = vae_family(source)
    if family is None:
        return "VAE of a layout the axis does not know"
    channels = latent_channels(source)
    return FAMILY_NAMES[family] + (f" ({channels} ch)" if channels is not None else "")


def mismatch(model_shapes, file_shapes) -> str:
    """A short account of why the strict load fails (for the console line)."""
    model_keys, file_keys = set(model_shapes or ()), set(file_shapes or ())
    missing, unexpected = model_keys - file_keys, file_keys - model_keys
    shaped = sorted(key for key in model_keys & file_keys if tuple(model_shapes[key]) != tuple(file_shapes[key]))
    parts = []
    if missing:
        parts.append(f"{len(missing)} of the model's VAE tensors are not in the file")
    if unexpected:
        parts.append(f"{len(unexpected)} tensors of the file are not in the model's VAE")
    if shaped:
        parts.append(f"{len(shaped)} tensors have other shapes (e.g. {shaped[0]})")
    return "; ".join(parts) or "the tensors differ"


def needs_conversion(shapes) -> bool:
    """A VAE file Forge's ``replace_state_dict`` renames before the model sees it (its names are not the model's)."""
    return any(key in _keys(shapes) for key in CONVERTED_KEYS)


def checkpoint_vae_shapes(header) -> dict | None:
    """The strict-load shapes of a checkpoint's own VAE (``None``: none, or not readable)."""
    keys = _keys(header) if isinstance(header, dict) else set()
    if not any(key.startswith(UNET_PREFIXES) for key in keys):
        return None
    for prefix in CHECKPOINT_VAE_PREFIXES:
        inner = {key[len(prefix):]: header[key] for key in keys if key.startswith(prefix)}
        if inner:
            return strict_shapes(inner)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Files and module lists
# ---------------------------------------------------------------------------------------------------------------------

def _basename(entry) -> str:
    return re.split(r"[\\/]", str(entry))[-1]


def file_shapes(path) -> dict | None:
    """``strict_shapes`` of a VAE file's header (safetensors / GGUF; ``None`` for a pickle or unreadable file), cached by
    (path, size, mtime) with the Text Encoder axis's headers."""
    if not path:
        return None
    path = str(path)
    if not path.lower().endswith(xte.SAFETENSORS_EXTENSIONS + (xte.GGUF_EXTENSION,)):
        return None
    return xte._cached("vae", path, lambda p: strict_shapes(xte.read_header(p)))


def checkpoint_file_shapes(path) -> dict | None:
    """``checkpoint_vae_shapes`` of a checkpoint file (cached like ``file_shapes``)."""
    if not path:
        return None
    path = str(path)
    if not path.lower().endswith(xte.SAFETENSORS_EXTENSIONS + (xte.GGUF_EXTENSION,)):
        return None
    return xte._cached("checkpoint vae", path, lambda p: checkpoint_vae_shapes(xte.read_header(p)))


def is_vae_entry(entry, classify, vae_names=()) -> bool:
    """An entry Forge loads as the VAE: a file whose header has a VAE branch of ``replace_state_dict``
    (``classify(path).kind == "vae"``), or a pickle / unreadable file of a VAE folder (``sd_vae.vae_dict``)."""
    kind = classify(entry)
    if kind.kind == "vae":
        return True
    return kind.kind in ("opaque", "unreadable") and _basename(entry) in set(vae_names or ())


def vae_entries(entries, classify, vae_names=()) -> list:
    return [entry for entry in entries or () if is_vae_entry(entry, classify, vae_names)]


def same_file(a, b) -> bool:
    try:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
    except (TypeError, ValueError):
        return a == b


def replace_vae(entries, vae, classify, vae_names=()) -> list:
    """``entries`` with every VAE entry replaced by ``vae`` (added when there is none: it then replaces the
    checkpoint's own VAE), sorted like Forge's ``modules_change`` - the list the dropdown would hold."""
    drop = vae_entries(entries, classify, vae_names)
    kept = [entry for entry in entries or () if entry not in drop and not same_file(entry, vae)]
    return sorted(dict.fromkeys([*kept, str(vae)]))


class CellModules(list):
    """Forge 2.31's ``VAE`` axis cell: the module list sam-extra writes into the cell's ``override_settings`` (a plain
    list to Forge's ``set_config``). That dict is shared by every cell, so the next cell finds this list there;
    ``start`` is the list the grid started from - an API caller's own, or the dropdown's as the grid's first cell read
    it - which the next cell's VAE replaces (or, for "None", keeps)."""

    def __init__(self, entries=(), start=None):
        super().__init__(entries)
        self.start = None if start is None else list(start)


def axis_cell_modules(before, value_path, selection, *, module_list, classify, vae_names=()) -> CellModules:
    """The list of a Forge 2.31 ``VAE`` axis cell, written over the one the stock axis wrote (the dropdown's selection
    without the axis's own values, plus the value: a VAE of the selection that is not one of the values stays, and the
    loader takes the last VAE of the sorted list). ``before``: what the cell's ``override_settings`` held before the
    stock axis wrote (an API caller's list, a ``CellModules`` of the previous cell, or nothing); ``value_path``: the
    picked file (``None`` for the value "None": the selection's own VAE); ``selection``: the dropdown's list now. The
    result is the grid's starting list with exactly the value as its VAE - what choosing the file in the "VAE / Text
    Encoder" dropdown loads, as the 2.30 axis rendered it.

    ``selection`` counts at the grid's first cell only (no list in ``before``): the start it gives the grid is carried
    to the later cells. Forge 2.31 leaves a cell's list in its settings - in the dropdown's list the next
    cell reads - when config.json has no module list (``process_images`` puts back only the keys the settings had) or
    the caller sent ``override_settings_restore_afterwards=False``."""
    start = before.start if isinstance(before, CellModules) else before
    if not isinstance(start, (list, tuple)):
        start = selection                  # the grid's first cell: the dropdown's list, kept for the later cells
    start = list(start or ())
    base = xte.normalise(start, module_list)
    modules = base if value_path is None else replace_vae(base, value_path, classify, vae_names)
    return CellModules(modules, start)


def predicted_shapes(entries, checkpoint_shapes, *, classify, read=file_shapes, vae_names=()) -> dict | None:
    """The strict-load shapes of the VAE Forge will build from the module list ``entries`` and a checkpoint whose own
    VAE has ``checkpoint_shapes``: the last VAE entry's (``None`` when its header cannot be read or Forge converts its
    format first), else the checkpoint's."""
    vaes = vae_entries(entries, classify, vae_names)
    if vaes:
        shapes = read(vaes[-1])
        if shapes is None or needs_conversion(shapes):
            return None
        return shapes
    return checkpoint_shapes
