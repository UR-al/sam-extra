"""XYZ axis ``[Modules] Text Encoder`` — pure logic (no Forge, no torch, no Gradio).

Forge Neo (2.30) keeps VAEs and text encoders in ONE list: the "VAE / Text Encoder" dropdown, saved as
``opts.forge_additional_modules`` (full paths). It tells them apart only while loading the model —
``backend/loader.py`` ``replace_state_dict`` looks at each file's keys and shapes and writes a text encoder into one
slot of the checkpoint (``text_encoders.<slot>.``, CLIP-L / CLIP-G where the checkpoint's family keeps them); the
family's ``clip_target`` then reads the slot(s) it uses (Anima ``qwen3_06b``, Krea2 ``qwen3vl_4b``, Flux ``clip_l`` +
``t5xxl`` …). A file that matches no branch is ignored.

The stock XYZ ``VAE`` axis hot-swaps VAE weights; there is no text-encoder axis. This module decides, from the file's
header only (no tensor is read: the safetensors JSON header, or a GGUF file's tensor table after Forge's
``gguf_remapping`` renames), which slot a file fills - ``te_slots`` mirrors replace_state_dict's branches in their order
(``tests/test_xyz_text_encoder_origin.py`` runs Forge's own functions on the same headers) - and which encoders a
checkpoint has built in (``checkpoint_profile``). It builds one cell's module list: the current list with the entry of
the picked file's slot replaced, or the pick added when the checkpoint has that slot built in (``compose``).
``sam3ext/xyz_text_encoder_cell.py`` hands that list to Forge's model loader for the cell without touching Forge's
settings (config.json and the dropdown stay as they are).

Values: ``Use Forge current`` (also ``None``, the stock VAE axis's word for it: no change), a module file (basename,
the dropdown's relative path, or the name without extension), or ``a + b`` (text mode) for families with two encoders
(Flux CLIP-L + T5). The axis may sit on two of X/Y/Z (CLIP-L files on one, T5 files on the other): the second builds
on the first's list, and two axes may not pick the same slot.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .anima38.files import ARCHITECTURE as _ANIMA38_ADAPTER, QWEN35_MARKERS, V2_ARCHITECTURE as _ANIMA38_V2
from .gguf_header import read_gguf_shapes
# The stdlib-only copy of the header reader (sam3ext/anima_core.py has the same function but imports torch).
from .vae_degrid_models import read_safetensors_header

AXIS_LABEL = "[Modules] Text Encoder"
LABEL = "Text Encoder"
CURRENT = "Use Forge current"
# Checkpoint (1.0) stays the outer loop, then this axis, then the stock VAE axis (0.7): a text-encoder change reloads
# the model like a checkpoint change does.
COST = 0.9
# Set on the cell's copy of ``p`` by the axis (``{"value": …, "modules": [...], "slots": frozenset, "origin": [...]}`` -
# ``origin`` is the selection the first Text Encoder axis of the cell started from; ``None`` for "Use Forge current")
# and read by the hook.
CELL_ATTR = "_sam_extra_xyz_text_encoder"
# Set on the grid's ``p`` by ``confirm``: the slots each Text Encoder axis picks (two axes may not share one).
CONFIRMED_ATTR = "_sam_extra_xyz_text_encoder_axes"
MODULES_KEY = "forge_additional_modules"
CHECKPOINT_KEY = "sd_model_checkpoint"
SAFETENSORS_EXTENSIONS = (".safetensors", ".sft")
GGUF_EXTENSION = ".gguf"
# A pick whose slot is not read (a pickle file): it conflicts with any pick of another axis.
UNKNOWN_SLOT = "?"

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
# Words that mean "the selection as it is" elsewhere in Forge (the stock VAE axis's "None", the Hires module dropdown's
# "Built-in" …). "None" is accepted as "Use Forge current"; the others get a pointer to it.
CURRENT_ALIASES = ("current", "none")
NOT_A_FILE_WORDS = ("default", "automatic", "auto", "built-in", "builtin", "built in", "use same choices")


class AxisValueError(ValueError):
    """A value the axis cannot use; raised from ``confirm`` before the first cell (like the stock "Unknown Checkpoint")."""


class ForgeLoadError(Exception):
    """Forge's replace_state_dict would raise on this file (a failed ``assert`` or a missing key it reads)."""


# ---------------------------------------------------------------------------------------------------------------------
# Which slot Forge puts a file in — backend/loader.py replace_state_dict (Forge Neo 2.30), branch for branch
# ---------------------------------------------------------------------------------------------------------------------

# CLIP-G / CLIP-L: Forge tries these keys and accepts the first whose shape[0] is 1280 / 768. (Where the weights then
# go depends on the checkpoint — nowhere for a family without CLIP, see PREFIX_L / PREFIX_G — but the file is a CLIP-G
# / CLIP-L encoder.)
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

    ``header`` is a safetensors header (``{key: {"dtype", "shape", "data_offsets"}, "__metadata__": …}``; a GGUF file's
    comes from ``gguf_header``). The order and the ``if`` / ``elif`` structure are Forge's: a Qwen3-VL file that also
    has the Qwen3 keys is Qwen3-VL. Raises ``ForgeLoadError`` where Forge itself would fail (its asserts, or a key it
    reads that is missing)."""
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
# GGUF files — backend/loader_gguf.py gguf_remapping (Forge Neo 2.30): the key renames, in Forge's order
# ---------------------------------------------------------------------------------------------------------------------

GGUF_T5_TEST_KEY = "enc.blk.0.attn_k.weight"
GGUF_T5_FORMAT = (
    ("enc.", "encoder."), (".blk.", ".block."), ("token_embd", "shared"), ("output_norm", "final_layer_norm"),
    ("attn_q", "layer.0.SelfAttention.q"), ("attn_k", "layer.0.SelfAttention.k"), ("attn_v", "layer.0.SelfAttention.v"),
    ("attn_o", "layer.0.SelfAttention.o"), ("attn_norm", "layer.0.layer_norm"),
    ("attn_rel_b", "layer.0.SelfAttention.relative_attention_bias"), ("ffn_up", "layer.1.DenseReluDense.wi_1"),
    ("ffn_down", "layer.1.DenseReluDense.wo"), ("ffn_gate", "layer.1.DenseReluDense.wi_0"),
    ("ffn_norm", "layer.1.layer_norm"),
)
GGUF_LLM_TEST_KEY = "blk.0.attn_norm.weight"
GGUF_LLM_FORMAT = (
    ("blk.", "model.layers."), ("attn_norm", "input_layernorm"), ("attn_q_norm.", "self_attn.q_norm."),
    ("attn_k_norm.", "self_attn.k_norm."), ("attn_v_norm.", "self_attn.v_norm."), ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"), ("attn_output", "self_attn.o_proj"),
    ("ffn_up", "mlp.up_proj"), ("ffn_down", "mlp.down_proj"), ("ffn_gate", "mlp.gate_proj"),
    ("ffn_norm", "post_attention_layernorm"), ("token_embd", "model.embed_tokens"), ("output_norm", "model.norm"),
    ("output.weight", "lm_head.weight"),
)
GGUF_PATCH_EMBED_KEY = "v.patch_embd.weight.1"
GGUF_CLIP_VISION_TEST_KEY = "mm.0.weight"
GGUF_CLIP_VISION_FORMAT = (
    ("mm.", "visual.merger.mlp."), ("v.post_ln.", "visual.merger.ln_q."), ("v.patch_embd", "visual.patch_embed.proj"),
    ("v.blk.", "visual.blocks."), ("ffn_up", "mlp.up_proj"), ("ffn_down", "mlp.down_proj"), ("ffn_gate", "mlp.gate_proj"),
    ("attn_out.", "attn.proj."), ("ln1.", "norm1."), ("ln2.", "norm2."),
)


def _rename(shapes: dict, table) -> dict:
    out = {}
    for key, shape in shapes.items():
        for old, new in table:
            key = key.replace(old, new)
        out[key] = shape
    return out


def gguf_remap(shapes: dict) -> dict:
    """The tensor names Forge's ``gguf_remapping`` gives a GGUF file before ``replace_state_dict`` looks at them (T5
    and LLM names to Hugging Face ones, the Qwen2.5-VL vision tower's ``mm.`` / ``v.`` names). Shapes stay; the vision
    tower's q/k/v merge and patch-embedding stack only rename keys no branch looks at."""
    if GGUF_T5_TEST_KEY in shapes:
        shapes = _rename(shapes, GGUF_T5_FORMAT)
    if GGUF_LLM_TEST_KEY in shapes:
        shapes = _rename(shapes, GGUF_LLM_FORMAT)
    if GGUF_PATCH_EMBED_KEY in shapes:
        shapes = {key: shape for key, shape in shapes.items() if key != GGUF_PATCH_EMBED_KEY}
    if GGUF_CLIP_VISION_TEST_KEY in shapes:
        shapes = _rename(shapes, GGUF_CLIP_VISION_FORMAT)
    return shapes


def gguf_header(shapes) -> dict | None:
    """A safetensors-style header (``{key: {"shape": [...]}}``) for a GGUF tensor table, after Forge's renames."""
    if shapes is None:
        return None
    return {key: {"shape": list(shape)} for key, shape in gguf_remap(dict(shapes)).items()}


# ---------------------------------------------------------------------------------------------------------------------
# What a checkpoint has built in — replace_state_dict's model type and where it writes CLIP-L / CLIP-G
# ---------------------------------------------------------------------------------------------------------------------

UNET_PREFIXES = ("model.diffusion_model.", "net.")
LEGACY_TEST_KEY = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
FLUX_TEST_KEYS = (
    "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
    "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight",
    "model.diffusion_model.single_transformer_blocks.0.mlp_fc1.qweight",
)
LEGACY_TYPES = {768: "sd1", 1280: "xlrf", 2048: "sdxl"}
PREFIX_L = {
    "-": None,
    "sd1": "cond_stage_model.transformer.",
    "xlrf": None,
    "sdxl": "conditioner.embedders.0.transformer.",
    "flux": "text_encoders.clip_l.transformer.",
}
PREFIX_G = {
    "-": None,
    "sd1": None,
    "xlrf": "conditioner.embedders.0.model.transformer.",
    "sdxl": "conditioner.embedders.1.model.transformer.",
    "flux": None,
}
TEXT_ENCODERS_PREFIX = "text_encoders."


@dataclass(frozen=True)
class Checkpoint:
    """What the axis knows of a checkpoint: ``model_type`` (replace_state_dict's: ``sd1`` / ``xlrf`` / ``sdxl`` /
    ``flux`` / ``-``) and ``slots``, the encoder slots it fills itself (SD1 / SDXL CLIP, an all-in-one Flux file's CLIP-L
    and T5 …) - a picked encoder of such a slot replaces the built-in one, as when it is chosen in the dropdown."""

    name: str
    model_type: str = "-"
    slots: frozenset = frozenset()

    def loads(self, slot) -> bool:
        """Whether Forge writes an encoder of this slot into the model at all (CLIP-L / CLIP-G only where the family
        keeps them; replace_state_dict drops them otherwise)."""
        if slot == "clip_l":
            return PREFIX_L.get(self.model_type) is not None
        if slot == "clip_g":
            return PREFIX_G.get(self.model_type) is not None
        return True


def checkpoint_profile(name: str, header) -> Checkpoint | None:
    """``Checkpoint`` from a checkpoint's header (``None``: not readable). Keys as replace_state_dict sees them:
    ``preprocess_state_dict`` puts ``model.diffusion_model.`` before every key of a file that has no UNet prefix (a
    UNet-only file - no built-in encoder)."""
    if not isinstance(header, dict):
        return None
    header = {key: value for key, value in header.items() if isinstance(key, str) and key != "__metadata__"}
    if not any(key.startswith(UNET_PREFIXES) for key in header):
        header = {UNET_PREFIXES[0] + key: value for key, value in header.items()}
    keys = list(header)
    model_type = "-"
    legacy = header.get(LEGACY_TEST_KEY)
    shape = legacy.get("shape") if isinstance(legacy, dict) else None
    if isinstance(shape, (list, tuple)):
        model_type = LEGACY_TYPES.get(shape[1] if len(shape) > 1 else None, "-")
    elif any(key in header for key in FLUX_TEST_KEYS):
        model_type = "flux"
    slots = set()
    prefix_l, prefix_g = PREFIX_L[model_type], PREFIX_G[model_type]
    for key in keys:
        if prefix_l and key.startswith(prefix_l):
            slots.add("clip_l")
        if prefix_g and key.startswith(prefix_g):
            slots.add("clip_g")
        if key.startswith(TEXT_ENCODERS_PREFIX):
            slot = key[len(TEXT_ENCODERS_PREFIX):].split(".", 1)[0]
            if slot in SLOT_NAMES and slot not in ("clip_l", "clip_g"):
                slots.add(slot)
    return Checkpoint(name, model_type, frozenset(slots))


# ---------------------------------------------------------------------------------------------------------------------
# What a module file is
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Kind:
    """``kind``: ``te`` (one slot - usable), ``multi`` (several slots), ``vae``, ``qwen35`` (Anima 3.8B's Qwen3.5,
    which Forge ignores), ``adapter`` (Anima 3.8B adapter), ``invalid`` (Forge would fail), ``unknown`` (no branch
    matches), ``unreadable`` (no header Forge could read), ``opaque`` (a pickle file - .pt / .pth / .bin / .ckpt - not
    read without loading it; ``in_te_dir`` says whether it sits in a text-encoder folder)."""

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
    """Anima 3.8B's Qwen3.5-4B file (sam3ext/anima38/files.py ``is_unused_qwen35_module``): a ``.safetensors`` file with
    the marker in the name and no ``model.`` key - replace_state_dict matches nothing and sam-extra's loader filter
    drops it from the list."""
    lowered = name.lower()
    if not lowered.endswith(".safetensors") or not any(marker in lowered for marker in QWEN35_MARKERS):
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


def _cached(kind: str, path: str, read):
    """``read(path)`` once per (path, size, mtime_ns); ``None`` when the file cannot be stat'ed."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (kind, os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns)
    if key not in _CACHE:
        _CACHE[key] = read(path)
    return _CACHE[key]


def read_header(path: str):
    """The header of a module or checkpoint file: safetensors JSON, or a GGUF tensor table after Forge's renames
    (``None``: not readable; other extensions: not read)."""
    lowered = path.lower()
    if lowered.endswith(SAFETENSORS_EXTENSIONS):
        return read_safetensors_header(path)
    if lowered.endswith(GGUF_EXTENSION):
        return gguf_header(read_gguf_shapes(path))
    return None


def classify_file(path: str, *, te_roots=()) -> Kind:
    """``Kind`` of a module file; headers are cached by (path, size, mtime_ns)."""
    path = str(path)
    if not path.lower().endswith(SAFETENSORS_EXTENSIONS + (GGUF_EXTENSION,)):
        return Kind("opaque", detail=os.path.splitext(path)[1].lower(), in_te_dir=_under(path, te_roots))
    kind = _cached("module", path, lambda p: classify_header(os.path.basename(p), read_header(p)))
    return kind if kind is not None else Kind("unreadable")


def checkpoint_file(path, name: str | None = None) -> Checkpoint | None:
    """``Checkpoint`` of a checkpoint file (safetensors or GGUF; ``None`` for anything else or an unreadable file)."""
    if not path:
        return None
    path = str(path)
    if not path.lower().endswith(SAFETENSORS_EXTENSIONS + (GGUF_EXTENSION,)):
        return None
    profile = _cached("checkpoint", path, lambda p: checkpoint_profile(os.path.basename(p), read_header(p)))
    if profile is not None and name and profile.name != name:
        profile = Checkpoint(name, profile.model_type, profile.slots)
    return profile


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
    return str(value).strip().casefold() in (CURRENT.casefold(), *CURRENT_ALIASES)


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
    if text.casefold() in NOT_A_FILE_WORDS:
        raise AxisValueError(f'{AXIS_LABEL}: "{text}" is not a file of Forge\'s VAE / Text Encoder list - for the '
                             f'encoders selected there now, use "{CURRENT}"')
    raise AxisValueError(f'{AXIS_LABEL}: unknown file "{text}" - it is not in Forge\'s VAE / Text Encoder list '
                         '(put it in models/text_encoder and press the refresh button next to that list; '
                         f'"{CURRENT}" keeps the encoders selected there)')


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
        return f"{head} is not a text encoder Forge can load (its header cannot be read)"
    if kind.kind == "opaque":
        return (f"{head} is not a text encoder Forge can load as far as the axis can tell "
                "(a pickle file outside a text-encoder folder)")
    return f"{head} is not a text encoder Forge can load (no text-encoder key Forge recognises)"


def _describe(kinds) -> str:
    kinds = list(kinds)
    slots = sorted({slot for kind in kinds for slot in kind.slots if kind.kind in ("te", "multi")})
    names = [slot_name(slot) for slot in slots]
    if any(kind.kind == "opaque" and kind.in_te_dir for kind in kinds):
        names.append("a pickle encoder")
    return ", ".join(names) if names else "no text encoder"


def _no_slot_message(name, slot, base_kinds, checkpoint, opaque) -> str:
    what = slot_name(slot)
    head = f'{AXIS_LABEL}: "{name}" is a {what} encoder'
    if opaque:
        return (f'{head}; the current VAE / Text Encoder selection has no other {what} encoder the axis can see, and '
                f'it cannot tell whether "{_basename(opaque[0])}" (a pickle file, not read without loading it) fills '
                f'that slot - put a safetensors or GGUF {what} encoder in that list first')
    base_kinds = list(base_kinds)
    if base_kinds:
        selection = f"the current VAE / Text Encoder selection has none (it has: {_describe(base_kinds)})"
    else:
        selection = "the current VAE / Text Encoder selection has no text encoder"
    if checkpoint is None:
        where = "and the axis cannot read the checkpoint to see whether it has one built in"
    else:
        built_in = ", ".join(slot_name(s) for s in sorted(checkpoint.slots)) or "no encoder"
        where = f'and the checkpoint "{checkpoint.name}" has none built in (it has: {built_in})'
    return (f"{head} but {selection} {where} - the axis only swaps an encoder the model already has. To compare "
            f"{what} encoders, add one to that list first")


def compose(base, picks, *, module_list, classify, checkpoint=None, added=()) -> list:
    """The module list of one cell: ``base`` (normalised) with the entries of each pick's slot replaced by the pick.

    * entries that are not text encoders (VAEs, Qwen3.5, …) stay;
    * every entry of the pick's slot is replaced; with none, the pick is added when the ``checkpoint`` has that slot
      built in (it replaces the built-in encoder, as when chosen in the dropdown) or when neither the selection nor
      the checkpoint has any encoder the axis can see (a checkpoint that cannot be read, or one with no built-in
      encoder) - otherwise it is an error: the axis swaps encoders the model has, it does not guess whether the model
      reads another slot;
    * CLIP-L / CLIP-G picks are an error for a checkpoint whose family Forge loads no CLIP-L / CLIP-G into;
    * a pickle encoder (its slot is not read) only replaces the selection's one text encoder, and only alone;
    * two picks of one slot are an error.

    ``classify(path) -> Kind``; ``checkpoint``: ``Checkpoint`` or ``None`` (unknown); ``added``: entries of ``base``
    that an earlier Text Encoder axis of the cell put in - replaced like any entry of their slot, but not the
    selection's encoders for the rules above (so the second axis accepts what its ``confirm`` accepted). Returns full
    paths, sorted (``modules_change``'s normalisation). Raises AxisValueError."""
    kept = normalise(base, module_list)
    kinds = {path: classify(path) for path in kept}
    added = set(normalise(added, module_list))
    base_te = [path for path in kept if kinds[path].is_te_entry and path not in added]
    opaque = [path for path in base_te if kinds[path].kind == "opaque"]
    built_in = checkpoint.slots if checkpoint is not None else frozenset()
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
                raise AxisValueError(f'{AXIS_LABEL}: "{name}" is a pickle file ({kind.detail}) whose slot cannot be read '
                                     'without loading it - it can only be used alone')
            if len(base_te) != 1:
                raise AxisValueError(
                    f'{AXIS_LABEL}: "{name}" is a pickle file ({kind.detail}) whose slot cannot be read without loading '
                    f'it - it can only replace the one text encoder of the current VAE / Text Encoder selection, '
                    f'which has {len(base_te)}')
            result = [entry for entry in result if entry != base_te[0]] + [path]
            continue
        slot = kind.slot
        if checkpoint is not None and not checkpoint.loads(slot):
            raise AxisValueError(f'{AXIS_LABEL}: "{name}" is a {slot_name(slot)} encoder, which Forge does not load '
                                 f'for the checkpoint "{checkpoint.name}" (its family has no {slot_name(slot)} slot)')
        partial = [entry for entry in kept if kinds[entry].kind == "multi" and slot in kinds[entry].slots]
        if partial:
            raise AxisValueError(f'{AXIS_LABEL}: "{name}" would replace only part of "{_basename(partial[0])}", which '
                                 f'fills {", ".join(slot_name(s) for s in kinds[partial[0]].slots)} together')
        same = [entry for entry in kept if kinds[entry].kind == "te" and kinds[entry].slot == slot]
        if same:
            result = [entry for entry in result if entry not in same] + [path]
        elif slot in built_in:
            result.append(path)
        elif len(base_te) == 1 and opaque and len(pick_paths) == 1:
            # the selection's one encoder is a pickle file whose slot is not read: the pick takes its place
            result = [entry for entry in result if entry != base_te[0]] + [path]
        elif not base_te and not built_in:
            result.append(path)
        else:
            raise AxisValueError(_no_slot_message(name, slot, (kinds[entry] for entry in base_te), checkpoint, opaque))
    return sorted(dict.fromkeys(result))


def pick_slots(value, module_list, classify) -> frozenset:
    """The slots one value picks (``UNKNOWN_SLOT`` for a pickle file); empty for "Use Forge current"."""
    slots = set()
    for name in parse_value(value, module_list):
        kind = classify(module_list[name])
        slots.update(kind.slots if kind.kind == "te" else (UNKNOWN_SLOT,))
    return frozenset(slots)


def overlap(first: frozenset, second: frozenset) -> frozenset:
    """Slots two axes would both pick (any slot against a pickle file, whose slot is not read)."""
    if (UNKNOWN_SLOT in first and second) or (UNKNOWN_SLOT in second and first):
        return frozenset(first | second)
    return frozenset(first & second)


def plan(value, *, base, module_list, classify, checkpoint=None, added=()):
    """The cell's module list for ``value`` (``None`` for "Use Forge current"). Raises AxisValueError."""
    picks = parse_value(value, module_list)
    if not picks:
        return None
    return compose(base, picks, module_list=module_list, classify=classify, checkpoint=checkpoint, added=added)


# ---------------------------------------------------------------------------------------------------------------------
# Dropdown choices, cell labels
# ---------------------------------------------------------------------------------------------------------------------

def choices(module_list, module_map, current, classify, *, use_short=True, checkpoint=None) -> list:
    """"Use Forge current", then the text encoders of a slot the model has - in the current selection or built into the
    ``checkpoint`` (every single-slot encoder Forge loads for it when neither has one the axis can see) - shown as the
    dropdown shows them (file name, or relative path). Pickle files are not listed (their slot is not read) but can be
    typed in text mode."""
    encoders = [kind for kind in map(classify, normalise(current, module_list)) if kind.is_te_entry]
    slots = {slot for kind in encoders for slot in kind.slots}
    if checkpoint is not None:
        slots |= checkpoint.slots
    has_encoder = bool(encoders) or bool(slots)
    found = []
    for name, path in module_list.items():
        kind = classify(path)
        if kind.kind != "te" or (has_encoder and kind.slot not in slots):
            continue
        if checkpoint is not None and not checkpoint.loads(kind.slot):
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
# The cell's Module N list
# ---------------------------------------------------------------------------------------------------------------------

def swap_entries(listing, base, cell) -> list:
    """``listing`` (Forge's ``Module N`` list for the selection, e.g. with the stock VAE axis's VAE swapped in) with the
    entries the cell took out of ``base`` removed and the ones it put in added (order kept, new ones last)."""
    removed = [entry for entry in base if entry not in cell]
    added = [entry for entry in cell if entry not in base]
    out = [entry for entry in listing if entry not in removed]
    return out + [entry for entry in added if entry not in out]
