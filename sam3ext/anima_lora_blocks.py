"""ANIMA LoRA block-layout compatibility for Forge.

ANIMA Base 1.0, 2.9B, and 3.8B use the same 2048-wide DiT block
implementation at different depths (28, 40, and 52 blocks).  Forge remaps
some upward combinations; this module owns the complete compatibility matrix
without modifying Forge itself:

* 28 <-> 40 (Base 1.0 <-> 2.9B)
* 28 <-> 52 (Base 1.0 <-> 3.8B)
* 40 <-> 52 (2.9B <-> 3.8B)

Mappings are expressed as ``target index -> source index``.  Expansion clones
the preceding source-lineage block into each inserted position.  Contraction
selects the original lineage positions and deliberately drops generation-only
blocks.  Only complete, contiguous LoRA layouts are remapped: guessing the
source architecture of a sparse/partial LoRA would silently attach weights to
the wrong semantic depth.
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, MutableMapping


LOGGER = logging.getLogger("sam-extra.anima-lora-blocks")

ANIMA_BASE_BLOCKS = 28
ANIMA_29B_BLOCKS = 40
ANIMA_38B_BLOCKS = 52
SUPPORTED_BLOCK_COUNTS = (
    ANIMA_BASE_BLOCKS,
    ANIMA_29B_BLOCKS,
    ANIMA_38B_BLOCKS,
)

LAYOUT_NAMES = {
    ANIMA_BASE_BLOCKS: "ANIMA Base 1.0",
    ANIMA_29B_BLOCKS: "ANIMA 2.9B",
    ANIMA_38B_BLOCKS: "ANIMA 3.8B",
}

_BLOCK_KEY_RE = re.compile(
    r"^(lora_unet_blocks_|diffusion_model\.blocks\.)(\d+)(.*)$"
)

# The semantic connector is bundled only into ANIMA 3.8B v1.1.  Qwen3.5 is a
# separate encoder and must not be classified by DiT block count, so only
# exact connector namespaces are filtered during a downward projection.
_ANIMA_38B_ONLY_KEY_PREFIXES = (
    "net.anima_v2_connector.",
    "diffusion_model.anima_v2_connector.",
    "lora_unet_anima_v2_connector_",
)

# Forge's Base 1.0 -> 2.9B mapping.  Repeated entries are the twelve blocks
# inserted by the 40-block model and inherit the preceding Base block.
_ANIMA_28_TO_40 = (
    0,
    1,
    1,
    2,
    3,
    3,
    4,
    5,
    5,
    6,
    7,
    7,
    8,
    9,
    9,
    10,
    11,
    11,
    12,
    13,
    14,
    14,
    15,
    16,
    16,
    17,
    18,
    18,
    19,
    20,
    20,
    21,
    22,
    22,
    23,
    24,
    24,
    25,
    26,
    27,
)

# These are the 28 original Base lineage positions inside the 40-block model.
# This must not be derived by taking the first occurrence in _ANIMA_28_TO_40:
# Forge's source checkpoints identify these exact blocks as the shared weights.
_ANIMA_40_TO_28 = (
    0,
    1,
    3,
    4,
    6,
    7,
    9,
    10,
    12,
    13,
    15,
    16,
    18,
    19,
    20,
    22,
    23,
    25,
    26,
    28,
    29,
    31,
    32,
    34,
    35,
    37,
    38,
    39,
)

# Anima-3.8B-v1.1.safetensors records this expansion in its own metadata:
# old_block_count=40, new_block_count=52, source_method="Anima-2.9B /
# LLaMA-Pro block expansion", and the following insertion/source pairs.
_ANIMA_38B_INSERTED_TO_29B_SOURCE = {
    3: 2,
    7: 5,
    11: 8,
    15: 11,
    19: 14,
    23: 17,
    27: 20,
    31: 23,
    35: 26,
    39: 29,
    43: 32,
    47: 35,
}


def _make_expansion_mapping(
    source_count: int,
    inserted_to_source: dict[int, int],
) -> tuple[int, ...]:
    """Build a target->source expansion mapping and validate every position."""

    target_count = source_count + len(inserted_to_source)
    mapping: list[int] = []
    next_source = 0
    for target_index in range(target_count):
        if target_index in inserted_to_source:
            source_index = inserted_to_source[target_index]
            if source_index != next_source - 1:
                raise ValueError(
                    "ANIMA inserted block must clone the immediately preceding "
                    f"source block: target={target_index}, source={source_index}, "
                    f"expected={next_source - 1}"
                )
            mapping.append(source_index)
            continue
        if next_source >= source_count:
            raise ValueError("ANIMA expansion consumed more source blocks than exist")
        mapping.append(next_source)
        next_source += 1

    if next_source != source_count:
        raise ValueError(
            f"ANIMA expansion consumed {next_source}/{source_count} source blocks"
        )
    return tuple(mapping)


def _make_contraction_mapping(
    target_count: int,
    source_count: int,
    inserted_positions: set[int],
) -> tuple[int, ...]:
    """Return target->source positions while excluding inserted source blocks."""

    mapping = tuple(
        source_index
        for source_index in range(source_count)
        if source_index not in inserted_positions
    )
    if len(mapping) != target_count:
        raise ValueError(
            f"ANIMA contraction produced {len(mapping)} positions, expected {target_count}"
        )
    return mapping


_ANIMA_40_TO_52 = _make_expansion_mapping(
    ANIMA_29B_BLOCKS,
    _ANIMA_38B_INSERTED_TO_29B_SOURCE,
)
_ANIMA_52_TO_40 = _make_contraction_mapping(
    ANIMA_29B_BLOCKS,
    ANIMA_38B_BLOCKS,
    set(_ANIMA_38B_INSERTED_TO_29B_SOURCE),
)

# Compose target->source mappings through the 40-block lineage.
_ANIMA_28_TO_52 = tuple(
    _ANIMA_28_TO_40[source_40] for source_40 in _ANIMA_40_TO_52
)
_ANIMA_52_TO_28 = tuple(
    _ANIMA_52_TO_40[source_40] for source_40 in _ANIMA_40_TO_28
)

BLOCK_MAPPINGS: dict[tuple[int, int], tuple[int, ...]] = {
    (ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS): _ANIMA_28_TO_40,
    (ANIMA_29B_BLOCKS, ANIMA_BASE_BLOCKS): _ANIMA_40_TO_28,
    (ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS): _ANIMA_40_TO_52,
    (ANIMA_38B_BLOCKS, ANIMA_29B_BLOCKS): _ANIMA_52_TO_40,
    (ANIMA_BASE_BLOCKS, ANIMA_38B_BLOCKS): _ANIMA_28_TO_52,
    (ANIMA_38B_BLOCKS, ANIMA_BASE_BLOCKS): _ANIMA_52_TO_28,
}


@dataclass(frozen=True)
class BlockRemapResult:
    """Observable result of one in-place LoRA compatibility pass."""

    source_blocks: int | None
    target_blocks: int
    supported: bool
    remapped: bool
    moved_adapter_keys: int = 0
    dropped_auxiliary_keys: int = 0

    @property
    def duplicated_blocks(self) -> int:
        if not self.remapped or self.source_blocks is None:
            return 0
        return max(0, self.target_blocks - self.source_blocks)

    @property
    def dropped_blocks(self) -> int:
        if not self.remapped or self.source_blocks is None:
            return 0
        return max(0, self.source_blocks - self.target_blocks)


def anima_lora_block_indices(lora: MutableMapping[str, Any]) -> set[int]:
    """Return the exact ANIMA DiT block indices represented by a LoRA."""

    return {
        int(match.group(2))
        for key in lora
        if (match := _BLOCK_KEY_RE.match(key)) is not None
    }


def detect_anima_lora_layout(lora: MutableMapping[str, Any]) -> int | None:
    """Detect a complete supported layout; refuse sparse or non-contiguous sets."""

    indices = anima_lora_block_indices(lora)
    for block_count in SUPPORTED_BLOCK_COUNTS:
        if indices == set(range(block_count)):
            return block_count
    return None


def _move_llm_adapter_keys(lora: MutableMapping[str, Any]) -> int:
    """Mirror Forge's ANIMA LLM-adapter namespace migration."""

    moved = 0
    for key in list(lora):
        if key.startswith("diffusion_model.llm_adapter"):
            lora[key.replace("diffusion_model", "text_encoders.qwen3_06b", 1)] = (
                lora.pop(key)
            )
            moved += 1
        elif key.startswith("lora_unet_llm_adapter"):
            lora[key.replace("lora_unet_llm_adapter", "lora_te_llm_adapter", 1)] = (
                lora.pop(key)
            )
            moved += 1
    return moved


def _drop_38b_only_keys(lora: MutableMapping[str, Any]) -> int:
    dropped = 0
    for key in list(lora):
        lowered = key.lower()
        if lowered.startswith(_ANIMA_38B_ONLY_KEY_PREFIXES):
            lora.pop(key)
            dropped += 1
    return dropped


def _38b_only_keys(lora: MutableMapping[str, Any]) -> list[str]:
    return [
        key
        for key in lora
        if key.lower().startswith(_ANIMA_38B_ONLY_KEY_PREFIXES)
    ]


def _clone_value(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else value


def _remap_blocks_in_place(
    lora: MutableMapping[str, Any],
    mapping: tuple[int, ...],
) -> None:
    """Replace all block keys using one validated target->source mapping."""

    source_blocks: dict[int, list[tuple[str, str, Any]]] = {}
    for key in list(lora):
        match = _BLOCK_KEY_RE.match(key)
        if match is None:
            continue
        source_blocks.setdefault(int(match.group(2)), []).append(
            (match.group(1), match.group(3), lora.pop(key))
        )

    emitted_sources: set[int] = set()
    for target_index, source_index in enumerate(mapping):
        duplicate_source = source_index in emitted_sources
        for prefix, suffix, value in source_blocks[source_index]:
            lora[f"{prefix}{target_index}{suffix}"] = (
                _clone_value(value) if duplicate_source else value
            )
        emitted_sources.add(source_index)


def remap_anima_lora_for_model(
    lora: MutableMapping[str, Any],
    target_blocks: int,
) -> BlockRemapResult:
    """Mutate one loaded LoRA for the active ANIMA model block count.

    The function is the module's main interface and has no Forge dependency,
    making the exact conversion matrix directly testable.  Unsupported sparse
    block layouts are not remapped and are reported with ``supported=False``;
    Forge's independent LLM-adapter namespace normalization still applies.
    """

    source_indices = anima_lora_block_indices(lora)
    source_blocks = detect_anima_lora_layout(lora)
    # Forge performs this namespace migration for every ANIMA LoRA, regardless
    # of its DiT block layout.  Keep that independent normalization even when
    # sparse block indices cannot be remapped safely.
    moved = _move_llm_adapter_keys(lora)

    # Never guess sparse DiT block semantics.  Only the independent LLM key
    # normalization above is allowed on this path.
    if source_indices and source_blocks is None:
        return BlockRemapResult(
            source_blocks=None,
            target_blocks=target_blocks,
            supported=False,
            remapped=False,
            moved_adapter_keys=moved,
        )
    if target_blocks not in SUPPORTED_BLOCK_COUNTS:
        return BlockRemapResult(
            source_blocks=source_blocks,
            target_blocks=target_blocks,
            supported=False,
            remapped=False,
            moved_adapter_keys=moved,
        )

    auxiliary_keys = _38b_only_keys(lora)
    # Never empty Forge's state dict.  load_lora_for_models() divides its
    # unmatched count by len(lora), so an empty connector-only LoRA would turn
    # a clean incompatibility report into ZeroDivisionError.  Keeping these
    # keys lets Forge reject the non-empty, wholly unmatched LoRA safely.
    if (
        target_blocks < ANIMA_38B_BLOCKS
        and not source_indices
        and auxiliary_keys
        and len(auxiliary_keys) == len(lora)
    ):
        return BlockRemapResult(
            source_blocks=None,
            target_blocks=target_blocks,
            supported=False,
            remapped=False,
            moved_adapter_keys=moved,
        )

    dropped_auxiliary = 0

    # Connector weights have no destination below 52 blocks.  At this
    # point the layout is either complete or contains another compatible
    # namespace, so removal cannot create the empty-dict Forge crash guarded
    # above.
    if target_blocks < ANIMA_38B_BLOCKS:
        dropped_auxiliary = _drop_38b_only_keys(lora)

    if not anima_lora_block_indices(lora):
        return BlockRemapResult(
            source_blocks=None,
            target_blocks=target_blocks,
            supported=True,
            remapped=False,
            moved_adapter_keys=moved,
            dropped_auxiliary_keys=dropped_auxiliary,
        )

    if source_blocks == target_blocks:
        return BlockRemapResult(
            source_blocks=source_blocks,
            target_blocks=target_blocks,
            supported=True,
            remapped=False,
            moved_adapter_keys=moved,
            dropped_auxiliary_keys=dropped_auxiliary,
        )

    mapping = BLOCK_MAPPINGS[(source_blocks, target_blocks)]
    _remap_blocks_in_place(lora, mapping)
    return BlockRemapResult(
        source_blocks=source_blocks,
        target_blocks=target_blocks,
        supported=True,
        remapped=True,
        moved_adapter_keys=moved,
        dropped_auxiliary_keys=dropped_auxiliary,
    )


_PATCH_OWNER = "sam-extra.anima-lora-blocks.v1"
_PATCHED_MODULE: ModuleType | None = None
_KNOWN_FORGE_MODULE: ModuleType | None = None
_ORIGINAL_PROCESS_ANIMA: Any = None


def _is_forge_lora_module(module: Any) -> bool:
    if not callable(getattr(module, "process_anima", None)):
        return False
    if not callable(getattr(module, "load_lora_for_models", None)):
        return False
    module_file = str(getattr(module, "__file__", "")).replace("\\", "/").lower()
    return not module_file or module_file.endswith("/sd_forge_lora/networks.py")


def _find_forge_lora_module() -> ModuleType | None:
    current = sys.modules.get("networks")
    if current is not None and _is_forge_lora_module(current):
        return current

    for module in tuple(sys.modules.values()):
        if module is not None and _is_forge_lora_module(module):
            return module

    # Anima Tile-Repair temporarily removes Forge's single-file ``networks``
    # module from sys.modules to import its vendored ``networks`` package.  A
    # UI reload during that window can still reattach to the live Forge module
    # through this direct reference; the context manager later restores the
    # very same object.
    if _KNOWN_FORGE_MODULE is not None and _is_forge_lora_module(
        _KNOWN_FORGE_MODULE
    ):
        return _KNOWN_FORGE_MODULE

    try:
        imported = importlib.import_module("networks")
    except Exception:
        return None
    return imported if _is_forge_lora_module(imported) else None


def _format_remap_message(result: BlockRemapResult) -> str:
    assert result.source_blocks is not None
    source_name = LAYOUT_NAMES[result.source_blocks]
    target_name = LAYOUT_NAMES[result.target_blocks]
    if result.duplicated_blocks:
        action = f"duplicating {result.duplicated_blocks} inserted lineage blocks"
    else:
        action = f"dropping {result.dropped_blocks} source-only blocks"
    return (
        "[sam-extra] Re-Mapping "
        f"{result.source_blocks}-Block ANIMA LoRA to {result.target_blocks}-Block "
        f"({source_name} -> {target_name}; {action})"
    )


def install_forge_lora_block_hook(
    networks_module: ModuleType | None = None,
) -> bool:
    """Patch Forge's narrow ``process_anima`` seam; return True when installed."""

    global _PATCHED_MODULE, _KNOWN_FORGE_MODULE, _ORIGINAL_PROCESS_ANIMA

    module = networks_module or _find_forge_lora_module()
    if module is None or not _is_forge_lora_module(module):
        return False

    _KNOWN_FORGE_MODULE = module

    current = module.process_anima
    if getattr(current, "_sam3_anima_block_owner", None) == _PATCH_OWNER:
        _PATCHED_MODULE = module
        _ORIGINAL_PROCESS_ANIMA = getattr(
            current,
            "_sam3_anima_block_original",
            _ORIGINAL_PROCESS_ANIMA,
        )
        return False


    # If another extension wrapped our already-installed callable, installing
    # again above it would create a stale nested SAM3 hook that cannot be
    # cleanly removed on UI reload.  Keep the existing lower hook instead.
    if _PATCHED_MODULE is module and _ORIGINAL_PROCESS_ANIMA is not None:
        LOGGER.warning(
            "[sam-extra] ANIMA LoRA hook is already installed below another "
            "wrapper; skipping a duplicate installation"
        )
        return False

    original = current
    log = getattr(module, "logger", LOGGER)

    def process_anima_with_38b(lora, blocks):
        target_blocks = int(blocks)
        result = remap_anima_lora_for_model(lora, target_blocks)
        if result.supported:
            if result.remapped:
                log.warning(_format_remap_message(result))
            if result.dropped_auxiliary_keys:
                log.warning(
                    "[sam-extra] Dropped %d ANIMA 3.8B-only connector "
                    "LoRA keys while projecting to %s; this direction is lossy",
                    result.dropped_auxiliary_keys,
                    LAYOUT_NAMES.get(target_blocks, f"{target_blocks} blocks"),
                )
            return True
        if target_blocks in SUPPORTED_BLOCK_COUNTS:
            indices = sorted(anima_lora_block_indices(lora))
            if not indices and _38b_only_keys(lora):
                log.warning(
                    "[sam-extra] ANIMA 3.8B-only connector LoRA "
                    "cannot be projected to %s; keys were left intact so "
                    "Forge can reject it safely",
                    LAYOUT_NAMES[target_blocks],
                )
            else:
                log.warning(
                    "[sam-extra] Refusing to guess a sparse/unsupported ANIMA "
                    "LoRA layout with indices %s for %s; DiT block keys were "
                    "left unchanged",
                    indices,
                    LAYOUT_NAMES[target_blocks],
                )
            # This adapter owns all known targets.  Delegating a sparse layout
            # would let Forge infer a generation from only a contiguous prefix.
            return False
        # Let a future Forge teach us a newer layout, but contain implementations
        # that may raise after partially writing remapped keys.
        before_fallback = dict(lora)
        try:
            return original(lora, blocks)
        except IndexError:
            lora.clear()
            lora.update(before_fallback)
            log.warning(
                "[sam-extra] Forge could not handle unknown ANIMA target "
                "%s; its partial LoRA changes were rolled back",
                target_blocks,
            )
            return False

    process_anima_with_38b._sam3_anima_block_owner = _PATCH_OWNER
    process_anima_with_38b._sam3_anima_block_original = original
    module.process_anima = process_anima_with_38b
    _PATCHED_MODULE = module
    _ORIGINAL_PROCESS_ANIMA = original
    return True


def uninstall_forge_lora_block_hook() -> bool:
    """Restore Forge's original callable when this extension owns the patch."""

    global _PATCHED_MODULE, _ORIGINAL_PROCESS_ANIMA

    module = _PATCHED_MODULE
    current = getattr(module, "process_anima", None) if module is not None else None
    restored = bool(
        module is not None
        and _ORIGINAL_PROCESS_ANIMA is not None
        and getattr(current, "_sam3_anima_block_owner", None) == _PATCH_OWNER
    )
    if restored:
        module.process_anima = _ORIGINAL_PROCESS_ANIMA
    _PATCHED_MODULE = None
    _ORIGINAL_PROCESS_ANIMA = None
    return restored


__all__ = [
    "ANIMA_29B_BLOCKS",
    "ANIMA_38B_BLOCKS",
    "ANIMA_BASE_BLOCKS",
    "BLOCK_MAPPINGS",
    "BlockRemapResult",
    "SUPPORTED_BLOCK_COUNTS",
    "anima_lora_block_indices",
    "detect_anima_lora_layout",
    "install_forge_lora_block_hook",
    "remap_anima_lora_for_model",
    "uninstall_forge_lora_block_hook",
]
