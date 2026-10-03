"""The MCP annotations of each tool, kept apart from the SDK so they can be checked anywhere.

Clients may use these hints to decide what to confirm with the user, so they describe the worst
a tool can do with any of its arguments (MCP ToolAnnotations):

- read_only: changes nothing outside this server's memory.
- destructive: may change or remove something that already exists in Forge or among the
  operator's files - the loaded model, a running job, an existing file. False means it only adds
  (new images, new files) or only touches this server's own cache.
- idempotent: repeating a call with the same arguments has no further effect.
- open_world: reaches beyond this PC's Forge, i.e. the internet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Hints:
    read_only: bool
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False
    why: str = ""


TOOL_HINTS: Mapping[str, Hints] = MappingProxyType(
    {
        "capabilities": Hints(
            read_only=True, idempotent=True,
            why="reads Forge and the output folder; refresh=True only rebuilds this server's indexes",
        ),
        "model_profile": Hints(read_only=True, idempotent=True, why="reads the loaded checkpoint and past generations"),
        "prompt_dialect": Hints(
            read_only=False, idempotent=True,
            why=(
                "confirm=... stores the operator's answer in this server's own dialects.json (replacing an "
                "earlier answer for that checkpoint); nothing in Forge changes. With FORGE_CIVITAI_LOOKUP=1 "
                "it also asks civitai.com about the checkpoint's hash (open_world_for)"
            ),
        ),
        "loras": Hints(read_only=True, idempotent=True, why="searches Forge's LoRA list"),
        "lora_info": Hints(read_only=True, idempotent=True, why="reads one LoRA's details"),
        "models": Hints(
            read_only=False, destructive=True, idempotent=True,
            why=(
                "action=load replaces the loaded checkpoint, preset, VAE and text encoders for everyone "
                "using the instance; refresh rescans the list; list only reads"
            ),
        ),
        "module_check": Hints(read_only=True, idempotent=True, why="compares the selected modules with what the architecture needs"),
        "module_download": Hints(
            read_only=False, idempotent=True, open_world=True,
            why="downloads a new file from Hugging Face into the models folder and never overwrites one",
        ),
        "generate": Hints(
            read_only=False,
            why="renders new images into the output folders; every call makes new ones, nothing is replaced",
        ),
        "progress": Hints(
            read_only=False, destructive=True,
            why="interrupt/skip stop the running job, whoever started it; status only reads",
        ),
    }
)

# Tools that reach civitai.com when the operator sets FORGE_CIVITAI_LOOKUP=1 (forgeneo.civitai).
CIVITAI_TOOLS = frozenset({"prompt_dialect"})


def open_world_for(name: str, *, civitai_lookup: bool = False) -> bool:
    return TOOL_HINTS[name].open_world or (civitai_lookup and name in CIVITAI_TOOLS)


def annotation_fields(name: str, *, civitai_lookup: bool = False) -> dict:
    """Keyword arguments for mcp.types.ToolAnnotations."""
    hints = TOOL_HINTS[name]
    return {
        "read_only_hint": hints.read_only,
        "destructive_hint": hints.destructive,
        "idempotent_hint": hints.idempotent,
        "open_world_hint": open_world_for(name, civitai_lookup=civitai_lookup),
    }
