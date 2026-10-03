# Vendored from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/presets.py) into sam-extra (mcp_server/sam_extra_mcp/forgeneo/presets.py).
#
# Copyright (c) 2026 Eduardo Abreu
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
# Changes in sam-extra: none - the code below is upstream's, byte for byte.

"""Architecture defaults mirrored from Forge Neo's modules_forge/presets.py.

These are a *fallback*. Measured history always wins: on the development
instance the anima preset declares 32 steps at CFG 4.0, while the operator's
actual 600 most recent generations ran at 11 steps and CFG 1.5, because the
checkpoints are turbo variants. Trusting this table over observed usage would
generate three times slower at the wrong guidance scale.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tags that mark a LoRA as changing the sampling regime rather than the image
# content. Matched exactly: substring matching turns "hyper-realistic" and
# "hyperass" into false accelerators.
ACCELERATOR_TAGS = frozenset({"turbo", "distill", "distillation", "dmd2", "lcm", "hyper", "lightning"})
ACCELERATOR_NAME_HINTS = ("turbo", "lcm", "hyper", "lightning", "distill", "dmd2")


@dataclass(frozen=True)
class ArchDefaults:
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    shift: float | None = None
    frames: int = 1  # >1 marks a video architecture

    @property
    def is_video(self) -> bool:
        return self.frames > 1


ARCH_DEFAULTS: dict[str, ArchDefaults] = {
    "sd": ArchDefaults("Euler a", "Automatic", 32, 6.0),
    "xl": ArchDefaults("Euler a", "Automatic", 24, 4.5, shift=-9.0),
    "flux": ArchDefaults("Euler", "Beta", 20, 1.0),
    "klein": ArchDefaults("Euler", "Beta", 4, 1.0),
    "qwen": ArchDefaults("LCM", "Normal", 8, 1.0),
    "lumina": ArchDefaults("Res Multistep", "Simple", 32, 4.0, shift=6.0),
    "zit": ArchDefaults("Euler", "Beta", 9, 1.0, shift=9.0),
    "wan": ArchDefaults("Euler", "Simple", 4, 1.0, shift=5.0, frames=16),
    "anima": ArchDefaults("ER SDE", "Beta", 32, 4.0, shift=3.0),
    "ernie": ArchDefaults("Euler", "Simple", 8, 1.0, shift=3.0),
    "pid": ArchDefaults("LCM", "Simple", 4, 1.0, shift=-1.5),
    "krea": ArchDefaults("Euler", "Simple", 8, 1.0, shift=-1.15),
}

# How each family expects prompts to be written. Used only as a prior; when the
# operator's own history is readable, the measured style overrides it.
PROMPT_STYLE: dict[str, str] = {
    "sd": "natural language, comma-separated descriptors",
    "xl": "natural language with quality tags",
    "anima": "hybrid: danbooru tags and natural language mix freely",
    "flux": "natural language, full sentences",
    "klein": "natural language, full sentences",
    "qwen": "natural language, strong prompt adherence",
    "zit": "natural language, concise",
    "krea": "natural language, photographic direction",
    "lumina": "natural language",
    "wan": "natural language describing motion",
}

# Lineages that share an architecture but not a prompt dialect. Detected from
# the checkpoint name or sidecar baseModel, never from tensor shape.
LINEAGE_PROMPT_STYLE: dict[str, str] = {
    "pony": "danbooru tags with score_9, score_8_up, score_7_up prefix",
    "illustrious": "danbooru tags, underscore form, quality tags first",
    "noobai": "danbooru tags, underscore form, quality tags first",
    "animagine": "danbooru tags, character/series/quality order",
}


def defaults_for(arch: str | None) -> ArchDefaults | None:
    if not arch:
        return None
    return ARCH_DEFAULTS.get(arch.lower())


def detect_lineage(name: str) -> str | None:
    """Best-effort lineage guess from a checkpoint name."""
    lowered = name.lower()
    for lineage in LINEAGE_PROMPT_STYLE:
        if lineage in lowered:
            return lineage
    return None


def looks_like_accelerator(name: str, tags: tuple[str, ...] = ()) -> bool:
    """Whether a LoRA alters the sampling regime instead of image content."""
    if any(tag.strip().lower() in ACCELERATOR_TAGS for tag in tags):
        return True
    lowered = name.lower()
    return any(hint in lowered for hint in ACCELERATOR_NAME_HINTS)
