# Vendored from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/modules.py) into sam-extra (mcp_server/sam_extra_mcp/forgeneo/modules.py).
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
# Changes in sam-extra (2026-10-03):
# - ArchModules.extras: recognised add-ons an architecture does not need but a legitimate setup
#   may still select under its preset. They are reported under "extras_loaded" and never as
#   "unrecognised". For `anima`: the Qwen3.5-4B text backbone and the Anima 3.8B expanded
#   adapter used by sam-extra's built-in Anima 3.8B (sam3ext/anima38; file markers as in
#   sam3ext/anima38/files.py QWEN35_MARKERS). Upstream flagged both as leftovers from another
#   preset, a false warning on every sam-extra 3.8B setup.
# - The Qwen-Image VAE requirement also matches the Qwen2D family by its "qwen2d" prefix
#   (e.g. Qwen2D-Anime builds), not only the "qwen2d_vae" spelling.

"""Which VAE and text encoders each architecture needs.

Source: the Forge Classic wiki's Download Models page, which is the reference
for what to fetch and where it goes:
https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

This table exists because the instance cannot answer the question. Forge stores
`forge_additional_modules_<arch>`, but that records the last selection made
under a preset, not what the architecture requires — loading a checkpoint while
the wrong preset is active overwrites it. Both that field and
`forge_checkpoint_<arch>` have been observed carrying another architecture's
values for exactly that reason.

Requirements are expressed as name patterns, never as exact filenames. A single
architecture is commonly served by several legitimate builds — quantised,
rescaled, or community-improved — and pinning filenames would reject all but
one of them.

Extras are the other direction: modules a setup may legitimately keep under a
preset although the architecture does not ask for them, such as the files
sam-extra's Anima 3.8B runtime works with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VAE = "vae"
TEXT_ENCODER = "text_encoder"


@dataclass(frozen=True)
class ModuleRequirement:
    kind: str
    label: str
    patterns: tuple[str, ...]
    optional: bool = False
    # For extras: what the module is for, reported next to it.
    note: str = ""

    def matches(self, name: str) -> bool:
        """Substring match, or anchored with "^" where a substring is too loose.

        "ae" is the Flux VAE's actual filename and also sits inside every other
        "*vae*" on disk, so it has to be anchored or it claims all of them.
        """
        lowered = name.lower()
        for pattern in self.patterns:
            if pattern.startswith("^"):
                if lowered.startswith(pattern[1:]):
                    return True
            elif pattern in lowered:
                return True
        return False


@dataclass(frozen=True)
class ArchModules:
    """What an architecture needs loaded alongside its checkpoint."""

    requirements: tuple[ModuleRequirement, ...] = field(default_factory=tuple)
    # Every diffusion model needs a VAE to turn latents into pixels; what varies
    # is whether it ships inside the checkpoint or as a separate file. This flag
    # means the reference does not name which external VAE belongs here - never
    # that one is unnecessary. Reported rather than guessed, since a mismatched
    # VAE degrades output without raising anything.
    vae_unspecified: bool = False
    note: str = ""
    # Recognised add-ons: never reported missing, never called unrecognised.
    extras: tuple[ModuleRequirement, ...] = field(default_factory=tuple)


# Architectures whose text encoder ships inside the checkpoint; a VAE can be
# supplied but the model loads without one.
_SELF_CONTAINED = ArchModules(note="text encoder is baked into the checkpoint; a VAE is optional")

# Shared building blocks, since several architectures reuse the same files.
_FLUX_VAE = ModuleRequirement(VAE, "Flux VAE (ae)", ("^ae.", "^ae_", "flux"))
_QWEN_IMAGE_VAE = ModuleRequirement(
    VAE, "Qwen-Image VAE", ("qwen_image_vae", "qwenimagevae", "qwen2d", "vaeanima")
)

# sam-extra's Anima 3.8B (sam3ext/anima38): markers as in sam3ext/anima38/files.py.
_ANIMA38_QWEN35 = ModuleRequirement(
    TEXT_ENCODER,
    "Qwen3.5 4B (sam-extra Anima 3.8B)",
    ("qwen35_4b", "qwen3.5-4b", "qwen3_5_4b"),
    optional=True,
    note=(
        "sam-extra's Anima 3.8B finds this file in models/text_encoder by itself, so it does not "
        "need to be selected here; when it is, sam-extra's loader filter skips it instead of letting "
        "Forge read and discard it on every model load. Harmless either way"
    ),
)
_ANIMA38_ADAPTER = ModuleRequirement(
    TEXT_ENCODER,
    "Anima 3.8B expanded adapter (sam-extra, v1 path)",
    ("anima-3.8b", "anima_3.8b", "anima_3_8b", "anima-3_8b"),
    optional=True,
    note=(
        "the separate adapter of the older Anima 3.8B v1 path, chosen in sam-extra's Anima 3.8B "
        "accordion; v1.1 bundles carry their connector inside the checkpoint. Selecting it here does "
        "nothing on its own"
    ),
)

ARCH_MODULES: dict[str, ArchModules] = {
    "sd": ArchModules(
        requirements=(
            ModuleRequirement(VAE, "SD1 VAE (vae-ft-mse-840000)", ("vae-ft-mse", "vae_ft_mse")),
        ),
        note="the reference lists a VAE and marks the text encoder as N/A",
    ),
    "xl": ArchModules(
        requirements=(
            ModuleRequirement(VAE, "SDXL VAE (sdxl-vae-fp16-fix)", ("sdxl-vae", "sdxl_vae")),
        ),
        note=(
            "applies to every SDXL lineage - Pony, Illustrious/NoobAI, Animagine and stock SDXL "
            "all run under this preset and share its VAE. They differ in prompt dialect, not in "
            "modules. The reference marks the text encoder N/A"
        ),
    ),
    "lumina": ArchModules(
        requirements=(
            _FLUX_VAE,
            ModuleRequirement(TEXT_ENCODER, "Gemma 2 2B", ("gemma_2_2b", "gemma2_2b", "gemma-2-2b")),
        ),
        note="Neta-Lumina and NetaYume-Lumina share these",
    ),
    "flux": ArchModules(
        requirements=(
            _FLUX_VAE,
            ModuleRequirement(TEXT_ENCODER, "CLIP-L", ("clip_l", "clip-l")),
            ModuleRequirement(TEXT_ENCODER, "T5-XXL", ("t5xxl", "t5_xxl", "t5-xxl")),
        ),
        note="Flux-Dev and Flux-Kontext share these",
    ),
    "klein": ArchModules(
        requirements=(
            ModuleRequirement(
                VAE, "Flux VAE (4B) or Flux.2 VAE (9B)",
                ("^ae.", "^ae_", "flux", "small_decoder"),
            ),
            ModuleRequirement(
                TEXT_ENCODER, "Qwen3 4B (Klein 4B) or Qwen3 8B (Klein 9B)",
                ("qwen_3_4b", "qwen3_4b", "qwen3-4b", "qwen_3_8b", "qwen3_8b", "qwen3-8b"),
            ),
        ),
        note=(
            "Flux.2-Klein ships in two sizes with different companions: 4B pairs Qwen3 4B with "
            "the Flux ae VAE, 9B pairs Qwen3 8B with flux2-vae plus small_decoder"
        ),
    ),
    "zit": ArchModules(
        requirements=(
            _FLUX_VAE,
            ModuleRequirement(TEXT_ENCODER, "Qwen3 4B", ("qwen_3_4b", "qwen3_4b", "qwen3-4b")),
        ),
        note="Z-Image and Z-Image-Turbo share these",
    ),
    "ernie": ArchModules(
        requirements=(
            ModuleRequirement(VAE, "Flux.2 VAE", ("flux2-vae", "flux2_vae", "flux", "small_decoder")),
            ModuleRequirement(TEXT_ENCODER, "Ministral 3 3B", ("ministral-3-3b", "ministral_3_3b", "ministral")),
        ),
        note="shares the Flux.2 VAE cell with Flux.2-Klein 9B in the source table",
    ),
    "wan": ArchModules(
        requirements=(
            ModuleRequirement(VAE, "Wan 2.1 VAE", ("wan_2.1_vae", "wan21_vae", "wan_vae")),
            ModuleRequirement(TEXT_ENCODER, "UMT5-XXL", ("umt5", "um_t5")),
        ),
        note="T2V and I2V share these; the DiT comes in high-noise and low-noise halves",
    ),
    "qwen": ArchModules(
        requirements=(
            _QWEN_IMAGE_VAE,
            ModuleRequirement(TEXT_ENCODER, "Qwen2.5-VL 7B", ("qwen_2.5_vl", "qwen2.5_vl", "qwen25vl")),
        ),
        note="Qwen-Image and Qwen-Image-Edit share these; the GGUF encoder also needs its mmproj file",
    ),
    "anima": ArchModules(
        requirements=(
            _QWEN_IMAGE_VAE,
            ModuleRequirement(TEXT_ENCODER, "Qwen3 0.6B", ("qwen_3_06b", "qwen3_06b", "qwen_3_0.6b")),
        ),
        note=(
            "Anima Base, 2.9B and 3.8B share these. sam-extra's Anima 3.8B adds Qwen3.5 4B on top of "
            "the Qwen3 0.6B encoder rather than replacing it"
        ),
        extras=(_ANIMA38_QWEN35, _ANIMA38_ADAPTER),
    ),
    "krea": ArchModules(
        requirements=(
            _QWEN_IMAGE_VAE,
            ModuleRequirement(TEXT_ENCODER, "Qwen3-VL 4B", ("qwen3vl_4b", "qwen3_vl_4b", "qwen3vl")),
        ),
    ),
    "pid": ArchModules(
        requirements=(
            ModuleRequirement(TEXT_ENCODER, "Gemma 2 2B IT", ("gemma_2_2b_it", "gemma2_2b_it", "gemma_2_2b")),
        ),
        vae_unspecified=True,
        note=(
            'a VAE is still required - the wiki lists it as "Model Dependent", meaning the '
            "checkpoint decides which one, so it has to be confirmed rather than assumed"
        ),
    ),
}


def requirements_for(arch: str | None) -> ArchModules | None:
    if not arch:
        return None
    return ARCH_MODULES.get(arch.lower())


def classify(filename: str) -> str | None:
    """VAE or text encoder, from the folder Forge keeps it in."""
    lowered = str(filename or "").replace("\\", "/").lower()
    if "/text_encoder" in lowered:
        return TEXT_ENCODER
    if "/vae" in lowered:
        return VAE
    return None


def audit(arch: str | None, selected: list[str], available: list[dict]) -> dict:
    """Compare what a preset has loaded against what its architecture needs.

    `available` is the /sdapi/v1/sd-modules listing. Returns what is satisfied,
    what is missing, and which installed files could fill each gap — candidates
    to offer, not a choice to make: several builds of the same module are all
    valid, and picking between them is the operator's call.
    """
    spec = requirements_for(arch)
    if spec is None:
        return {"known": False, "architecture": arch}

    pool = [
        {"name": str(item.get("model_name") or ""), "kind": classify(str(item.get("filename") or ""))}
        for item in available
        if isinstance(item, dict)
    ]
    selected_names = [str(name).replace("\\", "/").split("/")[-1] for name in selected]

    # Extras are claimed first, so a recognised add-on cannot pass as a requirement
    # (Qwen3.5 4B must not satisfy the Qwen3 0.6B encoder slot) nor be called a leftover.
    extras_loaded = []
    claimed: set[str] = set()
    for name in selected_names:
        extra = next((item for item in spec.extras if item.matches(name)), None)
        if extra is not None:
            claimed.add(name)
            extras_loaded.append({"module": name, "what": extra.label, "kind": extra.kind, "note": extra.note})
    candidates_selected = [name for name in selected_names if name not in claimed]

    satisfied, missing, optional_gaps = [], [], []
    for requirement in spec.requirements:
        hit = next((name for name in candidates_selected if requirement.matches(name)), None)
        if hit:
            satisfied.append({"need": requirement.label, "kind": requirement.kind, "loaded": hit})
            continue
        candidates = [
            item["name"]
            for item in pool
            if item["kind"] == requirement.kind
            and requirement.matches(item["name"])
            and not any(extra.matches(item["name"]) for extra in spec.extras)
        ]
        entry = {
            "need": requirement.label,
            "kind": requirement.kind,
            "candidates_installed": candidates,
            "action": "select one" if candidates else "download one",
        }
        if requirement.optional:
            # Absent is fine here - SD1 and SDXL load without an external VAE.
            entry["optional"] = True
            optional_gaps.append(entry)
        else:
            missing.append(entry)

    report = {
        "known": True,
        "architecture": arch,
        "satisfied": satisfied,
        "missing": missing,
        "optional_not_loaded": optional_gaps,
        "currently_selected": selected_names,
    }
    if extras_loaded:
        report["extras_loaded"] = extras_loaded
    if spec.note:
        report["note"] = spec.note
    if spec.vae_unspecified:
        vaes = [item["name"] for item in pool if item["kind"] == VAE]
        report["vae"] = {
            "status": "required, but the reference does not name which one for this architecture",
            "installed_vaes": vaes,
            "advice": "ask the operator which VAE they use here rather than assuming one",
        }
    # With no VAE requirement to compare against, a selected VAE cannot be
    # called wrong - the reference simply does not say what belongs here.
    kinds_specified = {requirement.kind for requirement in spec.requirements}
    selected_kinds = {
        item["name"]: item["kind"]
        for item in pool
        if item["name"] in selected_names
    }
    # A module absent from the listing has no known kind, and guessing one
    # produces false accusations. Only flag what can actually be classified.
    unexpected = [
        name
        for name in candidates_selected
        if not any(requirement.matches(name) for requirement in spec.requirements)
        and selected_kinds.get(name) in kinds_specified
    ]
    if unexpected and spec.requirements:
        report["unrecognised"] = {
            "modules": unexpected,
            "why": (
                "these match nothing this architecture asks for. Two innocent explanations exist "
                "before assuming a mistake: a community build named outside the usual convention "
                "is still perfectly valid, and matching is by name. The one worth checking is "
                "that Forge records the last selection made under a preset, so loading a "
                "checkpoint while another preset was active leaves that preset's modules here"
            ),
            "check": "confirm with the operator rather than treating this as an error",
        }
    return report
