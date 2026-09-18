"""Recipe constants and pure rules for Feature 6 (Anima Character Reference).

Values follow the public Anima ReStyler v1.2 workflow (animaRestyler_v12.json,
civitai model 2803070).  Nothing here imports Forge or Gradio, so every rule
the simplified panel relies on can be tested without a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

RECIPE_OUTPUT_SIZE: tuple[int, int] = (960, 1088)

KEEP_IDENTITY = "identity"
KEEP_OUTFIT = "outfit"
KEEP_STYLE = "style"
KEEP_SCOPES: tuple[str, ...] = (KEEP_IDENTITY, KEEP_OUTFIT, KEEP_STYLE)
# Provisional until the GPU A/B; the ReStyler v1.2 example image also carries
# a second AnimeEditV2 entry at 0.8.
OUTFIT_EDIT_STRENGTH = 0.8

FILL_ORIGINAL = "original"
FILL_SMEAR = "smear"

REFERENCE_OUTPUT_MARKER = "SAM3 Feature: 6"

WARN_WILDCARD = "와일드카드/동적 프롬프트 문법은 캐릭터 레퍼런스에서 처리되지 않습니다"
WARN_NEGPIP = "음수 가중치(negpip)는 캐릭터 레퍼런스에서 처리되지 않습니다"
WARN_LBW = "LoRA Block Weight(lbw=) 문법은 캐릭터 레퍼런스에서 처리되지 않습니다"
WARN_EXTEND_MISSING = "Extend LoRA를 찾지 못해 빼고 진행했습니다"

_LORA_FILE_SUFFIXES = (".safetensors", ".pt", ".ckpt")
_LORA_TOKEN = re.compile(r"<(?:lora|lyco):([^:>]+)(?::[^>]*)?>", re.IGNORECASE)
_PREFIX_GROUP = re.compile(
    r"\(\s*(?:split screen|multiple views)"
    r"(?:\s*,\s*(?:split screen|multiple views))*"
    r"\s*(?::\s*-?[\d.]+)?\s*\)",
    re.IGNORECASE,
)
_WILDCARD = re.compile(r"__[^\s_][^\s]*?__|\{[^{}]*\|[^{}]*\}")
_NEGATIVE_WEIGHT = re.compile(r":\s*-\d")
_LBW = re.compile(r"<(?:lora|lyco):[^>]*:lbw=", re.IGNORECASE)


@dataclass(frozen=True)
class DetectedLoras:
    edit: str | None
    extend: str | None


@dataclass(frozen=True)
class SamplingSettings:
    sampler: str = "Euler a"
    scheduler: str = "Simple"
    steps: int = 30
    cfg_scale: float = 5.0
    shift: float = 3.0


def lora_stem(name: str) -> str:
    """Return a LoRA name without folders or a real model-file extension."""

    base = str(name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    lowered = base.casefold()
    for suffix in _LORA_FILE_SUFFIXES:
        if lowered.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _lora_key(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", lora_stem(name)).casefold()


def is_edit_lora(name: str) -> bool:
    key = _lora_key(name)
    return ("animeedit" in key or "animaedit" in key) and "extend" not in key


def is_extend_lora(name: str) -> bool:
    return "extendimage" in _lora_key(name)


def detect_reference_loras(available: Iterable[str]) -> DetectedLoras:
    names = sorted(
        {str(name).strip() for name in (available or ()) if str(name).strip()},
        key=str.casefold,
    )
    edits = [name for name in names if is_edit_lora(name)]
    exact = [name for name in edits if _lora_key(name) == "animeeditv2"]
    edit = (exact or edits or [None])[0]
    extend = next((name for name in names if is_extend_lora(name)), None)
    return DetectedLoras(edit=edit, extend=extend)


def _tidy(text: str) -> str:
    text = re.sub(r"\s*,(?:\s*,)+", ",", text)
    return text.strip(" ,\t\r\n")


def strip_reference_tokens(prompt: str) -> str:
    """Drop Feature 6's own LoRA tokens and split-screen prefix from a prompt."""

    def keep_unless_reference(match: re.Match) -> str:
        name = match.group(1)
        return "" if is_edit_lora(name) or is_extend_lora(name) else match.group(0)

    text = _LORA_TOKEN.sub(keep_unless_reference, str(prompt or ""))
    text = _PREFIX_GROUP.sub("", text)
    return _tidy(text)


def strip_lora_tokens(prompt: str) -> str:
    return _tidy(_LORA_TOKEN.sub("", str(prompt or "")))


def prepare_inherited_prompt(prompt: str, keep_scope: str) -> str:
    text = strip_reference_tokens(prompt)
    if keep_scope == KEEP_STYLE:
        text = strip_lora_tokens(text)
    return text


def prompt_syntax_warnings(prompt: str) -> list[str]:
    text = str(prompt or "")
    warnings: list[str] = []
    if _WILDCARD.search(text):
        warnings.append(WARN_WILDCARD)
    if _NEGATIVE_WEIGHT.search(text):
        warnings.append(WARN_NEGPIP)
    if _LBW.search(text):
        warnings.append(WARN_LBW)
    return warnings


def edit_strength_for_scope(keep_scope: str, base_strength: float) -> float:
    base = float(base_strength)
    if keep_scope in (KEEP_OUTFIT, KEEP_STYLE):
        return max(base, OUTFIT_EDIT_STRENGTH)
    return base


def resolve_output_size(
    custom_enabled: bool,
    custom_width: int,
    custom_height: int,
    txt2img_width: Any,
    txt2img_height: Any,
) -> tuple[int, int]:
    if custom_enabled:
        return int(custom_width), int(custom_height)
    try:
        width = int(float(txt2img_width))
        height = int(float(txt2img_height))
    except (TypeError, ValueError):
        return RECIPE_OUTPUT_SIZE
    if not (64 <= width <= 8192 and 64 <= height <= 8192):
        return RECIPE_OUTPUT_SIZE
    return width, height


def fill_settings(fill_mode: str, color: str) -> tuple[str, str]:
    """Map the panel's fill choice to Forge masked content and panel colour.

    ``original`` keeps the solid panel pixels, so the reference latent is the
    same [reference | solid colour] canvas the ReStyler workflow encodes.
    """

    colour = str(color or "").strip() or "#000000"
    if fill_mode == FILL_SMEAR:
        return "latent noise", colour
    return "original", colour


def forge_preset_sampling(
    opts_data: Mapping[str, Any],
    fallback: SamplingSettings,
) -> SamplingSettings:
    def pick(key: str, cast, default):
        value = opts_data.get(key) if opts_data else None
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    return SamplingSettings(
        sampler=pick("anima_i2i_sampler", str, fallback.sampler),
        scheduler=pick("anima_i2i_scheduler", str, fallback.scheduler),
        steps=pick("anima_i2i_step", int, fallback.steps),
        cfg_scale=pick("anima_i2i_cfg", float, fallback.cfg_scale),
        shift=pick("anima_i2i_dcfg", float, fallback.shift),
    )


def is_reference_output(infotext: str) -> bool:
    return REFERENCE_OUTPUT_MARKER in str(infotext or "")


def choose_fallback_index(
    item_count: int,
    selected_index: Any,
    infotexts: Sequence[str],
) -> int:
    """Pick the gallery image to use when no reference was uploaded.

    The selected image wins, else the last one; earlier Feature 6 results are
    skipped so a second click never uses its own output as the reference.
    """

    infos = list(infotexts or [])

    def usable(index: int) -> bool:
        if not 0 <= index < item_count:
            return False
        info = infos[index] if index < len(infos) else ""
        return not is_reference_output(info)

    try:
        selected = int(selected_index)
    except (TypeError, ValueError):
        selected = -1
    if usable(selected):
        return selected
    for index in range(item_count - 1, -1, -1):
        if usable(index):
            return index
    return -1
