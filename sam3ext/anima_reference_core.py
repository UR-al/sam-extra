"""Pure image preparation for the Anima Character Reference workflow.

This module deliberately does not import Forge or Gradio.  Its interface is
small: prepare a split-screen canvas, crop the generated target region, and
compose the internal prompt.  Forge-specific sampling lives in the adapter
module so geometry can be tested without loading a diffusion model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageColor


Placement = Literal["right", "left"]
ResampleName = Literal["nearest", "bilinear", "bicubic", "lanczos"]

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class ReferenceCanvasConfig:
    """Every geometric value exposed by the Feature 6 UI."""

    output_width: int = 960
    output_height: int = 1088
    placement: Placement = "right"
    target_region_scale: float = 1.0
    target_color: str = "#000000"
    reference_matte_color: str = "#ffffff"
    composite_megapixels: float = 1.4
    dimension_multiple: int = 16
    resize_filter: ResampleName = "lanczos"
    mask_overlap: int = 0

    def validate(self) -> "ReferenceCanvasConfig":
        if not 64 <= int(self.output_width) <= 8192:
            raise ValueError("Output width must be between 64 and 8192.")
        if not 64 <= int(self.output_height) <= 8192:
            raise ValueError("Output height must be between 64 and 8192.")
        if self.placement not in ("right", "left"):
            raise ValueError("Placement must be 'right' or 'left'.")
        if not 0.05 <= float(self.target_region_scale) <= 8.0:
            raise ValueError("Target region scale must be between 0.05 and 8.0.")
        if not 0.0 <= float(self.composite_megapixels) <= 64.0:
            raise ValueError("Composite megapixels must be between 0 and 64.")
        if not 1 <= int(self.dimension_multiple) <= 256:
            raise ValueError("Dimension multiple must be between 1 and 256.")
        if self.resize_filter not in _RESAMPLING:
            raise ValueError(f"Unsupported resize filter: {self.resize_filter!r}")
        if not 0 <= int(self.mask_overlap) <= 4096:
            raise ValueError("Mask overlap must be between 0 and 4096.")
        # Validate both colors now so the handler fails before starting a job.
        ImageColor.getrgb(str(self.target_color))
        ImageColor.getrgb(str(self.reference_matte_color))
        return self


@dataclass(frozen=True)
class PreparedReferenceCanvas:
    canvas: Image.Image
    mask: Image.Image
    reference_box: tuple[int, int, int, int]
    target_box: tuple[int, int, int, int]
    mask_box: tuple[int, int, int, int]
    output_size: tuple[int, int]
    placement: Placement


def _round_to_multiple(value: float, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _flatten_rgb(image: Image.Image, matte_color: str) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        matte = Image.new("RGBA", rgba.size, ImageColor.getrgb(matte_color) + (255,))
        matte.alpha_composite(rgba)
        return matte.convert("RGB")
    return image.convert("RGB")


def prepare_reference_canvas(
    reference: Image.Image,
    config: ReferenceCanvasConfig,
) -> PreparedReferenceCanvas:
    """Create the split-screen init image and an exact rectangular mask.

    ``composite_megapixels=0`` preserves approximately the source height.
    Otherwise both panels are sized from the requested total pixel budget.
    All dimensions are rounded to ``dimension_multiple`` for Anima/Qwen VAE.
    """

    config = config.validate()
    if reference is None or reference.width < 1 or reference.height < 1:
        raise ValueError("A non-empty reference image is required.")

    source_ratio = reference.width / reference.height
    target_ratio = (
        config.output_width
        / config.output_height
        * float(config.target_region_scale)
    )
    ratio_sum = source_ratio + target_ratio
    multiple = int(config.dimension_multiple)

    if config.composite_megapixels > 0:
        height = math.sqrt(config.composite_megapixels * 1_000_000 / ratio_sum)
    else:
        height = float(reference.height)

    canvas_h = _round_to_multiple(height, multiple)
    reference_w = _round_to_multiple(canvas_h * source_ratio, multiple)
    target_w = _round_to_multiple(canvas_h * target_ratio, multiple)
    canvas_w = reference_w + target_w

    source = _flatten_rgb(reference, config.reference_matte_color).resize(
        (reference_w, canvas_h),
        _RESAMPLING[config.resize_filter],
    )
    target = Image.new(
        "RGB",
        (target_w, canvas_h),
        ImageColor.getrgb(config.target_color),
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h))
    mask = Image.new("L", (canvas_w, canvas_h), 0)

    overlap = min(int(config.mask_overlap), reference_w)
    if config.placement == "right":
        reference_box = (0, 0, reference_w, canvas_h)
        target_box = (reference_w, 0, canvas_w, canvas_h)
        mask_box = (reference_w - overlap, 0, canvas_w, canvas_h)
        canvas.paste(source, (0, 0))
        canvas.paste(target, (reference_w, 0))
    else:
        target_box = (0, 0, target_w, canvas_h)
        reference_box = (target_w, 0, canvas_w, canvas_h)
        mask_box = (0, 0, target_w + overlap, canvas_h)
        canvas.paste(target, (0, 0))
        canvas.paste(source, (target_w, 0))

    mask.paste(255, mask_box)
    return PreparedReferenceCanvas(
        canvas=canvas,
        mask=mask,
        reference_box=reference_box,
        target_box=target_box,
        mask_box=mask_box,
        output_size=(int(config.output_width), int(config.output_height)),
        placement=config.placement,
    )


def crop_reference_result(
    generated: Image.Image,
    prepared: PreparedReferenceCanvas,
    *,
    resize_filter: ResampleName = "lanczos",
) -> Image.Image:
    """Crop only the generated panel and resize to the exact requested size."""

    if generated is None or generated.width < 1 or generated.height < 1:
        raise ValueError("A non-empty generated image is required.")
    if resize_filter not in _RESAMPLING:
        raise ValueError(f"Unsupported resize filter: {resize_filter!r}")

    source_w, source_h = prepared.canvas.size
    scale_x = generated.width / source_w
    scale_y = generated.height / source_h
    left, top, right, bottom = prepared.target_box
    crop_box = (
        max(0, int(round(left * scale_x))),
        max(0, int(round(top * scale_y))),
        min(generated.width, int(round(right * scale_x))),
        min(generated.height, int(round(bottom * scale_y))),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError("Generated image does not contain a valid target region.")

    return generated.convert("RGB").crop(crop_box).resize(
        prepared.output_size,
        _RESAMPLING[resize_filter],
    )


def _lora_token(name: str, strength: float) -> str:
    cleaned = str(name or "").strip().replace("\\", "/")
    if not cleaned:
        return ""
    # Forge accepts relative subdirectories but the extension must be omitted.
    cleaned = str(Path(cleaned).with_suffix("")).replace("\\", "/")
    return f"<lora:{cleaned}:{float(strength):g}>"


def compose_reference_prompt(
    prompt: str,
    *,
    prefix_enabled: bool = True,
    prefix_text: str = "split screen, multiple views",
    prefix_strength: float = 1.2,
    edit_lora_enabled: bool = True,
    edit_lora_name: str = "",
    edit_lora_strength: float = 0.72,
    extend_lora_enabled: bool = True,
    extend_lora_name: str = "",
    extend_lora_strength: float = 0.4,
    extra_prefix: str = "",
    extra_suffix: str = "",
) -> str:
    """Build the internal prompt without mutating the visible txt2img prompt."""

    parts: list[str] = []
    if edit_lora_enabled and edit_lora_name:
        parts.append(_lora_token(edit_lora_name, edit_lora_strength))
    if extend_lora_enabled and extend_lora_name:
        parts.append(_lora_token(extend_lora_name, extend_lora_strength))
    if extra_prefix.strip():
        parts.append(extra_prefix.strip())
    if prefix_enabled and prefix_text.strip():
        parts.append(f"({prefix_text.strip()}:{float(prefix_strength):g})")
    if str(prompt or "").strip():
        parts.append(str(prompt).strip())
    if extra_suffix.strip():
        parts.append(extra_suffix.strip())
    return ", ".join(part for part in parts if part)
