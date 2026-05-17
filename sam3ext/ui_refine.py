"""Post-generation Refine panel — lives under the t2i gallery and runs SAM3
inpaint (optionally with a ControlNet unit) against the currently selected
gallery image, appending the result back into the same gallery."""
from __future__ import annotations

import io
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from .ui import (
    _controlnet_model_choices,
    _controlnet_module_choices,
    _default_cn_module,
)


@dataclass
class RefinePanel:
    """Container that exposes the panel's components to the script for
    later wiring (we wire the click handler in scripts/!sam3.py because the
    runtime callable depends on Forge's shared state)."""

    accordion: gr.Accordion
    selected_index_state: gr.Number  # hidden frontend slot — JS shim fills it
    detect_prompt: gr.Textbox
    inpaint_prompt: gr.Textbox
    negative_prompt: gr.Textbox
    inherit_main_prompt: gr.Checkbox
    inherit_main_neg_prompt: gr.Checkbox
    prompt_sr: gr.Textbox
    threshold: gr.Slider
    mask_dilation: gr.Slider
    mask_hull: gr.Checkbox
    mask_blur: gr.Slider
    unload_after: gr.Checkbox
    denoising_strength: gr.Slider
    inpaint_only_masked: gr.Checkbox
    inpaint_only_masked_padding: gr.Slider
    steps: gr.Slider
    cfg_scale: gr.Slider
    sampler: gr.Dropdown
    scheduler: gr.Dropdown
    checkpoint: gr.Dropdown
    mask_mode: gr.Radio
    cn_enable: gr.Checkbox
    cn_override_external: gr.Checkbox
    cn_pixel_perfect: gr.Checkbox
    cn_model: gr.Dropdown
    cn_module: gr.Dropdown
    cn_weight: gr.Slider
    cn_guidance_start: gr.Slider
    cn_guidance_end: gr.Slider
    cn_control_mode: gr.Radio
    cn_resize_mode: gr.Radio
    cn_processor_res: gr.Slider
    cn_threshold_a: gr.Slider
    cn_threshold_b: gr.Slider
    insert_mode: gr.Radio
    refine_button: gr.Button
    status: gr.HTML

    def all_widgets(self) -> list:
        """Ordered list of input widgets — must match ``REFINE_ARG_KEYS``."""
        return [
            self.detect_prompt,
            self.inpaint_prompt,
            self.negative_prompt,
            self.inherit_main_prompt,
            self.inherit_main_neg_prompt,
            self.prompt_sr,
            self.threshold,
            self.mask_dilation,
            self.mask_hull,
            self.mask_blur,
            self.unload_after,
            self.denoising_strength,
            self.inpaint_only_masked,
            self.inpaint_only_masked_padding,
            self.steps,
            self.cfg_scale,
            self.sampler,
            self.scheduler,
            self.checkpoint,
            self.mask_mode,
            self.cn_enable,
            self.cn_override_external,
            self.cn_pixel_perfect,
            self.cn_model,
            self.cn_module,
            self.cn_weight,
            self.cn_guidance_start,
            self.cn_guidance_end,
            self.cn_control_mode,
            self.cn_resize_mode,
            self.cn_processor_res,
            self.cn_threshold_a,
            self.cn_threshold_b,
            self.insert_mode,
        ]


REFINE_ARG_KEYS = (
    "detect_prompt",
    "inpaint_prompt",
    "negative_prompt",
    "inherit_main_prompt",
    "inherit_main_neg_prompt",
    "prompt_sr",
    "threshold",
    "mask_dilation",
    "mask_hull",
    "mask_blur",
    "unload_after",
    "denoising_strength",
    "inpaint_only_masked",
    "inpaint_only_masked_padding",
    "steps",
    "cfg_scale",
    "sampler",
    "scheduler",
    "checkpoint",
    "mask_mode",
    "cn_enable",
    "cn_override_external",
    "cn_pixel_perfect",
    "cn_model",
    "cn_module",
    "cn_weight",
    "cn_guidance_start",
    "cn_guidance_end",
    "cn_control_mode",
    "cn_resize_mode",
    "cn_processor_res",
    "cn_threshold_a",
    "cn_threshold_b",
    "insert_mode",
)


def build_refine_panel(
    samplers: list[str],
    schedulers: list[str],
    checkpoint_choices: list[str],
) -> RefinePanel:
    """Render the Refine accordion. Must be called inside an open
    ``gr.Blocks`` context that is a sibling of ``txt2img_gallery``."""

    cn_models = _controlnet_model_choices()
    cn_modules = _controlnet_module_choices()
    cn_module_default = _default_cn_module(cn_modules)

    with gr.Accordion("SAM3 Refine (post-generation)", open=False, elem_id="sam3_refine_panel") as acc:
        # Hidden Number (not gr.State) so the `_js` shim on the Refine button
        # can address this slot reliably: Gradio's _js handler only receives
        # frontend components in its args array — gr.State is server-side and
        # would shift every subsequent arg by one if we tried to overwrite it.
        selected_index_state = gr.Number(value=-1, precision=0, visible=False, elem_id="sam3_refine_selected_index")
        gr.Markdown(
            "Pick an image in the gallery above, then enter prompts and click **Refine**. "
            "The result is inserted next to the selected image — chain refines by reselecting."
        )

        with gr.Row():
            detect_prompt = gr.Textbox(
                value="",
                label="Detect Prompt",
                lines=1,
                placeholder="e.g. shirt, hair, face — what SAM3 should mask in the selected image",
            )
        with gr.Row():
            inpaint_prompt = gr.Textbox(
                value="",
                label="Inpaint Prompt",
                lines=2,
                placeholder="What to draw inside the mask (e.g. 'red leather jacket')",
            )
        with gr.Row():
            negative_prompt = gr.Textbox(
                value="",
                label="Negative Prompt",
                lines=1,
                placeholder="Optional",
            )

        with gr.Row():
            inherit_main_prompt = gr.Checkbox(
                label="Inherit main t2i prompt (carries LoRAs / style triggers — recommended)",
                value=True,
            )
            inherit_main_neg_prompt = gr.Checkbox(
                label="Inherit main t2i negative prompt",
                value=True,
            )

        with gr.Row():
            prompt_sr = gr.Textbox(
                value="",
                label="Prompt S/R (applied to inherited main prompt + negative before merge)",
                lines=2,
                placeholder=(
                    "one rule per line, pattern=replacement; empty replacement = delete.\n"
                    "example:  shirt=    (removes 'shirt' from inherited main so 'nude' in refine prompt isn't shouted down)"
                ),
            )

        with gr.Row():
            threshold = gr.Slider(label="SAM3 Threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.4)
            mask_dilation = gr.Slider(label="Mask Dilation (px)", minimum=0, maximum=256, step=1, value=4)
            mask_blur = gr.Slider(label="Mask Blur", minimum=0, maximum=64, step=1, value=4)
            mask_mode = gr.Radio(
                label="Mask Processing",
                choices=["Individual", "Combined"],
                value="Combined",
            )

        with gr.Row():
            mask_hull = gr.Checkbox(
                label="Convex Hull (wrap strands — recommended for hair/fur)",
                value=False,
            )
            unload_after = gr.Checkbox(
                label="Unload SAM3 from VRAM after detection (~3.5 GB — recommended for ≤12 GB GPUs)",
                value=False,
            )

        with gr.Row():
            denoising_strength = gr.Slider(
                label="Denoising Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.75
            )
            inpaint_only_masked = gr.Checkbox(label="Inpaint only masked", value=False)
            inpaint_only_masked_padding = gr.Slider(
                label="Inpaint padding", minimum=0, maximum=256, step=1, value=32
            )

        with gr.Row():
            steps = gr.Slider(label="Steps", minimum=1, maximum=150, step=1, value=28)
            cfg_scale = gr.Slider(label="CFG Scale", minimum=0.0, maximum=30.0, step=0.1, value=7.0)
            sampler = gr.Dropdown(
                label="Sampler",
                choices=samplers or ["Euler a"],
                value=samplers[0] if samplers else "Euler a",
                type="value",
            )
            scheduler = gr.Dropdown(
                label="Scheduler",
                choices=schedulers or ["Automatic"],
                value=schedulers[0] if schedulers else "Automatic",
                type="value",
            )

        with gr.Row():
            checkpoint = gr.Dropdown(
                label="SAM3 Checkpoint",
                choices=checkpoint_choices,
                value=checkpoint_choices[0] if checkpoint_choices else "sam3.pt",
                type="value",
            )

        with gr.Accordion("ControlNet", open=False):
            with gr.Row():
                cn_enable = gr.Checkbox(label="Enable ControlNet", value=False)
                cn_override_external = gr.Checkbox(
                    label="Override external CN units", value=False
                )
                cn_pixel_perfect = gr.Checkbox(label="Pixel Perfect", value=True)
            with gr.Row():
                cn_module = gr.Dropdown(
                    label="Preprocessor",
                    choices=cn_modules,
                    value=cn_module_default,
                    type="value",
                )
                cn_model = gr.Dropdown(
                    label="Model",
                    choices=cn_models,
                    value=cn_models[0] if cn_models else "None",
                    type="value",
                )
            with gr.Row():
                cn_weight = gr.Slider(label="Weight", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
                cn_guidance_start = gr.Slider(
                    label="Guidance Start", minimum=0.0, maximum=1.0, step=0.01, value=0.0
                )
                cn_guidance_end = gr.Slider(
                    label="Guidance End", minimum=0.0, maximum=1.0, step=0.01, value=1.0
                )
            with gr.Row():
                cn_control_mode = gr.Radio(
                    label="Control Mode",
                    choices=[
                        "Balanced",
                        "My prompt is more important",
                        "ControlNet is more important",
                    ],
                    value="Balanced",
                )
                cn_resize_mode = gr.Radio(
                    label="Resize Mode",
                    choices=["Just Resize", "Crop and Resize", "Resize and Fill"],
                    value="Crop and Resize",
                )
            with gr.Row():
                cn_processor_res = gr.Slider(
                    label="Preprocessor Resolution",
                    minimum=64,
                    maximum=2048,
                    step=8,
                    value=512,
                )
                cn_threshold_a = gr.Slider(
                    label="Threshold A", minimum=-1, maximum=256, step=1, value=-1
                )
                cn_threshold_b = gr.Slider(
                    label="Threshold B", minimum=-1, maximum=256, step=1, value=-1
                )

        with gr.Row():
            insert_mode = gr.Radio(
                label="Insert result",
                choices=["After selected", "At end"],
                value="After selected",
            )
            refine_button = gr.Button("▶ Refine", variant="primary")

        status = gr.HTML(value="", elem_id="sam3_refine_status")

    return RefinePanel(
        accordion=acc,
        selected_index_state=selected_index_state,
        detect_prompt=detect_prompt,
        inpaint_prompt=inpaint_prompt,
        negative_prompt=negative_prompt,
        inherit_main_prompt=inherit_main_prompt,
        inherit_main_neg_prompt=inherit_main_neg_prompt,
        prompt_sr=prompt_sr,
        threshold=threshold,
        mask_dilation=mask_dilation,
        mask_hull=mask_hull,
        mask_blur=mask_blur,
        unload_after=unload_after,
        denoising_strength=denoising_strength,
        inpaint_only_masked=inpaint_only_masked,
        inpaint_only_masked_padding=inpaint_only_masked_padding,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        scheduler=scheduler,
        checkpoint=checkpoint,
        mask_mode=mask_mode,
        cn_enable=cn_enable,
        cn_override_external=cn_override_external,
        cn_pixel_perfect=cn_pixel_perfect,
        cn_model=cn_model,
        cn_module=cn_module,
        cn_weight=cn_weight,
        cn_guidance_start=cn_guidance_start,
        cn_guidance_end=cn_guidance_end,
        cn_control_mode=cn_control_mode,
        cn_resize_mode=cn_resize_mode,
        cn_processor_res=cn_processor_res,
        cn_threshold_a=cn_threshold_a,
        cn_threshold_b=cn_threshold_b,
        insert_mode=insert_mode,
        refine_button=refine_button,
        status=status,
    )


def _coerce_gallery_item_to_pil(item: Any) -> Image.Image | None:
    """Gallery items come in a few shapes depending on Gradio version: PIL,
    dict with ``name``/``path``/``data``, tuple ``(path, caption)``, or a
    bare path string."""
    if item is None:
        return None
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, np.ndarray):
        return Image.fromarray(item).convert("RGB")
    if isinstance(item, (tuple, list)):
        return _coerce_gallery_item_to_pil(item[0]) if item else None
    if isinstance(item, dict):
        for key in ("name", "path", "data"):
            value = item.get(key)
            if value:
                pil = _coerce_gallery_item_to_pil(value)
                if pil is not None:
                    return pil
        return None
    if isinstance(item, (str, os.PathLike)):
        path = str(item)
        if path.startswith("data:"):
            try:
                import base64

                header, _, payload = path.partition(",")
                return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
            except Exception:
                return None
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None


def _apply_prompt_sr(text: str, rules_field: str) -> str:
    """Apply ``pattern=replacement`` rules (one per line) to ``text`` and
    normalize the result.

    Empty replacement deletes the pattern. After all substitutions, runs of
    commas / leading-or-trailing commas / collapsed whitespace get cleaned
    up so the merged prompt reads naturally.

    Returns ``text`` unchanged when ``rules_field`` is empty / has no
    parseable ``=`` lines.
    """
    if not text or not rules_field:
        return text
    out = text
    matched_any = False
    for raw_line in rules_field.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        pattern, _, replacement = line.partition("=")
        pattern = pattern.strip()
        if not pattern:
            continue
        if pattern in out:
            out = out.replace(pattern, replacement)
            matched_any = True
    if not matched_any:
        return text
    # Normalize: collapse whitespace, then comma-spacing, then repeated commas
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s*,\s*", ", ", out)
    out = re.sub(r"(,\s*){2,}", ", ", out)
    return out.strip(" ,")


def _as_float(value: Any, default: float) -> float:
    """Bulletproof float coercion: empty string / None / non-numeric → default.

    Sliders return floats and dropdowns return their selected string, but
    widget-order quirks or browser-side surprises can put an empty textbox
    value where a number was expected. We previously raised loudly, but the
    user-facing failure mode (refine button silently broken) is worse than
    silently falling back to the slider/component default.
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return default
    try:
        return int(float(value))  # float-first so "12.0" works
    except (TypeError, ValueError):
        return default


def map_widget_values_to_sam3_args(values: tuple) -> dict[str, Any]:
    """Translate the Refine panel's widget values into the dict shape that
    ``inpaint_core.run_sam3_refine`` expects (same keys as ``Sam3Args``)."""
    keyed = dict(zip(REFINE_ARG_KEYS, values))
    # Defensive: log when a string sneaks into a numeric slot — usually a
    # widget-order regression. Doesn't raise (we coerce below with _as_float
    # so the refine still runs with default values for those fields).
    for k in (
        "threshold", "denoising_strength", "cfg_scale", "cn_weight",
        "cn_guidance_start", "cn_guidance_end", "cn_threshold_a", "cn_threshold_b",
        "mask_dilation", "mask_blur", "inpaint_only_masked_padding", "steps",
        "cn_processor_res",
    ):
        v = keyed.get(k)
        if isinstance(v, str) and v.strip() != "":
            print(
                f"[-] SAM3 Refine: numeric slot '{k}' got a string ({v[:60]!r}...) — "
                f"falling back to default; check REFINE_ARG_KEYS alignment.",
                file=sys.stderr,
            )
    return {
        # SAM3 detection
        "sam3_prompt": str(keyed.get("detect_prompt") or "").strip(),
        "sam3_inpaint_prompt": str(keyed.get("inpaint_prompt") or ""),
        "sam3_negative_prompt": str(keyed.get("negative_prompt") or ""),
        "sam3_threshold": _as_float(keyed.get("threshold"), 0.4),
        "sam3_mask_dilation": _as_int(keyed.get("mask_dilation"), 4),
        "sam3_mask_hull": bool(keyed.get("mask_hull", False)),
        "sam3_unload_after": bool(keyed.get("unload_after", False)),
        "sam3_checkpoint": str(keyed.get("checkpoint") or "sam3.pt"),
        "sam3_device": "auto",
        "sam3_mask_mode": str(keyed.get("mask_mode") or "Combined"),
        # Inpaint
        "sam3_mask_blur": _as_int(keyed.get("mask_blur"), 4),
        "sam3_denoising_strength": _as_float(keyed.get("denoising_strength"), 0.75),
        "sam3_inpaint_only_masked": bool(keyed.get("inpaint_only_masked", False)),
        "sam3_inpaint_only_masked_padding": _as_int(keyed.get("inpaint_only_masked_padding"), 32),
        "sam3_use_inpaint_width_height": False,
        "sam3_inpaint_width": 512,
        "sam3_inpaint_height": 512,
        "sam3_steps": _as_int(keyed.get("steps"), 28),
        "sam3_cfg_scale": _as_float(keyed.get("cfg_scale"), 7.0),
        "sam3_sampler": str(keyed.get("sampler") or "Euler a"),
        "sam3_scheduler": str(keyed.get("scheduler") or "Automatic"),
        "sam3_noise_multiplier": 1.0,
        "sam3_restore_face": False,
        # ControlNet
        "sam3_cn_enable": bool(keyed.get("cn_enable", False)),
        "sam3_cn_override_external": bool(keyed.get("cn_override_external", False)),
        "sam3_cn_pixel_perfect": bool(keyed.get("cn_pixel_perfect", True)),
        "sam3_cn_model": str(keyed.get("cn_model") or "None"),
        "sam3_cn_module": str(keyed.get("cn_module") or "inpaint_only"),
        "sam3_cn_weight": _as_float(keyed.get("cn_weight"), 1.0),
        "sam3_cn_guidance_start": _as_float(keyed.get("cn_guidance_start"), 0.0),
        "sam3_cn_guidance_end": _as_float(keyed.get("cn_guidance_end"), 1.0),
        "sam3_cn_control_mode": str(keyed.get("cn_control_mode") or "Balanced"),
        "sam3_cn_resize_mode": str(keyed.get("cn_resize_mode") or "Crop and Resize"),
        "sam3_cn_processor_res": _as_int(keyed.get("cn_processor_res"), 512),
        "sam3_cn_threshold_a": _as_float(keyed.get("cn_threshold_a"), -1.0),
        "sam3_cn_threshold_b": _as_float(keyed.get("cn_threshold_b"), -1.0),
        "_insert_mode": str(keyed.get("insert_mode") or "After selected"),
        "_inherit_main_prompt": bool(keyed.get("inherit_main_prompt", True)),
        "_inherit_main_neg_prompt": bool(keyed.get("inherit_main_neg_prompt", True)),
        "_prompt_sr": str(keyed.get("prompt_sr") or ""),
    }


def handle_refine_click(gallery_value, selected_index, *all_values):
    """Refine-button handler. Returns ``(updated_gallery, status_html)``.

    ``all_values`` = ``(*widget_values, main_prompt, main_neg_prompt)``. The
    two main-prompt slots come last because they are appended by the wiring
    in ``scripts/!sam3.py``; we use them as a fallback when the Refine
    panel's own inpaint/negative prompts are blank.

    Out-of-scope behaviors are returned as HTML status messages rather than
    raised exceptions so the panel stays responsive.
    """
    expected_widget_count = len(REFINE_ARG_KEYS)
    if len(all_values) < expected_widget_count:
        return gallery_value, "<span style='color:#c33'>SAM3 Refine: missing widget values.</span>"

    widget_values = all_values[:expected_widget_count]
    extras = all_values[expected_widget_count:]
    main_prompt = str(extras[0]) if len(extras) > 0 else ""
    main_neg_prompt = str(extras[1]) if len(extras) > 1 else ""

    args = map_widget_values_to_sam3_args(widget_values)
    insert_mode = args.pop("_insert_mode", "After selected")
    inherit_main = args.pop("_inherit_main_prompt", True)
    inherit_main_neg = args.pop("_inherit_main_neg_prompt", True)
    sr_rules = args.pop("_prompt_sr", "")

    # Strip user-specified tokens from the inherited main prompt(s) BEFORE
    # merging — lets the user drop e.g. "shirt" so a refine prompt of "nude"
    # isn't shouted down by the original main prompt's "shirt" token while
    # still keeping LoRAs, style triggers, and important anatomy context
    # ("2girls", "grabbing pectorals", etc.) intact. Same rules apply to
    # negative — useful when main negative has "nude" anti-trigger.
    cleaned_main = _apply_prompt_sr(main_prompt, sr_rules) if main_prompt else main_prompt
    cleaned_neg = _apply_prompt_sr(main_neg_prompt, sr_rules) if main_neg_prompt else main_neg_prompt

    # Prompt resolution:
    # - Inherit ON  + refine empty  -> main only (preserves fallback semantics)
    # - Inherit ON  + refine filled -> "main, refine"  (LoRAs / style triggers
    #                                                   in main carry over; the
    #                                                   refine prompt adds the
    #                                                   new subject description)
    # - Inherit OFF + refine empty  -> "" (use the model's unconditional default)
    # - Inherit OFF + refine filled -> refine only (clean override)
    refine_p = args.get("sam3_inpaint_prompt") or ""
    if inherit_main and cleaned_main:
        args["sam3_inpaint_prompt"] = f"{cleaned_main}, {refine_p}".rstrip(", ") if refine_p else cleaned_main
    refine_n = args.get("sam3_negative_prompt") or ""
    if inherit_main_neg and cleaned_neg:
        args["sam3_negative_prompt"] = f"{cleaned_neg}, {refine_n}".rstrip(", ") if refine_n else cleaned_neg

    if not args["sam3_prompt"]:
        return gallery_value, "<span style='color:#c33'>SAM3 Refine: enter a detect prompt first.</span>"

    gallery_list = list(gallery_value or [])
    if not gallery_list:
        return gallery_value, "<span style='color:#c33'>SAM3 Refine: gallery is empty.</span>"

    # Hidden Number arrives as float; JS shim defaults to -1 when nothing is
    # selected. Coerce + clamp to a valid index.
    try:
        idx = int(selected_index) if selected_index is not None else -1
    except (TypeError, ValueError):
        idx = -1
    if idx < 0 or idx >= len(gallery_list):
        idx = len(gallery_list) - 1  # default to last item

    image = _coerce_gallery_item_to_pil(gallery_list[idx])
    if image is None:
        return gallery_value, f"<span style='color:#c33'>SAM3 Refine: could not load selected image (index {idx}).</span>"

    from modules import shared as _shared

    sd_model = getattr(_shared, "sd_model", None)
    if sd_model is None:
        return gallery_value, "<span style='color:#c33'>SAM3 Refine: no SD model loaded.</span>"

    outpath_samples = getattr(_shared.opts, "outdir_txt2img_samples", "outputs/txt2img-images")
    outpath_grids = getattr(_shared.opts, "outdir_txt2img_grids", "outputs/txt2img-grids")

    from .inpaint_core import run_sam3_refine

    try:
        new_images = run_sam3_refine(
            image,
            args,
            sd_model=sd_model,
            outpath_samples=outpath_samples,
            outpath_grids=outpath_grids,
        )
    except Exception:
        error = traceback.format_exc()
        print(f"[-] SAM3 Refine: handler failed:\n{error}", file=sys.stderr)
        return gallery_value, f"<pre style='color:#c33'>SAM3 Refine failed — see console.</pre>"

    if not new_images:
        return gallery_value, "<span style='color:#c80'>SAM3 Refine: no result (empty mask or interrupted).</span>"

    if insert_mode == "At end":
        updated = gallery_list + new_images
    else:
        updated = gallery_list[: idx + 1] + new_images + gallery_list[idx + 1 :]

    return updated, f"<span style='color:#383'>SAM3 Refine: added {len(new_images)} image(s).</span>"
