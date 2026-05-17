"""Post-generation Refine panel — lives under the t2i gallery and runs SAM3
inpaint (optionally with a ControlNet unit) against the currently selected
gallery image, appending the result back into the same gallery."""
from __future__ import annotations

import io
import json
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
    threshold: gr.Slider
    mask_dilation: gr.Slider
    mask_hull: gr.Checkbox
    mask_blur: gr.Slider
    unload_after: gr.Checkbox
    seed: gr.Number
    denoising_strength: gr.Slider
    inpainting_fill: gr.Dropdown
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
            self.threshold,
            self.mask_dilation,
            self.mask_hull,
            self.mask_blur,
            self.unload_after,
            self.seed,
            self.denoising_strength,
            self.inpainting_fill,
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
    "threshold",
    "mask_dilation",
    "mask_hull",
    "mask_blur",
    "unload_after",
    "seed",
    "denoising_strength",
    "inpainting_fill",
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
                label="Target (마스크/치환할 대상)",
                lines=1,
                placeholder=(
                    "what SAM3 masks AND what gets stripped from the inherited main prompt.\n"
                    "comma-separated: e.g. 'shirt, necktie' → 'white shirt' / 'black necktie' segments both go."
                ),
                elem_id="sam3_refine_target",
            )
        with gr.Row():
            inpaint_prompt = gr.Textbox(
                value="",
                label="Replacement (대체할 단어)",
                lines=2,
                placeholder=(
                    "what to draw inside the mask AND what to insert in place of the Target tokens.\n"
                    "single occurrence even when multiple Target tokens matched. e.g. 'nude'"
                ),
                elem_id="sam3_refine_replacement",
            )
        with gr.Row():
            negative_prompt = gr.Textbox(
                value="",
                label="Negative Prompt",
                lines=1,
                placeholder="Optional — appended after the (possibly cleaned) inherited main negative.",
                elem_id="sam3_refine_negative",
            )

        with gr.Row():
            inherit_main_prompt = gr.Checkbox(
                label="Inherit main t2i prompt (LoRAs / style triggers) — Target auto-stripped",
                value=True,
                elem_id="sam3_refine_inherit_main",
            )
            inherit_main_neg_prompt = gr.Checkbox(
                label="Inherit main t2i negative (Target also stripped here)",
                value=True,
                elem_id="sam3_refine_inherit_neg",
            )

        with gr.Row():
            threshold = gr.Slider(label="SAM3 Threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.4, elem_id="sam3_refine_threshold")
            mask_dilation = gr.Slider(label="Mask Dilation (px)", minimum=0, maximum=256, step=1, value=4, elem_id="sam3_refine_mask_dilation")
            mask_blur = gr.Slider(label="Mask Blur", minimum=0, maximum=64, step=1, value=4, elem_id="sam3_refine_mask_blur")
            mask_mode = gr.Radio(
                label="Mask Processing",
                choices=["Individual", "Combined"],
                value="Combined",
                elem_id="sam3_refine_mask_mode",
            )

        with gr.Row():
            mask_hull = gr.Checkbox(
                label="Convex Hull (wrap strands — recommended for hair/fur)",
                value=False,
                elem_id="sam3_refine_mask_hull",
            )
            unload_after = gr.Checkbox(
                label="Unload SAM3 from VRAM after detection (~3.5 GB — recommended for ≤12 GB GPUs)",
                value=False,
                elem_id="sam3_refine_unload_after",
            )
            seed = gr.Number(
                label="Seed (-1 = random)",
                value=-1,
                precision=0,
                elem_id="sam3_refine_seed",
            )

        with gr.Row():
            denoising_strength = gr.Slider(
                label="Denoising Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.75,
                elem_id="sam3_refine_denoising",
            )
            inpainting_fill = gr.Dropdown(
                label="Masked content",
                choices=["fill", "original", "latent noise", "latent nothing"],
                value="latent noise",
                type="value",
                elem_id="sam3_refine_inpainting_fill",
            )
            inpaint_only_masked = gr.Checkbox(label="Inpaint only masked", value=False, elem_id="sam3_refine_inpaint_only_masked")
            inpaint_only_masked_padding = gr.Slider(
                label="Inpaint padding", minimum=0, maximum=256, step=1, value=32,
                elem_id="sam3_refine_inpaint_padding",
            )

        with gr.Row():
            steps = gr.Slider(label="Steps", minimum=1, maximum=150, step=1, value=28, elem_id="sam3_refine_steps")
            cfg_scale = gr.Slider(label="CFG Scale", minimum=0.0, maximum=30.0, step=0.1, value=7.0, elem_id="sam3_refine_cfg")
            sampler = gr.Dropdown(
                label="Sampler",
                choices=samplers or ["Euler a"],
                value=samplers[0] if samplers else "Euler a",
                type="value",
                elem_id="sam3_refine_sampler",
            )
            scheduler = gr.Dropdown(
                label="Scheduler",
                choices=schedulers or ["Automatic"],
                value=schedulers[0] if schedulers else "Automatic",
                type="value",
                elem_id="sam3_refine_scheduler",
            )

        with gr.Row():
            checkpoint = gr.Dropdown(
                label="SAM3 Checkpoint",
                choices=checkpoint_choices,
                value=checkpoint_choices[0] if checkpoint_choices else "sam3.pt",
                type="value",
                elem_id="sam3_refine_checkpoint",
            )

        with gr.Accordion("ControlNet", open=False):
            with gr.Row():
                cn_enable = gr.Checkbox(label="Enable ControlNet", value=False, elem_id="sam3_refine_cn_enable")
                cn_override_external = gr.Checkbox(
                    label="Override external CN units", value=False, elem_id="sam3_refine_cn_override"
                )
                cn_pixel_perfect = gr.Checkbox(label="Pixel Perfect", value=True, elem_id="sam3_refine_cn_pp")
            with gr.Row():
                cn_module = gr.Dropdown(
                    label="Preprocessor",
                    choices=cn_modules,
                    value=cn_module_default,
                    type="value",
                    elem_id="sam3_refine_cn_module",
                )
                cn_model = gr.Dropdown(
                    label="Model",
                    choices=cn_models,
                    value=cn_models[0] if cn_models else "None",
                    type="value",
                    elem_id="sam3_refine_cn_model",
                )
            with gr.Row():
                cn_weight = gr.Slider(label="Weight", minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id="sam3_refine_cn_weight")
                cn_guidance_start = gr.Slider(
                    label="Guidance Start", minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                    elem_id="sam3_refine_cn_gstart",
                )
                cn_guidance_end = gr.Slider(
                    label="Guidance End", minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                    elem_id="sam3_refine_cn_gend",
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
                    elem_id="sam3_refine_cn_control_mode",
                )
                cn_resize_mode = gr.Radio(
                    label="Resize Mode",
                    choices=["Just Resize", "Crop and Resize", "Resize and Fill"],
                    value="Crop and Resize",
                    elem_id="sam3_refine_cn_resize_mode",
                )
            with gr.Row():
                cn_processor_res = gr.Slider(
                    label="Preprocessor Resolution",
                    minimum=64,
                    maximum=2048,
                    step=8,
                    value=512,
                    elem_id="sam3_refine_cn_procres",
                )
                cn_threshold_a = gr.Slider(
                    label="Threshold A", minimum=-1, maximum=256, step=1, value=-1,
                    elem_id="sam3_refine_cn_ta",
                )
                cn_threshold_b = gr.Slider(
                    label="Threshold B", minimum=-1, maximum=256, step=1, value=-1,
                    elem_id="sam3_refine_cn_tb",
                )

        with gr.Row():
            insert_mode = gr.Radio(
                label="Insert result",
                choices=["After selected", "At end"],
                value="After selected",
                elem_id="sam3_refine_insert_mode",
            )
            refine_button = gr.Button("▶ Refine", variant="primary", elem_id="sam3_refine_button")

        status = gr.HTML(value="", elem_id="sam3_refine_status")

    return RefinePanel(
        accordion=acc,
        selected_index_state=selected_index_state,
        detect_prompt=detect_prompt,
        inpaint_prompt=inpaint_prompt,
        negative_prompt=negative_prompt,
        inherit_main_prompt=inherit_main_prompt,
        inherit_main_neg_prompt=inherit_main_neg_prompt,
        threshold=threshold,
        mask_dilation=mask_dilation,
        mask_hull=mask_hull,
        mask_blur=mask_blur,
        unload_after=unload_after,
        seed=seed,
        denoising_strength=denoising_strength,
        inpainting_fill=inpainting_fill,
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


def _normalize_prompt(text: str) -> str:
    """Collapse whitespace + comma spacing + drop empty / repeated commas."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text.strip(" ,")


def _strip_patterns_with_replacement(text: str, patterns: list[str], replacement: str) -> str:
    """Walk comma-separated segments of ``text``; drop any segment that
    contains any of ``patterns``; insert ``replacement`` once at the first
    drop position.

    Comma-segment removal (vs naive substring replace) avoids orphan
    fragments like "white" left behind when stripping "shirt" from
    "white shirt".
    """
    if not patterns:
        return text
    segments = [s.strip() for s in text.split(",")]
    out: list[str] = []
    inserted = False
    matched_any = False
    for seg in segments:
        if not seg:
            continue
        seg_matches = any(pat and pat in seg for pat in patterns)
        if seg_matches:
            matched_any = True
            if not inserted and replacement:
                out.append(replacement)
                inserted = True
            continue
        out.append(seg)
    if not matched_any:
        return text
    return ", ".join(out)


def _apply_prompt_sr(text: str, rules_field: str) -> str:
    """Apply S/R rules (one per line) to ``text``.

    Rule grammar::

        pat                       = replacement
        pat1, pat2, ..., patN     = replacement

    Multi-pattern: every comma-segment containing any of the patterns is
    removed; ``replacement`` is inserted once at the first match position
    (so ``white shirt, black necktie = nude`` collapses both segments into
    a single ``nude``, not ``nude, nude``).

    Empty replacement deletes the matching segments without inserting
    anything.

    Returns ``text`` unchanged when ``rules_field`` is empty or no rule
    matches.
    """
    if not text or not rules_field:
        return text
    out = text
    matched_any = False
    for raw_line in rules_field.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        pattern_part, _, replacement = line.partition("=")
        replacement = replacement.strip()
        patterns = [p.strip() for p in pattern_part.split(",") if p.strip()]
        if not patterns:
            continue
        new_out = _strip_patterns_with_replacement(out, patterns, replacement)
        if new_out != out:
            matched_any = True
            out = new_out
    if not matched_any:
        return text
    return _normalize_prompt(out)


def _parse_detect_tokens(detect_prompt: str) -> list[str]:
    """Split a SAM3 detect prompt the same way SAM3 does internally — by
    ``,``, ``/``, ``;``, or newlines — and trim each token. Used by the
    auto-S/R derivation so every detected concept is stripped from the
    inherited main prompt.
    """
    if not detect_prompt:
        return []
    tokens = re.split(r"[,/;\n]", detect_prompt)
    return [t.strip() for t in tokens if t.strip()]


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
        "sam3_seed": _as_int(keyed.get("seed"), -1),
        "sam3_checkpoint": str(keyed.get("checkpoint") or "sam3.pt"),
        "sam3_device": "auto",
        "sam3_mask_mode": str(keyed.get("mask_mode") or "Combined"),
        # Inpaint
        "sam3_mask_blur": _as_int(keyed.get("mask_blur"), 4),
        "sam3_denoising_strength": _as_float(keyed.get("denoising_strength"), 0.75),
        "sam3_inpainting_fill": str(keyed.get("inpainting_fill") or "latent noise"),
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
    }


def _plaintext_to_html(text: str) -> str:
    """Light-weight stand-in for modules.ui.plaintext_to_html (avoids importing
    Forge modules at module load) — escape + preserve newlines for display in
    the txt2img sidebar's infotext HTML element."""
    if not text:
        return ""
    import html as _html

    escaped = _html.escape(text)
    return f"<p>{escaped.replace(chr(10), '<br>')}</p>"


def _refine_error_return(gallery_value, message: str):
    """4-tuple return shape used by handle_refine_click. ``gr.update()`` for
    html_info / generation_info leaves them untouched."""
    return gallery_value, message, gr.update(), gr.update()


def handle_refine_click(gallery_value, selected_index, *all_values):
    """Refine-button handler. Returns
    ``(updated_gallery, status_html, html_info, generation_info_json)``.

    ``all_values`` =
    ``(*widget_values, main_prompt, main_neg_prompt, current_generation_info)``
    — the three extras are appended by the wiring in ``scripts/!sam3.py``.

    The last two outputs update the gallery sidebar's prompt display:
    ``html_info`` is the immediately-visible HTML for the latest refine,
    ``generation_info_json`` is the per-image infotext array that
    ``update_generation_info`` (the standard click-handler) reads when the
    user clicks a different gallery thumbnail.

    Out-of-scope behaviors are returned as HTML status messages rather than
    raised exceptions so the panel stays responsive.
    """
    expected_widget_count = len(REFINE_ARG_KEYS)
    if len(all_values) < expected_widget_count:
        return _refine_error_return(
            gallery_value,
            "<span style='color:#c33'>SAM3 Refine: missing widget values.</span>",
        )

    widget_values = all_values[:expected_widget_count]
    extras = all_values[expected_widget_count:]
    main_prompt = str(extras[0]) if len(extras) > 0 else ""
    main_neg_prompt = str(extras[1]) if len(extras) > 1 else ""
    current_info_json = str(extras[2]) if len(extras) > 2 else ""

    args = map_widget_values_to_sam3_args(widget_values)
    insert_mode = args.pop("_insert_mode", "After selected")
    inherit_main = args.pop("_inherit_main_prompt", True)
    inherit_main_neg = args.pop("_inherit_main_neg_prompt", True)

    refine_p = args.get("sam3_inpaint_prompt") or ""
    detect_p = args.get("sam3_prompt") or ""
    detect_tokens = _parse_detect_tokens(detect_p)

    # Target → S/R: positive gets the Replacement injected once at the first
    # match site; negative just deletes (don't leak the new subject into the
    # anti-prompt).
    sr_positive = f"{', '.join(detect_tokens)} = {refine_p}" if detect_tokens else ""
    sr_negative = f"{', '.join(detect_tokens)} = " if detect_tokens else ""

    cleaned_main = _apply_prompt_sr(main_prompt, sr_positive) if main_prompt else main_prompt
    cleaned_neg = _apply_prompt_sr(main_neg_prompt, sr_negative) if main_neg_prompt else main_neg_prompt

    # Trace exactly what S/R produced — so a user can verify in the console
    # that 'shirt' really did strip 'white shirt' etc. instead of guessing.
    if cleaned_main != main_prompt or cleaned_neg != main_neg_prompt:
        print(
            f"[-] SAM3 Refine prompt transform:\n"
            f"      target  : {detect_p!r}\n"
            f"      replace : {refine_p!r}\n"
            f"      main +  : {main_prompt!r}\n"
            f"           -> : {cleaned_main!r}\n"
            f"      main -  : {main_neg_prompt!r}\n"
            f"           -> : {cleaned_neg!r}",
            file=sys.stderr,
        )

    # Positive resolution:
    # - Target set + inherit ON : cleaned_main already has the replacement
    #                             injected; that IS the final prompt.
    # - Target empty + inherit ON: append refine prompt to main (back-compat).
    # - inherit OFF             : refine prompt only.
    if inherit_main and cleaned_main:
        if detect_tokens:
            args["sam3_inpaint_prompt"] = cleaned_main
        else:
            args["sam3_inpaint_prompt"] = (
                f"{cleaned_main}, {refine_p}".rstrip(", ") if refine_p else cleaned_main
            )

    refine_n = args.get("sam3_negative_prompt") or ""
    if inherit_main_neg and cleaned_neg:
        args["sam3_negative_prompt"] = (
            f"{cleaned_neg}, {refine_n}".rstrip(", ") if refine_n else cleaned_neg
        )

    if not args["sam3_prompt"]:
        return _refine_error_return(
            gallery_value,
            "<span style='color:#c33'>SAM3 Refine: enter a Target (detect prompt) first.</span>",
        )

    gallery_list = list(gallery_value or [])
    if not gallery_list:
        return _refine_error_return(
            gallery_value,
            "<span style='color:#c33'>SAM3 Refine: gallery is empty.</span>",
        )

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
        return _refine_error_return(
            gallery_value,
            f"<span style='color:#c33'>SAM3 Refine: could not load selected image (index {idx}).</span>",
        )

    from modules import shared

    sd_model = getattr(shared, "sd_model", None)
    if sd_model is None:
        return _refine_error_return(
            gallery_value,
            "<span style='color:#c33'>SAM3 Refine: no SD model loaded.</span>",
        )

    outpath_samples = getattr(shared.opts, "outdir_txt2img_samples", "outputs/txt2img-images")
    outpath_grids = getattr(shared.opts, "outdir_txt2img_grids", "outputs/txt2img-grids")

    from .inpaint_core import run_sam3_refine

    try:
        new_pairs = run_sam3_refine(
            image,
            args,
            sd_model=sd_model,
            outpath_samples=outpath_samples,
            outpath_grids=outpath_grids,
        )
    except Exception:
        error = traceback.format_exc()
        print(f"[-] SAM3 Refine: handler failed:\n{error}", file=sys.stderr)
        return _refine_error_return(
            gallery_value,
            "<pre style='color:#c33'>SAM3 Refine failed — see console.</pre>",
        )

    if not new_pairs:
        return _refine_error_return(
            gallery_value,
            "<span style='color:#c80'>SAM3 Refine: no result (empty mask or interrupted).</span>",
        )

    new_images = [img for img, _ in new_pairs]
    new_infotexts = [info for _, info in new_pairs]

    if insert_mode == "At end":
        updated = gallery_list + new_images
    else:
        updated = gallery_list[: idx + 1] + new_images + gallery_list[idx + 1 :]

    # Splice the new infotexts into the existing generation_info JSON so the
    # standard "click a gallery item" handler shows the per-image transformed
    # prompt instead of falling back to the original t2i prompt.
    info_payload: dict[str, Any]
    try:
        info_payload = json.loads(current_info_json) if current_info_json else {}
    except Exception:
        info_payload = {}
    existing_infotexts = list(info_payload.get("infotexts") or [""] * len(gallery_list))
    while len(existing_infotexts) < len(gallery_list):
        existing_infotexts.append("")
    if insert_mode == "At end":
        merged_infotexts = existing_infotexts + new_infotexts
    else:
        merged_infotexts = (
            existing_infotexts[: idx + 1] + new_infotexts + existing_infotexts[idx + 1 :]
        )
    info_payload["infotexts"] = merged_infotexts
    new_info_json = json.dumps(info_payload, ensure_ascii=False)

    # Right side of the gallery — immediately show the latest refine's prompt
    # so the user can verify the transformation visually without clicking.
    latest_html = _plaintext_to_html(new_infotexts[-1] if new_infotexts else "")

    return (
        updated,
        f"<span style='color:#383'>SAM3 Refine: added {len(new_images)} image(s). Click the new thumbnail to recheck infotext.</span>",
        latest_html,
        new_info_json,
    )
