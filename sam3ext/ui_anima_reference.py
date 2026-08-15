"""Feature 6 UI for the Anima Character Reference / ReStyler workflow."""
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from .anima_reference_core import ReferenceCanvasConfig, prepare_reference_canvas
from .anima_reference_runner import (
    ReferenceGenerationRequest,
    ReferenceGenerationResult,
    run_anima_reference,
)
from .coerce import as_float, as_int
from .core import find_checkpoint_options
from .ui_refine import _coerce_gallery_item_to_pil, _plaintext_to_html


_RESTYLER_URL = "https://civitai.com/models/2803070/anima-restyler"
_EDIT_LORA_DEFAULT = "AnimeEditV2"
_EXTEND_LORA_DEFAULT = "Extend Image (Anima Edit) v1"


def list_installed_lora_names(*, refresh: bool = False) -> list[str]:
    """Return only LoRAs actually indexed by Forge."""

    try:
        import networks

        if refresh:
            networks.list_available_networks()
        names = {
            *(getattr(networks, "available_networks", {}) or {}),
            *(getattr(networks, "available_network_aliases", {}) or {}),
        }
    except Exception:
        names = set()
    return sorted(
        (str(name) for name in names if str(name).strip()),
        key=str.casefold,
    )


def list_lora_choices(*, refresh: bool = False) -> list[str]:
    """Return installed LoRAs plus the two editable workflow suggestions."""

    names = list_installed_lora_names(refresh=refresh)
    return sorted(
        {
            _EDIT_LORA_DEFAULT,
            _EXTEND_LORA_DEFAULT,
            *(str(name) for name in names if str(name).strip()),
        },
        key=str.casefold,
    )


def list_module_choices(*, refresh: bool = False) -> list[str]:
    """Return VAE/Text Encoder module names shown by Forge's model picker."""

    try:
        from modules import shared
        from modules_forge import main_entry

        if refresh:
            _, modules = main_entry.refresh_models()
            return [str(item) for item in modules]
        indexed = list(getattr(main_entry, "module_list", {}) or {})
        current = [
            Path(str(item)).name
            for item in getattr(
                shared.opts, "forge_additional_modules", []
            )
        ]
        # Do not rescan model directories during extension UI construction.
        # Forge's own picker populates this registry; the explicit refresh
        # button above performs a scan when the user asks for one.
        return sorted({*(str(item) for item in indexed), *current})
    except Exception:
        return []


def refresh_reference_model_choices():
    """Refresh LoRA, checkpoint, and VAE/TE dropdowns together."""

    loras = list_lora_choices(refresh=True)
    modules = list_module_choices(refresh=True)
    checkpoints = ["Use current", *find_checkpoint_options()]
    return (
        gr.update(choices=loras),
        gr.update(choices=loras),
        gr.update(choices=checkpoints),
        gr.update(choices=modules),
    )


@dataclass
class AnimaReferencePanel:
    accordion: gr.Accordion
    selected_index_state: gr.Number
    reference_image: gr.Image
    load_selected_button: gr.Button

    output_width: gr.Slider
    output_height: gr.Slider
    placement: gr.Radio
    target_region_scale: gr.Slider
    target_color: gr.ColorPicker
    reference_matte_color: gr.ColorPicker
    composite_megapixels: gr.Slider
    dimension_multiple: gr.Number
    resize_filter: gr.Radio
    mask_overlap: gr.Slider

    inherit_main_prompt: gr.Checkbox
    prompt: gr.Textbox
    inherit_main_negative: gr.Checkbox
    negative_prompt: gr.Textbox
    prefix_enabled: gr.Checkbox
    prefix_text: gr.Textbox
    prefix_strength: gr.Slider
    extra_prefix: gr.Textbox
    extra_suffix: gr.Textbox

    edit_lora_enabled: gr.Checkbox
    edit_lora_name: gr.Dropdown
    edit_lora_strength: gr.Slider
    extend_lora_enabled: gr.Checkbox
    extend_lora_name: gr.Dropdown
    extend_lora_strength: gr.Slider
    require_enabled_loras: gr.Checkbox

    checkpoint_override: gr.Dropdown
    override_additional_modules: gr.Checkbox
    additional_modules: gr.Dropdown
    refresh_models_button: gr.Button

    steps: gr.Slider
    cfg_scale: gr.Slider
    shift: gr.Slider
    sampler: gr.Dropdown
    scheduler: gr.Dropdown
    denoising_strength: gr.Slider
    resize_mode: gr.Radio
    inpainting_fill: gr.Radio
    mask_blur: gr.Slider
    mask_round: gr.Checkbox
    mask_invert: gr.Checkbox
    inpainting_mask_weight: gr.Slider
    initial_noise_multiplier: gr.Slider

    eta: gr.Slider
    s_min_uncond: gr.Slider
    s_churn: gr.Slider
    s_tmin: gr.Slider
    s_tmax: gr.Slider
    s_noise: gr.Slider

    seed: gr.Number
    seed_step: gr.Number
    candidate_count: gr.Slider
    seed_random_button: gr.Button
    seed_pull_button: gr.Button
    restore_faces: gr.Checkbox
    native_reference_enabled: gr.Checkbox

    save_target: gr.Checkbox
    save_generated_canvas: gr.Checkbox
    save_input_canvas: gr.Checkbox
    save_mask: gr.Checkbox
    gallery_content: gr.Radio
    insert_mode: gr.Radio

    preview_button: gr.Button
    preview_canvas: gr.Image
    preview_mask: gr.Image
    generate_button: gr.Button
    stop_button: gr.Button
    status: gr.HTML

    def all_widgets(self) -> list:
        return [getattr(self, name) for name in REFERENCE_ARG_KEYS]

    def geometry_widgets(self) -> list:
        return [
            self.reference_image,
            self.output_width,
            self.output_height,
            self.placement,
            self.target_region_scale,
            self.target_color,
            self.reference_matte_color,
            self.composite_megapixels,
            self.dimension_multiple,
            self.resize_filter,
            self.mask_overlap,
        ]


REFERENCE_ARG_KEYS: tuple[str, ...] = (
    "reference_image",
    "output_width",
    "output_height",
    "placement",
    "target_region_scale",
    "target_color",
    "reference_matte_color",
    "composite_megapixels",
    "dimension_multiple",
    "resize_filter",
    "mask_overlap",
    "inherit_main_prompt",
    "prompt",
    "inherit_main_negative",
    "negative_prompt",
    "prefix_enabled",
    "prefix_text",
    "prefix_strength",
    "extra_prefix",
    "extra_suffix",
    "edit_lora_enabled",
    "edit_lora_name",
    "edit_lora_strength",
    "extend_lora_enabled",
    "extend_lora_name",
    "extend_lora_strength",
    "require_enabled_loras",
    "checkpoint_override",
    "override_additional_modules",
    "additional_modules",
    "steps",
    "cfg_scale",
    "shift",
    "sampler",
    "scheduler",
    "denoising_strength",
    "resize_mode",
    "inpainting_fill",
    "mask_blur",
    "mask_round",
    "mask_invert",
    "inpainting_mask_weight",
    "initial_noise_multiplier",
    "eta",
    "s_min_uncond",
    "s_churn",
    "s_tmin",
    "s_tmax",
    "s_noise",
    "seed",
    "seed_step",
    "candidate_count",
    "restore_faces",
    "native_reference_enabled",
    "save_target",
    "save_generated_canvas",
    "save_input_canvas",
    "save_mask",
    "gallery_content",
    "insert_mode",
)

_REFERENCE_EXTRA_INPUTS = 3  # main prompt, main negative, generation_info


def _slider(
    label: str,
    *,
    value: float,
    minimum: float,
    maximum: float,
    step: float,
    elem_id: str,
    info: str | None = None,
) -> gr.Slider:
    return gr.Slider(
        label=label,
        value=value,
        minimum=minimum,
        maximum=maximum,
        step=step,
        elem_id=elem_id,
        info=info,
    )


def build_anima_reference_panel(
    samplers: list[str],
    schedulers: list[str],
) -> AnimaReferencePanel:
    """Render Feature 6 as a sibling of the extension's other accordions."""

    lora_choices = list_lora_choices()
    checkpoint_choices = ["Use current", *find_checkpoint_options()]
    module_choices = list_module_choices()
    sampler_choices = samplers or ["Euler a"]
    scheduler_choices = schedulers or ["Simple"]
    sampler_default = (
        "Euler a" if "Euler a" in sampler_choices else sampler_choices[0]
    )
    scheduler_default = next(
        (
            item
            for item in scheduler_choices
            if str(item).strip().casefold() == "simple"
        ),
        scheduler_choices[0],
    )

    with gr.Accordion(
        "Feature 6 — Anima Character Reference / ReStyler",
        open=False,
        elem_id="sam3_anima_reference_panel",
    ) as accordion:
        selected_index_state = gr.Number(
            value=-1,
            precision=0,
            visible=False,
            elem_id="sam3_anima_reference_selected_index",
        )
        gr.Markdown(
            "참조 캐릭터와 빈 생성 영역을 한 캔버스로 만든 뒤, 빈 영역만 "
            "Anima Edit로 인페인트하고 정확한 출력 크기로 잘라냅니다. "
            "Forge Neo 코어는 수정하지 않으며 **Anima 네이티브 Reference**를 "
            "작업 중에만 켭니다. 모델/LoRA 안내: "
            f"[Anima ReStyler]({_RESTYLER_URL})"
        )

        with gr.Row():
            reference_image = gr.Image(
                type="pil",
                sources=["upload", "clipboard"],
                label="Reference character image",
                height=360,
                elem_id="sam3_anima_reference_image",
            )
            with gr.Column():
                load_selected_button = gr.Button(
                    "📋 선택한 T2I 이미지를 Reference로",
                    elem_id="sam3_anima_reference_load_selected",
                )
                gr.Markdown(
                    "**입력 권장:** 단순한 배경, 중립 포즈. 왼쪽/오른쪽 배치, "
                    "빈 영역 색, 합성 해상도까지 아래에서 직접 바꿀 수 있습니다."
                )

        with gr.Accordion("1. Canvas / Mask / Crop", open=True):
            with gr.Row():
                output_width = _slider(
                    "Result width",
                    value=960,
                    minimum=64,
                    maximum=4096,
                    step=8,
                    elem_id="sam3_anima_reference_output_width",
                )
                output_height = _slider(
                    "Result height",
                    value=1088,
                    minimum=64,
                    maximum=4096,
                    step=8,
                    elem_id="sam3_anima_reference_output_height",
                )
            with gr.Row():
                placement = gr.Radio(
                    ["right", "left"],
                    value="right",
                    label="Generated target placement",
                    elem_id="sam3_anima_reference_placement",
                )
                target_region_scale = _slider(
                    "Target panel width scale",
                    value=1.0,
                    minimum=0.05,
                    maximum=4.0,
                    step=0.01,
                    elem_id="sam3_anima_reference_target_scale",
                    info="출력 종횡비에 곱해지는 빈 패널 폭입니다.",
                )
            with gr.Row():
                target_color = gr.ColorPicker(
                    value="#000000",
                    label="Empty target color",
                    elem_id="sam3_anima_reference_target_color",
                )
                reference_matte_color = gr.ColorPicker(
                    value="#ffffff",
                    label="Transparent reference matte",
                    elem_id="sam3_anima_reference_matte_color",
                )
            with gr.Row():
                composite_megapixels = _slider(
                    "Composite megapixels (0 = source height)",
                    value=1.4,
                    minimum=0.0,
                    maximum=16.0,
                    step=0.05,
                    elem_id="sam3_anima_reference_megapixels",
                )
                dimension_multiple = gr.Number(
                    value=16,
                    precision=0,
                    minimum=1,
                    maximum=256,
                    label="Dimension multiple",
                    elem_id="sam3_anima_reference_multiple",
                )
            with gr.Row():
                resize_filter = gr.Radio(
                    ["nearest", "bilinear", "bicubic", "lanczos"],
                    value="lanczos",
                    label="Resize / final crop filter",
                    elem_id="sam3_anima_reference_resize_filter",
                )
                mask_overlap = _slider(
                    "Mask overlap into reference (px)",
                    value=0,
                    minimum=0,
                    maximum=1024,
                    step=1,
                    elem_id="sam3_anima_reference_mask_overlap",
                )

        with gr.Accordion("2. Prompt / LoRA", open=True):
            with gr.Row():
                inherit_main_prompt = gr.Checkbox(
                    value=True,
                    label="Use current T2I prompt",
                    elem_id="sam3_anima_reference_inherit_prompt",
                )
                inherit_main_negative = gr.Checkbox(
                    value=True,
                    label="Use current T2I negative prompt",
                    elem_id="sam3_anima_reference_inherit_negative",
                )
            prompt = gr.Textbox(
                value="",
                lines=3,
                label="Feature 6 prompt (used when current prompt is OFF)",
                elem_id="sam3_anima_reference_prompt",
            )
            negative_prompt = gr.Textbox(
                value="",
                lines=2,
                label="Feature 6 negative prompt (used when current negative is OFF)",
                elem_id="sam3_anima_reference_negative",
            )
            with gr.Row():
                prefix_enabled = gr.Checkbox(
                    value=True,
                    label="Enable reference prefix",
                    elem_id="sam3_anima_reference_prefix_enabled",
                )
                prefix_text = gr.Textbox(
                    value="split screen, multiple views",
                    label="Reference prefix text",
                    elem_id="sam3_anima_reference_prefix_text",
                )
                prefix_strength = _slider(
                    "Prefix weight",
                    value=1.2,
                    minimum=-5.0,
                    maximum=5.0,
                    step=0.05,
                    elem_id="sam3_anima_reference_prefix_strength",
                )
            with gr.Row():
                extra_prefix = gr.Textbox(
                    value="",
                    label="Extra prefix",
                    elem_id="sam3_anima_reference_extra_prefix",
                )
                extra_suffix = gr.Textbox(
                    value="",
                    label="Extra suffix",
                    elem_id="sam3_anima_reference_extra_suffix",
                )
            gr.Markdown(
                "기본 권장: **AnimeEditV2 = 0.72**, "
                "**Extend Image (Anima Edit) v1 = 0.4**. 파일명과 강도를 "
                "모두 직접 바꿀 수 있습니다."
            )
            with gr.Row():
                edit_lora_enabled = gr.Checkbox(
                    value=True,
                    label="Enable Anima Edit LoRA",
                    elem_id="sam3_anima_reference_edit_enabled",
                )
                edit_lora_name = gr.Dropdown(
                    choices=lora_choices,
                    value=_EDIT_LORA_DEFAULT,
                    allow_custom_value=True,
                    label="Anima Edit LoRA",
                    elem_id="sam3_anima_reference_edit_lora",
                )
                edit_lora_strength = _slider(
                    "Edit LoRA strength",
                    value=0.72,
                    minimum=-5.0,
                    maximum=5.0,
                    step=0.01,
                    elem_id="sam3_anima_reference_edit_strength",
                )
            with gr.Row():
                extend_lora_enabled = gr.Checkbox(
                    value=True,
                    label="Enable Extend LoRA",
                    elem_id="sam3_anima_reference_extend_enabled",
                )
                extend_lora_name = gr.Dropdown(
                    choices=lora_choices,
                    value=_EXTEND_LORA_DEFAULT,
                    allow_custom_value=True,
                    label="Extend Image LoRA",
                    elem_id="sam3_anima_reference_extend_lora",
                )
                extend_lora_strength = _slider(
                    "Extend LoRA strength",
                    value=0.4,
                    minimum=-5.0,
                    maximum=5.0,
                    step=0.01,
                    elem_id="sam3_anima_reference_extend_strength",
                )
            require_enabled_loras = gr.Checkbox(
                value=True,
                label="Stop before generation if an enabled LoRA is missing",
                elem_id="sam3_anima_reference_require_loras",
            )

        with gr.Accordion("3. Model / VAE / Text Encoder", open=False):
            with gr.Row():
                checkpoint_override = gr.Dropdown(
                    choices=checkpoint_choices,
                    value="Use current",
                    allow_custom_value=True,
                    label="Checkpoint override",
                    elem_id="sam3_anima_reference_checkpoint",
                )
                refresh_models_button = gr.Button(
                    "🔄 Checkpoint / Module / LoRA 새로고침",
                    elem_id="sam3_anima_reference_refresh",
                )
            override_additional_modules = gr.Checkbox(
                value=False,
                label="Override VAE / Text Encoder for this Feature 6 run",
                elem_id="sam3_anima_reference_override_modules",
            )
            additional_modules = gr.Dropdown(
                choices=module_choices,
                value=[],
                multiselect=True,
                allow_custom_value=True,
                label="VAE / Text Encoder modules",
                elem_id="sam3_anima_reference_modules",
            )

        with gr.Accordion("4. Sampling", open=True):
            with gr.Row():
                steps = _slider(
                    "Steps",
                    value=30,
                    minimum=1,
                    maximum=200,
                    step=1,
                    elem_id="sam3_anima_reference_steps",
                )
                cfg_scale = _slider(
                    "CFG scale",
                    value=5.0,
                    minimum=0.0,
                    maximum=30.0,
                    step=0.1,
                    elem_id="sam3_anima_reference_cfg",
                )
                shift = _slider(
                    "Shift / Distilled CFG",
                    value=3.0,
                    minimum=0.0,
                    maximum=30.0,
                    step=0.1,
                    elem_id="sam3_anima_reference_shift",
                )
            with gr.Row():
                sampler = gr.Dropdown(
                    choices=sampler_choices,
                    value=sampler_default,
                    allow_custom_value=True,
                    label="Sampler",
                    elem_id="sam3_anima_reference_sampler",
                )
                scheduler = gr.Dropdown(
                    choices=scheduler_choices,
                    value=scheduler_default,
                    allow_custom_value=True,
                    label="Scheduler",
                    elem_id="sam3_anima_reference_scheduler",
                )
            with gr.Row():
                denoising_strength = _slider(
                    "Denoising strength",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    elem_id="sam3_anima_reference_denoise",
                    info="원본 워크플로우의 full-noise 기본값은 1.0입니다.",
                )
                initial_noise_multiplier = _slider(
                    "Initial noise multiplier",
                    value=1.0,
                    minimum=0.0,
                    maximum=3.0,
                    step=0.01,
                    elem_id="sam3_anima_reference_noise_multiplier",
                )
            with gr.Row():
                resize_mode = gr.Radio(
                    ["Just Resize", "Crop and Resize", "Resize and Fill"],
                    value="Just Resize",
                    label="Img2img resize mode",
                    elem_id="sam3_anima_reference_i2i_resize",
                )
                inpainting_fill = gr.Radio(
                    ["fill", "original", "latent noise", "latent nothing"],
                    value="latent noise",
                    label="Masked content",
                    elem_id="sam3_anima_reference_fill",
                )
            with gr.Row():
                mask_blur = _slider(
                    "Mask blur",
                    value=0,
                    minimum=0,
                    maximum=256,
                    step=1,
                    elem_id="sam3_anima_reference_mask_blur",
                )
                inpainting_mask_weight = _slider(
                    "Inpainting conditioning mask weight",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    elem_id="sam3_anima_reference_mask_weight",
                )
            with gr.Row():
                mask_round = gr.Checkbox(
                    value=True,
                    label="Round latent mask",
                    elem_id="sam3_anima_reference_mask_round",
                )
                mask_invert = gr.Checkbox(
                    value=False,
                    label="Invert mask",
                    elem_id="sam3_anima_reference_mask_invert",
                )

            with gr.Accordion("Sampler advanced values", open=False):
                with gr.Row():
                    eta = _slider(
                        "Eta",
                        value=1.0,
                        minimum=0.0,
                        maximum=10.0,
                        step=0.01,
                        elem_id="sam3_anima_reference_eta",
                    )
                    s_min_uncond = _slider(
                        "s_min_uncond",
                        value=0.0,
                        minimum=0.0,
                        maximum=20.0,
                        step=0.01,
                        elem_id="sam3_anima_reference_s_min_uncond",
                    )
                with gr.Row():
                    s_churn = _slider(
                        "s_churn",
                        value=0.0,
                        minimum=0.0,
                        maximum=100.0,
                        step=0.01,
                        elem_id="sam3_anima_reference_s_churn",
                    )
                    s_tmin = _slider(
                        "s_tmin",
                        value=0.0,
                        minimum=0.0,
                        maximum=10.0,
                        step=0.01,
                        elem_id="sam3_anima_reference_s_tmin",
                    )
                    s_tmax = _slider(
                        "s_tmax (0 = sampler default / infinity)",
                        value=0.0,
                        minimum=0.0,
                        maximum=999.0,
                        step=0.01,
                        elem_id="sam3_anima_reference_s_tmax",
                    )
                    s_noise = _slider(
                        "s_noise",
                        value=1.0,
                        minimum=0.0,
                        maximum=3.0,
                        step=0.001,
                        elem_id="sam3_anima_reference_s_noise",
                    )

        with gr.Accordion("5. Seed / Output / Diagnostics", open=True):
            with gr.Row():
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed (-1 = random)",
                    elem_id="sam3_anima_reference_seed",
                )
                seed_random_button = gr.Button(
                    "🎲 -1",
                    elem_id="sam3_anima_reference_seed_random",
                )
                seed_pull_button = gr.Button(
                    "🎯 선택 이미지 Seed",
                    elem_id="sam3_anima_reference_seed_pull",
                )
            with gr.Row():
                seed_step = gr.Number(
                    value=1,
                    precision=0,
                    label="Seed increment per candidate",
                    elem_id="sam3_anima_reference_seed_step",
                )
                candidate_count = _slider(
                    "Candidates",
                    value=1,
                    minimum=1,
                    maximum=16,
                    step=1,
                    elem_id="sam3_anima_reference_candidates",
                )
            with gr.Row():
                native_reference_enabled = gr.Checkbox(
                    value=True,
                    label="Enable Forge native Anima Reference",
                    elem_id="sam3_anima_reference_native",
                )
                restore_faces = gr.Checkbox(
                    value=False,
                    label="Restore faces",
                    elem_id="sam3_anima_reference_restore_faces",
                )
            with gr.Row():
                save_target = gr.Checkbox(
                    value=True,
                    label="Save cropped target",
                    elem_id="sam3_anima_reference_save_target",
                )
                save_generated_canvas = gr.Checkbox(
                    value=False,
                    label="Save generated split canvas",
                    elem_id="sam3_anima_reference_save_generated",
                )
                save_input_canvas = gr.Checkbox(
                    value=False,
                    label="Save input split canvas",
                    elem_id="sam3_anima_reference_save_input",
                )
                save_mask = gr.Checkbox(
                    value=False,
                    label="Save mask",
                    elem_id="sam3_anima_reference_save_mask",
                )
            with gr.Row():
                gallery_content = gr.Radio(
                    ["Target only", "Target + generated canvas"],
                    value="Target only",
                    label="Add to T2I gallery",
                    elem_id="sam3_anima_reference_gallery_content",
                )
                insert_mode = gr.Radio(
                    ["After selected", "At end", "Replace gallery"],
                    value="At end",
                    label="Gallery insertion",
                    elem_id="sam3_anima_reference_insert_mode",
                )

        with gr.Row():
            preview_button = gr.Button(
                "🧩 Canvas / Mask 미리보기",
                elem_id="sam3_anima_reference_preview",
            )
            generate_button = gr.Button(
                "▶ Generate Character Reference",
                variant="primary",
                elem_id="sam3_anima_reference_generate",
            )
            stop_button = gr.Button(
                "⏹ Stop",
                variant="stop",
                visible=False,
                elem_id="sam3_anima_reference_stop",
            )
        with gr.Row():
            preview_canvas = gr.Image(
                label="Prepared split canvas",
                interactive=False,
                elem_id="sam3_anima_reference_preview_canvas",
            )
            preview_mask = gr.Image(
                label="Inpaint mask",
                interactive=False,
                elem_id="sam3_anima_reference_preview_mask",
            )
        status = gr.HTML(
            "<span>Feature 6 ready — 모든 워크플로우 수치는 위에서 수정할 수 있습니다.</span>",
            elem_id="sam3_anima_reference_status",
        )

    return AnimaReferencePanel(
        **{
            name: value
            for name, value in locals().items()
            if name in AnimaReferencePanel.__dataclass_fields__
        }
    )


def _canvas_config_from_keyed(keyed: dict[str, Any]) -> ReferenceCanvasConfig:
    return ReferenceCanvasConfig(
        output_width=as_int(keyed.get("output_width"), 960),
        output_height=as_int(keyed.get("output_height"), 1088),
        placement=str(keyed.get("placement") or "right"),
        target_region_scale=as_float(keyed.get("target_region_scale"), 1.0),
        target_color=str(keyed.get("target_color") or "#000000"),
        reference_matte_color=str(
            keyed.get("reference_matte_color") or "#ffffff"
        ),
        composite_megapixels=as_float(
            keyed.get("composite_megapixels"), 1.4
        ),
        dimension_multiple=as_int(keyed.get("dimension_multiple"), 16),
        resize_filter=str(keyed.get("resize_filter") or "lanczos"),
        mask_overlap=as_int(keyed.get("mask_overlap"), 0),
    )


def preview_reference_layout(*values):
    """Preview handler for only the image and geometry controls."""

    keys = REFERENCE_ARG_KEYS[:11]
    keyed = dict(zip(keys, values))
    image = _coerce_gallery_item_to_pil(keyed.get("reference_image"))
    if image is None:
        return (
            gr.update(),
            gr.update(),
            "<span style='color:#c80'>Reference 이미지를 먼저 넣어 주세요.</span>",
        )
    try:
        prepared = prepare_reference_canvas(
            image,
            _canvas_config_from_keyed(keyed),
        )
        return (
            prepared.canvas,
            prepared.mask,
            (
                "<span style='color:#383'>Prepared canvas "
                f"{prepared.canvas.width}×{prepared.canvas.height}; "
                f"target crop {prepared.output_size[0]}×"
                f"{prepared.output_size[1]}.</span>"
            ),
        )
    except Exception as exc:
        return (
            gr.update(),
            gr.update(),
            f"<span style='color:#c33'>Preview failed: {exc}</span>",
        )


def _module_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(
            part.strip() for part in raw.split(",") if part.strip()
        )
    try:
        return tuple(str(item) for item in raw if str(item).strip())
    except TypeError:
        return ()


def request_from_values(
    values: tuple[Any, ...],
    *,
    reference_image: Image.Image,
    main_prompt: str,
    main_negative: str,
) -> tuple[ReferenceGenerationRequest, str, str]:
    """Map the UI vector to the typed request and gallery-only settings."""

    keyed = dict(zip(REFERENCE_ARG_KEYS, values))
    prompt = (
        str(main_prompt or "")
        if bool(keyed.get("inherit_main_prompt", True))
        else str(keyed.get("prompt") or "")
    )
    negative = (
        str(main_negative or "")
        if bool(keyed.get("inherit_main_negative", True))
        else str(keyed.get("negative_prompt") or "")
    )
    request = ReferenceGenerationRequest(
        reference_image=reference_image,
        canvas=_canvas_config_from_keyed(keyed),
        prompt=prompt,
        negative_prompt=negative,
        prefix_enabled=bool(keyed.get("prefix_enabled", True)),
        prefix_text=str(
            keyed.get("prefix_text") or "split screen, multiple views"
        ),
        prefix_strength=as_float(keyed.get("prefix_strength"), 1.2),
        extra_prefix=str(keyed.get("extra_prefix") or ""),
        extra_suffix=str(keyed.get("extra_suffix") or ""),
        edit_lora_enabled=bool(keyed.get("edit_lora_enabled", True)),
        edit_lora_name=str(keyed.get("edit_lora_name") or ""),
        edit_lora_strength=as_float(
            keyed.get("edit_lora_strength"), 0.72
        ),
        extend_lora_enabled=bool(keyed.get("extend_lora_enabled", True)),
        extend_lora_name=str(keyed.get("extend_lora_name") or ""),
        extend_lora_strength=as_float(
            keyed.get("extend_lora_strength"), 0.4
        ),
        require_enabled_loras=bool(
            keyed.get("require_enabled_loras", True)
        ),
        checkpoint_override=str(
            keyed.get("checkpoint_override") or "Use current"
        ),
        override_additional_modules=bool(
            keyed.get("override_additional_modules", False)
        ),
        additional_modules=_module_tuple(keyed.get("additional_modules")),
        steps=as_int(keyed.get("steps"), 30),
        cfg_scale=as_float(keyed.get("cfg_scale"), 5.0),
        shift=as_float(keyed.get("shift"), 3.0),
        sampler=str(keyed.get("sampler") or "Euler a"),
        scheduler=str(keyed.get("scheduler") or "Simple"),
        denoising_strength=as_float(
            keyed.get("denoising_strength"), 1.0
        ),
        resize_mode=str(keyed.get("resize_mode") or "Just Resize"),
        inpainting_fill=str(
            keyed.get("inpainting_fill") or "latent noise"
        ),
        mask_blur=as_int(keyed.get("mask_blur"), 0),
        mask_round=bool(keyed.get("mask_round", True)),
        mask_invert=bool(keyed.get("mask_invert", False)),
        inpainting_mask_weight=as_float(
            keyed.get("inpainting_mask_weight"), 1.0
        ),
        initial_noise_multiplier=as_float(
            keyed.get("initial_noise_multiplier"), 1.0
        ),
        eta=as_float(keyed.get("eta"), 1.0),
        s_min_uncond=as_float(keyed.get("s_min_uncond"), 0.0),
        s_churn=as_float(keyed.get("s_churn"), 0.0),
        s_tmin=as_float(keyed.get("s_tmin"), 0.0),
        s_tmax=as_float(keyed.get("s_tmax"), 0.0),
        s_noise=as_float(keyed.get("s_noise"), 1.0),
        seed=as_int(keyed.get("seed"), -1),
        seed_step=as_int(keyed.get("seed_step"), 1),
        candidate_count=as_int(keyed.get("candidate_count"), 1),
        restore_faces=bool(keyed.get("restore_faces", False)),
        native_reference_enabled=bool(
            keyed.get("native_reference_enabled", True)
        ),
        save_target=bool(keyed.get("save_target", True)),
        save_generated_canvas=bool(
            keyed.get("save_generated_canvas", False)
        ),
        save_input_canvas=bool(keyed.get("save_input_canvas", False)),
        save_mask=bool(keyed.get("save_mask", False)),
    )
    return (
        request,
        str(keyed.get("gallery_content") or "Target only"),
        str(keyed.get("insert_mode") or "At end"),
    )


def _normalize_lora_name(name: str) -> set[str]:
    text = str(name or "").strip().replace("\\", "/")
    if not text:
        return set()
    without_ext = str(Path(text).with_suffix("")).replace("\\", "/")
    return {
        text.casefold(),
        without_ext.casefold(),
        Path(text).name.casefold(),
        Path(without_ext).name.casefold(),
    }


def missing_enabled_loras(
    request: ReferenceGenerationRequest,
    available: list[str] | None = None,
) -> list[str]:
    if not request.require_enabled_loras:
        return []
    available_names: set[str] = set()
    installed = (
        available
        if available is not None
        else list_installed_lora_names()
    )
    for item in installed:
        available_names.update(_normalize_lora_name(item))
    missing = []
    for enabled, name in (
        (request.edit_lora_enabled, request.edit_lora_name),
        (request.extend_lora_enabled, request.extend_lora_name),
    ):
        if enabled and name and not (
            _normalize_lora_name(name) & available_names
        ):
            missing.append(name)
    return missing


def _selected_gallery_image(
    gallery_value,
    selected_index,
) -> tuple[Image.Image | None, int]:
    items = list(gallery_value or [])
    if not items:
        return None, -1
    try:
        index = int(selected_index)
    except (TypeError, ValueError):
        index = -1
    if index < 0 or index >= len(items):
        index = len(items) - 1
    return _coerce_gallery_item_to_pil(items[index]), index


def load_selected_reference(gallery_value, selected_index):
    image, _ = _selected_gallery_image(gallery_value, selected_index)
    return image


def _reference_error(gallery_value, message: str):
    return (
        gallery_value,
        message,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _new_gallery_items(
    result: ReferenceGenerationResult,
    gallery_content: str,
) -> tuple[list[Image.Image], list[str]]:
    images: list[Image.Image] = []
    infotexts: list[str] = []
    for output in result.outputs:
        images.append(output.target_image)
        infotexts.append(output.infotext)
        if gallery_content == "Target + generated canvas":
            images.append(output.generated_canvas)
            infotexts.append(output.infotext)
    return images, infotexts


def _merge_gallery(
    gallery_value,
    selected_index: int,
    new_images: list[Image.Image],
    new_infotexts: list[str],
    current_info_json: str,
    insert_mode: str,
) -> tuple[list[Any], str]:
    current = list(gallery_value or [])
    try:
        payload = json.loads(current_info_json) if current_info_json else {}
    except Exception:
        payload = {}
    existing_info = list(payload.get("infotexts") or [])
    while len(existing_info) < len(current):
        existing_info.append("")

    if insert_mode == "Replace gallery":
        updated = list(new_images)
        merged_info = list(new_infotexts)
    elif insert_mode == "After selected" and 0 <= selected_index < len(current):
        at = selected_index + 1
        updated = current[:at] + new_images + current[at:]
        merged_info = (
            existing_info[:at] + new_infotexts + existing_info[at:]
        )
    else:
        updated = current + new_images
        merged_info = existing_info + new_infotexts

    payload["infotexts"] = merged_info
    return updated, json.dumps(payload, ensure_ascii=False)


def handle_anima_reference_click(
    gallery_value,
    selected_index,
    *all_values,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate and insert Feature 6 outputs into the normal T2I gallery."""

    del progress  # Gradio injects it for tqdm forwarding; no direct calls needed.
    expected = len(REFERENCE_ARG_KEYS) + _REFERENCE_EXTRA_INPUTS
    if len(all_values) < expected:
        return _reference_error(
            gallery_value,
            "<span style='color:#c33'>Feature 6: missing UI values.</span>",
        )
    values = tuple(all_values[: len(REFERENCE_ARG_KEYS)])
    extras = all_values[len(REFERENCE_ARG_KEYS) :]
    main_prompt = str(extras[0] or "")
    main_negative = str(extras[1] or "")
    current_info_json = str(extras[2] or "")
    keyed = dict(zip(REFERENCE_ARG_KEYS, values))

    image = _coerce_gallery_item_to_pil(keyed.get("reference_image"))
    selected_image, resolved_index = _selected_gallery_image(
        gallery_value, selected_index
    )
    if image is None:
        image = selected_image
    if image is None:
        return _reference_error(
            gallery_value,
            "<span style='color:#c80'>Feature 6: Reference 이미지나 T2I Gallery 선택이 필요합니다.</span>",
        )

    try:
        request, gallery_content, insert_mode = request_from_values(
            values,
            reference_image=image,
            main_prompt=main_prompt,
            main_negative=main_negative,
        )
        request.validate()
        missing = missing_enabled_loras(request)
        if missing:
            names = ", ".join(missing)
            return _reference_error(
                gallery_value,
                (
                    "<span style='color:#c33'>Feature 6: enabled LoRA not "
                    f"found: {names}. Install/refresh it or turn off the "
                    "missing-LoRA guard.</span>"
                ),
            )

        result = run_anima_reference(request)
    except Exception as exc:
        print(
            f"[-] Feature 6 handler failed:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return _reference_error(
            gallery_value,
            f"<span style='color:#c33'>Feature 6 failed: {exc}</span>",
        )

    if not result.outputs:
        return _reference_error(
            gallery_value,
            "<span style='color:#c80'>Feature 6: interrupted or no result.</span>",
        )

    new_images, new_infotexts = _new_gallery_items(
        result, gallery_content
    )
    updated, info_json = _merge_gallery(
        gallery_value,
        resolved_index,
        new_images,
        new_infotexts,
        current_info_json,
        insert_mode,
    )
    latest = new_infotexts[-1] if new_infotexts else ""
    return (
        updated,
        (
            "<span style='color:#383'>Feature 6: "
            f"{len(result.outputs)} candidate(s), "
            f"{len(new_images)} gallery image(s) added.</span>"
        ),
        _plaintext_to_html(latest),
        info_json,
        result.prepared.canvas,
        result.prepared.mask,
    )
