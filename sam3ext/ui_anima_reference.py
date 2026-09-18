"""Feature 6 UI for the Anima Character Reference / ReStyler workflow."""
from __future__ import annotations

import html
import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any

import gradio as gr
from PIL import Image

from .anima_reference_core import ReferenceCanvasConfig, prepare_reference_canvas
from .anima_reference_recipe import (
    FILL_ORIGINAL,
    FILL_SMEAR,
    KEEP_IDENTITY,
    KEEP_OUTFIT,
    KEEP_SCOPES,
    KEEP_STYLE,
    WARN_EXTEND_MISSING,
    DetectedLoras,
    SamplingSettings,
    choose_fallback_index,
    detect_reference_loras,
    edit_strength_for_scope,
    fill_settings,
    forge_preset_sampling,
    prepare_inherited_prompt,
    prompt_syntax_warnings,
    resolve_output_size,
)
from .anima_reference_runner import (
    ReferenceGenerationRequest,
    ReferenceGenerationResult,
    run_anima_reference,
)
from .coerce import as_float, as_int
from .ui_refine import (
    _coerce_gallery_item_to_pil,
    _plaintext_to_html,
    _pull_seed_from_gallery_item,
)

_RESTYLER_URL = "https://civitai.com/models/2803070/anima-restyler"
# Every Feature 6 label starts with this.  Forge restores saved ui-config
# values by label, and bare labels such as "Steps" picked up txt2img values.
_LABEL = "[Ref] "

# Forge's modules/ui_loadsave.py::radio_choices compares a saved value
# against the display label, so (label, value) tuple choices never restore.
# These radios use plain string choices equal to the Korean labels, and
# request_from_values() maps label -> internal value (still accepting the
# internal values themselves, e.g. from a saved ui-config or an old caller).
_KEEP_SCOPE_LABEL_TO_VALUE = {
    "정체성": KEEP_IDENTITY,
    "정체성+의상": KEEP_OUTFIT,
    "정체성+의상+그림체": KEEP_STYLE,
}
_FILL_MODE_LABEL_TO_VALUE = {
    "원본 (단색 그대로)": FILL_ORIGINAL,
    "번진 이미지": FILL_SMEAR,
}


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


@dataclass
class AnimaReferencePanel:
    accordion: gr.Accordion
    selected_index_state: gr.Number
    txt2img_width_fallback: gr.Number
    txt2img_height_fallback: gr.Number
    reference_image: gr.Image
    load_selected_button: gr.Button
    keep_scope: gr.Radio
    candidate_count: gr.Slider
    prompt: gr.Textbox
    generate_button: gr.Button
    stop_button: gr.Button
    status: gr.HTML
    expert_accordion: gr.Accordion
    custom_size_enabled: gr.Checkbox
    output_width: gr.Slider
    output_height: gr.Slider
    composite_megapixels: gr.Slider
    edit_lora_strength: gr.Slider
    extend_lora_enabled: gr.Checkbox
    extend_lora_strength: gr.Slider
    prefix_text: gr.Textbox
    prefix_strength: gr.Slider
    follow_forge_preset: gr.Checkbox
    sampler: gr.Dropdown
    scheduler: gr.Dropdown
    steps: gr.Slider
    cfg_scale: gr.Slider
    shift: gr.Slider
    fill_mode: gr.Radio
    target_color: gr.ColorPicker
    seed: gr.Number
    seed_random_button: gr.Button
    seed_pull_button: gr.Button
    negative_prompt: gr.Textbox
    save_debug_images: gr.Checkbox
    preview_button: gr.Button
    preview_canvas: gr.Image
    preview_mask: gr.Image

    def all_widgets(self) -> list:
        return [getattr(self, name) for name in REFERENCE_ARG_KEYS]


REFERENCE_ARG_KEYS: tuple[str, ...] = (
    "reference_image",
    "keep_scope",
    "prompt",
    "candidate_count",
    "custom_size_enabled",
    "output_width",
    "output_height",
    "composite_megapixels",
    "edit_lora_strength",
    "extend_lora_enabled",
    "extend_lora_strength",
    "prefix_text",
    "prefix_strength",
    "follow_forge_preset",
    "sampler",
    "scheduler",
    "steps",
    "cfg_scale",
    "shift",
    "fill_mode",
    "target_color",
    "seed",
    "negative_prompt",
    "save_debug_images",
)

# main prompt, main negative, generation_info, txt2img width, txt2img height
_REFERENCE_EXTRA_INPUTS = 5


def build_anima_reference_panel(
    samplers: list[str],
    schedulers: list[str],
) -> AnimaReferencePanel:
    """Render Feature 6: one image in, everything else from the recipe."""

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
        "Feature 6 — Anima Character Reference",
        open=False,
        elem_id="sam3_anima_reference_panel",
    ) as accordion:
        selected_index_state = gr.Number(
            value=-1, precision=0, visible=False,
            elem_id="sam3_anima_reference_selected_index",
        )
        # Used only if Forge's txt2img width/height sliders were not captured.
        txt2img_width_fallback = gr.Number(
            value=0, precision=0, visible=False,
            elem_id="sam3_anima_reference_width_fallback",
        )
        txt2img_height_fallback = gr.Number(
            value=0, precision=0, visible=False,
            elem_id="sam3_anima_reference_height_fallback",
        )
        gr.Markdown(
            "캐릭터 이미지 하나만 넣으면 원본 "
            f"[Anima ReStyler v1.2]({_RESTYLER_URL}) 방식으로 그 캐릭터를 "
            "레퍼런스합니다. 비워 두면 선택한 T2I 이미지를 씁니다. 결과 크기와 "
            "프롬프트는 txt2img 설정을 따릅니다. 필요: `models/Lora`의 "
            "**AnimeEditV2**. 권장 입력: 단순한 배경, 중립 포즈."
        )
        with gr.Row():
            reference_image = gr.Image(
                type="pil",
                image_mode="RGBA",
                sources=["upload", "clipboard"],
                label=_LABEL + "캐릭터 이미지",
                height=360,
                elem_id="sam3_anima_reference_image",
            )
            with gr.Column():
                load_selected_button = gr.Button(
                    "📋 선택한 T2I 이미지 가져오기",
                    elem_id="sam3_anima_reference_load_selected",
                )
                keep_scope = gr.Radio(
                    choices=list(_KEEP_SCOPE_LABEL_TO_VALUE),
                    value="정체성",
                    label=_LABEL + "유지 범위",
                    elem_id="sam3_anima_reference_keep_scope",
                )
                candidate_count = gr.Slider(
                    label=_LABEL + "후보 수",
                    value=2, minimum=1, maximum=16, step=1,
                    elem_id="sam3_anima_reference_candidates",
                )
        prompt = gr.Textbox(
            value="",
            lines=2,
            label=_LABEL + "프롬프트 (비우면 txt2img 프롬프트)",
            elem_id="sam3_anima_reference_prompt",
        )
        with gr.Row():
            generate_button = gr.Button(
                "▶ 캐릭터 레퍼런스 생성",
                variant="primary",
                elem_id="sam3_anima_reference_generate",
            )
            stop_button = gr.Button(
                "⏹ 중지",
                variant="stop",
                visible=False,
                elem_id="sam3_anima_reference_stop",
            )
        status = gr.HTML(
            "<span>캐릭터 이미지를 넣거나 T2I 이미지를 선택한 뒤 생성하세요.</span>",
            elem_id="sam3_anima_reference_status",
        )

        with gr.Accordion(
            "전문가 설정",
            open=False,
            elem_id="sam3_anima_reference_expert",
        ) as expert_accordion:
            with gr.Row():
                custom_size_enabled = gr.Checkbox(
                    value=False,
                    label=_LABEL + "결과 크기 직접 지정 (끄면 txt2img 크기)",
                    elem_id="sam3_anima_reference_custom_size",
                )
                output_width = gr.Slider(
                    label=_LABEL + "결과 가로",
                    value=960, minimum=64, maximum=4096, step=8,
                    elem_id="sam3_anima_reference_output_width",
                )
                output_height = gr.Slider(
                    label=_LABEL + "결과 세로",
                    value=1088, minimum=64, maximum=4096, step=8,
                    elem_id="sam3_anima_reference_output_height",
                )
            composite_megapixels = gr.Slider(
                label=_LABEL + "캔버스 메가픽셀",
                value=1.4, minimum=0.5, maximum=4.0, step=0.05,
                elem_id="sam3_anima_reference_megapixels",
            )
            with gr.Row():
                edit_lora_strength = gr.Slider(
                    label=_LABEL + "Edit LoRA 강도",
                    value=0.72, minimum=0.0, maximum=2.0, step=0.01,
                    elem_id="sam3_anima_reference_edit_strength",
                )
                extend_lora_enabled = gr.Checkbox(
                    value=False,
                    label=_LABEL + "Extend LoRA 사용",
                    elem_id="sam3_anima_reference_extend_enabled",
                )
                extend_lora_strength = gr.Slider(
                    label=_LABEL + "Extend LoRA 강도",
                    value=0.4, minimum=0.0, maximum=2.0, step=0.01,
                    elem_id="sam3_anima_reference_extend_strength",
                )
            with gr.Row():
                prefix_text = gr.Textbox(
                    value="split screen, multiple views",
                    label=_LABEL + "앞머리 문구",
                    elem_id="sam3_anima_reference_prefix_text",
                )
                prefix_strength = gr.Slider(
                    label=_LABEL + "앞머리 가중치",
                    value=1.2, minimum=0.0, maximum=3.0, step=0.05,
                    elem_id="sam3_anima_reference_prefix_strength",
                )
            follow_forge_preset = gr.Checkbox(
                value=False,
                label=_LABEL + "Forge Anima img2img 프리셋 따르기 (켜면 아래 샘플링 값 무시)",
                elem_id="sam3_anima_reference_forge_preset",
            )
            with gr.Row():
                sampler = gr.Dropdown(
                    choices=sampler_choices,
                    value=sampler_default,
                    label=_LABEL + "샘플러",
                    elem_id="sam3_anima_reference_sampler",
                )
                scheduler = gr.Dropdown(
                    choices=scheduler_choices,
                    value=scheduler_default,
                    label=_LABEL + "스케줄러",
                    elem_id="sam3_anima_reference_scheduler",
                )
            with gr.Row():
                steps = gr.Slider(
                    label=_LABEL + "스텝",
                    value=30, minimum=1, maximum=150, step=1,
                    elem_id="sam3_anima_reference_steps",
                )
                cfg_scale = gr.Slider(
                    label=_LABEL + "CFG",
                    value=5.0, minimum=0.0, maximum=20.0, step=0.1,
                    elem_id="sam3_anima_reference_cfg",
                )
                shift = gr.Slider(
                    label=_LABEL + "Shift",
                    value=3.0, minimum=0.0, maximum=20.0, step=0.1,
                    elem_id="sam3_anima_reference_shift",
                )
            with gr.Row():
                fill_mode = gr.Radio(
                    choices=list(_FILL_MODE_LABEL_TO_VALUE),
                    value="원본 (단색 그대로)",
                    label=_LABEL + "빈 칸 채우기",
                    elem_id="sam3_anima_reference_fill_mode",
                )
                target_color = gr.ColorPicker(
                    value="#000000",
                    label=_LABEL + "빈 칸 색",
                    elem_id="sam3_anima_reference_target_color",
                )
            with gr.Row():
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label=_LABEL + "시드 (-1 = 무작위)",
                    elem_id="sam3_anima_reference_seed",
                )
                seed_random_button = gr.Button(
                    "🎲 -1", elem_id="sam3_anima_reference_seed_random"
                )
                seed_pull_button = gr.Button(
                    "🎯 선택 이미지 시드",
                    elem_id="sam3_anima_reference_seed_pull",
                )
            negative_prompt = gr.Textbox(
                value="",
                lines=2,
                label=_LABEL + "네거티브 프롬프트 (비우면 txt2img 네거티브)",
                elem_id="sam3_anima_reference_negative",
            )
            save_debug_images = gr.Checkbox(
                value=False,
                label=_LABEL + "디버그 이미지 저장 (생성 캔버스·입력 캔버스·마스크)",
                elem_id="sam3_anima_reference_save_debug",
            )
            preview_button = gr.Button(
                "🧩 캔버스 / 마스크 미리보기",
                elem_id="sam3_anima_reference_preview",
            )
            with gr.Row():
                preview_canvas = gr.Image(
                    label=_LABEL + "캔버스 미리보기",
                    interactive=False,
                    elem_id="sam3_anima_reference_preview_canvas",
                )
                preview_mask = gr.Image(
                    label=_LABEL + "마스크 미리보기",
                    interactive=False,
                    elem_id="sam3_anima_reference_preview_mask",
                )

    return AnimaReferencePanel(
        **{
            name: value
            for name, value in locals().items()
            if name in AnimaReferencePanel.__dataclass_fields__
        }
    )


def request_from_values(
    values: tuple[Any, ...],
    *,
    reference_image: Image.Image,
    main_prompt: str,
    main_negative: str,
    txt2img_width: Any = None,
    txt2img_height: Any = None,
    available_loras=(),
    forge_opts: dict[str, Any] | None = None,
) -> tuple[ReferenceGenerationRequest, DetectedLoras, list[str]]:
    """Map the panel vector to a request; everything hidden uses the recipe."""

    keyed = dict(zip(REFERENCE_ARG_KEYS, values))
    scope_raw = str(keyed.get("keep_scope") or KEEP_IDENTITY)
    scope = _KEEP_SCOPE_LABEL_TO_VALUE.get(scope_raw, scope_raw)
    if scope not in KEEP_SCOPES:
        scope = KEEP_IDENTITY
    warnings: list[str] = []

    panel_prompt = str(keyed.get("prompt") or "").strip()
    if panel_prompt:
        prompt = panel_prompt
    else:
        prompt = prepare_inherited_prompt(str(main_prompt or ""), scope)
        warnings.extend(prompt_syntax_warnings(str(main_prompt or "")))
    negative = (
        str(keyed.get("negative_prompt") or "").strip()
        or str(main_negative or "")
    )

    width, height = resolve_output_size(
        bool(keyed.get("custom_size_enabled", False)),
        as_int(keyed.get("output_width"), 960),
        as_int(keyed.get("output_height"), 1088),
        txt2img_width,
        txt2img_height,
    )
    fill_raw = str(keyed.get("fill_mode") or FILL_ORIGINAL)
    fill_choice = _FILL_MODE_LABEL_TO_VALUE.get(fill_raw, fill_raw)
    inpainting_fill, target_color = fill_settings(
        fill_choice,
        str(keyed.get("target_color") or "#000000"),
    )

    panel_sampling = SamplingSettings(
        sampler=str(keyed.get("sampler") or "Euler a"),
        scheduler=str(keyed.get("scheduler") or "Simple"),
        steps=as_int(keyed.get("steps"), 30),
        cfg_scale=as_float(keyed.get("cfg_scale"), 5.0),
        shift=as_float(keyed.get("shift"), 3.0),
    )
    follow_preset = bool(keyed.get("follow_forge_preset", False))
    sampling = (
        forge_preset_sampling(forge_opts or {}, panel_sampling)
        if follow_preset
        else panel_sampling
    )

    detected = detect_reference_loras(available_loras)
    extend_wanted = bool(keyed.get("extend_lora_enabled", False))
    if extend_wanted and detected.extend is None:
        warnings.append(WARN_EXTEND_MISSING)
    debug = bool(keyed.get("save_debug_images", False))

    request = ReferenceGenerationRequest(
        reference_image=reference_image,
        canvas=ReferenceCanvasConfig(
            output_width=width,
            output_height=height,
            target_color=target_color,
            composite_megapixels=as_float(
                keyed.get("composite_megapixels"), 1.4
            ),
        ),
        prompt=prompt,
        negative_prompt=negative,
        prefix_text=str(
            keyed.get("prefix_text") or "split screen, multiple views"
        ),
        prefix_strength=as_float(keyed.get("prefix_strength"), 1.2),
        edit_lora_name=detected.edit or "",
        edit_lora_strength=edit_strength_for_scope(
            scope, as_float(keyed.get("edit_lora_strength"), 0.72)
        ),
        extend_lora_enabled=extend_wanted and detected.extend is not None,
        extend_lora_name=detected.extend or "",
        extend_lora_strength=as_float(
            keyed.get("extend_lora_strength"), 0.4
        ),
        steps=sampling.steps,
        cfg_scale=sampling.cfg_scale,
        shift=sampling.shift,
        sampler=sampling.sampler,
        scheduler=sampling.scheduler,
        inpainting_fill=inpainting_fill,
        seed=as_int(keyed.get("seed"), -1),
        candidate_count=as_int(keyed.get("candidate_count"), 2),
        keep_scope=scope,
        sampling_source="forge_preset" if follow_preset else "recipe",
        save_generated_canvas=debug,
        save_input_canvas=debug,
        save_mask=debug,
    )
    return request, detected, warnings


_SOURCE_LABELS = {"uploaded": "업로드한 이미지", "gallery": "갤러리 이미지"}

# Edit LoRA is trained for the 28-block base Anima UNet; any other block
# count means the global 28->N compat hook (scripts/anima_lora_blocks.py)
# remapped it.
_BASE_MODEL_BLOCKS = 28
# anima38 labels that mean "no connector is active" (as opposed to "checked,
# and it is not applicable"/"installed") belong in the warning line, not the
# green status line.
_ANIMA38_OFF_PREFIXES = (
    "unavailable",
    "missing encoder",
    "install failed",
    "check failed",
)
WARN_ANIMA38_V1_ADAPTER = (
    "3.8B v1 체크포인트는 캐릭터 레퍼런스에서 어댑터를 쓰지 않고 기본 "
    "Anima로 진행했습니다"
)


def _gallery_infotexts(current_info_json: str) -> list[str]:
    try:
        payload = json.loads(current_info_json) if current_info_json else {}
    except Exception:
        return []
    return [str(item or "") for item in (payload.get("infotexts") or [])]


def resolve_reference_image(
    uploaded,
    gallery_value,
    selected_index,
    current_info_json: str,
) -> tuple[Image.Image | None, int, str]:
    """Return (image, gallery index, source) for a run or a preview."""

    if isinstance(uploaded, Image.Image):
        # Keep transparency: the canvas code flattens it onto the matte.
        return uploaded, -1, "uploaded"
    image = _coerce_gallery_item_to_pil(uploaded)
    if image is not None:
        return image, -1, "uploaded"
    items = list(gallery_value or [])
    index = choose_fallback_index(
        len(items), selected_index, _gallery_infotexts(current_info_json)
    )
    if index < 0:
        return None, -1, "none"
    return _coerce_gallery_item_to_pil(items[index]), index, "gallery"


def load_selected_reference(gallery_value, selected_index):
    """Explicit button: load exactly the selected (or last) gallery image."""

    items = list(gallery_value or [])
    if not items:
        return None
    try:
        index = int(selected_index)
    except (TypeError, ValueError):
        index = -1
    if not 0 <= index < len(items):
        index = len(items) - 1
    return _coerce_gallery_item_to_pil(items[index])


def _forge_opts_data() -> dict[str, Any]:
    try:
        from modules import shared

        return dict(getattr(shared.opts, "data", {}) or {})
    except Exception:
        return {}


def _run_exclusive(job: str, fn):
    """Run like Forge's own Generate: one job at a time, fresh stop flags."""

    from modules import shared
    from modules.call_queue import queue_lock

    with queue_lock:
        shared.state.begin(job=job)
        try:
            return fn()
        finally:
            shared.state.end()


def _split_inputs(all_values):
    expected = len(REFERENCE_ARG_KEYS) + _REFERENCE_EXTRA_INPUTS
    if len(all_values) < expected:
        return None
    values = tuple(all_values[: len(REFERENCE_ARG_KEYS)])
    main_prompt, main_negative, info_json, width, height = all_values[
        len(REFERENCE_ARG_KEYS) : expected
    ]
    return (
        values,
        str(main_prompt or ""),
        str(main_negative or ""),
        str(info_json or ""),
        width,
        height,
    )


def _reference_error(gallery_value, message: str):
    return (
        gallery_value,
        message,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def format_reference_status(
    result: ReferenceGenerationResult,
    request: ReferenceGenerationRequest,
    detected: DetectedLoras,
    warnings: list[str],
    source: str,
) -> str:
    diagnostics = dict(result.diagnostics or {})
    parts = [
        f"{len(result.outputs)}장 생성",
        f"레퍼런스: {_SOURCE_LABELS.get(source, source)}",
        f"Edit LoRA: {detected.edit}",
    ]
    if request.extend_lora_enabled:
        parts.append(f"Extend LoRA: {request.extend_lora_name}")
    panel = diagnostics.get("target_panel")
    if panel:
        parts.append(
            f"생성 영역 {panel[0]}×{panel[1]} "
            f"(확대 {float(diagnostics.get('upscale', 1.0)):.2f}배)"
        )
    blocks = diagnostics.get("model_blocks")
    parts.append(f"모델 {blocks}블록" if blocks else "모델 블록 수 확인 불가")
    if isinstance(blocks, int) and blocks != _BASE_MODEL_BLOCKS:
        parts.append(
            f"Edit LoRA {_BASE_MODEL_BLOCKS}블록 → {blocks}블록 모델 "
            "(블록 호환 훅으로 적용)"
        )

    anima38_label = str(diagnostics.get("anima38", "확인 불가"))
    extra_warnings = list(warnings)
    if anima38_label.startswith(_ANIMA38_OFF_PREFIXES):
        parts.append("3.8B 커넥터: 꺼짐")
        extra_warnings.append(f"3.8B 커넥터 사용 불가 ({anima38_label})")
    else:
        parts.append(f"3.8B 커넥터: {anima38_label}")
        if blocks == 52 and anima38_label == "not a 3.8B v2 bundle":
            extra_warnings.append(WARN_ANIMA38_V1_ADAPTER)

    parts.append("시드 " + ", ".join(str(item.seed) for item in result.outputs))
    text = (
        "<span style='color:#383'>"
        + html.escape(" · ".join(parts))
        + "</span>"
    )
    if diagnostics.get("interrupted"):
        text += "<br><span style='color:#c80'>중단되어 일부 후보만 생성했습니다.</span>"
    if extra_warnings:
        text += (
            "<br><span style='color:#c80'>⚠ "
            + html.escape(" / ".join(extra_warnings))
            + "</span>"
        )
    return text


def preview_reference_layout(gallery_value, selected_index, *all_values):
    """Preview exactly the canvas and mask a Generate click would use."""

    split = _split_inputs(all_values)
    if split is None:
        return (
            gr.update(),
            gr.update(),
            "<span style='color:#c33'>Feature 6: missing UI values.</span>",
        )
    values, main_prompt, main_negative, info_json, width, height = split
    keyed = dict(zip(REFERENCE_ARG_KEYS, values))
    image, _, source = resolve_reference_image(
        keyed.get("reference_image"), gallery_value, selected_index, info_json
    )
    if image is None:
        return (
            gr.update(),
            gr.update(),
            "<span style='color:#c80'>캐릭터 이미지를 넣거나 T2I 이미지를 선택해 주세요.</span>",
        )
    try:
        request, _, _ = request_from_values(
            values,
            reference_image=image,
            main_prompt=main_prompt,
            main_negative=main_negative,
            txt2img_width=width,
            txt2img_height=height,
        )
        prepared = prepare_reference_canvas(image, request.canvas)
    except Exception as exc:
        return (
            gr.update(),
            gr.update(),
            f"<span style='color:#c33'>미리보기 실패: {html.escape(str(exc))}</span>",
        )
    left, top, right, bottom = prepared.target_box
    return (
        prepared.canvas,
        prepared.mask,
        (
            "<span style='color:#383'>"
            f"캔버스 {prepared.canvas.width}×{prepared.canvas.height} · "
            f"생성 영역 {right - left}×{bottom - top} → 결과 "
            f"{prepared.output_size[0]}×{prepared.output_size[1]} · "
            f"레퍼런스: {_SOURCE_LABELS.get(source, source)}</span>"
        ),
    )


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
    """Generate candidates and append them to the normal T2I gallery."""

    del progress  # Gradio injects it for tqdm forwarding.
    split = _split_inputs(all_values)
    if split is None:
        return _reference_error(
            gallery_value,
            "<span style='color:#c33'>Feature 6: missing UI values.</span>",
        )
    values, main_prompt, main_negative, info_json, width, height = split
    keyed = dict(zip(REFERENCE_ARG_KEYS, values))
    image, index, source = resolve_reference_image(
        keyed.get("reference_image"), gallery_value, selected_index, info_json
    )
    if image is None:
        return _reference_error(
            gallery_value,
            "<span style='color:#c80'>캐릭터 이미지를 넣거나 T2I 이미지를 선택해 주세요.</span>",
        )

    def build(refresh: bool):
        return request_from_values(
            values,
            reference_image=image,
            main_prompt=main_prompt,
            main_negative=main_negative,
            txt2img_width=width,
            txt2img_height=height,
            available_loras=list_installed_lora_names(refresh=refresh),
            forge_opts=_forge_opts_data(),
        )

    try:
        request, detected, warnings = build(False)
        extend_wanted = bool(keyed.get("extend_lora_enabled", False))
        if detected.edit is None or (extend_wanted and detected.extend is None):
            # A LoRA copied in after startup is only seen after a rescan.
            request, detected, warnings = build(True)
        if detected.edit is None:
            return _reference_error(
                gallery_value,
                "<span style='color:#c33'>Edit LoRA(AnimeEditV2)를 "
                "models/Lora에서 찾지 못했습니다. 설치한 뒤 다시 시도하세요.</span>",
            )
        request.validate()
        result = _run_exclusive(
            "sam3_character_reference",
            lambda: run_anima_reference(request),
        )
    except Exception as exc:
        print(
            f"[-] Feature 6 handler failed:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return _reference_error(
            gallery_value,
            f"<span style='color:#c33'>Feature 6 실패: {html.escape(str(exc))}</span>",
        )

    if not result.outputs:
        return _reference_error(
            gallery_value,
            "<span style='color:#c80'>중단됐거나 결과가 없습니다.</span>",
        )

    new_images = [item.target_image for item in result.outputs]
    new_infotexts = [item.infotext for item in result.outputs]
    updated, info_json_out = _merge_gallery(
        gallery_value, index, new_images, new_infotexts, info_json, "At end"
    )
    pin = gr.update(value=image) if source == "gallery" else gr.update()
    return (
        updated,
        format_reference_status(result, request, detected, warnings, source),
        _plaintext_to_html(new_infotexts[-1]),
        info_json_out,
        result.prepared.canvas,
        result.prepared.mask,
        pin,
    )


# Job names the runner sets while a Feature 6 candidate is running (see
# run_anima_reference and _run_exclusive's "sam3_character_reference" job).
_FEATURE6_JOB_PREFIXES = (
    "sam3_character_reference",
    "Anima Reference",
    "Anima Character Reference",
)


def _stop_reference():
    """Interrupt only if Feature 6 itself currently holds the job.

    If txt2img holds the queue lock while Feature 6's click waits, blindly
    setting these flags would stop txt2img instead.
    """

    from modules import shared

    job = str(shared.state.job or "")
    if not job.startswith(_FEATURE6_JOB_PREFIXES):
        return
    shared.state.interrupted = True
    shared.state.skipped = True


def wire_anima_reference_panel(
    panel: AnimaReferencePanel,
    *,
    gallery,
    main_prompt,
    main_negative,
    html_info,
    generation_info,
    width,
    height,
    selected_index_js: str,
) -> None:
    """Wire Feature 6 without adding its controls to the main Generate."""

    width = width if width is not None else panel.txt2img_width_fallback
    height = height if height is not None else panel.txt2img_height_fallback
    run_inputs = [
        gallery,
        panel.selected_index_state,
        *panel.all_widgets(),
        main_prompt,
        main_negative,
        generation_info,
        width,
        height,
    ]

    panel.load_selected_button.click(
        fn=load_selected_reference,
        js=selected_index_js,
        inputs=[gallery, panel.selected_index_state],
        outputs=[panel.reference_image],
        queue=False,
        show_progress="hidden",
    )
    panel.preview_button.click(
        fn=preview_reference_layout,
        js=selected_index_js,
        inputs=run_inputs,
        outputs=[panel.preview_canvas, panel.preview_mask, panel.status],
        queue=False,
        show_progress="hidden",
    )
    show_stop = panel.generate_button.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[panel.generate_button, panel.stop_button],
        queue=False,
    )
    run = show_stop.then(
        fn=handle_anima_reference_click,
        js=selected_index_js,
        inputs=run_inputs,
        outputs=[
            gallery,
            panel.status,
            html_info,
            generation_info,
            panel.preview_canvas,
            panel.preview_mask,
            panel.reference_image,
        ],
    )
    run.then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[panel.generate_button, panel.stop_button],
        queue=False,
    )
    panel.stop_button.click(
        fn=_stop_reference, inputs=[], outputs=[], queue=False
    )
    panel.seed_random_button.click(
        fn=lambda: -1, inputs=[], outputs=[panel.seed], queue=False
    )
    panel.seed_pull_button.click(
        fn=_pull_seed_from_gallery_item,
        js=selected_index_js,
        inputs=[gallery, panel.selected_index_state, generation_info],
        outputs=[panel.seed],
        queue=False,
    )
