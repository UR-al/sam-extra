from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import gradio as gr

from .__version__ import __version__
from .args import ALL_ARGS
from .core import SAM3_NAME, find_checkpoint_options

try:
    from modules.sd_samplers import all_samplers as _all_samplers
except Exception:
    _all_samplers = []

try:
    from modules.sd_schedulers import schedulers as _all_schedulers
except Exception:
    _all_schedulers = []


_CN_MODEL_EXTS = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")


def _scan_sam3_dir_for_cn_models() -> dict[str, str]:
    """Scan ``models/sam3/`` for ControlNet-compatible weight files (LLLite
    inpaint, custom CN checkpoints kept next to the SAM3 weights, etc.) and
    return a ``{name: path}`` map.

    SAM3 detection checkpoints themselves are excluded by name pattern — they
    live in the SAM3 Checkpoint dropdown, not the CN model dropdown.
    """
    import os as _os
    from pathlib import Path as _Path

    try:
        from modules import paths as _paths
    except Exception:
        return {}

    models_root = _Path(_paths.models_path)
    candidates: list[_Path] = []
    # Match both `models/sam3` (existing convention) and `models/SAM3` (user
    # may rename); the actual folder on disk wins.
    for variant in ("sam3", "SAM3"):
        target = models_root / variant
        if not target.is_dir():
            continue
        for ext in _CN_MODEL_EXTS:
            candidates.extend(sorted(target.glob(f"*{ext}")))

    found: dict[str, str] = {}
    for path in candidates:
        stem = path.stem
        # Skip SAM3 detection checkpoints — those are handled by
        # find_checkpoint_options() and would only confuse the CN dropdown.
        if stem.lower().startswith("sam3"):
            continue
        # First-wins (sam3 vs SAM3 dedupe by stem)
        found.setdefault(stem, str(path))
    return found


def _controlnet_model_choices() -> list[str]:
    """Return ControlNet model filenames, with ``None`` when the
    sd_forge_controlnet extension isn't loaded so the UI still renders.

    Also surfaces non-SAM3 weights stored in ``models/sam3/`` (e.g.
    ``anima-lllite-inpainting-v2.safetensors``) by registering them into
    ``global_state.controlnet_filename_dict`` so the CN script can resolve
    the name → path mapping at load time.
    """
    try:
        from lib_controlnet import global_state

        global_state.update_controlnet_filenames()
        extras = _scan_sam3_dir_for_cn_models()
        if extras:
            global_state.controlnet_filename_dict.update(extras)
            global_state.controlnet_names = sorted(global_state.controlnet_filename_dict.keys())
        names = list(global_state.get_all_controlnet_names())
        return names or ["None"]
    except Exception:
        return ["None"]


def _controlnet_module_choices() -> list[str]:
    try:
        from lib_controlnet import global_state

        names = list(global_state.get_all_preprocessor_names())
        return names or ["None"]
    except Exception:
        return ["None"]


def _default_cn_module(choices: list[str]) -> str:
    for preferred in ("inpaint_only", "inpaint_global_harmonious", "None"):
        if preferred in choices:
            return preferred
    return choices[0] if choices else "None"


class Widgets(SimpleNamespace):
    def tolist(self):
        return [getattr(self, attr) for attr in ALL_ARGS.attrs]


@dataclass
class WebuiButtons:
    t2i_button: gr.Button | None
    i2i_button: gr.Button | None


def on_widget_change(state: dict, value: Any, *, attr: str):
    state = dict(state or {})
    state[attr] = value
    return state


def on_generate_click(state: dict, sam3_enable: Any, *values: Any):
    state = dict(state or {})
    state["sam3_enable"] = sam3_enable
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value
    state["is_api"] = ()
    return state


def state_init(w: Widgets) -> dict[str, Any]:
    return {
        "sam3_enable": getattr(w, "sam3_enable").value,
        **{attr: getattr(w, attr).value for attr in ALL_ARGS.attrs},
    }


def sam3_ui(is_img2img: bool, buttons: WebuiButtons):
    w = Widgets()
    tab = "img2img" if is_img2img else "txt2img"
    eid = lambda name: f"sam3_{tab}_{name}"

    with gr.Accordion(SAM3_NAME, open=False, elem_id=eid("accordion")):
        with gr.Row():
            with gr.Column(scale=3):
                w.sam3_enable = gr.Checkbox(label="Enable SAM3", value=False, elem_id=eid("enable"))
            with gr.Column(scale=5):
                gr.Markdown("SAM3 local mask refinement")
            with gr.Column(scale=1, min_width=180):
                gr.Markdown(f"v{__version__}")

        with gr.Row():
            w.sam3_prompt = gr.Textbox(
                value="face",
                label="SAM3 Detect Prompt",
                lines=2,
                placeholder="e.g. face, eyes, hair / hand   ( ',' = OR-merge into one mask, '/' = separate detailer pass )",
                elem_id=eid("prompt"),
            )

        with gr.Row():
            w.sam3_exclude_prompt = gr.Textbox(
                value="",
                label="SAM3 Exclude Prompt",
                lines=1,
                placeholder="Optional. Detect+subtract from main mask. e.g. 'face, eyes' protects those from a 'clothes' mask.",
                elem_id=eid("exclude_prompt"),
            )

        with gr.Row():
            w.sam3_inpaint_prompt = gr.Textbox(
                value="",
                label="SAM3 Inpaint Prompt",
                lines=2,
                placeholder="Optional inpaint prompt. Blank uses the main prompt.",
                elem_id=eid("inpaint_prompt"),
            )

        with gr.Row():
            w.sam3_negative_prompt = gr.Textbox(
                value="",
                label="SAM3 Negative Prompt",
                lines=2,
                placeholder="Optional inpaint negative prompt. Blank uses the main negative prompt.",
                elem_id=eid("negative_prompt"),
            )

        with gr.Row():
            w.sam3_mode = gr.Dropdown(
                label="SAM3 Mode",
                choices=["Mask only", "Inpaint"],
                value="Inpaint",
                type="value",
                elem_id=eid("mode"),
            )
            w.sam3_mask_mode = gr.Dropdown(
                label="Mask Processing",
                choices=["Individual", "Combined"],
                value="Individual",
                type="value",
                elem_id=eid("mask_mode"),
            )
            w.sam3_threshold = gr.Slider(
                label="SAM3 Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.40,
                elem_id=eid("threshold"),
            )
            w.sam3_mask_dilation = gr.Slider(
                label="Mask Dilation (px)",
                minimum=0,
                maximum=256,
                step=1,
                value=0,
                elem_id=eid("mask_dilation"),
            )
            w.sam3_mask_hull = gr.Checkbox(
                label="Convex Hull (wrap strands)",
                value=False,
                elem_id=eid("mask_hull"),
            )
            w.sam3_mask_outline_px = gr.Slider(
                label="Outline expand (edge-aware, px) — catches outline residue",
                minimum=0,
                maximum=64,
                step=1,
                value=0,
                elem_id=eid("mask_outline_px"),
            )
            checkpoint_choices = find_checkpoint_options()
            w.sam3_checkpoint = gr.Dropdown(
                label="SAM3 Checkpoint",
                elem_id=eid("checkpoint"),
                choices=checkpoint_choices,
                value=checkpoint_choices[0] if checkpoint_choices else "sam3.pt",
                type="value",
            )

        with gr.Row():
            w.sam3_device = gr.Dropdown(
                label="SAM3 Device",
                choices=["auto", "cuda", "cpu"],
                value="auto",
                type="value",
                elem_id=eid("device"),
            )
            w.sam3_preview_overlay = gr.Checkbox(
                label="Replace output with overlay preview",
                value=False,
                elem_id=eid("preview_overlay"),
            )
            w.sam3_save_artifacts = gr.Checkbox(
                label="Save mask/overlay artifacts",
                value=True,
                elem_id=eid("save_artifacts"),
            )
            w.sam3_unload_after = gr.Checkbox(
                label="Unload SAM3 from VRAM after detection (~3.5 GB) — recommended for ≤16 GB GPUs",
                value=True,
                elem_id=eid("unload_after"),
            )

        with gr.Accordion("Inpaint", open=False, elem_id=eid("inpaint_section")):
            with gr.Row():
                w.sam3_denoising_strength = gr.Slider(
                    label="Denoising Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.40,
                    elem_id=eid("denoising_strength"),
                )
                w.sam3_mask_blur = gr.Slider(
                    label="Mask Blur",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=4,
                    elem_id=eid("mask_blur"),
                )

            with gr.Row():
                w.sam3_inpainting_fill = gr.Dropdown(
                    label="Masked content (init for masked area)",
                    choices=["fill", "original", "latent noise", "latent nothing"],
                    value="original",
                    type="value",
                    elem_id=eid("inpainting_fill"),
                )
                w.sam3_inpaint_only_masked = gr.Checkbox(
                    label="Inpaint only masked",
                    value=True,
                    elem_id=eid("inpaint_only_masked"),
                )
                w.sam3_inpaint_only_masked_padding = gr.Slider(
                    label="Inpaint padding",
                    minimum=0,
                    maximum=256,
                    step=1,
                    value=32,
                    elem_id=eid("inpaint_padding"),
                )

            with gr.Row():
                w.sam3_use_inpaint_width_height = gr.Checkbox(
                    label="Use separate inpaint width/height",
                    value=False,
                    elem_id=eid("use_wh"),
                )
                w.sam3_inpaint_width = gr.Slider(
                    label="Inpaint Width",
                    minimum=64,
                    maximum=2048,
                    step=8,
                    value=512,
                    elem_id=eid("inpaint_width"),
                )
                w.sam3_inpaint_height = gr.Slider(
                    label="Inpaint Height",
                    minimum=64,
                    maximum=2048,
                    step=8,
                    value=512,
                    elem_id=eid("inpaint_height"),
                )

            with gr.Row():
                w.sam3_use_steps = gr.Checkbox(
                    label="Use separate steps",
                    value=False,
                    elem_id=eid("use_steps"),
                )
                w.sam3_steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=28,
                    elem_id=eid("steps"),
                )
                w.sam3_use_cfg_scale = gr.Checkbox(
                    label="Use separate CFG scale",
                    value=False,
                    elem_id=eid("use_cfg"),
                )
                w.sam3_cfg_scale = gr.Slider(
                    label="CFG Scale",
                    minimum=0.0,
                    maximum=30.0,
                    step=0.1,
                    value=7.0,
                    elem_id=eid("cfg_scale"),
                )

            with gr.Row():
                w.sam3_use_sampler = gr.Checkbox(
                    label="Use separate sampler",
                    value=False,
                    elem_id=eid("use_sampler"),
                )
                w.sam3_sampler = gr.Dropdown(
                    label="Sampler",
                    choices=["Use same sampler", *[s.name for s in _all_samplers]],
                    value="Use same sampler",
                    type="value",
                    elem_id=eid("sampler"),
                )
                w.sam3_use_scheduler = gr.Checkbox(
                    label="Use separate scheduler",
                    value=False,
                    elem_id=eid("use_scheduler"),
                )
                w.sam3_scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=["Use same scheduler", *[s.label for s in _all_schedulers]],
                    value="Use same scheduler",
                    type="value",
                    elem_id=eid("scheduler"),
                )

            with gr.Row():
                w.sam3_use_seed = gr.Checkbox(
                    label="Use specified seed (instead of parent's)",
                    value=False,
                    elem_id=eid("use_seed"),
                )
                w.sam3_seed = gr.Number(
                    label="Seed (-1 = random)",
                    value=-1,
                    precision=0,
                    elem_id=eid("seed"),
                )

            with gr.Row():
                w.sam3_use_noise_multiplier = gr.Checkbox(
                    label="Use noise multiplier",
                    value=False,
                    elem_id=eid("use_noise_mult"),
                )
                w.sam3_noise_multiplier = gr.Slider(
                    label="Noise Multiplier",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.01,
                    value=1.0,
                    elem_id=eid("noise_mult"),
                )
                w.sam3_restore_face = gr.Checkbox(
                    label="Restore face",
                    value=False,
                    elem_id=eid("restore_face"),
                )

        with gr.Accordion("ControlNet", open=False, elem_id=eid("cn_section")):
            cn_models = _controlnet_model_choices()
            cn_modules = _controlnet_module_choices()
            cn_module_default = _default_cn_module(cn_modules)

            gr.Markdown(
                "**Tip**: LLLite inpaint models (`anima-lllite-inpainting-*`) take a 4-channel "
                "RGB+mask cond and need the mask to survive preprocessing. The extension "
                "auto-overrides the Preprocessor to `None` when it detects such a model."
            )

            with gr.Row():
                w.sam3_cn_enable = gr.Checkbox(
                    label="Enable ControlNet for SAM3 inpaint",
                    value=False,
                    elem_id=eid("cn_enable"),
                )
                w.sam3_cn_override_external = gr.Checkbox(
                    label="Override external CN units (disable other slots during SAM3 pass)",
                    value=False,
                    elem_id=eid("cn_override"),
                )
                w.sam3_cn_pixel_perfect = gr.Checkbox(
                    label="Pixel Perfect",
                    value=True,
                    elem_id=eid("cn_pp"),
                )

            with gr.Row():
                w.sam3_cn_module = gr.Dropdown(
                    label="Preprocessor",
                    choices=cn_modules,
                    value=cn_module_default,
                    type="value",
                    elem_id=eid("cn_module"),
                )
                w.sam3_cn_model = gr.Dropdown(
                    label="Model",
                    choices=cn_models,
                    value=cn_models[0] if cn_models else "None",
                    type="value",
                    elem_id=eid("cn_model"),
                )

            with gr.Row():
                w.sam3_cn_weight = gr.Slider(
                    label="Weight",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                    elem_id=eid("cn_weight"),
                )
                w.sam3_cn_guidance_start = gr.Slider(
                    label="Guidance Start",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    elem_id=eid("cn_gstart"),
                )
                w.sam3_cn_guidance_end = gr.Slider(
                    label="Guidance End",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                    elem_id=eid("cn_gend"),
                )

            with gr.Row():
                w.sam3_cn_control_mode = gr.Radio(
                    label="Control Mode",
                    choices=[
                        "Balanced",
                        "My prompt is more important",
                        "ControlNet is more important",
                    ],
                    value="Balanced",
                    elem_id=eid("cn_control_mode"),
                )
                w.sam3_cn_resize_mode = gr.Radio(
                    label="Resize Mode",
                    choices=["Just Resize", "Crop and Resize", "Resize and Fill"],
                    value="Crop and Resize",
                    elem_id=eid("cn_resize_mode"),
                )

            with gr.Row():
                w.sam3_cn_processor_res = gr.Slider(
                    label="Preprocessor Resolution",
                    minimum=64,
                    maximum=2048,
                    step=8,
                    value=512,
                    elem_id=eid("cn_procres"),
                )
                w.sam3_cn_threshold_a = gr.Slider(
                    label="Threshold A (-1 = unused)",
                    minimum=-1,
                    maximum=256,
                    step=1,
                    value=-1,
                    elem_id=eid("cn_ta"),
                )
                w.sam3_cn_threshold_b = gr.Slider(
                    label="Threshold B (-1 = unused)",
                    minimum=-1,
                    maximum=256,
                    step=1,
                    value=-1,
                    elem_id=eid("cn_tb"),
                )

    state = gr.State(state_init(w))
    for attr in ("sam3_enable", *ALL_ARGS.attrs):
        widget = getattr(w, attr)
        on_change = partial(on_widget_change, attr=attr)
        widget.change(fn=on_change, inputs=[state, widget], outputs=state, queue=False)

    target_button = buttons.i2i_button if is_img2img else buttons.t2i_button
    if target_button is not None:
        all_inputs = [state, w.sam3_enable, *w.tolist()]
        target_button.click(fn=on_generate_click, inputs=all_inputs, outputs=state, queue=False)

    infotext_fields = [(getattr(w, attr), name) for attr, name in ALL_ARGS]
    return [w.sam3_enable, state], infotext_fields
