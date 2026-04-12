from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import gradio as gr

from .__version__ import __version__
from .args import ALL_ARGS
from .core import SAM3_NAME, find_checkpoint_options


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


def on_generate_click(state: dict, *values: Any):
    state = dict(state or {})
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value
    state["is_api"] = ()
    return state


def state_init(w: Widgets) -> dict[str, Any]:
    return {attr: getattr(w, attr).value for attr in ALL_ARGS.attrs}


def sam3_ui(is_img2img: bool, buttons: WebuiButtons):
    w = Widgets()

    with gr.Accordion(SAM3_NAME, open=False):
        with gr.Row():
            with gr.Column(scale=3):
                w.sam3_enable = gr.Checkbox(label="Enable SAM3", value=False)
            with gr.Column(scale=5):
                gr.Markdown("SAM3 local mask refinement")
            with gr.Column(scale=1, min_width=180):
                gr.Markdown(f"v{__version__}")

        with gr.Row():
            w.sam3_prompt = gr.Textbox(
                value="face",
                label="SAM3 Detect Prompt",
                lines=2,
                placeholder="What SAM3 should segment, e.g. face / hand / hair / person",
            )

        with gr.Row():
            w.sam3_inpaint_prompt = gr.Textbox(
                value="",
                label="SAM3 Inpaint Prompt",
                lines=2,
                placeholder="Optional inpaint prompt. Blank uses the main prompt.",
            )

        with gr.Row():
            w.sam3_negative_prompt = gr.Textbox(
                value="",
                label="SAM3 Negative Prompt",
                lines=2,
                placeholder="Optional inpaint negative prompt. Blank uses the main negative prompt.",
            )

        with gr.Row():
            w.sam3_mode = gr.Dropdown(
                label="SAM3 Mode",
                choices=["Mask only", "Inpaint"],
                value="Inpaint",
                type="value",
            )
            w.sam3_mask_mode = gr.Dropdown(
                label="Mask Processing",
                choices=["Individual", "Combined"],
                value="Individual",
                type="value",
            )
            w.sam3_threshold = gr.Slider(
                label="SAM3 Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.40,
            )
            checkpoint_choices = find_checkpoint_options()
            w.sam3_checkpoint = gr.Dropdown(
                label="SAM3 Checkpoint",
                choices=checkpoint_choices,
                value=checkpoint_choices[0] if checkpoint_choices else "models/sam3.pt",
                type="value",
            )

        with gr.Row():
            w.sam3_device = gr.Dropdown(
                label="SAM3 Device",
                choices=["auto", "cuda", "cpu"],
                value="auto",
                type="value",
            )
            w.sam3_preview_overlay = gr.Checkbox(
                label="Replace output with overlay preview",
                value=False,
            )
            w.sam3_save_artifacts = gr.Checkbox(
                label="Save mask/overlay artifacts",
                value=True,
            )

        with gr.Accordion("Inpaint", open=False):
            with gr.Row():
                w.sam3_denoising_strength = gr.Slider(
                    label="Denoising Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.40,
                )
                w.sam3_mask_blur = gr.Slider(
                    label="Mask Blur",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=4,
                )

            with gr.Row():
                w.sam3_inpaint_only_masked = gr.Checkbox(
                    label="Inpaint only masked",
                    value=True,
                )
                w.sam3_inpaint_only_masked_padding = gr.Slider(
                    label="Inpaint padding",
                    minimum=0,
                    maximum=256,
                    step=1,
                    value=32,
                )

            with gr.Row():
                w.sam3_use_inpaint_width_height = gr.Checkbox(
                    label="Use separate inpaint width/height",
                    value=False,
                )
                w.sam3_inpaint_width = gr.Slider(
                    label="Inpaint Width",
                    minimum=64,
                    maximum=2048,
                    step=8,
                    value=512,
                )
                w.sam3_inpaint_height = gr.Slider(
                    label="Inpaint Height",
                    minimum=64,
                    maximum=2048,
                    step=8,
                    value=512,
                )

            with gr.Row():
                w.sam3_use_steps = gr.Checkbox(
                    label="Use separate steps",
                    value=False,
                )
                w.sam3_steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=28,
                )
                w.sam3_use_cfg_scale = gr.Checkbox(
                    label="Use separate CFG scale",
                    value=False,
                )
                w.sam3_cfg_scale = gr.Slider(
                    label="CFG Scale",
                    minimum=0.0,
                    maximum=30.0,
                    step=0.1,
                    value=7.0,
                )

    state = gr.State(state_init(w))
    for attr in ("sam3_enable", *ALL_ARGS.attrs):
        widget = getattr(w, attr)
        on_change = partial(on_widget_change, attr=attr)
        widget.change(fn=on_change, inputs=[state, widget], outputs=state, queue=False)

    target_button = buttons.i2i_button if is_img2img else buttons.t2i_button
    if target_button is not None:
        all_inputs = [state, *w.tolist()]
        target_button.click(fn=on_generate_click, inputs=all_inputs, outputs=state, queue=False)

    infotext_fields = [(getattr(w, attr), name) for attr, name in ALL_ARGS]
    return [w.sam3_enable, state], infotext_fields
