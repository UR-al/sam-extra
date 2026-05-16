from __future__ import annotations

import sys
import traceback
from functools import partial
from typing import Any, NamedTuple

import gradio as gr
import numpy as np
from PIL import Image

from modules import script_callbacks, scripts, shared

try:
    from modules.sd_samplers import all_samplers as _all_samplers
except Exception:
    _all_samplers = []

try:
    from modules.sd_schedulers import schedulers as _all_schedulers
except Exception:
    _all_schedulers = []


from sam3ext import SAM3_NAME, Sam3Args, __version__, run_sam3_on_pil
from sam3ext.core import find_checkpoint_options, write_artifacts
from sam3ext.inpaint_core import apply_prompt_sr, copy_prompt, run_inpaint_passes
from sam3ext.ui import WebuiButtons, sam3_ui
from sam3ext.ui_refine import RefinePanel, build_refine_panel, handle_refine_click


txt2img_submit_button = img2img_submit_button = None


class PromptSR(NamedTuple):
    s: str
    r: str


def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_sam3_xyz"):
        p._sam3_xyz = {}
    p._sam3_xyz[field] = x


def search_and_replace_prompt(p, x: Any, xs: Any, replace_in_main_prompt: bool):
    if replace_in_main_prompt:
        p.prompt = p.prompt.replace(xs[0], x)
        p.negative_prompt = p.negative_prompt.replace(xs[0], x)

    if not hasattr(p, "_sam3_xyz_prompt_sr"):
        p._sam3_xyz_prompt_sr = []
    p._sam3_xyz_prompt_sr.append(PromptSR(s=xs[0], r=x))


def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    bool_choices = lambda: ["True", "False"]
    sampler_choices = lambda: [s.name for s in _all_samplers]
    scheduler_choices = lambda: [s.label for s in _all_schedulers]
    mode_choices = lambda: ["Mask only", "Inpaint"]
    mask_mode_choices = lambda: ["Combined", "Individual"]
    device_choices = lambda: ["auto", "cuda", "cpu"]
    format_path = (
        xyz_grid.format_remove_path
        if hasattr(xyz_grid, "format_remove_path")
        else xyz_grid.format_value
    )

    axis = [
        xyz_grid.AxisOption("[SAM3] Enable", str, partial(set_value, field="enabled"), choices=bool_choices),
        xyz_grid.AxisOption(
            "[SAM3] Checkpoint",
            str,
            partial(set_value, field="sam3_checkpoint"),
            format_value=format_path,
            choices=find_checkpoint_options,
        ),
        xyz_grid.AxisOption("[SAM3] Mode", str, partial(set_value, field="sam3_mode"), choices=mode_choices),
        xyz_grid.AxisOption("[SAM3] Mask Mode", str, partial(set_value, field="sam3_mask_mode"), choices=mask_mode_choices),
        xyz_grid.AxisOption("[SAM3] Device", str, partial(set_value, field="sam3_device"), choices=device_choices),
        xyz_grid.AxisOption("[SAM3] Detect Prompt", str, partial(set_value, field="sam3_prompt")),
        xyz_grid.AxisOption("[SAM3] Inpaint Prompt", str, partial(set_value, field="sam3_inpaint_prompt")),
        xyz_grid.AxisOption("[SAM3] Negative Prompt", str, partial(set_value, field="sam3_negative_prompt")),
        xyz_grid.AxisOption(
            "[SAM3] Prompt S/R (SAM3 inpaint)",
            str,
            partial(search_and_replace_prompt, replace_in_main_prompt=False),
        ),
        xyz_grid.AxisOption(
            "[SAM3] Prompt S/R (SAM3 inpaint and main prompt)",
            str,
            partial(search_and_replace_prompt, replace_in_main_prompt=True),
        ),
        xyz_grid.AxisOption("[SAM3] Threshold", float, partial(set_value, field="sam3_threshold")),
        xyz_grid.AxisOption("[SAM3] Mask Dilation", int, partial(set_value, field="sam3_mask_dilation")),
        xyz_grid.AxisOption("[SAM3] Mask Blur", int, partial(set_value, field="sam3_mask_blur")),
        xyz_grid.AxisOption("[SAM3] Denoising Strength", float, partial(set_value, field="sam3_denoising_strength")),
        xyz_grid.AxisOption("[SAM3] CFG Scale", float, partial(set_value, field="sam3_cfg_scale")),
        xyz_grid.AxisOption("[SAM3] Steps", int, partial(set_value, field="sam3_steps")),
        xyz_grid.AxisOption(
            "[SAM3] Inpaint Only Masked",
            str,
            partial(set_value, field="sam3_inpaint_only_masked"),
            choices=bool_choices,
        ),
        xyz_grid.AxisOption("[SAM3] Inpaint Padding", int, partial(set_value, field="sam3_inpaint_only_masked_padding")),
        xyz_grid.AxisOption("[SAM3] Inpaint Width", int, partial(set_value, field="sam3_inpaint_width")),
        xyz_grid.AxisOption("[SAM3] Inpaint Height", int, partial(set_value, field="sam3_inpaint_height")),
        xyz_grid.AxisOption("[SAM3] Sampler", str, partial(set_value, field="sam3_sampler"), choices=sampler_choices),
        xyz_grid.AxisOption("[SAM3] Scheduler", str, partial(set_value, field="sam3_scheduler"), choices=scheduler_choices),
        xyz_grid.AxisOption("[SAM3] Noise Multiplier", float, partial(set_value, field="sam3_noise_multiplier")),
        xyz_grid.AxisOption(
            "[SAM3] Restore Face",
            str,
            partial(set_value, field="sam3_restore_face"),
            choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[SAM3] CN Enable",
            str,
            partial(set_value, field="sam3_cn_enable"),
            choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[SAM3] CN Override External",
            str,
            partial(set_value, field="sam3_cn_override_external"),
            choices=bool_choices,
        ),
        xyz_grid.AxisOption("[SAM3] CN Model", str, partial(set_value, field="sam3_cn_model")),
        xyz_grid.AxisOption("[SAM3] CN Module", str, partial(set_value, field="sam3_cn_module")),
        xyz_grid.AxisOption("[SAM3] CN Weight", float, partial(set_value, field="sam3_cn_weight")),
        xyz_grid.AxisOption("[SAM3] CN Guidance Start", float, partial(set_value, field="sam3_cn_guidance_start")),
        xyz_grid.AxisOption("[SAM3] CN Guidance End", float, partial(set_value, field="sam3_cn_guidance_end")),
    ]

    if not any(x.label.startswith("[SAM3]") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(f"[-] SAM3: xyz_grid error:\n{error}", file=sys.stderr)


script_callbacks.on_before_ui(on_before_ui)


class Sam3MaskScript(scripts.Script):
    alwayson = True

    def title(self):
        return SAM3_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        components, infotext_fields = sam3_ui(
            is_img2img,
            WebuiButtons(
                t2i_button=txt2img_submit_button,
                i2i_button=img2img_submit_button,
            ),
        )
        self.infotext_fields = [(components[0], "SAM3 Enable"), *infotext_fields]
        return components

    def process(self, p, *args_):
        if getattr(p, "_sam3_inner", False):
            p._sam3_args = {"enabled": False}
            return

        xyz_values = getattr(p, "_sam3_xyz", {}) or {}
        enabled = False
        state = {}

        if args_:
            first = args_[0]
            if isinstance(first, bool):
                enabled = first
                if len(args_) > 1 and isinstance(args_[1], dict):
                    state = dict(args_[1] or {})
            elif isinstance(first, dict):
                state = dict(first or {})
                enabled = bool(state.get("sam3_enable", state.get("enabled", False)))

        if not state:
            state = next((dict(arg or {}) for arg in args_ if isinstance(arg, dict)), {})
            enabled = enabled or bool(state.get("sam3_enable", state.get("enabled", False)))

        if "enabled" in xyz_values:
            enabled = str(xyz_values.get("enabled")).lower() == "true"

        def _xyz_or(state_key: str, default: Any, *, legacy: str | None = None) -> Any:
            if state_key in xyz_values:
                return xyz_values[state_key]
            if legacy is not None and legacy in xyz_values:
                return xyz_values[legacy]
            return state.get(state_key, default)

        def _as_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            return str(value).strip().lower() in {"true", "1", "yes", "on"}

        sam3_sampler = str(_xyz_or("sam3_sampler", "Use same sampler"))
        sam3_scheduler = str(_xyz_or("sam3_scheduler", "Use same scheduler"))
        use_sampler = bool(state.get("sam3_use_sampler", False)) or (
            "sam3_sampler" in xyz_values or "sam3_scheduler" in xyz_values
        )

        payload = {
            "sam3_mode": str(_xyz_or("sam3_mode", "Inpaint")),
            "sam3_mask_mode": str(_xyz_or("sam3_mask_mode", "Individual")),
            "sam3_prompt": str(_xyz_or("sam3_prompt", "face", legacy="prompt")).strip() or "face",
            "sam3_inpaint_prompt": str(_xyz_or("sam3_inpaint_prompt", "")),
            "sam3_negative_prompt": str(_xyz_or("sam3_negative_prompt", "")),
            "sam3_threshold": float(_xyz_or("sam3_threshold", 0.4, legacy="threshold")),
            "sam3_mask_dilation": int(_xyz_or("sam3_mask_dilation", 0)),
            "sam3_checkpoint": str(_xyz_or("sam3_checkpoint", "sam3.pt", legacy="checkpoint")),
            "sam3_device": str(_xyz_or("sam3_device", "auto")),
            "sam3_mask_blur": int(_xyz_or("sam3_mask_blur", 4)),
            "sam3_denoising_strength": float(_xyz_or("sam3_denoising_strength", 0.4)),
            "sam3_inpaint_only_masked": _as_bool(
                _xyz_or("sam3_inpaint_only_masked", True), True
            ),
            "sam3_inpaint_only_masked_padding": int(_xyz_or("sam3_inpaint_only_masked_padding", 32)),
            "sam3_use_inpaint_width_height": bool(state.get("sam3_use_inpaint_width_height", False))
            or ("sam3_inpaint_width" in xyz_values or "sam3_inpaint_height" in xyz_values),
            "sam3_inpaint_width": int(_xyz_or("sam3_inpaint_width", 512)),
            "sam3_inpaint_height": int(_xyz_or("sam3_inpaint_height", 512)),
            "sam3_use_steps": bool(state.get("sam3_use_steps", False)) or ("sam3_steps" in xyz_values),
            "sam3_steps": int(_xyz_or("sam3_steps", 28)),
            "sam3_use_cfg_scale": bool(state.get("sam3_use_cfg_scale", False)) or ("sam3_cfg_scale" in xyz_values),
            "sam3_cfg_scale": float(_xyz_or("sam3_cfg_scale", 7.0)),
            "sam3_use_sampler": use_sampler,
            "sam3_sampler": sam3_sampler,
            "sam3_scheduler": sam3_scheduler,
            "sam3_use_noise_multiplier": bool(state.get("sam3_use_noise_multiplier", False))
            or ("sam3_noise_multiplier" in xyz_values),
            "sam3_noise_multiplier": float(_xyz_or("sam3_noise_multiplier", 1.0)),
            "sam3_restore_face": _as_bool(_xyz_or("sam3_restore_face", False), False),
            "sam3_preview_overlay": bool(state.get("sam3_preview_overlay", False)),
            "sam3_save_artifacts": bool(state.get("sam3_save_artifacts", True)),
            "sam3_cn_enable": _as_bool(_xyz_or("sam3_cn_enable", False), False),
            "sam3_cn_override_external": _as_bool(_xyz_or("sam3_cn_override_external", False), False),
            "sam3_cn_model": str(_xyz_or("sam3_cn_model", "None")),
            "sam3_cn_module": str(_xyz_or("sam3_cn_module", "inpaint_only")),
            "sam3_cn_weight": float(_xyz_or("sam3_cn_weight", 1.0)),
            "sam3_cn_guidance_start": float(_xyz_or("sam3_cn_guidance_start", 0.0)),
            "sam3_cn_guidance_end": float(_xyz_or("sam3_cn_guidance_end", 1.0)),
            "sam3_cn_pixel_perfect": _as_bool(_xyz_or("sam3_cn_pixel_perfect", True), True),
            "sam3_cn_control_mode": str(_xyz_or("sam3_cn_control_mode", "Balanced")),
            "sam3_cn_resize_mode": str(_xyz_or("sam3_cn_resize_mode", "Crop and Resize")),
            "sam3_cn_processor_res": int(_xyz_or("sam3_cn_processor_res", 512)),
            "sam3_cn_threshold_a": float(_xyz_or("sam3_cn_threshold_a", -1.0)),
            "sam3_cn_threshold_b": float(_xyz_or("sam3_cn_threshold_b", -1.0)),
        }

        try:
            validated = Sam3Args(**payload)
        except Exception:
            p._sam3_args = {"enabled": False}
            return

        p._sam3_args = {"enabled": bool(enabled), **validated.dict()}
        if not hasattr(p, "extra_generation_params"):
            p.extra_generation_params = {}
        if enabled:
            p.extra_generation_params["SAM3 Enable"] = True
            p.extra_generation_params.update(validated.extra_params())
            p.extra_generation_params["SAM3 Version"] = __version__
            print(
                f"[-] SAM3: mode={validated.sam3_mode}, mask_mode={validated.sam3_mask_mode}, "
                f"prompt={validated.sam3_prompt!r}",
                file=sys.stderr,
            )

    def postprocess_image(self, p, pp, *args_):
        args = getattr(p, "_sam3_args", None) or {}
        if not args.get("enabled"):
            return

        image = pp.image if isinstance(pp.image, Image.Image) else Image.fromarray(np.asarray(pp.image))
        allow_huggingface = not getattr(shared.cmd_opts, "sam3_no_huggingface", False)
        result = run_sam3_on_pil(
            image=image,
            prompt=args["sam3_prompt"],
            threshold=float(args["sam3_threshold"]),
            checkpoint_value=args["sam3_checkpoint"],
            device=args["sam3_device"],
            allow_huggingface=allow_huggingface,
            mask_dilation=int(args.get("sam3_mask_dilation", 0)),
        )

        if args.get("sam3_save_artifacts"):
            seed = None
            if hasattr(p, "all_seeds") and getattr(p, "all_seeds", None):
                seed = p.all_seeds[0]
            write_artifacts(result, seed)

        if not np.any(np.asarray(result.mask)):
            if args.get("sam3_preview_overlay"):
                pp.image = result.overlay
            return

        if args.get("sam3_mode") == "Inpaint":
            masks = [result.mask] if args.get("sam3_mask_mode") == "Combined" else (result.masks or [result.mask])
            prompt = copy_prompt(args.get("sam3_inpaint_prompt"), getattr(p, "prompt", ""))
            negative_prompt = copy_prompt(args.get("sam3_negative_prompt"), getattr(p, "negative_prompt", ""))
            prompt = apply_prompt_sr(p, prompt)
            negative_prompt = apply_prompt_sr(p, negative_prompt)
            print(
                f"[-] SAM3: starting inpaint mode with {len(masks)} mask(s), "
                f"processing={args.get('sam3_mask_mode')}, detect_prompt={args.get('sam3_prompt')!r}",
                file=sys.stderr,
            )

            pp.image = run_inpaint_passes(
                p,
                image,
                masks,
                prompt,
                negative_prompt,
                args,
                cn_args=args,
            )
            return

        if args.get("sam3_preview_overlay"):
            pp.image = result.overlay


txt2img_gallery_component = None
refine_panel: RefinePanel | None = None


def _wire_refine_panel(panel: RefinePanel, gallery):
    """Hook the gallery's select event + the Refine button's click handler.

    Lives here (not in ui_refine.py) because it needs the gallery component
    reference, which we only resolve at on_after_component time.
    """

    def _on_select(evt: gr.SelectData):
        return evt.index if evt is not None else None

    gallery.select(
        fn=_on_select,
        inputs=[],
        outputs=[panel.selected_index_state],
        queue=False,
    )

    panel.refine_button.click(
        fn=handle_refine_click,
        inputs=[gallery, panel.selected_index_state, *panel.all_widgets()],
        outputs=[gallery, panel.status],
    )


def on_after_component(component, **kwargs):
    global txt2img_submit_button, img2img_submit_button
    global txt2img_gallery_component, refine_panel

    elem_id = kwargs.get("elem_id")
    if elem_id == "txt2img_generate":
        txt2img_submit_button = component
    elif elem_id == "img2img_generate":
        img2img_submit_button = component
    elif elem_id == "txt2img_gallery":
        txt2img_gallery_component = component
    elif elem_id == "html_log_txt2img" and refine_panel is None and txt2img_gallery_component is not None:
        # We are still inside the t2i output panel's inner Group context; any
        # Gradio components built here become siblings of the html_log under
        # txt2img_results_panel.
        try:
            samplers = [s.name for s in _all_samplers]
            schedulers = [s.label for s in _all_schedulers]
            checkpoints = find_checkpoint_options()
            refine_panel = build_refine_panel(samplers, schedulers, checkpoints)
            _wire_refine_panel(refine_panel, txt2img_gallery_component)
        except Exception:
            error = traceback.format_exc()
            print(f"[-] SAM3: failed to render Refine panel:\n{error}", file=sys.stderr)


script_callbacks.on_after_component(on_after_component)
