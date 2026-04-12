from __future__ import annotations

from copy import copy
from pathlib import Path
import sys
import traceback
from functools import partial
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from modules import script_callbacks, scripts, shared
from modules.processing import StableDiffusionProcessingImg2Img, process_images

from sam3ext import SAM3_NAME, Sam3Args, __version__, run_sam3_on_pil
from sam3ext.core import find_checkpoint_options, write_artifacts
from sam3ext.ui import WebuiButtons, sam3_ui


txt2img_submit_button = img2img_submit_button = None


def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_sam3_xyz"):
        p._sam3_xyz = {}
    p._sam3_xyz[field] = x


def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption("[SAM3] Enable", str, partial(set_value, field="enabled"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption("[SAM3] Prompt", str, partial(set_value, field="prompt")),
        xyz_grid.AxisOption("[SAM3] Threshold", float, partial(set_value, field="threshold")),
        xyz_grid.AxisOption(
            "[SAM3] Checkpoint",
            str,
            partial(set_value, field="checkpoint"),
            format_value=xyz_grid.format_remove_path if hasattr(xyz_grid, "format_remove_path") else xyz_grid.format_value,
            choices=find_checkpoint_options,
        ),
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

    def process(self, p, enabled, sam3_state):
        if getattr(p, "_sam3_inner", False):
            p._sam3_args = {"enabled": False}
            return

        xyz_values = getattr(p, "_sam3_xyz", {}) or {}
        state = dict(sam3_state or {})
        if "enabled" in xyz_values:
            enabled = str(xyz_values.get("enabled")).lower() == "true"
        payload = {
            "sam3_mode": str(state.get("sam3_mode", "Inpaint")),
            "sam3_mask_mode": str(state.get("sam3_mask_mode", "Individual")),
            "sam3_prompt": str(xyz_values.get("prompt", state.get("sam3_prompt", "face"))).strip() or "face",
            "sam3_inpaint_prompt": str(state.get("sam3_inpaint_prompt", "")),
            "sam3_negative_prompt": str(state.get("sam3_negative_prompt", "")),
            "sam3_threshold": float(xyz_values.get("threshold", state.get("sam3_threshold", 0.4))),
            "sam3_checkpoint": str(xyz_values.get("checkpoint", state.get("sam3_checkpoint", "models/sam3.pt"))),
            "sam3_device": str(state.get("sam3_device", "auto")),
            "sam3_mask_blur": int(state.get("sam3_mask_blur", 4)),
            "sam3_denoising_strength": float(state.get("sam3_denoising_strength", 0.4)),
            "sam3_inpaint_only_masked": bool(state.get("sam3_inpaint_only_masked", True)),
            "sam3_inpaint_only_masked_padding": int(state.get("sam3_inpaint_only_masked_padding", 32)),
            "sam3_use_inpaint_width_height": bool(state.get("sam3_use_inpaint_width_height", False)),
            "sam3_inpaint_width": int(state.get("sam3_inpaint_width", 512)),
            "sam3_inpaint_height": int(state.get("sam3_inpaint_height", 512)),
            "sam3_use_steps": bool(state.get("sam3_use_steps", False)),
            "sam3_steps": int(state.get("sam3_steps", 28)),
            "sam3_use_cfg_scale": bool(state.get("sam3_use_cfg_scale", False)),
            "sam3_cfg_scale": float(state.get("sam3_cfg_scale", 7.0)),
            "sam3_preview_overlay": bool(state.get("sam3_preview_overlay", False)),
            "sam3_save_artifacts": bool(state.get("sam3_save_artifacts", True)),
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

    @staticmethod
    def _copy_prompt(prompt_value, fallback: str) -> str:
        text = str(prompt_value or "").strip()
        return text or str(fallback or "")

    @staticmethod
    def _script_args_copy(script_args):
        script_args = script_args or []
        type_ = type(script_args)
        result = []
        for arg in script_args:
            try:
                copied = copy(arg)
            except TypeError:
                copied = arg
            result.append(copied)
        return type_(result)

    @staticmethod
    def _script_filter(p):
        script_runner = copy(getattr(p, "scripts", None))
        script_args = Sam3MaskScript._script_args_copy(getattr(p, "script_args", []))
        if script_runner is None:
            return None, script_args

        filtered = []
        for script_object in getattr(script_runner, "alwayson_scripts", []):
            filename = Path(getattr(script_object, "filename", "")).stem.lower()
            if filename in {"!sam3", "sam3_mask"}:
                continue
            filtered.append(script_object)

        script_runner.alwayson_scripts = filtered
        return script_runner, script_args

    @staticmethod
    def _build_i2i(p, image: Image.Image, args: dict[str, Any]) -> StableDiffusionProcessingImg2Img:
        width = int(args["sam3_inpaint_width"]) if args.get("sam3_use_inpaint_width_height") else int(getattr(p, "width", image.width))
        height = int(args["sam3_inpaint_height"]) if args.get("sam3_use_inpaint_width_height") else int(getattr(p, "height", image.height))
        steps = int(args["sam3_steps"]) if args.get("sam3_use_steps") else int(getattr(p, "steps", 28))
        cfg_scale = float(args["sam3_cfg_scale"]) if args.get("sam3_use_cfg_scale") else float(getattr(p, "cfg_scale", 7.0))

        version_args = {}
        if hasattr(p, "scheduler"):
            version_args["scheduler"] = getattr(p, "scheduler")

        p2 = StableDiffusionProcessingImg2Img(
            init_images=[image],
            resize_mode=0,
            denoising_strength=float(args["sam3_denoising_strength"]),
            mask=None,
            mask_blur=int(args["sam3_mask_blur"]),
            inpainting_fill=1,
            inpaint_full_res=bool(args["sam3_inpaint_only_masked"]),
            inpaint_full_res_padding=int(args["sam3_inpaint_only_masked_padding"]),
            inpainting_mask_invert=0,
            sd_model=p.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt="",
            negative_prompt="",
            styles=getattr(p, "styles", []),
            seed=getattr(p, "seed", -1),
            subseed=getattr(p, "subseed", -1),
            subseed_strength=getattr(p, "subseed_strength", 0),
            seed_resize_from_h=getattr(p, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(p, "seed_resize_from_w", 0),
            sampler_name=getattr(p, "sampler_name", None),
            batch_size=1,
            n_iter=1,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            restore_faces=getattr(p, "restore_faces", False),
            tiling=getattr(p, "tiling", False),
            extra_generation_params=dict(getattr(p, "extra_generation_params", {})),
            do_not_save_samples=True,
            do_not_save_grid=True,
            **version_args,
        )
        p2.cached_c = [None, None]
        p2.cached_uc = [None, None]
        p2.scripts, p2.script_args = Sam3MaskScript._script_filter(p)
        p2._sam3_inner = True
        p2.enable_hr = getattr(p, "enable_hr", False)
        p2.hr_prompt = getattr(p, "hr_prompt", "")
        p2.hr_negative_prompt = getattr(p, "hr_negative_prompt", "")
        p2.all_hr_prompts = list(getattr(p, "all_hr_prompts", []) or [])
        p2.all_hr_negative_prompts = list(getattr(p, "all_hr_negative_prompts", []) or [])
        return p2

    def postprocess_image(self, p, pp, enabled, sam3_state):
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
            current_image = image.convert("RGB")
            prompt = self._copy_prompt(args.get("sam3_inpaint_prompt"), getattr(p, "prompt", ""))
            negative_prompt = self._copy_prompt(args.get("sam3_negative_prompt"), getattr(p, "negative_prompt", ""))
            print(
                f"[-] SAM3: starting inpaint mode with {len(masks)} mask(s), "
                f"processing={args.get('sam3_mask_mode')}, detect_prompt={args.get('sam3_prompt')!r}",
                file=sys.stderr,
            )

            for index, mask in enumerate(masks, start=1):
                if shared.state.interrupted or shared.state.skipped:
                    break
                p2 = self._build_i2i(p, current_image, args)
                p2.image_mask = mask
                p2.init_images[0] = current_image
                p2.prompt = prompt
                p2.negative_prompt = negative_prompt
                if not p2.all_hr_prompts:
                    p2.all_hr_prompts = [p2.prompt]
                if not p2.all_hr_negative_prompts:
                    p2.all_hr_negative_prompts = [p2.negative_prompt]
                try:
                    processed = process_images(p2)
                except Exception:
                    error = traceback.format_exc()
                    print(f"[-] SAM3: inpaint pass {index} failed:\n{error}", file=sys.stderr)
                    raise
                finally:
                    p2.close()

                if not processed.images:
                    print(f"[-] SAM3: inpaint pass {index} returned no images.", file=sys.stderr)
                    break
                print(f"[-] SAM3: inpaint pass {index} completed.", file=sys.stderr)
                current_image = processed.images[0].convert("RGB")

            pp.image = current_image
            return

        if args.get("sam3_preview_overlay"):
            pp.image = result.overlay


def on_after_component(component, **kwargs):
    global txt2img_submit_button, img2img_submit_button

    elem_id = kwargs.get("elem_id")
    if elem_id == "txt2img_generate":
        txt2img_submit_button = component
    elif elem_id == "img2img_generate":
        img2img_submit_button = component


script_callbacks.on_after_component(on_after_component)
