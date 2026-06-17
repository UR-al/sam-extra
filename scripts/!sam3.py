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
from sam3ext.anima_core import anima_available
from sam3ext.core import find_checkpoint_options, unload_sam3, write_artifacts
from sam3ext.inpaint_core import apply_prompt_sr, copy_prompt, run_inpaint_passes
from sam3ext.ui import WebuiButtons, sam3_ui
from sam3ext.ui_anima import AnimaPanel, build_anima_panel, handle_anima_click
from sam3ext.ui_refine import RefinePanel, _pull_seed_from_gallery_item, build_refine_panel, handle_refine_click


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
        xyz_grid.AxisOption("[SAM3] Exclude Prompt", str, partial(set_value, field="sam3_exclude_prompt")),
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
        xyz_grid.AxisOption(
            "[SAM3] Mask Hull",
            str,
            partial(set_value, field="sam3_mask_hull"),
            choices=bool_choices,
        ),
        xyz_grid.AxisOption("[SAM3] Mask Outline Expand", int, partial(set_value, field="sam3_mask_outline_px")),
        xyz_grid.AxisOption(
            "[SAM3] Unload After",
            str,
            partial(set_value, field="sam3_unload_after"),
            choices=bool_choices,
        ),
        xyz_grid.AxisOption("[SAM3] Mask Blur", int, partial(set_value, field="sam3_mask_blur")),
        xyz_grid.AxisOption("[SAM3] Denoising Strength", float, partial(set_value, field="sam3_denoising_strength")),
        xyz_grid.AxisOption(
            "[SAM3] Inpainting Fill",
            str,
            partial(set_value, field="sam3_inpainting_fill"),
            choices=lambda: ["fill", "original", "latent noise", "latent nothing"],
        ),
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
        xyz_grid.AxisOption("[SAM3] Seed", int, partial(set_value, field="sam3_seed")),
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
        # v0.8.1: render the Anima Tile-Repair accordion as a SIBLING of the
        # SAM3 mask accordion (same t2i scripts column). Previously the panel
        # was rendered in on_after_component into the gallery sidebar group
        # next to Refine — but the user wanted it directly below SAM3, which
        # is the alwayson-scripts area, so it has to be created here inside
        # the script's ui() callback while the same gr.Blocks context is open.
        #
        # Only render on t2i: i2i has no Refine/Anima companion panel either.
        # anima_panel is intentionally NOT added to the returned components
        # list — it's wired through its own button.click in on_after_component
        # so its widgets don't pollute the SAM3 alwayson script_args.
        global anima_panel
        if not is_img2img and anima_panel is None and anima_available():
            try:
                anima_panel = build_anima_panel()
            except Exception:
                error = traceback.format_exc()
                print(
                    f"[-] SAM3: failed to render Anima panel:\n{error}",
                    file=sys.stderr,
                )
        elif not is_img2img and not anima_available():
            print(
                "[-] SAM3: anima_vendor/ not present; Tile-Repair panel "
                "skipped. Re-run install.py to clone kohya-ss/sd-scripts.",
                file=sys.stderr,
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
            "sam3_exclude_prompt": str(_xyz_or("sam3_exclude_prompt", "")),
            "sam3_inpaint_prompt": str(_xyz_or("sam3_inpaint_prompt", "")),
            "sam3_negative_prompt": str(_xyz_or("sam3_negative_prompt", "")),
            "sam3_threshold": float(_xyz_or("sam3_threshold", 0.4, legacy="threshold")),
            "sam3_mask_dilation": int(_xyz_or("sam3_mask_dilation", 0)),
            "sam3_mask_hull": _as_bool(_xyz_or("sam3_mask_hull", False), False),
            "sam3_mask_outline_px": int(_xyz_or("sam3_mask_outline_px", 0)),
            "sam3_checkpoint": str(_xyz_or("sam3_checkpoint", "sam3.pt", legacy="checkpoint")),
            "sam3_device": str(_xyz_or("sam3_device", "auto")),
            "sam3_mask_blur": int(_xyz_or("sam3_mask_blur", 4)),
            "sam3_denoising_strength": float(_xyz_or("sam3_denoising_strength", 0.4)),
            "sam3_inpainting_fill": str(_xyz_or("sam3_inpainting_fill", "latent noise")),
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
            "sam3_use_scheduler": bool(state.get("sam3_use_scheduler", False))
            or ("sam3_scheduler" in xyz_values) or use_sampler,
            "sam3_scheduler": sam3_scheduler,
            "sam3_use_seed": _as_bool(_xyz_or("sam3_use_seed", False), False)
            or ("sam3_seed" in xyz_values),
            "sam3_seed": int(_xyz_or("sam3_seed", -1)),
            "sam3_use_noise_multiplier": bool(state.get("sam3_use_noise_multiplier", False))
            or ("sam3_noise_multiplier" in xyz_values),
            "sam3_noise_multiplier": float(_xyz_or("sam3_noise_multiplier", 1.0)),
            "sam3_restore_face": _as_bool(_xyz_or("sam3_restore_face", False), False),
            "sam3_preview_overlay": bool(state.get("sam3_preview_overlay", False)),
            "sam3_save_artifacts": bool(state.get("sam3_save_artifacts", True)),
            "sam3_unload_after": _as_bool(_xyz_or("sam3_unload_after", False), False),
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
            mask_hull=bool(args.get("sam3_mask_hull", False)),
            mask_outline_px=int(args.get("sam3_mask_outline_px", 0)),
            exclude_prompt=str(args.get("sam3_exclude_prompt") or ""),
        )

        if args.get("sam3_save_artifacts"):
            seed = None
            if hasattr(p, "all_seeds") and getattr(p, "all_seeds", None):
                seed = p.all_seeds[0]
            write_artifacts(result, seed, label=args.get("sam3_prompt"))

        if args.get("sam3_unload_after"):
            unload_sam3()
            print("[-] SAM3: model unloaded from VRAM (re-loads on next detection).", file=sys.stderr)

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
txt2img_prompt_component = None
txt2img_neg_prompt_component = None
txt2img_html_info_component = None
txt2img_generation_info_component = None
refine_panel: RefinePanel | None = None
anima_panel: AnimaPanel | None = None
# Set True the first time on_after_component wires the Anima panel's click
# chain. Distinct from `anima_panel is None` because the panel is created
# in Sam3MaskScript.ui() (alwayson script ui callback) but wiring needs the
# gallery component captured later via on_after_component.
anima_wired: bool = False


# JS shim: replace the placeholder selected_index slot (index 1 in the inputs
# array) with the current gallery selection from the DOM. This sidesteps
# Gradio 5.x's check_all_files_in_cache validation of SelectData event_data
# that we'd otherwise hit by subscribing to gallery.select.
_REFINE_JS = (
    "(...args) => {"
    "  try { args[1] = selected_gallery_index(); } catch (e) { args[1] = -1; }"
    "  return args;"
    "}"
)


def _wire_refine_panel(
    panel: RefinePanel,
    gallery,
    main_prompt,
    main_neg_prompt,
    html_info,
    generation_info,
):
    """Wire the Refine button. Index is injected client-side via ``_REFINE_JS``
    so we don't need a ``gallery.select`` handler (which would otherwise
    trigger Gradio's file-cache validation on the selected image's path).

    ``html_info`` and ``generation_info`` are the standard txt2img output-panel
    components — wiring them as outputs lets us push the per-refine prompt
    into the gallery sidebar so the user actually sees the transformed text,
    instead of the original t2i prompt leaking through.
    """

    # Refine button click chain: hide Refine + show Stop → run the
    # actual refine handler → restore Refine visibility. ``.then()`` chains
    # the steps so the visibility swap happens before sampling starts and
    # restores even if the handler raises (errors bubble through to the
    # last .then). The Stop button below sets shared.state.interrupted
    # which run_sam3_refine + process_images both poll.
    refine_show_stop = panel.refine_button.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[panel.refine_button, panel.stop_button],
        queue=False,
    )
    refine_run = refine_show_stop.then(
        fn=handle_refine_click,
        _js=_REFINE_JS,
        inputs=[
            gallery,
            panel.selected_index_state,
            *panel.all_widgets(),
            main_prompt,
            main_neg_prompt,
            generation_info,
        ],
        outputs=[gallery, panel.status, html_info, generation_info],
    )
    refine_run.then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[panel.refine_button, panel.stop_button],
        queue=False,
    )

    def _stop_refine():
        from modules import shared as _shared

        _shared.state.interrupted = True
        _shared.state.skipped = True

    panel.stop_button.click(
        fn=_stop_refine,
        inputs=[],
        outputs=[],
        queue=False,
    )

    # Seed convenience buttons:
    # - 🎲 → set the Seed Number to -1 (random)
    # - 🎯 → read the currently-selected gallery item's PNG metadata,
    #        extract "Seed: N", and put it in the Seed Number
    panel.seed_random_button.click(
        fn=lambda: -1,
        inputs=[],
        outputs=[panel.seed],
        queue=False,
    )
    panel.seed_pull_button.click(
        fn=_pull_seed_from_gallery_item,
        _js="(...args) => { try { args[1] = selected_gallery_index(); } catch (e) { args[1] = -1; } return args; }",
        inputs=[gallery, panel.selected_index_state, generation_info],
        outputs=[panel.seed],
        queue=False,
    )

    # Manual-mask "Load selected to canvas" button: copy the currently-
    # selected gallery image into the ForgeCanvas background slot so the
    # user can scribble over it. The JS shim populates args[1] with the
    # actual selection index just like the Refine button does.
    if panel.canvas_load_button is not None and panel.canvas_bg is not None:
        def _load_to_canvas(gallery_value, selected_index):
            from sam3ext.ui_refine import _coerce_gallery_item_to_pil

            items = list(gallery_value or [])
            if not items:
                return None
            try:
                idx = int(selected_index) if selected_index is not None else -1
            except (TypeError, ValueError):
                idx = -1
            if idx < 0 or idx >= len(items):
                idx = len(items) - 1
            return _coerce_gallery_item_to_pil(items[idx])

        panel.canvas_load_button.click(
            fn=_load_to_canvas,
            _js="(...args) => { try { args[1] = selected_gallery_index(); } catch (e) { args[1] = -1; } return args; }",
            inputs=[gallery, panel.selected_index_state],
            outputs=[panel.canvas_bg],
            queue=False,
        )

    # Auto seed pull on gallery change: when t2i Generate finishes (or our
    # Refine appends a new image), the gallery's value updates and Gradio
    # fires .change. Pull the seed from the LAST item — that's the freshly
    # generated/refined one and the most useful default for the next
    # Refine click. The user can still manually click 🎯 after selecting
    # a different older image.
    def _auto_pull_seed_from_latest(gallery_value, generation_info_json):
        items = list(gallery_value or [])
        if not items:
            return -1
        return _pull_seed_from_gallery_item(
            gallery_value, len(items) - 1, generation_info_json
        )

    gallery.change(
        fn=_auto_pull_seed_from_latest,
        inputs=[gallery, generation_info],
        outputs=[panel.seed],
        queue=False,
        show_progress=False,
    )


# JS shim identical to Refine's — replaces selected_index in args slot 1
# with the live frontend selection.
_ANIMA_JS = (
    "(...args) => {"
    "  try { args[1] = selected_gallery_index(); } catch (e) { args[1] = -1; }"
    "  return args;"
    "}"
)


def _wire_anima_panel(
    panel: AnimaPanel,
    gallery,
    html_info,
    generation_info,
):
    """Wire the Anima Tile-Repair button — mirror of ``_wire_refine_panel``
    minus the prompt-inheritance plumbing (Anima has its own prompt textbox)
    and minus the auto-seed-pull on gallery.change (Refine's already handles
    that, no need to double-fire).
    """
    # Refine→Stop visibility swap, same chain shape as the Refine panel.
    show_stop = panel.repair_button.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[panel.repair_button, panel.stop_button],
        queue=False,
    )
    run = show_stop.then(
        fn=handle_anima_click,
        _js=_ANIMA_JS,
        inputs=[
            gallery,
            panel.selected_index_state,
            *panel.all_widgets(),
            generation_info,
        ],
        outputs=[gallery, panel.status, html_info, generation_info],
    )
    run.then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[panel.repair_button, panel.stop_button],
        queue=False,
    )

    def _stop_anima():
        from modules import shared as _shared

        _shared.state.interrupted = True
        _shared.state.skipped = True

    panel.stop_button.click(
        fn=_stop_anima,
        inputs=[],
        outputs=[],
        queue=False,
    )

    # Seed convenience buttons — same handlers Refine uses (reuse imports).
    panel.seed_random_button.click(
        fn=lambda: -1,
        inputs=[],
        outputs=[panel.seed],
        queue=False,
    )
    panel.seed_pull_button.click(
        fn=_pull_seed_from_gallery_item,
        _js=_ANIMA_JS,
        inputs=[gallery, panel.selected_index_state, generation_info],
        outputs=[panel.seed],
        queue=False,
    )


def on_after_component(component, **kwargs):
    global txt2img_submit_button, img2img_submit_button
    global txt2img_gallery_component, txt2img_prompt_component, txt2img_neg_prompt_component
    global txt2img_html_info_component, txt2img_generation_info_component
    global refine_panel

    elem_id = kwargs.get("elem_id")
    if elem_id == "txt2img_generate":
        txt2img_submit_button = component
    elif elem_id == "img2img_generate":
        img2img_submit_button = component
    elif elem_id == "txt2img_prompt":
        txt2img_prompt_component = component
    elif elem_id == "txt2img_neg_prompt":
        txt2img_neg_prompt_component = component
    elif elem_id == "txt2img_gallery":
        txt2img_gallery_component = component
    elif elem_id == "html_info_txt2img":
        txt2img_html_info_component = component
    elif elem_id == "generation_info_txt2img":
        # Render the Refine panel right after generation_info Textbox so all
        # the components we need to wire as outputs (gallery, html_info,
        # generation_info) are already captured. The panel lands inside the
        # same hidden gr.Group as the infotext bits, but its accordion is
        # visible itself.
        txt2img_generation_info_component = component
        if refine_panel is not None or txt2img_gallery_component is None:
            return
        if txt2img_html_info_component is None:
            print(
                "[-] SAM3: html_info_txt2img not captured yet — skipping Refine panel render.",
                file=sys.stderr,
            )
            return
        try:
            samplers = [s.name for s in _all_samplers]
            schedulers = [s.label for s in _all_schedulers]
            checkpoints = find_checkpoint_options()
            refine_panel = build_refine_panel(samplers, schedulers, checkpoints)
            _wire_refine_panel(
                refine_panel,
                txt2img_gallery_component,
                txt2img_prompt_component,
                txt2img_neg_prompt_component,
                txt2img_html_info_component,
                txt2img_generation_info_component,  # local reference; not None at this point
            )
        except Exception:
            error = traceback.format_exc()
            print(f"[-] SAM3: failed to render Refine panel:\n{error}", file=sys.stderr)

        # v0.8.1: the Anima panel is now created inside Sam3MaskScript.ui()
        # (alongside the SAM3 mask accordion). Here we only wire its click
        # chain — we need the t2i gallery / generation_info components that
        # this callback captures.
        global anima_wired
        if anima_panel is not None and not anima_wired:
            try:
                _wire_anima_panel(
                    anima_panel,
                    txt2img_gallery_component,
                    txt2img_html_info_component,
                    txt2img_generation_info_component,
                )
                anima_wired = True
            except Exception:
                error = traceback.format_exc()
                print(
                    f"[-] SAM3: failed to wire Anima panel:\n{error}",
                    file=sys.stderr,
                )


script_callbacks.on_after_component(on_after_component)
