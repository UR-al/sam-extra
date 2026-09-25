from __future__ import annotations

import sys
import traceback
from functools import partial
from typing import Any, NamedTuple

import gradio as gr
import numpy as np
from gradio.context import Context
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


import sam3ext.core as _sam3_core

if not all(hasattr(_sam3_core, name) for name in ("release_sam3", "describe_unload", "drop_offloaded_sam3")):
    # Reload UI 는 이 스크립트만 다시 읽고 sam3ext.core 는 옛 모듈을 그대로 쓴다 — 옛 모듈의 캐시 번들을
    # 비운 뒤 새로 불러온다(run_sam3_on_pil 도 새 캐시를 쓰도록 아래 import 보다 먼저).
    import importlib

    try:
        (getattr(_sam3_core, "release_sam3", None) or _sam3_core.unload_sam3)()
    except Exception:
        pass
    _sam3_core = importlib.reload(_sam3_core)

from sam3ext import SAM3_NAME, Sam3Args, __version__, run_sam3_on_pil
from sam3ext.anima_core import anima_available
from sam3ext.core import (
    OPT_UNLOAD_KEEP_IN_RAM,
    describe_unload,
    drop_offloaded_sam3,
    find_checkpoint_options,
    release_sam3,
    unload_sam3,
    write_artifacts,
)
from sam3ext.inpaint_core import apply_prompt_sr, copy_prompt, is_oom, run_inpaint_passes
from sam3ext.notebook_store import register_notebook_routes
from sam3ext import layout_lanes, quick_button as sam3_quick, ui_dock
from sam3ext import ui_tipo
from sam3ext.ui import WebuiButtons, sam3_ui
from sam3ext.ui_anima import AnimaPanel, build_anima_panel, handle_anima_click, stop_anima
from sam3ext.anima_ipa.options import (
    DEFAULT_DUPLICATE_POLICY as IPA_DEFAULT_DUPLICATE_POLICY,
    OPT_DUPLICATE_POLICY as OPT_IPA_DUPLICATE_POLICY,
)
from sam3ext.ui_anima_reference import (
    AnimaReferencePanel,
    build_anima_reference_panel,
    wire_anima_reference_panel,
)
from sam3ext.ui_refine import (
    RefinePanel,
    _pull_seed_from_gallery_item,
    build_refine_panel,
    handle_refine_click,
    stop_refine,
)
try:
    from sam3ext.sampler_load_guard import (
        guard_sampler_app_started_callbacks,
        prune_stale_sampler_load_targets,
    )
except ImportError:
    # Forge can reload extension scripts without restarting Python. Reload this
    # helper too when an older cached module predates a newly-added guard.
    import importlib
    import sam3ext.sampler_load_guard as _sampler_load_guard

    _sampler_load_guard = importlib.reload(_sampler_load_guard)
    guard_sampler_app_started_callbacks = _sampler_load_guard.guard_sampler_app_started_callbacks
    prune_stale_sampler_load_targets = _sampler_load_guard.prune_stale_sampler_load_targets


# metadata.ini 의 콜백 순서 섹션([callbacks/forge_sam3_extension/...])은 폴더명에 묶여 있다 — Forge 는 확장
# 폴더명(소문자)을 canonical name 으로 쓰고, 섹션이 그 이름으로 시작하지 않으면 순서 지정을 버린다(감사 M28).
SAM3_EXTENSION_FOLDER = "forge_sam3_extension"


def _extension_folder_warning(script_path) -> str | None:
    """확장 폴더명이 ``SAM3_EXTENSION_FOLDER`` 와 다르면 시작 로그에 남길 한 줄, 같으면 None."""
    from pathlib import Path

    folder = Path(script_path).resolve().parent.parent.name
    if folder.lower().strip() == SAM3_EXTENSION_FOLDER:
        return None
    return (
        f"[-] SAM3: extension folder is '{folder}', not '{SAM3_EXTENSION_FOLDER}' — Forge ignores the callback "
        "order in metadata.ini (the RK/TDE sampler load guard then runs late). Rename the folder to "
        f"extensions/{SAM3_EXTENSION_FOLDER} and restart."
    )


_folder_warning = _extension_folder_warning(__file__)
if _folder_warning:
    print(_folder_warning, file=sys.stderr)


txt2img_submit_button = img2img_submit_button = None

# process() 가 UI 상태·XYZ 에 없는 키를 채울 때 쓰는 기본값 — Sam3Args(= UI 기본값)와 한 곳에서 맞춘다.
# 예전엔 여기 따로 적힌 'latent noise' / unload_after=False 가 API 호출자에게 UI 와 다른 결과를 줬다.
_SAM3_DEFAULTS: dict[str, Any] = Sam3Args().dict()

# XYZ '[SAM3] Checkpoint'·'[SAM3] Device' 축 cost — make_axis_on_xyz_grid 참고.
SAM3_CHECKPOINT_AXIS_COST = 0.9


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
            # 값이 바뀌면 SAM3 번들을 버리고 다시 빌드한다(3~5 초). xyz_grid 는 cost 가 큰 축을 바깥 루프에
            # 두므로 Checkpoint(1.0)보다 작고 [DoRA] 축(0.8)·VAE(0.7)보다 크게 — 각 칸의 결과는 같고 순서만 바뀐다.
            cost=SAM3_CHECKPOINT_AXIS_COST,
            choices=find_checkpoint_options,
        ),
        xyz_grid.AxisOption("[SAM3] Mode", str, partial(set_value, field="sam3_mode"), choices=mode_choices),
        xyz_grid.AxisOption("[SAM3] Mask Mode", str, partial(set_value, field="sam3_mask_mode"), choices=mask_mode_choices),
        xyz_grid.AxisOption(
            "[SAM3] Device",
            str,
            partial(set_value, field="sam3_device"),
            # 장치도 번들 캐시 키라 값이 바뀌면 버리고 다시 빌드한다 — Checkpoint 축과 같은 cost 로 바깥 루프에.
            cost=SAM3_CHECKPOINT_AXIS_COST,
            choices=device_choices,
        ),
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
    guarded = guard_sampler_app_started_callbacks(script_callbacks.callback_map)
    if guarded:
        print(f"[SAM3] guarded sampler app-start callbacks: {guarded}")
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(f"[-] SAM3: xyz_grid error:\n{error}", file=sys.stderr)


script_callbacks.on_before_ui(on_before_ui)

# 'Unload after' 는 SAM3 번들을 CPU 캐시에 남긴다(sam3ext.core.unload_sam3) — Reload UI·확장 언로드 때는
# 그 캐시(RAM 약 3.4 GB)와 GPU 에 남은 것까지 진짜로 버린다.
script_callbacks.on_script_unloaded(release_sam3, name="sam3-release-bundle")


def _on_keep_in_ram_changed():
    # 끄는 즉시 RAM 에 보관 중인 번들도 해제한다(다음 'Unload after' 까지 3.4 GB 가 남지 않게). 켤 때는 할 일 없음.
    if not getattr(shared.opts, OPT_UNLOAD_KEEP_IN_RAM, True):
        drop_offloaded_sam3()


def on_ui_settings():
    section = ("sam3_mask", "SAM Extra SAM3")
    shared.opts.add_option(
        OPT_UNLOAD_KEEP_IN_RAM,
        shared.OptionInfo(
            True,
            "'Unload after' 뒤 SAM3 모델을 CPU RAM 에 보관 (약 3.4 GB)",
            gr.Checkbox,
            onchange=_on_keep_in_ram_changed,
            section=section,
        ).info(
            "켜면 VRAM 에서 내린 모델을 RAM 에 두었다가 다음 검출 때 옮기기만 합니다(이미지당 2~4 초 절약). "
            "끄면 예전처럼 완전히 해제해 RAM 을 비우고, 다음 검출마다 체크포인트를 다시 읽습니다. "
            "끄는 즉시 보관 중인 모델도 해제합니다. 결과 이미지는 같습니다."
        ),
    )
    shared.opts.add_option(
        OPT_IPA_DUPLICATE_POLICY,
        shared.OptionInfo(
            IPA_DEFAULT_DUPLICATE_POLICY,
            "캐릭터 레퍼런스 IP-Adapter: 28블록 어댑터를 2.9B·3.8B 에 얹을 때 끼워 넣은 블록 처리 (결과가 바뀜)",
            gr.Radio,
            {
                "choices": [
                    ("원래 계보 블록에만 주입 (lineage · 권장)", "lineage"),
                    ("끼워 넣은 블록에도 전부 주입 (all · 0.21 까지의 동작)", "all"),
                    ("끼워 넣은 블록과 강도를 나눔 (split)", "split"),
                ]
            },
            section=("sam3_reference", "SAM Extra Character Reference"),
        ).info(
            "IP-Adapter 는 28블록 베이스로 학습돼, 2.9B(40블록)·3.8B(52블록)에서는 한 어댑터 블록이 블록 계보에 따라 "
            "여러 모델 블록에 대응합니다. 모두에 주입하면(all) 같은 참조가 두세 번 더해져 2.9B 는 강도 1.0, 3.8B 는 "
            "0.5 에서도 격자 무늬로 깨집니다. lineage(기본)는 각 어댑터 블록의 원래 자리에만 주입해 두 모델 모두 "
            "깨지지 않았고(2026-09-23 GPU 확인, 참조 1장·시드 1개), split 은 복제들이 강도를 나눠 덜 깨지지만 격자 "
            "무늬가 남았습니다. 28블록 베이스 모델에서는 어느 값이든 결과가 같습니다. 결과 infotext 의 "
            "'SAM3 IPA Duplicates' 에 남습니다."
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)


def on_app_started_services(demo, app):
    if register_notebook_routes(app):
        print("[SAM3 Notebook] storage API: /sam3-notebook")
    guarded = guard_sampler_app_started_callbacks(script_callbacks.callback_map)
    if guarded:
        print(f"[SAM3] guarded late sampler app-start callbacks: {guarded}")
    if sam3_quick_button is not None and not sam3_quick_wired:
        print(
            "[-] SAM3: quick button could not find the hires-fix button wiring; it stays inactive.",
            file=sys.stderr,
        )
    removed = prune_stale_sampler_load_targets(demo)
    total = sum(removed.values())
    if total:
        print(
            "[SAM3] removed stale sampler page-load targets: "
            f"RK={removed['rk']}, TDE={removed['tde']}"
        )


script_callbacks.on_app_started(
    on_app_started_services,
    name="notebook-and-sampler-load-guard",
)


class Sam3MaskScript(scripts.Script):
    alwayson = True
    # Lead the SAM3 extension block. Without an explicit priority this accordion
    # fell in with the unprioritised third-party scripts, letting them sit
    # between SAM3 and the anima feature accordions (which carry explicit
    # priorities and so sank below the unprioritised ones). A contiguous
    # negative block keeps the whole SAM3 family together at the top, in order:
    #   SAM3 (-30) → Detail Daemon (-29) → Skimmed CFG (-28) → Safe PAG (-27)
    #   → VAE 2x (-26) → Reference PoC / log toggles (-25).
    # (lower sorting_priority = higher up.)
    sorting_priority = -30

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
        # v0.22: Tile-Repair 와 캐릭터 레퍼런스는 갤러리 밑 "선택 이미지" 탭에서 만든다(sam3ext/ui_dock.py).
        # 여기서 만들지 않으므로 API 가 ui() 를 throwaway Blocks 에서 다시 불러도 죽은 컴포넌트가 생기지 않는다.

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

        # Fast path: SAM3 off and no XYZ override → skip building the ~50-field
        # payload + Sam3Args validation on every (incl. plain non-Anima) gen.
        # postprocess_image only reads args["enabled"] when disabled.
        if not enabled and not xyz_values:
            p._sam3_args = {"enabled": False}
            return

        def _xyz_or(state_key: str, *, legacy: str | None = None) -> Any:
            # XYZ 축 → UI 상태 → Sam3Args 기본값 순.
            if state_key in xyz_values:
                return xyz_values[state_key]
            if legacy is not None and legacy in xyz_values:
                return xyz_values[legacy]
            return state.get(state_key, _SAM3_DEFAULTS[state_key])

        def _as_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            return str(value).strip().lower() in {"true", "1", "yes", "on"}

        def _bool_or(state_key: str) -> bool:
            return _as_bool(_xyz_or(state_key), _SAM3_DEFAULTS[state_key])

        sam3_sampler = str(_xyz_or("sam3_sampler"))
        sam3_scheduler = str(_xyz_or("sam3_scheduler"))
        use_sampler = bool(state.get("sam3_use_sampler", False)) or (
            "sam3_sampler" in xyz_values or "sam3_scheduler" in xyz_values
        )

        # 수치·Literal 은 문자열 그대로 넘긴다 — 범위 클램프와 표기 정규화는 Sam3Args 의 validator 가 한다.
        payload = {
            "sam3_mode": _xyz_or("sam3_mode"),
            "sam3_mask_mode": _xyz_or("sam3_mask_mode"),
            "sam3_prompt": str(_xyz_or("sam3_prompt", legacy="prompt")).strip() or _SAM3_DEFAULTS["sam3_prompt"],
            "sam3_exclude_prompt": str(_xyz_or("sam3_exclude_prompt")),
            "sam3_inpaint_prompt": str(_xyz_or("sam3_inpaint_prompt")),
            "sam3_negative_prompt": str(_xyz_or("sam3_negative_prompt")),
            "sam3_threshold": _xyz_or("sam3_threshold", legacy="threshold"),
            "sam3_mask_dilation": _xyz_or("sam3_mask_dilation"),
            "sam3_mask_hull": _bool_or("sam3_mask_hull"),
            "sam3_mask_outline_px": _xyz_or("sam3_mask_outline_px"),
            "sam3_checkpoint": str(_xyz_or("sam3_checkpoint", legacy="checkpoint")),
            "sam3_device": str(_xyz_or("sam3_device")),
            "sam3_mask_blur": _xyz_or("sam3_mask_blur"),
            "sam3_denoising_strength": _xyz_or("sam3_denoising_strength"),
            "sam3_inpainting_fill": _xyz_or("sam3_inpainting_fill"),
            "sam3_inpaint_only_masked": _bool_or("sam3_inpaint_only_masked"),
            "sam3_inpaint_only_masked_padding": _xyz_or("sam3_inpaint_only_masked_padding"),
            "sam3_use_inpaint_width_height": bool(state.get("sam3_use_inpaint_width_height", False))
            or ("sam3_inpaint_width" in xyz_values or "sam3_inpaint_height" in xyz_values),
            "sam3_inpaint_width": _xyz_or("sam3_inpaint_width"),
            "sam3_inpaint_height": _xyz_or("sam3_inpaint_height"),
            "sam3_use_steps": bool(state.get("sam3_use_steps", False)) or ("sam3_steps" in xyz_values),
            "sam3_steps": _xyz_or("sam3_steps"),
            "sam3_use_cfg_scale": bool(state.get("sam3_use_cfg_scale", False)) or ("sam3_cfg_scale" in xyz_values),
            "sam3_cfg_scale": _xyz_or("sam3_cfg_scale"),
            "sam3_use_sampler": use_sampler,
            "sam3_sampler": sam3_sampler,
            "sam3_use_scheduler": bool(state.get("sam3_use_scheduler", False))
            or ("sam3_scheduler" in xyz_values) or use_sampler,
            "sam3_scheduler": sam3_scheduler,
            "sam3_use_seed": _bool_or("sam3_use_seed") or ("sam3_seed" in xyz_values),
            "sam3_seed": _xyz_or("sam3_seed"),
            "sam3_use_noise_multiplier": bool(state.get("sam3_use_noise_multiplier", False))
            or ("sam3_noise_multiplier" in xyz_values),
            "sam3_noise_multiplier": _xyz_or("sam3_noise_multiplier"),
            "sam3_restore_face": _bool_or("sam3_restore_face"),
            "sam3_preview_overlay": bool(state.get("sam3_preview_overlay", _SAM3_DEFAULTS["sam3_preview_overlay"])),
            "sam3_save_artifacts": bool(state.get("sam3_save_artifacts", _SAM3_DEFAULTS["sam3_save_artifacts"])),
            "sam3_unload_after": _bool_or("sam3_unload_after"),
            "sam3_cn_enable": _bool_or("sam3_cn_enable"),
            "sam3_cn_override_external": _bool_or("sam3_cn_override_external"),
            "sam3_cn_model": str(_xyz_or("sam3_cn_model")),
            "sam3_cn_module": str(_xyz_or("sam3_cn_module")),
            "sam3_cn_weight": _xyz_or("sam3_cn_weight"),
            "sam3_cn_guidance_start": _xyz_or("sam3_cn_guidance_start"),
            "sam3_cn_guidance_end": _xyz_or("sam3_cn_guidance_end"),
            "sam3_cn_pixel_perfect": _bool_or("sam3_cn_pixel_perfect"),
            "sam3_cn_control_mode": _xyz_or("sam3_cn_control_mode"),
            "sam3_cn_resize_mode": _xyz_or("sam3_cn_resize_mode"),
            "sam3_cn_processor_res": _xyz_or("sam3_cn_processor_res"),
            "sam3_cn_threshold_a": _xyz_or("sam3_cn_threshold_a"),
            "sam3_cn_threshold_b": _xyz_or("sam3_cn_threshold_b"),
        }

        if not hasattr(p, "extra_generation_params"):
            p.extra_generation_params = {}

        try:
            validated = Sam3Args(**payload)
        except Exception as exc:
            # 검증 실패 → 이 생성은 SAM3 없이 간다. 예전엔 로그 한 줄 없이 꺼져서 "켰는데 아무 일도 없다" 가 됐다.
            p._sam3_args = {"enabled": False}
            if enabled:
                reason = " ".join(str(exc).split())   # pydantic 의 여러 줄 메시지를 infotext 한 줄로
                print(
                    f"[-] SAM3: invalid settings, SAM3 disabled for this generation: {reason}",
                    file=sys.stderr,
                )
                p.extra_generation_params.pop("SAM3 Enable", None)
                p.extra_generation_params["SAM3 Error"] = reason
            return

        p._sam3_args = {"enabled": bool(enabled), **validated.dict()}
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

        try:
            self._run_sam3_on_image(p, pp, args)
        except Exception as exc:
            # Forge 의 ScriptRunner.postprocess_image 는 예외를 콘솔에만 보고하고 이미지를 그대로 저장한다 —
            # 그 이미지가 'SAM3 Enable: True' 를 달고 나가지 않게 infotext 를 고치고, 검출 번들이 VRAM 에
            # 남지 않게 내린 뒤 예외는 그대로 올린다(🎯 빠른 버튼도 같은 경로).
            p._sam3_mask_found = False
            params = getattr(p, "extra_generation_params", None)
            if isinstance(params, dict):
                if "SAM3 Enable" in params:
                    # 배치의 다음 장이 성공하면 같은 자리에 되돌린다(_restore_infotext_after_failure).
                    p._sam3_enable_index = list(params).index("SAM3 Enable")
                params.pop("SAM3 Enable", None)
                params["SAM3 Error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())}"
            print(f"[-] SAM3: failed, image saved without SAM3: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.get("sam3_unload_after") or is_oom(exc):
                try:
                    kept = unload_sam3()
                    print(f"[-] SAM3: {describe_unload(bool(kept), after_failure=True)}", file=sys.stderr)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
            raise

    @staticmethod
    def _restore_infotext_after_failure(p) -> None:
        """배치(n_iter>1·batch_size>1)의 앞 장이 실패해 남긴 'SAM3 Error' 를 걷고 'SAM3 Enable' 을 되돌린다.

        extra_generation_params 는 배치 전체가 함께 쓴다 — 되돌리지 않으면 그 뒤 성공한 장도 오류 infotext 로
        저장된다. SAM3 가 켜진 채 여기 왔다면 'SAM3 Error' 는 postprocess_image 의 실패 기록뿐이다(process()
        의 검증 실패는 SAM3 를 끄므로 여기 오지 않는다). 실패가 없던 배치와 같은 순서가 되게 원래 자리에 넣는다.
        """
        params = getattr(p, "extra_generation_params", None)
        if not isinstance(params, dict) or "SAM3 Error" not in params:
            return
        params.pop("SAM3 Error", None)
        if "SAM3 Enable" in params:
            return
        items = list(params.items())
        index = getattr(p, "_sam3_enable_index", len(items))
        items.insert(min(max(int(index), 0), len(items)), ("SAM3 Enable", True))
        params.clear()
        params.update(items)

    def _run_sam3_on_image(self, p, pp, args: dict[str, Any]) -> None:
        """postprocess_image 의 본문 — 검출 → (아티팩트 저장) → (unload) → 오버레이/인페인트."""
        self._restore_infotext_after_failure(p)
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

        # 🎯 빠른 버튼이 결과를 판단한다(마스크 없음 / Mask only / 인페인트) — 이미지 객체 비교로는 알 수 없다
        p._sam3_mask_found = bool(np.any(np.asarray(result.mask)))

        if args.get("sam3_save_artifacts"):
            seed = None
            if hasattr(p, "all_seeds") and getattr(p, "all_seeds", None):
                seed = p.all_seeds[0]
            write_artifacts(result, seed, label=args.get("sam3_prompt"))

        if args.get("sam3_unload_after"):
            kept = unload_sam3()
            print(f"[-] SAM3: {describe_unload(bool(kept))}", file=sys.stderr)

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
txt2img_width_component = None
txt2img_height_component = None
# SAM3 빠른 버튼(🎯): ✨(txt2img_upscale) 옆에 만들고, ✨ 의 click 이 등록된 뒤 같은 입력으로 연결한다.
txt2img_upscale_button = None
sam3_quick_button = None
sam3_quick_wired: bool = False
# TIPO 프롬프트 확장: 🪄 는 스타일 적용 버튼(txt2img_style_apply) 옆, 설정 칸은 스타일 줄 아래. 프롬프트·가로·세로가 다
# 잡힌 뒤 한 번 연결한다.
tipo_button = None
tipo_panel = None
tipo_wired: bool = False
refine_panel: RefinePanel | None = None
anima_panel: AnimaPanel | None = None
anima_reference_panel: AnimaReferencePanel | None = None
# v0.22: 세 패널은 갤러리 밑 "선택 이미지" 탭에서 함께 만든다(sam3ext/ui_dock.py). 예전에 쓰던 일회성 플래그
# 네 개(build_attempted / wired) 대신, 화면 하나당 한 번만 만들고 패널마다 한 번만 연결하도록 이 상태가 기억한다.
dock_state = ui_dock.DockState()


# JS shim: replace the placeholder selected_index slot (index 1 in the inputs
# array) with the current gallery selection from the DOM. This sidesteps
# Gradio 5.x's check_all_files_in_cache validation of SelectData event_data
# that we'd otherwise hit by subscribing to gallery.select. Shared by every
# Refine/Anima handler and seed/canvas button that needs the live selection.
_SELECTED_INDEX_JS = (
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
    """Wire the Refine button. Index is injected client-side via ``_SELECTED_INDEX_JS``
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
    # which run_sam3_refine + process_images both poll — but only while the
    # Refine job holds shared.state (ui_refine.stop_refine), so a txt2img
    # that holds the queue lock is never stopped in its place.
    refine_show_stop = panel.refine_button.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[panel.refine_button, panel.stop_button],
        queue=False,
    )
    refine_run = refine_show_stop.then(
        fn=handle_refine_click,
        _js=_SELECTED_INDEX_JS,
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

    panel.stop_button.click(
        fn=stop_refine,
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
        _js=_SELECTED_INDEX_JS,
        inputs=[gallery, panel.selected_index_state, generation_info],
        outputs=[panel.seed],
        queue=False,
    )

    # 🎭 Regional Swap preset — set the guide's (RegionalSampler) values on the
    # existing Refine widgets in one click. Only touches widgets we're confident
    # exist by choice/label; scheduler is left for the user (sgm_uniform label
    # varies) as noted in the panel. See docs/REGIONAL_STYLE_SWAP.md.
    def _apply_regional_preset():
        return (
            gr.update(value="Euler"),      # sampler (deterministic)
            gr.update(value=5.0),          # cfg_scale
            gr.update(value=33),           # steps
            gr.update(value=0.76),         # denoising_strength (= base_only_steps 8)
            gr.update(value=16),           # mask_blur (= overlap_factor 16)
            gr.update(value=False),        # inherit_main_prompt
            gr.update(value=False),        # inherit_main_neg_prompt
            gr.update(value=True),         # inpaint_only_masked
            gr.update(value=32),           # inpaint_only_masked_padding
            gr.update(value="original"),   # inpainting_fill
        )

    panel.regional_preset_button.click(
        fn=_apply_regional_preset,
        inputs=[],
        outputs=[
            panel.sampler,
            panel.cfg_scale,
            panel.steps,
            panel.denoising_strength,
            panel.mask_blur,
            panel.inherit_main_prompt,
            panel.inherit_main_neg_prompt,
            panel.inpaint_only_masked,
            panel.inpaint_only_masked_padding,
            panel.inpainting_fill,
        ],
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
            _js=_SELECTED_INDEX_JS,
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
        _js=_SELECTED_INDEX_JS,
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

    panel.stop_button.click(
        fn=stop_anima,
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
        _js=_SELECTED_INDEX_JS,
        inputs=[gallery, panel.selected_index_state, generation_info],
        outputs=[panel.seed],
        queue=False,
    )


def _create_sam3_quick_button():
    try:
        from modules.ui_components import ToolButton

        return ToolButton(
            sam3_quick.BUTTON_ICON,
            elem_id=sam3_quick.BUTTON_ELEM_ID,
            tooltip=sam3_quick.BUTTON_TOOLTIP,
        )
    except Exception:
        print(f"[-] SAM3: failed to create the quick button:\n{traceback.format_exc()}", file=sys.stderr)
        return None


def _create_tipo_button():
    try:
        return ui_tipo.create_tipo_button()
    except Exception:
        print(f"[-] SAM3: failed to create the TIPO button:\n{traceback.format_exc()}", file=sys.stderr)
        return None


def _build_tipo_panel():
    try:
        from sam3ext.tipo.runtime import shared_runtime

        return ui_tipo.build_tipo_panel(model_missing=bool(shared_runtime().missing_files()))
    except Exception:
        print(f"[-] SAM3: failed to build the TIPO panel:\n{traceback.format_exc()}", file=sys.stderr)
        return None


def _wire_tipo() -> None:
    global tipo_wired
    if tipo_wired or tipo_button is None or tipo_panel is None:
        return
    if txt2img_prompt_component is None or txt2img_width_component is None or txt2img_height_component is None:
        return
    tipo_wired = True
    try:
        ui_tipo.wire_tipo(
            tipo_button, tipo_panel, txt2img_prompt_component, txt2img_width_component, txt2img_height_component,
        )
    except Exception:
        print(f"[-] SAM3: failed to wire the TIPO button:\n{traceback.format_exc()}", file=sys.stderr)


def _wire_sam3_quick_button() -> None:
    """✨ 의 click 이 txt2img Blocks 에 등록되면(그 뒤 만들어지는 엑스트라 네트워크 UI 컴포넌트 때) 한 번 연결한다."""
    global sam3_quick_wired
    if sam3_quick_wired or sam3_quick_button is None or txt2img_upscale_button is None:
        return
    dependency = sam3_quick.find_click_dependency(Context.root_block, txt2img_upscale_button)
    if dependency is None:
        return
    sam3_quick_wired = True
    try:
        sam3_quick.wire_quick_button(sam3_quick_button, dependency)
    except Exception:
        print(f"[-] SAM3: failed to wire the quick button:\n{traceback.format_exc()}", file=sys.stderr)


def on_after_component(component, **kwargs):
    global txt2img_submit_button, img2img_submit_button
    global txt2img_gallery_component, txt2img_prompt_component, txt2img_neg_prompt_component
    global txt2img_html_info_component, txt2img_generation_info_component
    global txt2img_width_component, txt2img_height_component
    global txt2img_upscale_button, sam3_quick_button
    global tipo_button, tipo_panel
    global refine_panel, anima_panel, anima_reference_panel

    # Gradio rebuilds a component at *request* time whenever a handler returns
    # gr.update() for it — blocks.py postprocess_data does
    #     state[block._id] = block.__class__(**constructor_args | {"render": False})
    # and constructor_args still carries the original elem_id. webui patches
    # gradio.components.Component.__init__ (modules/gradio_extensions.py), so
    # this callback also fires for those throwaway instances, with the very
    # elem_ids we match on below.
    #
    # gradio forces render=False on that rebuild (blocks.py:1740), so render()
    # never runs and the instance's _id never lands in demo.default_config.blocks.
    # SessionState.blocks_config is only a shallow snapshot of that dict
    # (BlocksConfig.__copy__), so caching such an instance and later wiring it as
    # an event output makes gradio raise
    #     KeyError: <id>   in state_holder.__contains__
    # the next time the event fires.
    #
    # Testing registration directly is what we actually care about, and it is
    # exact: by the time webui fires this callback, Component.__init__ has already
    # returned, so every genuine component is registered — including one declared
    # inside a render=False container (only the container itself defers, cf. the
    # Compact prompt layout in modules/ui_toprow.py). Checking Context.root_block
    # alone would still leak if an in-flight request echoed during a Reload UI
    # rebuild, since it is a process-wide global rather than a ContextVar.
    if Context.root_block is None:
        return
    if component._id not in Context.root_block.default_config.blocks:
        return

    _wire_sam3_quick_button()
    _wire_tipo()
    elem_id = kwargs.get("elem_id")
    if elem_id == "txt2img_generate":
        txt2img_submit_button = component
    elif elem_id == "img2img_generate":
        img2img_submit_button = component
    elif elem_id == "txt2img_prompt":
        txt2img_prompt_component = component
    elif elem_id == "txt2img_neg_prompt":
        txt2img_neg_prompt_component = component
    elif elem_id == "txt2img_style_apply":
        tipo_button = _create_tipo_button()   # 도구 줄에서 스타일 적용 버튼 오른쪽
    elif elem_id == "txt2img_styles_row":
        tipo_panel = _build_tipo_panel()      # 스타일 줄(Row)이 만들어진 직후 — 같은 열에서 그 아래에 붙는다
    elif elem_id == "txt2img_upscale":
        # ✨ 는 갤러리 아래 버튼 줄의 마지막 — 지금 만들면 그 오른쪽에 붙는다
        txt2img_upscale_button = component
        sam3_quick_button = _create_sam3_quick_button()
    elif elem_id == "txt2img_gallery":
        txt2img_gallery_component = component
    elif elem_id == "txt2img_width":
        txt2img_width_component = component
    elif elem_id == "txt2img_height":
        txt2img_height_component = component
    elif elem_id == "html_info_txt2img":
        txt2img_html_info_component = component
    elif elem_id == "download_files_txt2img":
        # 갤러리 버튼 줄 바로 뒤, 생성 정보 그룹(overflow:hidden) 밖 — 선택 이미지 탭을 여기에 만든다
        # (modules/ui_common.py:190-228).
        if not dock_state.needs_build(Context.root_block):
            return
        try:
            panels = ui_dock.build_selected_image_dock(
                [s.name for s in _all_samplers],
                [s.label for s in _all_schedulers],
                find_checkpoint_options(),
                anima_ok=anima_available(),
            )
            dock_state.mark_built(panels, Context.root_block)
            refine_panel = panels.refine
            anima_panel = panels.anima
            anima_reference_panel = panels.reference
        except Exception:
            error = traceback.format_exc()
            print(f"[-] SAM3: failed to build the selected-image dock:\n{error}", file=sys.stderr)
    elif elem_id == "generation_info_txt2img":
        # 여기서는 연결만 한다. 패널은 위의 download_files 분기에서 이미 만들었고(없으면 예비로 만든다),
        # 연결에 필요한 갤러리·infotext 컴포넌트가 이 시점에 모두 잡혀 있다.
        txt2img_generation_info_component = component
        try:
            import modules.scripts as _scripts

            layout_lanes.tag_runner(_scripts.scripts_txt2img)
        except Exception:
            error = traceback.format_exc()
            print(f"[-] SAM3: failed to tag script slots:\n{error}", file=sys.stderr)
        if txt2img_gallery_component is None or txt2img_html_info_component is None:
            print(
                "[-] SAM3: gallery/html_info not captured yet — skipping panel wiring.",
                file=sys.stderr,
            )
            return
        if dock_state.needs_build(Context.root_block):   # 예비: download_files 훅이 없었거나 실패한 경우
            try:
                panels = ui_dock.build_selected_image_dock(
                    [s.name for s in _all_samplers],
                    [s.label for s in _all_schedulers],
                    find_checkpoint_options(),
                    anima_ok=anima_available(),
                )
                dock_state.mark_built(panels, Context.root_block)
                refine_panel = panels.refine
                anima_panel = panels.anima
                anima_reference_panel = panels.reference
            except Exception:
                error = traceback.format_exc()
                print(f"[-] SAM3: failed to build the dock (fallback):\n{error}", file=sys.stderr)
        if dock_state.needs_wire("refine", refine_panel):
            try:
                _wire_refine_panel(
                    refine_panel,
                    txt2img_gallery_component,
                    txt2img_prompt_component,
                    txt2img_neg_prompt_component,
                    txt2img_html_info_component,
                    txt2img_generation_info_component,
                )
                dock_state.mark_wired("refine")
            except Exception:
                error = traceback.format_exc()
                print(f"[-] SAM3: failed to wire Refine panel:\n{error}", file=sys.stderr)
        if dock_state.needs_wire("anima", anima_panel):
            try:
                _wire_anima_panel(
                    anima_panel,
                    txt2img_gallery_component,
                    txt2img_html_info_component,
                    txt2img_generation_info_component,
                )
                dock_state.mark_wired("anima")
            except Exception:
                error = traceback.format_exc()
                print(f"[-] SAM3: failed to wire Anima panel:\n{error}", file=sys.stderr)
        if dock_state.needs_wire("reference", anima_reference_panel):
            try:
                wire_anima_reference_panel(
                    anima_reference_panel,
                    gallery=txt2img_gallery_component,
                    main_prompt=txt2img_prompt_component,
                    main_negative=txt2img_neg_prompt_component,
                    html_info=txt2img_html_info_component,
                    generation_info=txt2img_generation_info_component,
                    width=txt2img_width_component,
                    height=txt2img_height_component,
                    selected_index_js=_SELECTED_INDEX_JS,
                )
                dock_state.mark_wired("reference")
            except Exception:
                error = traceback.format_exc()
                print(
                    "[-] SAM3: failed to wire Feature 6 Anima Reference "
                    f"panel:\n{error}",
                    file=sys.stderr,
                )


script_callbacks.on_after_component(on_after_component)
