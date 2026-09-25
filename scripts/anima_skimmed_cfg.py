"""Skimmed CFG — anti-burn skimming of the cond/uncond predictions.

Port of the ``CFG_Skimming_Single_Scale_Pre_CFG`` node of
https://github.com/Extraltodeus/Skimmed_CFG (commit d8300583, Apache-2.0). The
upstream maths (``get_skimming_mask``/``skimmed_CFG``) are vendored unmodified
in ``sam3ext/guidance/skimmed_cfg.py`` together with the node's sigma gate, flip
rule and skim order; the notice is in ``THIRD_PARTY_NOTICES.md``.

Behaviour matches upstream:

* The start/end/flip percentages become sigmas through the sampling model's own
  ``predictor.percent_to_sigma`` (ComfyUI: ``model_sampling.percent_to_sigma``)
  and the current sigma is read from the hook args, not from a step counter. A
  step is skimmed only when ``end_sigma < sigma < start_sigma`` (strict), so on
  a flow model such as Anima the first step (sigma 1.0 = percent 0) is never
  skimmed, and a reversed start/end skims nothing.
* The filter is flipped while ``flip_at > 0 and sigma > flip_sigma``.
* The negative is skimmed first, then the positive against the skimmed negative
  at ``cond_scale - 1``; a negative prediction that is all zeros (Forge's CFG 1
  path) is left alone.

Host differences that remain:

* Upstream is a ComfyUI *pre*-CFG node that rewrites ``conds_out`` before the
  CFG combine. Forge's ``sampler_pre_cfg_function`` runs before the predictions
  exist, so this script hooks the front of the post-CFG list instead, skims the
  predictions, writes them back into Forge's tensors in place and re-runs
  Forge's own CFG step on them: the registered ``sampler_cfg_function``
  (RescaleCFG, Dynamic Thresholding, ...) with Forge's argument dict, otherwise
  the linear combine with Forge's ``edit_strength``. That is what every later
  consumer would have seen after an upstream pre-CFG rewrite. A registered
  ``sampler_cfg_function`` therefore runs twice on skimmed steps (Forge's call
  on the raw predictions is discarded).
* CFG exactly 1 with the uncond pass forced on is skipped: upstream divides by
  ``cond_scale - 1`` there.
* The Skimming CFG slider allows -1 (use the live CFG) so one script covers the
  Clean Skim and Timed flip presets; upstream's main node hides that behind the
  preset nodes.

Forge's post-CFG args are rebuilt per registered function but reuse the same
prediction tensors, so the in-place write lets later hooks (Safe PAG's
SMC/APG/CWM base, the PAG/SEG/SLG delta, DCW) see the skim exactly as a ComfyUI
graph would.

Forge Neo currently defines ``ScriptRunner.process_before_every_sampling``
twice; the later definition iterates the raw ``alwayson_scripts`` list and
therefore bypasses ``sorting_priority``. To retain pre-CFG semantics without a
Forge core edit, this script explicitly prepends its post-CFG hook to the
cloned UNet's callback list. The insertion is owner-tagged and de-duplicated so
hires passes and script reloads cannot stack stale copies.
"""

from __future__ import annotations

import math
import sys
import traceback
from functools import partial

import gradio as gr

from modules import script_callbacks, scripts

from sam3ext import layout_lanes

try:
    import torch
except Exception:  # pragma: no cover - torch is always present under Forge
    torch = None

# Deliberately unguarded (like Safe PAG's sam3ext.guidance imports): if the
# vendored maths cannot be imported, the script must fail to load with Forge's
# traceback instead of leaving a checkbox that silently does nothing.
from sam3ext.guidance.skimmed_cfg import (  # noqa: E402
    flip_filter_at,
    skim_active,
    skim_pair,
    skim_sigmas,
)


# ---------------------------------------------------------------------------
# Neutral by default: installing or updating the extension cannot change a
# generation until the checkbox is ticked.
# ---------------------------------------------------------------------------

_SKIM: dict = {
    "on": False,
    "skimming_cfg": 7.0,
    "full_skim_negative": False,
    "disable_flipping_filter": False,
    "start": 0.0,
    "end": 1.0,
    "flip_at": 0.0,
    "steps": 0,
    "warned": False,
    # (predictor, (start_sigma, end_sigma, flip_sigma)) for the current pass.
    "sigmas": None,
}

_MIN_SCALE = 1e-6
_POST_CFG_OWNER = "sam-extra/anima-skimmed-cfg"


def _log(message: str) -> None:
    """Print a diagnostic without ever being able to abort a generation.

    Some of these messages carry non-ASCII characters, and on a Windows console
    running a legacy code page (cp949, cp1252, ...) print raises
    UnicodeEncodeError. _log is called from inside the post-CFG hook, so an
    exception here would propagate into the sampler and kill the run for the
    sake of a log line. Degrade to an ASCII-safe rendering instead.
    """
    text = f"[AnimaSkimmedCFG] {message}"
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(encoding, "replace").decode(encoding, "replace"))
        except Exception:
            pass
    except Exception:
        pass


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "on", "enable", "enabled"):
            return True
        if text in ("false", "0", "no", "off", "disable", "disabled"):
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# XYZ plot integration
# ---------------------------------------------------------------------------


def _skim_xyz_set(p, x, xs, *, field: str):
    if not hasattr(p, "_anima_skimmed_cfg_xyz"):
        p._anima_skimmed_cfg_xyz = {}
    p._anima_skimmed_cfg_xyz[field] = x


def _make_skim_xyz_axis() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return
    bool_choices = lambda: ["True", "False"]  # noqa: E731
    axis = [
        xyz_grid.AxisOption(
            "[Anima Skim] Enable", str,
            partial(_skim_xyz_set, field="enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima Skim] Skimming CFG", float,
            partial(_skim_xyz_set, field="skimming_cfg"),
        ),
        xyz_grid.AxisOption(
            "[Anima Skim] Full Skim Negative", str,
            partial(_skim_xyz_set, field="full_skim_negative"),
            choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima Skim] Disable Flipping Filter", str,
            partial(_skim_xyz_set, field="disable_flipping_filter"),
            choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima Skim] Start", float,
            partial(_skim_xyz_set, field="start"),
        ),
        xyz_grid.AxisOption(
            "[Anima Skim] End", float,
            partial(_skim_xyz_set, field="end"),
        ),
        xyz_grid.AxisOption(
            "[Anima Skim] Flip At", float,
            partial(_skim_xyz_set, field="flip_at"),
        ),
    ]
    if not any(a.label.startswith("[Anima Skim]") for a in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def _skim_on_before_ui() -> None:
    try:
        _make_skim_xyz_axis()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_skim_on_before_ui)


def _warn_once(message: str) -> None:
    if not _SKIM["warned"]:
        _SKIM["warned"] = True
        _log(message)


def _gate_sigmas(model):
    """``(start, end, flip)`` sigmas from the sampling model's own schedule.

    Upstream converts the percentages once, when the node patches the model
    (after any shift node). Here the predictor comes from the post-CFG args —
    the KModel Forge is sampling with, after Forge applied its shift — and the
    conversion is cached for the pass. ``None`` when there is no predictor.
    """
    predictor = getattr(model, "predictor", None)
    percent_to_sigma = getattr(predictor, "percent_to_sigma", None)
    if not callable(percent_to_sigma):
        return None
    cached = _SKIM["sigmas"]
    if cached is not None and cached[0] is predictor:
        return cached[1]
    sigmas = skim_sigmas(
        percent_to_sigma,
        float(_SKIM["start"]), float(_SKIM["end"]), float(_SKIM["flip_at"]),
    )
    _SKIM["sigmas"] = (predictor, sigmas)
    start_sigma, end_sigma, flip_sigma = sigmas
    flip_text = (
        f", filter flipped while sigma > {flip_sigma:.4f}"
        if float(_SKIM["flip_at"]) > 0 else ""
    )
    _log(
        f"sigma window: skims while {end_sigma:.4f} < sigma < "
        f"{start_sigma:.4f}{flip_text}"
    )
    return sigmas


def _edit_strength(conds) -> float:
    """Forge's ``edit_strength`` for the positive conds (1 when unknown)."""
    if conds is None:
        return 1.0
    try:
        return float(sum(
            (item["strength"] if "strength" in item else 1) for item in conds
        ))
    except Exception:
        return 1.0


def _forge_cfg(args, cond_pred, uncond_pred):
    """Forge's own CFG step, run on the given predictions.

    Same branches and argument dict as ``sampling_function_inner`` in
    ``backend/sampling/sampling_function.py``: a registered
    ``sampler_cfg_function`` gets Forge's dict and its result is subtracted
    from the input; otherwise the linear combine, scaled by ``edit_strength``
    when the positive conds carry a strength.
    """
    x = args["input"]
    cond_scale = args["cond_scale"]
    model_options = args.get("model_options") or {}
    if "sampler_cfg_function" in model_options:
        cfg_args = {
            "cond": x - cond_pred,
            "uncond": x - uncond_pred,
            "cond_scale": cond_scale,
            "timestep": args.get("sigma"),
            "input": x,
            "sigma": args.get("sigma"),
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "model": args.get("model"),
            "model_options": model_options,
        }
        return x - model_options["sampler_cfg_function"](cfg_args)
    edit_strength = _edit_strength(args.get("cond"))
    if not math.isclose(edit_strength, 1.0):
        return uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    return uncond_pred + (cond_pred - uncond_pred) * cond_scale


def _post_cfg(args):
    """Skim the predictions (upstream pre-CFG patch) and redo Forge's CFG step."""
    denoised = args["denoised"]
    if not _SKIM["on"] or torch is None:
        return denoised

    try:
        x = args["input"]
        cond = args["cond_denoised"]
        uncond = args["uncond_denoised"]
        cond_scale = args["cond_scale"]
        sigma = args["sigma"][0].item()
    except Exception:
        return denoised

    if not (torch.is_tensor(x) and torch.is_tensor(cond)):
        return denoised
    if not torch.is_tensor(uncond) or uncond.shape != cond.shape:
        return denoised

    try:
        sigmas = _gate_sigmas(args.get("model"))
    except Exception as exc:
        _warn_once(f"percent_to_sigma failed, skipped: {type(exc).__name__}: {exc}")
        return denoised
    if sigmas is None:
        _warn_once("the sampling model has no predictor.percent_to_sigma — "
                   "the sigma window cannot be placed; skipped.")
        return denoised
    start_sigma, end_sigma, flip_sigma = sigmas
    if not skim_active(sigma, start_sigma, end_sigma):
        return denoised
    if not torch.any(uncond):
        return denoised  # upstream: no negative prediction (CFG 1 path)
    if abs(float(cond_scale) - 1.0) < _MIN_SCALE:
        _warn_once("CFG scale is 1 — skimming would divide by zero; skipped.")
        return denoised

    try:
        flip_filter = flip_filter_at(
            sigma, bool(_SKIM["disable_flipping_filter"]),
            float(_SKIM["flip_at"]), flip_sigma,
        )
        cond_skimmed, uncond_skimmed = skim_pair(
            x, cond.clone(), uncond.clone(), cond_scale,
            float(_SKIM["skimming_cfg"]), bool(_SKIM["full_skim_negative"]),
            flip_filter,
        )
        result = _forge_cfg(args, cond_skimmed, uncond_skimmed)

        # Publish the skimmed predictions the way upstream's pre-CFG node does.
        # Forge rebuilds the args dict per post-CFG function but reuses the same
        # cond_pred/uncond_pred tensors (backend/sampling/sampling_function.py),
        # so writing in place is what lets the rest of the suite — Safe PAG's
        # SMC/APG/CWM base, the PAG/SEG/SLG delta, DCW — compose on top of the
        # skim instead of rebuilding from the unskimmed originals.
        cond.copy_(cond_skimmed)
        uncond.copy_(uncond_skimmed)

        _SKIM["steps"] += 1
        return result
    except Exception as exc:  # keep the generation alive on any surprise
        _warn_once(f"fallback (earlier guidance kept): {type(exc).__name__}: {exc}")
        return denoised


_post_cfg._sam_extra_post_cfg_owner = _POST_CFG_OWNER


def _prepend_post_cfg_function(unet, function=_post_cfg) -> None:
    """Install ``function`` first while preserving every unrelated callback.

    ``ModelPatcher.set_model_sampler_post_cfg_function`` is intentionally an
    append-only ComfyUI API. Skimmed CFG emulates a *pre*-CFG prediction rewrite,
    though, so it must run before every ordinary post-CFG transform. Owner
    tagging removes both this module's current function and a stale function
    left by WebUI's "Reload scripts" before inserting exactly one copy.
    """
    model_options = getattr(unet, "model_options", None)
    if not isinstance(model_options, dict):
        model_options = {}

    callbacks = model_options.get("sampler_post_cfg_function", [])
    if not isinstance(callbacks, (list, tuple)):
        callbacks = []
    kept = [
        callback
        for callback in callbacks
        if getattr(callback, "_sam_extra_post_cfg_owner", None)
        != _POST_CFG_OWNER
    ]
    function._sam_extra_post_cfg_owner = _POST_CFG_OWNER

    # Copy the top-level dict instead of mutating a possibly shared options
    # object. ModelPatcher.clone() already deep-copies it in current Forge, but
    # this also keeps compatible forks safe.
    model_options = dict(model_options)
    model_options["sampler_post_cfg_function"] = [function, *kept]
    unet.model_options = model_options


class AnimaSkimmedCFG(scripts.Script):
    @property
    def section(self):
        # Forge 는 사용자 섹션을 설정값 칼럼(#txt2img_settings) 안에 만든다 → 1열 "ANIMA 튜닝" 자리.
        # 설정(sam3_layout_sections)을 끄거나 img2img 면 None 이라 예전과 똑같이 스크립트 컨테이너로 간다.
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    # Larger sorting_priority appears further down. This places the accordion
    # directly under Anima Detail Daemon (-29) and above Anima Safe PAG (-27).
    # Runtime hook precedence does NOT rely on this value;
    # _prepend_post_cfg_function enforces it explicitly because current Forge
    # ignores the sorted runner method.
    sorting_priority = -28

    def title(self):
        return "Anima Skimmed CFG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima Skimmed CFG (CFG 과포화 완화)", open=False):
            gr.Markdown(
                "높은 CFG에서 **타는 듯한 과포화·번짐**을 만드는 성분만 골라 낮은 CFG "
                "값으로 되돌립니다(anti-burn). 추가 forward가 없어 속도 비용이 거의 "
                "없고, CFG를 평소보다 높게 쓸 수 있게 해 줍니다. **CFG > 1 전용**입니다."
            )
            enabled = gr.Checkbox(
                label="Enable Skimmed CFG",
                value=False,
                elem_id="anima_skim_enable",
                elem_classes=["sam3-on"],
            )
            skimming_cfg = gr.Slider(
                label="Skimming CFG (되돌릴 기준 스케일 · -1 = 현재 CFG 사용)",
                minimum=-1.0, maximum=10.0, step=0.5, value=7.0,
                info="과포화가 남으면 낮추세요. 너무 낮으면 대비·채도가 함께 죽습니다.",
                elem_id="anima_skim_cfg",
            )
            full_skim_negative = gr.Checkbox(
                label="Full skim negative (네거티브를 0까지 완전히 skim)",
                value=False,
                info="upstream의 Clean Skim 프리셋은 이 옵션 + Skimming CFG = -1 조합입니다.",
                elem_id="anima_skim_full_negative",
            )
            with gr.Accordion("Skimmed CFG Advanced (세부값)", open=False):
                gr.Markdown(
                    "**start/end**=적용 구간(%)입니다. 원본 노드와 같이 모델의 노이즈 "
                    "스케줄로 %를 σ로 바꿔, σ가 start의 σ보다 작고 end의 σ보다 큰 "
                    "스텝만 skim합니다(경계 제외). 그래서 스케줄러마다 해당 스텝이 "
                    "다르고, Anima 같은 flow 모델에서는 첫 스텝(σ=1)을 깎지 않습니다. "
                    "**flip at**=그 지점의 σ보다 앞선(σ가 큰) 스텝에서 flipping filter를 "
                    "뒤집어 초반 구도를 다르게 잡습니다(0=사용 안 함), "
                    "**disable flipping filter**=필터를 아예 끄면 더 거칠어집니다."
                )
                disable_flipping_filter = gr.Checkbox(
                    label="Disable flipping filter",
                    value=False,
                    elem_id="anima_skim_disable_flip",
                )
                with gr.Row():
                    start_percent = gr.Slider(
                        label="Start at (%)",
                        minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                        elem_id="anima_skim_start",
                    )
                    end_percent = gr.Slider(
                        label="End at (%)",
                        minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                        elem_id="anima_skim_end",
                    )
                flip_at = gr.Slider(
                    label="Flip at (%) · 0 = 사용 안 함",
                    minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                    info="upstream Timed flip 노드의 기본값은 0.3입니다. 0에 가까울수록 부드럽습니다.",
                    elem_id="anima_skim_flip_at",
                )
        return [
            enabled, skimming_cfg, full_skim_negative,
            disable_flipping_filter, start_percent, end_percent, flip_at,
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if torch is None:
            return

        xyz = getattr(p, "_anima_skimmed_cfg_xyz", {}) or {}

        def _arg(i, default):
            return args[i] if len(args) > i else default

        def _xyz_num(key, cur):
            if key in xyz:
                try:
                    return float(xyz[key])
                except (TypeError, ValueError):
                    return cur
            return cur

        _SKIM.update(on=False, steps=0, warned=False, sigmas=None)

        try:
            enabled = bool(_arg(0, False))
        except Exception:
            enabled = False
        if "enabled" in xyz:
            enabled = _as_bool(xyz["enabled"], enabled)
        if not enabled:
            return

        try:
            # Passed through as-is, like upstream: no swap (a reversed window
            # skims nothing) and no clamp (percent_to_sigma saturates at 0/1).
            _SKIM.update(
                on=True,
                skimming_cfg=_xyz_num("skimming_cfg", float(_arg(1, 7.0))),
                full_skim_negative=_as_bool(
                    xyz.get("full_skim_negative", _arg(2, False)),
                    bool(_arg(2, False)),
                ),
                disable_flipping_filter=_as_bool(
                    xyz.get("disable_flipping_filter", _arg(3, False)),
                    bool(_arg(3, False)),
                ),
                start=_xyz_num("start", float(_arg(4, 0.0))),
                end=_xyz_num("end", float(_arg(5, 1.0))),
                flip_at=_xyz_num("flip_at", float(_arg(6, 0.0))),
            )
        except Exception as exc:
            _SKIM["on"] = False
            _log(f"invalid arguments, skipped: {type(exc).__name__}: {exc}")
            return

        sd_model = getattr(p, "sd_model", None)
        forge_objects = getattr(sd_model, "forge_objects", None)
        unet = getattr(forge_objects, "unet", None)
        if unet is None:
            _SKIM["on"] = False
            _log("no forge_objects.unet — cannot attach skimming.")
            return

        # Clone from the CURRENT unet so other patching scripts compose. Do not
        # use the append-only setter here: Forge's raw always-on script order
        # currently runs anima_safe_pag.py before this file, and appending would
        # make Skimmed discard the already-computed PAG/CWM/SMC/DCW result.
        unet = unet.clone()
        _prepend_post_cfg_function(unet)
        p.sd_model.forge_objects.unet = unet

        if not hasattr(p, "extra_generation_params"):
            p.extra_generation_params = {}
        p.extra_generation_params["Anima Skimmed CFG"] = (
            f"skimming_cfg={_SKIM['skimming_cfg']}, "
            f"full_skim_negative={_SKIM['full_skim_negative']}, "
            f"flip_filter_off={_SKIM['disable_flipping_filter']}, "
            f"range={_SKIM['start']:.2f}-{_SKIM['end']:.2f}, "
            f"flip_at={_SKIM['flip_at']:.2f}"
        )
        _log(
            f"attached ✅ skimming_cfg={_SKIM['skimming_cfg']} "
            f"full_skim_negative={_SKIM['full_skim_negative']} "
            f"flip_filter={'off' if _SKIM['disable_flipping_filter'] else 'on'} "
            f"range={_SKIM['start']:.2f}-{_SKIM['end']:.2f} "
            f"flip_at={_SKIM['flip_at']:.2f}"
        )

    def postprocess(self, p, processed, *args):
        if _SKIM["on"]:
            _log(f"skimmed steps={_SKIM['steps']}")
        _SKIM.update(on=False, steps=0, warned=False, sigmas=None)
