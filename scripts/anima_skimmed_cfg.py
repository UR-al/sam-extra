"""Skimmed CFG — anti-burn skimming of the cond/uncond predictions.

Reimplemented for Forge from the public formulas of
https://github.com/Extraltodeus/Skimmed_CFG (no LICENSE file is published in
that repository, so nothing is vendored here — only the described maths are
rewritten in this extension's own style, as with the other guidance sources
credited in ``docs/GUIDANCE.md``).

Upstream is a ComfyUI *pre*-CFG node: it rewrites ``conds_out`` in place before
the CFG combine, which is why everything downstream of it — the combine itself
and any further guidance — operates on the skimmed predictions. Forge's
``sampler_pre_cfg_function`` has a different contract (it runs on the
conditioning lists *before* the predictions exist), so the same maths are
applied here from a post-CFG hook instead: Forge hands us ``cond_denoised``,
``uncond_denoised``, ``input`` and ``cond_scale``, which is everything the
skimming formula needs.

To keep upstream's composition semantics, the skimmed predictions are written
back into Forge's own tensors in place. Forge rebuilds the post-CFG args dict
per registered function but reuses the same prediction tensors, so later hooks
(Safe PAG's SMC/APG/CWM base, the PAG/SEG/SLG delta, DCW) see the skim exactly
as a ComfyUI graph would.

Forge Neo currently defines ``ScriptRunner.process_before_every_sampling``
twice; the later definition iterates the raw ``alwayson_scripts`` list and
therefore bypasses ``sorting_priority``. To retain pre-CFG semantics without a
Forge core edit, this script explicitly prepends its post-CFG hook to the
cloned UNet's callback list. The insertion is owner-tagged and de-duplicated so
hires passes and script reloads cannot stack stale copies.
"""

from __future__ import annotations

import sys
import traceback
from functools import partial

import gradio as gr

from modules import script_callbacks, scripts, shared

from sam3ext import layout_lanes

try:
    import torch
except Exception:  # pragma: no cover - torch is always present under Forge
    torch = None


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


def _sampling_position() -> tuple[int, int]:
    """Forge's authoritative 0-based sampler position (see anima_safe_pag)."""
    try:
        state = getattr(shared, "state", None)
        step = int(getattr(state, "sampling_step"))
        total = int(getattr(state, "sampling_steps"))
        if total > 0:
            return max(0, step), total
    except Exception:
        pass
    return 0, 1


def _pct_now() -> float:
    step, total = _sampling_position()
    return min(1.0, max(0.0, step / max(total - 1, 1)))


def _skim_predictions(x, target, reference, scale, skimming_scale, flip_filter):
    """Return ``target`` with its 'burning' elements pulled toward ``skimming_scale``.

    ``target``/``reference`` are denoised (x0) predictions and ``scale`` is the
    CFG scale that would be applied to them, so
    ``denoised = reference + scale * (target - reference)`` matches Forge's own
    linear combine. Elements are skimmed only where the guidance pushes the
    prediction further out along a direction it already agrees with — that is
    the set upstream calls ``outer_influence``.
    """
    if abs(float(scale)) < _MIN_SCALE:
        return target  # CFG 1: the combine contributes nothing to skim

    denoised = reference + scale * (target - reference)
    matching_pred_signs = (target - reference).sign() == target.sign()
    matching_diff_after = target.sign() == denoised.sign()
    outer_influence = matching_pred_signs & matching_diff_after
    if not flip_filter:
        outer_influence &= denoised.sign() == (denoised - x).sign()

    # torch.where 로 고른다. 예전의 ``outer_influence.any()`` 확인과 불리언 마스크
    # gather/scatter(skimmed[mask] = target[mask] - correction[mask])는 호출마다
    # GPU→CPU 동기화를 4번 일으켰다. 원소별 계산이 같으므로 결과는 비트 단위로
    # 같고, 마스크가 비어 있으면 target 과 같은 값이 그대로 나온다.
    low_scale_denoised = reference + skimming_scale * (target - reference)
    correction = (denoised - low_scale_denoised) / scale
    return torch.where(outer_influence, target - correction, target)


def _post_cfg(args):
    """Recompute the CFG combine from skimmed predictions."""
    denoised = args["denoised"]
    if not _SKIM["on"] or torch is None:
        return denoised

    pct = _pct_now()
    if not (float(_SKIM["start"]) <= pct <= float(_SKIM["end"])):
        return denoised

    try:
        x = args["input"]
        cond = args["cond_denoised"]
        uncond = args["uncond_denoised"]
        cond_scale = float(args.get("cond_scale", 1.0))
    except Exception:
        return denoised

    if not torch.is_tensor(x) or not torch.is_tensor(cond):
        return denoised
    if not torch.is_tensor(uncond) or uncond.shape != cond.shape:
        return denoised  # CFG 1 / positive-only path has no uncond to skim
    if abs(cond_scale - 1.0) < _MIN_SCALE:
        if not _SKIM["warned"]:
            _SKIM["warned"] = True
            _log("CFG scale is 1 — skimming has nothing to correct; skipped.")
        return denoised

    try:
        requested = float(_SKIM["skimming_cfg"])
        practical_scale = cond_scale if requested < 0 else requested

        flip_filter = bool(_SKIM["disable_flipping_filter"])
        flip_at = float(_SKIM["flip_at"])
        if flip_at > 0 and pct < flip_at:
            flip_filter = not flip_filter

        x_f = x.float()
        cond_f = cond.float()
        uncond_f = uncond.float()

        # Upstream order: skim the negative against the positive first, then
        # the positive against the freshly skimmed negative at ``scale - 1``.
        uncond_skimmed = _skim_predictions(
            x_f, uncond_f, cond_f, cond_scale,
            0.0 if _SKIM["full_skim_negative"] else practical_scale,
            flip_filter,
        )
        cond_skimmed = _skim_predictions(
            x_f, cond_f, uncond_skimmed, cond_scale - 1.0,
            practical_scale, flip_filter,
        )

        result = uncond_skimmed + cond_scale * (cond_skimmed - uncond_skimmed)
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # Publish the skimmed predictions the way upstream's pre-CFG node does.
        # Forge rebuilds the args dict per post-CFG function but reuses the same
        # cond_pred/uncond_pred tensors (backend/sampling/sampling_function.py),
        # so writing in place is what lets the rest of the suite — Safe PAG's
        # SMC/APG/CWM base, the PAG/SEG/SLG delta, DCW — compose on top of the
        # skim instead of rebuilding from the unskimmed originals.
        cond.copy_(cond_skimmed.to(cond.dtype))
        uncond.copy_(uncond_skimmed.to(uncond.dtype))

        _SKIM["steps"] += 1
        return result.to(denoised.dtype)
    except Exception as exc:  # keep the generation alive on any surprise
        if not _SKIM["warned"]:
            _SKIM["warned"] = True
            _log(f"fallback (earlier guidance kept): {type(exc).__name__}: {exc}")
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
                    "**start/end**=적용 스텝 구간, **flip at**=지정 지점 이전에서 "
                    "flipping filter를 뒤집어 초반 구도를 다르게 잡습니다(0=사용 안 함), "
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
                    minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                    info="0.3 부근이 upstream 기본값입니다. 0에 가까울수록 부드럽습니다.",
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

        _SKIM.update(on=False, steps=0, warned=False)

        try:
            enabled = bool(_arg(0, False))
        except Exception:
            enabled = False
        if "enabled" in xyz:
            enabled = _as_bool(xyz["enabled"], enabled)
        if not enabled:
            return

        try:
            start = _xyz_num("start", float(_arg(4, 0.0)))
            end = _xyz_num("end", float(_arg(5, 1.0)))
            if end < start:
                start, end = end, start
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
                start=min(max(start, 0.0), 1.0),
                end=min(max(end, 0.0), 1.0),
                flip_at=min(max(_xyz_num("flip_at", float(_arg(6, 0.0))), 0.0), 1.0),
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
        _SKIM.update(on=False, steps=0, warned=False)
