# Derived from (MIT License, notice below):
#   aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 scripts/speed_forge.py:187-413 (the
#     Forge Neo script: accordion, inputs and their defaults, the ``SPEED`` infotext, wrapping the
#     selected sampler) — Copyright (c) 2026 A. Izzuddin Al Faruq (the Forge Neo script, presets,
#     adaptive delta and neo_shift in that fork are Oleg Afonin's work); based on howardhx/speed
#     (MIT, Copyright (c) 2026 Howard Xiao). The respace mode follows
#     sorryhyun/ComfyUI-Spectrum-KSampler@b46a364a (MIT, Copyright (c) 2026 sorryhyun).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Modified by sam-extra, 2026-10-03 (part of a GPL-3.0-only extension; the MIT notice stays):
#   * sits in the Anima lane (layout_lanes), title "Anima SPEED", default preset ``anima``;
#   * a ``Mode`` switch: ``transition`` (default, official/aoleg) or ``respace`` (sorryhyun, the mode
#     the desktop app's ComfyUI pack runs through SpectrumSPDKSampler);
#   * aoleg's ``Mode`` (neo_shift / delta_optimal / manual) became ``Transition sigma``;
#   * no state on the Script class; settings are coerced once and closed over by the wrapper;
#   * infotext ``Anima SPEED`` (settings, restored by paste) and ``Anima SPEED status`` (what each
#     pass did or why it was skipped) replace aoleg's single ``SPEED`` string;
#   * XYZ axes, two Forge settings (console log, the img2img init rescale);
#   * the run itself is sam3ext/speed (see the module headers for the remaining changes).
"""Anima SPEED — Spectral Progressive Diffusion (Xiao et al., arXiv 2605.18736). Default OFF.

The early, noise-dominated steps run on a DCT-truncated coarse latent; at a transition sigma the
coarse DCT block is embedded in the full grid, the new high frequencies are filled with sigma-scaled
noise, the state is rescaled by ``kappa = r / (1 + (r - 1) sigma)`` and the selected sampler continues
at full size. Fewer tokens in the early steps = less time; the image changes (it is not a lossless
speed-up).

Modes:

* ``transition`` (default) — official howardhx/speed and aoleg/ComfyUI-SPEED: only the transition
  step's sigma is patched to the aligned time ``sigma * kappa``; coarse grids are ``round(s * H)``.
* ``respace`` — sorryhyun/ComfyUI-Spectrum-KSampler (``SpectrumSPDKSampler``, which the desktop app's
  ComfyUI pack calls): every remaining sigma is scaled by ``sigma_aligned / sigma`` and coarse grids
  are snapped to even sizes. Upstream forces Euler; here the selected sampler runs per segment (Euler
  gives sorryhyun's loop). Upstream's Spectrum forecasting of the full-size tail is not part of it:
  a geometric match only, and odd latents differ (sorryhyun pads them to even before sampling).

The transition sigma comes from a power-spectrum preset (``delta`` tolerance, adaptive delta pinned
to the 1024 px reference, ``neo_shift`` divisor 1.03 for Forge's flat shift) or is given by hand.

Skipped, with the reason in ``Anima SPEED status``: masks / inpaint models, reference latents (Anima,
Flux Kontext / Flux.2 Klein / Qwen-Image-Edit / Krea 2), Wan 2.2 I2V conditioning, PiD, ControlNet
(incl. LLLite), non-flow models (SDXL/SD1.x), Forge's Spectrum Integrated, Restart/UniPC and samplers
without a sigma list, passes with no coarse step (img2img/hires starting at or below the transition
sigma), and the hires pass unless ``Apply to Hires pass`` is on.
"""

from __future__ import annotations

import sys
import traceback
from functools import partial

import gradio as gr

from modules import script_callbacks, scripts

try:
    from modules import shared as _shared
except Exception:  # pragma: no cover - always present under Forge
    _shared = None

from sam3ext import layout_lanes
from sam3ext.speed import forge_host as fh
from sam3ext.speed import schedule

TITLE = "Anima SPEED"
KEY = fh.KEY
STATUS_KEY = fh.STATUS_KEY
OPT_LOG = fh.OPT_LOG
OPT_IMG2IMG_RESCALE = fh.OPT_IMG2IMG_RESCALE
INFOTEXT_IMG2IMG_RESCALE = fh.INFOTEXT_IMG2IMG_RESCALE
XYZ_PREFIX = "[Anima SPEED]"


def _option(name: str, default):
    try:
        return getattr(_shared.opts, name, default)
    except Exception:
        return default


def _log(message: str) -> None:
    """Console line; never raises (legacy Windows code pages, closed streams)."""
    if not bool(_option(OPT_LOG, True)):
        return
    text = f"[AnimaSPEED] {message}"
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


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------


def _on_ui_settings() -> None:
    shared = _shared
    section = ("sam3_speed", "SAM Extra SPEED")
    shared.opts.add_option(
        OPT_LOG,
        shared.OptionInfo(
            True,
            "Anima SPEED: 콘솔에 전환 계획과 결과 출력",
            gr.Checkbox,
            section=section,
        ).info(
            "켜면(기본) 패스마다 전환 스텝·σ·격자와 결과를 `[AnimaSPEED]` 줄로 남깁니다. 이미지는 바뀌지 않습니다."
        ),
    )
    shared.opts.add_option(
        OPT_IMG2IMG_RESCALE,
        shared.OptionInfo(
            True,
            "Anima SPEED: img2img·Hires 의 저해상도 시작 latent 를 flow 형태로 맞추기 (원본 노드와 다름)",
            gr.Checkbox,
            section=section,
            infotext=INFOTEXT_IMG2IMG_RESCALE,
        ).info(
            "DCT 로 자른 저해상도 latent 는 이미지 성분이 √(HW/hw) 배 커지고 노이즈는 그대로라, 원본 노드처럼 그대로 "
            "쓰면 img2img·Hires(σ<1 시작)에서 이미지가 과하게 강한 상태로 저해상도 스텝이 돕니다. 켜면(기본) 이미지 "
            "성분을 원래 크기로 줄이고 모자란 노이즈를 시드별로 채워 (1−σ)·이미지 + σ·노이즈 형태로 맞춥니다. txt2img "
            "(σ=1 시작)는 켜고 끔에 관계없이 같습니다. 끄면 원본(official·aoleg·sorryhyun)과 같은 동작입니다."
        ),
    )


script_callbacks.on_ui_settings(_on_ui_settings)


# ---------------------------------------------------------------------------------------------
# XYZ plot
# ---------------------------------------------------------------------------------------------


def _xyz_set(p, x, xs, *, field: str):
    values = dict(getattr(p, fh.XYZ_ATTR, None) or {})
    values[field] = x
    setattr(p, fh.XYZ_ATTR, values)


def _make_xyz_axes() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return
    axes = [
        xyz_grid.AxisOption(
            f"{XYZ_PREFIX} Enable", str, partial(_xyz_set, field="enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            f"{XYZ_PREFIX} Mode", str, partial(_xyz_set, field="mode"),
            choices=lambda: list(schedule.MODES),
        ),
        xyz_grid.AxisOption(f"{XYZ_PREFIX} Manual sigma", float, partial(_xyz_set, field="manual")),
        xyz_grid.AxisOption(f"{XYZ_PREFIX} Delta", float, partial(_xyz_set, field="delta")),
        xyz_grid.AxisOption(f"{XYZ_PREFIX} Sigma divisor", float, partial(_xyz_set, field="divisor")),
        xyz_grid.AxisOption(f"{XYZ_PREFIX} Scale", float, partial(_xyz_set, field="scale")),
    ]
    if not any(str(a.label).startswith(XYZ_PREFIX) for a in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axes)


def _on_before_ui() -> None:
    try:
        _make_xyz_axes()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_on_before_ui)


# ---------------------------------------------------------------------------------------------
# Infotext paste
# ---------------------------------------------------------------------------------------------


def _paste(name: str, convert):
    def read(params):
        if KEY not in params:
            return None
        values = fh.parse_summary(params[KEY])
        if name not in values:
            return None
        try:
            return convert(values[name])
        except (TypeError, ValueError):
            return None

    return read


def _paste_bool(text: str) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes", "on")


def _paste_int(text: str) -> int:
    return int(float(text))


# ---------------------------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------------------------


class AnimaSpeed(scripts.Script):
    # Directly under Anima 3.8B (-35) in the Anima lane: it changes how the whole run samples.
    sorting_priority = -31

    @property
    def section(self):
        # Forge 는 사용자 섹션을 설정값 칼럼(#txt2img_settings) 안에 만든다 → 1열 "ANIMA 튜닝" 자리.
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        d = fh.DEFAULTS
        with gr.Accordion("Anima SPEED (저해상도 선행 샘플링 · 실험)", open=False):
            gr.Markdown(
                "초반의 노이즈가 지배적인 스텝을 **DCT로 줄인 저해상도 latent**에서 돌린 뒤, 전환 σ에서 고주파를 σ 크기의 "
                "노이즈로 채워 원래 크기로 넓히고(κ로 보정) 고른 샘플러로 이어서 샘플링합니다(SPEED, arXiv 2605.18736). "
                "앞쪽 스텝의 토큰이 줄어 빨라지지만 **이미지가 달라집니다**. 기본 OFF · Anima 품질은 미검증입니다. "
                "flow 모델 전용이며 SDXL·인페인트·마스크·ControlNet·레퍼런스 latent(Anima·Kontext·Klein·Qwen-Edit·"
                "Krea 2)·Wan I2V·PiD·Forge Spectrum과 함께면 쉬고, 그 이유를 infotext `Anima SPEED status`에 남깁니다."
            )
            enabled = gr.Checkbox(
                label="Enable SPEED", value=False,
                elem_id="anima_speed_enable", elem_classes=["sam3-on"],
            )
            with gr.Row():
                mode = gr.Radio(
                    label="Mode", choices=list(schedule.MODES), value=d["mode"],
                    info="transition = 공식·aoleg (전환 스텝 σ만 바꿈) · respace = sorryhyun (남은 σ를 비율로 다시 배치, "
                         "짝수 격자 — 데스크톱 앱 ComfyUI 팩과 같은 방식)",
                    elem_id="anima_speed_mode",
                )
                threshold = gr.Radio(
                    label="Transition sigma", choices=list(schedule.THRESHOLDS), value=d["threshold"],
                    info="neo_shift = 프리셋 σ* ÷ divisor (Forge의 고정 shift용) · delta_optimal = 프리셋 σ* 그대로 · "
                         "manual = 아래 σ 값",
                    elem_id="anima_speed_threshold",
                )
            with gr.Row():
                preset = gr.Dropdown(
                    label="Spectrum preset", choices=list(schedule.PRESET_NAMES), value=d["preset"],
                    info="잠재 공간 파워 스펙트럼(A·β). anima = Anima 32스텝 기준(스펙트럼 유도·미검증), flux·flux2 = 측정값, "
                         "custom = 아래 A·β",
                    elem_id="anima_speed_preset",
                )
                scales = gr.Textbox(
                    label="Scales", value=d["scales"],
                    info="1.0으로 끝나는 해상도 비율, 쉼표 구분 (예: 0.5,1.0 또는 0.25,0.5,1.0)",
                    elem_id="anima_speed_scales",
                )
            with gr.Row():
                delta = gr.Slider(
                    label="Delta (δ)", minimum=0.0001, maximum=0.5, step=0.0001, value=d["delta"],
                    info="노이즈 지배 허용치 — 작을수록 일찍 전환(품질↑ 속도↓)",
                    elem_id="anima_speed_delta",
                )
                divisor = gr.Slider(
                    label="Sigma divisor (neo_shift)", minimum=0.5, maximum=2.0, step=0.01,
                    value=d["sigma_divisor"],
                    info="σ*를 이 값으로 나눔. 1.0 = delta_optimal과 같음, 클수록 늦게 전환(빠름·위험)",
                    elem_id="anima_speed_divisor",
                )
            with gr.Row():
                manual = gr.Textbox(
                    label="Manual sigma(s)", value=d["manual_sigmas"],
                    info="전환마다 하나, transition 모드는 내림차순 (manual일 때만). 앱 기본값은 0.7",
                    elem_id="anima_speed_manual",
                )
                adaptive = gr.Checkbox(
                    label="Adaptive delta", value=d["adaptive"],
                    info="1024 px 기준 σ*·스텝 비율에 고정 (해상도가 커져도 저해상도 구간이 늘지 않게)",
                    elem_id="anima_speed_adaptive",
                )
            hires = gr.Checkbox(
                label="Apply to Hires pass", value=d["hires"], visible=not is_img2img,
                elem_id="anima_speed_hires",
            )
            with gr.Accordion("SPEED Advanced (세부값)", open=False):
                transform = gr.Radio(
                    label="Transform", choices=list(schedule.TRANSFORMS), value=d["transform"],
                    info="전환 때 쓰는 스펙트럼 기저. dct(기본)·fft = 아무 비율, dwt = 단계마다 정확히 2배",
                    elem_id="anima_speed_transform",
                )
                with gr.Row():
                    spectrum_a = gr.Number(
                        label="Spectrum A (custom)", value=d["spectrum_A"], elem_id="anima_speed_spectrum_a",
                    )
                    spectrum_beta = gr.Number(
                        label="Spectrum beta (custom)", value=d["spectrum_beta"],
                        elem_id="anima_speed_spectrum_beta",
                    )
                seed = gr.Number(
                    label="Spectral noise seed (-1 = 이미지 시드)", value=d["seed"], precision=0,
                    elem_id="anima_speed_seed",
                )

        self.infotext_fields = [
            (enabled, lambda params: KEY in params),
            (mode, _paste("mode", str)),
            (threshold, _paste("threshold", str)),
            (preset, _paste("preset", str)),
            (scales, _paste("scales", str)),
            (delta, _paste("delta", float)),
            (divisor, _paste("divisor", float)),
            (manual, _paste("manual", str)),
            (adaptive, _paste("adaptive", _paste_bool)),
            (transform, _paste("transform", str)),
            (spectrum_a, _paste("A", float)),
            (spectrum_beta, _paste("beta", float)),
            (seed, _paste("seed", _paste_int)),
            (hires, _paste("hires", _paste_bool)),
        ]
        # Order = sam3ext.speed.forge_host.ARG_NAMES (append-only for API callers).
        return [
            enabled, mode, threshold, preset, scales, delta, divisor, manual, adaptive,
            transform, spectrum_a, spectrum_beta, seed, hires,
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        settings = fh.coerce_settings(args, getattr(p, fh.XYZ_ATTR, None))
        label = fh.pass_label(p)
        params = getattr(p, "extra_generation_params", None)
        if not isinstance(params, dict):
            params = {}
            try:
                p.extra_generation_params = params
            except Exception:
                pass
        if not settings.enabled:
            if label != "hires":
                for key in (KEY, STATUS_KEY, INFOTEXT_IMG2IMG_RESCALE):
                    params.pop(key, None)
            return
        fh.begin_pass(p, label)
        params[KEY] = settings.summary()
        if settings.error:
            fh.record_status(p, label, f"invalid settings - {settings.error}")
            _log(f"{label}: invalid settings, not applied - {settings.error}")
            return
        if label == "hires" and not settings.hires:
            fh.record_status(p, label, "not applied - Apply to Hires pass is off")
            return
        reason = fh.attach(
            p, settings, label,
            img2img_rescale=bool(_option(OPT_IMG2IMG_RESCALE, True)), log=_log,
        )
        if reason is not None:
            fh.record_status(p, label, f"skipped - {reason}")
            _log(f"{label}: skipped - {reason}")
            return
        fh.record_status(p, label, "pending - sampler wrapped, waiting for sampling")
