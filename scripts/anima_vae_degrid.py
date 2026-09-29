"""Anima VAE DeGrid (NAFNet) — txt2img·img2img 결과 이미지마다 VAE 격자 제거를 맨 마지막에 한 번.

계산은 ``sam3ext/vae_degrid.py``, 모델 찾기·불러오기는 ``vae_degrid_models.py``, 장치·정밀도·Forge 메모리 관리는
``vae_degrid_runtime.py``, 인자·infotext·컨트롤은 ``ui_vae_degrid.py``. 여기는 Forge 훅과 설정 등록만 한다.
Extras 탭은 ``scripts/anima_vae_degrid_extras.py``.

왜 ``postprocess_image`` 가 아니라 ``postprocess_image_after_composite`` 인가 — Forge ``modules/processing.py``
(process_images_inner 의 이미지 루프)는 이미지마다 ① 모든 always-on 스크립트의 ``postprocess_image``(ADetailer,
SAM3 in-flight 인페인트, img2img-hires-fix …) → ② ``postprocess_maskoverlay`` → ③ img2img 색 보정 → ④ ``apply_overlay``
(인페인트 합성) → ⑤ ``postprocess_image_after_composite`` → ⑥ 저장·``infotext(i)`` 순으로 부른다. ① 안의 순서는 스크립트
로드 순서(확장 폴더 이름순 + metadata.ini)와 사용자 콜백 우선순위가 정해 '맨 끝' 을 보장할 수 없지만, ⑤ 는 코드 구조상
① 이 모두 끝난 뒤라 어떤 설치 순서에서도 마지막 후처리다. ⑤ 를 쓰는 다른 것은 Settings 에서 메인 탭에 켠 Extras
후처리(``ScriptPostprocessingForMainUI`` — Upscale 등)뿐인데, 그들은 always-on 목록 맨 앞이라 기본으로는 먼저 돈다.
확대한 뒤에는 격자 간격이 달라지므로 ``metadata.ini`` 의 콜백 순서(Before)로 Upscale 보다 앞에 둔다. ⑥ 이 ⑤ 뒤라 여기서
쓴 infotext 는 그 이미지의 저장 파일에 들어간다.

SAM3 인페인트·ADetailer 의 내부 패스(``_sam3_inner``·``_ad_inner``)에서는 돌지 않는다 — 합친 뒤 바깥 이미지에 한 번만.
"""
from __future__ import annotations

import gradio as gr

from modules import script_callbacks, scripts

from sam3ext import ui_vae_degrid as uvd
from sam3ext import vae_degrid as vd
from sam3ext import vae_degrid_models as vdm
from sam3ext import vae_degrid_runtime as vdr


def inner_pass(p) -> bool:
    """SAM3 인페인트·Refine(``_sam3_inner``)·ADetailer(``_ad_inner``)의 내부 img2img."""
    return bool(getattr(p, "_sam3_inner", False) or getattr(p, "_ad_inner", False))


def _params(p) -> dict:
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        params = {}
        p.extra_generation_params = params
    return params


class AnimaVaeDegrid(scripts.Script):
    # SAM3 묶음(-30 … -25) 바로 뒤. 섹션 정리가 켜져 있으면 2열 "디테일러"(SAM3·ADetailer 와 같은 후처리 묶음).
    sorting_priority = -24

    def title(self):
        return uvd.TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        controls = uvd.build_controls(self.elem_id)
        self.infotext_fields = uvd.infotext_fields(controls)
        return list(controls)

    def process(self, p, *args):
        p._sam3_degrid = None
        if inner_pass(p):
            return
        cfg = uvd.coerce_args(args)
        if not cfg.enabled:
            return
        entry = vdm.resolve(cfg.model, vdm.discover())
        if entry is None:
            wanted = cfg.model or "auto"
            vdr.log(
                f"모델을 찾지 못해 건너뜁니다: {wanted!r} — models/ESRGAN · models/DeGrid 에 NAFNet DeGrid 파일"
                "(예: qwenVAEDegridNafnet_v11.safetensors)을 넣으세요. 이미지는 DeGrid 없이 저장됩니다."
            )
            uvd.record_failure(_params(p), f"model not found: {wanted}")
            return
        p._sam3_degrid = (cfg, entry)
        vdr.log(
            f"켬 — {entry.name} · {vd.MODE_LABELS[cfg.mode]} · 강도 {cfg.strength:g} · "
            f"타일 {cfg.tile or '없음'} (모든 후처리 뒤, 저장 직전)"
        )

    def postprocess_image_after_composite(self, p, pp, *args):
        state = getattr(p, "_sam3_degrid", None)
        if not state or inner_pass(p):
            return
        cfg, entry = state
        try:
            outcome = vdr.shared_runtime().run(pp.image, entry, mode=cfg.mode, strength=cfg.strength, tile=cfg.tile)
        except Exception as exc:
            # Forge 는 이 훅의 예외를 콘솔에만 보고하고 이미지를 그대로 저장한다 — 여기서 받아 infotext 를 고친다.
            reason = f"{type(exc).__name__}: {exc}"
            vdr.log(f"실패 — 이 이미지는 DeGrid 없이 저장합니다: {reason}")
            uvd.record_failure(_params(p), reason)
            try:
                vdr.shared_runtime().release()
            except Exception:
                pass
            return
        pp.image = outcome.image
        uvd.record_success(_params(p), uvd.outcome_infotext(outcome))   # 실제로 쓴 타일(OOM 으로 줄였으면 그 값)·정밀도
        vdr.log(outcome.summary())


def _on_ui_settings() -> None:
    from modules import shared

    section = ("sam3_degrid", "SAM Extra VAE DeGrid")
    shared.opts.add_option(
        vdr.OPT_DEVICE,
        shared.OptionInfo(
            vdr.DEFAULT_DEVICE,
            "VAE DeGrid: 계산 장치",
            gr.Radio,
            {"choices": [("auto (Forge 가 쓰는 GPU)", vdr.DEVICE_AUTO), ("cpu", vdr.DEVICE_CPU)]},
            section=section,
        ).info(
            "cpu 는 VRAM 을 전혀 쓰지 않습니다(같은 GPU 로 학습 중일 때 등). NAFNet-small 이라 CPU 로도 1216×1856 한 장에 "
            "몇 초입니다(개발 PC 실측 약 6.5 초, 타일 512). 계산식은 장치와 관계없이 같습니다."
        ),
    )
    shared.opts.add_option(
        vdr.OPT_PRECISION,
        shared.OptionInfo(
            vdr.DEFAULT_PRECISION,
            "VAE DeGrid: GPU 계산 정밀도",
            gr.Radio,
            {"choices": [
                ("fp16 autocast (학습과 같음)", vdr.PRECISION_FP16),
                ("fp32 (ComfyUI 노드와 같음)", vdr.PRECISION_FP32),
            ]},
            section=section,
        ).info(
            "fp16 은 가중치를 fp32 로 두고 autocast 로 돕니다 — 모델이 fp16 AMP 로 학습됐습니다. 값이 넘친 타일은 그 타일만 "
            "fp32 로 다시 합니다. fp32 와의 차이는 CPU 실측 최대 0.27/255(평균 0.04/255). CPU 는 늘 fp32 입니다."
        ),
    )
    shared.opts.add_option(
        vdr.OPT_KEEP_LOADED,
        shared.OptionInfo(
            vdr.DEFAULT_KEEP_LOADED,
            "VAE DeGrid: 모델을 VRAM 에 남기기 (약 117 MB)",
            gr.Checkbox,
            section=section,
        ).info(
            "켜면 Forge 메모리 관리에 맡겨 다음 이미지에서 다시 올리지 않습니다(자리가 필요하면 Forge 가 내림). 끄면(기본) "
            "이미지마다 올렸다가 끝나면 내리고 VRAM 캐시를 비웁니다. 결과 이미지는 같습니다."
        ),
    )


script_callbacks.on_ui_settings(_on_ui_settings)
