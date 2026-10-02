"""Anima VAE DeGrid (NAFNet) — Extras 탭(한 장·배치·폴더). 이미 만든 이미지에 생성 탭과 같은 계산을 적용한다.

Forge 의 Extras 는 ``ScriptPostprocessing.order`` 가 작은 것부터 돈다(Upscale 1000). 확대하면 격자 간격이 달라지므로
900 으로 Upscale 보다 앞에 둔다(Settings 의 postprocessing_operation_order 로 바꿀 수 있음). 결과 정보(``pp.info``)는
Extras 가 PNG 의 ``postprocessing`` 항목에 적는다.

이름은 ``Anima VAE DeGrid (NAFNet, Extras)`` 로 생성 탭 스크립트(``Anima VAE DeGrid (NAFNet)``)와 다르다 — Settings 에서
이 항목을 메인 탭에도 켜면 Forge 가 ``ScriptPostprocessingForMainUI``(title() = name)로 감싸 always-on 으로 넣는데, 이름이
같으면 API ``alwayson_scripts`` 의 이름이 둘 중 어느 것인지 모호해진다.
"""
from __future__ import annotations

from modules import scripts_postprocessing

from sam3ext import ui_vae_degrid as uvd
from sam3ext import vae_degrid_models as vdm
from sam3ext import vae_degrid_runtime as vdr


class ScriptPostprocessingVaeDegrid(scripts_postprocessing.ScriptPostprocessing):
    name = uvd.EXTRAS_TITLE   # 생성 탭 스크립트와 다른 이름(메인 탭에 켰을 때 always-on 제목이 겹치지 않게)
    order = 900

    def ui(self):
        enabled, model, mode, strength, tile = uvd.build_controls(lambda item: f"extras_{item}", extras=True)
        return {
            "enabled": enabled,
            "model": model,
            "mode": mode,
            "strength": strength,
            "tile": tile,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enabled=False, model=None, mode=None,
                strength=None, tile=None, **_kwargs):
        cfg = uvd.coerce_args([{"enabled": enabled, "model": model, "mode": mode, "strength": strength, "tile": tile}])
        if not cfg.enabled:
            return
        entry = vdm.resolve(cfg.model, vdm.discover())
        if entry is None:
            wanted = cfg.model or "auto"
            vdr.log(f"Extras: 모델을 찾지 못해 건너뜁니다: {wanted!r} (models/ESRGAN · models/DeGrid)")
            pp.info[uvd.KEY_ERROR] = f"model not found: {wanted}"
            return
        try:
            outcome = vdr.shared_runtime().run(pp.image, entry, mode=cfg.mode, strength=cfg.strength, tile=cfg.tile)
        except Exception as exc:
            reason = vdr.failure_reason(exc)
            vdr.log(f"Extras: 실패 — 원본을 그대로 둡니다: {reason}")
            pp.info[uvd.KEY_ERROR] = " ".join(reason.split())
            try:
                vdr.shared_runtime().release()
            except Exception:
                pass
            return
        pp.image = outcome.image
        pp.info.update(uvd.outcome_infotext(outcome))
        vdr.log(f"Extras: {outcome.summary()}")
