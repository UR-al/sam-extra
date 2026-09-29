"""Anima VAE DeGrid (NAFNet) — Extras 탭(한 장·배치·폴더). 이미 만든 이미지에 생성 탭과 같은 계산을 적용한다.

Forge 의 Extras 는 ``ScriptPostprocessing.order`` 가 작은 것부터 돈다(Upscale 1000). 확대하면 격자 간격이 달라지므로
900 으로 Upscale 보다 앞에 둔다(Settings 의 postprocessing_operation_order 로 바꿀 수 있음). 결과 정보(``pp.info``)는
Extras 가 PNG 의 ``postprocessing`` 항목에 적는다.
"""
from __future__ import annotations

from modules import scripts_postprocessing

from sam3ext import ui_vae_degrid as uvd
from sam3ext import vae_degrid_models as vdm
from sam3ext import vae_degrid_runtime as vdr


class ScriptPostprocessingVaeDegrid(scripts_postprocessing.ScriptPostprocessing):
    name = uvd.TITLE
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
            reason = f"{type(exc).__name__}: {exc}"
            vdr.log(f"Extras: 실패 — 원본을 그대로 둡니다: {reason}")
            pp.info[uvd.KEY_ERROR] = " ".join(reason.split())
            try:
                vdr.shared_runtime().release()
            except Exception:
                pass
            return
        pp.image = outcome.image
        pp.info.update(uvd.infotext_items(entry.name, cfg.mode, cfg.strength, cfg.tile))
        vdr.log(f"Extras: {outcome.summary()}")
