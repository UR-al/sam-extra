"""Install sam-extra's ANIMA 28/40/52-block LoRA compatibility adapter."""

from __future__ import annotations

import sys
import traceback

from modules import script_callbacks, scripts

from sam3ext import anima_lora_blocks as alb
from sam3ext import dora_infer_mode as dim
from sam3ext.anima_lora_blocks import (
    install_forge_lora_block_hook,
    uninstall_forge_lora_block_hook,
)


def _install_anima_lora_block_hook() -> None:
    try:
        if install_forge_lora_block_hook():
            print(
                "[sam-extra] ANIMA LoRA block compatibility installed "
                "(28/40/52 blocks)."
            )
    except Exception:
        print(
            "[-] sam-extra: failed to install ANIMA LoRA block compatibility:\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
        )


# Built-in extensions normally load first, so install immediately.  on_before_ui
# is a fallback for unusual extension orders and remains idempotent.
_install_anima_lora_block_hook()
script_callbacks.on_before_ui(_install_anima_lora_block_hook)

try:
    script_callbacks.on_script_unloaded(uninstall_forge_lora_block_hook)
except AttributeError:
    pass


# ── 부분(sparse) LoRA 순정 추측 변환 토글 ──

def _on_ui_settings() -> None:
    import gradio as gr
    from modules import shared

    shared.opts.add_option(
        alb.OPT_SPARSE_FORGE_GUESS,
        shared.OptionInfo(
            False,
            f"Anima {alb.SPARSE_GUESS_LABEL} (일부 블록만 담은 LoRA — 블록 대응이 틀릴 수 있음)",
            gr.Checkbox,
            section=("sam3_lora", "SAM Extra LoRA"),
            infotext=alb.INFOTEXT_SPARSE_GUESS_KEY,
        ).info(
            "끄면(기본) 일부 블록만 담은 Anima LoRA 는 순정 Forge 판정(가장 큰 블록 인덱스+1 이 들어가는 가장 작은 "
            "레이아웃: 28·40·52)이 현재 모델과 같을 때만 그대로 로드하고, 다르면 건너뛰며 알림을 띄웁니다. "
            "켜면 순정 Forge 처럼 그 판정 레이아웃에서 현재 모델로 변환해 로드합니다 — 이 확장의 블록 대응표와 "
            "끼워 넣은 블록 정책(약한 복사 포함)을 그대로 쓰고, 순정이 거부하는 하향(예: 52→40)도 변환합니다. "
            "부분 LoRA 는 원래 학습한 모델을 알 수 없어(앞 21블록만 있는 LoRA 는 2.9B·3.8B 에서 학습했을 수도 있음) "
            "블록 대응이 틀릴 수 있습니다. 추측 변환한 생성은 infotext 에 'Anima sparse LoRA' 로 남고, "
            "그 infotext 를 붙여 넣으면 이 설정을 켜는 덮어쓰기가 됩니다. 인덱스 52 이상이 있는 LoRA 는 켜도 건너뜁니다."
        ),
    )


try:
    script_callbacks.on_ui_settings(_on_ui_settings)
except AttributeError:
    pass


def _lora_model(p):
    # 블록 훅은 shared.sd_model(= networks.load_networks 의 current_sd)에 기록한다.
    model = alb._forge_sd_model()
    return model if model is not None else getattr(p, "sd_model", None)


def _record_infotext(p) -> None:
    guessed = alb.sparse_guess_record(_lora_model(p))
    if not guessed:
        return
    seen = getattr(p, "_sam3_sparse_lora_guessed", None)
    if not isinstance(seen, dict):
        seen = {}
    seen.update(guessed)
    try:
        p._sam3_sparse_lora_guessed = seen
    except Exception:
        pass
    params = getattr(p, "extra_generation_params", None)
    if isinstance(params, dict):
        detail = ", ".join(f"{name} {conversion}" for name, conversion in seen.items())
        params[alb.INFOTEXT_SPARSE_GUESS_KEY] = f"Forge guess ({detail})"


class AnimaSparseLoraGuess(scripts.Script):
    """UI 없음: 토글이 바뀐 뒤 옛 LoRA 합성을 버리고, 추측 변환한 생성을 infotext 에 남긴다."""

    def title(self):
        return "SAM Extra Anima sparse LoRA"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process(self, p, *args):
        # extra_networks.activate(LoRA 로드)보다 먼저 돈다. XYZ 칸들은 extra_generation_params 를 같이 쓴다.
        params = getattr(p, "extra_generation_params", None)
        if isinstance(params, dict):
            params.pop(alb.INFOTEXT_SPARSE_GUESS_KEY, None)
        try:
            p._sam3_sparse_lora_guessed = {}
        except Exception:
            pass
        model = _lora_model(p)
        # Forge 는 LoRA 목록 해시가 같으면 다시 합치지 않는다 — 설정을 바꾸거나 infotext 덮어쓰기(onchange 없음)로
        # 토글이 달라졌는데 그 토글이 영향을 준 부분 LoRA 가 합쳐져 있으면 캐시를 버린다.
        if alb.sparse_cache_stale(model, alb.sparse_forge_guess_enabled()):
            dim.invalidate_lora_cache(p, model)
            alb.forget_sparse_state(model)

    def process_batch(self, p, *args, **kwargs):
        _record_infotext(p)

    def postprocess_batch(self, p, *args, **kwargs):
        # hires 패스가 다른 LoRA 목록을 합쳤을 수 있다(processing.py hr_extra_network_data).
        _record_infotext(p)
