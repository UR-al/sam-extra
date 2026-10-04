"""DoRA 추론 방식 — 합치는 공식(Forge/Comfy 순정 · Forge 공식 fp32 · LyCORIS)과 3.8B 끼워 넣은 블록 처리.

계산과 설치는 ``sam3ext/dora_infer_mode.py``, 블록 복제는 ``sam3ext/anima_lora_blocks.py``. 여기는 UI·infotext·XYZ
축·설치 연결만 한다.

API(``alwayson_scripts["DoRA Inference Mode"]``)는 위치 인자 ``[enabled, mode, inserted, weak_strength, weak_scope]``
도, 키 이름을 적은 dict 하나도 받는다: ``{"args": [{"enabled": true, "mode": "lycoris", "inserted": "weak",
"weak_strength": 0.12, "weak_scope": "attn"}]}``. mode 는 ``lycoris`` / ``forge_fp32`` / ``forge`` / ``no_magnitude``,
inserted 는 ``keep`` / ``additive`` / ``skip`` / ``weak`` 또는 UI 라벨, weak_scope 는 ``attn`` / ``attn_mlp`` / ``all``.
뒤의 인자는 빼도 된다(기본값). 꺼져 있으면(또는 이 스크립트 인자가 없으면) 순정이다.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

import gradio as gr

from modules import script_callbacks, scripts

from sam3ext import anima_lora_blocks as alb
from sam3ext import dora_infer_mode as dim

try:
    from modules.ui_components import InputAccordion
except Exception:  # pragma: no cover - 옛 Forge 는 InputAccordion 이 없다
    InputAccordion = None


DORA_INFER_NAME = "DoRA Inference Mode"

LABEL_LYCORIS = "LyCORIS (학습과 동일 · fp32)"
LABEL_FORGE_FP32 = "Forge/Comfy 공식 · fp32"
LABEL_FORGE = "Forge/Comfy (순정)"
LABEL_NO_MAGNITUDE = "DoRA 끔 (크기 보정 없이 ΔW만 · 일반 LoKr처럼)"
MODE_CHOICES = [LABEL_LYCORIS, LABEL_FORGE_FP32, LABEL_FORGE, LABEL_NO_MAGNITUDE]
_LABEL_BY_MODE = {
    dim.MODE_LYCORIS: LABEL_LYCORIS,
    dim.MODE_FORGE_FP32: LABEL_FORGE_FP32,
    dim.MODE_FORGE: LABEL_FORGE,
    dim.MODE_NO_MAGNITUDE: LABEL_NO_MAGNITUDE,
}

LABEL_DUP_KEEP = "그대로 복제 (순정 · Forge 기본)"
LABEL_DUP_ADDITIVE = "덧셈형 (끼워 넣은 블록만 DoRA 크기 보정 끔)"
LABEL_DUP_SKIP = "넣지 않음 (원래 블록에만 · 모든 LoRA)"
LABEL_DUP_WEAK = "약한 복사 (브리지식 · 강도·범위 조절)"
DUP_CHOICES = [LABEL_DUP_KEEP, LABEL_DUP_ADDITIVE, LABEL_DUP_SKIP, LABEL_DUP_WEAK]
_LABEL_BY_DUP = {
    alb.DUPLICATE_ADDITIVE: LABEL_DUP_ADDITIVE,
    alb.DUPLICATE_KEEP: LABEL_DUP_KEEP,
    alb.DUPLICATE_SKIP: LABEL_DUP_SKIP,
    alb.DUPLICATE_WEAK: LABEL_DUP_WEAK,
}

LABEL_SCOPE_ATTN = "어텐션만 (브리지 기본)"
LABEL_SCOPE_ATTN_MLP = "어텐션+MLP"
LABEL_SCOPE_ALL = "전체 (모듈레이션·노름 포함)"
SCOPE_CHOICES = [LABEL_SCOPE_ATTN, LABEL_SCOPE_ATTN_MLP, LABEL_SCOPE_ALL]
_LABEL_BY_SCOPE = {
    alb.WEAK_SCOPE_ATTN: LABEL_SCOPE_ATTN,
    alb.WEAK_SCOPE_ATTN_MLP: LABEL_SCOPE_ATTN_MLP,
    alb.WEAK_SCOPE_ALL: LABEL_SCOPE_ALL,
}

ARG_NAMES = ("enabled", "mode", "inserted", "weak_strength", "weak_scope")  # 뒤에만 덧붙인다(API 인자 위치 고정)

INFOTEXT_DUP_KEY = "DoRA inserted"  # 값: additive / skip / weak 0.12 attn (그대로 복제는 기록하지 않는다)

# XYZ 격자 범례는 Forge 기본 글꼴(Roboto)로 그려져 한글이 네모로 나온다 — 선택지는 영어로.
XYZ_MODE_LABEL = "[DoRA] Inference mode"
XYZ_MODE_CHOICES = ["Forge (stock)", "Forge fp32", "LyCORIS fp32", "DoRA off (no magnitude)"]
XYZ_DUP_LABEL = "[DoRA] Inserted blocks"
XYZ_DUP_CHOICES = ["keep", "additive", "skip", "weak"]
XYZ_WEAK_LABEL = "[DoRA] Weak copy strength"  # 이 축만 있으면(끼워 넣은 블록 축 없이) 약한 복사로 돈다
XYZ_SCOPE_LABEL = "[DoRA] Weak copy scope"
XYZ_SCOPE_CHOICES = list(alb.WEAK_SCOPES)
_XYZ_MODE_ATTR = "_sam3_dora_mode_xyz"
_XYZ_DUP_ATTR = "_sam3_dora_dup_xyz"
_XYZ_WEAK_ATTR = "_sam3_dora_weak_xyz"
_XYZ_SCOPE_ATTR = "_sam3_dora_scope_xyz"
# 이 축들의 값이 바뀌면 LoRA 를 다시 읽고 합친다(칸당 1~2 초). xyz_grid 는 cost 가 큰 축을 바깥 루프에 두므로
# Checkpoint(1.0) 와 VAE(0.7) 사이 값을 줘서 시드·프롬프트 같은 값싼 축보다 덜 자주 바뀌게 한다.
XYZ_AXIS_COST = 0.8


def _log(message: str) -> None:
    """생성 경로에서 부르므로 cp949 콘솔의 UnicodeEncodeError 로 생성을 죽이지 않는다."""
    text = f"[DoRA Inference] {message}"
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(encoding, "backslashreplace").decode(encoding, "replace"))
        except Exception:
            pass
    except Exception:
        pass


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def coerce_args(args: tuple | list) -> tuple[bool, str, str]:
    """스크립트 인자 → (enabled, mode, inserted). 모르거나 빠진 값은 UI 기본값(LyCORIS, 그대로 복제)."""
    if args and isinstance(args[0], dict):
        raw = args[0]
    else:
        raw = {name: value for name, value in zip(ARG_NAMES, args)}
    enabled = _truthy(raw.get("enabled", False))
    mode = dim.normalize_mode(raw.get("mode")) or dim.MODE_LYCORIS
    inserted = alb.normalize_duplicate_policy(raw.get("inserted")) or alb.DUPLICATE_KEEP
    return enabled, mode, inserted


def _clamp_strength(value: Any) -> float | None:
    try:
        return round(min(2.0, max(0.0, float(value))), 3)
    except (TypeError, ValueError):
        return None


def coerce_weak(args: tuple | list) -> tuple[float, str]:
    """스크립트 인자 → 약한 복사 (강도, 범위). 빠지거나 모르는 값은 브리지 기본값(0.12, 어텐션만)."""
    if args and isinstance(args[0], dict):
        raw = args[0]
    else:
        raw = {name: value for name, value in zip(ARG_NAMES, args)}
    strength = _clamp_strength(raw.get("weak_strength"))
    scope = alb.normalize_weak_scope(raw.get("weak_scope"))
    return (
        alb.DEFAULT_WEAK_STRENGTH if strength is None else strength,
        scope or alb.WEAK_SCOPE_ATTN,
    )


def resolve_state(p, args: tuple | list) -> tuple[str, str]:
    """이번 생성의 (방식, 끼워 넣은 블록 정책). XYZ 값이 있으면 그 항목만 아코디언보다 먼저다."""
    enabled, mode, inserted = coerce_args(args)
    if not enabled:
        mode, inserted = dim.MODE_FORGE, alb.DUPLICATE_KEEP
    mode = dim.normalize_mode(getattr(p, _XYZ_MODE_ATTR, None)) or mode
    xyz_dup = alb.normalize_duplicate_policy(getattr(p, _XYZ_DUP_ATTR, None))
    if xyz_dup is None and (getattr(p, _XYZ_WEAK_ATTR, None) is not None or getattr(p, _XYZ_SCOPE_ATTR, None) is not None):
        xyz_dup = alb.DUPLICATE_WEAK  # 약한 복사 강도·범위 축만 걸면 그 축을 비교하려는 것이다
    return mode, xyz_dup or inserted


def resolve_weak(p, args: tuple | list) -> tuple[float, str]:
    """이번 생성의 약한 복사 (강도, 범위). XYZ 값이 아코디언보다 먼저다(정책이 약한 복사일 때만 쓰인다)."""
    strength, scope = coerce_weak(args)
    xyz_strength = _clamp_strength(getattr(p, _XYZ_WEAK_ATTR, None))
    xyz_scope = alb.normalize_weak_scope(getattr(p, _XYZ_SCOPE_ATTR, None))
    return (strength if xyz_strength is None else xyz_strength), (xyz_scope or scope)


def _paste_enabled(params: dict) -> bool:
    # 켜져 있을 때만 기록하므로, 두 키가 모두 없는 이미지는 순정으로 만든 것이다(붙여 넣으면 꺼진다).
    return dim.INFOTEXT_KEY in params or INFOTEXT_DUP_KEY in params


def _paste_mode(params: dict) -> str | None:
    mode = dim.normalize_mode(params.get(dim.INFOTEXT_KEY))
    if mode is not None:
        return _LABEL_BY_MODE[mode]
    return LABEL_FORGE if INFOTEXT_DUP_KEY in params else None


def _paste_dup(params: dict) -> str | None:
    if not _paste_enabled(params):
        return None
    policy = alb.normalize_duplicate_policy(params.get(INFOTEXT_DUP_KEY)) or alb.DUPLICATE_KEEP
    return _LABEL_BY_DUP[policy]


def _pasted_weak(params: dict) -> tuple[float | None, str | None]:
    value = params.get(INFOTEXT_DUP_KEY)
    if alb.normalize_duplicate_policy(value) != alb.DUPLICATE_WEAK:
        return None, None
    return alb.parse_weak_key(value)


def _paste_weak_strength(params: dict) -> float | None:
    return _clamp_strength(_pasted_weak(params)[0]) if _pasted_weak(params)[0] is not None else None


def _paste_weak_scope(params: dict) -> str | None:
    scope = _pasted_weak(params)[1]
    return _LABEL_BY_SCOPE[scope] if scope else None


# ── 설치 ──

def _install_hook() -> None:
    try:
        if dim.install():
            print(f"[sam-extra] DoRA inference mode hook installed ({', '.join(dim.installed_adapters())}).")
    except Exception:
        print(
            "[-] sam-extra: failed to install DoRA inference mode hook:\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
        )


def _on_unloaded() -> None:
    try:
        alb.set_duplicate_policy(alb.DUPLICATE_KEEP)
        alb.set_weak_copy(alb.DEFAULT_WEAK_STRENGTH, alb.WEAK_SCOPE_ATTN)
    except Exception:
        pass
    dim.uninstall()


def _current_state() -> tuple[str, str]:
    # 정책 자리는 policy_state_key — 약한 복사는 강도·범위까지 담아, 강도만 바꿔도 다시 합친다.
    return dim.current_mode(), alb.policy_state_key()


def _on_model_loaded(sd_model) -> None:
    # 체크포인트가 process() 뒤에 바뀌어도(✨ 루프 안 재로드, hires 체크포인트, 🎯·Refine·IPA) 그 모델의 LoRA 는
    # 지금 전역 상태로 합쳐진다 — 그 상태를 표시해 두어야 다음 생성이 낡은 가중치를 알아본다.
    dim.tag_model(sd_model, _current_state())


def _record_state(p, mode: str, inserted: str) -> None:
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        return
    # XYZ 칸들은 이 dict 하나를 같이 쓴다(xyz_grid.py:769) — 순정 칸에서 앞 칸의 키를 지운다.
    if mode == dim.MODE_FORGE:
        params.pop(dim.INFOTEXT_KEY, None)
    else:
        params[dim.INFOTEXT_KEY] = dim.INFOTEXT_VALUES[mode]
    if inserted == alb.DUPLICATE_KEEP:
        params.pop(INFOTEXT_DUP_KEY, None)
    else:
        params[INFOTEXT_DUP_KEY] = inserted


def _shared_sd_model():
    try:
        from modules import shared

        return getattr(shared, "sd_model", None)
    except Exception:  # pragma: no cover - Forge 밖
        return None


# weight_adapter 는 Forge 본체 패키지라 확장 순서와 상관없이 import 된다. on_before_ui 는 멱등 예비.
_install_hook()
script_callbacks.on_before_ui(_install_hook)

try:
    script_callbacks.on_script_unloaded(_on_unloaded)
except AttributeError:  # pragma: no cover - 옛 Forge
    pass

try:
    script_callbacks.on_model_loaded(_on_model_loaded)
except AttributeError:  # pragma: no cover - 옛 Forge
    pass


# ── XYZ ──

def _xyz_set_mode(p, x, xs) -> None:
    setattr(p, _XYZ_MODE_ATTR, x)


def _xyz_set_dup(p, x, xs) -> None:
    setattr(p, _XYZ_DUP_ATTR, x)


def _xyz_set_weak(p, x, xs) -> None:
    setattr(p, _XYZ_WEAK_ATTR, x)


def _xyz_set_scope(p, x, xs) -> None:
    setattr(p, _XYZ_SCOPE_ATTR, x)


def _make_xyz_axis() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return
    existing = {str(a.label) for a in xyz_grid.axis_options}
    for label, setter, choices in (
        (XYZ_MODE_LABEL, _xyz_set_mode, XYZ_MODE_CHOICES),
        (XYZ_DUP_LABEL, _xyz_set_dup, XYZ_DUP_CHOICES),
        (XYZ_SCOPE_LABEL, _xyz_set_scope, XYZ_SCOPE_CHOICES),
    ):
        if label not in existing:
            xyz_grid.axis_options.append(
                xyz_grid.AxisOption(label, str, setter, cost=XYZ_AXIS_COST, choices=lambda c=choices: list(c))
            )
    if XYZ_WEAK_LABEL not in existing:
        xyz_grid.axis_options.append(xyz_grid.AxisOption(XYZ_WEAK_LABEL, float, _xyz_set_weak, cost=XYZ_AXIS_COST))


def _on_before_ui_xyz() -> None:
    try:
        _make_xyz_axis()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_on_before_ui_xyz)


class DoraInferenceMode(scripts.Script):
    def title(self):
        return DORA_INFER_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        label = "DoRA 추론 방식 (Forge / LyCORIS · 끼워 넣은 블록)"
        if InputAccordion is not None:
            context = InputAccordion(False, label=label, elem_id=self.elem_id("dora_infer_mode"))
        else:  # pragma: no cover
            context = gr.Accordion(label, open=False)
        with context as enabled:
            if InputAccordion is None:  # pragma: no cover
                enabled = gr.Checkbox(label="Enable", value=False, elem_classes=["sam3-on"])
            gr.Markdown(
                "`dora_scale` 이 든 LoRA·LoKr·LoHa 를 어떻게 합칠지 고릅니다. 꺼 두면 Forge/ComfyUI 순정입니다.\n\n"
                "**계산 방식**\n"
                "- **LyCORIS** — 학습 때와 같은 공식(합친 가중치의 노름으로 나눔)을 fp32 로 계산합니다. "
                "강도 1.0 에서 학습 샘플을 만들던 가중치와 같습니다.\n"
                "- **Forge/Comfy 공식 · fp32** — 순정 공식은 두고 계산만 fp32 로 합니다(순정은 대부분의 GPU 에서 fp16). "
                "출력 축 DoRA 에서는 순정보다 오히려 학습에서 멀어지는 레이어도 있어, 입력 축 파일에 알맞습니다.\n"
                "- **DoRA 끔** — 모든 블록에서 크기 보정을 빼고 `W₀+ΔW` 만 더합니다(일반 LoKr 처럼). 학습한 크기 조정분이 "
                "빠져 학습 결과와 달라지는 실험용입니다. 원래 블록에서는 차이가 1\\~2% 정도로 작습니다.\n"
                "- 공식 차이는 출력 축 DoRA(`wd_on_output=True`, LyCORIS 기본값)에만 있습니다. 강도는 모두 Forge 처럼 "
                "선형으로 섞습니다(0 이면 LoRA 없음).\n\n"
                "**끼워 넣은 블록** — 작은 Anima LoRA 를 큰 모델에 얹을 때(2.9B→3.8B 등) 새로 끼워 넣은 블록에 앞 블록의 "
                "LoRA 를 복제합니다. 3.8B 의 끼워 넣은 블록은 출력 가중치가 원본보다 훨씬 작아서, DoRA 의 절대 크기를 "
                "그대로 복제하면 그 행들이 원본 크기로 부풀려집니다(학습량과 무관, 1에포크 파일도 같음).\n"
                "- **그대로 복제 (순정, 기본값)** — Forge 와 똑같이 복제합니다.\n"
                "- **덧셈형** — 끼워 넣은 블록의 복제본에서만 크기 보정(`dora_scale`)을 빼고 `W₀+ΔW` 로 더합니다. "
                "원래 블록은 그대로 DoRA 입니다.\n"
                "- **넣지 않음** — 끼워 넣은 블록에는 LoRA 를 넣지 않습니다(DoRA 가 아닌 LoRA 도 포함).\n"
                "- **약한 복사** — Civitai 'Anima 2B LoRA Bridge' 의 방식입니다. 복제본을 덧셈형으로 두고 ΔW 를 아래 "
                "강도만큼 줄여, 고른 범위의 모듈에만 넣습니다(브리지 권장: 0.08\\~0.18, 어텐션만). 강도 0 은 넣지 않음과, "
                "강도 1·전체는 덧셈형과 같습니다.\n"
                "- 렌더 비교(시드 3개)에서 덧셈형·넣지 않음이 순정보다 일관되게 낫지는 않았습니다. 특정 시드가 무너질 때 "
                "바꿔 보는 대안입니다.\n\n"
                "방식을 바꾸면 다음 생성에서 LoRA 를 한 번 다시 합칩니다. 키가 없는 이미지를 PNG Info 로 붙여 넣으면 "
                "순정으로 돌아갑니다."
            )
            mode = gr.Radio(
                label="계산 방식",
                choices=MODE_CHOICES,
                value=LABEL_LYCORIS,
                elem_id=self.elem_id("dora_infer_mode_choice"),
            )
            inserted = gr.Radio(
                label="끼워 넣은 블록 (상향 블록 변환 시)",
                choices=DUP_CHOICES,
                value=LABEL_DUP_KEEP,
                elem_id=self.elem_id("dora_infer_inserted"),
            )
            with gr.Row():
                weak_strength = gr.Slider(
                    label="약한 복사 강도 (약한 복사일 때만)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=alb.DEFAULT_WEAK_STRENGTH,
                    elem_id=self.elem_id("dora_infer_weak_strength"),
                )
                weak_scope = gr.Radio(
                    label="약한 복사 범위",
                    choices=SCOPE_CHOICES,
                    value=LABEL_SCOPE_ATTN,
                    elem_id=self.elem_id("dora_infer_weak_scope"),
                )
        self.infotext_fields = [
            (enabled, _paste_enabled),
            (mode, _paste_mode),
            (inserted, _paste_dup),
            (weak_strength, _paste_weak_strength),
            (weak_scope, _paste_weak_scope),
        ]
        return [enabled, mode, inserted, weak_strength, weak_scope]

    def process(self, p, *args):
        # extra_networks.activate(processing.py:970)와 가중치 합치기(setup_conds·sampling_prepare)보다 먼저 돈다.
        if getattr(p, "_sam3_inner", False):
            # SAM3 인페인트·Refine 의 내부 패스 — 바깥 생성(또는 마지막 생성)의 방식을 그대로 쓴다. 여기서 다시
            # 고르면 XYZ 값을 모르는 채 아코디언 값(또는 UI 기본값)으로 전역 방식을 뒤집는다. 쓰는 상태는 기록한다 —
            # 단독 Refine 은 빈 infotext 로 저장되므로(inpaint_core.py:473) 기록이 없으면 순정으로 붙여 넣어진다.
            state = _current_state()
            dim.sync_merged_state(p, _shared_sd_model(), state)
            _record_state(p, *state)
            return
        mode, inserted = resolve_state(p, args)
        weak_strength, weak_scope = resolve_weak(p, args)
        _install_hook()
        if mode != dim.MODE_FORGE and not dim.installed_adapters():
            _log("weight_adapter 훅이 없어 순정 방식으로 진행합니다(Forge 버전 차이).")
            mode = dim.MODE_FORGE
        if inserted != alb.DUPLICATE_KEEP and not alb.forge_hook_installed():
            _log("ANIMA LoRA 블록 변환 훅이 없어 끼워 넣은 블록은 Forge 기본대로 복제합니다.")
            inserted = alb.DUPLICATE_KEEP
        dim.set_mode(mode)
        alb.set_duplicate_policy(inserted)
        alb.set_weak_copy(weak_strength, weak_scope)
        dim.take_counters()
        state = _current_state()
        sd_model = _shared_sd_model()
        before = getattr(sd_model if sd_model is not None else getattr(p, "sd_model", None), dim.MERGED_STATE_ATTR, None)
        if dim.sync_merged_state(p, sd_model, state) and before is not None:
            detail = f" ({state[1]})" if inserted == alb.DUPLICATE_WEAK else ""
            _log(
                f"방식 변경 → {_LABEL_BY_MODE[mode]} / 끼워 넣은 블록: {_LABEL_BY_DUP[inserted]}{detail}"
                " — LoRA 를 다시 합칩니다."
            )
        _record_state(p, *state)

    def postprocess(self, p, processed, *args):
        if getattr(p, "_sam3_inner", False):
            return
        counters = dim.take_counters()
        mode = dim.current_mode()
        if counters.layers:
            extra = f" (합친 횟수 {counters.calls} — Low VRAM·온라인 LoRA 는 매 스텝 합칩니다)" if counters.calls > counters.layers else ""
            _log(f"DoRA 레이어 {counters.layers}개를 {_LABEL_BY_MODE[mode]} 방식으로 합쳤습니다{extra}.")
        if counters.fallbacks:
            _log(f"DoRA 레이어 {counters.fallbacks}번은 순정으로 합쳤습니다 — 첫 원인: {counters.fallback_reason}")
