"""Anima VAE DeGrid — 인자 해석·infotext 키·붙여 넣기·Gradio 컨트롤(생성 탭과 Extras 탭이 같이 쓴다).

API 인자(생성 탭 ``alwayson_scripts["Anima VAE DeGrid (NAFNet)"]``)는 위치 인자 ``[enabled, model, mode, strength, tile]``
또는 키 이름을 적은 dict 하나. 뒤의 인자는 빼도 된다(기본값). model 을 비우거나 ``"None"``/``"auto"`` 면 찾은 첫 NAFNet 파일,
mode 는 ``full``/``dark``/``bright`` 나 노드 팩 이름(``Dark Pixels Mainly``)·UI 라벨, strength 0~1.5, tile 0(나누지 않음)
또는 128~4096.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import vae_degrid as vd
from . import vae_degrid_models as vdm

TITLE = "Anima VAE DeGrid (NAFNet)"
# Extras 항목 이름(Settings → Postprocessing 목록·순서의 키). 생성 탭 스크립트와 달라야 한다 — Extras 항목을 메인 탭에도
# 켜면 Forge 가 ``ScriptPostprocessingForMainUI``(title() = 이 이름)로 감싸 always-on 으로 넣는데, 이름이 같으면
# API 의 ``alwayson_scripts[TITLE]`` 가 둘 중 어느 것인지 모호해진다.
EXTRAS_TITLE = "Anima VAE DeGrid (NAFNet, Extras)"
ARG_NAMES = ("enabled", "model", "mode", "strength", "tile")  # 뒤에만 덧붙인다(API 인자 위치 고정)

KEY_MODEL = "Anima DeGrid model"
KEY_MODE = "Anima DeGrid mode"
KEY_STRENGTH = "Anima DeGrid strength"
KEY_TILE = "Anima DeGrid tile"            # 실제로 쓴 타일(OOM 으로 줄였으면 줄인 값) — 붙여 넣으면 같은 결과
KEY_PRECISION = "Anima DeGrid precision"  # fp32(기본) / fp16-autocast — 설정이라 붙여 넣지 않는다(기록만)
KEY_ERROR = "Anima DeGrid error"
RESULT_KEYS = (KEY_MODEL, KEY_MODE, KEY_STRENGTH, KEY_TILE, KEY_PRECISION)

MODE_CHOICES = [
    "Full (전체)",
    "Dark Pixels Mainly (어두운 점 위주)",
    "Bright Pixels Mainly (밝은 점 위주)",
]
_MODE_CHOICE = dict(zip(vd.MODES, MODE_CHOICES))

# 타일 슬라이더: 0(나누지 않음)과 128 단위 — 모든 칸이 ``coerce_tile`` 에서 그대로이고(64 처럼 조용히 128 로 바뀌는 칸이
# 없음), 최대가 API·붙여 넣기 범위(MAX_TILE)와 같다. 칸 사이 값(API·OOM 으로 줄인 타일)은 숫자 칸에 그대로 들어간다.
TILE_SLIDER_STEP = vd.MIN_TILE
TILE_SLIDER_MAX = vd.MAX_TILE


@dataclass(frozen=True)
class DegridArgs:
    enabled: bool = False
    model: str = ""
    mode: str = vd.DEFAULT_MODE
    strength: float = vd.DEFAULT_STRENGTH
    tile: int = vd.DEFAULT_TILE


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def coerce_args(args) -> DegridArgs:
    """스크립트 인자(위치 또는 dict 하나) → DegridArgs. 모르거나 빠진 값은 기본값."""
    args = list(args or ())
    if args and isinstance(args[0], dict):
        raw = dict(args[0])
    else:
        raw = {name: value for name, value in zip(ARG_NAMES, args)}
    model = raw.get("model")
    model = "" if model is None or str(model).strip() in ("", vdm.NONE_NAME) else str(model).strip()
    return DegridArgs(
        enabled=_truthy(raw.get("enabled", False)),
        model=model,
        mode=vd.normalize_mode(raw.get("mode")) or vd.DEFAULT_MODE,
        strength=vd.coerce_strength(raw.get("strength", vd.DEFAULT_STRENGTH)),
        tile=vd.coerce_tile(raw.get("tile", vd.DEFAULT_TILE)),
    )


def mode_choice(mode: str) -> str:
    return _MODE_CHOICE.get(vd.normalize_mode(mode) or vd.DEFAULT_MODE, MODE_CHOICES[0])


def format_strength(strength: float) -> str:
    return f"{round(float(strength), 3):g}"


def infotext_items(model_name: str, mode: str, strength: float, tile: int, precision: str | None = None) -> dict:
    """infotext 항목. ``tile`` 은 **실제로 쓴** 타일(``DegridOutcome.tile_used``)을 넘긴다 — 타일 크기에 따라 결과가
    달라서(채널 어텐션이 타일 평균을 씀) 요청값을 적으면 OOM 으로 줄인 이미지를 붙여 넣어도 같은 결과가 나오지 않는다.
    ``precision`` 은 실행 정밀도(``fp32``·``fp16-autocast`` — 잔차 차이 GPU 최대 0.43/255, 8비트로 많아야 1 단계. 기록만)."""
    items = {
        KEY_MODEL: model_name,
        KEY_MODE: vd.MODE_LABELS[vd.normalize_mode(mode) or vd.DEFAULT_MODE],
        KEY_STRENGTH: format_strength(strength),
        KEY_TILE: int(tile),
    }
    if precision and str(precision).strip() not in ("", "-"):
        items[KEY_PRECISION] = str(precision).strip()
    return items


def outcome_infotext(outcome) -> dict:
    """``DegridOutcome`` → infotext 항목(실제로 쓴 타일·정밀도). 생성 탭과 Extras 가 같이 쓴다."""
    return infotext_items(outcome.model_name, outcome.mode, outcome.strength, outcome.tile_used, outcome.precision)


def record_success(params: dict, items: dict) -> None:
    """성공한 이미지의 infotext — 같은 배치 앞 장의 오류 기록은 걷는다(extra_generation_params 는 배치가 함께 쓴다)."""
    params.pop(KEY_ERROR, None)
    params.update(items)


def record_failure(params: dict, reason: str) -> None:
    """실패한 이미지의 infotext — 앞 장이 남긴 성공 키를 걷고 이유만 남긴다."""
    for key in RESULT_KEYS:
        params.pop(key, None)
    params[KEY_ERROR] = " ".join(str(reason).split())


# ── 붙여 넣기(PNG Info / Send to) ──
def paste_enabled(params: dict) -> bool:
    return KEY_MODEL in params


def make_paste_model(choices: Callable[[], list[str]]):
    def paste_model(params: dict):
        name = params.get(KEY_MODEL)
        if name is None:
            return None
        names = list(choices() or [])
        entry = vdm.resolve(name, [vdm.ModelEntry(n, n) for n in names if n != vdm.NONE_NAME])
        return entry.name if entry is not None and str(name).strip() else None

    return paste_model


def paste_mode(params: dict):
    mode = vd.normalize_mode(params.get(KEY_MODE))
    return None if mode is None else _MODE_CHOICE[mode]


def paste_strength(params: dict):
    if KEY_STRENGTH not in params:
        return None
    return vd.coerce_strength(params.get(KEY_STRENGTH))


def paste_tile(params: dict):
    if KEY_TILE not in params:
        return None
    return vd.coerce_tile(params.get(KEY_TILE))


# ── 모델 목록 ──
def model_choices() -> list[str]:
    names = [entry.name for entry in vdm.discover()]
    return names or [vdm.NONE_NAME]


def refreshed_choices() -> dict:
    return {"choices": model_choices()}


# ── Gradio ──
HELP_GENERATION = (
    "[DraconicDragon NAFNet VAE DeGrid](https://huggingface.co/DraconicDragon/NAFNet-VAE-DeGrid) 로 Anima(Qwen·Wan VAE) "
    "이미지의 **격자 무늬**를 지웁니다. 이미지마다 ADetailer·SAM3 인페인트 등 **모든 후처리가 끝난 뒤, 저장 직전에** "
    "한 번 적용합니다.\n\n"
    "- 모델: `models/ESRGAN` 또는 `models/DeGrid` 의 NAFNet 파일만 목록에 나옵니다(예: `qwenVAEDegridNafnet_v11`). "
    "이 모델은 이미지가 아니라 **잔차**를 내므로 Forge 의 일반 업스케일러로 고르면 안 됩니다. 거꾸로 이미지를 내는 일반 "
    "복원 NAFNet(노이즈 제거 등)도 목록에 보이지만 골라도 적용하지 않습니다.\n"
    "- **Full** 잔차 전체(어두운·밝은 격자 모두) · **Dark Pixels Mainly** 양의 잔차만(ComfyUI 기본 노드 경로와 같음) · "
    "**Bright Pixels Mainly** 음의 잔차만.\n"
    "- 강도: 잔차에 곱하는 배율(1 = 원본 노드). 타일: 512 = 원본 노드, 0 = 나누지 않음(VRAM 더 씀).\n"
    "- 장치·fp16/fp32·VRAM 에 남기기는 Settings → **SAM Extra VAE DeGrid**."
)
HELP_EXTRAS = (
    "이미 만든 이미지(한 장·배치·폴더)에 **VAE DeGrid** 를 적용합니다 — 생성 탭의 같은 기능과 계산이 같습니다. "
    "Upscale 보다 먼저 돕니다(확대하면 격자 간격이 달라짐). 결과 정보에 `Anima DeGrid …` 가 남습니다."
)


def _accordion(label: str, elem_id: str):
    try:
        from modules.ui_components import InputAccordion
    except Exception:  # pragma: no cover - 옛 Forge·테스트
        InputAccordion = None
    import gradio as gr

    if InputAccordion is not None:
        return InputAccordion(False, label=label, elem_id=elem_id), None
    accordion = gr.Accordion(label, open=False)
    return accordion, "checkbox"


def build_controls(elem_id: Callable[[str], str], *, extras: bool = False):
    """(enabled, model, mode, strength, tile) — 생성 탭은 ``Script.elem_id``, Extras 는 접두사 함수를 넘긴다."""
    import gradio as gr

    choices = model_choices()
    context, fallback = _accordion(EXTRAS_TITLE if extras else TITLE, elem_id("sam3_degrid"))
    with context as enabled:
        if fallback:
            enabled = gr.Checkbox(label="Enable", value=False, elem_classes=["sam3-on"],
                                  elem_id=elem_id("sam3_degrid_enable"))
        gr.Markdown(HELP_EXTRAS if extras else HELP_GENERATION)
        with gr.Row():
            model = gr.Dropdown(
                label="DeGrid 모델 (NAFNet)",
                choices=choices,
                value=choices[0],
                elem_id=elem_id("sam3_degrid_model"),
            )
            try:
                from modules.ui_common import create_refresh_button

                create_refresh_button(model, lambda: None, refreshed_choices, elem_id("sam3_degrid_refresh"))
            except Exception:  # pragma: no cover - Forge 밖
                pass
        mode = gr.Radio(
            label="적용 방식",
            choices=MODE_CHOICES,
            value=MODE_CHOICES[0],
            elem_id=elem_id("sam3_degrid_mode"),
        )
        with gr.Row():
            strength = gr.Slider(
                label="강도 (잔차 배율)",
                minimum=vd.STRENGTH_MIN,
                maximum=vd.STRENGTH_MAX,
                step=0.05,
                value=vd.DEFAULT_STRENGTH,
                elem_id=elem_id("sam3_degrid_strength"),
            )
            tile = gr.Slider(
                label=f"타일 크기 (0 = 나누지 않음, {vd.MIN_TILE}~{vd.MAX_TILE})",
                minimum=0,
                maximum=TILE_SLIDER_MAX,
                step=TILE_SLIDER_STEP,
                value=vd.DEFAULT_TILE,
                elem_id=elem_id("sam3_degrid_tile"),
            )
    return enabled, model, mode, strength, tile


def infotext_fields(controls) -> list:
    enabled, model, mode, strength, tile = controls
    return [
        (enabled, paste_enabled),
        (model, make_paste_model(model_choices)),
        (mode, paste_mode),
        (strength, paste_strength),
        (tile, paste_tile),
    ]
