"""Anima 3.8B (Qwen3.5 / Semantic Connector v2) — 편입된 forge-anima-3.8B.

Anima-3.8B v1.1 체크포인트는 Qwen3.5-4B 의미 특징을 매 디노이징 스텝에 주입하는
Semantic Connector v2 를 **파일 안에** 담고 있다. Forge 본체는 그 가중치를 모르는 키로
버리므로(``Anima Unexpected: anima_v2_connector…``) 확장이 없으면 어댑터·qwen35_4b 를
텍스트 인코더 목록에 넣어도 결과가 한 픽셀도 안 바뀐다. 이 스크립트가
``sam3ext.anima38`` 런타임을 붙여 그 경로를 되살린다.

동작(원본 https://github.com/GumGum10/forge-anima-3.8B, MIT 와 동일):
- v2 번들(safetensors metadata 로 판별)은 아코디언이 접혀 있어도 **자동으로** 켜진다.
- 구형 v1 (베이스 + 별도 ``Anima-3.8B-expanded_adapter.safetensors``)은 아코디언을 켜고
  어댑터·강도를 고른다.
- 긍정 프롬프트는 두 인코더(Qwen3 0.6B + Qwen3.5 4B)를 순차로 돌려 융합하고, 부정
  프롬프트는 기본적으로 네이티브 경로를 쓴다(스위치로 바꿀 수 있다).
- 생성이 끝나면 전부 원복한다 — 아코디언을 끄면 재시작 없이 순정 Anima 로 돌아간다.

API(``alwayson_scripts["Anima 3.8B (Qwen3.5 / v2)"]``)는 위치 인자 목록도, SAM3 처럼
키 이름을 적은 dict 하나도 받는다: ``{"args": [{"enabled": true, "negative": false}]}``.
``bypass: true`` 를 주면 v2 번들이라도 순정 Anima 로 생성한다(확장 ON/OFF 비교용).

qwen35_4b.safetensors 가 없거나 런타임이 못 뜨면 **생성을 죽이지 않고** 한 번 경고한 뒤
순정 Anima 로 진행한다 — v2 번들은 자동 활성이라, 여기서 raise 하면 3.8B 생성이 전부
막히기 때문이다.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

import gradio as gr

from modules import scripts

try:
    from modules.ui_components import InputAccordion
except Exception:  # pragma: no cover - 옛 Forge 는 InputAccordion 이 없다
    InputAccordion = None

ANIMA38_NAME = "Anima 3.8B (Qwen3.5 / v2)"
DEFAULT_ADAPTER = "Anima-3.8B-expanded_adapter.safetensors"
ARG_NAMES = ("enabled", "adapter", "strength", "negative", "negative_strength", "bypass")
ARG_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "adapter": DEFAULT_ADAPTER,
    "strength": 1.0,
    "negative": False,
    "negative_strength": 1.0,
    # v2 번들은 자동 활성이라 '끄는' 손잡이가 따로 필요하다 — A/B 비교와 순정 확인용.
    "bypass": False,
}


def _log(message: str) -> None:
    """콘솔 코드페이지(cp949 등)에서 print 가 UnicodeEncodeError 로 죽지 않게."""
    text = f"[Anima38] {message}"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _number(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return out


def coerce_args(args: tuple | list) -> dict[str, Any]:
    """스크립트 인자 → {enabled, adapter, strength, negative, negative_strength}.

    UI 는 위치 인자 다섯 개를 준다. API 는 같은 목록이거나, 첫 원소로 키 이름을 적은
    dict 하나를 줄 수 있다(SAM3 와 같은 관례). 모르는 키는 무시하고 빠진 키는 기본값.
    """
    values = dict(ARG_DEFAULTS)
    if args and isinstance(args[0], dict):
        raw = args[0]
    else:
        raw = {name: value for name, value in zip(ARG_NAMES, args)}
    if "enabled" in raw:
        values["enabled"] = _truthy(raw["enabled"])
    if raw.get("adapter"):
        values["adapter"] = str(raw["adapter"])
    if "strength" in raw:
        values["strength"] = max(0.0, min(2.0, _number(raw["strength"], 1.0)))
    if "negative" in raw:
        values["negative"] = _truthy(raw["negative"])
    if "negative_strength" in raw:
        values["negative_strength"] = max(0.0, min(2.0, _number(raw["negative_strength"], 1.0)))
    if "bypass" in raw:
        values["bypass"] = _truthy(raw["bypass"])
    return values


def _adapter_choices() -> list[str]:
    try:
        from sam3ext.anima38.files import adapters

        choices = list(adapters())
    except Exception:
        choices = []
    return choices or [DEFAULT_ADAPTER]


class Anima38Script(scripts.Script):
    sorting_priority = 260209301   # 원본과 같은 자리 — 다른 컨디셔닝 패치보다 먼저

    def __init__(self):
        super().__init__()
        self._runtime = None
        self._runtime_error: str | None = None
        self._warned_missing = False
        self._installed_for = None

    # ── 런타임은 게으르게 — 임포트가 무거워(torch·backend) 스크립트 로드를 늦추지 않게 ──
    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        if self._runtime_error is not None:
            return None
        try:
            from sam3ext.anima38.runtime import shared_runtime

            self._runtime = shared_runtime()
        except Exception as exc:  # pragma: no cover - Forge 버전 차이
            self._runtime_error = f"{type(exc).__name__}: {exc}"
            _log(f"runtime unavailable — {self._runtime_error}")
            traceback.print_exc(file=sys.stderr)
            return None
        return self._runtime

    def title(self):
        return ANIMA38_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        choices = _adapter_choices()
        accordion = InputAccordion if InputAccordion is not None else None
        if accordion is not None:
            context = accordion(False, label=ANIMA38_NAME)
        else:  # pragma: no cover
            context = gr.Accordion(ANIMA38_NAME, open=False)
        with context as enabled_component:
            if accordion is None:  # pragma: no cover
                enabled_component = gr.Checkbox(label="Enable", value=False)
            gr.Markdown(
                "v2 번들(Anima-3.8B v1.1 등)은 이 아코디언이 꺼져 있어도 자동으로 켜집니다. "
                "아래 어댑터·강도는 구형 v1 체크포인트에서만 쓰입니다."
            )
            adapter = gr.Dropdown(
                label="Adapter (v1 only)",
                choices=choices,
                value=choices[0],
                info="Legacy v1 only. Bundled v2 checkpoints activate automatically and ignore this selector.",
            )
            strength = gr.Slider(
                label="Adapter strength (v1 only)",
                minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                info="Bundled v2 is fixed at its trained strength 1.0.",
            )
            negative = gr.Checkbox(
                label="Use adapter on negative prompt",
                value=False,
                info="Off keeps negatives on the native Anima encoder.",
            )
            negative_strength = gr.Slider(
                label="Negative adapter strength",
                minimum=0.0, maximum=2.0, value=1.0, step=0.05, visible=False,
            )
            negative.change(
                fn=lambda value: gr.update(visible=bool(value)),
                inputs=[negative], outputs=[negative_strength],
                show_progress=False, queue=False,
            )
            bypass = gr.Checkbox(
                label="Bypass — v2 번들도 순정 Anima(0.6B) 로 생성",
                value=False,
                info="자동 활성을 끕니다. 확장 ON/OFF 비교나 순정 확인용.",
            )
        return [enabled_component, adapter, strength, negative, negative_strength, bypass]

    # ── 생성 ──
    def process_batch(self, p, *args_, **kwargs):
        if self._installed_for is not None:
            # 이전 생성이 샘플링 중 예외로 끝나면 postprocess 가 안 불려 패치가 남는다 —
            # bypass/비활성 경로로 들어와도 먼저 원복해 순정 상태에서 시작한다.
            self._safe_restore(p)
        values = coerce_args(args_)
        if values["bypass"]:
            _log("bypass — native Anima")
            return
        runtime = self._get_runtime()
        if runtime is None:
            return
        try:
            is_v2 = runtime.is_v2_bundle(p.sd_model)
        except Exception:
            is_v2 = False
        if not values["enabled"] and not is_v2:
            return
        try:
            runtime.install(
                p,
                values["adapter"],
                values["strength"],
                float(values["negative_strength"]) if values["negative"] else None,
            )
            self._installed_for = p
            mode = "v2 bundle" if is_v2 else f"v1 adapter {values['adapter']} @ {values['strength']}"
            _log(f"active — {mode}" + (" (negative too)" if values["negative"] else ""))
        except FileNotFoundError as exc:
            # qwen35_4b 가 없다 — 3.8B 생성 자체는 순정 경로로 계속 간다
            if not self._warned_missing:
                self._warned_missing = True
                _log(f"disabled — {exc}")
            self._safe_restore(p)
        except Exception as exc:
            _log(f"install failed — {type(exc).__name__}: {exc}; continuing with native Anima")
            traceback.print_exc(file=sys.stderr)
            self._safe_restore(p)

    def postprocess(self, p, processed, *args_):
        self._safe_restore(p)

    def _safe_restore(self, p) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            target = self._installed_for if self._installed_for is not None else p
            runtime.restore(target)
        except Exception as exc:  # pragma: no cover - 복구는 실패해도 생성은 끝났다
            _log(f"restore failed — {type(exc).__name__}: {exc}")
        self._installed_for = None
