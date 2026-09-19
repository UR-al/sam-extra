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

API(``alwayson_scripts["Anima 3.8B (Qwen3.5 / v2)"]``)는 위치 인자 여섯 개
``[enabled, adapter, strength, negative, negative_strength, bypass]`` 도, SAM3 처럼
키 이름을 적은 dict 하나도 받는다: ``{"args": [{"enabled": true, "negative": false}]}``.
``bypass: true`` 를 주면 v2 번들이라도 순정 Anima 로 생성한다(확장 ON/OFF 비교용).

qwen35_4b.safetensors 는 models/text_encoder 에서 자동으로 찾는다(VAE/Text Encoder 목록 불필요).
없거나 런타임이 못 뜨면 **생성을 죽이지 않고** 한 번 경고한 뒤 순정 Anima 로 진행한다 — v2 번들은
자동 활성이라, 여기서 raise 하면 3.8B 생성이 전부 막히기 때문이다. 어느 쪽으로 돌았는지는
infotext 의 ``Anima 3.8B`` 키(v2 bundle / v1 adapter / bypass / off: …)에 남는다.
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


def _install_loader_filter() -> None:
    """VAE/Text Encoder 목록의 Qwen3.5 파일을 Forge 로더가 모델마다 4.8 GB 씩 읽고 버리지 않게 한다.
    3.8B 는 런타임이 파일을 직접 찾으므로 1.0/2.9B/3.8B 를 같은 모듈 목록으로 XYZ 비교할 수 있다."""
    loader = sys.modules.get("backend.loader")   # Forge 는 스크립트보다 먼저 불러 둔다 (테스트엔 없다)
    if loader is None:
        return
    try:
        from sam3ext.anima38 import loader_filter

        loader_filter.install(loader)
    except Exception as exc:  # pragma: no cover - Forge 버전 차이
        _log(f"loader filter unavailable — {type(exc).__name__}: {exc}")


_install_loader_filter()


def _release_stale_runtime_caches(sd_model) -> None:
    """다른 체크포인트가 로드되면 공용 런타임이 이전 모델에 묶어 둔 커넥터·어댑터를 놓게 한다."""
    module = sys.modules.get("sam3ext.anima38.runtime")   # 아직 안 떴으면 놓을 것도 없다
    runtime = getattr(module, "_SHARED_RUNTIME", None)
    if runtime is None:
        return
    try:
        runtime.release_stale_caches(sd_model)
    except Exception as exc:  # pragma: no cover - 모델 로드를 막지 않는다
        _log(f"cache release failed — {type(exc).__name__}: {exc}")


def _register_model_loaded_hook() -> None:
    try:
        from modules import script_callbacks
    except Exception:  # pragma: no cover - Forge 밖(테스트)
        return
    script_callbacks.on_model_loaded(_release_stale_runtime_caches)


_register_model_loaded_hook()


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
    """스크립트 인자 → {enabled, adapter, strength, negative, negative_strength, bypass}.

    UI 는 위치 인자 여섯 개를 준다. API 는 같은 목록이거나, 첫 원소로 키 이름을 적은
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


# infotext 키 — Forge 파서(modules/infotext_utils.py re_param_code)는 키에 [\w\s\-/] 만 읽는다.
# 옛 키 "Anima 3.8B …" 는 '.' 때문에 "8B …" 로 잘려 읽히므로 붙여 넣기에서만 그 이름도 받는다.
STATUS_KEY = "Anima38"   # v2 bundle / v1 adapter / bypass / off: <이유>
_LEGACY_PREFIX = "8B"


def _record_status(p, status: str) -> None:
    params = getattr(p, "extra_generation_params", None)
    if isinstance(params, dict):
        params[STATUS_KEY] = status


def _v1_architecture() -> str:
    from sam3ext.anima38.files import ARCHITECTURE

    return ARCHITECTURE


def _param(params: dict, suffix: str = ""):
    """새 키(Anima38 …) 또는 옛 키가 Forge 파서에서 잘린 이름(8B …)."""
    for prefix in (STATUS_KEY, _LEGACY_PREFIX):
        key = f"{prefix} {suffix}".strip()
        if key in params:
            return params[key]
    return None


# ── PNG Info / Send to 로 붙여 넣기. None 이면 그 칸은 건드리지 않는다(3.8B 흔적이 없는 이미지). ──
def _paste_bypass(params: dict) -> bool | None:
    status = _param(params)
    return None if status is None else status == "bypass"


def _paste_negative(params: dict) -> bool | None:
    negative = _param(params, "negative")
    if negative is not None:
        return negative == "connector"
    if _param(params, "negative strength") is not None:
        return True
    return None


def _paste_negative_strength(params: dict) -> float | None:
    value = _param(params, "negative strength")
    return None if value is None else _number(value, 1.0)


def _paste_v1_enabled(params: dict) -> bool | None:
    architecture = _param(params, "architecture")
    return None if architecture is None else architecture == _v1_architecture()


def _paste_v1_adapter(params: dict) -> str | None:
    if _param(params, "architecture") != _v1_architecture():
        return None   # v2 어댑터 이름은 v1 드롭다운에 없다
    return _param(params, "adapter")


def _paste_v1_strength(params: dict) -> float | None:
    if _param(params, "architecture") != _v1_architecture():
        return None
    return _number(_param(params, "strength"), 1.0)


def _selected_checkpoint_path(sd_model, loading_parameters: dict) -> str | None:
    """다음 생성에 쓰일 체크포인트 — 드롭다운 선택(forge_loading_parameters)이 먼저다. Forge 는 첫 생성
    전까지 shared.sd_model 에 체크포인트 정보 없는 FakeInitialModel 을 둔다."""
    info = (loading_parameters or {}).get("checkpoint_info")
    path = getattr(info, "filename", None)
    if path:
        return str(path)
    info = getattr(sd_model, "sd_checkpoint_info", None)
    path = getattr(info, "filename", None) or getattr(sd_model, "filename", None)
    return str(path) if path else None


def _status_markdown(checkpoint_path: str | None, last_status: str | None) -> str:
    """아코디언의 상태 칸 — 인코더를 찾았는지, 선택한 체크포인트가 v2 번들인지, 이 탭의 마지막 생성."""
    from sam3ext.anima38 import files as anima_files

    found = anima_files.qwen35_models()
    if found:
        name = "qwen35_4b.safetensors" if "qwen35_4b.safetensors" in found else next(iter(found))
        encoder = f"`{name}` (models/text_encoder 에서 자동으로 찾음 — VAE/Text Encoder 목록에 넣을 필요 없음)"
    else:
        encoder = "없음 — `models/text_encoder` 에 `qwen35_4b.safetensors` 를 두세요 (없으면 순정 Anima 로 생성)"
    if not checkpoint_path:
        model = "아직 선택된 체크포인트가 없습니다"
    elif anima_files.bundle_metadata(checkpoint_path) is not None:
        model = "v2 번들 — 생성 때 자동으로 켜집니다 (끄려면 Bypass)"
    else:
        model = "3.8B 번들이 아닙니다 — 구형 v1 은 아코디언을 켜야 동작"
    return (
        f"- **Qwen3.5 인코더:** {encoder}\n"
        f"- **선택한 체크포인트:** {model}\n"
        f"- **이 탭의 마지막 생성:** {last_status or '아직 없음'}"
    )


def _adapter_choices() -> list[str]:
    try:
        from sam3ext.anima38.files import adapters

        choices = list(adapters())
    except Exception:
        choices = []
    return choices or [DEFAULT_ADAPTER]


class Anima38Script(scripts.Script):
    sorting_priority = 260209301   # 원본과 같은 값 — UI 순서만 정한다(실행 순서는 로드 순서, NegPiP 는 어느 순서든 동작)

    def __init__(self):
        super().__init__()
        self._runtime = None
        self._runtime_error: str | None = None
        self._warned_missing = False
        self._installed_for = None
        self._installed_values: dict[str, Any] | None = None   # 설치할 때의 인자 (batch count 판별용)
        self._last_status: str | None = None

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
            status = gr.Markdown("상태 확인을 누르면 인코더·모델·마지막 생성 상태를 보여줍니다.")
            status_button = gr.Button("상태 확인", size="sm")
            status_button.click(
                fn=self._status_for_ui,
                inputs=[], outputs=[status],
                show_progress=False, queue=False,
            )
        self.infotext_fields = [
            (bypass, _paste_bypass),
            (enabled_component, _paste_v1_enabled),
            (adapter, _paste_v1_adapter),
            (strength, _paste_v1_strength),
            (negative, _paste_negative),
            (negative_strength, _paste_negative_strength),
        ]
        return [enabled_component, adapter, strength, negative, negative_strength, bypass]

    def _status_for_ui(self) -> str:
        try:
            from modules import sd_models, shared

            loading = getattr(sd_models.model_data, "forge_loading_parameters", {}) or {}
            return _status_markdown(_selected_checkpoint_path(shared.sd_model, loading), self._last_status)
        except Exception as exc:  # pragma: no cover - UI 버튼은 죽지 않는다
            return f"상태를 읽지 못했습니다 — {type(exc).__name__}: {exc}"

    def _set_status(self, p, status: str) -> None:
        self._last_status = status
        _record_status(p, status)

    # ── 생성 ──
    def process_batch(self, p, *args_, **kwargs):
        if self._installed_for is not None and getattr(p, "_sam3_outer", None) is self._installed_for:
            # SAM3 가 생성 도중 돌리는 인페인트 패스 — 이 스크립트 인스턴스를 공유한다. 바깥 설치를 그대로
            # 물려받는다: 여기서 내렸다 다시 달면 그 뒤의 ADetailer 패스가 0.6B 조건으로 떨어진다.
            self._ensure_attached(p)
            p._anima38_inherited = True
            return
        values = coerce_args(args_)
        if (
            self._installed_for is p
            and self._installed_values == values
            and self._runtime is not None
            and self._runtime.install_is_current(p)
        ):
            # 같은 생성의 다음 batch count — 내렸다 다시 달면 조건 캐시·run 이 끊겨 Qwen3.5 로 다시 인코딩한다.
            # 모델이 새로 로드됐으면(Hires 체크포인트·Refiner) 이 길로 오지 않고 아래에서 다시 설치한다.
            self._ensure_attached(p)
            return
        self._release_stale_install(p)
        if self._installed_for is not None:
            # 이전 생성이 샘플링 중 예외로 끝나면 postprocess 가 안 불려 패치가 남는다 —
            # bypass/비활성 경로로 들어와도 먼저 원복해 순정 상태에서 시작한다.
            self._safe_restore(p)
        runtime = self._get_runtime()
        if runtime is None:
            return
        try:
            is_v2 = runtime.is_v2_bundle(p.sd_model)
        except Exception:
            is_v2 = False
        if values["bypass"]:
            _log("bypass — native Anima")
            if is_v2:
                self._set_status(p, "bypass")   # 붙여 넣어 다시 뽑아도 Bypass 로 돌게
            return
        if not values["enabled"] and not is_v2:
            self._last_status = "off: not a 3.8B checkpoint"   # 상태 칸용 — infotext 엔 남기지 않는다
            return
        try:
            runtime.install(
                p,
                values["adapter"],
                values["strength"],
                float(values["negative_strength"]) if values["negative"] else None,
            )
            self._installed_for = p
            self._installed_values = values
            self._set_status(p, "v2 bundle" if is_v2 else "v1 adapter")
            mode = "v2 bundle" if is_v2 else f"v1 adapter {values['adapter']} @ {values['strength']}"
            _log(f"active — {mode}" + (" (negative too)" if values["negative"] else ""))
        except FileNotFoundError as exc:
            # qwen35_4b 가 없다 — 3.8B 생성 자체는 순정 경로로 계속 간다
            if not self._warned_missing:
                self._warned_missing = True
                _log(f"disabled — {exc}")
            self._set_status(p, f"off: {exc}")
            self._safe_restore(p)
        except Exception as exc:
            _log(f"install failed — {type(exc).__name__}: {exc}; continuing with native Anima")
            traceback.print_exc(file=sys.stderr)
            self._set_status(p, f"off: install failed ({type(exc).__name__})")
            self._safe_restore(p)

    def process_before_every_sampling(self, p, *args_, **kwargs):
        """샘플링 직전(1차·하이레스·img2img, 그 패스의 LoRA 활성화 뒤) — 커넥터 사본의 llm_adapter LoRA 패치를
        지금 TE 에 맞춘다. 인코딩 때 맞춘 세트는 하이레스 조건 인코딩이나 ADetailer 가 바꿔 놓을 수 있다."""
        runtime = self._runtime
        if runtime is None:
            return
        if self._installed_for is not p and not getattr(p, "_anima38_inherited", False):
            return
        try:
            runtime._sync_adapter_lora(p.sd_model.forge_objects.clip)
        except Exception as exc:  # pragma: no cover - 동기화가 실패해도 샘플링은 진행한다
            _log(f"LoRA sync failed — {type(exc).__name__}: {exc}")

    def _ensure_attached(self, p) -> None:
        try:
            self._runtime.ensure_attached(p)
        except Exception as exc:  # pragma: no cover - 다시 걸기가 실패해도 생성은 진행한다
            _log(f"re-attach failed — {type(exc).__name__}: {exc}")

    def _release_stale_install(self, p) -> None:
        """다른 탭(스크립트 인스턴스)의 생성이 샘플링 중 예외로 끝나 postprocess 가 안 불리면 그 설치가
        켜진 채 남는다. 인스턴스는 탭마다 따로지만 런타임은 하나라, 여기서 런타임 기준으로 먼저 내린다 —
        안 그러면 이 탭의 Bypass·비활성 생성이 모르는 새 v2 로 돈다."""
        runtime = self._runtime
        if runtime is None:
            # 아직 런타임을 안 잡은 인스턴스 — 무거운 임포트 없이 이미 떠 있는 공용 런타임만 본다
            module = sys.modules.get("sam3ext.anima38.runtime")
            runtime = getattr(module, "_SHARED_RUNTIME", None)
        stale = getattr(runtime, "_installed_processing", None)
        if stale is None or stale is p:
            return
        try:
            runtime.restore(stale)
        except Exception as exc:  # pragma: no cover - 복구는 실패해도 이번 생성은 진행한다
            _log(f"stale restore failed — {type(exc).__name__}: {exc}")
        if self._installed_for is stale:
            self._installed_for = None   # 방금 내렸다 — 아래 _safe_restore 가 한 번 더 내리지 않게

    def postprocess(self, p, processed, *args_):
        if getattr(p, "_anima38_inherited", False):
            return   # 바깥 생성의 설치 — 바깥 postprocess 가 내린다
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
