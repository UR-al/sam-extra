"""'모델 내리기' 버튼 — Forge 의 모델과 sam-extra 가 쥔 모델을 모두 내리고 VRAM 을 비운다(순수 로직: Forge·torch 는 Host 로 받는다).

버튼은 notebook 레이아웃 탭 줄의 '기본값 저장' 바로 오른쪽에 있다(javascript/notebook_unload_models.js). 누르면
sam3ext/ui_unload_models.py 의 숨은 Gradio 버튼이 ``run(forge_host())`` 을 부른다:

1. **생성 중이면 거절** — "생성 중에는 내릴 수 없습니다". Forge 의 ``modules.call_queue.queue_lock`` 을 기다리지 않고
   (``acquire(blocking=False)``) 잡아 본다: txt2img/img2img Generate(``wrap_gradio_gpu_call``), API 생성(``Api.queue_lock``),
   sam-extra 의 Refine · Tile-Repair · 캐릭터 레퍼런스(``forge_exclusive.run_exclusive``)가 이 잠금을 쥐고 돈다(기다리는 중이어도
   잠금은 잡혀 있다). 잡았어도 ``shared.state`` 에 작업이 있으면(``job`` 이름이나 ``job_count`` ≠ 0 — 잠금 밖에서 begin 한 작업)
   거절한다. ``interrupted`` 만으로는 거절하지 않는다: 쉬는 동안 Interrupt 를 누르면 다음 ``begin()`` 까지 True 로 남는다
   (modules/shared_state.py) — 중단 중인 작업은 ``job`` 이 남아 있어 위에서 걸린다. 내리는 동안 잠금을 쥐고 있으므로 그사이 누른
   Generate 는 끝날 때까지 기다렸다가 시작한다(그리고 체크포인트를 다시 읽는다).
2. **Forge 의 모델** — ``modules.sd_models.unload_model_weights`` 를 그대로 부른다(Settings 의 'Unload All Models' 버튼과
   ``POST /sdapi/v1/unload-checkpoint`` 가 부르는 함수): ``memory_management.unload_all_models`` → ``sd_model`` 을
   ``FakeInitialModel`` 로, ``forge_hash`` 를 "" 로 → ``soft_empty_cache`` → ``gc.collect``. ``model_data.forge_loading_parameters``
   (고른 체크포인트 · VAE/TE · Low Bits — main_entry.refresh_model_loading_parameters 가 정함)는 건드리지 않으므로 다음 생성의
   ``forge_model_reload`` 가 해시가 다른 것을 보고 같은 설정으로 다시 읽는다(processing.manage_model_and_prompt_cache).
3. **sam-extra 가 쥔 것** — sam3ext/unload_registry.py 의 훅을 등록 순서대로. 하나가 던져도 나머지는 부르고, 던진 것은 결과 글에
   이름이 남는다.
4. ``gc.collect()`` 뒤 CUDA 가 있으면 ``torch.cuda.synchronize`` · ``empty_cache`` · ``ipc_collect``.
5. **결과 한 줄** — ``VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3) · 이 프로세스 예약 15.2 → 0.0 GB``. 앞은 GPU 전체의
   사용량(``torch.cuda.mem_get_info`` 의 전체 − 남은 것 — 같은 GPU 의 다른 프로그램 몫 포함), 뒤는 이 Forge 의 PyTorch 예약
   (``torch.cuda.memory_reserved``). 괄호는 실제로 무언가를 놓은 것만. CUDA 가 없으면 VRAM 은 재지 않았다고 쓴다.
"""
from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .save_defaults import clean_request_id
from .unload_registry import UnloadHook

BUSY_MESSAGE = "생성 중에는 내릴 수 없습니다"
RUNNING_MESSAGE = "이미 모델을 내리는 중입니다"
FORGE_KEY = "forge"
FORGE_LABEL = "Forge 모델"
COLLECT_KEY = "cuda_cache"
COLLECT_LABEL = "CUDA 캐시 비우기"
MEASURE_KEY = "vram"
MEASURE_LABEL = "VRAM 재기"
NOTHING_LABEL = "내릴 모델 없음"
RELOAD_NOTE = "다음 생성 때 고른 체크포인트(VAE / Text Encoder · Low Bits 그대로)를 다시 읽습니다."
GB = 1024 ** 3

_RUNNING = threading.Lock()   # 두 탭에서 동시에 눌러도 한 번씩


@dataclass(frozen=True)
class Memory:
    used: int        # GPU 전체 사용량(바이트) = mem_get_info 의 total − free
    total: int
    reserved: int    # 이 프로세스의 PyTorch 예약(memory_reserved)


@dataclass
class Host:
    """Forge 쪽 — 실제 값은 sam3ext/ui_unload_models.py forge_host(), 테스트는 가짜."""

    try_lock: Callable[[], bool]                   # queue_lock.acquire(blocking=False)
    unlock: Callable[[], None]                     # queue_lock.release
    busy: Callable[[], bool]                       # state_busy(shared.state)
    unload_forge: Callable[[], Any]                # unload_forge_model(sd_models, memory_management)
    hooks: Callable[[], Sequence[UnloadHook]]      # unload_registry.hooks
    collect: Callable[[], None]                    # collect_garbage(gc, torch)
    memory: Callable[[], "Memory | None"]          # measure_cuda(torch, device) — CUDA 가 없으면 None


@dataclass(frozen=True)
class Step:
    key: str
    label: str


@dataclass(frozen=True)
class Failure:
    key: str
    label: str
    error: str          # "TypeName: message"
    trace: str = ""     # 콘솔용


@dataclass
class Outcome:
    refused: str | None = None
    released: list[Step] = field(default_factory=list)   # 무언가를 놓은 것
    idle: list[Step] = field(default_factory=list)       # 쥔 것이 없던 것
    errors: list[Failure] = field(default_factory=list)
    before: Memory | None = None
    after: Memory | None = None
    measure_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.refused is None and not self.errors


# ---- Forge 사실 (모듈을 받아서 — 테스트는 가짜 모듈) ----------------------------------------------------------------

def state_busy(state: Any) -> bool:
    """shared.state 에 작업이 있는가 — begin() 이 job 이름과 job_count(-1)를, end() 와 wrap_gradio_call 이 ""·0 을 넣는다."""
    if state is None:
        return False
    job = getattr(state, "job", "")
    try:
        count = int(getattr(state, "job_count", 0) or 0)
    except (TypeError, ValueError):
        count = 0
    return bool(job) or count != 0


def unload_forge_model(sd_models: Any, memory_management: Any = None) -> bool:
    """Forge 의 unload_model_weights 를 한 번 부른다. 그 전에 모델이 있었으면(FakeInitialModel 이 아니거나 GPU 목록이 있으면) True."""
    fake = getattr(sd_models, "FakeInitialModel", None)
    current = getattr(getattr(sd_models, "model_data", None), "sd_model", None)
    held = current is not None and not (isinstance(fake, type) and isinstance(current, fake))
    if memory_management is not None:
        held = held or bool(getattr(memory_management, "current_loaded_models", None))
    sd_models.unload_model_weights()
    return held


def collect_garbage(gc_module: Any, torch_module: Any) -> None:
    gc_module.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()
        cuda.empty_cache()
        cuda.ipc_collect()


def measure_cuda(torch_module: Any, device: Any = None) -> Memory | None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    free, total = cuda.mem_get_info(device)
    return Memory(used=int(total) - int(free), total=int(total), reserved=int(cuda.memory_reserved(device)))


# ---- 실행 ------------------------------------------------------------------------------------------------------

def _describe(error: BaseException) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _failure(key: str, label: str, error: BaseException) -> Failure:
    return Failure(key, label, _describe(error), traceback.format_exc())


def _call(outcome: Outcome, key: str, label: str, fn: Callable[[], Any]) -> None:
    try:
        released = fn()
    except Exception as error:      # 하나가 실패해도 다음 것은 부른다
        outcome.errors.append(_failure(key, label, error))
        return
    (outcome.released if released else outcome.idle).append(Step(key, label))


def _measure(host: Host, outcome: Outcome) -> Memory | None:
    try:
        return host.memory()
    except Exception as error:
        outcome.measure_failed = True
        if not any(failure.key == MEASURE_KEY for failure in outcome.errors):
            outcome.errors.append(_failure(MEASURE_KEY, MEASURE_LABEL, error))
        return None


def _unload(host: Host) -> Outcome:
    outcome = Outcome()
    outcome.before = _measure(host, outcome)
    _call(outcome, FORGE_KEY, FORGE_LABEL, host.unload_forge)
    for hook in list(host.hooks()):
        _call(outcome, hook.key, hook.label, hook.release)
    try:
        host.collect()
    except Exception as error:
        outcome.errors.append(_failure(COLLECT_KEY, COLLECT_LABEL, error))
    outcome.after = _measure(host, outcome)
    return outcome


def run(host: Host) -> Outcome:
    """버튼 한 번. 생성 중이면 아무것도 하지 않고 refused 를 채운다."""
    if not _RUNNING.acquire(blocking=False):
        return Outcome(refused=RUNNING_MESSAGE)
    try:
        if not host.try_lock():
            return Outcome(refused=BUSY_MESSAGE)
        try:
            if host.busy():
                return Outcome(refused=BUSY_MESSAGE)
            return _unload(host)
        finally:
            host.unlock()
    finally:
        _RUNNING.release()


# ---- 결과 글 ---------------------------------------------------------------------------------------------------

def gb(value: int) -> str:
    return f"{value / GB:.1f} GB"


def _labels(steps: Sequence[Step | Failure]) -> list[str]:
    seen: list[str] = []
    for step in steps:
        if step.label not in seen:
            seen.append(step.label)
    return seen


def summary(outcome: Outcome) -> str:
    """버튼 옆 한 줄."""
    if outcome.refused is not None:
        return outcome.refused
    names = " · ".join(_labels(outcome.released)) or NOTHING_LABEL
    before, after = outcome.before, outcome.after
    if before is not None and after is not None:
        text = (f"VRAM 사용 {gb(before.used)} → {gb(after.used)} ({names}) · "
                f"이 프로세스 예약 {before.reserved / GB:.1f} → {gb(after.reserved)}")
    elif outcome.measure_failed:
        text = f"모델 내림 ({names}) · VRAM 을 재지 못했습니다"
    else:
        text = f"모델 내림 ({names}) · CUDA GPU 가 없어 VRAM 은 재지 않았습니다"
    if outcome.errors:
        text += " · 실패: " + ", ".join(_labels(outcome.errors))
    return text


def detail(outcome: Outcome) -> str:
    """결과 글의 툴팁 — 숫자의 뜻, 놓은 것 · 쥔 것이 없던 것 · 실패한 것과 그 이유."""
    if outcome.refused is not None:
        return outcome.refused + " — 생성이 끝난 뒤 다시 누르세요." if outcome.refused == BUSY_MESSAGE else outcome.refused
    lines = []
    before, after = outcome.before, outcome.after
    if before is not None and after is not None:
        lines.append(f"GPU 전체 사용: {gb(before.used)} → {gb(after.used)} ({gb(after.total)} 중 — 같은 GPU 의 다른 프로그램 몫 포함)")
        lines.append(f"이 Forge 프로세스의 PyTorch 예약: {gb(before.reserved)} → {gb(after.reserved)}")
    elif not outcome.measure_failed:
        lines.append("CUDA GPU 가 없어 VRAM 은 재지 않았습니다")
    lines.append("내림: " + (", ".join(_labels(outcome.released)) or "없음"))
    idle = [label for label in _labels(outcome.idle) if label not in _labels(outcome.released)]
    if idle:
        lines.append("쥔 것이 없던 것: " + ", ".join(idle))
    for failure in outcome.errors:
        lines.append(f"실패 — {failure.label}: {failure.error}")
    lines.append(RELOAD_NOTE)
    return "\n".join(lines)


def _memory_json(memory: Memory | None) -> dict | None:
    if memory is None:
        return None
    return {"used": memory.used, "total": memory.total, "reserved": memory.reserved}


def payload(request_id: Any, outcome: Outcome) -> dict:
    return {
        "ok": outcome.ok,
        "request": clean_request_id(request_id),
        "refused": outcome.refused is not None,
        "text": summary(outcome),
        "detail": detail(outcome),
        "released": [step.key for step in outcome.released],
        "idle": [step.key for step in outcome.idle],
        "errors": [{"key": f.key, "label": f.label, "error": f.error} for f in outcome.errors],
        "before": _memory_json(outcome.before),
        "after": _memory_json(outcome.after),
    }


def error_payload(request_id: Any, error: BaseException) -> dict:
    """run 밖에서 난 예외(Forge 를 못 읽음 등) — 페이지가 요청 id 를 기다리다 시간 초과로 끝나지 않게."""
    text = f"모델 내리기 실패: {_describe(error)}"
    return {"ok": False, "request": clean_request_id(request_id), "refused": False, "text": text, "detail": text,
            "released": [], "idle": [], "errors": [{"key": "bridge", "label": "모델 내리기", "error": _describe(error)}],
            "before": None, "after": None}


def to_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False)
