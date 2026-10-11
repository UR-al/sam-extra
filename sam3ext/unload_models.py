"""'모델 내리기' 버튼 — Forge 의 모델과 sam-extra 가 쥔 모델을 모두 내리고 VRAM 을 비운다(순수 로직: Forge·torch 는 Host 로 받는다).

버튼은 notebook 레이아웃 탭 줄의 '기본값 저장' 바로 오른쪽에 있다(javascript/notebook_unload_models.js). 누르면
sam3ext/ui_unload_models.py 의 숨은 Gradio 버튼이 ``run(forge_host())`` 을 부른다:

1. **생성 중이면 거절** — "생성 중에는 내릴 수 없습니다". Forge 의 ``modules.call_queue.queue_lock`` 을 기다리지 않고
   (``acquire(blocking=False)``) 잡아 본다: txt2img/img2img Generate · Extras · Checkpoint Merger(``wrap_gradio_gpu_call``),
   API 생성 · Extras API(``Api.queue_lock``), sam-extra 의 Refine · Tile-Repair · 캐릭터 레퍼런스(``forge_exclusive.run_exclusive``)
   가 이 잠금을 쥐고 돈다(기다리는 중이어도 잠금은 잡혀 있다). **잠금이 판단의 기준이다**: 이 체크아웃에서 ``shared.state.begin()``
   을 부르는 곳은 모두 잠금 안이다. 잠금을 잡았는데 ``shared.state`` 에 작업 기록(``job`` 이름이나 ``job_count`` ≠ 0)이 남아
   있으면, 끝 정리 없이 실패한 작업이 남긴 것이다 — 예: 예외로 끝난 Extras API 호출(modules/postprocessing.py 는 ``end()`` 를
   성공 경로에서만 부르고, API 경로엔 wrap_gradio_call 의 finally 가 없다). 그때는 그 작업 이름을 알리고
   ("Forge 에 끝나지 않은 작업 기록(extras)이 남아 있습니다 — 생성 중이 아니면 한 번 더 누르세요") **같은 기록에 다시 누르면**
   내린다(``job_timestamp`` 까지 같은 기록 — 그사이 새 작업이 begin 했으면 다시 묻는다). 잠금 밖에서 begin 하는 다른 확장의
   작업이 정말 도는 중이라면 첫 클릭이 그것을 지켜 준다. ``interrupted`` 만으로는 거절하지 않는다: 쉬는 동안 Interrupt 를 누르면
   다음 ``begin()`` 까지 True 로 남는다(modules/shared_state.py). 내리는 동안 잠금을 쥐고 있으므로 그사이 누른 Generate 는 끝날
   때까지 기다렸다가 시작한다(그리고 체크포인트를 다시 읽는다).
2. **Forge 의 모델** — ``modules.sd_models.unload_model_weights`` 를 그대로 부른다(Settings 의 'Unload All Models' 버튼과
   ``POST /sdapi/v1/unload-checkpoint`` 가 부르는 함수): ``memory_management.unload_all_models`` → ``sd_model`` 을
   ``FakeInitialModel`` 로, ``forge_hash`` 를 "" 로 → ``soft_empty_cache`` → ``gc.collect``. ``model_data.forge_loading_parameters``
   (고른 체크포인트 · VAE/TE · Low Bits — main_entry.refresh_model_loading_parameters 가 정함)는 건드리지 않으므로 다음 생성의
   ``forge_model_reload`` 가 해시가 다른 것을 보고 같은 설정으로 다시 읽는다(processing.manage_model_and_prompt_cache).
   예외 하나: Forge 의 ✨(갤러리 이미지 고해상도, ``p.txt2img_upscale``)는 그 다시 읽기를 건너뛴다 — scripts/layout_unload_models.py
   의 before_process 가 ``reload_for_upscale`` 로 대신 읽는다. 결과 글의 'Forge 모델' 은 체크포인트가 올라 있었을 때만이다(Forge
   의 메모리 목록에 sam-extra 의 패처 — TIPO · DeGrid … — 만 있었으면 그 이름은 각자의 훅이 적는다).
3. **Forge 가 메모리 목록 밖에 쥔 모델** — ``unload_all_models`` 는 ``current_loaded_models`` 만 내린다. 그 밖에서 GPU 에 남는 것:
   업스케일러(modules/esrgan_model.py ``UpscalerESRGAN.load_model`` 의 lru_cache(4) — 이 Forge 의 모델 업스케일러는 모두 이 길),
   TAESD · VAE Approx 디코더(sd_vae_taesd.py · sd_vae_approx.py 의 모듈 dict), 얼굴 복원(GFPGAN · CodeFormer 의 ``net`` 과
   ``face_helper`` — 'Move the model to CPU after restoration' 기본 끔). ``FORGE_CACHE_HOOKS`` 가 이것들을 버린다(다음에 쓸 때
   Forge 가 파일에서 다시 읽는다). import 되지 않은 모듈은 건너뛴다.
4. **sam-extra 가 쥔 것** — sam3ext/unload_registry.py 의 훅을 등록 순서대로. 하나가 던져도 나머지는 부르고, 던진 것은 결과 글에
   이름이 남는다.
5. ``gc.collect()`` 뒤 CUDA 가 있으면 ``torch.cuda.synchronize`` · ``empty_cache`` · ``ipc_collect``.
6. **결과 한 줄** — ``VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3) · 이 프로세스 예약 15.2 → 0.0 GB``. 앞은 GPU 전체의
   사용량(``torch.cuda.mem_get_info`` 의 전체 − 남은 것 — 같은 GPU 의 다른 프로그램 몫 포함), 뒤는 이 Forge 의 PyTorch 예약
   (``torch.cuda.memory_reserved``). 괄호는 실제로 무언가를 놓은 것만. CUDA 가 없으면 VRAM 은 재지 않았다고 쓴다.
"""
from __future__ import annotations

import functools
import json
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .save_defaults import clean_request_id
from .unload_registry import UnloadHook

BUSY_MESSAGE = "생성 중에는 내릴 수 없습니다"
RUNNING_MESSAGE = "이미 모델을 내리는 중입니다"
STALE_MESSAGE = "Forge 에 끝나지 않은 작업 기록({job})이 남아 있습니다 — 생성 중이 아니면 한 번 더 누르세요"
UNNAMED_JOB = "이름 없음"
FORGE_KEY = "forge"
FORGE_LABEL = "Forge 모델"
COLLECT_KEY = "cuda_cache"
COLLECT_LABEL = "CUDA 캐시 비우기"
MEASURE_KEY = "vram"
MEASURE_LABEL = "VRAM 재기"
NOTHING_LABEL = "내릴 모델 없음"
RELOAD_NOTE = ("다음 생성 때 고른 체크포인트(VAE / Text Encoder · Low Bits 그대로)를 다시 읽습니다 — "
               "갤러리 이미지의 ✨(고해상도)도 먼저 다시 읽습니다.")
GB = 1024 ** 3

_RUNNING = threading.Lock()   # 두 탭에서 동시에 눌러도 한 번씩
_STALE_SEEN: list = [None]    # 지난 클릭이 알린 '남은 작업 기록' — 같은 기록에 다시 누르면 내린다


@dataclass(frozen=True)
class Memory:
    used: int        # GPU 전체 사용량(바이트) = mem_get_info 의 total − free
    total: int
    reserved: int    # 이 프로세스의 PyTorch 예약(memory_reserved)


def _no_hooks() -> Sequence[UnloadHook]:
    return ()


@dataclass
class Host:
    """Forge 쪽 — 실제 값은 sam3ext/ui_unload_models.py forge_host(), 테스트는 가짜."""

    try_lock: Callable[[], bool]                   # queue_lock.acquire(blocking=False)
    unlock: Callable[[], None]                     # queue_lock.release
    busy: Callable[[], Any]                        # state_job(shared.state) — 작업 기록이 없으면 거짓 값
    unload_forge: Callable[[], Any]                # unload_forge_model(sd_models)
    hooks: Callable[[], Sequence[UnloadHook]]      # unload_registry.hooks
    collect: Callable[[], None]                    # collect_garbage(gc, torch)
    memory: Callable[[], "Memory | None"]          # measure_cuda(torch, device) — CUDA 가 없으면 None
    forge_caches: Callable[[], Sequence[UnloadHook]] = _no_hooks   # forge_cache_hooks(sys.modules.get)


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
    stale_job: str | None = None    # 거절: 잠금은 비었는데 남은 작업 기록(그 이름) — 같은 기록에 다시 누르면 내린다
    forced_job: str | None = None   # 그 기록에 두 번째로 눌러 내렸다(그 이름)

    @property
    def ok(self) -> bool:
        return self.refused is None and not self.errors


# ---- Forge 사실 (모듈을 받아서 — 테스트는 가짜 모듈) ----------------------------------------------------------------

def state_job(state: Any) -> tuple | None:
    """shared.state 에 남은 작업 기록 ``(job, job_count, job_timestamp)`` — 없으면 None.

    begin() 이 job 이름 · job_count(-1) · job_timestamp 를, end() 와 wrap_gradio_call 의 finally 가 "" · 0 을 넣는다."""
    if state is None:
        return None
    job = getattr(state, "job", "")
    try:
        count = int(getattr(state, "job_count", 0) or 0)
    except (TypeError, ValueError):
        count = 0
    if not job and count == 0:
        return None
    return (str(job or ""), count, str(getattr(state, "job_timestamp", "") or ""))


def state_busy(state: Any) -> bool:
    return state_job(state) is not None


def job_name(job: Any) -> str:
    """거절 글에 쓰는 작업 이름 — ``state_job`` 의 튜플(또는 가짜 host 의 글자/True)."""
    if isinstance(job, tuple) and job:
        name = str(job[0] or "")
    elif isinstance(job, str):
        name = job
    else:
        name = ""
    return name.strip() or UNNAMED_JOB


def checkpoint_loaded(sd_models: Any) -> bool:
    """Forge 에 체크포인트가 올라 있는가 — ``model_data.sd_model`` 이 None 도 FakeInitialModel 도 아니다."""
    fake = getattr(sd_models, "FakeInitialModel", None)
    current = getattr(getattr(sd_models, "model_data", None), "sd_model", None)
    return current is not None and not (isinstance(fake, type) and isinstance(current, fake))


def unload_forge_model(sd_models: Any) -> bool:
    """Forge 의 unload_model_weights 를 한 번 부른다(체크포인트가 없어도 — 메모리 목록도 비운다). 체크포인트가 있었으면 True.

    Forge 의 메모리 목록(current_loaded_models)에 체크포인트 없이 남은 것은 sam-extra 의 패처(TIPO · DeGrid · VAE 2x …)다 —
    그 이름은 각자의 훅이 적으므로 여기서 'Forge 모델' 로 세지 않는다."""
    held = checkpoint_loaded(sd_models)
    sd_models.unload_model_weights()
    return held


def reload_for_upscale(p: Any, sd_models: Any) -> bool:
    """Forge 의 ✨(갤러리 이미지 고해상도) 앞에서 — 내린 체크포인트를 읽는다. 읽었으면 True.

    ✨ 는 ``p.txt2img_upscale = True`` (modules/txt2img.py) 로 process_images 를 부르고, process_images 는 그때
    ``manage_model_and_prompt_cache`` 를 건너뛴다(modules/processing.py — '이미 올라 있을 모델을 다시 읽지 않으려고'). 버튼으로
    내린 뒤라 FakeInitialModel 이면 process_images_inner 의 첫 ``shared.sd_model.is_wan`` 에서 AttributeError 가 난다. 그래서
    그 건너뛰기보다 먼저 도는 ScriptRunner.before_process(scripts/layout_unload_models.py)에서, ✨ 이고 체크포인트가 없을 때만
    sam-extra 🎯 처럼 ``forge_model_reload()`` 를 부른다 — 고른 체크포인트 · VAE/TE · Low Bits(forge_loading_parameters) 그대로.
    체크포인트가 있으면(다른 것을 골라 두었어도) Forge 의 동작 그대로 둔다."""
    if not getattr(p, "txt2img_upscale", False) or checkpoint_loaded(sd_models):
        return False
    sd_models.forge_model_reload()
    return True


def clear_upscaler_cache(esrgan_model: Any) -> bool:
    """modules/esrgan_model.py — ``UpscalerESRGAN.load_model`` 의 ``lru_cache(maxsize=4)`` 가 GPU(device_esrgan)에 둔 채 쥔
    업스케일러 모델. 이 Forge 의 모델 업스케일러(hires fix · Extras)는 모두 이 길로 읽는다."""
    load_model = getattr(getattr(esrgan_model, "UpscalerESRGAN", None), "load_model", None)
    clear = getattr(load_model, "cache_clear", None)
    if not callable(clear):
        return False
    info = getattr(load_model, "cache_info", None)
    held = bool(info().currsize) if callable(info) else True
    clear()
    return held


def clear_model_cache(models: Any) -> bool:
    """``{이름: 모델}`` 모듈 dict — sd_vae_taesd.py ``sd_vae_taesd_models`` · sd_vae_approx.py ``sd_vae_approx_models``
    (devices.device 에 둔 채 쥔다). 비우면 다음 미리보기 · TAESD 디코드가 파일에서 다시 읽는다."""
    if not isinstance(models, dict) or not models:
        return False
    models.clear()
    return True


def release_face_restorers(restorers: Any) -> bool:
    """``shared.face_restorers`` 의 GFPGAN · CodeFormer(modules/face_restoration_utils.py CommonFaceRestoration) — 불러 둔 ``net``
    과 캐시된 ``face_helper``(얼굴 검출 · 파싱 모델). Settings 의 'Move the model to CPU after restoration'(기본 끔)이 꺼져
    있으면 둘 다 GPU 에 남는다. 다시 읽는 길이 있는 것만 버린다: ``net`` 은 ``load_net`` 이 있을 때(restore_with_helper 가 None
    이면 다시 읽음), ``face_helper`` 는 cached_property 일 때(다음 접근에서 다시 만듦)."""
    held = False
    for restorer in list(restorers or ()):
        if callable(getattr(restorer, "load_net", None)) and getattr(restorer, "net", None) is not None:
            restorer.net = None
            held = True
        if isinstance(getattr(type(restorer), "face_helper", None), functools.cached_property):
            if getattr(restorer, "__dict__", {}).pop("face_helper", None) is not None:
                held = True
    return held


# (key, 결과 글의 이름, Forge 모듈, 그 모듈을 받아 놓는 함수) — Forge 의 unload 바로 뒤, sam-extra 훅 앞에 이 순서대로.
FORGE_CACHE_HOOKS: tuple[tuple[str, str, str, Callable[[Any], Any]], ...] = (
    ("forge_upscalers", "업스케일러", "modules.esrgan_model", clear_upscaler_cache),
    ("forge_taesd", "TAESD", "modules.sd_vae_taesd",
     lambda module: clear_model_cache(getattr(module, "sd_vae_taesd_models", None))),
    ("forge_vae_approx", "VAE Approx", "modules.sd_vae_approx",
     lambda module: clear_model_cache(getattr(module, "sd_vae_approx_models", None))),
    ("forge_face_restorers", "얼굴 복원", "modules.shared",
     lambda module: release_face_restorers(getattr(module, "face_restorers", None))),
)


def forge_cache_hooks(find_module: Callable[[str], Any]) -> list[UnloadHook]:
    """``FORGE_CACHE_HOOKS`` 를 훅으로 — ``find_module`` 은 ``sys.modules.get``(내리려고 import 하지 않는다: 없으면 쥔 것도 없다)."""

    def bind(module_name: str, release: Callable[[Any], Any]) -> Callable[[], Any]:
        def run_hook() -> Any:
            module = find_module(module_name)
            return False if module is None else release(module)

        run_hook.__qualname__ = f"{module_name} cache"
        return run_hook

    return [UnloadHook(key, label, bind(module, release)) for key, label, module, release in FORGE_CACHE_HOOKS]


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


def _hooks(outcome: Outcome, source: Callable[[], Sequence[UnloadHook]], key: str, label: str) -> list[UnloadHook]:
    try:
        return list(source())
    except Exception as error:      # 목록을 못 읽어도 나머지 단계는 한다
        outcome.errors.append(_failure(key, label, error))
        return []


def _unload(host: Host) -> Outcome:
    outcome = Outcome()
    outcome.before = _measure(host, outcome)
    _call(outcome, FORGE_KEY, FORGE_LABEL, host.unload_forge)
    for hook in _hooks(outcome, host.forge_caches, "forge_caches", "Forge 캐시 목록"):
        _call(outcome, hook.key, hook.label, hook.release)
    for hook in _hooks(outcome, host.hooks, "hooks", "sam-extra 훅 목록"):
        _call(outcome, hook.key, hook.label, hook.release)
    try:
        host.collect()
    except Exception as error:
        outcome.errors.append(_failure(COLLECT_KEY, COLLECT_LABEL, error))
    outcome.after = _measure(host, outcome)
    return outcome


def run(host: Host) -> Outcome:
    """버튼 한 번. 생성 중이면 아무것도 하지 않고 refused 를 채운다.

    잠금은 비었는데 작업 기록이 남았으면 첫 클릭은 그 이름을 알리고(refused · stale_job), 같은 기록에 다시 누르면 내린다
    (forced_job)."""
    if not _RUNNING.acquire(blocking=False):
        return Outcome(refused=RUNNING_MESSAGE)
    try:
        if not host.try_lock():
            return Outcome(refused=BUSY_MESSAGE)
        try:
            job = host.busy()
            forced = None
            if job:
                if _STALE_SEEN[0] != job:
                    _STALE_SEEN[0] = job
                    name = job_name(job)
                    return Outcome(refused=STALE_MESSAGE.format(job=name), stale_job=name)
                forced = job_name(job)
            _STALE_SEEN[0] = None
            outcome = _unload(host)
            outcome.forced_job = forced
            return outcome
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
    if outcome.stale_job is not None:
        return (f"Forge 의 작업 잠금은 비어 있는데 shared.state 에 작업 기록({outcome.stale_job})이 남아 있습니다. "
                "끝 정리 없이 실패한 작업(예: 예외로 끝난 Extras API 호출)이 남긴 기록이면, 생성 중이 아닐 때 한 번 더 "
                "누르면 내립니다. 그사이 새 작업이 시작되면 다시 묻습니다.")
    if outcome.refused is not None:
        return outcome.refused + " — 생성이 끝난 뒤 다시 누르세요." if outcome.refused == BUSY_MESSAGE else outcome.refused
    lines = []
    if outcome.forced_job is not None:
        lines.append(f"잠금이 비어 있어 남은 작업 기록({outcome.forced_job})을 지난 것으로 보고 내렸습니다")
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
        "stale_job": outcome.stale_job,
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
    return {"ok": False, "request": clean_request_id(request_id), "refused": False, "stale_job": None,
            "text": text, "detail": text,
            "released": [], "idle": [], "errors": [{"key": "bridge", "label": "모델 내리기", "error": _describe(error)}],
            "before": None, "after": None}


def to_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False)
