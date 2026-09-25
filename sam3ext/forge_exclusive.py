"""Forge 의 Generate 와 같은 규칙으로 확장 버튼 작업을 돌리는 공용 헬퍼.

Refine · Tile-Repair · PiD · 캐릭터 레퍼런스처럼 ``process_images`` 나 벤더 추론을 직접 부르는 버튼
핸들러는 Forge 의 ``wrap_gradio_gpu_call``(modules/call_queue.py:23-48) 을 거치지 않는다. 그 래퍼가
해 주던 것을 여기서 대신한다.

- ``queue_lock``: txt2img Generate 와 동시에 sd_model·state·VRAM 을 만지지 않는다
  (webui.py 는 ``demo.queue(default_concurrency_limit=32)`` 라 Gradio 는 막아 주지 않는다).
- ``state.begin/end``: ⏹ Stop 이 세운 ``interrupted``/``skipped`` 를 새 작업 시작 때 되돌린다
  (modules/shared_state.py:121-136). 이게 없으면 Stop 한 번 뒤의 모든 Refine 이 첫 검사에서 바로 끝난다.
- 끝난 뒤 ``wrap_gradio_call`` 의 finally(call_queue.py:51-63) 처럼 중단 플래그를 지운다.
- 원하면 Forge 의 메인 스레드(modules_forge/main_thread.py)에서 실행한다 — txt2img/img2img 와 같은
  스레드에서 모델을 옮기면 빠르고, 그 스레드가 곧 "지금 GPU 를 쓰는 곳" 이라는 Forge 의 전제가 지켜진다.

Stop 버튼은 ``stop_if_job()`` 으로 job 이름 접두어를 확인한 뒤에만 플래그를 세운다 — txt2img 가 잠금을
쥐고 있고 우리 클릭이 기다리는 중이면 그냥 세웠을 때 txt2img 가 대신 중단된다.
"""
from __future__ import annotations

import contextvars
import threading
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def run_exclusive(
    job: str, fn: Callable[[], T], *, on_main_thread: bool = False
) -> T:
    """``fn`` 을 Forge Generate 하나처럼 실행한다: 잠금 하나, 새 job, 끝나면 중단 플래그 정리.

    중첩 호출 금지: ``queue_lock`` 은 재진입이 안 되는 FIFOLock(modules/fifo_lock.py)이라 ``fn`` 안에서
    다시 부르면 잠금에서 영원히 기다린다(메인 스레드 우회까지 가지도 못한다).
    """

    from modules import shared
    from modules.call_queue import queue_lock

    with queue_lock:
        shared.state.begin(job=job)
        try:
            if on_main_thread:
                return _call_on_main_thread(fn)
            return fn()
        finally:
            shared.state.end()
            _clear_stop_flags(shared.state)


def _clear_stop_flags(state) -> None:
    """call_queue.wrap_gradio_call 의 finally 와 같다 — 다음 작업이 Stop 잔상을 보지 않게."""
    state.skipped = False
    state.interrupted = False
    state.stopping_generation = False


def _call_on_main_thread(fn: Callable[[], T]) -> T:
    """Forge 메인 스레드에서 ``fn`` 을 돌리고 결과·예외를 그대로 돌려준다.

    ``main_thread.run_and_wait_result`` 는 예외를 삼키고 ``None`` 을 돌려주므로(main_thread.py:27-43)
    직접 상자에 담아 옮긴다. Gradio ``Progress(track_tqdm=True)`` 는 ContextVar(``LocalContext``)로
    현재 이벤트를 찾으므로 호출한 쪽의 컨텍스트를 복사해 넘겨야 브라우저 진행 표시가 유지된다.
    """

    try:
        from modules_forge import main_thread
    except Exception:
        return fn()
    if threading.current_thread() is threading.main_thread():
        # Forge 는 launch_utils.start() 끝에서 Python 메인 스레드로 loop() 를 돈다 — 그 안에서 큐에
        # 넣으면 영원히 기다린다. 메인 스레드 작업(work) 자체도 이 스레드에서 돌므로 별도의 스레드 로컬
        # 재진입 표시는 필요 없다.
        return fn()

    context = contextvars.copy_context()
    box: dict = {}

    def work():
        try:
            box["result"] = fn()
        except Exception as exc:
            box["error"] = exc

    main_thread.run_and_wait_result(context.run, work)
    if "error" in box:
        raise box["error"]
    if "result" not in box:
        raise RuntimeError("Forge main thread returned without a result")
    return box["result"]


def stop_if_job(prefixes: Sequence[str]) -> bool:
    """지금 ``shared.state.job`` 이 ``prefixes`` 중 하나로 시작할 때만 중단 플래그를 세운다.

    돌려주는 값: 플래그를 세웠으면 True. 다른 작업(txt2img 등)이 잠금을 쥐고 있으면 건드리지 않는다.
    """

    from modules import shared

    job = str(shared.state.job or "")
    if not job.startswith(tuple(prefixes)):
        return False
    shared.state.interrupted = True
    shared.state.skipped = True
    return True


def is_interrupted() -> bool:
    """⏹ Stop 이 눌렸는지. Forge 밖(테스트·CLI)에서는 항상 False."""

    try:
        from modules import shared

        state = shared.state
    except Exception:
        return False
    return bool(getattr(state, "interrupted", False) or getattr(state, "skipped", False))
