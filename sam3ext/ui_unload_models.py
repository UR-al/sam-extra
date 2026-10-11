"""'모델 내리기' 버튼의 Gradio 다리 — 숨은 버튼 하나(요청 id 입력 → JSON 결과)를 demo 에 만든다.

보이는 버튼은 javascript/notebook_unload_models.js 가 notebook 레이아웃의 탭 줄에서 '기본값 저장' 바로 오른쪽에 만든다.
누르면 그 스크립트가 여기 숨은 버튼(#sam3_unload_models_run)을 누르고, 결과 JSON 이 숨은 Textbox(#sam3_unload_models_result)
에 돌아온다. 요청 id 는 이벤트의 ``js`` 가 첫 입력 칸에 넣고, 같은 id 의 결과만 페이지가 받아들인다(기본값 저장과 같은 방식 —
sam3ext/ui_save_defaults.py).

이벤트는 Gradio 큐로 돈다(``concurrency_limit=1`` — 두 탭에서 눌러도 하나씩). 큰 모델을 CPU 로 옮기는 데 몇 초가 걸려도 요청이
끊기지 않고, Forge 의 Generate 와 겹치지 않는 것은 sam3ext/unload_models.py 가 Forge 의 queue_lock 으로 지킨다.
연결 시점은 기본값 저장과 같다: footer 의 on_after_component — 모든 컴포넌트가 demo 에 등록된 뒤, loadsave.setup_ui() 뒤라
숨은 컴포넌트가 ui-config 에 들어가지 않는다. Reload UI 는 새 demo 를 만든다 — 만든 화면은 약한 참조로 기억한다.
"""
from __future__ import annotations

import sys
import weakref

from sam3ext import unload_models, unload_registry

ANCHOR_ELEM_ID = "footer"
BRIDGE_ELEM_ID = "sam3_unload_models_bridge"
REQUEST_ELEM_ID = "sam3_unload_models_request"
RUN_ELEM_ID = "sam3_unload_models_run"
RESULT_ELEM_ID = "sam3_unload_models_result"
# Gradio 4 는 입력값(+ 출력값)을 이 함수에 넘기고 돌려받은 배열을 백엔드에 보낸다 — 첫 칸에 요청 id.
ARGS_JS = "(...args) => (window.sam3UnloadModelsArgs ? window.sam3UnloadModelsArgs(args) : args)"
LOG_PREFIX = "[sam-extra] unload models"

_built_root = None   # weakref.ref(root Blocks) — 이 화면에는 이미 만들었다


def _log(text: str, error: bool = False) -> None:
    """콘솔 기록 — 콘솔이 글자를 못 써도(UnicodeEncodeError 등) 버튼의 응답은 막지 않는다."""
    stream = sys.stderr if error else sys.stdout
    try:
        print(text, file=stream)
    except (UnicodeError, OSError, ValueError):
        try:
            print(text.encode("ascii", "backslashreplace").decode("ascii"), file=stream)
        except Exception:
            pass


def _cuda_device(torch):
    """Forge 가 계산에 쓰는 CUDA 장치(memory_management.get_torch_device) — CPU 로 돌리면 PyTorch 의 현재 장치."""
    try:
        from backend import memory_management

        device = torch.device(memory_management.get_torch_device())
        if device.type == "cuda":
            return device
    except Exception:
        pass
    return None


def forge_host() -> unload_models.Host:
    """버튼을 누를 때의 Forge: queue_lock · shared.state · sd_models · Forge 의 모델 캐시(sys.modules) · torch."""
    import gc

    import torch
    from modules import sd_models, shared
    from modules.call_queue import queue_lock

    device = _cuda_device(torch)
    return unload_models.Host(
        try_lock=lambda: queue_lock.acquire(blocking=False),
        unlock=queue_lock.release,
        busy=lambda: unload_models.state_job(shared.state),
        unload_forge=lambda: unload_models.unload_forge_model(sd_models),
        hooks=unload_registry.hooks,
        collect=lambda: unload_models.collect_garbage(gc, torch),
        memory=lambda: unload_models.measure_cuda(torch, device),
        forge_caches=lambda: unload_models.forge_cache_hooks(sys.modules.get),
    )


def _report(outcome: unload_models.Outcome) -> None:
    if outcome.stale_job is not None:
        _log(f"{LOG_PREFIX}: refused once - Forge's queue lock is free but shared.state still shows the job "
             f"{outcome.stale_job!r} (a failed job left it); the same record on the next click is unloaded")
        return
    if outcome.refused is not None:
        _log(f"{LOG_PREFIX}: refused - "
             + ("a generation is running" if outcome.refused == unload_models.BUSY_MESSAGE else "already running"))
        return
    released = ", ".join(step.key for step in outcome.released) or "nothing held"
    line = f"{LOG_PREFIX}: released {released}"
    if outcome.forced_job is not None:
        line += f" (past the stale job record {outcome.forced_job!r})"
    before, after = outcome.before, outcome.after
    if before is not None and after is not None:
        line += (f"; GPU used {unload_models.gb(before.used)} -> {unload_models.gb(after.used)}, "
                 f"reserved {unload_models.gb(before.reserved)} -> {unload_models.gb(after.reserved)}")
    _log(line)
    for failure in outcome.errors:
        _log(f"{LOG_PREFIX}: {failure.key} failed - {failure.error}\n{failure.trace}".rstrip(), error=True)


def handle_unload(request_id, *_rest, host_factory=None) -> str:
    """숨은 버튼의 click — 언제나 JSON 글을 돌려준다(던지면 페이지가 요청 id 를 기다리다 시간 초과로 끝난다).

    ``_rest``: 페이지의 js 가 Gradio 의 출력값을 덧붙여 보낼 수 있다(쓰지 않는다)."""
    try:
        host = (host_factory or forge_host)()
        outcome = unload_models.run(host)
    except Exception as error:
        import traceback

        _log(f"{LOG_PREFIX}: unexpected error\n{traceback.format_exc()}", error=True)
        return unload_models.to_json(unload_models.error_payload(request_id, error))
    _report(outcome)
    return unload_models.to_json(unload_models.payload(request_id, outcome))


def build_bridge():
    """숨은 [요청 id · 실행 버튼 · 결과] 묶음을 지금 컨텍스트(demo)에 만들고 연결한다."""
    import gradio as gr

    with gr.Group(visible=False, elem_id=BRIDGE_ELEM_ID):
        request = gr.Textbox(value="", show_label=False, elem_id=REQUEST_ELEM_ID)
        run = gr.Button(value="sam3_unload_models", elem_id=RUN_ELEM_ID)
        result = gr.Textbox(value="", show_label=False, elem_id=RESULT_ELEM_ID)
    run.click(fn=handle_unload, js=ARGS_JS, inputs=[request], outputs=[result],
              queue=True, concurrency_limit=1, show_progress="hidden")
    return request, run, result


def on_after_component(component, **kwargs):
    """script_callbacks.on_after_component — footer 가 demo 에 생긴 직후 한 번(화면마다)."""
    global _built_root
    if kwargs.get("elem_id") != ANCHOR_ELEM_ID:
        return
    from gradio.context import Context

    root = Context.root_block
    # 핸들러가 gr.update() 를 돌려줄 때 Gradio 가 요청 중에 만드는 일회용 인스턴스(render=False)는 등록되지 않는다
    # — sam3ext/ui_save_defaults.py 의 같은 검사.
    if root is None or component._id not in root.default_config.blocks:
        return
    if _built_root is not None and _built_root() is root:
        return
    try:
        build_bridge()
    except Exception:
        import traceback

        _log(f"{LOG_PREFIX}: the button is not wired (the page's button reports an error)\n"
             f"{traceback.format_exc()}", error=True)
        return
    _built_root = weakref.ref(root)
