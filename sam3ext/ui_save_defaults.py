"""'기본값 저장' 버튼의 Gradio 다리 — 숨은 버튼 하나를 main_entry 가 쓰는 23개 컴포넌트에 Gradio 방식으로 연결한다.

보이는 버튼은 javascript/notebook_save_defaults.js 가 notebook 레이아웃의 탭 줄(Generation · 임베딩 · 체크포인트 · 로라 ·
Manage) 오른쪽 끝에 만든다. 누르면 그 스크립트가 (1) Forge 의 Settings → Defaults Apply(#ui_defaults_apply)를 누르고
결과 글("Wrote N changes." / "No changes.")을 읽은 다음 (2) 여기 숨은 버튼(#sam3_save_defaults_run)을 누른다.

(2) 의 입력은 [요청 id, UI Preset, Checkpoint, VAE / Text Encoder, Diffusion in Low Bits, 그리고 forge_main_entry 가
get_a1111_ui_component 로 찾는 19개] — modules_forge/main_entry.py 가 on_preset_change 의 출력으로 쓰는 바로 그
컴포넌트들이다. Gradio 가 화면의 지금 값을 그대로 넘겨 주므로(드롭다운의 실제 값, 슬라이더의 숫자) JS 가 DOM 에서
값을 읽어 맞추지 않는다. 결과는 JSON 글로 숨은 Textbox(#sam3_save_defaults_result)에 돌아오고, 요청 id 가 같은
결과만 JS 가 받아들인다. 요청 id 는 이벤트의 ``js`` 가 입력의 첫 칸에 넣는다(Forge 의 submit 이 id_task 를 넣는 방식).

연결 시점: 이 컴포넌트들은 서로 다른 Blocks 에서 만들어진다(txt2img·img2img 인터페이스, 그리고 demo 의 빠른 설정 줄).
modules/ui.py create_ui 는 탭을 모두 렌더하고 loadsave.setup_ui() 를 부른 뒤 demo 안에서 footer(gr.HTML,
elem_id "footer")를 만들고, 그 뒤에 main_entry.forge_main_entry() 로 같은 컴포넌트들을 연결한다. 그래서 footer 의
on_after_component 에서 — 모든 컴포넌트가 demo 에 등록된 뒤, 같은 demo 컨텍스트 안에서 — 숨은 버튼을 만들고
연결한다. loadsave.setup_ui() 뒤라 이 숨은 컴포넌트들은 ui-config 에 들어가지 않는다.

Reload UI 는 새 demo 를 만든다 — 만든 화면은 약한 참조로 기억한다(sam3ext/ui_dock.py DockState 와 같은 이유).
"""
from __future__ import annotations

import sys
import traceback
import weakref

from sam3ext import save_defaults

ANCHOR_ELEM_ID = "footer"
BRIDGE_ELEM_ID = "sam3_save_defaults_bridge"
REQUEST_ELEM_ID = "sam3_save_defaults_request"
RUN_ELEM_ID = "sam3_save_defaults_run"
RESULT_ELEM_ID = "sam3_save_defaults_result"
# Gradio 4 는 입력값(+ 출력값)을 이 함수에 넘기고 돌려받은 배열을 백엔드에 보낸다 — 첫 칸에 요청 id.
ARGS_JS = "(...args) => (window.sam3SaveDefaultsArgs ? window.sam3SaveDefaultsArgs(args) : args)"
LOG_PREFIX = "[sam-extra] save defaults"

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


def forge_host() -> save_defaults.Host:
    """버튼을 누를 때의 Forge 사실들(설정 객체, 프리셋·체크포인트·모듈·샘플러·스케줄러 목록)."""
    from modules import sd_models, shared, shared_items
    from modules_forge import main_entry
    from modules_forge.presets import PresetArch

    return save_defaults.Host(
        opts=shared.opts,
        config_filename=shared.config_filename,
        presets=PresetArch.choices(),
        checkpoint_known=lambda name: sd_models.get_closet_checkpoint_match(name) is not None,
        module_paths=dict(main_entry.module_list),
        dtypes=list(main_entry.forge_unet_storage_dtype_options),
        samplers=[sampler.name for sampler in shared_items.list_samplers()],
        schedulers=list(shared_items.list_schedulers()),
        frozen=bool(getattr(shared.cmd_opts, "freeze_settings", False)),
    )


def handle_save(request_id, preset, *values, host_factory=None) -> str:
    """숨은 버튼의 click — 언제나 JSON 글을 돌려준다(던지면 페이지가 요청 id 를 기다리다 시간 초과로 끝난다)."""
    try:
        host = (host_factory or forge_host)()
        payload = save_defaults.save_preset(request_id, preset, list(values), host)
    except Exception as error:
        _log(f"{LOG_PREFIX}: unexpected error\n{traceback.format_exc()}", error=True)
        payload = {"ok": False, "request": save_defaults.clean_request_id(request_id),
                   "preset": preset if isinstance(preset, str) else None, "error": f"{type(error).__name__}: {error}"}
    if payload.get("ok"):
        kept = payload.get("kept") or []
        _log(f"{LOG_PREFIX}: preset {payload['preset']}: wrote {payload['written']} options, "
             f"{payload['changed']} changed"
             + (f", kept {len(kept)} at 0 (Forge keeps the screen value)" if kept else ""))
    else:
        _log(f"{LOG_PREFIX}: failed - {payload.get('error')}", error=True)
    return save_defaults.to_json(payload)


def find_inputs():
    """[UI Preset, Checkpoint, VAE / Text Encoder, Low Bits, …19개] — 하나라도 없으면 LookupError(이름과 함께)."""
    from modules_forge import main_entry

    lookup = getattr(main_entry, "get_a1111_ui_component", None)
    if lookup is None:
        raise LookupError("modules_forge.main_entry.get_a1111_ui_component 이 없습니다")
    named = [
        ("UI Preset", getattr(main_entry, "ui_forge_preset", None)),
        ("Checkpoint", getattr(main_entry, "ui_checkpoint", None)),
        ("VAE / Text Encoder", getattr(main_entry, "ui_vae", None)),
        ("Diffusion in Low Bits", getattr(main_entry, "ui_forge_unet_dtype", None)),
    ]
    named += [(f"{tab} {label}", lookup(tab, label)) for tab, label in save_defaults.UI_COMPONENTS]
    missing = [name for name, component in named if component is None]
    if missing:
        raise LookupError("컴포넌트를 찾지 못했습니다: " + ", ".join(missing))
    return [component for _name, component in named]


def build_bridge(inputs):
    """숨은 [요청 id · 실행 버튼 · 결과] 묶음을 지금 컨텍스트(demo)에 만들고 연결한다."""
    import gradio as gr

    with gr.Group(visible=False, elem_id=BRIDGE_ELEM_ID):
        request = gr.Textbox(value="", show_label=False, elem_id=REQUEST_ELEM_ID)
        run = gr.Button(value="sam3_save_defaults", elem_id=RUN_ELEM_ID)
        result = gr.Textbox(value="", show_label=False, elem_id=RESULT_ELEM_ID)
    run.click(fn=handle_save, js=ARGS_JS, inputs=[request, *inputs], outputs=[result],
              queue=False, show_progress="hidden")
    return request, run, result


def on_after_component(component, **kwargs):
    """script_callbacks.on_after_component — footer 가 demo 에 생긴 직후 한 번(화면마다)."""
    global _built_root
    if kwargs.get("elem_id") != ANCHOR_ELEM_ID:
        return
    from gradio.context import Context

    root = Context.root_block
    # 핸들러가 gr.update() 를 돌려줄 때 Gradio 가 요청 중에 만드는 일회용 인스턴스(render=False)는 등록되지 않는다
    # — scripts/!sam3.py on_after_component 의 설명과 같은 검사.
    if root is None or component._id not in root.default_config.blocks:
        return
    if _built_root is not None and _built_root() is root:
        return
    try:
        inputs = find_inputs()
        unregistered = [c for c in inputs if c._id not in root.default_config.blocks]
        if unregistered:
            raise LookupError(f"컴포넌트 {len(unregistered)}개가 이 화면에 등록되어 있지 않습니다")
        build_bridge(inputs)
    except Exception:
        _log(f"{LOG_PREFIX}: the button is not wired (the page's button reports an error)\n"
             f"{traceback.format_exc()}", error=True)
        return
    _built_root = weakref.ref(root)
