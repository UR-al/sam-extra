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

Settings 탭의 칸: 프리셋 설정은 Settings 탭(과 빠른 설정 줄)의 칸이기도 하다(modules/ui_settings.py — Forge 가 페이지를
열 때 demo.load 로 그 칸들에 설정 값을 넣는다). 저장한 뒤에도 그 칸들이 옛 값을 들고 있으면, 같은 페이지에서 누른
Apply settings 가 그 옛 값을 다시 써서 방금 저장한 값을 되돌린다. 그래서 숨은 버튼의 click 뒤에 ``.then`` 을 하나 더
걸어, 결과 JSON 의 ``written_keys`` 에 든 키의 칸만 Forge 가 페이지를 열 때 넣는 그대로(ui_settings.get_value_for_setting)
새 값으로 바꾸고 나머지 칸은 그대로 둔다(gr.update()). 출력은 ``shared.settings_components``(Forge 의 키 → 칸 표) 중
버튼이 쓸 수 있는 모든 프리셋의 키 — 누를 때 어느 프리셋이 활성일지 모른다. Settings 의 칸은 서버가 보낸 값을 Gradio
가 페이지의 상태에 넣으므로 Settings 탭을 한 번도 열지 않았어도 Apply settings 가 새 값을 보낸다. 칸이 없는 키(숨긴
설정 · Settings 탭을 숨긴 화면)는 출력에 넣지 않는다. 이 연결이 실패해도 저장 버튼은 그대로 연결한다.
연결 시점(footer)에는 Settings 의 칸이 모두 있다: create_ui 가 settings.create_ui 와 빠른 설정 줄을 footer 보다 먼저 만든다.

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


def find_settings_outputs(root):
    """[(설정 키, Settings 칸)] — 버튼이 쓸 수 있는 프리셋 키 중 이 화면에 칸이 있는 것(Settings 탭 · 빠른 설정 줄)."""
    from modules import shared
    from modules_forge.presets import PresetArch

    components = getattr(shared, "settings_components", None) or {}
    blocks = root.default_config.blocks
    found = []
    for key in save_defaults.preset_option_keys(PresetArch.choices()):
        component = components.get(key)
        # 칸이 없는 키(숨긴 설정 — Forge 는 체크포인트 · 모듈 · Low Bits 를 숨긴다), 값을 보이지 않는 gr.State 자리,
        # 이 화면에 없는 칸(Settings 탭을 숨긴 화면)은 넣지 않는다
        if component is None or getattr(component, "stateful", False) or getattr(component, "_id", None) not in blocks:
            continue
        found.append((key, component))
    return found


def settings_value(key):
    """Forge 가 페이지를 열 때 Settings 칸에 넣는 것 그대로(modules/ui_settings.py get_value_for_setting — 값과 선택지 ·
    슬라이더 범위). 그 함수가 없는 Forge 면 값만."""
    from modules import shared, ui_settings

    get_value = getattr(ui_settings, "get_value_for_setting", None)
    if get_value is not None:
        return get_value(key)
    import gradio as gr

    return gr.update(value=getattr(shared.opts, key))


def settings_updates(result_text, keys, value_for=None):
    """저장 click 의 ``.then`` — ``keys``(출력 칸의 키, 같은 순서) 중 이번에 쓴 키의 칸만 새 값, 나머지는 그대로(gr.update()).

    저장이 실패했거나 결과 글이 다른 모양이면 아무 칸도 바꾸지 않는다. 칸 하나의 값을 못 만들면 그 칸만 그대로 둔다.
    출력이 하나면 Gradio 는 돌려준 것을 그 칸의 값으로 읽으므로 목록이 아니라 그 하나를 돌려준다."""
    import gradio as gr

    value_for = value_for or settings_value
    written = set(save_defaults.written_keys_of(result_text))
    updates = []
    for key in keys:
        update = None
        if key in written:
            try:
                update = value_for(key)
            except Exception:
                _log(f"{LOG_PREFIX}: Settings field {key} not refreshed (reload the page before Apply settings)\n"
                     f"{traceback.format_exc()}", error=True)
        updates.append(update if update is not None else gr.update())
    return updates[0] if len(updates) == 1 else updates


def build_bridge(inputs, settings_outputs=()):
    """숨은 [요청 id · 실행 버튼 · 결과] 묶음을 지금 컨텍스트(demo)에 만들고 연결한다. ``settings_outputs`` 는
    find_settings_outputs 의 [(키, 칸)] — 있으면 저장 뒤 같은 클릭에서 쓴 키의 칸을 새 값으로 바꾼다."""
    import gradio as gr

    with gr.Group(visible=False, elem_id=BRIDGE_ELEM_ID):
        request = gr.Textbox(value="", show_label=False, elem_id=REQUEST_ELEM_ID)
        run = gr.Button(value="sam3_save_defaults", elem_id=RUN_ELEM_ID)
        result = gr.Textbox(value="", show_label=False, elem_id=RESULT_ELEM_ID)
    saved = run.click(fn=handle_save, js=ARGS_JS, inputs=[request, *inputs], outputs=[result],
                      queue=False, show_progress="hidden")
    if settings_outputs:
        keys = [key for key, _component in settings_outputs]

        def refresh_settings(result_text):
            return settings_updates(result_text, keys)

        saved.then(fn=refresh_settings, inputs=[result], outputs=[component for _key, component in settings_outputs],
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
        try:
            settings_outputs = find_settings_outputs(root)
        except Exception:
            settings_outputs = []
            _log(f"{LOG_PREFIX}: Settings fields are not refreshed after a save (reload the page before Apply "
                 f"settings)\n{traceback.format_exc()}", error=True)
        build_bridge(inputs, settings_outputs)
    except Exception:
        _log(f"{LOG_PREFIX}: the button is not wired (the page's button reports an error)\n"
             f"{traceback.format_exc()}", error=True)
        return
    _built_root = weakref.ref(root)
