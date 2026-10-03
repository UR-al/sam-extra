"""구도 · 카메라 — txt2img·img2img 스타일 줄 아래(Generate 옆 열) 접힌 칸("프롬프트로 시점 잡기")의 자리와 그 설정.

사용자 앱 UR_IV 의 CompositionControl(태그·구도 문구로 시점을 유도하는 패널 — 3D 카메라가 아님)을 Forge 로 옮긴
것이다. 패널은 브라우저 JS 가 그린다: 계산은 ``javascript/composition_prompt.js``(앱 compositionPrompt.ts 이식),
패널은 ``javascript/composition_ui.js``. 이 파일은 두 가지만 한다.

* Settings → **SAM Extra Appearance** 의 ``sam3_composition_panel``(기본 켬). UI 를 만들 때 읽으므로 바꾸면
  Reload UI 또는 재시작 뒤에 적용된다.
* 탭마다 빈 ``gr.HTML`` 하나(``#sam3_composition_<탭>`` 안의 ``div.sam3-composition-mount``)를 Forge 의 스타일 줄
  (``<탭>_styles_row``, modules/ui_prompt_styles.py) 바로 다음, 같은 열에 만든다. JS 가 그 div 를 채운다.

자리: webui 는 Row 가 만들어진 직후(``with`` 로 들어가기 전) after_component 를 부른다(modules/gradio_extensions.py
BlockContext_init). 그래서 여기서 만든 컴포넌트는 스타일 줄의 부모 열에서 그 줄 바로 뒤에 붙는다 — !sam3.py 가 TIPO
칸을 같은 자리에 만드는 방식과 같다. Forge 의 기본 배치(Default·Scrollable·Accordion)에서 그 열은 Generate 버튼 옆의
``<탭>_actions_column`` 이고, Compact 배치(modules/ui_toprow.py create_inline_toprow_prompts)에서는 설정 열 안
``toprow-compact-stylerow`` 줄의 두 번째(스타일) 열이다 — 어느 쪽이든 스타일 줄은 늘 만들어지므로 칸도 늘 생긴다.
txt2img 에서는 TIPO 칸 다음이어야 한다: Forge 는 한 확장의 scripts/ 를 파일 이름순으로 불러 콜백을 그 순서로 부르고
(modules/extensions.py list_files · script_callbacks.add_callback), ``!sam3.py`` 의 '!' 가 이 파일보다 앞선다.
metadata.ini 에는 after_component 순서 지시가 없다(tests/test_composition_camera.py 가 둘 다 지킨다). Always-on 스크립트가
아니어서 layout_lanes 의 칸도 없다.
"""
from __future__ import annotations

import gradio as gr
from gradio.context import Context

from modules import script_callbacks, shared


OPT_COMPOSITION_PANEL = "sam3_composition_panel"
SECTION = ("sam3_appearance", "SAM Extra Appearance")   # scripts/appearance_theme.py 와 같은 섹션

# Forge 의 스타일 줄 elem_id → 탭. 자리 이름은 javascript/composition_ui.js 의 HOST_ID_PREFIX · MOUNT_CLASS 와 같다.
TRIGGERS = {"txt2img_styles_row": "txt2img", "img2img_styles_row": "img2img"}
MOUNT_ELEM_ID = "sam3_composition_{tab}"
MOUNT_CLASS = "sam3-composition-mount"
HOST_CLASS = "sam3-composition-host"

# 탭 → 그 칸을 만든 root Blocks. 같은 Blocks 에 두 번 만들지 않는다(같은 elem_id 가 둘이면 JS 가 하나만 찾는다).
# Reload UI 는 새 Blocks 를 만들므로 다시 만든다.
_built: dict[str, object] = {}


def mount_html(tab: str) -> str:
    return f'<div class="{MOUNT_CLASS}" data-sam3-composition-tab="{tab}"></div>'


def panel_enabled() -> bool:
    return bool(getattr(shared.opts, OPT_COMPOSITION_PANEL, True))


def on_ui_settings() -> None:
    shared.opts.add_option(
        OPT_COMPOSITION_PANEL,
        shared.OptionInfo(
            True,
            "구도 · 카메라 칸 표시 (txt2img·img2img 스타일 줄 아래) — Reload UI 또는 Forge 재시작 후 적용",
            gr.Checkbox,
            section=SECTION,
        ).info(
            "태그·구도 문구(facing viewer, from above, cowboy shot …)로 시점을 유도하는 접힌 칸입니다 — 실제 3D 카메라 제어가 "
            "아닙니다. '메인 태그에 추가' 를 눌러야 프롬프트 끝에 없는 태그만 붙고, 적어 둔 글자는 바꾸지 않습니다. 끄면 칸을 "
            "만들지 않습니다. 조작값은 탭마다 이 브라우저에 기억합니다."
        ),
    )


def on_after_component(component, **kwargs):
    tab = TRIGGERS.get(kwargs.get("elem_id"))   # 거의 모든 컴포넌트는 여기서 끝난다
    if tab is None:
        return

    # Gradio rebuilds a component at *request* time whenever a handler returns
    # gr.update() for it — blocks.py postprocess_data does
    #     state[block._id] = block.__class__(**constructor_args | {"render": False})
    # and constructor_args still carries the original elem_id. webui patches
    # Component.__init__ and BlockContext.__init__ (modules/gradio_extensions.py),
    # so this callback also fires for those throwaway instances, with the very
    # elem_ids we match on above.
    #
    # gradio forces render=False on that rebuild (blocks.py:1740), so render()
    # never runs and the instance's _id never lands in demo.default_config.blocks.
    # Building our gr.HTML from such a callback would be worse than useless: at
    # request time there is no Blocks being built (the HTML would float free),
    # or — during a Reload UI rebuild — Context.root_block / Context.block are the
    # *new* UI's, so the HTML would be pushed into whatever container that rebuild
    # happens to be in (a second, misplaced mount with a duplicate elem_id).
    # !sam3.py documents the same trap for components it caches and later wires as
    # event outputs (KeyError: <id> in state_holder.__contains__).
    #
    # Testing registration directly is what we actually care about, and it is
    # exact: by the time webui fires this callback, BlockContext.__init__ has
    # already returned, so every genuine row is registered — including one
    # declared inside a render=False container (only the container itself defers,
    # cf. the Compact prompt layout in modules/ui_toprow.py). Checking
    # Context.root_block alone would still leak if an in-flight request echoed
    # during a Reload UI rebuild, since it is a process-wide global rather than a
    # ContextVar.
    root = Context.root_block
    if root is None:
        return
    if component._id not in root.default_config.blocks:
        return

    if not panel_enabled() or _built.get(tab) is root:
        return
    _built[tab] = root
    # 스타일 줄은 아직 열리지 않았다(with 전) — 이 HTML 은 그 줄의 부모 열, 줄 바로 다음에 들어간다.
    gr.HTML(mount_html(tab), elem_id=MOUNT_ELEM_ID.format(tab=tab), elem_classes=[HOST_CLASS])


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)
