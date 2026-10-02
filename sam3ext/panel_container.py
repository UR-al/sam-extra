"""후처리 패널(Refine · Tile-Repair · 캐릭터 레퍼런스)을 담는 바깥 상자.

예전에는 각자 접히는 아코디언이었다. 갤러리 밑 "선택 이미지" 탭(sam3ext/ui_dock.py)에 들어갈 때는 탭이 이미
여닫는 역할을 하므로 같은 elem_id 의 칼럼으로 바꾼다. 안에 만드는 위젯의 순서와 라벨은 어느 쪽이든 같아야 한다
— ui-config 키가 `<탭>/<라벨>/<필드>` 라 저장된 기본값이 그대로 적용되어야 하기 때문이다.
"""
from __future__ import annotations

import gradio as gr

ACCORDION = "accordion"
COLUMN = "column"


def panel_container(kind: str, title: str, elem_id: str):
    """``kind`` 가 ``"column"`` 이면 칼럼, 아니면 접힌 아코디언."""
    if kind == COLUMN:
        return gr.Column(elem_id=elem_id, elem_classes=["sam3-dock-panel"])
    return gr.Accordion(title, open=False, elem_id=elem_id)
