"""갤러리 밑 "선택 이미지" 탭 — 고른 결과 이미지에 쓰는 도구 세 가지를 한곳에 모은다.

SAM3 Refine(마스크로 다시 그리기) · Tile-Repair(타일 복원) · 캐릭터 레퍼런스. 예전에는 Refine 만 갤러리 밑에 있고
나머지 둘은 왼쪽 설정 열 중간에 있었다 — 셋 다 갤러리에서 고른 이미지에 작용하므로 이미지 옆으로 모았다.

이 묶음은 ``scripts/!sam3.py`` 의 ``download_files_txt2img`` 훅에서 만든다. 그 자리는 갤러리 버튼 줄 바로 뒤이고,
생성 정보 그룹(overflow:hidden) 밖이다(modules/ui_common.py:190-228).
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Any

import gradio as gr

from .panel_container import COLUMN
from .ui_anima import build_anima_panel
from .ui_anima_reference import build_anima_reference_panel
from .ui_refine import build_refine_panel

DOCK_ELEM_ID = "sam3_selected_image_dock"
DOCK_WRAP_ELEM_ID = "sam3_selected_image_dock_wrap"
DOCK_TITLE = "선택 이미지 도구"


@dataclass
class DockPanels:
    refine: Any = None
    anima: Any = None
    reference: Any = None


class DockState:
    """도크를 화면당 한 번만 만들고, 패널을 한 번만 연결하기 위한 기억.

    예전에는 ``!sam3.py`` 의 전역 플래그 네 개(refine_panel / anima_wired / …)가 같은 일을 했다.
    화면(root Blocks)은 Reload UI 때 새로 만들어지므로 약한 참조로 비교한다 — ``id()`` 는 재사용되어 위험하다.
    """

    def __init__(self) -> None:
        self.panels: DockPanels | None = None
        self._root = None
        self._wired: set[str] = set()

    def needs_build(self, root_block) -> bool:
        return self.panels is None or self._root is None or self._root() is not root_block

    def mark_built(self, panels: DockPanels, root_block) -> None:
        import weakref

        self.panels = panels
        self._root = weakref.ref(root_block) if root_block is not None else None
        self._wired.clear()

    def needs_wire(self, name: str, panel) -> bool:
        return panel is not None and name not in self._wired

    def mark_wired(self, name: str) -> None:
        self._wired.add(name)


def _tab(title: str, tab_id: str, elem_id: str, builder):
    """탭 하나. 안쪽 빌더가 실패해도 다른 탭은 만든다."""
    with gr.Tab(title, id=tab_id, elem_id=elem_id):
        try:
            return builder()
        except Exception:
            print(
                f"[-] SAM3: failed to build the {title} tab:\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            gr.Markdown(f"{title} 를 만들지 못했습니다 — 콘솔 로그를 확인해 주세요.")
            return None


def build_selected_image_dock(
    samplers: list[str],
    schedulers: list[str],
    checkpoints: list[str],
    *,
    anima_ok: bool,
) -> DockPanels:
    """열린 ``gr.Blocks`` 안에서 부른다. 갤러리 바로 아래에 놓이는 탭 묶음을 만든다.

    탭 자체는 접히지 않으므로 접히는 칸으로 한 번 감싼다(기본 접힘) — 안 그러면 패널 하나를 펼친 뒤
    그 아래의 Notebook 까지 내려가려면 한참을 스크롤해야 한다.
    """
    panels = DockPanels()
    with gr.Accordion(DOCK_TITLE, open=False, elem_id=DOCK_WRAP_ELEM_ID), \
            gr.Tabs(elem_id=DOCK_ELEM_ID, elem_classes=["sam3-dock"]):
        panels.refine = _tab(
            "SAM3 Refine",
            "refine",
            "sam3_dock_refine",
            lambda: build_refine_panel(samplers, schedulers, checkpoints, container=COLUMN),
        )
        if anima_ok:
            panels.anima = _tab(
                "Tile-Repair",
                "tile",
                "sam3_dock_tile",
                lambda: build_anima_panel(container=COLUMN),
            )
        else:
            print(
                "[-] SAM3: anima_vendor/ not present; Tile-Repair tab skipped. "
                "Re-run install.py to clone kohya-ss/sd-scripts.",
                file=sys.stderr,
            )
        panels.reference = _tab(
            "캐릭터 레퍼런스",
            "reference",
            "sam3_dock_reference",
            lambda: build_anima_reference_panel(samplers, schedulers, container=COLUMN),
        )
    return panels
