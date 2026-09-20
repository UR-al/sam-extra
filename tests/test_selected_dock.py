"""갤러리 밑 '선택 이미지' 탭 — 같은 위젯을 아코디언 대신 칼럼에 담아도 ui-config 키가 그대로인지.

ui-config 키는 `<탭>/<라벨>/<필드>` 라 컨테이너 종류는 키에 들어가지 않는다(modules/ui_loadsave.py:41-46).
그래서 바깥 아코디언을 칼럼으로 바꿔도 저장된 기본값이 그대로 적용돼야 한다.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import ui_dock  # noqa: E402
from sam3ext.ui_anima import build_anima_panel  # noqa: E402
from sam3ext.ui_anima_reference import build_anima_reference_panel  # noqa: E402
from sam3ext.ui_refine import build_refine_panel  # noqa: E402

SAMPLERS = ["Euler a", "ER SDE"]
SCHEDULERS = ["Simple", "Beta57"]
CHECKPOINTS = ["a.safetensors"]


def _widget_signature(panel):
    """(타입, 라벨, elem_id) 순서 — ui-config 키가 이 세 가지로 만들어진다."""
    return [
        (type(w).__name__, getattr(w, "label", None), getattr(w, "elem_id", None))
        for w in panel.all_widgets()
    ]


class ContainerParityTests(unittest.TestCase):
    def test_refine_panel_is_identical_in_both_containers(self):
        with gr.Blocks():
            as_accordion = build_refine_panel(SAMPLERS, SCHEDULERS, CHECKPOINTS)
        with gr.Blocks():
            as_column = build_refine_panel(SAMPLERS, SCHEDULERS, CHECKPOINTS, container="column")
        self.assertEqual(_widget_signature(as_accordion), _widget_signature(as_column))
        self.assertIsInstance(as_column.container, gr.Column)
        self.assertEqual(as_column.container.elem_id, "sam3_refine_panel")

    def test_reference_panel_is_identical_in_both_containers(self):
        with gr.Blocks():
            as_accordion = build_anima_reference_panel(SAMPLERS, SCHEDULERS)
        with gr.Blocks():
            as_column = build_anima_reference_panel(SAMPLERS, SCHEDULERS, container="column")
        self.assertEqual(_widget_signature(as_accordion), _widget_signature(as_column))
        self.assertEqual(as_column.container.elem_id, "sam3_anima_reference_panel")

    def test_tile_repair_panel_is_identical_in_both_containers(self):
        with gr.Blocks():
            as_accordion = build_anima_panel()
        with gr.Blocks():
            as_column = build_anima_panel(container="column")
        self.assertEqual(_widget_signature(as_accordion), _widget_signature(as_column))
        self.assertEqual(as_column.container.elem_id, "sam3_anima_panel")


class DockTests(unittest.TestCase):
    def test_dock_has_three_tabs_with_stable_ids(self):
        with gr.Blocks() as demo:
            panels = ui_dock.build_selected_image_dock(
                SAMPLERS, SCHEDULERS, CHECKPOINTS, anima_ok=True
            )
        self.assertIsNotNone(panels.refine)
        self.assertIsNotNone(panels.anima)
        self.assertIsNotNone(panels.reference)
        ids = [block.elem_id for block in demo.blocks.values() if getattr(block, "elem_id", None)]
        for elem_id in ("sam3_selected_image_dock", "sam3_dock_refine", "sam3_dock_tile", "sam3_dock_reference"):
            self.assertIn(elem_id, ids)

    def test_the_dock_can_be_collapsed_and_starts_closed(self):
        """탭은 접을 수 없으므로 접히는 칸으로 한 번 감싼다 — 안 그러면 Notebook 이 저 아래로 밀린다."""
        with gr.Blocks() as demo:
            ui_dock.build_selected_image_dock(SAMPLERS, SCHEDULERS, CHECKPOINTS, anima_ok=True)
        wraps = [b for b in demo.blocks.values()
                 if getattr(b, "elem_id", None) == ui_dock.DOCK_WRAP_ELEM_ID]
        self.assertEqual(len(wraps), 1)
        self.assertIsInstance(wraps[0], gr.Accordion)
        self.assertIs(wraps[0].open, False, "기본은 접힘")
        self.assertEqual(wraps[0].label, "선택 이미지 도구")

    def test_tile_tab_is_skipped_without_the_anima_vendor(self):
        with mock.patch("sys.stderr"):
            with gr.Blocks() as demo:
                panels = ui_dock.build_selected_image_dock(
                    SAMPLERS, SCHEDULERS, CHECKPOINTS, anima_ok=False
                )
        self.assertIsNone(panels.anima)
        ids = [block.elem_id for block in demo.blocks.values() if getattr(block, "elem_id", None)]
        self.assertNotIn("sam3_dock_tile", ids)

    def test_one_failing_builder_does_not_take_the_others_down(self):
        with mock.patch("sam3ext.ui_dock.build_anima_panel", side_effect=RuntimeError("boom")):
            with mock.patch("sys.stderr"):
                with gr.Blocks():
                    panels = ui_dock.build_selected_image_dock(
                        SAMPLERS, SCHEDULERS, CHECKPOINTS, anima_ok=True
                    )
        self.assertIsNone(panels.anima)
        self.assertIsNotNone(panels.refine)
        self.assertIsNotNone(panels.reference)

    def test_no_component_id_ends_with_gallery(self):
        """ui.js 의 갤러리 찾기가 이 탭 안을 갤러리로 오인하면 안 된다."""
        with gr.Blocks() as demo:
            ui_dock.build_selected_image_dock(SAMPLERS, SCHEDULERS, CHECKPOINTS, anima_ok=True)
        for block in demo.blocks.values():
            elem_id = getattr(block, "elem_id", None) or ""
            self.assertFalse(elem_id.endswith("_gallery"), elem_id)


class _Root:
    """실제 화면(gr.Blocks) 대역 — 약한 참조를 만들 수 있어야 한다."""


class DockStateTests(unittest.TestCase):
    """도크를 한 번만 만들고 패널을 한 번만 연결하게 하는 기억."""

    def test_builds_once_per_screen_and_again_after_a_ui_reload(self):
        state = ui_dock.DockState()
        first = _Root()
        self.assertTrue(state.needs_build(first))
        state.mark_built(ui_dock.DockPanels(refine=object()), first)
        self.assertFalse(state.needs_build(first), "같은 화면에서는 다시 만들지 않는다")
        second = _Root()
        self.assertTrue(state.needs_build(second), "Reload UI 로 화면이 새로 만들어지면 다시 만든다")

    def test_each_panel_is_wired_once(self):
        state = ui_dock.DockState()
        panel = object()
        state.mark_built(ui_dock.DockPanels(refine=panel, anima=None, reference=panel), _Root())
        self.assertTrue(state.needs_wire("refine", panel))
        state.mark_wired("refine")
        self.assertFalse(state.needs_wire("refine", panel))
        self.assertFalse(state.needs_wire("anima", None), "없는 패널은 연결하지 않는다")
        self.assertTrue(state.needs_wire("reference", panel))

    def test_rebuilding_clears_the_wiring_marks(self):
        state = ui_dock.DockState()
        panel = object()
        state.mark_built(ui_dock.DockPanels(refine=panel), _Root())
        state.mark_wired("refine")
        state.mark_built(ui_dock.DockPanels(refine=panel), _Root())
        self.assertTrue(state.needs_wire("refine", panel), "새 화면에서는 다시 연결해야 한다")


class HookTests(unittest.TestCase):
    SOURCE = (ROOT / "scripts" / "!sam3.py").read_text(encoding="utf-8")

    def test_the_hook_builds_the_dock_and_wires_through_the_state(self):
        """소스 검사 — 도크는 download_files 훅에서, 연결은 dock_state 판단으로."""
        self.assertIn('elif elem_id == "download_files_txt2img":', self.SOURCE)
        self.assertIn("ui_dock.build_selected_image_dock(", self.SOURCE)
        self.assertIn("dock_state.needs_build(", self.SOURCE)
        self.assertIn("dock_state.mark_built(", self.SOURCE)
        for name in ("refine", "anima", "reference"):
            self.assertIn(f'dock_state.needs_wire("{name}"', self.SOURCE)
            self.assertIn(f'dock_state.mark_wired("{name}")', self.SOURCE)
        branch = self.SOURCE[self.SOURCE.index('elif elem_id == "generation_info_txt2img":'):]
        self.assertNotIn("if refine_panel is not None or txt2img_gallery_component is None:", branch,
                         "도크가 refine_panel 을 먼저 채우므로 이 early return 은 모든 연결을 막는다")
        self.assertIn("layout_lanes.tag_runner(", branch)

    def test_the_old_one_shot_globals_are_gone(self):
        for name in ("anima_build_attempted", "anima_reference_build_attempted", "anima_wired",
                     "anima_reference_wired"):
            self.assertNotIn(name, self.SOURCE, f"{name} 은 dock_state 가 대신한다")

    def test_script_ui_no_longer_builds_the_moved_panels(self):
        ui_body = self.SOURCE[self.SOURCE.index("    def ui(self, is_img2img):"):
                              self.SOURCE.index("    def process(self, p, *args_):")]
        self.assertNotIn("build_anima_panel(", ui_body)
        self.assertNotIn("build_anima_reference_panel(", ui_body)
        self.assertIn("return components", ui_body)
        self.assertNotIn("anima_build_attempted", self.SOURCE, "쓰이지 않는 전역은 지운다")
        self.assertNotIn("anima_reference_build_attempted", self.SOURCE)

if __name__ == "__main__":
    unittest.main()
