from __future__ import annotations

import contextlib
import importlib
import io
import sys
import types
import unittest
from threading import Lock
from unittest import mock

import gradio as gr
from PIL import Image

from sam3ext import ui_refine
from sam3ext.ui_refine import REFINE_ARG_KEYS, _refine_widget_count, build_refine_panel


class RefineInputAlignmentTests(unittest.TestCase):
    def test_optional_canvas_keeps_three_txt2img_extras_aligned(self):
        self.assertEqual(_refine_widget_count(45), 42)  # canvas + 3 extras
        self.assertEqual(_refine_widget_count(43), 40)  # no canvas + 3 extras
        self.assertIsNone(_refine_widget_count(42))


class _FakeState:
    def __init__(self, job=""):
        self.job = job
        self.job_count = 0
        self.interrupted = False
        self.skipped = False
        self.textinfo = None


def _forge_modules(state):
    package = types.ModuleType("modules")
    package.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.state = state
    shared.sd_model = object()
    shared.opts = types.SimpleNamespace()
    shared.cmd_opts = types.SimpleNamespace()
    package.shared = shared
    return package, shared


def _refine_values(**changes):
    with gr.Blocks():
        panel = build_refine_panel(["Euler a"], ["Simple"], ["Use current"])
    widgets = panel.all_widgets()
    keys = REFINE_ARG_KEYS[: len(widgets)]
    keyed = {key: widget.value for key, widget in zip(keys, widgets)}
    keyed.update(changes)
    return tuple(keyed[key] for key in keys)


class RefineHandlerTests(unittest.TestCase):
    """감사 H1: Refine 은 Forge 의 queue_lock·state.begin/end 안에서, 제 job 이름으로 돌아야 한다."""

    def _click(self, values, *, refine=None):
        gallery = [Image.new("RGB", (64, 64), "white")]
        calls: list = []
        state = _FakeState()
        package, shared = _forge_modules(state)
        inpaint = types.ModuleType("sam3ext.inpaint_core")
        inpaint.run_sam3_refine = refine or (lambda *a, **k: [(gallery[0], "refined")])

        def fake_exclusive(job, fn, *, on_main_thread=False):
            calls.append((job, on_main_thread))
            return fn()

        modules = {"modules": package, "modules.shared": shared, "sam3ext.inpaint_core": inpaint}
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(ui_refine, "run_exclusive", side_effect=fake_exclusive),
        ):
            outputs = ui_refine.handle_refine_click(gallery, 0, *values, "1girl", "", "")
        return outputs, calls

    def test_refine_runs_as_a_forge_job_on_the_main_thread(self):
        outputs, calls = self._click(_refine_values(detect_prompt="face", inpaint_prompt="smile"))
        self.assertEqual(calls, [("sam3_refine", True)])
        self.assertEqual(len(outputs[0]), 2)

    def test_validation_errors_do_not_take_the_lock(self):
        outputs, calls = self._click(_refine_values(detect_prompt="", inpaint_prompt="smile"))
        self.assertEqual(calls, [])
        self.assertIn("enter a Target", outputs[1])

    def test_a_failed_refine_still_reports_through_the_status_line(self):
        def boom(*a, **k):
            raise RuntimeError("inside")

        outputs, calls = self._click(
            _refine_values(detect_prompt="face", inpaint_prompt="smile"), refine=boom
        )
        self.assertEqual(calls, [("sam3_refine", True)])
        self.assertIn("failed", outputs[1])


class RefineStopTests(unittest.TestCase):
    """⏹ Stop 은 Refine 이 job 을 쥐고 있을 때만 — 기다리는 동안 누르면 txt2img 가 대신 멈추면 안 된다."""

    def test_stop_interrupts_only_a_refine_job(self):
        for job, expected in (
            ("sam3_refine", True),
            ("SAM3 Refine pass 1/2", True),
            ("task(abc)", False),
            ("sam3_tile_repair", False),
            ("", False),
        ):
            state = _FakeState(job)
            package, shared = _forge_modules(state)
            with mock.patch.dict(sys.modules, {"modules": package, "modules.shared": shared}):
                ui_refine.stop_refine()
            self.assertEqual(state.interrupted, expected, job)
            self.assertEqual(state.skipped, expected, job)


class RunSam3RefineStateTests(unittest.TestCase):
    """run_sam3_refine 이 예외로 끝나도 진행 표시(textinfo)·job 이 남지 않는다."""

    def _load_inpaint_core(self, state):
        package, shared = _forge_modules(state)
        processing = types.ModuleType("modules.processing")
        processing.StableDiffusionProcessingImg2Img = object
        processing.process_images = lambda p: None
        scripts = types.ModuleType("modules.scripts")
        scripts.scripts_img2img = types.SimpleNamespace(alwayson_scripts=[], inputs=[])
        package.scripts = scripts
        call_queue = types.ModuleType("modules.call_queue")
        call_queue.queue_lock = Lock()
        package.call_queue = call_queue
        core = types.ModuleType("sam3ext.core")

        def run_sam3_on_pil(**kwargs):
            raise RuntimeError("detection exploded")

        core.run_sam3_on_pil = run_sam3_on_pil
        core.unload_sam3 = lambda: None
        return {
            "modules": package,
            "modules.shared": shared,
            "modules.processing": processing,
            "modules.scripts": scripts,
            "modules.call_queue": call_queue,
            "sam3ext.core": core,
        }

    def _run_failing_refine(self, state, *, extra_args=None, core_changes=None) -> str:
        """가짜 modules 로 sam3ext.inpaint_core 를 새로 불러 run_sam3_refine 을 실패시킨다. stderr 를 돌려준다."""
        import sam3ext

        modules = self._load_inpaint_core(state)
        for name, value in (core_changes or {}).items():
            setattr(modules["sam3ext.core"], name, value)
        missing = object()
        saved_attr = sam3ext.__dict__.get("inpaint_core", missing)
        stderr = io.StringIO()
        try:
            with mock.patch.dict(sys.modules, modules), contextlib.redirect_stderr(stderr):
                sys.modules.pop("sam3ext.inpaint_core", None)
                inpaint_core = importlib.import_module("sam3ext.inpaint_core")
                args = {
                    "sam3_prompt": "face",
                    "sam3_threshold": 0.5,
                    "sam3_checkpoint": "sam3.pt",
                    "sam3_device": "cpu",
                    **(extra_args or {}),
                }
                with self.assertRaises(RuntimeError):
                    inpaint_core.run_sam3_refine(
                        Image.new("RGB", (8, 8)),
                        args,
                        sd_model=object(),
                        outpath_samples="out",
                        outpath_grids="out",
                    )
        finally:
            # patch.dict 는 sys.modules 만 되돌린다 — import_module 이 바꿔 둔 패키지 속성도 원래대로(순서 의존 방지).
            if saved_attr is missing:
                sam3ext.__dict__.pop("inpaint_core", None)
            else:
                sam3ext.inpaint_core = saved_attr
        return stderr.getvalue()

    def test_textinfo_and_job_are_restored_when_detection_raises(self):
        state = _FakeState("sam3_refine")
        self._run_failing_refine(state)
        self.assertEqual(state.textinfo, "")
        self.assertEqual(state.job, "sam3_refine")

    def test_failure_log_says_what_unload_actually_did(self):
        for kept in (True, False):
            log = self._run_failing_refine(
                _FakeState("sam3_refine"),
                extra_args={"sam3_unload_after": True},
                core_changes={
                    "unload_sam3": lambda kept=kept: kept,
                    "describe_unload": lambda kept, after_failure=False: f"kept={kept} after_failure={after_failure}",
                },
            )
            self.assertIn(f"[-] SAM3 Refine: kept={kept} after_failure=True", log)
            self.assertNotIn("unloaded from VRAM", log)

    def test_the_fake_inpaint_core_does_not_leak_into_the_package(self):
        # import_module 이 sam3ext 패키지 속성 inpaint_core 를 가짜 modules 로 묶인 모듈로 바꿔 두면, 뒤에 도는
        # 테스트의 `from sam3ext import inpaint_core`(예: test_sam3_script.InnerPassTests)가 그것을 받는다(순서 의존).
        import sam3ext

        missing = object()
        before = sam3ext.__dict__.get("inpaint_core", missing)
        self.test_textinfo_and_job_are_restored_when_detection_raises()
        self.assertIs(sam3ext.__dict__.get("inpaint_core", missing), before)


if __name__ == "__main__":
    unittest.main()
