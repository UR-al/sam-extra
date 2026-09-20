"""TIPO 버튼·설정 칸 — 핸들러(가짜 런타임)와 실제 Gradio 연결."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import ui_tipo  # noqa: E402
from sam3ext.tipo import prompt as tp  # noqa: E402
from sam3ext.tipo.runtime import GenerationResult  # noqa: E402

PROMPT = "masterpiece, 1girl, hatsune miku, (smile:1.2)"
EXPANDED = PROMPT + ", outdoors, cherry blossoms\nA girl \\(smiling\\)."


class _FakeRuntime:
    def __init__(self, missing=(), text=", outdoors, cherry blossoms\nlong: A girl (smiling).", fail=None, note=None,
                 finished=True, busy=False):
        self.missing = list(missing)
        self.text = text
        self.fail = fail
        self.note = note
        self.finished = finished
        self.busy = busy
        self.calls = []
        self.downloads = 0

    def missing_files(self):
        return list(self.missing)

    def download(self):
        if self.busy:
            return False
        self.downloads += 1
        if self.fail:
            raise self.fail
        self.missing = []
        return True

    def generate(self, prompt_text, *, requested_device, max_new_tokens, seed):
        self.calls.append((prompt_text, requested_device, max_new_tokens, seed))
        if self.fail:
            raise self.fail
        return GenerationResult(self.text, 5 if seed < 0 else seed, "cuda" if requested_device == "GPU" else "cpu",
                                1.25, self.note, self.finished)


def _gradio():
    gradio = types.ModuleType("gradio")
    gradio.update = lambda **kw: {"update": kw}
    return gradio


NO_CHANGE = {"update": {}}


class ExpandTests(unittest.TestCase):
    def _expand(self, runtime, prompt=PROMPT, mode=tp.MODE_TAGS_NL, length=tp.LENGTH_NORMAL, allow=False,
                device="GPU", seed=-1):
        with mock.patch.dict(sys.modules, {"gradio": _gradio()}):
            return ui_tipo.handle_expand(prompt, 832, 1216, mode, length, allow, device, seed,
                                         runtime=runtime, index=tp.TagIndex({"hatsune miku": 4}))

    def test_success_hands_the_result_to_the_apply_step(self):
        runtime = _FakeRuntime()
        pending, status, download = self._expand(runtime)
        self.assertEqual(pending, {"before": PROMPT, "after": EXPANDED}, "적용은 다음 단계 — 그때 프롬프트가 그대로인지 본다")
        for part in ("태그 2개", "설명", "시드 5", "GPU", "1.2"):
            self.assertIn(part, status)
        self.assertEqual(download, {"update": {"visible": False}})
        tipo_input, device, max_tokens, seed = runtime.calls[0]
        self.assertIn("aspect ratio: 0.7", tipo_input)
        self.assertIn("characters: hatsune miku", tipo_input)
        budget = ui_tipo.max_new_tokens(tp.MODE_TAGS_NL, tp.LENGTH_NORMAL)
        self.assertEqual((device, max_tokens, seed), ("GPU", budget, -1))

    def test_token_budgets_fit_the_mode_and_length(self):
        for mode in tp.MODES:
            budgets = [ui_tipo.max_new_tokens(mode, length) for length in tp.LENGTHS]
            self.assertEqual(budgets, sorted(budgets), mode)
        self.assertGreaterEqual(ui_tipo.max_new_tokens(tp.MODE_TAGS_NL, tp.LENGTH_NORMAL), 512,
                                "태그 48개 + 설명 문장이 들어갈 만큼")
        self.assertGreaterEqual(ui_tipo.max_new_tokens(tp.MODE_TAGS_NL, tp.LENGTH_LONG), 768)
        self.assertGreater(ui_tipo.max_new_tokens(tp.MODE_TAGS_NL, tp.LENGTH_SHORT),
                           ui_tipo.max_new_tokens(tp.MODE_TAGS, tp.LENGTH_SHORT))

    def test_lengths_and_cpu(self):
        runtime = _FakeRuntime()
        self._expand(runtime, length=tp.LENGTH_LONG, device="CPU", seed=7)
        self.assertEqual(runtime.calls[0][1:], ("CPU", ui_tipo.max_new_tokens(tp.MODE_TAGS_NL, tp.LENGTH_LONG), 7))
        self._expand(runtime, mode=tp.MODE_TAGS, length=tp.LENGTH_SHORT)
        self.assertEqual(runtime.calls[1][2], ui_tipo.max_new_tokens(tp.MODE_TAGS, tp.LENGTH_SHORT))

    def test_unknown_labels_fall_back_to_defaults(self):
        runtime = _FakeRuntime()
        self._expand(runtime, mode="?", length="?", device="?")
        tipo_input, device, max_tokens, _ = runtime.calls[0]
        self.assertIn("<|long|> <|tag_to_long|>", tipo_input)
        self.assertEqual((device, max_tokens), ("GPU", ui_tipo.max_new_tokens(tp.MODE_TAGS_NL, tp.LENGTH_NORMAL)))

    def test_cut_off_output_keeps_only_complete_parts(self):
        runtime = _FakeRuntime(text=", outdoors, cherry blossoms\nlong: A girl smiles. Her dre", finished=False)
        pending, status, _ = self._expand(runtime)
        self.assertEqual(pending["after"], PROMPT + ", outdoors, cherry blossoms\nA girl smiles.")
        self.assertIn("토큰 한도", status)

    def test_empty_prompt_and_missing_model_leave_the_prompt_alone(self):
        pending, status, _ = self._expand(_FakeRuntime(), prompt="   ")
        self.assertIsNone(pending)
        self.assertIn("프롬프트", status)
        runtime = _FakeRuntime(missing=["model.safetensors"])
        pending, status, download = self._expand(runtime)
        self.assertIsNone(pending)
        self.assertIn("1.98 GB", status)
        self.assertEqual(download, {"update": {"visible": True}})
        self.assertEqual(runtime.calls, [])

    def test_failure_leaves_the_prompt_alone(self):
        with mock.patch("sys.stderr"):
            pending, status, _ = self._expand(_FakeRuntime(fail=RuntimeError("boom")))
        self.assertIsNone(pending)
        self.assertIn("boom", status)

    def test_nothing_new_is_reported(self):
        pending, status, _ = self._expand(_FakeRuntime(text=", smile, hatsune miku\n"), mode=tp.MODE_TAGS)
        self.assertIsNone(pending)
        self.assertIn("찾지 못했습니다", status)

    def test_cpu_fallback_note_is_shown(self):
        _, status, _ = self._expand(_FakeRuntime(note="GPU 여유가 1.0 GB 라 CPU 로 돌렸습니다"))
        self.assertIn("1.0 GB", status)


class ApplyTests(unittest.TestCase):
    def _apply(self, current, pending, undo="earlier"):
        with mock.patch.dict(sys.modules, {"gradio": _gradio()}):
            return ui_tipo.handle_apply(current, pending, undo)

    def test_applies_when_the_prompt_is_unchanged(self):
        text, undo, status, pending = self._apply(PROMPT, {"before": PROMPT, "after": EXPANDED})
        self.assertEqual((text, undo, status, pending), (EXPANDED, PROMPT, NO_CHANGE, None),
                         "되돌리기는 확장 전 프롬프트, 상태 줄은 확장 단계가 쓴 그대로")

    def test_edits_made_while_waiting_are_kept(self):
        text, undo, status, pending = self._apply(PROMPT + ", night", {"before": PROMPT, "after": EXPANDED})
        self.assertEqual((text, undo, pending), (NO_CHANGE, "earlier", None))
        self.assertIn("바뀌어", status)

    def test_nothing_pending(self):
        self.assertEqual(self._apply(PROMPT, None), (NO_CHANGE, "earlier", NO_CHANGE, None))


class UndoAndDownloadTests(unittest.TestCase):
    def test_undo(self):
        with mock.patch.dict(sys.modules, {"gradio": _gradio()}):
            self.assertEqual(ui_tipo.handle_undo("before", "after")[:2], ("before", None))
            text, undo, status = ui_tipo.handle_undo(None, "after")
        self.assertEqual((text, undo), (NO_CHANGE, None))
        self.assertIn("없습니다", status)

    def test_download(self):
        runtime = _FakeRuntime(missing=["model.safetensors"])
        with mock.patch.dict(sys.modules, {"gradio": _gradio()}):
            status, button = ui_tipo.handle_download(runtime=runtime)
            self.assertEqual((runtime.downloads, button), (1, {"update": {"visible": False}}))
            self.assertIn("받았습니다", status)
            status, _ = ui_tipo.handle_download(runtime=runtime)
            self.assertEqual(runtime.downloads, 1, "이미 있으면 다시 받지 않는다")
            failing = _FakeRuntime(missing=["model.safetensors"], fail=OSError("network down"))
            with mock.patch("sys.stderr"):
                status, button = ui_tipo.handle_download(runtime=failing)
            self.assertIn("network down", status)
            self.assertEqual(button, {"update": {"visible": True}})
            status, button = ui_tipo.handle_download(runtime=_FakeRuntime(missing=["model.safetensors"], busy=True))
        self.assertIn("받는 중", status)
        self.assertEqual(button, {"update": {"visible": False}}, "받는 동안 또 누르지 않게")


class GradioWiringTests(unittest.TestCase):
    """실제 Gradio 4.40 으로 버튼·설정 칸을 만들고 연결한다 (Forge 의 ToolButton 만 대역)."""

    def test_panel_button_and_click_wiring(self):
        import gradio as gr

        ui_components = types.ModuleType("modules.ui_components")
        ui_components.ToolButton = lambda value, elem_id=None, tooltip=None: gr.Button(value, elem_id=elem_id)
        package = types.ModuleType("modules")
        package.__path__ = []
        package.ui_components = ui_components
        with mock.patch.dict(sys.modules, {"modules": package, "modules.ui_components": ui_components}):
            with gr.Blocks() as demo:
                prompt = gr.Textbox(elem_id="txt2img_prompt")
                width, height = gr.Slider(elem_id="txt2img_width"), gr.Slider(elem_id="txt2img_height")
                with gr.Row(elem_id="txt2img_tools") as tools:
                    gr.Button("apply styles", elem_id="txt2img_style_apply")
                    button = ui_tipo.create_tipo_button()
                panel = ui_tipo.build_tipo_panel(model_missing=True)
                ui_tipo.wire_tipo(button, panel, prompt, width, height)
        self.assertIs(tools.children[-1], button)
        self.assertEqual(button.elem_id, "txt2img_tipo_expand")
        self.assertEqual(panel.mode.value, tp.MODE_TAGS_NL)
        self.assertEqual(panel.length.value, tp.LENGTH_NORMAL)
        self.assertEqual(panel.device.value, "GPU")
        self.assertEqual(panel.device.elem_id, "sam3_tipo_device", "javascript/tipo_device.js 가 이 id 로 찾는다")
        self.assertIs(panel.allow_new_names.value, False)
        self.assertTrue(panel.download_button.visible)
        fns = list(demo.fns.values()) if isinstance(demo.fns, dict) else list(demo.fns)

        def ids(blocks):
            return [b._id for b in blocks]

        def clicked(component):
            return [f for f in fns if (component._id, "click") in [tuple(t) for t in f.targets]][0]

        def after(dep):
            return [f for f in fns if f.trigger_after == fns.index(dep)][0]

        expand = clicked(button)
        self.assertEqual(ids(expand.inputs), ids((prompt, width, height, panel.mode, panel.length,
                                                  panel.allow_new_names, panel.device, panel.seed)))
        self.assertEqual(ids(expand.outputs), ids((panel.pending, panel.status, panel.download_button)),
                         "확장 단계는 프롬프트 칸에 쓰지 않는다")
        apply = after(expand)
        self.assertEqual(ids(apply.inputs), ids((prompt, panel.pending, panel.undo_state)),
                         "적용 단계는 지금 프롬프트 칸의 값을 다시 읽는다")
        self.assertEqual(ids(apply.outputs), ids((prompt, panel.undo_state, panel.status, panel.pending)))
        self.assertIn("update_txt2img_tokens", after(apply).js, "토큰 수를 다시 센다")
        undo = clicked(panel.undo_button)
        self.assertEqual(ids(undo.outputs), ids((prompt, panel.undo_state, panel.status)))
        self.assertIn("update_txt2img_tokens", after(undo).js)
        self.assertEqual(ids(clicked(panel.download_button).outputs), ids((panel.status, panel.download_button)))


class DeviceMemoryScriptTests(unittest.TestCase):
    def test_script_ships_with_the_extension(self):
        script = (ROOT / "javascript" / "tipo_device.js").read_text(encoding="utf-8")
        self.assertIn("sam3_tipo_device", script)


if __name__ == "__main__":
    unittest.main()
