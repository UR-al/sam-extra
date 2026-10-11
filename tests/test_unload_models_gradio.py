"""sam3ext/ui_unload_models.py + scripts/layout_unload_models.py — '모델 내리기' 숨은 버튼의 Gradio 연결(진짜 Gradio 4.40)과
Forge 쪽 사실(forge_host), 그리고 Forge 의 unload/reload 경로가 아직 버튼이 기대하는 모양인지(Forge 체크아웃의 AST).

Forge 의 화면 짜임은 tests/test_save_defaults_gradio.py 의 흉내를 그대로 쓴다(footer 의 on_after_component).
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network (Blocks construction)

import ast  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

import gradio as gr  # noqa: E402
from gradio.state_holder import SessionState  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import unload_models as um  # noqa: E402
from sam3ext import unload_registry as ur  # noqa: E402
from sam3ext import ui_unload_models as uum  # noqa: E402
from tests import test_save_defaults_gradio as sdg  # noqa: E402  (helpers only — no TestCase is imported by name)
from tests._forge_checkout import require_forge_file  # noqa: E402

GB = 1024 ** 3


def fake_host(calls, *, busy=False):
    memory = [um.Memory(used=12 * GB, total=24 * GB, reserved=10 * GB), um.Memory(used=2 * GB, total=24 * GB,
                                                                                  reserved=0)]
    return um.Host(
        try_lock=lambda: calls.append("lock") or True,
        unlock=lambda: calls.append("unlock"),
        busy=lambda: busy,
        unload_forge=lambda: calls.append("forge") or True,
        hooks=lambda: [ur.UnloadHook("sam3", "SAM3", lambda: calls.append("sam3") or True)],
        collect=lambda: calls.append("collect"),
        memory=lambda: memory.pop(0),
    )


class WiringTests(unittest.TestCase):
    def setUp(self):
        uum._built_root = None
        self.addCleanup(setattr, uum, "_built_root", None)

    def build(self, *, footers=1, throwaway_footer=False):
        with sdg._forge_after_component(uum.on_after_component):
            with gr.Blocks() as demo:
                gr.Textbox(label="Prompt", elem_id="txt2img_prompt")
                if throwaway_footer:
                    gr.HTML("<div>footer</div>", elem_id="footer", render=False)
                for _ in range(footers):
                    gr.HTML("<div>footer</div>", elem_id="footer")
        return demo

    def test_a_hidden_button_after_the_footer_with_the_request_id_in_and_json_out(self):
        demo = self.build()
        (run,) = sdg.by_elem_id(demo, uum.RUN_ELEM_ID)
        (request,) = sdg.by_elem_id(demo, uum.REQUEST_ELEM_ID)
        (result,) = sdg.by_elem_id(demo, uum.RESULT_ELEM_ID)
        (bridge,) = sdg.by_elem_id(demo, uum.BRIDGE_ELEM_ID)
        self.assertIs(bridge.visible, False)
        dep = sdg.click_dependency(demo, run)
        self.assertEqual([c._id for c in dep.inputs], [request._id])
        self.assertEqual([c._id for c in dep.outputs], [result._id])
        self.assertEqual(dep.js, uum.ARGS_JS)
        self.assertTrue(dep.queue, "through Gradio's queue: a long unload does not hold an HTTP request")
        self.assertEqual(dep.concurrency_limit, 1, "two tabs: one unload at a time")
        config = demo.get_config_file()
        (entry,) = [d for d in config["dependencies"] if d["js"] == uum.ARGS_JS]
        self.assertEqual((entry["inputs"], entry["outputs"]), ([request._id], [result._id]))
        ids = [child.elem_id for child in demo.children if getattr(child, "elem_id", None)]
        self.assertEqual(ids[-2:], ["footer", uum.BRIDGE_ELEM_ID])

    def _click(self, demo, *, extra_output=True, request_id="um-7"):
        (run,) = sdg.by_elem_id(demo, uum.RUN_ELEM_ID)
        dep = sdg.click_dependency(demo, run)
        inputs = [request_id] + ([""] if extra_output else [])
        state = SessionState(demo)
        return json.loads(asyncio.run(demo.process_api(block_fn=dep, inputs=inputs, state=state))["data"][0])

    def test_a_click_through_gradio_unloads_and_reports(self):
        demo = self.build()
        calls = []
        with mock.patch.object(uum, "forge_host", lambda: fake_host(calls)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            payload = self._click(demo)
        self.assertEqual(calls, ["lock", "forge", "sam3", "collect", "unlock"])
        self.assertEqual((payload["ok"], payload["request"], payload["refused"]), (True, "um-7", False))
        self.assertEqual(payload["text"],
                         "VRAM 사용 12.0 GB → 2.0 GB (Forge 모델 · SAM3) · 이 프로세스 예약 10.0 → 0.0 GB")
        self.assertIn("[sam-extra] unload models: released forge, sam3; GPU used 12.0 GB -> 2.0 GB", out.getvalue())
        with mock.patch.object(uum, "forge_host", lambda: fake_host([])), contextlib.redirect_stdout(io.StringIO()):
            again = self._click(demo, extra_output=False, request_id="um-8")
        self.assertEqual((again["ok"], again["request"]), (True, "um-8"))

    def test_a_refusal_reaches_the_page_as_json(self):
        demo = self.build()
        calls = []
        with mock.patch.object(uum, "forge_host", lambda: fake_host(calls, busy=True)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            payload = self._click(demo)
        self.assertEqual((payload["ok"], payload["refused"], payload["text"]), (False, True, "생성 중에는 내릴 수 없습니다"))
        self.assertEqual(calls, ["lock", "unlock"])
        self.assertIn("refused - a generation is running", out.getvalue())

    def test_the_handler_never_raises(self):
        def broken():
            raise RuntimeError("Forge is not ready")

        with contextlib.redirect_stderr(io.StringIO()) as err:
            payload = json.loads(uum.handle_unload("abc", host_factory=broken))
        self.assertEqual((payload["ok"], payload["request"], payload["text"]),
                         (False, "abc", "모델 내리기 실패: RuntimeError: Forge is not ready"))
        self.assertIn("Traceback", err.getvalue())

    def test_hook_failures_go_to_the_console_with_their_traceback(self):
        def host():
            calls = []
            base = fake_host(calls)
            base.hooks = lambda: [ur.UnloadHook("tipo", "TIPO", lambda: 1 / 0)]
            return base

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            payload = json.loads(uum.handle_unload("abc", "", host_factory=host))
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["text"].endswith(" · 실패: TIPO"))
        self.assertIn("tipo failed - ZeroDivisionError: division by zero", err.getvalue())
        self.assertIn("Traceback", err.getvalue())

    def test_built_once_per_screen_and_again_for_a_new_screen(self):
        demo = self.build(footers=2)
        self.assertEqual(len(sdg.by_elem_id(demo, uum.RUN_ELEM_ID)), 1)
        reloaded = self.build()                    # Reload UI: a new demo
        self.assertEqual(len(sdg.by_elem_id(reloaded, uum.RUN_ELEM_ID)), 1)
        self.assertIs(uum._built_root(), reloaded)

    def test_a_throwaway_footer_builds_nothing(self):
        demo = self.build(footers=0, throwaway_footer=True)
        self.assertEqual(sdg.by_elem_id(demo, uum.RUN_ELEM_ID), [])
        self.assertIsNone(uum._built_root)
        uum.on_after_component(types.SimpleNamespace(_id=-1), elem_id="footer")   # outside any Blocks
        self.assertIsNone(uum._built_root)


class FakeFifoLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls = []

    def acquire(self, blocking=True):
        self.calls.append(("acquire", blocking))
        return self._lock.acquire(blocking)

    def release(self):
        self.calls.append(("release",))
        self._lock.release()


class ForgeHostTests(unittest.TestCase):
    """forge_host(): what every real click reads from Forge (modules/call_queue.py, shared.py, sd_models.py,
    backend/memory_management.py) on stand-ins with Forge's shapes. torch is the real one (no GPU in this process)."""

    def stubs(self):
        modules = types.ModuleType("modules")
        modules.__path__ = []
        call_queue = types.ModuleType("modules.call_queue")
        call_queue.queue_lock = self.lock
        shared = types.ModuleType("modules.shared")
        shared.state = self.state
        sd_models = types.ModuleType("modules.sd_models")

        class FakeInitialModel:
            pass

        sd_models.FakeInitialModel = FakeInitialModel
        sd_models.model_data = types.SimpleNamespace(sd_model=object())
        sd_models.unload_model_weights = lambda *a, **k: self.unloads.append("unload_model_weights")
        for name, module in (("call_queue", call_queue), ("shared", shared), ("sd_models", sd_models)):
            setattr(modules, name, module)
        backend = types.ModuleType("backend")
        backend.__path__ = []
        memory_management = types.ModuleType("backend.memory_management")
        memory_management.current_loaded_models = []
        memory_management.get_torch_device = lambda: "cpu"
        backend.memory_management = memory_management
        return {"modules": modules, "modules.call_queue": call_queue, "modules.shared": shared,
                "modules.sd_models": sd_models, "backend": backend, "backend.memory_management": memory_management}

    def setUp(self):
        self.lock = FakeFifoLock()
        self.state = types.SimpleNamespace(job="", job_count=0, interrupted=True)
        self.unloads = []

    def test_every_fact_comes_from_forge(self):
        with sdg._modules(self.stubs()):
            host = uum.forge_host()
            self.assertTrue(host.try_lock())
            self.assertEqual(self.lock.calls, [("acquire", False)], "never waits for a generation")
            host.unlock()
            self.assertFalse(host.busy(), "an idle Interrupt left over is not a job")
            self.state.job = "txt2img"
            self.assertTrue(host.busy())
            self.assertTrue(host.unload_forge())
            self.assertEqual(self.unloads, ["unload_model_weights"])
            self.assertIs(host.hooks, ur.hooks)
            import torch

            if not torch.cuda.is_available():
                self.assertIsNone(host.memory(), "CPU only: nothing to measure")
                host.collect()

    def test_a_click_while_forge_holds_the_lock_is_refused(self):
        with sdg._modules(self.stubs()):
            host = uum.forge_host()
            self.lock._lock.acquire()
            try:
                outcome = um.run(host)
            finally:
                self.lock._lock.release()
        self.assertEqual(outcome.refused, um.BUSY_MESSAGE)
        self.assertEqual(self.unloads, [])


class ForgeLockTests(unittest.TestCase):
    """Forge's own FIFOLock (modules/fifo_lock.py): acquire(blocking=False) answers at once while a generation holds it."""

    def test_non_blocking_acquire_on_forges_lock(self):
        path = require_forge_file("modules/fifo_lock.py")
        spec = importlib.util.spec_from_file_location("_test_unload_fifo_lock", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        lock = module.FIFOLock()
        calls = []
        host = fake_host(calls)
        host = um.Host(try_lock=lambda: lock.acquire(blocking=False), unlock=lock.release, busy=host.busy,
                       unload_forge=host.unload_forge, hooks=host.hooks, collect=host.collect, memory=host.memory)
        with lock:                                # a generation (wrap_gradio_gpu_call: with queue_lock)
            self.assertEqual(um.run(host).refused, um.BUSY_MESSAGE)
        self.assertEqual(calls, [])
        self.assertIsNone(um.run(host).refused)
        self.assertTrue(lock.acquire(blocking=False), "given back")
        lock.release()


class ForgeUnloadPathTests(unittest.TestCase):
    """The Forge checkout still unloads and reloads the way the button relies on (modules/sd_models.py, processing.py)."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(require_forge_file("modules/sd_models.py").read_text(encoding="utf-8"))
        cls.functions = {node.name: node for node in cls.tree.body if isinstance(node, ast.FunctionDef)}

    def test_unload_model_weights_drops_the_model_and_the_hash_only(self):
        body = ast.unparse(self.functions["unload_model_weights"])
        for needle in ("memory_management.unload_all_models()", "model_data.sd_model = FakeInitialModel()",
                       "model_data.forge_hash = ''", "memory_management.soft_empty_cache()", "gc.collect()"):
            self.assertIn(needle, body)
        # the selected checkpoint · VAE/TE · Low Bits stay: the next generation reloads exactly those
        self.assertNotIn("forge_loading_parameters", body)

    def test_the_next_generation_reloads_because_the_hash_differs(self):
        body = ast.unparse(self.functions["forge_model_reload"])
        self.assertIn("current_hash = str(model_data.forge_loading_parameters)", body)
        self.assertIn("if model_data.forge_hash == current_hash:", body)
        self.assertIn("checkpoint_info = model_data.forge_loading_parameters['checkpoint_info']", body)
        self.assertIn("additional_state_dicts = model_data.forge_loading_parameters.get('additional_modules', [])",
                      body)
        processing = require_forge_file("modules/processing.py").read_text(encoding="utf-8")
        self.assertIn("p.sd_model, just_reloaded = forge_model_reload()", processing)

    def test_forges_own_buttons_call_the_same_function(self):
        settings = require_forge_file("modules/ui_settings.py").read_text(encoding="utf-8")
        self.assertIn("sd_models.unload_model_weights", settings)
        api = require_forge_file("modules/api/api.py").read_text(encoding="utf-8")
        self.assertIn('"/sdapi/v1/unload-checkpoint"', api)
        self.assertIn("sd_models.unload_model_weights()", api)


class ScriptFileTests(unittest.TestCase):
    def test_the_script_registers_one_after_component_callback(self):
        registered = []
        package = types.ModuleType("modules")
        package.__path__ = []
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_after_component = lambda fn, **kw: registered.append(("on_after_component", fn))
        package.script_callbacks = callbacks
        with sdg._modules({"modules": package, "modules.script_callbacks": callbacks}):
            spec = importlib.util.spec_from_file_location("_test_layout_unload_models",
                                                          ROOT / "scripts" / "layout_unload_models.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertEqual(registered, [("on_after_component", uum.on_after_component)])
        source = (ROOT / "scripts" / "layout_unload_models.py").read_text(encoding="utf-8")
        self.assertNotIn("class ", source.split('"""', 2)[2], "no Script class: nothing in the scripts column")

    def test_negpip_stays_the_last_script(self):
        names = sorted(name for name in os.listdir(ROOT / "scripts") if name.endswith(".py"))
        self.assertLess(names.index("layout_unload_models.py"), names.index("negpip.py"))
        self.assertEqual(names[-1], "negpip.py")

    def test_the_page_script_loads_after_the_save_button_script(self):
        names = sorted(os.listdir(ROOT / "javascript"))
        self.assertLess(names.index("notebook_save_defaults.js"), names.index("notebook_unload_models.js"))


if __name__ == "__main__":
    unittest.main()
