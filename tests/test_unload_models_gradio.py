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


def fake_host(calls, *, busy=False, locked=False):
    memory = [um.Memory(used=12 * GB, total=24 * GB, reserved=10 * GB), um.Memory(used=2 * GB, total=24 * GB,
                                                                                  reserved=0)]
    return um.Host(
        try_lock=lambda: calls.append("lock") or not locked,
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
        um._STALE_SEEN[0] = None
        self.addCleanup(um._STALE_SEEN.__setitem__, 0, None)

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
        with mock.patch.object(uum, "forge_host", lambda: fake_host(calls, locked=True)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            payload = self._click(demo)
        self.assertEqual((payload["ok"], payload["refused"], payload["text"]), (False, True, "생성 중에는 내릴 수 없습니다"))
        self.assertEqual(calls, ["lock"])
        self.assertIn("refused - a generation is running", out.getvalue())

    def test_a_stale_job_record_is_named_then_the_next_click_unloads(self):
        demo = self.build()
        calls = []
        record = ("extras", 1, "20261011213000")
        with mock.patch.object(uum, "forge_host", lambda: fake_host(calls, busy=record)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            first = self._click(demo)
            second = self._click(demo, request_id="um-8")
        self.assertEqual((first["ok"], first["refused"], first["stale_job"]), (False, True, "extras"))
        self.assertEqual(first["text"],
                         "Forge 에 끝나지 않은 작업 기록(extras)이 남아 있습니다 — 생성 중이 아니면 한 번 더 누르세요")
        self.assertEqual((second["ok"], second["refused"], second["request"]), (True, False, "um-8"))
        self.assertEqual(calls, ["lock", "unlock", "lock", "forge", "sam3", "collect", "unlock"])
        self.assertIn("refused once - Forge's queue lock is free but shared.state still shows the job 'extras'",
                      out.getvalue())
        self.assertIn("released forge, sam3 (past the stale job record 'extras')", out.getvalue())

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
        shared.face_restorers = []
        taesd = types.ModuleType("modules.sd_vae_taesd")
        taesd.sd_vae_taesd_models = self.taesd_models
        backend = types.ModuleType("backend")
        backend.__path__ = []
        memory_management = types.ModuleType("backend.memory_management")
        memory_management.current_loaded_models = []
        memory_management.get_torch_device = lambda: "cpu"
        backend.memory_management = memory_management
        return {"modules": modules, "modules.call_queue": call_queue, "modules.shared": shared,
                "modules.sd_models": sd_models, "modules.sd_vae_taesd": taesd,
                "backend": backend, "backend.memory_management": memory_management}

    def setUp(self):
        self.lock = FakeFifoLock()
        self.state = types.SimpleNamespace(job="", job_count=0, interrupted=True, job_timestamp="20261011213000")
        self.unloads = []
        self.taesd_models = {"taesd_decoder.pth": object()}
        um._STALE_SEEN[0] = None
        self.addCleanup(um._STALE_SEEN.__setitem__, 0, None)

    def test_every_fact_comes_from_forge(self):
        with sdg._modules(self.stubs()):
            host = uum.forge_host()
            self.assertTrue(host.try_lock())
            self.assertEqual(self.lock.calls, [("acquire", False)], "never waits for a generation")
            host.unlock()
            self.assertFalse(host.busy(), "an idle Interrupt left over is not a job")
            self.state.job = "txt2img"
            self.assertEqual(host.busy(), ("txt2img", 0, "20261011213000"))
            self.assertTrue(host.unload_forge())
            self.assertEqual(self.unloads, ["unload_model_weights"])
            self.assertIs(host.hooks, ur.hooks)
            caches = {h.key: h for h in host.forge_caches()}
            self.assertEqual(list(caches), ["forge_upscalers", "forge_taesd", "forge_vae_approx",
                                            "forge_face_restorers"])
            self.assertFalse(caches["forge_upscalers"].release(), "modules.esrgan_model is not imported here")
            self.assertTrue(caches["forge_taesd"].release())
            self.assertEqual(self.taesd_models, {})
            self.assertFalse(caches["forge_face_restorers"].release())
            import torch

            if not torch.cuda.is_available():
                self.assertIsNone(host.memory(), "CPU only: nothing to measure")
                host.collect()

    def test_no_checkpoint_and_only_a_sam_extra_patcher_in_forges_list_is_not_forge_model(self):
        # fresh start (FakeInitialModel) + TIPO kept on the GPU (sam3ext/tipo/runtime.py load_models_gpu)
        stubs = self.stubs()
        sd_models = stubs["modules.sd_models"]
        sd_models.model_data.sd_model = sd_models.FakeInitialModel()
        stubs["backend.memory_management"].current_loaded_models.append(types.SimpleNamespace(model="tipo patcher"))
        with sdg._modules(stubs):
            self.assertFalse(uum.forge_host().unload_forge(), "TIPO's own hook names it, not 'Forge 모델'")
        self.assertEqual(self.unloads, ["unload_model_weights"], "Forge's unload still runs")

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

    def setUp(self):
        um._STALE_SEEN[0] = None
        self.addCleanup(um._STALE_SEEN.__setitem__, 0, None)

    def _forge_lock(self):
        path = require_forge_file("modules/fifo_lock.py")
        spec = importlib.util.spec_from_file_location("_test_unload_fifo_lock", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.FIFOLock()

    def test_a_failed_extras_api_call_leaves_a_record_that_the_second_click_gets_past(self):
        # modules/postprocessing.py run_postprocessing: state.begin(job="extras") (:15), job_count (:51), end() only on
        # success (:109); modules/api/api.py runs it under queue_lock with no state reset (the UI route has
        # wrap_gradio_call's finally). The lock is free afterwards, the record stays.
        post = ast.parse(require_forge_file("modules/postprocessing.py").read_text(encoding="utf-8"))
        (run_post,) = [n for n in post.body if isinstance(n, ast.FunctionDef) and n.name == "run_postprocessing"]
        body = ast.unparse(run_post)
        self.assertIn("shared.state.begin(job='extras')", body)
        finally_ends = [n for n in ast.walk(run_post) if isinstance(n, ast.Try)
                        and "shared.state.end()" in "".join(ast.unparse(s) for s in n.finalbody)]
        self.assertEqual(finally_ends, [], "Forge now ends the job in a finally: this case is gone")
        lock = self._forge_lock()
        state = types.SimpleNamespace(job="", job_count=0, job_timestamp="")

        def extras_api_call():
            with lock:                                                # api.py extras_single_image_api
                state.job, state.job_count, state.job_timestamp = "extras", -1, "20261011213000"   # State.begin
                state.job_count = 1
                raise RuntimeError("upscaler failed")                 # before state.end()

        with self.assertRaises(RuntimeError):
            extras_api_call()
        calls = []
        base = fake_host(calls)
        host = um.Host(try_lock=lambda: lock.acquire(blocking=False), unlock=lock.release,
                       busy=lambda: um.state_job(state), unload_forge=base.unload_forge, hooks=base.hooks,
                       collect=base.collect, memory=base.memory)
        first = um.run(host)
        self.assertEqual(first.stale_job, "extras")
        self.assertEqual(calls, [], "the first click names the record and unloads nothing")
        second = um.run(host)
        self.assertIsNone(second.refused)
        self.assertEqual(second.forced_job, "extras")
        self.assertEqual(calls, ["forge", "sam3", "collect"])
        self.assertTrue(lock.acquire(blocking=False), "given back")
        lock.release()

    def test_non_blocking_acquire_on_forges_lock(self):
        lock = self._forge_lock()
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

    def test_the_hires_quick_button_skips_the_reload_after_before_process(self):
        """✨ after the button: process_images skips manage_model_and_prompt_cache, and process_images_inner's first look
        at the model is an attribute FakeInitialModel lacks — so the reload has to come from before_process, which runs
        first (scripts/layout_unload_models.py)."""
        processing = ast.parse(require_forge_file("modules/processing.py").read_text(encoding="utf-8"))
        functions = {n.name: n for n in processing.body if isinstance(n, ast.FunctionDef)}
        process_images = functions["process_images"]
        before = [n.lineno for n in ast.walk(process_images)
                  if isinstance(n, ast.Call) and ast.unparse(n.func) == "p.scripts.before_process"]
        (skip,) = [n for n in ast.walk(process_images)
                   if isinstance(n, ast.If) and ast.unparse(n.test) == "getattr(p, 'txt2img_upscale', False)"]
        self.assertEqual([ast.unparse(s) for s in skip.body], ["pass"], "✨ loads no model in process_images")
        self.assertIn("manage_model_and_prompt_cache(p)", [ast.unparse(s) for s in skip.orelse])
        self.assertEqual(len(before), 1)
        self.assertLess(before[0], skip.lineno, "before_process runs before the skip")
        reads = sorted((n.lineno, n.col_offset, n.attr) for n in ast.walk(functions["process_images_inner"])
                       if isinstance(n, ast.Attribute) and ast.unparse(n.value) == "shared.sd_model")
        first = reads[0][2]
        fake = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == "FakeInitialModel")
        members = {n.name for n in fake.body if isinstance(n, ast.FunctionDef)}
        self.assertNotIn(first, members, f"FakeInitialModel has no {first!r}: ✨ would raise without the reload")
        txt2img = ast.parse(require_forge_file("modules/txt2img.py").read_text(encoding="utf-8"))
        (upscale,) = [n for n in txt2img.body if isinstance(n, ast.FunctionDef) and n.name == "txt2img_upscale_function"]
        body = ast.unparse(upscale)
        self.assertIn("p.txt2img_upscale = True", body)
        self.assertIn("processing.process_images(p)", body)
        scripts_tree = ast.parse(require_forge_file("modules/scripts.py").read_text(encoding="utf-8"))
        runner = next(n for n in scripts_tree.body if isinstance(n, ast.ClassDef) and n.name == "ScriptRunner")
        (runner_before,) = [n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "before_process"]
        self.assertIn("script.before_process(p, *script_args)", ast.unparse(runner_before))

    def test_the_models_forge_keeps_outside_its_memory_list(self):
        """unload_all_models only walks current_loaded_models; these stay on the GPU (FORGE_CACHE_HOOKS)."""
        memory = ast.parse(require_forge_file("backend/memory_management.py").read_text(encoding="utf-8"))
        (unload_all,) = [n for n in memory.body if isinstance(n, ast.FunctionDef) and n.name == "unload_all_models"]
        self.assertEqual([ast.unparse(s) for s in unload_all.body], ["free_memory(1e+30, get_torch_device())"])
        esrgan = ast.parse(require_forge_file("modules/esrgan_model.py").read_text(encoding="utf-8"))
        upscaler = next(n for n in esrgan.body if isinstance(n, ast.ClassDef) and n.name == "UpscalerESRGAN")
        (load_model,) = [n for n in upscaler.body if isinstance(n, ast.FunctionDef) and n.name == "load_model"]
        self.assertTrue(any(ast.unparse(d).startswith("lru_cache(") for d in load_model.decorator_list))
        self.assertIn("model.to(devices.device_esrgan)", ast.unparse(load_model))
        for relpath, name in (("modules/sd_vae_taesd.py", "sd_vae_taesd_models"),
                              ("modules/sd_vae_approx.py", "sd_vae_approx_models")):
            tree = ast.parse(require_forge_file(relpath).read_text(encoding="utf-8"))
            globals_ = {ast.unparse(t) for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])}
            self.assertIn(name, globals_, relpath)
            self.assertIn(f"{name}[model_name] = loaded_model", ast.unparse(tree), relpath)
            self.assertIn("loaded_model.to(devices.device, devices.dtype)", ast.unparse(tree), relpath)
        shared = ast.parse(require_forge_file("modules/shared.py").read_text(encoding="utf-8"))
        self.assertIn("face_restorers", {ast.unparse(n.target) for n in shared.body if isinstance(n, ast.AnnAssign)})
        utils = ast.parse(require_forge_file("modules/face_restoration_utils.py").read_text(encoding="utf-8"))
        common = next(n for n in utils.body if isinstance(n, ast.ClassDef) and n.name == "CommonFaceRestoration")
        methods = {n.name: n for n in common.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("load_net", methods)
        self.assertEqual([ast.unparse(d) for d in methods["face_helper"].decorator_list], ["cached_property"])
        self.assertIn("if self.net is None:\n        self.net = self.load_net()",
                      ast.unparse(methods["restore_with_helper"]).replace("            ", "        "))


class ScriptFileTests(unittest.TestCase):
    def _load(self):
        registered = []
        package = types.ModuleType("modules")
        package.__path__ = []
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_after_component = lambda fn, **kw: registered.append(("on_after_component", fn))
        scripts_module = types.ModuleType("modules.scripts")

        class Script:
            create_group = True

        scripts_module.Script = Script
        scripts_module.AlwaysVisible = object()
        package.script_callbacks = callbacks
        package.scripts = scripts_module
        with sdg._modules({"modules": package, "modules.script_callbacks": callbacks, "modules.scripts": scripts_module}):
            spec = importlib.util.spec_from_file_location("_test_layout_unload_models",
                                                          ROOT / "scripts" / "layout_unload_models.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module, registered, scripts_module

    def test_the_script_registers_one_after_component_callback(self):
        module, registered, scripts_module = self._load()
        self.assertEqual(registered, [("on_after_component", uum.on_after_component)])
        source = (ROOT / "scripts" / "layout_unload_models.py").read_text(encoding="utf-8")
        classes = [n.name for n in ast.parse(source).body if isinstance(n, ast.ClassDef)]
        self.assertEqual(classes, ["UnloadModelsReload"], "one script, for the ✨ reload only")

    def test_the_reload_script_puts_nothing_on_screen(self):
        module, _, scripts_module = self._load()
        script = module.UnloadModelsReload()
        self.assertIsInstance(script, scripts_module.Script)
        self.assertIs(script.create_group, False, "no empty group in the scripts column")
        self.assertIs(script.show(False), scripts_module.AlwaysVisible)
        self.assertIs(script.show(True), False, "✨ is a txt2img gallery button")
        self.assertEqual(script.ui(False), [])

    def test_after_the_button_the_hires_quick_button_loads_the_checkpoint_in_before_process(self):
        module, _, _ = self._load()
        calls = []

        class FakeInitialModel:
            pass

        class Checkpoint:
            is_wan = False

        sd_models = types.ModuleType("modules.sd_models")
        sd_models.FakeInitialModel = FakeInitialModel
        sd_models.model_data = types.SimpleNamespace(sd_model=Checkpoint())

        def unload_model_weights():
            calls.append("unload_model_weights")
            sd_models.model_data.sd_model = FakeInitialModel()

        def forge_model_reload():
            calls.append("forge_model_reload")
            sd_models.model_data.sd_model = Checkpoint()

        sd_models.unload_model_weights = unload_model_weights
        sd_models.forge_model_reload = forge_model_reload
        package = types.ModuleType("modules")
        package.__path__ = []
        package.sd_models = sd_models
        script = module.UnloadModelsReload()
        with sdg._modules({"modules": package, "modules.sd_models": sd_models}):
            script.before_process(types.SimpleNamespace(txt2img_upscale=True))   # checkpoint loaded: Forge's way
            self.assertEqual(calls, [])
            um.unload_forge_model(sd_models)                                    # the button
            script.before_process(types.SimpleNamespace())                       # Generate: process_images reloads
            self.assertEqual(calls, ["unload_model_weights"])
            script.before_process(types.SimpleNamespace(txt2img_upscale=True))   # ✨
        self.assertEqual(calls, ["unload_model_weights", "forge_model_reload"])
        self.assertIs(sd_models.model_data.sd_model.is_wan, False, "process_images_inner's first read works")

    def test_negpip_stays_the_last_script(self):
        names = sorted(name for name in os.listdir(ROOT / "scripts") if name.endswith(".py"))
        self.assertLess(names.index("layout_unload_models.py"), names.index("negpip.py"))
        self.assertEqual(names[-1], "negpip.py")

    def test_the_page_script_loads_after_the_save_button_script(self):
        names = sorted(os.listdir(ROOT / "javascript"))
        self.assertLess(names.index("notebook_save_defaults.js"), names.index("notebook_unload_models.js"))


if __name__ == "__main__":
    unittest.main()
