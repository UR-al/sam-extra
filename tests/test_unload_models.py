"""sam3ext/unload_models.py + sam3ext/unload_registry.py — '모델 내리기' 버튼의 순서·거절·결과 글, 그리고 기능마다의 해제 훅.

Forge·CUDA 는 가짜다(Host 의 함수들, 가짜 torch.cuda). 기능 훅은 실제 모듈의 함수를 작은 가짜 모델로 부른다 — Anima 3.8B
런타임은 tests/test_anima38.py 의 _load_lifecycle_runtime(Forge backend 스텁), 두 스크립트(anima_vae_2x · anima_safe_pag)는
modules 스텁으로 읽어 따로 만든 목록에 등록하게 한다.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import unload_models as um  # noqa: E402
from sam3ext import unload_registry as ur  # noqa: E402

GB = 1024 ** 3


class FakeHost:
    """Forge 대신 — 부른 순서를 기록한다."""

    def __init__(self, *, locked=False, busy=False, forge=True, hooks=(), memory=None, collect_error=None,
                 forge_error=None):
        self.calls: list[str] = []
        self.locked = locked
        self.is_busy = busy
        self.forge = forge
        self.forge_error = forge_error
        self.hook_list = list(hooks)
        self.memory_values = list(memory) if memory is not None else None
        self.collect_error = collect_error

    def try_lock(self):
        self.calls.append("try_lock")
        return not self.locked

    def unlock(self):
        self.calls.append("unlock")

    def busy(self):
        self.calls.append("busy")
        return self.is_busy

    def unload_forge(self):
        self.calls.append("forge")
        if self.forge_error is not None:
            raise self.forge_error
        return self.forge

    def hooks(self):
        return self.hook_list

    def collect(self):
        self.calls.append("collect")
        if self.collect_error is not None:
            raise self.collect_error

    def memory(self):
        self.calls.append("memory")
        if self.memory_values is None:
            return None
        value = self.memory_values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def host(self):
        return um.Host(try_lock=self.try_lock, unlock=self.unlock, busy=self.busy, unload_forge=self.unload_forge,
                       hooks=self.hooks, collect=self.collect, memory=self.memory)


def hook(host, key, label, result=True, error=None):
    def release():
        host.calls.append(key)
        if error is not None:
            raise error
        return result

    return ur.UnloadHook(key, label, release)


def gib(value):
    return int(value * GB)


BEFORE = um.Memory(used=gib(19.6), total=gib(24.0), reserved=gib(15.2))
AFTER = um.Memory(used=gib(3.1), total=gib(24.0), reserved=gib(0.0))


class RefusalTests(unittest.TestCase):
    def test_a_generation_holding_forges_queue_lock_is_refused_and_nothing_is_touched(self):
        fake = FakeHost(locked=True, memory=[BEFORE, AFTER])
        fake.hook_list = [hook(fake, "sam3", "SAM3")]
        outcome = um.run(fake.host())
        self.assertEqual(outcome.refused, "생성 중에는 내릴 수 없습니다")
        self.assertEqual(fake.calls, ["try_lock"], "no unlock of a lock it never took, no unload, no hook")
        self.assertEqual(um.summary(outcome), "생성 중에는 내릴 수 없습니다")
        self.assertFalse(outcome.ok)

    def test_a_job_in_shared_state_is_refused_and_the_lock_given_back(self):
        fake = FakeHost(busy=True, memory=[BEFORE, AFTER])
        fake.hook_list = [hook(fake, "sam3", "SAM3")]
        outcome = um.run(fake.host())
        self.assertEqual(outcome.refused, um.BUSY_MESSAGE)
        self.assertEqual(fake.calls, ["try_lock", "busy", "unlock"])
        payload = um.payload("req-1", outcome)
        self.assertEqual((payload["ok"], payload["refused"], payload["text"]), (False, True, um.BUSY_MESSAGE))
        self.assertEqual(payload["detail"], "생성 중에는 내릴 수 없습니다 — 생성이 끝난 뒤 다시 누르세요.")

    def test_a_click_while_another_unload_runs_is_refused(self):
        fake = FakeHost(memory=[BEFORE, AFTER])
        with um._RUNNING:
            outcome = um.run(fake.host())
        self.assertEqual(outcome.refused, "이미 모델을 내리는 중입니다")
        self.assertEqual(fake.calls, [])
        self.assertIsNone(um.run(FakeHost(memory=[BEFORE, AFTER]).host()).refused, "the guard is released")

    def test_shared_state_busy_reads_job_and_job_count_not_interrupted(self):
        state = types.SimpleNamespace
        self.assertTrue(um.state_busy(state(job="txt2img", job_count=1, interrupted=False)))
        self.assertTrue(um.state_busy(state(job="", job_count=-1, interrupted=False)), "begin() sets -1")
        self.assertTrue(um.state_busy(state(job="task(abc)", job_count=0, interrupted=True)), "stopping job")
        # Interrupt pressed while idle stays True until the next begin(): not a running job
        self.assertFalse(um.state_busy(state(job="", job_count=0, interrupted=True)))
        self.assertFalse(um.state_busy(state(job="", job_count=0)))
        self.assertFalse(um.state_busy(None))
        self.assertFalse(um.state_busy(state(job=None, job_count=None)))


class OrderTests(unittest.TestCase):
    def test_forge_once_then_every_hook_then_the_cache_then_the_numbers(self):
        fake = FakeHost(memory=[BEFORE, AFTER])
        fake.hook_list = [hook(fake, "sam3", "SAM3"), hook(fake, "tipo", "TIPO", result=False),
                          hook(fake, "vae_degrid", "DeGrid")]
        outcome = um.run(fake.host())
        self.assertEqual(fake.calls, ["try_lock", "busy", "memory", "forge", "sam3", "tipo", "vae_degrid", "collect",
                                      "memory", "unlock"])
        self.assertEqual(fake.calls.count("forge"), 1)
        self.assertEqual([s.key for s in outcome.released], ["forge", "sam3", "vae_degrid"])
        self.assertEqual([s.key for s in outcome.idle], ["tipo"])
        self.assertTrue(outcome.ok)
        self.assertEqual(um.summary(outcome),
                         "VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3 · DeGrid) · 이 프로세스 예약 15.2 → 0.0 GB")

    def test_every_hook_runs_even_when_some_raise_and_the_failures_are_named(self):
        fake = FakeHost(memory=[BEFORE, AFTER], forge_error=RuntimeError("Forge is busy loading"))
        fake.hook_list = [hook(fake, "sam3", "SAM3", error=RuntimeError("boom")),
                          hook(fake, "anima38", "Anima 3.8B"),
                          hook(fake, "tipo", "TIPO", error=ValueError()),
                          hook(fake, "colorcraft", "Colorcraft")]
        outcome = um.run(fake.host())
        self.assertEqual(fake.calls, ["try_lock", "busy", "memory", "forge", "sam3", "anima38", "tipo", "colorcraft",
                                      "collect", "memory", "unlock"])
        self.assertFalse(outcome.ok)
        self.assertEqual([(f.key, f.label, f.error) for f in outcome.errors], [
            ("forge", "Forge 모델", "RuntimeError: Forge is busy loading"),
            ("sam3", "SAM3", "RuntimeError: boom"),
            ("tipo", "TIPO", "ValueError"),
        ])
        self.assertIn("Traceback", outcome.errors[1].trace)
        self.assertEqual(um.summary(outcome),
                         "VRAM 사용 19.6 GB → 3.1 GB (Anima 3.8B · Colorcraft) · 이 프로세스 예약 15.2 → 0.0 GB"
                         " · 실패: Forge 모델, SAM3, TIPO")
        detail = um.detail(outcome)
        self.assertIn("실패 — SAM3: RuntimeError: boom", detail)
        self.assertIn("실패 — Forge 모델: RuntimeError: Forge is busy loading", detail)
        payload = um.payload("req-2", outcome)
        self.assertEqual(payload["errors"][1], {"key": "sam3", "label": "SAM3", "error": "RuntimeError: boom"})
        self.assertEqual(payload["released"], ["anima38", "colorcraft"])
        self.assertEqual(fake.calls[-1], "unlock", "the lock is given back after failures too")

    def test_a_failing_cache_flush_is_named_and_the_numbers_still_come(self):
        fake = FakeHost(memory=[BEFORE, AFTER], collect_error=RuntimeError("CUDA error"))
        outcome = um.run(fake.host())
        self.assertEqual(outcome.after, AFTER)
        self.assertTrue(um.summary(outcome).endswith(" · 실패: CUDA 캐시 비우기"))

    def test_a_label_shared_by_several_hooks_is_shown_once(self):
        fake = FakeHost(memory=[BEFORE, AFTER], forge=False)
        fake.hook_list = [hook(fake, "speed_dct", "샘플링 캐시"), hook(fake, "sam3", "SAM3"),
                          hook(fake, "hiflow_mask", "샘플링 캐시")]
        outcome = um.run(fake.host())
        self.assertIn("(샘플링 캐시 · SAM3)", um.summary(outcome))
        self.assertIn("내림: 샘플링 캐시, SAM3", um.detail(outcome))
        self.assertIn("쥔 것이 없던 것: Forge 모델", um.detail(outcome))

    def test_nothing_held(self):
        fake = FakeHost(memory=[AFTER, AFTER], forge=False)
        fake.hook_list = [hook(fake, "sam3", "SAM3", result=None), hook(fake, "tipo", "TIPO", result=False)]
        outcome = um.run(fake.host())
        self.assertEqual(um.summary(outcome), "VRAM 사용 3.1 GB → 3.1 GB (내릴 모델 없음) · 이 프로세스 예약 0.0 → 0.0 GB")
        self.assertTrue(outcome.ok)

    def test_a_hook_list_changed_by_a_hook_does_not_skip_or_repeat_hooks(self):
        fake = FakeHost(memory=[BEFORE, AFTER])
        later = hook(fake, "late", "Late")

        def register_more():
            fake.calls.append("first")
            fake.hook_list.append(later)
            return True

        fake.hook_list = [ur.UnloadHook("first", "First", register_more), hook(fake, "second", "Second")]
        um.run(fake.host())
        self.assertEqual([c for c in fake.calls if c in ("first", "second", "late")], ["first", "second"])


class TextTests(unittest.TestCase):
    def outcome(self, **kwargs):
        defaults = dict(released=[um.Step("forge", "Forge 모델"), um.Step("sam3", "SAM3")], before=BEFORE, after=AFTER)
        defaults.update(kwargs)
        return um.Outcome(**defaults)

    def test_the_one_line_result(self):
        self.assertEqual(um.summary(self.outcome()),
                         "VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3) · 이 프로세스 예약 15.2 → 0.0 GB")

    def test_the_tooltip_explains_the_numbers(self):
        self.assertEqual(um.detail(self.outcome(idle=[um.Step("tipo", "TIPO")])), "\n".join([
            "GPU 전체 사용: 19.6 GB → 3.1 GB (24.0 GB 중 — 같은 GPU 의 다른 프로그램 몫 포함)",
            "이 Forge 프로세스의 PyTorch 예약: 15.2 GB → 0.0 GB",
            "내림: Forge 모델, SAM3",
            "쥔 것이 없던 것: TIPO",
            "다음 생성 때 고른 체크포인트(VAE / Text Encoder · Low Bits 그대로)를 다시 읽습니다.",
        ]))

    def test_cpu_only(self):
        outcome = self.outcome(before=None, after=None)
        self.assertEqual(um.summary(outcome), "모델 내림 (Forge 모델 · SAM3) · CUDA GPU 가 없어 VRAM 은 재지 않았습니다")
        self.assertTrue(um.detail(outcome).startswith("CUDA GPU 가 없어 VRAM 은 재지 않았습니다\n내림: Forge 모델, SAM3"))

    def test_a_failed_measurement(self):
        fake = FakeHost(memory=[RuntimeError("no context"), RuntimeError("still none")])
        outcome = um.run(fake.host())
        self.assertEqual(um.summary(outcome),
                         "모델 내림 (Forge 모델) · VRAM 을 재지 못했습니다 · 실패: VRAM 재기")
        self.assertEqual(len(outcome.errors), 1, "one entry for both measurements")

    def test_payload_json(self):
        payload = um.payload("um-1_x", self.outcome())
        text = um.to_json(payload)
        self.assertEqual(json.loads(text), payload)
        self.assertIn("VRAM 사용", text, "not ascii-escaped")
        self.assertEqual(payload["request"], "um-1_x")
        self.assertEqual(payload["before"], {"used": BEFORE.used, "total": BEFORE.total, "reserved": BEFORE.reserved})
        self.assertEqual((payload["ok"], payload["refused"], payload["released"]), (True, False, ["forge", "sam3"]))
        self.assertEqual(um.payload("<script>", self.outcome())["request"], "", "only a plain request id comes back")
        self.assertIsNone(um.payload(None, self.outcome(before=None, after=None))["before"])

    def test_error_payload(self):
        payload = um.error_payload("abc", RuntimeError("Forge is not ready"))
        self.assertEqual((payload["ok"], payload["request"], payload["text"]),
                         (False, "abc", "모델 내리기 실패: RuntimeError: Forge is not ready"))


class FakeCuda:
    def __init__(self, available, calls):
        self.available = available
        self.calls = calls

    def is_available(self):
        return self.available

    def synchronize(self):
        self.calls.append("synchronize")

    def empty_cache(self):
        self.calls.append("empty_cache")

    def ipc_collect(self):
        self.calls.append("ipc_collect")

    def mem_get_info(self, device=None):
        self.calls.append(("mem_get_info", device))
        return gib(4.0), gib(24.0)

    def memory_reserved(self, device=None):
        self.calls.append(("memory_reserved", device))
        return gib(1.5)


class ForgeFactsTests(unittest.TestCase):
    def test_collect_frees_python_then_cuda(self):
        calls = []
        gc_module = types.SimpleNamespace(collect=lambda: calls.append("gc"))
        um.collect_garbage(gc_module, types.SimpleNamespace(cuda=FakeCuda(True, calls)))
        self.assertEqual(calls, ["gc", "synchronize", "empty_cache", "ipc_collect"])

    def test_collect_on_cpu_only_touches_no_cuda(self):
        calls = []
        um.collect_garbage(types.SimpleNamespace(collect=lambda: calls.append("gc")),
                           types.SimpleNamespace(cuda=FakeCuda(False, calls)))
        self.assertEqual(calls, ["gc"])
        # this test process hides the GPU (CUDA_VISIBLE_DEVICES=-1): the real torch takes the same path
        if not torch.cuda.is_available():
            um.collect_garbage(types.SimpleNamespace(collect=lambda: None), torch)
            self.assertIsNone(um.measure_cuda(torch))

    def test_measure_whole_gpu_used_and_this_process_reserved(self):
        calls = []
        memory = um.measure_cuda(types.SimpleNamespace(cuda=FakeCuda(True, calls)), "cuda:0")
        self.assertEqual(memory, um.Memory(used=gib(20.0), total=gib(24.0), reserved=gib(1.5)))
        self.assertEqual(calls, [("mem_get_info", "cuda:0"), ("memory_reserved", "cuda:0")])
        self.assertIsNone(um.measure_cuda(types.SimpleNamespace(cuda=FakeCuda(False, []))))

    def _sd_models(self, current, calls):
        class FakeInitialModel:
            pass

        module = types.SimpleNamespace(FakeInitialModel=FakeInitialModel, calls=calls)
        module.model_data = types.SimpleNamespace(sd_model=current(FakeInitialModel))

        def unload_model_weights(*args, **kwargs):
            calls.append("unload_model_weights")
            module.model_data.sd_model = FakeInitialModel()

        module.unload_model_weights = unload_model_weights
        return module

    def test_forges_unload_is_called_once_and_says_whether_a_model_was_there(self):
        calls = []
        loaded = self._sd_models(lambda fake: object(), calls)
        self.assertTrue(um.unload_forge_model(loaded, types.SimpleNamespace(current_loaded_models=[])))
        self.assertEqual(calls, ["unload_model_weights"])
        self.assertIsInstance(loaded.model_data.sd_model, loaded.FakeInitialModel)
        # nothing loaded yet (FakeInitialModel) and Forge's GPU list empty: still called (it also flushes), False
        calls.clear()
        empty = self._sd_models(lambda fake: fake(), calls)
        self.assertFalse(um.unload_forge_model(empty, types.SimpleNamespace(current_loaded_models=[])))
        self.assertEqual(calls, ["unload_model_weights"])
        # a model Forge still keeps on the GPU (e.g. a ControlNet) counts
        self.assertTrue(um.unload_forge_model(empty, types.SimpleNamespace(current_loaded_models=[object()])))
        self.assertTrue(um.unload_forge_model(self._sd_models(lambda fake: object(), []), None))


class RegistryTests(unittest.TestCase):
    def test_builtin_hooks_in_order(self):
        self.assertEqual([row[0] for row in ur.BUILTIN_HOOKS], [
            "sam3", "anima38", "anima_ipa", "vae_degrid", "tipo", "modulation", "colorcraft",
            "speed_dct", "hiflow_mask", "history_dct"])
        registry = ur.UnloadRegistry(ur.BUILTIN_HOOKS)
        self.assertEqual([(h.key, h.label) for h in registry.hooks()], [(row[0], row[1]) for row in ur.BUILTIN_HOOKS])

    def test_every_builtin_hook_names_a_module_level_function(self):
        for key, _label, module, attr in ur.BUILTIN_HOOKS:
            with self.subTest(key=key):
                parts = module.split(".")
                path = ROOT.joinpath(*parts).with_suffix(".py")
                if not path.exists():
                    path = ROOT.joinpath(*parts, "__init__.py")
                tree = ast.parse(path.read_text(encoding="utf-8"))
                names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
                self.assertIn(attr, names, f"{module}.{attr}")

    def test_a_module_not_imported_holds_nothing_and_is_not_imported(self):
        name = "sam3ext._never_imported_for_unload_test"
        release = ur.module_function(name, "release")
        self.assertIs(release(), False)
        self.assertNotIn(name, sys.modules)

    def test_an_imported_module_function_is_called_when_pressed(self):
        module = types.ModuleType("sam3ext._unload_test_module")
        module.release = mock.Mock(return_value=True)
        release = ur.module_function(module.__name__, "release")
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            self.assertTrue(release())
            module.release = mock.Mock(return_value=False)    # looked up at call time (a reloaded module)
            self.assertFalse(release())

    def test_an_old_module_without_the_function_says_restart(self):
        module = types.ModuleType("sam3ext._unload_test_old")
        release = ur.module_function(module.__name__, "release")
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaises(ur.StaleModuleError) as caught:
                release()
        self.assertIn("Forge 를 다시 시작하세요", str(caught.exception))

    def test_register_replaces_the_same_key_in_place_and_unregister(self):
        registry = ur.UnloadRegistry()
        first = lambda: True  # noqa: E731
        registry.register("a", "A", first)
        registry.register("b", "B", lambda: False)
        again = lambda: False  # noqa: E731
        registry.register("a", "A2", again)
        self.assertEqual([(h.key, h.label, h.release) for h in registry.hooks()][0], ("a", "A2", again))
        self.assertEqual([h.key for h in registry.hooks()], ["a", "b"])
        self.assertTrue(registry.unregister("a"))
        self.assertFalse(registry.unregister("a"))
        self.assertEqual([h.key for h in registry.hooks()], ["b"])
        for bad in (("", "X", first), ("k", "", first)):
            with self.assertRaises(ValueError):
                registry.register(*bad)
        with self.assertRaises(TypeError):
            registry.register("k", "K", None)

    def test_the_module_level_functions_use_the_shared_registry(self):
        fresh = ur.UnloadRegistry()
        with mock.patch.object(ur, "REGISTRY", fresh):
            ur.register("face", "얼굴 검출기", lambda: True)
            self.assertEqual([h.key for h in ur.hooks()], ["face"])
            self.assertTrue(ur.unregister("face"))
        self.assertIn("sam3", [h.key for h in ur.hooks()])


# ---- 기능 훅 ------------------------------------------------------------------------------------------------------

class Sam3HookTests(unittest.TestCase):
    def test_release_sam3_says_whether_a_bundle_was_dropped(self):
        from sam3ext import core

        dropped = []
        with mock.patch.object(core, "_discard_bundle", lambda *bundle: dropped.append(bundle)), \
                mock.patch.object(core, "_empty_cuda_cache", lambda: None):
            core.release_sam3()                                  # whatever an earlier test left
            dropped.clear()
            with core._BUNDLE_LOCK:
                core._BUNDLE, core._BUNDLE_KEY, core._BUNDLE_OFFLOADED = ("model", "processor"), ("k",), {1: "cuda"}
            self.assertIs(core.release_sam3(), True)
            self.assertEqual(dropped, [("model", "processor")])
            self.assertIsNone(core._BUNDLE)
            self.assertIsNone(core._BUNDLE_OFFLOADED, "the copy 'Unload after' keeps in RAM goes too")
            self.assertIs(core.release_sam3(), False)


class RuntimeHookTests(unittest.TestCase):
    def test_tipo_drops_the_model_from_ram(self):
        from sam3ext.tipo import runtime as tr

        loads = []

        def loader(directory):
            loads.append(directory)
            return torch.nn.Linear(2, 2), object()

        with tempfile.TemporaryDirectory() as tmp:
            runtime = tr.TipoRuntime(loader=loader, directory=tmp, cuda_info=lambda: (False, 0),
                                     forge_memory=lambda: None)
            self.assertFalse(runtime.unload())
            runtime._ensure_loaded()
            self.assertTrue(runtime.unload())
            self.assertIsNone(runtime._model)
            self.assertIsNone(runtime._tokenizer)
            self.assertFalse(runtime.unload())
            runtime._ensure_loaded()
            self.assertEqual(len(loads), 2, "the next use reads the files again")

    def test_tipo_shared_hook_does_not_create_a_runtime(self):
        from sam3ext.tipo import runtime as tr

        with mock.patch.object(tr, "_SHARED", None):
            self.assertFalse(tr.unload_shared_runtime())
            self.assertIsNone(tr._SHARED)
        fake = mock.Mock()
        fake.unload.return_value = True
        with mock.patch.object(tr, "_SHARED", fake):
            self.assertTrue(tr.unload_shared_runtime())
        fake.unload.assert_called_once_with()

    def test_degrid_drops_the_nafnet(self):
        from sam3ext import vae_degrid_runtime as vdr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nafnet.safetensors"
            path.write_bytes(b"x")
            loads = []

            def loader(p):
                loads.append(p)
                return torch.nn.Linear(2, 2)

            runtime = vdr.DegridRuntime(loader=loader, forge_memory=lambda: None, logger=lambda m: None)
            self.assertFalse(runtime.unload())
            runtime.model_for(path)
            self.assertIsNotNone(runtime.loaded_path)
            self.assertTrue(runtime.unload())
            self.assertIsNone(runtime._model)
            self.assertIsNone(runtime.loaded_path)
            self.assertFalse(runtime.unload())
            runtime.model_for(path)
            self.assertEqual(len(loads), 2)
        with mock.patch.object(vdr, "_SHARED", None):
            self.assertFalse(vdr.unload_shared_runtime())
            self.assertIsNone(vdr._SHARED)

    def test_ipa_drops_the_adapter_and_siglip(self):
        from sam3ext.anima_ipa import runtime as ir

        loads = []
        runtime = ir.IpaRuntime(models_dir=Path("nowhere"), loader=lambda d: loads.append(d) or "loaded")
        self.assertFalse(runtime.unload())
        runtime.load()
        self.assertTrue(runtime.unload())
        self.assertFalse(runtime.unload())
        runtime.load()
        self.assertEqual(len(loads), 2)
        with mock.patch.object(ir, "_SHARED", None):
            self.assertFalse(ir.unload_shared_runtime())
            self.assertIsNone(ir._SHARED)


class Anima38HookTests(unittest.TestCase):
    def setUp(self):
        from tests import test_anima38 as anima38_tests

        self.module = anima38_tests._load_lifecycle_runtime()
        self.unloaded = []
        self.loaded = self.module.memory_management.current_loaded_models = []

    def _loaded(self, patcher):
        entry = types.SimpleNamespace(model=patcher, model_unload=lambda: self.unloaded.append(patcher))
        self.loaded.append(entry)

    def _filled(self):
        runtime = self.module.Anima3BRuntime()
        qwen_patcher, connector = object(), object()
        runtime._qwen, runtime._tokenizer, runtime._qwen_path = object(), object(), "qwen35_4b.safetensors"
        runtime._qwen_clip = types.SimpleNamespace(patcher=qwen_patcher)
        runtime._adapter, runtime._adapter_key = object(), ("adapter", 1)
        runtime._v2_models, runtime._v2_key = object(), ("bundle", 2)
        runtime._v2_connector_patcher = connector
        run = types.SimpleNamespace(device_copies={("cuda", torch.float32): torch.zeros(1)})
        runtime._v2_runs[1] = run
        runtime._v2_device_runs[id(run)] = run
        runtime._v2_run_counter = 3
        runtime._semantic_cache[("q", "line", torch.float32)] = (torch.zeros(1),)
        runtime._native_cache[("te", "line")] = (torch.zeros(1),)
        self._loaded(connector)
        self._loaded(qwen_patcher)
        return runtime, run, (connector, qwen_patcher)

    def test_everything_it_holds_is_dropped(self):
        runtime, run, patchers = self._filled()
        self.assertTrue(runtime.unload_models())
        for name in ("_qwen", "_qwen_clip", "_tokenizer", "_qwen_path", "_adapter", "_adapter_key", "_v2_models",
                     "_v2_key", "_v2_source", "_v2_connector_patcher", "_v2_sampling_patcher"):
            self.assertIsNone(getattr(runtime, name), name)
        self.assertEqual((runtime._v2_runs, dict(runtime._v2_device_runs), run.device_copies), ({}, {}, {}))
        self.assertEqual((len(runtime._semantic_cache), len(runtime._native_cache)), (0, 0))
        self.assertEqual(runtime._v2_run_counter, 0)
        self.assertEqual(self.loaded, [], "its patchers left Forge's list")
        self.assertEqual(sorted(map(id, self.unloaded)), sorted(map(id, patchers)))
        self.assertFalse(runtime.unload_models(), "nothing left")

    def test_a_stale_install_is_restored_first_and_a_failing_restore_still_drops_everything(self):
        runtime, _run, _patchers = self._filled()
        stale = object()
        runtime._installed_processing = stale
        with mock.patch.object(runtime, "restore", side_effect=RuntimeError("restore failed")) as restore:
            with self.assertRaises(RuntimeError):
                runtime.unload_models()
        restore.assert_called_once_with(stale)
        self.assertIsNone(runtime._qwen)
        self.assertIsNone(runtime._installed_processing)
        self.assertIsNone(runtime._v2_models)
        self.assertEqual(self.loaded, [], "its patchers still left Forge's list")

    def test_the_shared_hook_does_not_create_a_runtime(self):
        self.assertIsNone(self.module._SHARED_RUNTIME)
        self.assertFalse(self.module.unload_shared_runtime())
        self.assertIsNone(self.module._SHARED_RUNTIME)
        runtime, _run, _patchers = self._filled()
        self.module._SHARED_RUNTIME = runtime
        self.assertTrue(self.module.unload_shared_runtime())


class CacheHookTests(unittest.TestCase):
    def test_modulation(self):
        from sam3ext.guidance import modulation

        modulation.clear_modulation_caches()
        self.assertFalse(modulation.clear_modulation_caches())
        modulation._POOLED_CACHE[("k",)] = torch.zeros(1)
        self.assertTrue(modulation.clear_modulation_caches())
        self.assertEqual(len(modulation._POOLED_CACHE), 0)

    def test_colorcraft(self):
        from sam3ext.colorcraft import hook as cc_hook

        cc_hook.release_caches()
        self.assertFalse(cc_hook.release_caches())
        cc_hook._BASIS_CACHE["anima"] = {"v": torch.zeros(2)}
        cc_hook._BASIS_DEVICE_CACHE[("anima", "cuda", torch.float32)] = {"v": torch.zeros(2)}
        cc_hook._ANCHOR_CACHE[("k",)] = (None, torch.zeros(1))
        self.assertTrue(cc_hook.release_caches())
        self.assertEqual((cc_hook._BASIS_CACHE, cc_hook._BASIS_DEVICE_CACHE, len(cc_hook._ANCHOR_CACHE)), ({}, {}, 0))

    def test_small_sampling_matrices(self):
        from sam3ext.guidance import hiflow, history
        from sam3ext.speed import spectral

        for build, clear in ((lambda: spectral.dct_basis(4), spectral.clear_basis_cache),
                             (lambda: hiflow.butterworth_mask(4, 4, 0.5), hiflow.clear_mask_cache),
                             (lambda: history._dct_matrix(4, "cpu", torch.float32), history.clear_dct_cache)):
            with self.subTest(clear=clear.__module__):
                clear()
                self.assertFalse(clear())
                build()
                self.assertTrue(clear())
                self.assertFalse(clear())


@contextlib.contextmanager
def _modules_stub(callbacks=None):
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    modules_stub.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules_stub.script_callbacks = callbacks or types.SimpleNamespace(on_before_ui=lambda fn: None,
                                                                       on_cfg_denoiser=lambda fn: None)
    with mock.patch.dict(sys.modules, {"modules": modules_stub}):
        yield


def _load_script(filename, registry):
    with _modules_stub(), mock.patch.object(ur, "REGISTRY", registry):
        spec = importlib.util.spec_from_file_location(f"_test_unload_{Path(filename).stem}", ROOT / "scripts" / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ScriptHookTests(unittest.TestCase):
    def test_vae_2x_registers_its_decoder_cache(self):
        registry = ur.UnloadRegistry()
        module = _load_script("anima_vae_2x.py", registry)
        (registered,) = registry.hooks()
        self.assertEqual((registered.key, registered.label), ("anima_vae_2x", "VAE 2x 디코더"))
        self.assertIs(registered.release, module.release_decoders)
        self.assertFalse(module.release_decoders())
        module._DECODER_CACHE[("not-12ch",)] = None
        self.assertFalse(module.release_decoders(), "a remembered 'not a 2x VAE' is not a decoder")
        module._DECODER_CACHE[("a.safetensors", "cuda:0", "torch.bfloat16")] = object()
        self.assertTrue(module.release_decoders())
        self.assertEqual(module._DECODER_CACHE, {})

    def test_safe_pag_registers_its_sampling_leftovers(self):
        registry = ur.UnloadRegistry()
        module = _load_script("anima_safe_pag.py", registry)
        (registered,) = registry.hooks()
        self.assertEqual((registered.key, registered.label), ("anima_safe_pag", "샘플링 캐시"))
        self.assertIs(registered.release, module.release_sampling_caches)
        self.assertFalse(module.release_sampling_caches())
        module._index_tensor([0, 1], "cpu")
        module._RUNTIME.smc_prev = torch.zeros(1)
        module._RUNTIME.cns_x_t = torch.zeros(1)
        self.assertTrue(module.release_sampling_caches())
        self.assertEqual(module._INDEX_TENSOR_CACHE, {})
        self.assertIsNone(module._RUNTIME.smc_prev)
        self.assertIsNone(module._RUNTIME.cns_x_t)
        self.assertFalse(module.release_sampling_caches())


class ThreadedLockTests(unittest.TestCase):
    """A real lock held by another thread (a generation): the click is refused at once, it does not wait."""

    def test_refused_without_waiting(self):
        lock = threading.Lock()
        fake = FakeHost(memory=[BEFORE, AFTER])
        host = um.Host(try_lock=lambda: lock.acquire(blocking=False), unlock=lock.release, busy=lambda: False,
                       unload_forge=fake.unload_forge, hooks=lambda: [], collect=fake.collect, memory=fake.memory)
        started, finish = threading.Event(), threading.Event()

        def generation():
            with lock:
                started.set()
                finish.wait(5)

        worker = threading.Thread(target=generation)
        worker.start()
        try:
            self.assertTrue(started.wait(5))
            self.assertEqual(um.run(host).refused, um.BUSY_MESSAGE)
            self.assertNotIn("forge", fake.calls)
        finally:
            finish.set()
            worker.join(5)
        self.assertIsNone(um.run(host).refused)
        self.assertFalse(lock.locked(), "given back after the unload")


if __name__ == "__main__":
    unittest.main()
