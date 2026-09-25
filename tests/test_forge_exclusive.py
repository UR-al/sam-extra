"""forge_exclusive — Refine·Tile-Repair·PiD 버튼이 Forge Generate 와 같은 규칙(잠금·job·Stop)으로 도는지.

감사 H1: 세 핸들러가 queue_lock·state.begin/end 밖에서 돌아 Stop 한 번 뒤 다음 실행이 즉시 끝나고, Generate 와
동시에 돌며, Stop 이 진행 중인 txt2img 를 대신 중단시키고, Tile-Repair 는 Stop 이 아무것도 멈추지 못했다.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import sys
import threading
import types
import unittest
from threading import Lock
from unittest import mock

import gradio as gr
import torch
from PIL import Image

from sam3ext import anima_core
from sam3ext import forge_exclusive as fx
from sam3ext import ui_anima
from sam3ext.ui_anima import ANIMA_ARG_KEYS, build_anima_panel


class _FakeState:
    """modules/shared_state.py:121-146 의 begin/end 만 흉내 낸다."""

    def __init__(self):
        self.job = ""
        self.job_count = 0
        self.interrupted = False
        self.skipped = False
        self.stopping_generation = False
        self.textinfo = None
        self.events: list = []

    def begin(self, job="(unknown)"):
        self.events.append(("begin", job))
        self.job = job
        self.job_count = -1
        self.interrupted = False
        self.skipped = False
        self.stopping_generation = False
        self.textinfo = None

    def end(self):
        self.events.append("end")
        self.job = ""
        self.job_count = 0


class _FakeMainThread:
    """modules_forge/main_thread.py 처럼 다른 스레드에서 돌리고 예외는 삼킨다(None 반환)."""

    def __init__(self):
        self.calls = 0
        self.thread_idents: list[int] = []

    def run_and_wait_result(self, func, *args, **kwargs):
        self.calls += 1
        box: dict = {}

        def work():
            self.thread_idents.append(threading.get_ident())
            try:
                box["result"] = func(*args, **kwargs)
            except Exception:
                box["result"] = None

        worker = threading.Thread(target=work)
        worker.start()
        worker.join()
        return box.get("result")


def _fake_modules(state, *, main_thread=None, lock=None):
    package = types.ModuleType("modules")
    package.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.state = state
    call_queue = types.ModuleType("modules.call_queue")
    call_queue.queue_lock = lock if lock is not None else Lock()
    package.shared = shared
    package.call_queue = call_queue
    mods: dict = {
        "modules": package,
        "modules.shared": shared,
        "modules.call_queue": call_queue,
    }
    if main_thread is None:
        # None 을 넣으면 import 가 ImportError 를 낸다 — Forge 밖에서 도는 경우.
        mods["modules_forge"] = None
        mods["modules_forge.main_thread"] = None
    else:
        forge = types.ModuleType("modules_forge")
        forge.__path__ = []
        mt = types.ModuleType("modules_forge.main_thread")
        mt.run_and_wait_result = main_thread.run_and_wait_result
        forge.main_thread = mt
        mods["modules_forge"] = forge
        mods["modules_forge.main_thread"] = mt
    return mods


def _in_worker(fn):
    """Forge 메인 스레드 우회 규칙 때문에 Gradio 워커처럼 다른 스레드에서 부른다."""
    box: dict = {}

    def work():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 — 테스트 스레드로 그대로 옮긴다
            box["error"] = exc

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


class RunExclusiveTests(unittest.TestCase):
    def test_run_starts_a_job_holds_the_lock_and_clears_stop_flags_after(self):
        state = _FakeState()
        lock = Lock()
        seen: dict = {}

        def fn():
            seen["locked"] = lock.locked()
            seen["job"] = state.job
            state.interrupted = True   # 도중에 ⏹ Stop
            state.skipped = True
            return "ok"

        with mock.patch.dict(sys.modules, _fake_modules(state, lock=lock)):
            result = fx.run_exclusive("sam3_refine", fn)

        self.assertEqual(result, "ok")
        self.assertEqual(seen, {"locked": True, "job": "sam3_refine"})
        self.assertEqual(state.events, [("begin", "sam3_refine"), "end"])
        self.assertFalse(lock.locked())
        # call_queue.wrap_gradio_call 의 finally 와 같이 Stop 잔상이 남지 않는다.
        self.assertFalse(state.interrupted)
        self.assertFalse(state.skipped)
        self.assertEqual(state.job, "")

    def test_a_previous_stop_does_not_end_the_next_run_immediately(self):
        # H1(1): Stop 뒤 interrupted 가 남아 다음 Refine 이 첫 검사에서 바로 끝나던 문제.
        state = _FakeState()
        state.interrupted = True
        state.skipped = True
        seen: dict = {}

        def fn():
            seen["interrupted"] = fx.is_interrupted()

        with mock.patch.dict(sys.modules, _fake_modules(state)):
            fx.run_exclusive("sam3_tile_repair", fn)
        self.assertFalse(seen["interrupted"])

    def test_the_job_ends_and_the_lock_opens_even_when_fn_raises(self):
        state = _FakeState()
        lock = Lock()

        def boom():
            raise RuntimeError("x")

        with mock.patch.dict(sys.modules, _fake_modules(state, lock=lock)):
            with self.assertRaises(RuntimeError):
                fx.run_exclusive("sam3_refine", boom)
        self.assertEqual(state.events, [("begin", "sam3_refine"), "end"])
        self.assertFalse(lock.locked())

    def test_main_thread_runs_fn_there_and_hands_back_the_result(self):
        state = _FakeState()
        main = _FakeMainThread()
        seen: dict = {}

        def fn():
            seen["ident"] = threading.get_ident()
            return 42

        def click():
            seen["caller"] = threading.get_ident()
            return fx.run_exclusive("sam3_refine", fn, on_main_thread=True)

        with mock.patch.dict(sys.modules, _fake_modules(state, main_thread=main)):
            result = _in_worker(click)
        self.assertEqual(result, 42)
        self.assertEqual(main.calls, 1)
        self.assertEqual(seen["ident"], main.thread_idents[0])
        self.assertNotEqual(seen["ident"], seen["caller"])
        self.assertEqual(state.events, [("begin", "sam3_refine"), "end"])

    def test_main_thread_exception_is_not_swallowed(self):
        # main_thread.Task.work 는 예외를 삼키고 None 을 돌려준다 — 핸들러의 빨간 상태줄이 사라지면 안 된다.
        state = _FakeState()
        main = _FakeMainThread()

        def boom():
            raise ValueError("inside")

        def click():
            return fx.run_exclusive("sam3_refine", boom, on_main_thread=True)

        with mock.patch.dict(sys.modules, _fake_modules(state, main_thread=main)):
            with self.assertRaises(ValueError):
                _in_worker(click)
        self.assertEqual(state.events, [("begin", "sam3_refine"), "end"])

    def test_main_thread_call_keeps_the_callers_context(self):
        # Gradio Progress(track_tqdm=True) 는 ContextVar(LocalContext) 로 이벤트를 찾는다.
        state = _FakeState()
        main = _FakeMainThread()
        var: contextvars.ContextVar = contextvars.ContextVar("gradio_progress", default=None)
        seen: dict = {}

        def fn():
            seen["value"] = var.get()

        def click():
            var.set("progress-object")
            fx.run_exclusive("sam3_refine", fn, on_main_thread=True)

        with mock.patch.dict(sys.modules, _fake_modules(state, main_thread=main)):
            _in_worker(click)
        self.assertEqual(seen["value"], "progress-object")

    def test_main_thread_is_bypassed_from_the_main_thread_itself(self):
        # Forge 는 Python 메인 스레드로 main_thread.loop() 를 돈다 — 그 안에서 큐에 넣으면 자기 자신을 기다린다.
        state = _FakeState()
        main = _FakeMainThread()
        self.assertIs(threading.current_thread(), threading.main_thread())
        with mock.patch.dict(sys.modules, _fake_modules(state, main_thread=main)):
            result = fx.run_exclusive("sam3_refine", lambda: "direct", on_main_thread=True)
        self.assertEqual(result, "direct")
        self.assertEqual(main.calls, 0)

    def test_main_thread_is_optional_outside_forge(self):
        state = _FakeState()
        with mock.patch.dict(sys.modules, _fake_modules(state, main_thread=None)):
            result = _in_worker(
                lambda: fx.run_exclusive("sam3_refine", lambda: "plain", on_main_thread=True)
            )
        self.assertEqual(result, "plain")

    def test_nested_run_stops_at_queue_lock_before_the_main_thread_hop(self):
        # 검토 지적: 예전 ``_local.inside`` 재진입 우회는 도달할 수 없었다. 중첩 run_exclusive 는
        # 비재진입 queue_lock(modules/fifo_lock.FIFOLock) 에서 먼저 막히고, Forge 의 메인 스레드 작업은
        # 원래 Python 메인 스레드라 current_thread() 검사로 이미 바로 실행된다.
        state = _FakeState()
        main = _FakeMainThread()
        lock = _NonReentrantLock()
        seen: dict = {}

        def inner():
            seen["inner_ran"] = True

        def outer():
            try:
                fx.run_exclusive("sam3_refine", inner, on_main_thread=True)
            except _WouldDeadlock:
                seen["blocked_at_lock"] = True

        with mock.patch.dict(sys.modules, _fake_modules(state, main_thread=main, lock=lock)):
            _in_worker(lambda: fx.run_exclusive("sam3_refine", outer, on_main_thread=True))
        self.assertEqual(seen, {"blocked_at_lock": True})
        self.assertEqual(main.calls, 1, "중첩 호출은 메인 스레드 큐에 닿지 못한다")
        self.assertFalse(hasattr(fx, "_local"), "도달 불가한 재진입 우회는 지웠다")


class _WouldDeadlock(RuntimeError):
    pass


class _NonReentrantLock:
    """FIFOLock 처럼 재진입이 안 된다 — 이미 잡혀 있으면 기다리는 대신 알린다."""

    def __init__(self):
        self._held = False

    def __enter__(self):
        if self._held:
            raise _WouldDeadlock("queue_lock is not reentrant")
        self._held = True
        return self

    def __exit__(self, *exc):
        self._held = False
        return False


class StopIfJobTests(unittest.TestCase):
    def _state(self, job):
        state = _FakeState()
        state.job = job
        return state

    def test_stop_only_when_our_job_holds_the_state(self):
        for job in ("sam3_refine", "SAM3 Refine pass 1/2"):
            state = self._state(job)
            with mock.patch.dict(sys.modules, _fake_modules(state)):
                self.assertTrue(fx.stop_if_job(("sam3_refine", "SAM3 Refine")), job)
            self.assertTrue(state.interrupted and state.skipped, job)

    def test_stop_leaves_txt2img_alone(self):
        # Generate 가 잠금을 쥔 채 우리 클릭이 기다리는 동안 Stop 을 누른 경우.
        for job in ("task(abc123)", "", "sam3_tile_repair"):
            state = self._state(job)
            with mock.patch.dict(sys.modules, _fake_modules(state)):
                self.assertFalse(fx.stop_if_job(("sam3_refine", "SAM3 Refine")), job)
            self.assertFalse(state.interrupted or state.skipped, job)

    def test_is_interrupted_is_false_outside_forge(self):
        with mock.patch.dict(sys.modules, {"modules": None, "modules.shared": None}):
            self.assertFalse(fx.is_interrupted())
        state = self._state("sam3_refine")
        state.skipped = True
        with mock.patch.dict(sys.modules, _fake_modules(state)):
            self.assertTrue(fx.is_interrupted())


class _FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, *args, **kwargs):
        self.calls += 1
        return x


def _fake_body(args, anima, context, context_null, device, seed):
    latents = torch.zeros(1)
    for _ in range(3):
        latents = anima(latents, None, None, padding_mask=None)
    return latents


class TileRepairInterruptTests(unittest.TestCase):
    """H1(4): 벤더 ami.generate 루프는 shared.state 를 보지 않아 Stop 이 아무것도 멈추지 못했다."""

    def _ami(self):
        return types.SimpleNamespace(generate_body=_fake_body)

    def test_stop_raises_interrupted_error_at_the_next_denoising_step(self):
        ami = self._ami()
        state = _FakeState()
        model = _FakeDiT()
        with mock.patch.dict(sys.modules, _fake_modules(state)):
            with anima_core._vendor_interrupt_guard(ami):
                self.assertIsNot(ami.generate_body, _fake_body)
                state.interrupted = True
                with self.assertRaises(InterruptedError):
                    ami.generate_body(None, model, {}, None, "cpu", 0)
        self.assertEqual(model.calls, 0, "첫 스텝 전에 멈춘다")
        # 가드가 끝나면 벤더 함수와 모델은 원래대로다.
        self.assertIs(ami.generate_body, _fake_body)
        state.interrupted = True
        with mock.patch.dict(sys.modules, _fake_modules(state)):
            ami.generate_body(None, model, {}, None, "cpu", 0)
        self.assertEqual(model.calls, 3)

    def test_without_stop_the_vendor_loop_runs_to_the_end(self):
        ami = self._ami()
        state = _FakeState()
        model = _FakeDiT()
        with mock.patch.dict(sys.modules, _fake_modules(state)):
            with anima_core._vendor_interrupt_guard(ami):
                ami.generate_body(None, model, {}, None, "cpu", 0)
        self.assertEqual(model.calls, 3)
        self.assertFalse(model._forward_pre_hooks, "훅이 남으면 다음 실행에서 또 검사한다")


def _fake_ami(model, *, fail=None):
    """벤더 ``ami.generate`` 처럼 모듈 전역 ``generate_body`` 를 호출 시점에 찾는다."""
    ami = types.ModuleType("anima_minimal_inference")
    ami.generate_body = _fake_body
    ami.generate_calls = 0

    def generate(args, gen_settings):
        ami.generate_calls += 1
        if fail is not None:
            raise fail
        return ami.generate_body(args, model, {}, None, "cpu", 0)

    ami.generate = generate
    ami.get_generation_settings = lambda args: {"steps": 3}
    ami.decode_latent = lambda vae, latent, device: torch.full((1, 3, 8, 8), 0.5)
    return ami


class GenerateWithStopTests(unittest.TestCase):
    """후속: run_tile_repair 가 쓰는 ``_generate_with_stop`` — 가드와 InterruptedError→None 을 한곳에서."""

    def test_stop_returns_none_before_the_first_step_and_restores_the_vendor(self):
        model = _FakeDiT()
        ami = _fake_ami(model)
        state = _FakeState()
        state.interrupted = True
        with mock.patch.dict(sys.modules, _fake_modules(state)):
            self.assertIsNone(anima_core._generate_with_stop(ami, None, {}))
        self.assertEqual(model.calls, 0)
        self.assertIs(ami.generate_body, _fake_body)
        self.assertFalse(model._forward_pre_hooks)

    def test_without_stop_the_latent_comes_back(self):
        model = _FakeDiT()
        ami = _fake_ami(model)
        with mock.patch.dict(sys.modules, _fake_modules(_FakeState())):
            latent = anima_core._generate_with_stop(ami, None, {})
        self.assertIsInstance(latent, torch.Tensor)
        self.assertEqual(model.calls, 3)
        self.assertIs(ami.generate_body, _fake_body)

    def test_other_errors_still_propagate(self):
        ami = _fake_ami(_FakeDiT(), fail=RuntimeError("size mismatch"))
        with mock.patch.dict(sys.modules, _fake_modules(_FakeState())):
            with self.assertRaises(RuntimeError):
                anima_core._generate_with_stop(ami, None, {})
        self.assertIs(ami.generate_body, _fake_body)


class _Strategy:
    current = "previous"

    @classmethod
    def get_strategy(cls):
        return cls.current

    @classmethod
    def set_strategy(cls, value):
        cls.current = value


class RunTileRepairStopWiringTests(unittest.TestCase):
    """run_tile_repair 전체를 가짜 벤더로 돌려 ⏹ Stop 배선(가드 → 빈 결과 → 정리)을 확인한다."""

    def _run(self, state, model):
        ami = _fake_ami(model)
        tok = type("TokenizeStrategy", (_Strategy,), {"current": "prev-tok"})
        te = type("TextEncodingStrategy", (_Strategy,), {"current": "prev-te"})
        strategy_base = types.SimpleNamespace(TokenizeStrategy=tok, TextEncodingStrategy=te)
        strategy_anima = types.SimpleNamespace(
            AnimaTokenizeStrategy=lambda **kw: "anima-tok",
            AnimaTextEncodingStrategy=lambda: "anima-te",
        )
        vae_loads: list = []
        train_utils = types.SimpleNamespace(
            load_qwen_image_vae=lambda *a, **k: vae_loads.append(1) or torch.nn.Identity()
        )
        library = types.ModuleType("library")
        library.__path__ = []
        library.strategy_base = strategy_base
        library.strategy_anima = strategy_anima
        library.anima_train_utils = train_utils
        mods = _fake_modules(state)
        mods.update({
            "anima_minimal_inference_control_net_lllite": types.ModuleType("lllite"),
            "anima_minimal_inference": ami,
            "library": library,
            "library.strategy_base": strategy_base,
            "library.strategy_anima": strategy_anima,
            "library.anima_train_utils": train_utils,
        })
        control: dict = {}

        def fake_args(repair, control_image_path):
            control["path"] = control_image_path
            return types.SimpleNamespace(dit="dit.safetensors", text_encoder="te.safetensors",
                                         vae="vae.safetensors", seed=7)

        repair = anima_core.AnimaTileRepairArgs(
            lllite_model="animaTileRepair_v10.safetensors", unload_forge_before=False
        )
        with (
            mock.patch.dict(sys.modules, mods),
            mock.patch.object(anima_core, "_ensure_vendor_importable", return_value=True),
            mock.patch.object(anima_core, "ANIMA_LLLITE_SENTINEL",
                              types.SimpleNamespace(exists=lambda: True)),
            mock.patch.object(anima_core, "_vendor_sys_modules", contextlib.nullcontext),
            mock.patch.object(anima_core, "_build_anima_args", side_effect=fake_args),
        ):
            out = anima_core.run_tile_repair(Image.new("RGB", (64, 64), "white"), repair)
        return out, ami, tok, te, vae_loads, control

    def test_stop_during_sampling_returns_empty_and_cleans_up(self):
        state = _FakeState()
        state.interrupted = True   # run_exclusive 안에서 Stop 이 눌린 상태
        model = _FakeDiT()
        out, ami, tok, te, vae_loads, control = self._run(state, model)
        self.assertEqual(out, [], "빈 결과 → 패널이 'no output produced (interrupted?)' 를 띄운다")
        self.assertEqual(ami.generate_calls, 1)
        self.assertEqual(model.calls, 0, "디노이징 첫 스텝 전에 멈춘다")
        self.assertEqual(vae_loads, [], "멈춘 뒤 VAE 디코드로 넘어가지 않는다")
        self.assertIs(ami.generate_body, _fake_body)
        self.assertEqual((tok.current, te.current), ("prev-tok", "prev-te"), "전역 전략이 복원된다")
        self.assertFalse(os.path.exists(control["path"]), "임시 control 이미지가 지워진다")

    def test_without_stop_the_same_path_produces_one_image(self):
        model = _FakeDiT()
        out, ami, tok, te, vae_loads, control = self._run(_FakeState(), model)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0][0], Image.Image)
        self.assertIn("Seed: 7", out[0][1])
        self.assertEqual(model.calls, 3)
        self.assertEqual(vae_loads, [1])
        self.assertEqual((tok.current, te.current), ("prev-tok", "prev-te"))
        self.assertFalse(os.path.exists(control["path"]))


def _anima_values(**changes):
    with gr.Blocks():
        panel = build_anima_panel()
    keyed = {key: widget.value for key, widget in zip(ANIMA_ARG_KEYS, panel.all_widgets())}
    keyed["lllite_model"] = "animaTileRepair_v10.safetensors"
    keyed.update(changes)
    return tuple(keyed[key] for key in ANIMA_ARG_KEYS)


class AnimaHandlerTests(unittest.TestCase):
    """handle_anima_click 이 Tile-Repair·PiD 를 각각 제 job 이름으로 Forge 잠금 안에서 돌린다."""

    def _click(self, values, *, tile=None, pid=None):
        gallery = [Image.new("RGB", (64, 64), "white")]
        calls: list = []

        def fake_exclusive(job, fn, *, on_main_thread=False):
            calls.append((job, on_main_thread))
            return fn()

        with (
            mock.patch.object(ui_anima, "run_exclusive", side_effect=fake_exclusive),
            mock.patch.object(ui_anima, "anima_available", return_value=True),
            mock.patch.object(
                ui_anima, "run_tile_repair", side_effect=tile or (lambda *a, **k: [(gallery[0], "tile")])
            ),
            mock.patch.object(
                ui_anima, "run_pid_upscale", side_effect=pid or (lambda *a, **k: [(gallery[0], "pid")])
            ),
        ):
            outputs = ui_anima.handle_anima_click(gallery, 0, *values, "")
        return outputs, calls

    def test_tile_repair_runs_as_its_own_forge_job_on_the_main_thread(self):
        outputs, calls = self._click(_anima_values())
        self.assertEqual(calls, [("sam3_tile_repair", True)])
        self.assertEqual(len(outputs[0]), 2)

    def test_pid_upscale_runs_as_its_own_forge_job_on_the_main_thread(self):
        outputs, calls = self._click(
            _anima_values(restore_mode="PiD Upscale", pid_checkpoint="PiD_v1.safetensors")
        )
        self.assertEqual(calls, [("sam3_pid_upscale", True)])
        self.assertEqual(len(outputs[0]), 2)

    def test_validation_errors_do_not_take_the_lock(self):
        outputs, calls = self._click(_anima_values(positive=""))
        self.assertEqual(calls, [])
        self.assertIn("prompt is empty", outputs[1])

    def test_a_stopped_tile_repair_shows_the_interrupted_status_not_a_traceback(self):
        def stopped(*a, **k):
            return []

        outputs, calls = self._click(_anima_values(), tile=stopped)
        self.assertEqual(calls, [("sam3_tile_repair", True)])
        self.assertIn("interrupted", outputs[1])


class AnimaStopTests(unittest.TestCase):
    def test_stop_interrupts_only_a_tile_repair_or_pid_job(self):
        for job, expected in (
            ("sam3_tile_repair", True),
            ("sam3_pid_upscale", True),
            ("task(abc)", False),
            ("sam3_refine", False),
        ):
            state = _FakeState()
            state.job = job
            with mock.patch.dict(sys.modules, _fake_modules(state)):
                ui_anima.stop_anima()
            self.assertEqual(state.interrupted, expected, job)
            self.assertEqual(state.skipped, expected, job)


if __name__ == "__main__":
    unittest.main()
