"""Anima Tile & Repair JSON route (sam3ext/tile_repair_api.py, UR_IV 원본 동등성 계획 §7.3 A · 패키지 TR-API).

The route must run what the Tile-Repair panel runs, with the panel's defaults — which are the originals:
kohya-ss/sd-scripts@690ea7f9 anima_minimal_inference_control_net_lllite.py argparse defaults and the
kohya-ss/ComfyUI-Anima-LLLite@b7495bd8 node's strength range. No GPU, no vendor: the pipeline is injected.
"""
from __future__ import annotations

import base64
import contextlib
import dataclasses
import importlib.util
import io
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import anima_core, tile_repair_api as api, ui_anima  # noqa: E402
from sam3ext.anima_core import AnimaTileRepairArgs  # noqa: E402
from sam3ext.tile_repair_api import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULTS,
    RANGES,
    TILE_REPAIR_API_PATH,
    TILE_REPAIR_OPTIONS_PATH,
    TILE_REPAIR_STOP_PATH,
    TileRepairChoices,
    TileRepairRequestError,
    parse_request,
    register_tile_repair_routes,
)

HEADERS = {"X-SAM3-Notebook": "1"}
V10 = "animaTileRepair_v10.safetensors"
V20 = "animaTileRepair_v20.safetensors"
LINEART = "anima_lineart_lllite.safetensors"
TE = "qwen_3_06b_base.safetensors"
VAE = "qwen_image_vae.safetensors"
DIT = "anima-preview.safetensors"

CHOICES = TileRepairChoices(
    lllite=(V10, V20, LINEART),
    dit=("Use Forge current", DIT),
    text_encoder=("Use Forge current", "h3_qwen3vl_8b_tap24.safetensors", TE),
    vae=("Use Forge current", "sdxl_vae.safetensors", VAE),
    default_lllite=V20,
    default_text_encoder=TE,
    default_vae=VAE,
)


def png_b64(size=(40, 24), color=(200, 120, 40), fmt="PNG") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Pipeline:
    """Injected ``execute`` — records the request, answers like run_tile_repair."""

    def __init__(self, result=None, error: Exception | None = None):
        self.calls = []
        self.result = result
        self.error = error

    def __call__(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        out = request.source.convert("RGB").resize((64, 96))
        return [(out, f"{request.prompt}\nSteps: {request.steps}, CFG scale: {request.cfg_scale}, "
                      f"Seed: 1234, Size: 64x96, Flow shift: {request.flow_shift}, "
                      f"LLLite: {request.model} (mult {request.multiplier})\nAnima Tile-Repair: on")]


def _app(pipeline=None, *, available=True, stop=None, choices=CHOICES):
    app = FastAPI()
    register_tile_repair_routes(
        app,
        choices_provider=lambda: choices,
        execute=pipeline or _Pipeline(),
        available=lambda: available,
        stop=stop or (lambda: False),
    )
    return app


class RegistrationTests(unittest.TestCase):
    def test_routes_are_registered_once_with_their_methods(self):
        app = FastAPI()
        self.assertTrue(register_tile_repair_routes(app, choices_provider=lambda: CHOICES))
        self.assertFalse(register_tile_repair_routes(app, choices_provider=lambda: CHOICES))
        methods = {
            (route.path, method)
            for route in app.routes
            if route.path.startswith("/sam-extra/")
            for method in route.methods
        }
        self.assertEqual(methods, {
            (TILE_REPAIR_API_PATH, "POST"),
            (TILE_REPAIR_OPTIONS_PATH, "GET"),
            (TILE_REPAIR_STOP_PATH, "POST"),
        })

    def test_get_on_the_run_route_is_405_so_clients_can_probe_without_side_effects(self):
        pipeline = _Pipeline()
        with TestClient(_app(pipeline)) as client:
            self.assertEqual(client.get(TILE_REPAIR_API_PATH).status_code, 405)
        self.assertEqual(pipeline.calls, [])

    def test_every_route_needs_the_notebook_same_origin_header(self):
        pipeline = _Pipeline()
        stopped = []
        with TestClient(_app(pipeline, stop=lambda: stopped.append(1) or True)) as client:
            self.assertEqual(client.get(TILE_REPAIR_OPTIONS_PATH).status_code, 403)
            self.assertEqual(client.post(TILE_REPAIR_API_PATH, json={"image": png_b64()}).status_code, 403)
            self.assertEqual(client.post(TILE_REPAIR_STOP_PATH).status_code, 403)
        self.assertEqual((pipeline.calls, stopped), ([], []))

    def test_routes_reuse_the_gradio_login_guard(self):
        app = FastAPI()

        @app.get("/login_check")
        def login_check(x_test_auth: str | None = Header(default=None)):
            if x_test_auth != "ok":
                raise HTTPException(status_code=401, detail="Not authenticated")

        register_tile_repair_routes(app, choices_provider=lambda: CHOICES, execute=_Pipeline(),
                                    available=lambda: True)
        with TestClient(app) as client:
            self.assertEqual(client.get(TILE_REPAIR_OPTIONS_PATH, headers=HEADERS).status_code, 401)
            ok = client.get(TILE_REPAIR_OPTIONS_PATH, headers={**HEADERS, "X-Test-Auth": "ok"})
            self.assertEqual(ok.status_code, 200)

    def test_script_registers_the_routes_on_app_started(self):
        registered = []
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_app_started = lambda fn, *, name=None: registered.append((fn, name))
        modules = types.ModuleType("modules")
        modules.script_callbacks = callbacks
        path = ROOT / "scripts" / "anima_tile_repair_api.py"
        spec = importlib.util.spec_from_file_location("_sam3_tile_repair_api_script", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"modules": modules, "modules.script_callbacks": callbacks}):
            spec.loader.exec_module(module)
        self.assertEqual(len(registered), 1)
        callback, name = registered[0]
        self.assertEqual(name, "sam-extra-tile-repair-api")
        app = FastAPI()
        with contextlib.redirect_stdout(io.StringIO()):
            callback(None, app)
            callback(None, app)  # Reload UI re-fires app start
        self.assertEqual(sum(route.path == TILE_REPAIR_API_PATH for route in app.routes), 1)


class OptionsTests(unittest.TestCase):
    def test_options_carry_choices_defaults_and_bounds(self):
        with TestClient(_app()) as client:
            response = client.get(TILE_REPAIR_OPTIONS_PATH, headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")
        body = response.json()
        self.assertEqual(body["version"], 1)
        self.assertTrue(body["available"])
        self.assertEqual(body["models"], [V10, V20, LINEART])
        self.assertEqual(body["default_model"], V20)
        self.assertEqual(body["dit"], ["Use Forge current", DIT])
        defaults = body["defaults"]
        # origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:127-134
        self.assertEqual((defaults["negative_prompt"], defaults["steps"], defaults["cfg_scale"],
                          defaults["flow_shift"]), ("", 50, 3.5, 5.0))
        # origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:167-170 (1.0)
        self.assertEqual(defaults["multiplier"], 1.0)
        self.assertEqual((defaults["model"], defaults["text_encoder"], defaults["vae"], defaults["dit"]),
                         (V20, TE, VAE, "Use Forge current"))
        # origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:130 — strength -10..10 step .01
        self.assertEqual(body["ranges"]["multiplier"], [-10.0, 10.0])
        self.assertEqual(body["increments"]["multiplier"], 0.01)
        self.assertEqual(body["ranges"]["short_side"], [256, 4096])

    def test_no_lllite_installed_means_no_default_model(self):
        empty = TileRepairChoices((), ("Use Forge current",), ("Use Forge current",),
                                  ("Use Forge current",), "None", "Use Forge current", "Use Forge current")
        with TestClient(_app(choices=empty, available=False)) as client:
            body = client.get(TILE_REPAIR_OPTIONS_PATH, headers=HEADERS).json()
        self.assertFalse(body["available"])
        self.assertEqual((body["models"], body["default_model"]), ([], None))


class DefaultsMatchThePanelTests(unittest.TestCase):
    """The route's defaults and bounds are the panel's (which are the originals — TR-F)."""

    def _panel(self):
        with mock.patch.object(ui_anima, "list_lllite_choices", return_value=["None", V10, V20]):
            with gr.Blocks():
                return ui_anima.build_anima_panel()

    def test_defaults_and_bounds_equal_the_panel_widgets(self):
        panel = self._panel()
        self.assertEqual(panel.positive.value, DEFAULT_PROMPT)
        self.assertEqual(panel.negative.value, DEFAULTS["negative_prompt"])
        self.assertEqual(panel.seed.value, DEFAULTS["seed"])
        self.assertEqual(panel.unload_forge_before.value, DEFAULTS["unload_forge_before"])
        sliders = {
            "steps": panel.steps, "cfg_scale": panel.cfg, "flow_shift": panel.flow_shift,
            "multiplier": panel.lllite_multiplier, "short_side": panel.short_side,
        }
        self.assertEqual(set(sliders), set(RANGES))
        for key, slider in sliders.items():
            with self.subTest(key=key):
                self.assertEqual(slider.value, DEFAULTS[key])
                self.assertEqual((slider.minimum, slider.maximum), RANGES[key])
                self.assertEqual(slider.step, api.INCREMENTS[key])

    def test_a_default_request_builds_the_same_args_as_the_default_panel_click(self):
        panel = self._panel()
        from_panel = ui_anima._map_widget_values(tuple(w.value for w in panel.all_widgets()))
        with mock.patch.object(anima_core, "list_lllite_choices", return_value=["None", V10, V20]):
            choices = api.panel_choices()
        from_api = parse_request({"image": png_b64()}, choices).to_repair_args()
        self.assertIsInstance(from_api, AnimaTileRepairArgs)
        # PiD fields belong to the other restoration mode and insert_mode to the gallery splice.
        panel_only = {"pid_checkpoint", "pid_scale", "pid_steps", "pid_degrade", "insert_mode"}
        strip = lambda args: {k: v for k, v in dataclasses.asdict(args).items() if k not in panel_only}  # noqa: E731
        self.assertEqual(strip(from_api), strip(from_panel))
        self.assertEqual(from_panel.restore_mode, "Anima Tile-Repair")


class RunTests(unittest.TestCase):
    def _post(self, body, pipeline=None, **kwargs):
        with TestClient(_app(pipeline, **kwargs)) as client:
            return client.post(TILE_REPAIR_API_PATH, headers=HEADERS, json=body)

    def test_image_only_request_runs_with_the_original_defaults(self):
        pipeline = _Pipeline()
        response = self._post({"image": png_b64()}, pipeline)
        self.assertEqual(response.status_code, 200, response.text)
        (request,) = pipeline.calls
        args = request.to_repair_args()
        self.assertEqual(args.lllite_model, V20)                  # newest Tile & Repair (panel default)
        # origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:127-134
        self.assertEqual(args.negative, "")                        # --negative_prompt ""
        self.assertEqual(args.steps, 50)                           # --infer_steps 50
        self.assertEqual(args.cfg, 3.5)                            # --guidance_scale 3.5
        self.assertEqual(args.flow_shift, 5.0)                     # --flow_shift 5.0
        # origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:167-170
        self.assertEqual(args.lllite_multiplier, 1.0)              # --lllite_multiplier 1.0
        self.assertEqual((args.short_side, args.seed), (1024, -1))
        self.assertEqual((args.dit_override, args.te_override, args.vae_override),
                         ("Use Forge current", TE, VAE))
        self.assertEqual(args.lora_slots, [("None", 0.0)] * 4)
        self.assertEqual(args.positive, DEFAULT_PROMPT)
        self.assertEqual(request.source.size, (40, 24))

        body = response.json()
        self.assertFalse(body["interrupted"])
        self.assertEqual((body["width"], body["height"], body["seed"], body["model"]), (64, 96, 1234, V20))
        self.assertIn("Anima Tile-Repair: on", body["info"])
        with Image.open(io.BytesIO(base64.b64decode(body["image"]))) as out:
            self.assertEqual((out.format, out.size), ("PNG", (64, 96)))
            self.assertEqual(out.text["parameters"], body["info"])

    def test_explicit_values_reach_the_pipeline(self):
        pipeline = _Pipeline()
        response = self._post({
            "image": "data:image/png;base64," + png_b64(),
            "model": LINEART, "prompt": "fix it", "negative_prompt": "blurry",
            "steps": 30, "cfg_scale": 5, "flow_shift": 3.0, "multiplier": -10,
            "short_side": 768, "seed": 42, "dit": DIT, "text_encoder": TE, "vae": VAE,
            "unload_forge_before": False,
        }, pipeline)
        self.assertEqual(response.status_code, 200, response.text)
        args = pipeline.calls[0].to_repair_args()
        self.assertEqual(
            (args.lllite_model, args.positive, args.negative, args.steps, args.cfg, args.flow_shift,
             args.lllite_multiplier, args.short_side, args.seed, args.dit_override,
             args.unload_forge_before),
            (LINEART, "fix it", "blurry", 30, 5.0, 3.0, -10.0, 768, 42, DIT, False),
        )

    def test_jpeg_and_webp_sources_decode(self):
        for fmt in ("JPEG", "WEBP"):
            with self.subTest(fmt=fmt):
                self.assertEqual(self._post({"image": png_b64(fmt=fmt)}).status_code, 200)

    def test_refused_requests_never_reach_the_pipeline(self):
        cases = {
            "unknown key": {"image": png_b64(), "lllite_multiplier": 1.0},
            "no image": {"steps": 20},
            "bad base64": {"image": "not base64 !"},
            "not an image": {"image": base64.b64encode(b"hello").decode()},
            "multiplier above 10": {"image": png_b64(), "multiplier": 10.01},
            "multiplier below -10": {"image": png_b64(), "multiplier": -10.5},
            "steps 0": {"image": png_b64(), "steps": 0},
            "fractional steps": {"image": png_b64(), "steps": 20.5},
            "steps as text": {"image": png_b64(), "steps": "20"},
            "cfg as bool": {"image": png_b64(), "cfg_scale": True},
            "short side too small": {"image": png_b64(), "short_side": 128},
            "nan shift": {"image": png_b64(), "flow_shift": "NaN"},
            "empty prompt": {"image": png_b64(), "prompt": "   "},
            "negative seed": {"image": png_b64(), "seed": -2},
            "4-channel or unknown model": {"image": png_b64(), "model": "anima-lllite-inpainting-v2.safetensors"},
            "unknown TE": {"image": png_b64(), "text_encoder": "missing.safetensors"},
            "unload as text": {"image": png_b64(), "unload_forge_before": "yes"},
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                pipeline = _Pipeline()
                response = self._post(body, pipeline)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertTrue(response.json()["detail"])
                self.assertEqual(pipeline.calls, [])

    def test_non_object_and_invalid_json(self):
        with TestClient(_app()) as client:
            self.assertEqual(client.post(TILE_REPAIR_API_PATH, headers=HEADERS, json=[1]).status_code, 400)
            broken = client.post(TILE_REPAIR_API_PATH, headers={**HEADERS, "Content-Type": "application/json"},
                                 content=b"{broken")
            self.assertEqual(broken.status_code, 400)

    def test_too_large_request_is_413(self):
        with mock.patch.object(api, "MAX_REQUEST_BYTES", 100):
            response = self._post({"image": png_b64()})
        self.assertEqual(response.status_code, 413)

    def test_no_three_channel_lllite_installed_is_422(self):
        empty = TileRepairChoices((), ("Use Forge current",), ("Use Forge current", TE),
                                  ("Use Forge current", VAE), "None", TE, VAE)
        response = self._post({"image": png_b64()}, choices=empty)
        self.assertEqual(response.status_code, 422)
        self.assertIn("models/ControlNet", response.json()["detail"])

    def test_vendor_missing_is_503(self):
        pipeline = _Pipeline()
        response = self._post({"image": png_b64()}, pipeline, available=False)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(pipeline.calls, [])

    def test_stop_during_the_run_answers_interrupted(self):
        response = self._post({"image": png_b64()}, _Pipeline(result=[]))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["interrupted"])
        self.assertIsNone(body["image"])
        self.assertEqual(body["model"], V20)

    def test_pipeline_setup_errors_are_422_and_others_500(self):
        message = "Anima는 Qwen3 Text Encoder가 필수입니다."
        setup = self._post({"image": png_b64()}, _Pipeline(error=RuntimeError(message)))
        self.assertEqual((setup.status_code, setup.json()["detail"]), (422, message))
        with contextlib.redirect_stderr(io.StringIO()), self.assertLogs(api.logger, "ERROR"):
            crash = self._post({"image": png_b64()}, _Pipeline(error=KeyError("x")))
        self.assertEqual(crash.status_code, 500)
        self.assertIn("KeyError", crash.json()["detail"])

    def test_stop_route_reports_whether_a_tile_repair_job_was_stopped(self):
        with TestClient(_app(stop=lambda: True)) as client:
            response = client.post(TILE_REPAIR_STOP_PATH, headers=HEADERS)
        self.assertEqual((response.status_code, response.json()), (200, {"stopped": True}))


class PanelPipelineWiringTests(unittest.TestCase):
    """Forge defaults: the panel's pipeline, lock and main thread under the route's own job name."""

    def test_run_goes_through_run_exclusive_with_the_route_job_on_the_main_thread(self):
        request = parse_request({"image": png_b64(), "seed": 7}, CHOICES)
        seen = {}

        def fake_exclusive(job, fn, *, on_main_thread=False):
            seen["job"], seen["main"] = job, on_main_thread
            return fn()

        def fake_run(source, repair):
            seen["source"], seen["repair"] = source, repair
            return [("pil", "info")]

        with (
            mock.patch("sam3ext.forge_exclusive.run_exclusive", side_effect=fake_exclusive),
            mock.patch.object(anima_core, "run_tile_repair", side_effect=fake_run),
        ):
            self.assertEqual(api.run_panel_pipeline(request), [("pil", "info")])
        self.assertEqual((seen["job"], seen["main"]), (api.ROUTE_JOB, True))
        self.assertIs(seen["source"], request.source)
        self.assertEqual(seen["repair"].seed, 7)

    def test_route_and_panel_jobs_never_match_each_others_stop(self):
        # forge_exclusive.stop_if_job matches by prefix: neither name may start with the other.
        self.assertFalse(api.ROUTE_JOB.startswith(ui_anima.TILE_REPAIR_JOB))
        self.assertFalse(ui_anima.TILE_REPAIR_JOB.startswith(api.ROUTE_JOB))
        self.assertFalse(api.ROUTE_JOB.startswith(ui_anima._ANIMA_JOB_PREFIXES))

    def test_stop_only_targets_the_route_job(self):
        with mock.patch("sam3ext.forge_exclusive.stop_if_job", return_value=True) as stop_if_job:
            self.assertTrue(api.stop_tile_repair())
        stop_if_job.assert_called_once_with((api.ROUTE_JOB,))

    def test_a_cancel_made_while_queued_runs_nothing_once_the_queue_is_free(self):
        request = parse_request({"image": png_b64()}, CHOICES)

        def cancel_while_waiting(job, fn, *, on_main_thread=False):
            request.cancel.set()        # the stop route, while this request waited for queue_lock
            return fn()

        with (
            mock.patch("sam3ext.forge_exclusive.run_exclusive", side_effect=cancel_while_waiting),
            mock.patch.object(anima_core, "run_tile_repair") as run,
        ):
            self.assertEqual(api.run_panel_pipeline(request), [])
        run.assert_not_called()
        with mock.patch("sam3ext.forge_exclusive.run_exclusive") as exclusive:
            self.assertEqual(api.run_panel_pipeline(request), [])   # already cancelled: no queue wait
        exclusive.assert_not_called()

    def test_seed_is_read_back_from_the_infotext(self):
        info = anima_core._build_infotext(AnimaTileRepairArgs(positive="p", seed=-1), 987654)
        self.assertEqual(api.seed_from_infotext(info), 987654)
        self.assertIsNone(api.seed_from_infotext("no seed here"))

    def test_seed_in_the_prompts_is_not_taken_for_the_seed_used(self):
        tricky = "1girl, Seed: 5, masterpiece\nSteps: 1, CFG scale: 1.0, Seed: 6, Size: 8x8"
        for positive, negative in ((tricky, ""), ("p", "Seed: 7, blurry"), (tricky, tricky)):
            with self.subTest(positive=positive, negative=negative):
                info = anima_core._build_infotext(
                    AnimaTileRepairArgs(positive=positive, negative=negative, seed=-1), 987654)
                self.assertEqual(api.seed_from_infotext(info), 987654)

    def test_request_error_status_defaults_to_400(self):
        self.assertEqual(TileRepairRequestError("x").status, 400)


class _State:
    """modules/shared_state.py begin/end — only what forge_exclusive touches."""

    def __init__(self):
        self.job = ""
        self.interrupted = self.skipped = self.stopping_generation = False
        self.jobs: list = []

    def begin(self, job="(unknown)"):
        self.jobs.append(job)
        self.job = job
        self.interrupted = self.skipped = self.stopping_generation = False

    def end(self):
        self.job = ""


class _QueueLock:
    """Forge's queue_lock stand-in that says when someone starts waiting for it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.waiting = threading.Event()

    def __enter__(self):
        self.waiting.set()
        self._lock.acquire()

    def __exit__(self, *exc):
        self._lock.release()

    def acquire(self):
        self._lock.acquire()

    def release(self):
        self._lock.release()


class RouteStopTests(unittest.TestCase):
    """Stop reaches a route request in every phase and never the job that holds Forge's queue."""

    def setUp(self):
        self.state = _State()
        self.lock = _QueueLock()
        package = types.ModuleType("modules")
        package.__path__ = []
        shared = types.ModuleType("modules.shared")
        shared.state = self.state
        call_queue = types.ModuleType("modules.call_queue")
        call_queue.queue_lock = self.lock
        package.shared, package.call_queue = shared, call_queue
        patcher = mock.patch.dict(sys.modules, {
            "modules": package, "modules.shared": shared, "modules.call_queue": call_queue,
            "modules_forge": None, "modules_forge.main_thread": None,   # run fn on this thread
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runs = []

    def _app(self, run=None, choices_provider=None):
        def fake_run(source, repair):
            self.runs.append(repair)
            return run() if run else [(source.convert("RGB"), "p\nSteps: 50, Seed: 3, Size: 8x8")]

        patcher = mock.patch.object(anima_core, "run_tile_repair", side_effect=fake_run)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = FastAPI()
        register_tile_repair_routes(app, choices_provider=choices_provider or (lambda: CHOICES),
                                    available=lambda: True)   # default execute and stop
        return app

    @staticmethod
    def _post_in_background(client):
        box: dict = {}

        def work():
            box["response"] = client.post(TILE_REPAIR_API_PATH, headers=HEADERS, json={"image": png_b64()})

        worker = threading.Thread(target=work)
        worker.start()
        return worker, box

    def _stop(self, client) -> dict:
        response = client.post(TILE_REPAIR_STOP_PATH, headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _queued_behind(self, job: str):
        self.lock.acquire()
        self.state.begin(job=job)
        with TestClient(self._app()) as client:
            worker, box = self._post_in_background(client)
            self.assertTrue(self.lock.waiting.wait(5), "route request never queued")
            self.assertEqual(self._stop(client), {"stopped": True})
            holder_interrupted = self.state.interrupted
            self.state.end()
            self.lock.release()
            worker.join(10)
        return box["response"], holder_interrupted

    def test_stop_while_queued_behind_a_txt2img_cancels_the_request_not_the_txt2img(self):
        response, txt2img_interrupted = self._queued_behind("txt2img")
        self.assertFalse(txt2img_interrupted)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["interrupted"])
        self.assertEqual(self.runs, [])                       # nothing loaded, nothing unloaded
        self.assertEqual(self.state.jobs, ["txt2img", api.ROUTE_JOB])
        self.assertFalse(self.state.interrupted or self.state.skipped)

    def test_stop_while_queued_behind_a_panel_click_leaves_the_panel_run_alone(self):
        response, panel_interrupted = self._queued_behind(ui_anima.TILE_REPAIR_JOB)
        self.assertFalse(panel_interrupted)
        self.assertTrue(response.json()["interrupted"])
        self.assertEqual(self.runs, [])

    def test_stop_while_running_interrupts_the_route_job(self):
        entered, release = threading.Event(), threading.Event()
        seen = {}

        def blocking_run():
            entered.set()
            release.wait(5)
            seen["interrupted"] = self.state.interrupted
            return []                                     # run_tile_repair's answer after ⏹

        with TestClient(self._app(run=blocking_run)) as client:
            worker, box = self._post_in_background(client)
            self.assertTrue(entered.wait(5))
            self.assertEqual(self._stop(client), {"stopped": True})
            release.set()
            worker.join(10)
        self.assertTrue(seen["interrupted"])
        self.assertTrue(box["response"].json()["interrupted"])
        self.assertFalse(self.state.interrupted)          # run_exclusive cleared the flags after

    def test_stop_while_the_request_is_still_being_read_runs_nothing(self):
        gate, reading = threading.Event(), threading.Event()

        def slow_choices():
            reading.set()
            gate.wait(5)
            return CHOICES

        with TestClient(self._app(choices_provider=slow_choices)) as client:
            worker, box = self._post_in_background(client)
            self.assertTrue(reading.wait(5))
            self.assertEqual(self._stop(client), {"stopped": True})
            gate.set()
            worker.join(10)
        self.assertTrue(box["response"].json()["interrupted"])
        self.assertEqual((self.runs, self.state.jobs), ([], []))     # never even queued

    def test_stop_with_nothing_in_flight_reports_false(self):
        with TestClient(self._app()) as client:
            self.assertEqual(self._stop(client), {"stopped": False})
            done = client.post(TILE_REPAIR_API_PATH, headers=HEADERS, json={"image": png_b64()})
            self.assertEqual(self._stop(client), {"stopped": False})   # finished requests are gone
        self.assertFalse(done.json()["interrupted"])
        self.assertEqual(done.json()["seed"], 3)
        self.assertEqual(len(self.runs), 1)


if __name__ == "__main__":
    unittest.main()
