"""SAM Extra Progress Bar — step/ETA snapshot logic, the GET /sam-extra/progress route, its Settings and
the contract the page (javascript/progress_bar.js) and style.css share with it.

Upstream (diamfang/sd-webui-smooth-progress@7fe5810, MIT) parity of the ETA arithmetic is in
tests/test_progress_origin.py. Forge's own ``progressapi`` (modules/progress.py, read from the Forge checkout
next to the extension when there is one) is the oracle for the whole-job progress and the queue text.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib.util
import io
import json
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr  # noqa: F401 - the script under test imports it; keep it loaded outside patch.dict
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import progress_api as pa  # noqa: E402
from sam3ext.progress_api import (  # noqa: E402
    PROGRESS_API_PATH,
    ProgressTracker,
    StateView,
    TaskView,
    average_step_seconds,
    job_progress,
    pass_progress,
    progress_snapshot,
    read_state,
    read_tasks,
    register_progress_routes,
    step_eta,
)

HEADERS = {"X-SAM3-Notebook": "1"}
FORGE_PROGRESS_PY = ROOT.parents[1] / "modules" / "progress.py"
MIT_LINES = (
    "Copyright (c) 2026 diamfang",
    "Permission is hereby granted, free of charge, to any person obtaining a copy",
    "The above copyright notice and this permission notice shall be included in all",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR',
)


def _state(**fields):
    """A Forge ``shared.state`` look-alike (modules/shared_state.py attribute names)."""

    base = dict(job="task(a)", job_no=0, job_count=1, sampling_step=0, sampling_steps=20, time_start=100.0,
                interrupted=False, skipped=False, stopping_generation=False,
                processing_has_refined_job_count=False, textinfo=None)
    base.update(fields)
    return types.SimpleNamespace(**base)


def _view(**fields):
    return read_state(_state(**fields))


def _forge_progressapi(state, current, pending, finished, now):
    """Forge's own ``progressapi`` (modules/progress.py), compiled from the Forge checkout as is."""

    tree = ast.parse(FORGE_PROGRESS_PY.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "progressapi")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "ProgressRequest": object,
        "ProgressResponse": lambda **fields: fields,
        "current_task": current,
        "pending_tasks": dict(pending),
        "finished_tasks": list(finished),
        "shared": types.SimpleNamespace(state=state),
        "opts": types.SimpleNamespace(live_previews_enable=False),
        "time": types.SimpleNamespace(time=lambda: now),
    }
    exec(compile(module, str(FORGE_PROGRESS_PY), "exec"), namespace)
    return lambda id_task: namespace["progressapi"](
        types.SimpleNamespace(id_task=id_task, id_live_preview=-1, live_preview=False))


# ---------------------------------------------------------------------------
# Per-step ETA
# ---------------------------------------------------------------------------


class StepEtaTests(unittest.TestCase):
    def test_unknown_until_a_step_finishes(self):
        eta, tracker = step_eta(None, 0, 10, 0.0)
        self.assertEqual((eta, tracker), (0.0, {"step": 0, "step_ts": 0.0, "dur_sum": 0.0, "dur_n": 0}))
        self.assertIsNone(average_step_seconds(tracker))
        eta, tracker = step_eta(tracker, 0, 10, 0.7)
        self.assertEqual(eta, 0.0)
        self.assertIsNone(average_step_seconds(tracker))

    def test_counts_down_inside_a_step_and_drops_one_step_at_the_boundary(self):
        _, tracker = step_eta(None, 0, 10, 0.0)
        eta, tracker = step_eta(tracker, 1, 10, 1.0)
        self.assertEqual(eta, 9.0)
        self.assertEqual(average_step_seconds(tracker), 1.0)
        eta, tracker = step_eta(tracker, 1, 10, 1.5)
        self.assertEqual(eta, 8.5)
        eta, tracker = step_eta(tracker, 2, 10, 2.0)
        self.assertEqual(eta, 8.0)

    def test_a_late_poll_never_drives_the_estimate_below_one_step(self):
        _, tracker = step_eta(None, 0, 10, 0.0)
        _, tracker = step_eta(tracker, 1, 10, 1.0)
        eta, _ = step_eta(tracker, 1, 10, 3.5)   # 2.5 s in a step that averages 1 s
        self.assertEqual(eta, 8.0)

    def test_several_steps_between_looks_share_the_time(self):
        _, tracker = step_eta(None, 0, 10, 0.0)
        _, tracker = step_eta(tracker, 1, 10, 1.0)
        eta, tracker = step_eta(tracker, 4, 10, 4.0)
        self.assertEqual((tracker["dur_sum"], tracker["dur_n"]), (4.0, 4))
        self.assertEqual(eta, 6.0)

    def test_a_step_going_back_restarts_the_clock_and_adds_nothing(self):
        _, tracker = step_eta(None, 0, 10, 0.0)
        _, tracker = step_eta(tracker, 3, 10, 3.0)
        eta, tracker = step_eta(tracker, 0, 10, 5.0)     # next pass
        self.assertEqual((tracker["dur_sum"], tracker["dur_n"], tracker["step"], tracker["step_ts"]),
                         (3.0, 3, 0, 5.0))
        self.assertEqual(eta, 10.0)

    def test_the_callers_tracker_is_not_modified(self):
        _, tracker = step_eta(None, 0, 10, 0.0)
        before = dict(tracker)
        step_eta(tracker, 2, 10, 2.0)
        self.assertEqual(tracker, before)

    def test_never_negative(self):
        _, tracker = step_eta(None, 8, 10, 0.0)
        _, tracker = step_eta(tracker, 9, 10, 1.0)
        eta, _ = step_eta(tracker, 9, 10, 9.0)
        self.assertEqual(eta, 0.0)
        self.assertEqual(average_step_seconds(tracker), 1.0)


# ---------------------------------------------------------------------------
# Progress fractions
# ---------------------------------------------------------------------------


class ProgressFractionTests(unittest.TestCase):
    CASES = (
        # job_no, job_count, step, steps
        (0, 1, 0, 20),
        (0, 1, 7, 20),
        (0, 1, 19, 20),
        (1, 4, 5, 20),
        (3, 4, 19, 20),
        (4, 4, 0, 20),          # last nextjob before the end
        (5, 4, 3, 12),          # ADetailer's own pass inside the last image
        (0, -1, 0, 28),         # between State.begin and the processing setup
        (0, 0, 5, 20),
        (2, 6, 0, 0),
        (1, 2, 14, 15),         # hires pass (job_count doubled)
    )

    def test_whole_job_progress_is_forges_formula_clamped(self):
        for job_no, job_count, step, steps in self.CASES:
            with self.subTest(job_no=job_no, job_count=job_count, step=step, steps=steps):
                expected = 0.0
                if job_count > 0:
                    expected += job_no / job_count
                if steps > 0 and job_count > 0:
                    expected += 1 / job_count * step / steps
                self.assertEqual(job_progress(job_no, job_count, step, steps), min(expected, 1.0))

    @unittest.skipUnless(FORGE_PROGRESS_PY.is_file(), "Forge's modules/progress.py is not next to the extension")
    def test_matches_forges_own_progressapi(self):
        # origin: <Forge>/modules/progress.py progressapi (the native bar's job_no/job_count progress)
        for job_no, job_count, step, steps in self.CASES:
            with self.subTest(job_no=job_no, job_count=job_count, step=step, steps=steps):
                state = _state(job_no=job_no, job_count=job_count, sampling_step=step, sampling_steps=steps)
                forge = _forge_progressapi(state, "task(a)", {}, [], 130.0)("task(a)")
                ours = progress_snapshot("task(a)", read_state(state), TaskView(current="task(a)"),
                                         ProgressTracker(), 130.0)
                self.assertTrue(forge["active"])
                self.assertEqual(ours["progress"], round(forge["progress"], 4))

    def test_pass_progress_is_upstreams_fraction(self):
        self.assertEqual(pass_progress(5, 20), 0.25)
        self.assertEqual(pass_progress(0, 0), 0.0)
        self.assertEqual(pass_progress(25, 20), 1.0)


# ---------------------------------------------------------------------------
# Tracker (pass ETA and whole-job ETA)
# ---------------------------------------------------------------------------


class TrackerTests(unittest.TestCase):
    def test_single_pass_job_eta_is_the_pass_eta(self):
        tracker = ProgressTracker()
        self.assertEqual(tracker.observe(_view(sampling_step=0, sampling_steps=4), 100.0), (None, None))
        self.assertEqual(tracker.observe(_view(sampling_step=1, sampling_steps=4), 101.0), (3.0, 3.0))
        self.assertEqual(tracker.observe(_view(sampling_step=1, sampling_steps=4), 101.5), (2.5, 2.5))

    def test_batch_passes_add_the_last_finished_pass(self):
        tracker = ProgressTracker()
        look = lambda job_no, step, now: tracker.observe(  # noqa: E731
            _view(job_count=3, job_no=job_no, sampling_step=step, sampling_steps=4, time_start=0.0), now)
        self.assertEqual(look(0, 0, 0.0), (None, None))
        # No pass finished yet: the remaining two are assumed as long as this one will be (1 s + 3 s).
        self.assertEqual(look(0, 1, 1.0), (3.0, 3.0 + 2 * 4.0))
        self.assertEqual(look(0, 3, 3.0), (1.0, 1.0 + 2 * 4.0))
        # Pass 0 took 5 s from its first look to the next pass (decode, save, setup included).
        self.assertEqual(look(1, 0, 5.0), (4.0, 4.0 + 5.0))
        self.assertEqual(look(1, 1, 6.0), (3.0, 3.0 + 5.0))
        self.assertEqual(look(2, 1, 12.0), (3.0, 3.0))

    def test_hires_passes_keep_their_own_step_average(self):
        tracker = ProgressTracker()
        look = lambda job_no, step, steps, now: tracker.observe(  # noqa: E731
            _view(job_count=2, job_no=job_no, sampling_step=step, sampling_steps=steps, time_start=0.0,
                  processing_has_refined_job_count=True), now)
        self.assertEqual(look(0, 0, 4, 0.0), (None, None))
        self.assertEqual(look(0, 1, 4, 1.0), (3.0, 3.0 + 4.0))
        # Hires pass: base steps still in sampling_steps, then the hires count; its first step (setup
        # included) took 4.5 s. Upstream's single tracker would mix it with the 1 s base steps (2.75 s).
        self.assertEqual(look(1, 0, 4, 4.5), (None, None))
        self.assertEqual(look(1, 0, 2, 6.0), (None, None))
        self.assertEqual(look(1, 1, 2, 9.0), (4.5, 4.5))

    def test_second_image_counts_the_hires_pass_still_to_run(self):
        tracker = ProgressTracker()
        look = lambda job_no, step, steps, now: tracker.observe(  # noqa: E731
            _view(job_count=4, job_no=job_no, sampling_step=step, sampling_steps=steps, time_start=0.0,
                  processing_has_refined_job_count=True), now)
        look(0, 0, 4, 0.0)
        look(0, 1, 4, 1.0)
        look(1, 0, 2, 5.0)        # base pass: 5 s
        look(1, 1, 2, 8.0)
        pass_eta, job_eta = look(2, 0, 4, 13.0)   # hires pass: 8 s; second image starts
        self.assertEqual(pass_eta, 4.0)            # base steps average 1 s (1 s, the hires steps kept apart)
        self.assertEqual(job_eta, 4.0 + 8.0)       # this base pass + one hires pass like the last one

    def test_a_new_job_starts_over(self):
        tracker = ProgressTracker()
        tracker.observe(_view(sampling_step=0, sampling_steps=4, time_start=1.0), 1.0)
        self.assertEqual(tracker.observe(_view(sampling_step=1, sampling_steps=4, time_start=1.0), 2.0)[0], 3.0)
        self.assertEqual(tracker.observe(_view(sampling_step=2, sampling_steps=4, time_start=50.0), 51.0),
                         (None, None))

    def test_interrupted_or_skipped_looks_are_not_timed(self):
        for flag in ("interrupted", "skipped"):
            with self.subTest(flag=flag):
                tracker = ProgressTracker()
                tracker.observe(_view(sampling_step=0, sampling_steps=4), 100.0)
                self.assertEqual(tracker.observe(_view(sampling_step=2, sampling_steps=4, **{flag: True}), 102.0),
                                 (None, None))
                # The skipped look was not timed: the next step counts from the first look.
                self.assertEqual(tracker.observe(_view(sampling_step=3, sampling_steps=4), 103.0), (1.0, 1.0))

    def test_a_pass_joined_in_the_middle_is_not_measured(self):
        tracker = ProgressTracker()
        look = lambda job_no, step, now: tracker.observe(  # noqa: E731
            _view(job_count=3, job_no=job_no, sampling_step=step, sampling_steps=4, time_start=0.0), now)
        look(0, 2, 10.0)                     # a client that started looking half way through pass 0
        look(0, 3, 11.0)
        pass_eta, job_eta = look(1, 1, 13.0)
        self.assertEqual(pass_eta, 3.0)
        # Pass 0's 3 s are a fragment: the remaining pass is projected from this one (0 s in + 3 s left).
        self.assertEqual(job_eta, 3.0 + 3.0)

    def test_a_missed_step_zero_does_not_count_the_pass_change_as_step_time(self):
        tracker = ProgressTracker()
        look = lambda job_no, step, now: tracker.observe(  # noqa: E731
            _view(job_count=2, job_no=job_no, sampling_step=step, sampling_steps=4, time_start=0.0), now)
        look(0, 0, 0.0)
        look(0, 3, 3.0)                      # 1 s steps
        # Next look is already in pass 1 at step 2 (decode + save + setup in between): the clock
        # restarts there, the 1 s average stays.
        self.assertEqual(look(1, 2, 9.0), (2.0, 2.0))
        self.assertEqual(look(1, 3, 10.0), (1.0, 1.0))

    def test_no_job_eta_past_the_counted_passes(self):
        tracker = ProgressTracker()
        tracker.observe(_view(job_count=1, job_no=1, sampling_step=0, sampling_steps=12), 0.0)
        self.assertEqual(tracker.observe(_view(job_count=1, job_no=1, sampling_step=1, sampling_steps=12), 1.0),
                         (11.0, None))
        self.assertEqual(tracker.observe(_view(job_count=-1, job_no=0, sampling_step=1, sampling_steps=12,
                                               time_start=7.0), 8.0), (None, None))


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


TASKS = TaskView(current="task(a)", pending=("task(c)", "task(b)"), finished=frozenset({"task(z)"}))


class SnapshotTests(unittest.TestCase):
    def test_the_running_task_gets_the_job_fields(self):
        state = _state(job_no=1, job_count=4, sampling_step=5, sampling_steps=20, time_start=100.0,
                       textinfo="Loading B")
        payload = progress_snapshot("task(a)", read_state(state), TASKS, ProgressTracker(), 112.5)
        self.assertEqual(payload, {
            "version": 1, "id_task": "task(a)", "busy": True, "active": True, "queued": False,
            "completed": False, "queue_position": None, "queue_size": 2, "server_time": 112.5,
            "time_start": 100.0, "elapsed": 12.5, "job_no": 1, "job_count": 4, "passes_per_image": 1,
            "step": 5, "steps": 20, "progress": 0.3125, "pass_progress": 0.25, "pass_eta": None, "eta": None,
            "interrupted": False, "skipped": False, "stopping": False, "textinfo": "Loading B",
        })

    def test_other_tasks_get_queue_and_completion_only(self):
        view = _view()
        tracker = ProgressTracker()
        queued = progress_snapshot("task(b)", view, TASKS, tracker, 1.0)
        self.assertEqual((queued["active"], queued["queued"], queued["completed"]), (False, True, False))
        self.assertEqual((queued["queue_position"], queued["queue_size"]), (2, 2))
        for field in pa.JOB_FIELDS:
            self.assertIsNone(queued[field], field)
        finished = progress_snapshot("task(z)", view, TASKS, tracker, 1.0)
        self.assertEqual((finished["active"], finished["queued"], finished["completed"]), (False, False, True))
        unknown = progress_snapshot("task(q)", view, TASKS, tracker, 1.0)
        self.assertEqual((unknown["active"], unknown["queued"], unknown["completed"], unknown["busy"]),
                         (False, False, False, True))

    @unittest.skipUnless(FORGE_PROGRESS_PY.is_file(), "Forge's modules/progress.py is not next to the extension")
    def test_queue_flags_and_text_match_forge(self):
        # The page shows Forge's own queue text; built from our fields it must read the same.
        state = _state()
        pending = {"task(b)": 20.0, "task(c)": 10.0}
        forge = _forge_progressapi(state, "task(a)", pending, ["task(z)"], 5.0)
        tasks = read_tasks(types.SimpleNamespace(current_task="task(a)", pending_tasks=pending,
                                                 finished_tasks=["task(z)"]))
        for id_task in ("task(a)", "task(b)", "task(c)", "task(z)", "task(q)"):
            with self.subTest(id_task=id_task):
                native = forge(id_task)
                ours = progress_snapshot(id_task, read_state(state), tasks, ProgressTracker(), 5.0)
                self.assertEqual((ours["active"], ours["queued"], ours["completed"]),
                                 (native["active"], native["queued"], native["completed"]))
                if ours["queued"]:
                    self.assertEqual(f"In queue: {ours['queue_position']}/{ours['queue_size']}", native["textinfo"])
                elif not ours["active"]:
                    self.assertEqual(native["textinfo"], "Waiting...")

    def test_without_a_task_id_it_says_nothing_job_specific(self):
        # Forge's own /internal/progress tells nothing about a job without its id; neither does this route.
        state = _state(sampling_step=3, textinfo="SAM3 Refine: pass 1/2 — sampling")
        for asked in (None, ""):
            with self.subTest(asked=asked):
                busy = progress_snapshot(asked, read_state(state), TASKS, ProgressTracker(), 1.0)
                self.assertEqual((busy["id_task"], busy["busy"], busy["active"], busy["queued"], busy["completed"],
                                  busy["queue_position"], busy["queue_size"]), (None, True, False, False, False, None, 2))
                for field in pa.JOB_FIELDS:
                    self.assertIsNone(busy[field], field)
        idle = progress_snapshot(None, _view(job=""), TaskView(), ProgressTracker(), 1.0)
        self.assertEqual((idle["busy"], idle["active"], idle["textinfo"]), (False, False, None))

    def test_flags_are_reported_separately(self):
        payload = progress_snapshot(
            "task(a)", _view(interrupted=True, skipped=True, stopping_generation=True), TASKS, ProgressTracker(), 1.0)
        self.assertEqual((payload["interrupted"], payload["skipped"], payload["stopping"]), (True, True, True))

    def test_hires_reports_two_passes_per_image(self):
        payload = progress_snapshot("task(a)", _view(job_count=2, processing_has_refined_job_count=True),
                                    TASKS, ProgressTracker(), 1.0)
        self.assertEqual(payload["passes_per_image"], 2)

    def test_odd_state_values_read_as_idle_defaults(self):
        view = read_state(types.SimpleNamespace(job=None, job_no="x", job_count=float("nan"), sampling_step=None,
                                                sampling_steps="12", time_start="soon", textinfo=7))
        self.assertEqual(view, StateView(job="", job_no=0, job_count=0, step=0, steps=12, time_start=None,
                                         textinfo=None))
        self.assertEqual(read_state(None), StateView())

    def test_read_tasks_orders_the_queue_like_forge(self):
        module = types.SimpleNamespace(current_task="task(a)", pending_tasks={"task(b)": 3.0, "task(c)": 1.0,
                                                                               "task(d)": 2.0},
                                       finished_tasks=["task(z)"])
        self.assertEqual(read_tasks(module), TaskView("task(a)", ("task(c)", "task(d)", "task(b)"),
                                                      frozenset({"task(z)"})))
        self.assertEqual(read_tasks(None), TaskView())


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _forge_api_auth(value):
    """Forge started with ``--api --api-auth value`` (its loaded ``modules.shared.cmd_opts``)."""

    missing = object()
    saved = sys.modules.get("modules.shared", missing)
    sys.modules["modules.shared"] = types.SimpleNamespace(
        cmd_opts=types.SimpleNamespace(api_auth=value, api=True, nowebui=False))
    try:
        yield
    finally:
        if saved is missing:
            sys.modules.pop("modules.shared", None)
        else:
            sys.modules["modules.shared"] = saved


class _Forge:
    """Mutable state, tasks and clock behind the route."""

    def __init__(self):
        self.state = _state(sampling_step=0, sampling_steps=4)
        self.tasks = TaskView(current="task(a)", pending=("task(b)",), finished=frozenset({"task(z)"}))
        self.now = 100.0

    def app(self, app=None):
        app = app or FastAPI()
        register_progress_routes(app, state_provider=lambda: self.state, tasks_provider=lambda: self.tasks,
                                 clock=lambda: self.now)
        return app


class RouteTests(unittest.TestCase):
    def test_registered_once_as_a_get_route_outside_the_schema(self):
        app = FastAPI()
        forge = _Forge()
        self.assertTrue(forge.app(app) is app)
        self.assertFalse(register_progress_routes(app))
        routes = [route for route in app.routes if getattr(route, "path", None) == PROGRESS_API_PATH]
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].methods, {"GET"})
        self.assertFalse(routes[0].include_in_schema)
        with TestClient(app) as client:
            self.assertEqual(client.post(PROGRESS_API_PATH, headers=HEADERS).status_code, 405)
            self.assertNotIn(PROGRESS_API_PATH, client.get("/openapi.json").text)

    def test_answers_for_the_asked_task_without_caching(self):
        forge = _Forge()
        with TestClient(forge.app()) as client:
            response = client.get(PROGRESS_API_PATH, params={"id_task": "task(a)"}, headers=HEADERS)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")
            self.assertEqual(response.headers["pragma"], "no-cache")
            body = response.json()
            self.assertEqual((body["active"], body["step"], body["steps"], body["eta"]), (True, 0, 4, None))
            queued = client.get(PROGRESS_API_PATH, params={"id_task": "task(b)"}, headers=HEADERS).json()
            self.assertEqual((queued["queued"], queued["queue_position"], queued["step"]), (True, 1, None))
            done = client.get(PROGRESS_API_PATH, params={"id_task": "task(z)"}, headers=HEADERS).json()
            self.assertTrue(done["completed"])
            anything = client.get(PROGRESS_API_PATH, headers=HEADERS).json()
            self.assertEqual((anything["id_task"], anything["busy"], anything["active"], anything["step"]),
                             (None, True, False, None))

    def test_without_a_task_id_no_status_text_or_job_fields(self):
        forge = _Forge()
        forge.state = _state(sampling_step=2, sampling_steps=4, textinfo="Loading B")
        with TestClient(forge.app()) as client:
            asked = client.get(PROGRESS_API_PATH, params={"id_task": "task(a)"}, headers=HEADERS).json()
            self.assertEqual((asked["active"], asked["textinfo"], asked["step"]), (True, "Loading B", 2))
            for params in ({}, {"id_task": ""}):
                with self.subTest(params=params):
                    body = client.get(PROGRESS_API_PATH, params=params, headers=HEADERS).json()
                    self.assertEqual((body["id_task"], body["busy"], body["active"]), (None, True, False))
                    for field in pa.JOB_FIELDS:
                        self.assertIsNone(body[field], field)
                    self.assertNotIn("Loading B", json.dumps(body))

    def test_the_eta_memory_spans_requests(self):
        forge = _Forge()
        with TestClient(forge.app()) as client:
            ask = lambda: client.get(PROGRESS_API_PATH, params={"id_task": "task(a)"}, headers=HEADERS).json()  # noqa: E731
            self.assertIsNone(ask()["eta"])
            forge.now, forge.state = 101.0, _state(sampling_step=1, sampling_steps=4)
            body = ask()
            self.assertEqual((body["pass_eta"], body["eta"], body["elapsed"]), (3.0, 3.0, 1.0))

    def test_needs_the_same_origin_header(self):
        with TestClient(_Forge().app()) as client:
            self.assertEqual(client.get(PROGRESS_API_PATH).status_code, 403)
            self.assertEqual(client.get(PROGRESS_API_PATH, headers={"X-SAM3-Notebook": "0"}).status_code, 403)

    def test_refuses_an_overlong_task_id(self):
        with TestClient(_Forge().app()) as client:
            ok = client.get(PROGRESS_API_PATH, params={"id_task": "t" * 256}, headers=HEADERS)
            self.assertEqual(ok.status_code, 200)
            too_long = client.get(PROGRESS_API_PATH, params={"id_task": "t" * 257}, headers=HEADERS)
            self.assertEqual(too_long.status_code, 400)

    def test_reuses_the_gradio_login_guard(self):
        app = FastAPI()

        @app.get("/login_check")
        def login_check(x_test_auth: str | None = Header(default=None)):
            if x_test_auth != "ok":
                raise HTTPException(status_code=401, detail="Not authenticated")

        _Forge().app(app)
        with TestClient(app) as client:
            self.assertEqual(client.get(PROGRESS_API_PATH, headers=HEADERS).status_code, 401)
            ok = client.get(PROGRESS_API_PATH, headers={**HEADERS, "X-Test-Auth": "ok"})
            self.assertEqual(ok.status_code, 200)

    def test_api_auth_guards_it_like_sdapi(self):
        with _forge_api_auth("user:secret"):
            app = _Forge().app()
        with TestClient(app) as client:
            refused = client.get(PROGRESS_API_PATH, headers=HEADERS)
            self.assertEqual(refused.status_code, 401)
            self.assertEqual(refused.headers["www-authenticate"], "Basic")
            self.assertEqual(client.get(PROGRESS_API_PATH, headers=HEADERS, auth=("user", "nope")).status_code, 401)
            self.assertEqual(client.get(PROGRESS_API_PATH, headers=HEADERS, auth=("user", "secret")).status_code, 200)

    def test_defaults_read_forges_state_and_task_bookkeeping(self):
        state = _state(sampling_step=2, sampling_steps=8)
        progress = types.SimpleNamespace(current_task="task(a)", pending_tasks={"task(b)": 1.0}, finished_tasks=[])
        modules = types.ModuleType("modules")
        modules.shared = types.SimpleNamespace(state=state)
        modules.progress = progress
        with mock.patch.dict(sys.modules, {"modules": modules, "modules.shared": modules.shared,
                                           "modules.progress": progress}):
            self.assertIs(pa.forge_state(), state)
            self.assertEqual(pa.forge_tasks(), TaskView("task(a)", ("task(b)",), frozenset()))
            app = FastAPI()
            register_progress_routes(app)
            with TestClient(app) as client:
                body = client.get(PROGRESS_API_PATH, params={"id_task": "task(a)"}, headers=HEADERS).json()
        self.assertEqual((body["active"], body["step"], body["steps"]), (True, 2, 8))

    def test_outside_forge_it_reports_idle(self):
        with mock.patch.dict(sys.modules, {"modules": None}):
            self.assertIsNone(pa.forge_state())
            self.assertEqual(pa.forge_tasks(), TaskView())

    def test_handler_runs_without_a_threadpool(self):
        # A cheap read: the handler is a coroutine and never blocks the event loop.
        app = _Forge().app()
        route = next(route for route in app.routes if getattr(route, "path", None) == PROGRESS_API_PATH)
        self.assertTrue(asyncio.iscoroutinefunction(route.endpoint))


# ---------------------------------------------------------------------------
# Settings (scripts/appearance_progress_bar.py)
# ---------------------------------------------------------------------------


class _OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None,
                 **_ignored):
        self.default, self.label, self.component = default, label, component
        self.component_args, self.section, self.onchange = component_args, section, onchange
        self.info_text = None

    def info(self, text):
        self.info_text = text
        return self


def _load_script():
    """scripts/appearance_progress_bar.py with Forge's modules stubbed; returns (module, options, callbacks)."""

    added: dict = {}
    settings_callbacks, app_started = [], []
    callbacks = types.ModuleType("modules.script_callbacks")
    callbacks.on_ui_settings = lambda fn, *args, **kwargs: settings_callbacks.append(fn)
    callbacks.on_app_started = lambda fn, *, name=None: app_started.append((fn, name))
    shared = types.ModuleType("modules.shared")
    shared.OptionInfo = _OptionInfo
    shared.opts = types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info))
    ui_components = types.ModuleType("modules.ui_components")

    class FormColorPicker:  # noqa: D401 - stand-in for Forge's colour picker component
        pass

    ui_components.FormColorPicker = FormColorPicker
    modules = types.ModuleType("modules")
    modules.script_callbacks, modules.shared, modules.ui_components = callbacks, shared, ui_components
    stubs = {"modules": modules, "modules.script_callbacks": callbacks, "modules.shared": shared,
             "modules.ui_components": ui_components}
    path = ROOT / "scripts" / "appearance_progress_bar.py"
    spec = importlib.util.spec_from_file_location("_sam3_progress_bar_script", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
        for callback in settings_callbacks:
            callback()
    return module, added, app_started, FormColorPicker


class SettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.options, cls.app_started, cls.color_picker = _load_script()

    def test_every_option_is_in_the_progress_section_with_its_default(self):
        self.assertEqual(list(self.options), list(pa.DEFAULTS))
        for key, info in self.options.items():
            with self.subTest(key=key):
                self.assertTrue(key.startswith("sam3_progress_"))
                self.assertEqual(info.section, ("sam3_progress", "SAM Extra Progress Bar"))
                self.assertEqual(info.default, pa.DEFAULTS[key])
        self.assertIs(self.options[pa.OPT_ENABLED].default, False)

    def test_components_choices_and_ranges(self):
        options = self.options
        self.assertIs(options[pa.OPT_ENABLED].component, gr.Checkbox)
        self.assertIs(options[pa.OPT_CUSTOM_COLOR].component, self.color_picker)
        for key, component in ((pa.OPT_SMOOTHNESS, gr.Radio), (pa.OPT_TEXT_FORMAT, gr.Dropdown),
                               (pa.OPT_AFTER_FINISH, gr.Radio), (pa.OPT_INTERRUPT_STYLE, gr.Radio),
                               (pa.OPT_COLOR, gr.Dropdown)):
            with self.subTest(key=key):
                self.assertIs(options[key].component, component)
                values = [value for _label, value in options[key].component_args["choices"]]
                self.assertEqual(tuple(values), pa.CHOICE_VALUES[key])
                self.assertIn(options[key].default, values)
        for key, bounds in ((pa.OPT_HEIGHT, {"minimum": 10, "maximum": 50, "step": 5}),
                            (pa.OPT_FADE_SECONDS, {"minimum": 0.1, "maximum": 4.0, "step": 0.1}),
                            (pa.OPT_TEXT_ALIGN, {"minimum": 0, "maximum": 100, "step": 5})):
            with self.subTest(key=key):
                self.assertIs(options[key].component, gr.Slider)
                self.assertEqual(options[key].component_args, bounds)
                self.assertTrue(bounds["minimum"] <= options[key].default <= bounds["maximum"])

    def test_upstreams_modes_are_all_there(self):
        # upstream SMOOTHNESS_KEYS / FORMAT_KEYS / FINISH_KEYS / INTERRUPT_KEYS (smooth-progress.js:30-33)
        self.assertEqual(pa.CHOICE_VALUES[pa.OPT_SMOOTHNESS], ("smooth_gt_acc", "smooth_eq_acc", "smooth_lt_acc"))
        self.assertEqual(pa.CHOICE_VALUES[pa.OPT_TEXT_FORMAT],
                         ("steps_pct_eta", "steps_eta", "steps_pct", "pct_eta", "eta_only", "steps_only",
                          "pct_only", "none"))
        self.assertEqual(pa.CHOICE_VALUES[pa.OPT_AFTER_FINISH], ("fade", "fade_text_only", "keep"))
        # All four interruption styles are selectable (upstream's slider stopped at index 2).
        self.assertEqual(pa.CHOICE_VALUES[pa.OPT_INTERRUPT_STYLE],
                         ("text", "red_text", "text_red_bar", "red_text_red_bar"))
        self.assertEqual(pa.DEFAULTS[pa.OPT_INTERRUPT_STYLE], "red_text_red_bar")

    def test_routes_are_registered_at_app_start_once(self):
        self.assertEqual(len(self.app_started), 1)
        callback, name = self.app_started[0]
        self.assertEqual(name, "sam-extra-progress-api")
        app = FastAPI()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            callback(None, app)
            callback(None, app)   # Reload UI re-fires app start
        self.assertEqual(sum(getattr(route, "path", None) == PROGRESS_API_PATH for route in app.routes), 1)
        self.assertEqual(out.getvalue().count(PROGRESS_API_PATH), 1)

    def test_a_registration_failure_never_breaks_app_start(self):
        callback, _name = self.app_started[0]
        with mock.patch.object(pa, "register_progress_routes", side_effect=RuntimeError("boom")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            callback(None, FastAPI())
        self.assertIn("RuntimeError: boom", err.getvalue())


# ---------------------------------------------------------------------------
# What javascript/progress_bar.js and style.css must agree on
# ---------------------------------------------------------------------------


def _strip_js_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$|\s//[^\n\"']*$", "", source)


class PageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = (ROOT / "javascript" / "progress_bar.js").read_text(encoding="utf-8")
        cls.code = _strip_js_comments(cls.js)
        cls.css = (ROOT / "style.css").read_text(encoding="utf-8")
        start = cls.css.index("/* sam3-progress:begin */")
        end = cls.css.index("/* sam3-progress:end */")
        cls.block = re.sub(r"/\*.*?\*/", "", cls.css[start:end], flags=re.DOTALL)

    def _js_object(self, name):
        match = re.search(r"var " + name + r" = Object\.freeze\(\{(.*?)\n    \}\);", self.js, re.DOTALL)
        self.assertIsNotNone(match, name)
        return {key: json.loads(value) for key, value in
                re.findall(r"^\s*(\w+):\s*(.+?),\s*$", match.group(1), re.MULTILINE)}

    def test_page_defaults_choices_and_ranges_match_the_settings(self):
        self.assertEqual(self._js_object("DEFAULTS"), dict(pa.DEFAULTS))
        self.assertEqual({key: tuple(values) for key, values in self._js_object("CHOICES").items()},
                         dict(pa.CHOICE_VALUES))
        ranges = self._js_object("RANGES")
        self.assertEqual(ranges, {
            pa.OPT_TEXT_ALIGN: [pa.ALIGN_RANGE["minimum"], pa.ALIGN_RANGE["maximum"]],
            pa.OPT_FADE_SECONDS: [pa.FADE_RANGE["minimum"], pa.FADE_RANGE["maximum"]],
            pa.OPT_HEIGHT: [pa.HEIGHT_RANGE["minimum"], pa.HEIGHT_RANGE["maximum"]],
        })
        self.assertEqual(self._js_object("PRESET_COLORS"), dict(pa.PRESET_COLORS))
        self.assertIn(f'var API_PATH = "{PROGRESS_API_PATH}";', self.js)

    def test_page_wraps_requestprogress_and_never_touches_the_native_bar_or_storage(self):
        self.assertIn("window.requestProgress = wrapper;", self.code)
        self.assertIn('"X-SAM3-Notebook": "1"', self.code)
        self.assertIn('credentials: "same-origin"', self.code)
        for forbidden in ("progressDiv", "localStorage", "sessionStorage", ".textContent", ".innerHTML",
                          "setInterval"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.code)
        self.assertIn("node.data = text", self.code)
        self.assertIn('getElementById(UPSTREAM_STYLE_ID)', self.code)
        self.assertIn('var UPSTREAM_STYLE_ID = "spb-dynamic-css";', self.js)

    def test_native_bar_is_hidden_not_removed_and_only_next_to_ours(self):
        # Hidden only next to our bar and only while it holds a job Forge's requestProgress still runs.
        self.assertIn('html[data-sam3-progress="on"] .sam3-progress[data-native="hide"] ~ .progressDiv,', self.block)
        self.assertIn('html[data-sam3-progress="on"] .sam3-progress[data-native="hide"] ~ * .progressDiv {\n'
                      '    display: none !important;', self.block.replace("\r\n", "\n"))
        self.assertEqual(self.block.count("progressDiv"), 2)
        self.assertIn('setAttr(bar, "data-native", alive(bar.job) || bar.backlog.some(alive) ? "hide" : "show");',
                      self.code)

    def test_bar_css_uses_theme_tokens_and_no_gradient_or_glow(self):
        lowered = self.block.lower()
        for forbidden in ("gradient", "drop-shadow", "text-shadow", "box-shadow", "filter:", "z-index",
                          "animation", "@keyframes", "#", "rgb("):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)
        for token in ("var(--sam3-color-accent)", "var(--sam3-color-error)", "var(--sam3-color-error-ink)",
                      "var(--sam3-color-paper-3)", "var(--sam3-progress-height)"):
            self.assertIn(token, self.block)

    def test_every_derived_file_keeps_the_upstream_notice(self):
        for relative in ("javascript/progress_bar.js", "sam3ext/progress_api.py",
                         "scripts/appearance_progress_bar.py"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            for line in MIT_LINES:
                with self.subTest(file=relative, line=line):
                    self.assertIn(line, text)
            self.assertIn("7fe5810", text, relative)


if __name__ == "__main__":
    unittest.main()
