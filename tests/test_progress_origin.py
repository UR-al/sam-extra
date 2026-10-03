"""Origin-parity tests for the progress bar's per-step ETA and route (sam3ext/progress_api.py).

The oracle is upstream itself: diamfang/sd-webui-smooth-progress@7fe58101 ``scripts/smooth-progress.py``,
loaded verbatim from ``tests/_origin_smooth_progress.py`` (MIT, Copyright (c) 2026 diamfang — the notice
is in that file's header and in THIRD_PARTY_NOTICES.md; its SHA-256 is pinned below) with Forge's
``modules`` stubbed, a fake FastAPI app that captures the ``/smooth-progress/api`` handler and a fake
clock. Both sides see the same ``shared.state`` at the same instants.

What must match upstream:

* ``step_eta`` — every estimate and tracker on the same observations as ``_step_eta``: steady and
  jittery steps, several steps between looks, a step going back, late looks.
* The route, for jobs without Hires. fix whose passes are each seen from step 0: ``step``,
  ``total_steps``/``steps``, ``progress``/``pass_progress``, ``elapsed``, ``time_start``,
  ``job_running``/``busy`` and ``eta``/``pass_eta`` (upstream reports at least 0.1 s).

Kept differences, each asserted below so it stays deliberate: an unknown ETA is ``null`` (upstream
0.1, shown as a flickering "1s"), ``skipped`` is not ``interrupted``, hires passes keep their own step
average, and a new pass restarts the step clock even when no look saw its step 0.
"""
from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.progress_api import (  # noqa: E402
    ProgressTracker,
    TaskView,
    progress_snapshot,
    read_state,
    step_eta,
)

ORIGIN_FILE = ROOT / "tests" / "_origin_smooth_progress.py"
ORIGIN_MARKER = "# ---- upstream scripts/smooth-progress.py below (verbatim) ----\n"
# SHA-256 of diamfang/sd-webui-smooth-progress@7fe58101ae315af1b5e6eda6080e72ac45bfe846:scripts/smooth-progress.py
# (LF, no final newline — the file exactly as the GitHub API serves it; same blob at 2b966fc).
ORIGIN_SHA256 = "754639abd8ffd910d97f9d890506e6be03698aae8a39c2de96de61516f6c3ed5"


class _FakeApp:
    def __init__(self):
        self.routes = {}

    def get(self, path):
        def register(handler):
            self.routes[path] = handler
            return handler
        return register


class _Upstream:
    """Upstream's module: stubbed ``modules``, its handler captured, ``time.time`` driven by the test."""

    def __init__(self):
        self.now = 0.0
        started = []
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_app_started = started.append
        self.shared = types.ModuleType("modules.shared")
        self.shared.state = None
        modules = types.ModuleType("modules")
        modules.script_callbacks, modules.shared = callbacks, self.shared
        spec = importlib.util.spec_from_file_location(f"_origin_smooth_progress_{id(self)}", ORIGIN_FILE)
        self.module = importlib.util.module_from_spec(spec)
        stubs = {"modules": modules, "modules.script_callbacks": callbacks, "modules.shared": self.shared}
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(self.module)
        self.module.time = types.SimpleNamespace(time=lambda: self.now)
        app = _FakeApp()
        self_started = started[0]
        self_started(None, app)
        self.handler = app.routes["/smooth-progress/api"]

    def get(self, state, now):
        self.shared.state, self.now = state, now
        coroutine = self.handler()
        try:
            coroutine.send(None)      # the handler never awaits: it finishes on the first send
        except StopIteration as done:
            return done.value
        raise AssertionError("upstream handler awaited something")


def _state(job_no=0, job_count=1, step=0, steps=20, time_start=1000.0, **flags):
    fields = dict(job="task(a)", job_no=job_no, job_count=job_count, sampling_step=step, sampling_steps=steps,
                  time_start=time_start, interrupted=False, skipped=False, stopping_generation=False,
                  processing_has_refined_job_count=False, textinfo=None)
    fields.update(flags)
    return types.SimpleNamespace(**fields)


def _job_looks(passes, *, poll, time_start=1000.0, setup=0.35, gap=0.6, rng=None, refined=False):
    """Looks at one Forge job: ``passes`` = [(steps, step_seconds), ...], one per ``job_no``.

    Each pass starts with ``setup`` seconds at step 0, its steps then take ``step_seconds`` (times a
    random factor when ``rng`` is given), and ``gap`` seconds of decode/save follow (still the old
    ``job_no``, like Forge between ``sample`` and ``nextjob``). ``sampling_step`` is the index of the
    step being computed, as Forge's sampler callback leaves it.
    """

    timeline = []          # (start, end, job_no, step, steps)
    t = time_start
    for job_no, (steps, seconds) in enumerate(passes):
        timeline.append((t, t + setup, job_no, 0, steps))
        t += setup
        for step in range(steps):
            duration = seconds * (rng.uniform(0.6, 1.6) if rng else 1.0)
            timeline.append((t, t + duration, job_no, step, steps))
            t += duration
        timeline.append((t, t + gap, job_no, steps - 1, steps))
        t += gap
    looks = []
    now = time_start + poll / 2
    index = 0
    while now < t:
        while timeline[index][1] <= now:
            index += 1
        _start, _end, job_no, step, steps = timeline[index]
        looks.append((_state(job_no=job_no, job_count=len(passes), step=step, steps=steps,
                             time_start=time_start, processing_has_refined_job_count=refined), now))
        now += poll * (rng.uniform(0.5, 2.5) if rng else 1.0)
    return looks


class OriginCopyTests(unittest.TestCase):
    def test_vendored_upstream_file_is_unchanged(self):
        text = ORIGIN_FILE.read_text(encoding="utf-8")
        self.assertEqual(text.count(ORIGIN_MARKER), 1)
        body = text.split(ORIGIN_MARKER, 1)[1]
        self.assertEqual(hashlib.sha256(body.encode("utf-8")).hexdigest(), ORIGIN_SHA256)
        self.assertIn("Copyright (c) 2026 diamfang", text.split(ORIGIN_MARKER, 1)[0])


class StepEtaOriginTests(unittest.TestCase):
    """origin: scripts/smooth-progress.py:21-54 ``_step_eta``."""

    def setUp(self):
        self.upstream = _Upstream().module._step_eta

    def _compare(self, observations, steps):
        ours = theirs = None
        for step, now in observations:
            our_eta, ours = step_eta(ours, step, steps, now)
            their_eta, theirs = self.upstream(theirs, step, steps, now, 0.0)
            self.assertEqual(our_eta, their_eta, (step, now))
            self.assertEqual(ours, dict(theirs), (step, now))

    def test_steady_steps_looked_at_every_100_ms(self):
        self._compare([(min(19, int(i * 0.1 / 0.45)), 50.0 + i * 0.1) for i in range(200)], 20)

    def test_jittery_steps_and_irregular_looks(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                now, step, observations = 10.0, 0, []
                for _ in range(300):
                    now += rng.uniform(0.01, 1.5)
                    step = min(39, step + rng.choice((0, 0, 0, 1, 1, 2, 5)))
                    observations.append((step, now))
                self._compare(observations, 40)

    def test_a_step_going_back_and_late_looks(self):
        self._compare([(0, 0.0), (3, 3.0), (3, 9.0), (0, 9.5), (1, 10.0), (1, 30.0), (7, 31.0), (2, 31.0)], 8)

    def test_many_jobs_of_random_shape(self):
        rng = random.Random(1234)
        for job in range(30):
            with self.subTest(job=job):
                steps = rng.randint(1, 60)
                now, step, observations = 0.0, 0, []
                while step < steps - 1:
                    now += rng.uniform(0.0, 0.7)
                    step = min(steps - 1, step + rng.choice((0, 1, 1, 3)))
                    observations.append((step, now))
                self._compare(observations, steps)


class RouteOriginTests(unittest.TestCase):
    """origin: scripts/smooth-progress.py:57-112 ``register_api`` → ``GET /smooth-progress/api``."""

    def _compare(self, looks):
        upstream = _Upstream()
        tracker = ProgressTracker()
        tasks = TaskView(current="task(a)")
        compared = 0
        for state, now in looks:
            theirs = upstream.get(state, now)
            ours = progress_snapshot("task(a)", read_state(state), tasks, tracker, now)
            with self.subTest(now=now, job_no=state.job_no, step=state.sampling_step):
                self.assertEqual(ours["step"], theirs["step"])
                self.assertEqual(ours["steps"], theirs["total_steps"])
                self.assertEqual(ours["pass_progress"], theirs["progress"])
                self.assertEqual(ours["elapsed"], theirs["elapsed"])
                self.assertEqual(ours["time_start"], theirs["time_start"])
                self.assertEqual(ours["busy"], theirs["job_running"])
                self.assertEqual(ours["interrupted"] or ours["skipped"], theirs["interrupted"])
                if theirs["active"]:
                    # Upstream reports at least 0.1 s; an unknown estimate is null here.
                    self.assertEqual(max(0.1, ours["pass_eta"] or 0.0), theirs["eta"])
                    compared += 1
        return compared

    def test_single_pass_job(self):
        self.assertGreater(self._compare(_job_looks([(20, 0.45)], poll=0.1)), 90)   # 9.95 s at 100 ms

    def test_batch_of_passes(self):
        self.assertGreater(self._compare(_job_looks([(12, 0.3), (12, 0.3), (12, 0.3)], poll=0.1)), 130)   # 13.65 s

    def test_jittery_steps_and_irregular_polls(self):
        # Looks at most 0.5 s apart and 0.6 s of setup per pass: every pass is first seen at step 0,
        # as upstream's tracker needs (see KeptDifferenceTests for a pass first seen later).
        for seed in range(8):
            rng = random.Random(seed)
            passes = [(rng.randint(3, 30), rng.uniform(0.05, 0.6)) for _ in range(rng.randint(1, 4))]
            with self.subTest(seed=seed, passes=passes):
                self.assertGreater(self._compare(_job_looks(passes, poll=0.2, setup=0.6, rng=rng)), 0)

    def test_two_jobs_in_a_row(self):
        looks = _job_looks([(8, 0.25)], poll=0.1, time_start=1000.0)
        looks += _job_looks([(6, 0.4), (6, 0.4)], poll=0.1, time_start=1010.0)
        self.assertGreater(self._compare(looks), 80)


class KeptDifferenceTests(unittest.TestCase):
    def test_an_unknown_eta_is_null_where_upstream_said_a_tenth_of_a_second(self):
        upstream, tracker = _Upstream(), ProgressTracker()
        state = _state(step=0, steps=20)
        theirs = upstream.get(state, 1000.2)
        ours = progress_snapshot("task(a)", read_state(state), TaskView("task(a)"), tracker, 1000.2)
        self.assertEqual((theirs["active"], theirs["eta"]), (True, 0.1))
        self.assertIsNone(ours["pass_eta"])
        self.assertIsNone(ours["eta"])

    def test_skip_is_not_an_interruption_here(self):
        upstream = _Upstream()
        state = _state(step=4, steps=20, skipped=True)
        theirs = upstream.get(state, 1003.0)
        ours = progress_snapshot("task(a)", read_state(state), TaskView("task(a)"), ProgressTracker(), 1003.0)
        self.assertTrue(theirs["interrupted"])
        self.assertEqual((ours["interrupted"], ours["skipped"]), (False, True))

    def _run(self, looks):
        upstream, tracker = _Upstream(), ProgressTracker()
        last = None
        for state, now in looks:
            theirs = upstream.get(state, now)
            ours = progress_snapshot("task(a)", read_state(state), TaskView("task(a)"), tracker, now)
            last = (theirs, ours)
        return last

    def test_hires_steps_keep_their_own_average(self):
        # Base pass: 4 steps at 0.5 s; hires pass: 4 steps at 2 s. One hires step in, 3 remain (~6 s).
        looks = [(state, now) for state, now in
                 _job_looks([(4, 0.5), (4, 2.0)], poll=0.1, setup=0.0, gap=0.0, refined=True)
                 if state.job_no == 0 or state.sampling_step <= 1]
        theirs, ours = self._run(looks)
        self.assertEqual((ours["job_no"], ours["step"]), (1, 1))
        true_remaining = 3 * 2.0 - (looks[-1][1] - (1000.0 + 4 * 0.5 + 2.0))
        self.assertAlmostEqual(ours["pass_eta"], true_remaining, delta=0.11)
        # Upstream mixes the 0.5 s base steps in and promises the hires pass far too early.
        self.assertLess(theirs["eta"], true_remaining - 1.5)

    def test_a_new_pass_seen_after_its_step_zero_restarts_the_step_clock(self):
        # 1 s steps; the look after step 1 of pass 0 lands at step 3 of pass 1, 8 s later (the rest of
        # pass 0, its decode and save, and pass 1's setup and first steps).
        looks = [(_state(job_no=0, job_count=2, step=0, steps=4), 1000.0),
                 (_state(job_no=0, job_count=2, step=1, steps=4), 1001.0),
                 (_state(job_no=1, job_count=2, step=3, steps=4), 1009.0)]
        theirs, ours = self._run(looks)
        self.assertEqual(ours["pass_eta"], 1.0)          # one 1 s step left
        self.assertEqual(theirs["eta"], 3.0)             # 8 s counted as two steps: average 3 s
