"""Smooth progress bar backend — ``GET /sam-extra/progress`` and its step/ETA snapshot logic.

Derived from diamfang/sd-webui-smooth-progress@7fe58101ae315af1b5e6eda6080e72ac45bfe846 (the code is
identical at 2b966fc), ``scripts/smooth-progress.py``: ``_step_eta`` (:21-54) and the
``/smooth-progress/api`` handler (:57-112). The page side is ``javascript/progress_bar.js``; the
Settings are registered by ``scripts/appearance_progress_bar.py``. Upstream licence:

    MIT License

    Copyright (c) 2026 diamfang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Changes made by sam-extra (2026-10-03):

* ``step_eta`` is ``_step_eta`` without its unused ``elapsed`` argument, returning a new tracker
  instead of editing the caller's; the arithmetic is unchanged (``tests/test_progress_origin.py``
  runs both on the same observations).
* One step tracker per *pass kind* instead of one per job. Without Hires. fix every pass is one kind,
  which is upstream's behaviour; with it, first and hires passes alternate (Forge doubles
  ``job_count``) and get separate step averages, so a slow hires step no longer skews the first pass.
  A new pass (``job_no`` changed) restarts its step clock even when no look saw its step 0, so the
  decode/save/setup between passes is never counted as step time (upstream only restarted when it
  saw the step go back). Known limitation: the kind is read from ``job_no`` parity. ADetailer and
  SAM3's postprocess passes raise ``job_count`` and run their own ``nextjob`` inside an image, so with
  Hires. fix, batch count > 1 and an odd number of such passes the later images' first and hires
  passes swap kinds and the two step averages mix. Only the ETA suffers; progress is Forge's formula.
* Whole-job progress and ETA. ``progress`` is Forge's own formula (``modules/progress.py``
  ``progressapi``: ``job_no``/``job_count`` plus the current pass); ``eta`` adds the passes still to
  run, each taking as long as the last finished pass of its kind (or, before one has finished, as the
  current pass will). ``pass_eta`` is the per-step estimate upstream reported.
* The route answers for one Forge task (``?id_task=``): ``active``/``queued``/``completed`` and the
  queue position, so a page draws only the jobs it started and shows Forge's queue text otherwise.
  Without ``id_task`` it says nothing job-specific (no job fields, no status text), like Forge's own
  ``/internal/progress`` without the task's id; upstream's route reported whatever ran to anyone.
* An unknown ETA is ``null``. Upstream reported ``0.1`` s, which the page showed as a flickering "1s".
* ``interrupted`` no longer folds in ``skipped``: Skip moves on to the next batch item, the job goes
  on. ``stopping`` ("Don't Interrupt in the middle") is reported separately.
* Guarded like every sam-extra route (``notebook_store.extension_auth_dependencies`` plus the
  ``X-SAM3-Notebook: 1`` header), ``Cache-Control: no-store`` and left out of the OpenAPI schema.
  Upstream's route had no guard and was polled every 100 ms by every open page.

Route — ``GET /sam-extra/progress[?id_task=task(...)]`` → 200::

    {"version": 1, "id_task", "busy", "active", "queued", "completed", "queue_position",
     "queue_size", "server_time",
     "time_start", "elapsed", "job_no", "job_count", "passes_per_image", "step", "steps",
     "progress", "pass_progress", "pass_eta", "eta", "interrupted", "skipped", "stopping",
     "textinfo"}

The second row is ``null`` unless ``active``. Without ``id_task`` only ``busy``, ``queue_size`` and
``server_time`` are filled: ``active``/``queued``/``completed`` are false and no job field or status
text is given. 400 when ``id_task`` is longer than 256 characters; 401/403 from the guards.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .notebook_store import extension_auth_dependencies, require_same_origin_header


__all__ = [
    "API_VERSION",
    "DEFAULTS",
    "PROGRESS_API_PATH",
    "ProgressTracker",
    "StateView",
    "TaskView",
    "average_step_seconds",
    "job_progress",
    "pass_kind",
    "pass_progress",
    "progress_snapshot",
    "read_state",
    "read_tasks",
    "register_progress_routes",
    "step_eta",
]

PROGRESS_API_PATH = "/sam-extra/progress"
API_VERSION = 1
MAX_TASK_ID_LENGTH = 256
_NO_STORE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}

# The fields that describe the asked-about job; ``null`` while it is not the running one.
JOB_FIELDS = (
    "time_start",
    "elapsed",
    "job_no",
    "job_count",
    "passes_per_image",
    "step",
    "steps",
    "progress",
    "pass_progress",
    "pass_eta",
    "eta",
    "interrupted",
    "skipped",
    "stopping",
    "textinfo",
)


# ---------------------------------------------------------------------------
# Settings (registered by scripts/appearance_progress_bar.py, read by javascript/progress_bar.js — the page keeps
# the same defaults and choice values, tests/test_progress_api.py compares the three files).
# ---------------------------------------------------------------------------

SETTINGS_SECTION = ("sam3_progress", "SAM Extra Progress Bar")

OPT_ENABLED = "sam3_progress_enabled"
OPT_SMOOTHNESS = "sam3_progress_smoothness"
OPT_TEXT_FORMAT = "sam3_progress_text_format"
OPT_TEXT_ALIGN = "sam3_progress_text_align"
OPT_AFTER_FINISH = "sam3_progress_after_finish"
OPT_FADE_SECONDS = "sam3_progress_fade_seconds"
OPT_INTERRUPT_STYLE = "sam3_progress_interrupt_style"
OPT_HEIGHT = "sam3_progress_height"
OPT_COLOR = "sam3_progress_color"
OPT_CUSTOM_COLOR = "sam3_progress_custom_color"

# (label, value). Values are upstream's keys (SMOOTHNESS_KEYS, FORMAT_KEYS, FINISH_KEYS, INTERRUPT_KEYS).
SMOOTHNESS_CHOICES = (
    ("Smooth > Accurate — ETA 에 맞춰 일정한 속도로", "smooth_gt_acc"),
    ("Smooth ~ Accurate — 서버 진행률을 부드럽게 따라감", "smooth_eq_acc"),
    ("Smooth < Accurate — 스텝 단위로 정확하게", "smooth_lt_acc"),
)
TEXT_FORMAT_CHOICES = (
    ("Steps • % • ETA", "steps_pct_eta"),
    ("Steps • ETA", "steps_eta"),
    ("Steps • %", "steps_pct"),
    ("% • ETA", "pct_eta"),
    ("ETA 만", "eta_only"),
    ("Steps 만", "steps_only"),
    ("% 만", "pct_only"),
    ("글자 없음", "none"),
)
AFTER_FINISH_CHOICES = (
    ("막대와 글자를 서서히 숨김", "fade"),
    ("글자만 서서히 숨김", "fade_text_only"),
    ("그대로 둠", "keep"),
)
# All four of upstream's modes. Its slider stopped at index 2, so the fourth (its own default) could
# only be reached by never touching the slider.
INTERRUPT_STYLE_CHOICES = (
    ("글자", "text"),
    ("빨간 글자", "red_text"),
    ("글자 + 빨간 막대", "text_red_bar"),
    ("빨간 글자 + 빨간 막대", "red_text_red_bar"),
)
# Upstream's solid presets (COLOR_PRESETS blue/green/red/dandelion) plus the two theme colours. Its
# gradient and animated presets are not ported (design.md: no gradients, no glow).
PRESET_COLORS = MappingProxyType({
    "blue": "#2563eb",
    "green": "#059669",
    "red": "#dc2626",
    "dandelion": "#FEDF08",
})
COLOR_CHOICES = (
    ("테마 강조색", "accent"),
    ("테마 성공색", "success"),
    ("Blue (#2563eb)", "blue"),
    ("Green (#059669)", "green"),
    ("Red (#dc2626)", "red"),
    ("Dandelion (#FEDF08)", "dandelion"),
    ("직접 지정 — 아래 색", "custom"),
)

# Upstream: 20 px x 0.5..2.5 in 0.25 steps; its stored value was never clamped.
HEIGHT_RANGE = MappingProxyType({"minimum": 10, "maximum": 50, "step": 5})
# Upstream fadeDurationSec: 0.1..4.0, step 0.1, default 0.4.
FADE_RANGE = MappingProxyType({"minimum": 0.1, "maximum": 4.0, "step": 0.1})
# Upstream textAlignIdx 0..20 = 0..100 % in 5 % steps, default 10 = centre.
ALIGN_RANGE = MappingProxyType({"minimum": 0, "maximum": 100, "step": 5})

DEFAULTS: Mapping[str, Any] = MappingProxyType({
    OPT_ENABLED: False,
    OPT_SMOOTHNESS: "smooth_gt_acc",
    OPT_TEXT_FORMAT: "steps_pct_eta",
    OPT_TEXT_ALIGN: 50,
    OPT_AFTER_FINISH: "fade",
    OPT_FADE_SECONDS: 0.4,
    OPT_INTERRUPT_STYLE: "red_text_red_bar",
    OPT_HEIGHT: 20,
    OPT_COLOR: "accent",
    OPT_CUSTOM_COLOR: "#ef256c",   # upstream DEFAULT_CUSTOM_COLORS.solid
})

CHOICE_VALUES: Mapping[str, tuple] = MappingProxyType({
    OPT_SMOOTHNESS: tuple(value for _label, value in SMOOTHNESS_CHOICES),
    OPT_TEXT_FORMAT: tuple(value for _label, value in TEXT_FORMAT_CHOICES),
    OPT_AFTER_FINISH: tuple(value for _label, value in AFTER_FINISH_CHOICES),
    OPT_INTERRUPT_STYLE: tuple(value for _label, value in INTERRUPT_STYLE_CHOICES),
    OPT_COLOR: tuple(value for _label, value in COLOR_CHOICES),
})


# ---------------------------------------------------------------------------
# Per-step ETA (upstream _step_eta)
# ---------------------------------------------------------------------------


def step_eta(tracker: Mapping[str, Any] | None, step, steps, now: float) -> tuple[float, dict]:
    """Remaining seconds of the current pass from per-step durations — upstream ``_step_eta``.

    The raw ``elapsed / progress - elapsed`` ratio climbs inside each step because ``sampling_step``
    stays put while the clock runs. Instead the tracker keeps how long completed steps took and the
    estimate subtracts the time already spent in the current step::

        eta = (steps - step) * avg_step_sec - min(time_in_step, avg_step_sec)

    so it counts down within a step and drops by one step at each boundary. Returns
    ``(eta, tracker)``; ``eta`` is ``0.0`` until a step has finished (``average_step_seconds`` tells
    the two zeros apart). A step that goes back (a new pass) restarts the step clock and adds nothing.
    """

    if tracker is None:
        return 0.0, {"step": int(step), "step_ts": now, "dur_sum": 0.0, "dur_n": 0}

    tracker = dict(tracker)
    if tracker["step"] != step:
        advanced = step - tracker["step"]
        dt = max(0.0, now - tracker["step_ts"])
        if advanced > 0:
            tracker["dur_sum"] += dt
            tracker["dur_n"] += advanced
        tracker["step"] = int(step)
        tracker["step_ts"] = now

    if tracker["dur_n"] <= 0:
        return 0.0, tracker

    avg_step_sec = tracker["dur_sum"] / tracker["dur_n"]
    if avg_step_sec <= 0.0:
        return 0.0, tracker

    # Time already spent in the current step, capped at one step so late polls never go negative.
    in_step_sec = min(now - tracker["step_ts"], avg_step_sec)
    eta = (steps - step) * avg_step_sec - in_step_sec
    return max(0.0, eta), tracker


def average_step_seconds(tracker: Mapping[str, Any] | None) -> float | None:
    """Mean finished-step duration of a ``step_eta`` tracker, ``None`` while unknown."""

    if not tracker or tracker.get("dur_n", 0) <= 0:
        return None
    average = tracker["dur_sum"] / tracker["dur_n"]
    return average if average > 0.0 else None


# ---------------------------------------------------------------------------
# Forge state, read defensively
# ---------------------------------------------------------------------------


def _int(value, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return int(number)


def _finite_or_none(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class StateView:
    """The ``shared.state`` fields the bar needs (``modules/shared_state.py`` ``State``)."""

    job: str = ""
    job_no: int = 0
    job_count: int = 0
    step: int = 0
    steps: int = 0
    time_start: float | None = None
    interrupted: bool = False
    skipped: bool = False
    stopping: bool = False
    refined: bool = False
    textinfo: str | None = None

    @property
    def busy(self) -> bool:
        """``State.begin`` sets ``job`` (the task id, later "Batch n out of m"); ``State.end`` clears it."""
        return bool(self.job)


def read_state(state: Any) -> StateView:
    """Snapshot of Forge's ``shared.state``; missing or odd attributes read as idle values."""

    if state is None:
        return StateView()
    textinfo = getattr(state, "textinfo", None)
    job = getattr(state, "job", "")
    return StateView(
        job=job if isinstance(job, str) else ("" if job is None else str(job)),
        job_no=_int(getattr(state, "job_no", 0)),
        job_count=_int(getattr(state, "job_count", 0)),
        step=_int(getattr(state, "sampling_step", 0)),
        steps=_int(getattr(state, "sampling_steps", 0)),
        time_start=_finite_or_none(getattr(state, "time_start", None)),
        interrupted=bool(getattr(state, "interrupted", False)),
        skipped=bool(getattr(state, "skipped", False)),
        stopping=bool(getattr(state, "stopping_generation", False)),
        refined=bool(getattr(state, "processing_has_refined_job_count", False)),
        textinfo=textinfo if isinstance(textinfo, str) and textinfo else None,
    )


@dataclass(frozen=True)
class TaskView:
    """Forge's task bookkeeping (``modules/progress.py``): running, queued (oldest first), finished."""

    current: str | None = None
    pending: tuple[str, ...] = ()
    finished: frozenset = frozenset()


def read_tasks(progress_module: Any) -> TaskView:
    """``current_task`` / ``pending_tasks`` / ``finished_tasks`` of ``modules.progress``.

    The queue is ordered by the time each task was added, like Forge's own "In queue: i/n".
    """

    if progress_module is None:
        return TaskView()
    current = getattr(progress_module, "current_task", None)
    pending_map = getattr(progress_module, "pending_tasks", None) or {}
    try:
        items = list(pending_map.items())
    except (AttributeError, RuntimeError):   # not a mapping, or changed while it was copied
        items = []
    pending = tuple(task for task, _added in sorted(items, key=lambda item: item[1]))
    try:
        finished = frozenset(getattr(progress_module, "finished_tasks", ()) or ())
    except (TypeError, RuntimeError):
        finished = frozenset()
    return TaskView(current if isinstance(current, str) else None, pending, finished)


def forge_state() -> Any:
    """Forge's ``shared.state`` (``None`` outside Forge)."""

    try:
        from modules import shared
    except Exception:  # noqa: BLE001 - outside Forge there is nothing to report
        return None
    return getattr(shared, "state", None)


def forge_tasks() -> TaskView:
    """Forge's task bookkeeping, read from ``modules.progress`` at call time (module globals change)."""

    try:
        from modules import progress
    except Exception:  # noqa: BLE001
        return TaskView()
    return read_tasks(progress)


# ---------------------------------------------------------------------------
# Progress and ETA
# ---------------------------------------------------------------------------


def job_progress(job_no: int, job_count: int, step: int, steps: int) -> float:
    """Whole-job fraction — Forge's ``progressapi`` formula, clamped to ``[0, 1]``.

    ``job_no`` can pass ``job_count`` (an ADetailer pass inside the last image calls ``nextjob``
    again), and ``job_count`` is ``-1`` between ``State.begin`` and the processing setup.
    """

    progress = 0.0
    if job_count > 0:
        progress += job_no / job_count
    if steps > 0 and job_count > 0:
        progress += 1 / job_count * step / steps
    return min(max(progress, 0.0), 1.0)


def pass_progress(step: int, steps: int) -> float:
    """Current pass fraction — upstream's ``progress``."""

    if steps <= 0:
        return 0.0
    return min(1.0, max(0.0, step / steps))


def pass_kind(job_no: int, refined: bool) -> int:
    """With Hires. fix Forge doubles ``job_count``: even passes are first passes, odd ones hires."""

    return job_no % 2 if refined else 0


def _remaining_pass_counts(job_no: int, job_count: int, refined: bool) -> dict[int, int]:
    first = job_no + 1
    count = max(0, job_count - first)
    if not refined:
        return {0: count}
    parity = first % 2
    return {parity: (count + 1) // 2, 1 - parity: count // 2}


class ProgressTracker:
    """Cross-request memory for the running job's ETA (upstream kept ``_tr_key``/``_tr`` globals).

    Every look at the state — from any page or client — is an observation; more observers only make
    the step timing finer. A new ``time_start`` (``State.begin``) starts over.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start(None)

    def _start(self, job_key) -> None:
        self._job_key = job_key
        self._step_trackers: dict[int, dict] = {}
        self._pass_no: int | None = None
        self._pass_started: float | None = None
        self._pass_partial = False
        self._last_pass_seconds: dict[int, float] = {}

    def observe(self, view: StateView, now: float) -> tuple[float | None, float | None]:
        """One look at the state; returns ``(pass_eta, job_eta)`` in seconds, ``None`` = unknown."""

        with self._lock:
            if view.time_start != self._job_key:
                self._start(view.time_start)
            self._note_pass(view, now)
            kind = pass_kind(view.job_no, view.refined)
            pass_eta = None
            # Upstream only tracks while a step is running and nothing was interrupted or skipped.
            if (view.steps > 0 and view.step < view.steps
                    and not (view.skipped or view.interrupted)):
                eta, tracker = step_eta(self._step_trackers.get(kind), view.step, view.steps, now)
                self._step_trackers[kind] = tracker
                if average_step_seconds(tracker) is not None:
                    pass_eta = eta
            return pass_eta, self._job_eta(view, pass_eta, now)

    def _note_pass(self, view: StateView, now: float) -> None:
        if self._pass_no is None or view.job_no < self._pass_no:
            self._pass_no, self._pass_started = view.job_no, now
            # Joined in the middle of a pass: its measured length would be too short.
            self._pass_partial = view.step > 0
            return
        if view.job_no == self._pass_no:
            return
        advanced = view.job_no - self._pass_no
        if not self._pass_partial:
            # Several passes between two looks share the time, like upstream splits skipped steps.
            seconds = max(0.0, now - self._pass_started) / advanced
            if advanced == 1:
                kinds = {pass_kind(self._pass_no, view.refined)}
            else:
                kinds = {0, 1} if view.refined else {0}
            for kind in kinds:
                self._last_pass_seconds[kind] = seconds
        self._pass_no, self._pass_started = view.job_no, now
        self._pass_partial = False
        # A new pass restarts its kind's step clock and keeps the average. Upstream's tracker only
        # restarts when it happens to see the step go back: a look that misses the new pass's step 0
        # would count the decode/save/setup between passes as step time.
        kind = pass_kind(view.job_no, view.refined)
        tracker = self._step_trackers.get(kind)
        if tracker is not None:
            self._step_trackers[kind] = dict(tracker, step=int(view.step), step_ts=now)

    def _job_eta(self, view: StateView, pass_eta: float | None, now: float) -> float | None:
        if pass_eta is None or view.job_count <= 0 or view.job_no >= view.job_count:
            return None
        counts = _remaining_pass_counts(view.job_no, view.job_count, view.refined)
        projected = None
        if self._pass_started is not None:
            projected = max(0.0, now - self._pass_started) + pass_eta
        total = pass_eta
        for kind, count in counts.items():
            if count <= 0:
                continue
            seconds = self._last_pass_seconds.get(kind)
            if seconds is None:
                seconds = projected   # no finished pass of this kind yet: one like the current
            if seconds is None:
                return None
            total += count * seconds
        return total


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _job_fields(view: StateView, now: float, pass_eta: float | None, eta: float | None) -> dict:
    elapsed = max(0.0, now - view.time_start) if view.time_start else 0.0
    return {
        "time_start": view.time_start,
        "elapsed": round(elapsed, 2),
        "job_no": view.job_no,
        "job_count": view.job_count,
        "passes_per_image": 2 if view.refined else 1,
        "step": view.step,
        "steps": view.steps,
        "progress": round(job_progress(view.job_no, view.job_count, view.step, view.steps), 4),
        "pass_progress": round(pass_progress(view.step, view.steps), 4),
        "pass_eta": _rounded(pass_eta, 2),
        "eta": _rounded(eta, 2),
        "interrupted": view.interrupted,
        "skipped": view.skipped,
        "stopping": view.stopping,
        "textinfo": view.textinfo,
    }


def progress_snapshot(
    id_task: str | None,
    view: StateView,
    tasks: TaskView,
    tracker: ProgressTracker,
    now: float,
) -> dict:
    """The route's answer for ``id_task``; without one only ``busy``, ``queue_size`` and ``server_time``."""

    asked = id_task or None
    busy = view.busy or tasks.current is not None
    if asked is None:
        # Nothing job-specific without the task's id, as Forge's own /internal/progress.
        active, queued, completed = False, False, False
    else:
        active = tasks.current == asked
        queued = not active and asked in tasks.pending
        completed = not active and not queued and asked in tasks.finished
    pass_eta = eta = None
    if busy:
        pass_eta, eta = tracker.observe(view, now)
    payload = {
        "version": API_VERSION,
        "id_task": asked,
        "busy": busy,
        "active": active,
        "queued": queued,
        "completed": completed,
        "queue_position": tasks.pending.index(asked) + 1 if queued else None,
        "queue_size": len(tasks.pending),
        "server_time": round(now, 3),
    }
    payload.update(_job_fields(view, now, pass_eta, eta) if active else dict.fromkeys(JOB_FIELDS))
    return payload


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


def register_progress_routes(
    app: Any,
    *,
    state_provider: Callable[[], Any] | None = None,
    tasks_provider: Callable[[], TaskView] | None = None,
    clock: Callable[[], float] | None = None,
    tracker: ProgressTracker | None = None,
) -> bool:
    """Register ``GET /sam-extra/progress`` once (Forge re-fires app start after Reload UI).

    The keyword hooks exist for tests; Forge uses ``shared.state`` and ``modules.progress``.
    """

    for route in getattr(app, "routes", ()):
        if getattr(route, "path", None) == PROGRESS_API_PATH:
            return False

    state_provider = state_provider or forge_state
    tasks_provider = tasks_provider or forge_tasks
    clock = clock or time.time
    tracker = tracker or ProgressTracker()
    auth_dependencies = extension_auth_dependencies(app)

    async def get_progress(request: Request) -> JSONResponse:
        require_same_origin_header(request)
        id_task = request.query_params.get("id_task") or None
        if id_task is not None and len(id_task) > MAX_TASK_ID_LENGTH:
            raise HTTPException(status_code=400, detail="id_task is too long")
        payload = progress_snapshot(
            id_task, read_state(state_provider()), tasks_provider(), tracker, clock()
        )
        return JSONResponse(payload, headers=_NO_STORE)

    app.add_api_route(
        PROGRESS_API_PATH,
        get_progress,
        methods=["GET"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam-extra-progress",
        dependencies=auth_dependencies,
    )
    return True
