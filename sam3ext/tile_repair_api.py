"""Anima Tile & Repair as a JSON route — ``POST /sam-extra/tile-repair``.

Runs exactly what the Gradio panel runs in Tile-Repair mode (``ui_anima.handle_anima_click``):
``anima_core.run_tile_repair`` under ``forge_exclusive.run_exclusive`` (queue lock, fresh
``shared.state``, Forge main thread) — under its own job name ``ROUTE_JOB``, so the route's stop and
the panel's ⏹ never interrupt each other's run. The panel needs 30 positional Gradio inputs
plus gallery state; this route takes one JSON object, so the UR_IV desktop app (or any same-origin
client) can use the feature. Registered once at app start by ``scripts/anima_tile_repair_api.py``.

Routes — same guards as the Notebook routes (``notebook_store``): the Gradio login dependency when
Forge runs with ``--gradio-auth`` (401) and the same-origin header ``X-SAM3-Notebook: 1`` (403):

``GET /sam-extra/tile-repair/options``
    ``{"version", "available", "models", "default_model", "dit", "text_encoder", "vae", "defaults",
    "ranges", "increments"}`` — the panel's dropdown choices (only 3-channel Anima LLLites, read from
    the safetensors header) and slider defaults/bounds.

``POST /sam-extra/tile-repair`` — JSON object; unknown keys are refused (400) so a typo never runs
with a silent default:

=====================  ==========================================================================
``image``              base64 PNG (JPEG/WebP also open; a ``data:image/...;base64,`` prefix is fine).
                       Required. At most 64 MB decoded and 64 MP.
``model``              3-channel Anima LLLite basename in ``models/ControlNet`` (default: the newest
                       Tile & Repair file, v20 over v10 — the panel's default).
``prompt``             default: the panel's prompt. Empty is refused, as the panel does.
``negative_prompt``    default ``""`` (sd-scripts ``--negative_prompt``).
``steps``              default 50, 1..150 (sd-scripts ``--infer_steps``; panel slider bounds).
``cfg_scale``          default 3.5, 0..20 (sd-scripts ``--guidance_scale``).
``flow_shift``         default 5.0, 0..30 (sd-scripts ``--flow_shift``).
``multiplier``         default 1.0, -10..10 (sd-scripts ``--lllite_multiplier``; kohya
                       ComfyUI-Anima-LLLite ``strength``).
``short_side``         default 1024, 256..4096 — the output keeps the source aspect ratio
                       (``anima_core.tile_repair_size``: shorter edge, multiples of 32).
``seed``               default -1 = random; the seed actually used is returned.
``dit``                default ``"Use Forge current"``; otherwise one of the panel's DiT choices.
``text_encoder``       default: the panel's Qwen3 0.6B pick; otherwise one of its choices.
``vae``                default: the panel's Qwen-Image pick; otherwise one of its choices.
``unload_forge_before`` default ``true``.
=====================  ==========================================================================

→ 200 ``{"version", "interrupted": false, "image" (base64 PNG, infotext in its ``parameters`` text
chunk), "info", "seed", "width", "height", "model"}``; ``{"interrupted": true, "image": null, ...}``
when the stop route (or Forge's own Interrupt) ended the run. 400 bad body/values, 413 too large,
422 the pipeline's own setup errors (no TE/VAE/DiT resolved, vendor files, model load — the panel's
red banner text), 503 vendor missing, 500 anything else.

``POST /sam-extra/tile-repair/stop`` → ``{"stopped": bool}``. Cancels this route's requests from the
moment they arrive: one still uploading or waiting for Forge's queue (behind a txt2img or a panel
click) answers ``interrupted`` as soon as it gets the queue, without loading or unloading anything;
the running one is interrupted through ``forge_exclusive.stop_if_job((ROUTE_JOB,))``. Never touches a
txt2img or a Tile-Repair panel click. ``true`` when there was a request to cancel.

LoRA slots, the PiD Upscale mode and gallery insertion stay panel-only.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import math
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, PngImagePlugin
from starlette.concurrency import run_in_threadpool

from .notebook_store import _gradio_auth_dependencies, require_same_origin_header

logger = logging.getLogger(__name__)

TILE_REPAIR_API_PATH = "/sam-extra/tile-repair"
TILE_REPAIR_OPTIONS_PATH = "/sam-extra/tile-repair/options"
TILE_REPAIR_STOP_PATH = "/sam-extra/tile-repair/stop"
API_VERSION = 1

MAX_REQUEST_BYTES = 96 * 1024 * 1024        # base64 of the 64 MB image limit plus the other fields
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_PIXELS = 64 * 1024 * 1024        # 8192 x 8192
MAX_PROMPT_CHARS = 20_000
MAX_SEED = 2**63 - 1
_SOURCE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
_DATA_URL_PREFIX = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,")
# The parameters line of anima_core._build_infotext ("Steps: .., CFG scale: .., Seed: N, Size: ..").
# The prompt and negative prompt come before it and may hold "Seed: N" (even a whole look-alike
# line) themselves, so seed_from_infotext takes the last such line.
_SEED_IN_INFOTEXT = re.compile(r"^Steps: [^\n]*?, Seed: (-?\d+)(?:,|$)", re.M)
_NO_STORE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}

# shared.state.job of a route run. Not the panel's ``ui_anima.TILE_REPAIR_JOB`` and neither name is
# a prefix of the other (``stop_if_job`` matches prefixes): the route's stop never interrupts a
# panel click and the panel's ⏹ never interrupts a route run. Forge's own Interrupt stops either.
ROUTE_JOB = "sam3_route_tile_repair"

USE_FORGE_CURRENT = "Use Forge current"
# The panel's prompt textbox value (ui_anima.build_anima_panel) — tests keep the two identical.
DEFAULT_PROMPT = (
    "repair the low-quality anime image, reduce blur and compression artifacts, "
    "preserve the original composition"
)

# Panel defaults = kohya-ss/sd-scripts@690ea7f9 anima_minimal_inference_control_net_lllite.py
# argparse defaults (:127-134, :167-170) and the panel's short side / seed.
DEFAULTS: Mapping[str, Any] = MappingProxyType({
    "prompt": DEFAULT_PROMPT,
    "negative_prompt": "",
    "steps": 50,
    "cfg_scale": 3.5,
    "flow_shift": 5.0,
    "multiplier": 1.0,
    "short_side": 1024,
    "seed": -1,
    "unload_forge_before": True,
})
# Panel slider bounds (multiplier = kohya ComfyUI-Anima-LLLite@b7495bd8 nodes.py:130 strength).
RANGES: Mapping[str, tuple] = MappingProxyType({
    "steps": (1, 150),
    "cfg_scale": (0.0, 20.0),
    "flow_shift": (0.0, 30.0),
    "multiplier": (-10.0, 10.0),
    "short_side": (256, 4096),
})
INCREMENTS: Mapping[str, float] = MappingProxyType({
    "steps": 1, "cfg_scale": 0.1, "flow_shift": 0.1, "multiplier": 0.01, "short_side": 32,
})
_INTEGER_FIELDS = frozenset({"steps", "short_side"})
REQUEST_KEYS = frozenset({
    "image", "model", "prompt", "negative_prompt", "steps", "cfg_scale", "flow_shift",
    "multiplier", "short_side", "seed", "dit", "text_encoder", "vae", "unload_forge_before",
})


class TileRepairRequestError(ValueError):
    """A request the route refuses before running anything. ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class TileRepairChoices:
    """The panel's dropdown choices. ``lllite`` holds only real files (no ``"None"``)."""

    lllite: tuple
    dit: tuple
    text_encoder: tuple
    vae: tuple
    default_lllite: str
    default_text_encoder: str
    default_vae: str


@dataclass
class TileRepairRequest:
    """A validated request. ``to_repair_args`` builds what ``run_tile_repair`` takes."""

    source: Image.Image
    model: str
    prompt: str
    negative_prompt: str
    steps: int
    cfg_scale: float
    flow_shift: float
    multiplier: float
    short_side: int
    seed: int
    dit: str
    text_encoder: str
    vae: str
    unload_forge_before: bool
    # Set by the stop route. The route hands in the token it registered when the request arrived.
    cancel: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    def to_repair_args(self):
        from .anima_core import AnimaTileRepairArgs

        # Same field mapping as ui_anima._map_widget_values for the Tile-Repair mode; the LoRA
        # slots stay at the panel default (4 x ("None", 0.0)).
        return AnimaTileRepairArgs(
            lllite_model=self.model,
            dit_override=self.dit,
            te_override=self.text_encoder,
            vae_override=self.vae,
            positive=self.prompt,
            negative=self.negative_prompt,
            steps=self.steps,
            cfg=self.cfg_scale,
            flow_shift=self.flow_shift,
            seed=self.seed,
            short_side=self.short_side,
            lllite_multiplier=self.multiplier,
            unload_forge_before=self.unload_forge_before,
            restore_mode="Anima Tile-Repair",
        )


# ---------------------------------------------------------------------------
# Panel-backed providers (the defaults the route uses in Forge)
# ---------------------------------------------------------------------------


def panel_choices() -> TileRepairChoices:
    """The Tile-Repair panel's dropdown lists and defaults, computed the same way."""

    from . import anima_core as core

    lllite = core.list_lllite_choices()
    te = core.list_te_choices()
    vae = core.list_vae_choices()
    return TileRepairChoices(
        lllite=tuple(name for name in lllite if name != "None"),
        dit=tuple(core.list_dit_choices()),
        text_encoder=tuple(te),
        vae=tuple(vae),
        default_lllite=core.default_lllite_choice(lllite),
        default_text_encoder=core.default_te_choice(te),
        default_vae=core.default_vae_choice(vae),
    )


def vendor_available() -> bool:
    from .anima_core import anima_available

    return anima_available()


def run_panel_pipeline(request: TileRepairRequest) -> list:
    """``[(PIL, infotext)]`` — or ``[]`` when stopped — like the panel's Tile-Repair click."""

    from .anima_core import run_tile_repair
    from .forge_exclusive import run_exclusive

    repair = request.to_repair_args()

    def work() -> list:
        # A cancel that came while this request waited for queue_lock: shared.state.job was someone
        # else's then, so stop_if_job could not reach it. Checked after state.begin(ROUTE_JOB), so a
        # cancel after this check finds our job and sets the stop flags instead.
        if request.cancel.is_set():
            return []
        return run_tile_repair(request.source, repair)

    if request.cancel.is_set():
        return []
    return run_exclusive(ROUTE_JOB, work, on_main_thread=True)


def stop_tile_repair() -> bool:
    """Interrupt the running route job (not a panel click, not a txt2img)."""

    from .forge_exclusive import stop_if_job

    return stop_if_job((ROUTE_JOB,))


class _InFlight:
    """Cancel tokens of the route's requests, from arrival until the pipeline returns."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: set = set()

    @contextmanager
    def track(self) -> Iterator[threading.Event]:
        token = threading.Event()
        with self._lock:
            self._tokens.add(token)
        try:
            yield token
        finally:
            with self._lock:
                self._tokens.discard(token)

    def cancel_all(self) -> bool:
        with self._lock:
            tokens = tuple(self._tokens)
        for token in tokens:
            token.set()
        return bool(tokens)


# ---------------------------------------------------------------------------
# Request parsing (pure — no Forge, no torch)
# ---------------------------------------------------------------------------


def _number(body: Mapping[str, Any], key: str):
    value = body.get(key, DEFAULTS[key])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TileRepairRequestError(f"{key} must be a number")
    if key in _INTEGER_FIELDS:
        if isinstance(value, float):
            if not value.is_integer():
                raise TileRepairRequestError(f"{key} must be a whole number")
            value = int(value)
    else:
        value = float(value)
        if not math.isfinite(value):
            raise TileRepairRequestError(f"{key} must be a finite number")
    low, high = RANGES[key]
    if not low <= value <= high:
        raise TileRepairRequestError(f"{key} must be between {low} and {high} (got {value})")
    return value


def _text(body: Mapping[str, Any], key: str, default: str) -> str:
    value = body.get(key, default)
    if not isinstance(value, str):
        raise TileRepairRequestError(f"{key} must be a string")
    if len(value) > MAX_PROMPT_CHARS:
        raise TileRepairRequestError(f"{key} is longer than {MAX_PROMPT_CHARS} characters")
    return value


def _choice(body: Mapping[str, Any], key: str, default: str, choices: Sequence[str], what: str) -> str:
    value = body.get(key, default)
    if not isinstance(value, str) or value not in choices:
        raise TileRepairRequestError(f"{key}: {value!r} is not one of the {what} choices")
    return value


def decode_source_image(value: Any) -> Image.Image:
    """base64 (optionally a data URL) → a loaded single still image."""

    if not isinstance(value, str) or not value:
        raise TileRepairRequestError("image must be a base64 PNG string")
    text = _DATA_URL_PREFIX.sub("", value.strip(), count=1)
    if len(text) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise TileRepairRequestError("image is larger than 64 MB", status=413)
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise TileRepairRequestError("image is not valid base64") from error
    if not data:
        raise TileRepairRequestError("image is empty")
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in _SOURCE_FORMATS:
            raise TileRepairRequestError(f"image must be PNG, JPEG or WebP (got {image.format})")
        width, height = image.size
        if width < 1 or height < 1 or width * height > MAX_SOURCE_PIXELS:
            raise TileRepairRequestError(f"image must be at most 64 MP (got {width}x{height})")
        image.load()
    except TileRepairRequestError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise TileRepairRequestError("image could not be decoded") from error
    return image


def parse_request(body: Any, choices: TileRepairChoices) -> TileRepairRequest:
    """Validate a JSON body against the panel's choices and bounds. Raises TileRepairRequestError."""

    if not isinstance(body, Mapping):
        raise TileRepairRequestError("Tile-Repair request must be a JSON object")
    unknown = sorted(str(key) for key in body if key not in REQUEST_KEYS)
    if unknown:
        raise TileRepairRequestError(f"unknown field(s): {', '.join(unknown)}")
    if "image" not in body:
        raise TileRepairRequestError("image is required")

    if "model" not in body and choices.default_lllite in ("", "None"):
        raise TileRepairRequestError(
            "No 3-channel Anima ControlNet-LLLite in models/ControlNet — put the Tile & Repair "
            "model (e.g. animaTileRepair_v20.safetensors) there.",
            status=422,
        )
    model = _choice(body, "model", choices.default_lllite, choices.lllite,
                    "3-channel Anima LLLite (models/ControlNet)")
    prompt = _text(body, "prompt", DEFAULT_PROMPT)
    if not prompt.strip():
        raise TileRepairRequestError("prompt is empty")
    negative = _text(body, "negative_prompt", DEFAULTS["negative_prompt"])

    seed = body.get("seed", DEFAULTS["seed"])
    if isinstance(seed, float) and seed.is_integer():
        seed = int(seed)
    if isinstance(seed, bool) or not isinstance(seed, int) or not -1 <= seed <= MAX_SEED:
        raise TileRepairRequestError("seed must be -1 (random) or a whole number from 0")

    unload = body.get("unload_forge_before", DEFAULTS["unload_forge_before"])
    if not isinstance(unload, bool):
        raise TileRepairRequestError("unload_forge_before must be true or false")

    numbers = {key: _number(body, key) for key in RANGES}
    return TileRepairRequest(
        source=decode_source_image(body["image"]),
        model=model,
        prompt=prompt,
        negative_prompt=negative,
        steps=numbers["steps"],
        cfg_scale=numbers["cfg_scale"],
        flow_shift=numbers["flow_shift"],
        multiplier=numbers["multiplier"],
        short_side=numbers["short_side"],
        seed=seed,
        dit=_choice(body, "dit", USE_FORGE_CURRENT, choices.dit, "DiT"),
        text_encoder=_choice(body, "text_encoder", choices.default_text_encoder,
                             choices.text_encoder, "Text Encoder"),
        vae=_choice(body, "vae", choices.default_vae, choices.vae, "VAE"),
        unload_forge_before=unload,
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def options_payload(choices: TileRepairChoices, *, available: bool) -> dict:
    default_model = choices.default_lllite if choices.default_lllite not in ("", "None") else None
    return {
        "version": API_VERSION,
        "available": bool(available),
        "models": list(choices.lllite),
        "default_model": default_model,
        "dit": list(choices.dit),
        "text_encoder": list(choices.text_encoder),
        "vae": list(choices.vae),
        "defaults": {
            **DEFAULTS,
            "model": default_model,
            "dit": USE_FORGE_CURRENT,
            "text_encoder": choices.default_text_encoder,
            "vae": choices.default_vae,
        },
        "ranges": {key: list(bounds) for key, bounds in RANGES.items()},
        "increments": dict(INCREMENTS),
    }


def encode_png(image: Image.Image, infotext: str) -> bytes:
    """PNG bytes with the infotext in a ``parameters`` text chunk (what Forge writes for generations)."""

    info = PngImagePlugin.PngInfo()
    if infotext:
        info.add_text("parameters", infotext)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def seed_from_infotext(infotext: str) -> int | None:
    found = _SEED_IN_INFOTEXT.findall(infotext or "")
    return int(found[-1]) if found else None


def result_payload(pairs: Sequence, request: TileRepairRequest) -> dict:
    """``run_tile_repair`` result → response body. ``[]`` = interrupted by Stop."""

    if not pairs:
        return {"version": API_VERSION, "interrupted": True, "image": None, "info": "",
                "seed": None, "width": None, "height": None, "model": request.model}
    image, infotext = pairs[-1]
    infotext = str(infotext or "")
    png = encode_png(image, infotext)
    return {
        "version": API_VERSION,
        "interrupted": False,
        "image": base64.b64encode(png).decode("ascii"),
        "info": infotext,
        "seed": seed_from_infotext(infotext),
        "width": image.width,
        "height": image.height,
        "model": request.model,
    }


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_tile_repair_routes(
    app: Any,
    *,
    choices_provider: Callable[[], TileRepairChoices] | None = None,
    execute: Callable[[TileRepairRequest], list] | None = None,
    available: Callable[[], bool] | None = None,
    stop: Callable[[], bool] | None = None,
) -> bool:
    """Register the three Tile-Repair routes once (Forge re-fires app start after Reload UI).

    The keyword hooks exist for tests; Forge uses the panel-backed defaults.
    """

    for route in getattr(app, "routes", ()):
        if getattr(route, "path", None) == TILE_REPAIR_API_PATH:
            return False

    choices_provider = choices_provider or panel_choices
    execute = execute or run_panel_pipeline
    available = available or vendor_available
    stop = stop or stop_tile_repair
    auth_dependencies = _gradio_auth_dependencies(app)
    in_flight = _InFlight()

    async def get_options(request: Request) -> JSONResponse:
        require_same_origin_header(request)
        try:
            choices = await run_in_threadpool(choices_provider)
            ready = await run_in_threadpool(available)
        except Exception as error:  # noqa: BLE001 - reported as one message
            logger.exception("SAM3 Anima Tile-Repair API: listing models failed")
            raise HTTPException(status_code=500, detail="Tile-Repair model lists could not be read") from error
        return JSONResponse(options_payload(choices, available=ready), headers=_NO_STORE)

    async def post_tile_repair(request: Request) -> JSONResponse:
        require_same_origin_header(request)
        # Registered before the body is read: a stop while the image uploads, while the request
        # waits for Forge's queue or while it runs all reach this request.
        with in_flight.track() as cancel:
            parsed, pairs = await _parse_and_run(request, cancel)
        payload = await run_in_threadpool(result_payload, pairs, parsed)
        return JSONResponse(payload, headers=_NO_STORE)

    async def _parse_and_run(request: Request, cancel: threading.Event):
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Tile-Repair request is too large")
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Tile-Repair request is too large")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise HTTPException(status_code=400, detail="Tile-Repair request is not valid JSON") from error
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="Tile-Repair request must be a JSON object")
        if not await run_in_threadpool(available):
            raise HTTPException(
                status_code=503,
                detail="Anima vendor missing — install.py didn't clone kohya-ss/sd-scripts into anima_vendor/.",
            )
        try:
            choices = await run_in_threadpool(choices_provider)
            parsed = await run_in_threadpool(parse_request, body, choices)
        except TileRepairRequestError as error:
            raise HTTPException(status_code=error.status, detail=str(error)) from error
        parsed.cancel = cancel
        if cancel.is_set():
            return parsed, []   # stopped while uploading — answer interrupted, run nothing
        try:
            pairs = await run_in_threadpool(execute, parsed)
        except Exception as error:  # noqa: BLE001 - mapped below, like the panel's banner
            if type(error) is RuntimeError:
                # run_tile_repair's own setup messages (no TE/VAE/DiT, vendor files, model load).
                raise HTTPException(status_code=422, detail=str(error)) from error
            logger.exception("SAM3 Anima Tile-Repair API: run failed")
            raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error
        return parsed, pairs

    async def post_stop(request: Request) -> JSONResponse:
        require_same_origin_header(request)
        # Tokens first, then the job: a request that takes the queue after this sees its token set;
        # one that already checked it holds shared.state.job == ROUTE_JOB, which stop() interrupts.
        cancelled = in_flight.cancel_all()
        stopped = await run_in_threadpool(stop)
        return JSONResponse({"stopped": bool(cancelled or stopped)}, headers=_NO_STORE)

    app.add_api_route(
        TILE_REPAIR_OPTIONS_PATH,
        get_options,
        methods=["GET"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam-extra-tile-repair-options",
        dependencies=auth_dependencies,
    )
    app.add_api_route(
        TILE_REPAIR_STOP_PATH,
        post_stop,
        methods=["POST"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam-extra-tile-repair-stop",
        dependencies=auth_dependencies,
    )
    app.add_api_route(
        TILE_REPAIR_API_PATH,
        post_tile_repair,
        methods=["POST"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam-extra-tile-repair",
        dependencies=auth_dependencies,
    )
    return True
