"""Light LoRA list — ``GET /sam3-lora/lite``.

Forge's ``GET /sdapi/v1/loras`` answers every LoRA with its whole parsed
safetensors header. On a library of ~800 kohya-trained LoRAs that is ~143 MB
of JSON (``ss_datasets`` ~97 MiB, ``ss_tag_frequency`` ~29 MiB): ~5 s per call
with Forge's metadata cache warm, 72 s cold. Clients that keep only a few
fields (the AI Studio Pro app, sam-extra's MCP server) pay for all of it.

This route answers the same list with just those fields::

    {"version": 1, "count": <int>, "items": [
      {"name": str,              # /sdapi/v1/loras "name"  (NetworkOnDisk.name)
       "alias": str,             # /sdapi/v1/loras "alias" (NetworkOnDisk.alias)
       "path": str,              # /sdapi/v1/loras "path"  (NetworkOnDisk.filename)
       "activation_text": str,   # metadata "activation text", else "ss_activation_text", else ""
       "tag_frequency_top": [[tag, count], ...]},   # see tag_frequency_top()
      ...]}

The items come from the very objects ``/sdapi/v1/loras`` serialises —
``networks.available_networks.values()`` of Forge's built-in ``sd_forge_lora``
— in the same order, so names, aliases, paths and order match it exactly.
Nothing is read from disk per call: the metadata is what Forge already holds
in memory. ``?refresh=1`` first rescans the LoRA folders with
``networks.list_available_networks()``, which is what
``POST /sdapi/v1/refresh-loras`` runs.

The one per-LoRA cost that grows with the library is merging and sorting
``ss_tag_frequency``. It is memoised per LoRA file on the identity of the
metadata value it was computed from: Forge builds new ``NetworkOnDisk``
objects (and new metadata) whenever it rescans, so an unchanged object means
unchanged input — no ``os.stat`` per file — and a rescan recomputes. The memo
only holds LoRAs seen by the last call.

Framework-agnostic (FastAPI is imported inside ``register_lora_lite_route``),
like the rest of ``sam3ext``.
"""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Iterable

LORA_LITE_PATH = "/sam3-lora/lite"
LITE_VERSION = 1
TAG_FREQUENCY_TOP = 32
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def activation_text(metadata: Any) -> str:
    """``metadata["activation text"]``, else ``metadata["ss_activation_text"]``, else "".

    Exactly the value a client of ``/sdapi/v1/loras`` reads (AI Studio Pro's
    ``get_loras``: ``md.get("activation text", "") or
    md.get("ss_activation_text", "")``), so both lists give the same words.
    Forge parses a header value that starts with ``{`` into an object; when
    that object is the value such a client reads, it is not text and gives
    "" — not the key below it — instead of breaking the list.
    """
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get("activation text", "") or metadata.get("ss_activation_text", "")
    return value if isinstance(value, str) else ""


def _merged_tag_frequency(raw: Any) -> dict:
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    merged: dict = {}
    for group in raw.values():
        if isinstance(group, dict):
            merged.update(group)
    return merged


def tag_frequency_top(metadata: Any, limit: int = TAG_FREQUENCY_TOP) -> list[list[Any]]:
    """The most frequent training tags of ``ss_tag_frequency``.

    ``ss_tag_frequency`` is ``{dataset_dir: {tag: count}}`` — a JSON string in
    the file, already a dict when Forge's ``read_metadata_from_safetensors``
    parsed it. The per-dataset dicts are merged with ``dict.update`` in
    iteration order (a later dataset's count replaces an earlier one; the tag
    keeps its first position), then ``sorted(..., key=count, reverse=True)``,
    which is stable, so tied counts keep that order. First ``limit`` entries,
    ``[]`` when there is nothing.

    A count that cannot be compared with the others (not a number) drops only
    that tag rather than the whole LoRA; so does a NaN or infinite count.
    """
    if not isinstance(metadata, dict):
        return []
    merged = _merged_tag_frequency(metadata.get("ss_tag_frequency"))
    if not merged:
        return []
    items = list(merged.items())
    try:
        ranked = sorted(items, key=lambda kv: kv[1], reverse=True)
    except TypeError:
        numeric = [(tag, count) for tag, count in items
                   if isinstance(count, (int, float)) and not isinstance(count, bool)]
        ranked = sorted(numeric, key=lambda kv: kv[1], reverse=True)
    # JSON has no NaN/Infinity (the response would fail to encode): such a
    # count is dropped, as a non-number is.
    return [[_text(tag), count] for tag, count in ranked[:limit]
            if not (isinstance(count, float) and not math.isfinite(count))]


class LoraLiteBuilder:
    """Builds the lite payload from Forge's in-memory LoRA objects.

    ``_memo`` maps a LoRA file path to ``(raw ss_tag_frequency, top list)``.
    A hit needs the raw value to be the *same object* (``is``) Forge holds
    now; the memo keeps that object alive, so its identity cannot be reused.
    Each call replaces the memo with the entries it used, so LoRAs that left
    the list do not linger. The cost: after a rescan the previous objects stay
    alive until the next call (one generation — ~70 MiB of parsed
    ``ss_tag_frequency`` for an 825-LoRA kohya library).
    """

    def __init__(self, limit: int = TAG_FREQUENCY_TOP) -> None:
        self._limit = limit
        self._memo: dict[str, tuple[Any, list]] = {}
        self._lock = threading.Lock()

    def item(self, entry: Any, memo: dict, fresh: dict) -> dict:
        metadata = getattr(entry, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        path = _text(getattr(entry, "filename", ""))
        raw = metadata.get("ss_tag_frequency")
        cached = memo.get(path)
        if cached is not None and cached[0] is raw:
            top = cached[1]
        else:
            top = tag_frequency_top(metadata, self._limit)
        fresh[path] = (raw, top)
        return {
            "name": _text(getattr(entry, "name", "")),
            "alias": _text(getattr(entry, "alias", "")),
            "path": path,
            "activation_text": activation_text(metadata),
            # A copy per answer: callers may mutate theirs, the memo stays intact.
            "tag_frequency_top": [list(pair) for pair in top],
        }

    def build(self, entries: Iterable[Any]) -> dict:
        snapshot = list(entries)
        with self._lock:
            memo = self._memo
        fresh: dict[str, tuple[Any, list]] = {}
        items = [self.item(entry, memo, fresh) for entry in snapshot]
        with self._lock:
            self._memo = fresh
        return {"version": LITE_VERSION, "count": len(items), "items": items}


_BUILDER = LoraLiteBuilder()
# One ?refresh=1 rescan at a time (see lora_lite_data).
_RESCAN_LOCK = threading.Lock()


def _forge_networks() -> Any:
    """Forge's ``sd_forge_lora/networks.py`` (its folder is on ``sys.path`` in Forge)."""
    import networks  # type: ignore[import-not-found]

    return networks


def _entries(module: Any) -> list:
    available = getattr(module, "available_networks", None) or {}
    # One snapshot of the values: a concurrent rescan clears and refills the
    # dict, and list() over it runs without releasing the GIL.
    return list(available.values())


def lora_lite_data(*, refresh: bool = False, networks: Any = None,
                   builder: LoraLiteBuilder | None = None) -> dict:
    """The ``/sam3-lora/lite`` payload. ``networks``/``builder`` are for tests.

    ``list_available_networks()`` clears Forge's tables and refills them file
    by file. Forge's own ``POST /sdapi/v1/refresh-loras`` runs it on the event
    loop, so it never overlaps itself; this route runs it in the threadpool,
    where two ``?refresh=1`` at once would interleave (an alias seen twice
    lands in ``forbidden_network_aliases``, and a caller can answer a
    half-filled table). So rescans from here run one at a time, and each
    caller answers the table its own rescan left.
    """
    module = networks if networks is not None else _forge_networks()
    if refresh:
        with _RESCAN_LOCK:
            module.list_available_networks()
            entries = _entries(module)
    else:
        entries = _entries(module)
    return (builder or _BUILDER).build(entries)


def refresh_requested(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _TRUTHY


def register_lora_lite_route(
    app: Any,
    *,
    auth_dependencies: list[Any] | None = None,
    networks: Any = None,
    builder: LoraLiteBuilder | None = None,
) -> bool:
    """Register ``GET /sam3-lora/lite`` once. Returns True if newly added
    (idempotent — Forge re-fires app-start after Reload UI).

    Guarded by the same login guards as every sam-extra route
    (``extension_auth_dependencies``: ``--gradio-auth`` and, under
    ``--api``/``--nowebui``, ``--api-auth``). Unlike the Notebook / memo /
    Tile & Repair / LoRA Manager config-spawn routes it does not ask for the
    ``X-SAM3-Notebook: 1`` header: it answers a subset of what
    ``GET /sdapi/v1/loras`` already answers without one, and its ``?refresh=1``
    does what ``POST /sdapi/v1/refresh-loras`` does. It is stricter than those
    two, though: ``sd_forge_lora`` adds them straight to the app, not through
    ``modules/api``'s ``add_api_route``, so neither login guards them. Under
    ``--gradio-auth`` a client without the Gradio login cookie, or under
    ``--api-auth`` one without the Basic credentials, gets 401 here and is
    expected to fall back to ``/sdapi/v1/loras``. ``auth_dependencies``,
    ``networks`` and ``builder`` exist for tests.
    """
    existing = {getattr(r, "path", None) for r in getattr(app, "routes", ())}
    if LORA_LITE_PATH in existing:
        return False

    from fastapi import Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    from .notebook_store import extension_auth_dependencies

    dependencies = (
        list(auth_dependencies)
        if auth_dependencies is not None
        else extension_auth_dependencies(app)
    )
    _no_store = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}

    def _answer(refresh: bool) -> dict:
        return lora_lite_data(refresh=refresh, networks=networks, builder=builder)

    async def _lite_route(request: Request) -> JSONResponse:
        refresh = refresh_requested(request.query_params.get("refresh"))
        try:
            # A rescan walks the LoRA folders; even the plain answer walks
            # ~800 objects. Off the event loop like the other sam-extra routes.
            data = await run_in_threadpool(_answer, refresh)
        except ImportError:
            return JSONResponse(
                {"detail": "Forge's built-in LoRA support (sd_forge_lora) is not loaded"},
                status_code=503, headers=_no_store)
        return JSONResponse(data, headers=_no_store)

    # See register_lora_routes: string annotations would make FastAPI read
    # ``request`` as a query parameter.
    _lite_route.__annotations__ = {"request": Request, "return": JSONResponse}

    app.add_api_route(
        LORA_LITE_PATH, _lite_route, methods=["GET"],
        include_in_schema=False, name="sam3-lora-lite",
        dependencies=dependencies,
    )
    return True
