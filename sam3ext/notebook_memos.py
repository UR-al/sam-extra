"""SAM3 Notebook memo pad: storage and HTTP routes.

Memos live in ``memos.json`` next to ``notebook.json`` (default
``<Forge data path>/sam-extra/memos.json``) so the preset store's revision
does not move on every keystroke.  The UR_IV desktop app syncs the same file
over these routes, which is why every memo carries its own ``updated_at`` and
deletions are kept as tombstones instead of being removed outright.

Routes (same Gradio login guard and ``X-SAM3-Notebook: 1`` header as the
Notebook routes):

* ``GET    /sam3-notebook/memos``            live memos, newest first
  (``?include_deleted=1`` adds tombstones)
* ``PUT    /sam3-notebook/memos/{memo_id}``  upsert; ``base_updated_at`` guards
  against overwriting a newer copy (409 with the stored memo)
* ``DELETE /sam3-notebook/memos/{memo_id}``  turn the memo into a tombstone;
  an optional ``?base_updated_at=`` query guards it the same way as PUT (409
  with the stored memo when it changed after that version)

A 200 from the GET route is the capability signal: older extensions answer 404.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import uuid
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .notebook_store import (
    _ID_RE,
    NOTEBOOK_API_PATH,
    _gradio_auth_dependencies,
    _safe_id,
    _safe_text,
    _utc_now,
    default_notebook_path,
    require_same_origin_header,
)


logger = logging.getLogger(__name__)

MEMO_API_PATH = NOTEBOOK_API_PATH + "/memos"
MEMO_FILE_NAME = "memos.json"
MEMO_SCHEMA_VERSION = 1
MAX_MEMOS = 500  # tombstones count toward the limit; the oldest are pruned first
MAX_MEMO_TITLE_LENGTH = 120
MAX_MEMO_TEXT_LENGTH = 100_000
# One memo is at most ~1.2 MB even when a Python client escapes every
# character as a surrogate pair; the whole store is capped separately.
MAX_MEMO_REQUEST_BYTES = 2 * 1024 * 1024
MAX_MEMO_STORE_BYTES = 32 * 1024 * 1024
MAX_TIMESTAMP_LENGTH = 64
# A client clock this far ahead would win every last-writer-wins merge.
MAX_CLIENT_CLOCK_SKEW = timedelta(minutes=10)

_NO_STORE_HEADERS = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# json.loads turns a lone "\ud83d" escape (a JS string cut inside an emoji)
# into a surrogate code point that UTF-8 cannot encode.
_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


class MemoConflictError(RuntimeError):
    """The stored memo changed after the version the client edited."""

    def __init__(self, message: str, memo: dict[str, Any]):
        super().__init__(message)
        self.memo = memo


class MemoNotFoundError(LookupError):
    """DELETE named a memo id the store has never seen."""


class MemoCapacityError(RuntimeError):
    """The memo count or the store size limit would be exceeded."""


class MemoCorruptError(RuntimeError):
    """Stored memo data exists but no valid copy can be read."""


class MemoSchemaError(ValueError):
    """Raised rather than silently downgrading data from a newer schema."""


def default_memo_path() -> Path:
    return default_notebook_path().with_name(MEMO_FILE_NAME)


def memo_path_for_notebook(notebook_store: Any) -> Path:
    """``memos.json`` next to the Notebook store's own file."""

    notebook_path = getattr(notebook_store, "path", None)
    if isinstance(notebook_path, (str, os.PathLike)):
        return Path(notebook_path).with_name(MEMO_FILE_NAME)
    return default_memo_path()


def empty_memo_store(*, revision: int = 0) -> dict[str, Any]:
    return {
        "schema_version": MEMO_SCHEMA_VERSION,
        "revision": max(0, int(revision)),
        "memos": [],
    }


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp that carries a time zone, as UTC."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > MAX_TIMESTAMP_LENGTH:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _clean_timestamp(value: Any) -> str | None:
    """Keep a valid UTC string verbatim; convert other offsets to UTC."""

    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    text = value.strip()
    if datetime.fromisoformat(text).utcoffset() == timedelta(0):
        return text
    return parsed.isoformat()


def _timestamp_key(value: Any) -> datetime:
    return parse_timestamp(value) or _EPOCH


def _later_than(previous: str | None) -> str:
    """Now, or 1 ms after ``previous`` when the clock has not passed it."""

    now = _utc_now()
    before = parse_timestamp(previous)
    if before is None or parse_timestamp(now) > before:
        return now
    return (before + timedelta(milliseconds=1)).isoformat()


def validate_memo_id(value: Any) -> str:
    memo_id = "" if value is None else str(value)
    if not _ID_RE.fullmatch(memo_id):
        raise ValueError(
            "Memo id must match ^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$"
        )
    return memo_id


def sanitize_memo(raw: Any) -> dict[str, Any]:
    """Validate one stored memo; bad ids get a fresh id so text is not lost.

    A missing timestamp becomes the epoch, not "now": the value must stay the
    same across reads or a syncing client would see a change on every GET.
    """

    if not isinstance(raw, dict):
        raise ValueError("Memo entries must be objects")
    updated_at = _clean_timestamp(raw.get("updated_at")) or _EPOCH.isoformat()
    deleted = raw.get("deleted") is True
    return {
        "id": _safe_id(raw.get("id"), "memo"),
        "title": _safe_text(raw.get("title"), MAX_MEMO_TITLE_LENGTH),
        "text": "" if deleted else _safe_text(raw.get("text"), MAX_MEMO_TEXT_LENGTH),
        "created_at": _clean_timestamp(raw.get("created_at")) or updated_at,
        "updated_at": updated_at,
        "deleted": deleted,
    }


def prune_tombstones(memos: list[dict[str, Any]], limit: int | None = None) -> int:
    """Drop the oldest tombstones until ``memos`` fits ``limit`` (MAX_MEMOS)."""

    excess = len(memos) - (MAX_MEMOS if limit is None else limit)
    if excess <= 0:
        return 0
    tombstones = sorted(
        (memo for memo in memos if memo.get("deleted")),
        key=lambda memo: _timestamp_key(memo.get("updated_at")),
    )
    doomed = {id(memo) for memo in tombstones[:excess]}
    if not doomed:
        return 0
    memos[:] = [memo for memo in memos if id(memo) not in doomed]
    return len(doomed)


def sanitize_memo_store(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Memo store must be an object")

    try:
        version = int(payload.get("schema_version", MEMO_SCHEMA_VERSION))
    except (TypeError, ValueError) as error:
        raise ValueError("Memo schema version must be an integer") from error
    if version != MEMO_SCHEMA_VERSION:
        raise MemoSchemaError(
            f"Unsupported memo schema version {version}; "
            f"expected {MEMO_SCHEMA_VERSION}"
        )

    try:
        revision = int(payload.get("revision", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("Memo revision must be an integer") from error
    if revision < 0:
        raise ValueError("Memo revision must not be negative")

    raw_memos = payload.get("memos", [])
    if not isinstance(raw_memos, list):
        raise ValueError("Memo store memos must be a list")

    result = empty_memo_store(revision=revision)
    seen_ids: set[str] = set()
    for raw in raw_memos:
        memo = sanitize_memo(raw)
        while memo["id"] in seen_ids:
            memo["id"] = _safe_id(None, "memo")
        seen_ids.add(memo["id"])
        result["memos"].append(memo)
    # A hand-edited file may exceed the limit. Only tombstones are dropped
    # here; live memos are never discarded, new ones are refused instead.
    prune_tombstones(result["memos"])
    return result


def _field_text(value: Any, limit: int, name: str) -> str:
    """Validate a client field; lone surrogates become U+FFFD.

    Rejecting them instead would leave a browser retrying the same PUT forever,
    and failing later in ``_write`` would report a confusing codec error.
    """

    if value is not None and not isinstance(value, str):
        raise ValueError(f"Memo {name} must be a string")
    if value:
        value = _LONE_SURROGATE_RE.sub("�", value)
    return _safe_text(value, limit)


def _public_memo(memo: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": memo["id"],
        "title": memo["title"],
        "text": memo["text"],
        "created_at": memo["created_at"],
        "updated_at": memo["updated_at"],
        "deleted": bool(memo["deleted"]),
    }


def _newest_first(memos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        memos,
        key=lambda memo: (_timestamp_key(memo.get("updated_at")), memo["id"]),
        reverse=True,
    )


def _parse_base(raw_base: Any) -> datetime | None:
    """``base_updated_at``: absent/empty means unguarded; anything else must parse."""

    if raw_base in (None, ""):
        return None
    base = parse_timestamp(raw_base)
    if base is None:
        raise ValueError(
            "base_updated_at must be an ISO 8601 timestamp with a time zone"
        )
    return base


def _check_base(stored: dict[str, Any], base: datetime | None, verb: str) -> None:
    """409 when the stored memo is newer than the version the client saw."""

    if base is not None and _timestamp_key(stored["updated_at"]) > base:
        raise MemoConflictError(
            f"Memo changed elsewhere after the version being {verb}",
            _public_memo(stored),
        )


class MemoStore:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else default_memo_path()
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        self._lock = threading.RLock()

    @staticmethod
    def _read_path(path: Path) -> dict[str, Any]:
        raw = path.read_bytes()
        if len(raw) > MAX_MEMO_STORE_BYTES:
            raise ValueError("Memo file exceeds the 32 MiB safety limit")
        return sanitize_memo_store(json.loads(raw.decode("utf-8")))

    def load(self) -> dict[str, Any]:
        with self._lock:
            unreadable: list[Path] = []
            for candidate in (self.path, self.backup_path):
                try:
                    data = self._read_path(candidate)
                except FileNotFoundError:
                    continue
                except MemoSchemaError:
                    # A newer schema is not damage; never fall back to an older
                    # backup that a later save would then write over it.
                    raise
                except (OSError, UnicodeError, ValueError):
                    unreadable.append(candidate)
                    continue
                if unreadable:
                    # The next write keeps the damaged primary as
                    # memos.json.corrupt-* before replacing it (see _write).
                    logger.warning(
                        "SAM3 Notebook memo storage %s is unreadable; serving "
                        "the older backup %s",
                        unreadable[0],
                        candidate.name,
                    )
                return deepcopy(data)
            if unreadable:
                names = ", ".join(path.name for path in unreadable)
                raise MemoCorruptError(
                    "Memo storage is damaged and was not overwritten "
                    f"({names}). Restore a valid .bak file or move the damaged "
                    "file aside before saving."
                )
            return empty_memo_store()

    def list_memos(self, *, include_deleted: bool = False) -> dict[str, Any]:
        data = self.load()
        memos = [
            _public_memo(memo)
            for memo in _newest_first(data["memos"])
            if include_deleted or not memo["deleted"]
        ]
        return {"revision": data["revision"], "memos": memos}

    def upsert(self, memo_id: Any, payload: Any) -> dict[str, Any]:
        memo_id = validate_memo_id(memo_id)
        if not isinstance(payload, dict):
            raise ValueError("Memo payload must be an object")
        base = _parse_base(payload.get("base_updated_at"))

        with self._lock:
            data = self.load()
            memos = data["memos"]
            index = next(
                (i for i, memo in enumerate(memos) if memo["id"] == memo_id),
                None,
            )
            stored = memos[index] if index is not None else None
            if stored is not None:
                _check_base(stored, base, "edited")

            # A missing field keeps the stored value instead of wiping it.
            title = _field_text(
                payload["title"] if "title" in payload
                else (stored["title"] if stored else ""),
                MAX_MEMO_TITLE_LENGTH,
                "title",
            )
            text = _field_text(
                payload["text"] if "text" in payload
                else (stored["text"] if stored else ""),
                MAX_MEMO_TEXT_LENGTH,
                "text",
            )
            updated_at = self._accepted_timestamp(
                payload.get("updated_at"),
                stored["updated_at"] if stored else None,
            )
            if stored is None:
                memo = {
                    "id": memo_id,
                    "title": title,
                    "text": text,
                    "created_at": _clean_timestamp(payload.get("created_at"))
                    or updated_at,
                    "updated_at": updated_at,
                    "deleted": False,
                }
                memos.append(memo)
                prune_tombstones(memos)
                if len(memos) > MAX_MEMOS:
                    raise MemoCapacityError(
                        f"Memo storage holds at most {MAX_MEMOS} memos"
                    )
            else:
                memo = {
                    **stored,
                    "title": title,
                    "text": text,
                    "updated_at": updated_at,
                    "deleted": False,
                }
                memos[index] = memo
            saved = self._write(data)
            return {"revision": saved["revision"], "memo": _public_memo(memo)}

    def delete(self, memo_id: Any, base_updated_at: Any = None) -> dict[str, Any]:
        """Tombstone a memo.

        ``base_updated_at`` (optional, the version the client decided to delete)
        refuses to wipe an edit made elsewhere after it, like ``upsert`` does.
        """

        memo_id = validate_memo_id(memo_id)
        base = _parse_base(base_updated_at)
        with self._lock:
            data = self.load()
            memos = data["memos"]
            index = next(
                (i for i, memo in enumerate(memos) if memo["id"] == memo_id),
                None,
            )
            if index is None:
                raise MemoNotFoundError(f"Unknown memo id: {memo_id}")
            stored = memos[index]
            if stored["deleted"]:
                # Idempotent: a repeated DELETE must not make the tombstone
                # newer than a later re-creation elsewhere.
                return {"revision": data["revision"], "memo": _public_memo(stored)}
            _check_base(stored, base, "deleted")
            tombstone = {
                **stored,
                "text": "",
                "deleted": True,
                "updated_at": _later_than(stored["updated_at"]),
            }
            memos[index] = tombstone
            saved = self._write(data)
            return {"revision": saved["revision"], "memo": _public_memo(tombstone)}

    @staticmethod
    def _accepted_timestamp(client_value: Any, stored_value: str | None) -> str:
        """The client's edit time when it is usable, else the server's now.

        Unusable means missing/invalid, too far in the future, or not newer
        than the copy it replaces (an edit never moves a memo back in time).
        """

        candidate = _clean_timestamp(client_value)
        parsed = parse_timestamp(candidate)
        if parsed is None:
            return _later_than(stored_value)
        if parsed > datetime.now(UTC) + MAX_CLIENT_CLOCK_SKEW:
            return _later_than(stored_value)
        stored_at = parse_timestamp(stored_value)
        if stored_at is not None and parsed <= stored_at:
            return _later_than(stored_value)
        return candidate

    def _damaged_copy_path(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        candidate = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        suffix = 1
        while candidate.exists():
            candidate = self.path.with_name(
                f"{self.path.name}.corrupt-{stamp}-{suffix}"
            )
            suffix += 1
        return candidate

    def _write(self, data: dict[str, Any]) -> dict[str, Any]:
        """Atomically replace the store with ``data`` at the next revision."""

        with self._lock:
            result = empty_memo_store(revision=int(data.get("revision", 0)) + 1)
            result["memos"] = [
                {
                    "id": memo["id"],
                    "title": memo["title"],
                    "text": memo["text"],
                    "created_at": memo["created_at"],
                    "updated_at": memo["updated_at"],
                    "deleted": bool(memo["deleted"]),
                }
                for memo in data["memos"]
            ]
            encoded = json.dumps(
                result,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) > MAX_MEMO_STORE_BYTES:
                raise MemoCapacityError("Memo storage exceeds the 32 MiB limit")

            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_name(
                f".{self.path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temp_path.open("wb") as file:
                    file.write(encoded)
                    file.flush()
                    os.fsync(file.fileno())
                if self.path.exists():
                    # Never replace a good backup with a damaged primary that
                    # ``load()`` just recovered from. The damaged file may hold
                    # the only copy of the newest revision, so keep it aside
                    # (a failed copy aborts the write rather than lose it).
                    try:
                        self._read_path(self.path)
                    except (OSError, UnicodeError, ValueError):
                        damaged_copy = self._damaged_copy_path()
                        shutil.copy2(self.path, damaged_copy)
                        logger.warning(
                            "Kept the unreadable SAM3 Notebook memo storage as %s",
                            damaged_copy,
                        )
                    else:
                        shutil.copy2(self.path, self.backup_path)
                os.replace(temp_path, self.path)
            finally:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
            return result


def _error(status: int, detail: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        {"detail": detail, **extra},
        status_code=status,
        headers=_NO_STORE_HEADERS,
    )


def _storage_error(error: Exception, action: str) -> JSONResponse:
    """Map store exceptions to responses; unknown errors propagate."""

    if isinstance(error, MemoConflictError):
        return _error(409, str(error), memo=error.memo)
    if isinstance(error, MemoSchemaError):
        return _error(409, str(error))
    if isinstance(error, MemoNotFoundError):
        return _error(404, str(error.args[0] if error.args else error))
    if isinstance(error, MemoCapacityError):
        return _error(413, str(error))
    if isinstance(error, MemoCorruptError):
        return _error(500, str(error))
    if isinstance(error, OSError):
        logger.exception("Failed to %s SAM3 Notebook memo storage", action)
        return _error(500, f"Memo storage I/O failed while {action}")
    if isinstance(error, (UnicodeError, ValueError, TypeError)):
        return _error(400, str(error))
    raise error


def register_memo_routes(
    app: Any,
    store: MemoStore | None = None,
    *,
    auth_dependencies: list[Any] | None = None,
) -> bool:
    """Register the memo routes exactly once. Returns False if already there."""

    for route in getattr(app, "routes", ()):
        if getattr(route, "path", None) == MEMO_API_PATH:
            return False

    memo_store = store or MemoStore()
    dependencies = (
        list(auth_dependencies)
        if auth_dependencies is not None
        else _gradio_auth_dependencies(app)
    )

    async def list_memos(request: Request) -> JSONResponse:
        require_same_origin_header(request)
        flag = request.query_params.get("include_deleted", "").strip().lower()
        try:
            payload = await run_in_threadpool(
                memo_store.list_memos,
                include_deleted=flag in ("1", "true", "yes"),
            )
        except Exception as error:  # noqa: BLE001 - mapped or re-raised below
            return _storage_error(error, "reading")
        return JSONResponse(payload, headers=_NO_STORE_HEADERS)

    async def put_memo(memo_id: str, request: Request) -> JSONResponse:
        require_same_origin_header(request)
        body = await request.body()
        if len(body) > MAX_MEMO_REQUEST_BYTES:
            return _error(413, "Memo payload is too large")
        try:
            payload = json.loads(body.decode("utf-8"))
            saved = await run_in_threadpool(memo_store.upsert, memo_id, payload)
        except Exception as error:  # noqa: BLE001 - mapped or re-raised below
            return _storage_error(error, "writing")
        return JSONResponse(saved, headers=_NO_STORE_HEADERS)

    async def delete_memo(memo_id: str, request: Request) -> JSONResponse:
        require_same_origin_header(request)
        base = request.query_params.get("base_updated_at")
        try:
            deleted = await run_in_threadpool(memo_store.delete, memo_id, base)
        except Exception as error:  # noqa: BLE001 - mapped or re-raised below
            return _storage_error(error, "writing")
        return JSONResponse(deleted, headers=_NO_STORE_HEADERS)

    item_path = MEMO_API_PATH + "/{memo_id}"
    app.add_api_route(
        MEMO_API_PATH,
        list_memos,
        methods=["GET"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam3-notebook-memos-list",
        dependencies=dependencies,
    )
    app.add_api_route(
        item_path,
        put_memo,
        methods=["PUT"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam3-notebook-memo-put",
        dependencies=dependencies,
    )
    app.add_api_route(
        item_path,
        delete_memo,
        methods=["DELETE"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam3-notebook-memo-delete",
        dependencies=dependencies,
    )
    return True


def register_memo_routes_for_notebook(
    app: Any,
    notebook_store: Any,
    auth_dependencies: list[Any],
) -> bool:
    """Called from ``register_notebook_routes``; never breaks the Notebook."""

    try:
        registered = register_memo_routes(
            app,
            MemoStore(memo_path_for_notebook(notebook_store)),
            auth_dependencies=auth_dependencies,
        )
    except Exception:  # noqa: BLE001 - the preset Notebook must keep working
        logger.exception("Failed to register SAM3 Notebook memo routes")
        return False
    if registered:
        logger.info("SAM3 Notebook memo API: %s", MEMO_API_PATH)
    return registered
