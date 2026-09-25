"""Persistent storage and HTTP routes for the SAM3 Notebook.

The Notebook deliberately lives outside the extension checkout so extension
updates cannot overwrite user presets.  The default location is
``<Forge data path>/sam-extra/notebook.json``.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool


logger = logging.getLogger(__name__)

NOTEBOOK_API_PATH = "/sam3-notebook"
NOTEBOOK_SCHEMA_VERSION = 1
MAX_NOTEBOOK_BYTES = 2 * 1024 * 1024
MAX_PRESETS = 200
MAX_ENTRIES_PER_PRESET = 64
MAX_NAME_LENGTH = 80
MAX_VALUE_LENGTH = 100_000

SUPPORTED_TARGETS = frozenset(
    {
        "prompt",
        "negative_prompt",
        "lora_tokens",
        "xyz_x",
        "xyz_y",
        "xyz_z",
        "sampling_method",
        "schedule_type",
        "sampling_steps",
        "width",
        "height",
        "batch_count",
        "batch_size",
        "shift",
        "cfg_scale",
        "rescale_cfg",
        "seed",
    }
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")


class NotebookConflictError(RuntimeError):
    """Raised when a stale browser attempts to replace newer notebook data."""


class NotebookCorruptError(RuntimeError):
    """Raised when stored Notebook data exists but no valid copy can be read."""


class NotebookSchemaError(ValueError):
    """Raised rather than silently downgrading data from a newer schema."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_notebook_path() -> Path:
    try:
        from modules.paths_internal import data_path

        root = Path(data_path)
    except Exception:
        root = Path.cwd()
    return root / "sam-extra" / "notebook.json"


def empty_notebook(*, revision: int = 0) -> dict[str, Any]:
    return {
        "version": NOTEBOOK_SCHEMA_VERSION,
        "revision": max(0, int(revision)),
        "updated_at": "",
        "presets": [],
    }


def _safe_id(value: Any, prefix: str) -> str:
    candidate = str(value or "").strip()
    if _ID_RE.fullmatch(candidate):
        return candidate
    return f"{prefix}-{uuid.uuid4().hex}"


def _safe_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) > limit:
        raise ValueError(f"Notebook text exceeds the {limit} character limit")
    return text


def sanitize_notebook(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Notebook payload must be an object")

    version = payload.get("version", NOTEBOOK_SCHEMA_VERSION)
    try:
        version = int(version)
    except (TypeError, ValueError) as error:
        raise ValueError("Notebook schema version must be an integer") from error
    if version != NOTEBOOK_SCHEMA_VERSION:
        raise NotebookSchemaError(
            f"Unsupported Notebook schema version {version}; "
            f"expected {NOTEBOOK_SCHEMA_VERSION}"
        )

    revision = payload.get("revision", 0)
    try:
        revision = int(revision)
    except (TypeError, ValueError) as error:
        raise ValueError("Notebook revision must be an integer") from error
    if revision < 0:
        raise ValueError("Notebook revision must not be negative")

    result = empty_notebook(revision=revision)
    raw_presets = payload.get("presets", [])
    if not isinstance(raw_presets, list):
        raise ValueError("Notebook presets must be a list")
    if len(raw_presets) > MAX_PRESETS:
        raise ValueError(f"Notebook supports at most {MAX_PRESETS} presets")

    seen_preset_ids: set[str] = set()
    for preset_index, raw_preset in enumerate(raw_presets, start=1):
        if not isinstance(raw_preset, dict):
            raise ValueError(f"Notebook preset {preset_index} must be an object")
        preset_id = _safe_id(raw_preset.get("id"), "preset")
        while preset_id in seen_preset_ids:
            preset_id = _safe_id(None, "preset")
        seen_preset_ids.add(preset_id)

        name = _safe_text(raw_preset.get("name"), MAX_NAME_LENGTH).strip()
        if not name:
            name = f"preset{preset_index}"

        raw_entries = raw_preset.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError(f"Notebook preset {preset_index} entries must be a list")
        if len(raw_entries) > MAX_ENTRIES_PER_PRESET:
            raise ValueError(
                f"Notebook preset {preset_index} supports at most "
                f"{MAX_ENTRIES_PER_PRESET} entries"
            )

        entries: list[dict[str, str]] = []
        seen_entry_ids: set[str] = set()
        for entry_index, raw_entry in enumerate(raw_entries, start=1):
            if not isinstance(raw_entry, dict):
                raise ValueError(
                    f"Notebook preset {preset_index} entry {entry_index} "
                    "must be an object"
                )
            target = str(raw_entry.get("target") or "").strip()
            if target not in SUPPORTED_TARGETS:
                raise ValueError(
                    f"Unsupported Notebook target in preset {preset_index}: "
                    f"{target or '<empty>'}"
                )
            entry_id = _safe_id(raw_entry.get("id"), "entry")
            while entry_id in seen_entry_ids:
                entry_id = _safe_id(None, "entry")
            seen_entry_ids.add(entry_id)
            entry = {
                "id": entry_id,
                "target": target,
                "value": _safe_text(raw_entry.get("value"), MAX_VALUE_LENGTH),
            }
            if target.startswith("xyz_"):
                entry["axis_type"] = _safe_text(
                    raw_entry.get("axis_type"), MAX_NAME_LENGTH * 4
                )
            entries.append(entry)

        result["presets"].append(
            {
                "id": preset_id,
                "name": name,
                "entries": entries,
            }
        )

    updated_at = payload.get("updated_at")
    if isinstance(updated_at, str):
        result["updated_at"] = updated_at[:80]
    return result


class NotebookStore:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else default_notebook_path()
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        self._lock = threading.RLock()

    @staticmethod
    def _read_path(path: Path) -> dict[str, Any]:
        raw = path.read_bytes()
        if len(raw) > MAX_NOTEBOOK_BYTES:
            raise ValueError("Notebook file exceeds the 2 MiB safety limit")
        return sanitize_notebook(json.loads(raw.decode("utf-8")))

    def load(self) -> dict[str, Any]:
        with self._lock:
            unreadable: list[tuple[Path, Exception]] = []
            for candidate in (self.path, self.backup_path):
                try:
                    return deepcopy(self._read_path(candidate))
                except FileNotFoundError:
                    continue
                except NotebookSchemaError:
                    # A future schema is not damaged data. Do not silently
                    # substitute an older backup and later overwrite it.
                    raise
                except (
                    OSError,
                    UnicodeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as error:
                    unreadable.append((candidate, error))
                    continue
            if unreadable:
                names = ", ".join(path.name for path, _error in unreadable)
                raise NotebookCorruptError(
                    "Notebook storage is damaged and was not overwritten "
                    f"({names}). Restore a valid .bak file or move the damaged "
                    "file aside before saving."
                )
            return empty_notebook()

    def save(
        self,
        payload: Any,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        sanitized = sanitize_notebook(payload)
        with self._lock:
            current = self.load()
            current_revision = int(current.get("revision", 0) or 0)
            if (
                expected_revision is not None
                and int(expected_revision) != current_revision
            ):
                raise NotebookConflictError(
                    f"Notebook revision changed: expected {expected_revision}, "
                    f"current {current_revision}"
                )

            sanitized["revision"] = current_revision + 1
            sanitized["updated_at"] = _utc_now()
            encoded = json.dumps(
                sanitized,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) > MAX_NOTEBOOK_BYTES:
                raise ValueError("Notebook data exceeds the 2 MiB safety limit")

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
                    # Never replace a known-good backup with a corrupt primary.
                    # ``load()`` may have recovered from the backup above.
                    try:
                        self._read_path(self.path)
                    except (OSError, UnicodeError, ValueError):
                        pass
                    else:
                        shutil.copy2(self.path, self.backup_path)
                os.replace(temp_path, self.path)
            finally:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
            return deepcopy(sanitized)


def _gradio_auth_dependencies(app: Any) -> list[Any]:
    """Reuse Gradio's own login guard when the host app provides one."""

    for route in getattr(app, "routes", ()):
        if getattr(route, "path", None) != "/login_check":
            continue
        endpoint = getattr(route, "endpoint", None)
        if callable(endpoint):
            return [Depends(endpoint)]
    return []


def require_same_origin_header(request: Request) -> None:
    """Shared by the Notebook and memo routes (sam3ext/notebook_memos.py)."""

    if request.headers.get("X-SAM3-Notebook") != "1":
        raise HTTPException(
            status_code=403,
            detail="Missing same-origin Notebook request header",
        )


def register_notebook_routes(app: Any, store: NotebookStore | None = None) -> bool:
    """Register same-origin GET/PUT Notebook routes exactly once.

    The memo pad routes (``/sam3-notebook/memos``) are registered here too, so
    ``scripts/!sam3.py`` keeps a single call.
    """

    for route in getattr(app, "routes", ()):
        if getattr(route, "path", None) == NOTEBOOK_API_PATH:
            return False

    notebook_store = store or NotebookStore()
    auth_dependencies = _gradio_auth_dependencies(app)

    async def get_notebook(request: Request) -> JSONResponse:
        require_same_origin_header(request)
        try:
            payload = await run_in_threadpool(notebook_store.load)
        except NotebookSchemaError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except NotebookCorruptError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        except OSError as error:
            logger.exception("Failed to read SAM3 Notebook storage")
            raise HTTPException(
                status_code=500,
                detail="Notebook storage I/O failed while reading",
            ) from error
        return JSONResponse(payload, headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        })

    async def put_notebook(request: Request) -> JSONResponse:
        require_same_origin_header(request)
        body = await request.body()
        if len(body) > MAX_NOTEBOOK_BYTES:
            raise HTTPException(status_code=413, detail="Notebook payload is too large")
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Notebook payload must be an object")
            expected_revision = payload.get("revision", 0)
            saved = await run_in_threadpool(
                notebook_store.save,
                payload,
                expected_revision=int(expected_revision),
            )
        except NotebookConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except NotebookSchemaError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except NotebookCorruptError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        except OSError as error:
            logger.exception("Failed to write SAM3 Notebook storage")
            raise HTTPException(
                status_code=500,
                detail="Notebook storage I/O failed while writing",
            ) from error
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return JSONResponse(
            saved,
            headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
        )

    app.add_api_route(
        NOTEBOOK_API_PATH,
        get_notebook,
        methods=["GET"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam3-notebook-get",
        dependencies=auth_dependencies,
    )
    app.add_api_route(
        NOTEBOOK_API_PATH,
        put_notebook,
        methods=["PUT"],
        response_class=JSONResponse,
        include_in_schema=False,
        name="sam3-notebook-put",
        dependencies=auth_dependencies,
    )
    # Imported here: notebook_memos imports this module's helpers.
    from .notebook_memos import register_memo_routes_for_notebook

    register_memo_routes_for_notebook(app, notebook_store, auth_dependencies)
    return True
