# Vendored from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/fetcher.py) into sam-extra (mcp_server/sam_extra_mcp/forgeneo/fetcher.py).
#
# Copyright (c) 2026 Eduardo Abreu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes in sam-extra (2026-10-03):
# - Verification. A download is kept only when its size matches the size the server announced
#   and, where Hugging Face states it (X-Linked-ETag on the /resolve/ redirect, the SHA256 of an
#   LFS file), its SHA256 matches too; a .safetensors file must also open as one. A download
#   whose size cannot be learnt beforehand is refused rather than kept unverified. Upstream
#   checked the size only when a Content-Length happened to arrive.
# - fetch() takes a keyword-only `permitted` with no default: the caller states the outcome of
#   the policy check (Settings -> SAM Extra MCP -> sam3_mcp_allow_download) and nothing is
#   requested when it is False.
# - plan(probe=False) works out the local side without any network request; local refusals
#   (file present, folder missing or read-only) are decided before the size is asked for.
# - httpx replaces urllib (injectable transport for tests); the HEAD request does not follow
#   redirects, so the size probe can never turn into a GET of the whole file. A stream that
#   runs past the announced size is cut off. The file is never overwritten: the finished
#   download is moved into place with an operation that fails when the name exists
#   (place_new_file: os.rename on Windows, a hard link elsewhere), not checked and then replaced.

"""Fetching a missing module, only when told to.

Everything else in this bridge is read-only. This is the one place that writes,
and it writes multi-gigabyte files into someone's models folder — often across a
network share, sometimes onto a disk with no room. So the default is to report
what would happen and stop: the caller has to pass an explicit confirmation for
anything to be downloaded.

Downloads land in a .part file and are renamed on completion, so an interrupted
transfer never leaves something that looks like a usable model. A finished
transfer is renamed only after its size (and, when the host states it, its
SHA256) has been checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

try:  # only a live download needs httpx; planning without a probe does not
    import httpx
except ImportError:  # pragma: no cover - httpx ships with the server's own environment
    httpx = None  # type: ignore[assignment]

CHUNK = 1024 * 1024
TIMEOUT = 60.0
USER_AGENT = "sam-extra-mcp (forgeneo-mcp)"
# Refuse to start a download that would leave the disk under this much room.
HEADROOM_BYTES = 2 * 1024 ** 3
MAX_REDIRECTS = 5
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024


@dataclass(frozen=True)
class RemoteFile:
    size: int | None
    sha256: str | None
    error: str | None = None


@dataclass(frozen=True)
class Plan:
    url: str
    destination: str
    size_bytes: int | None
    free_bytes: int | None
    already_present: bool
    writable: bool
    blocked: str | None = None
    sha256: str | None = None
    probed: bool = False

    @property
    def verification(self) -> str:
        if self.size_bytes and self.sha256:
            return "size and sha256"
        if self.size_bytes:
            return "size only"
        return "not possible (size unknown)" if self.probed else "not checked yet"

    def as_dict(self) -> dict:
        def gb(value):
            return None if value is None else round(value / 1024 ** 3, 2)

        return {
            "url": self.url,
            "destination": self.destination,
            "size_gb": gb(self.size_bytes),
            "free_gb": gb(self.free_bytes),
            "already_present": self.already_present,
            "writable": self.writable,
            "blocked": self.blocked,
            "sha256": self.sha256,
            "verification": self.verification,
            "probed": self.probed,
        }


def _int_header(headers: Any, name: str) -> int | None:
    raw = headers.get(name)
    try:
        value = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and value >= 0 else None


def _sha256_from_etag(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip()
    if value.startswith("W/"):
        value = value[2:]
    value = value.strip('"').lower()
    return value if SHA256_PATTERN.match(value) else None


def _http_client(transport: Any = None, follow_redirects: bool = False):
    if httpx is None:  # pragma: no cover - see the import above
        raise RuntimeError("httpx is required to download modules")
    return httpx.Client(
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=follow_redirects,
        transport=transport,
    )


def remote_file(url: str, transport: Any = None) -> RemoteFile:
    """Ask how large a file is, and its SHA256 where the host states it.

    Hugging Face answers a HEAD on /resolve/ with a redirect carrying
    X-Linked-Size and X-Linked-ETag, which describe the file itself; for LFS
    files the ETag is the SHA256 of the content (huggingface_hub's
    get_hf_file_metadata reads the same headers). Other hosts give a
    Content-Length. Redirects are followed by hand, with HEAD, never GET.
    """
    try:
        with _http_client(transport) as client:
            response = client.head(url)
            size = _int_header(response.headers, "x-linked-size")
            sha256 = _sha256_from_etag(response.headers.get("x-linked-etag"))
            hops = 0
            current = url
            while response.is_redirect and size is None and hops < MAX_REDIRECTS:
                location = response.headers.get("location")
                if not location:
                    break
                current = urljoin(current, location)
                response = client.head(current)
                hops += 1
                size = _int_header(response.headers, "x-linked-size")
                sha256 = sha256 or _sha256_from_etag(response.headers.get("x-linked-etag"))
            if response.status_code >= 400:
                return RemoteFile(None, None, f"HTTP {response.status_code} for {current}")
            if size is None and not response.is_redirect:
                size = _int_header(response.headers, "content-length")
            return RemoteFile(size, sha256)
    except Exception as exc:  # httpx.HTTPError, socket errors, bad URLs
        return RemoteFile(None, None, f"cannot reach {url}: {str(exc)[:120]}")


def plan(url: str, folder: str, filename: str, *, probe: bool = True, transport: Any = None) -> Plan:
    """Work out what fetching this would involve, without doing any of it.

    With probe=False nothing leaves the machine: the size is simply not known yet.
    """
    destination = os.path.join(folder, filename)
    present = os.path.isfile(destination)
    writable = os.path.isdir(folder) and os.access(folder, os.W_OK)

    free = None
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        pass

    blocked = None
    if present:
        blocked = "file already exists; nothing to do"
    elif not os.path.isdir(folder):
        blocked = f"target folder does not exist or is unreachable: {folder}"
    elif not writable:
        blocked = f"target folder is not writable from here: {folder}"

    size = sha256 = None
    probed = False
    if blocked is None and probe:
        remote = remote_file(url, transport=transport)
        probed = True
        size, sha256 = remote.size, remote.sha256
        if remote.error:
            blocked = f"cannot check the file before downloading it: {remote.error}"
        elif size and free is not None and free - size < HEADROOM_BYTES:
            blocked = (
                f"not enough room: needs {size / 1024 ** 3:.1f} GB and only "
                f"{free / 1024 ** 3:.1f} GB is free"
            )

    return Plan(url, destination, size, free, present, writable, blocked, sha256, probed)


class _Oversize(Exception):
    pass


def fetch(url: str, folder: str, filename: str, *, permitted: bool, transport: Any = None) -> dict:
    """Download the file. Callers are expected to have confirmed with a human,
    and to pass the policy decision as `permitted`."""
    if not permitted:
        return {"ok": False, "error": "downloads are not permitted (Settings -> SAM Extra MCP)"}
    intent = plan(url, folder, filename, transport=transport)
    if intent.blocked:
        return {"ok": False, "error": intent.blocked, "plan": intent.as_dict()}
    if not intent.size_bytes:
        return {
            "ok": False,
            "error": (
                "the server did not state the file size, so the download could not be verified; "
                f"fetch {url} into {folder} by hand if you trust it"
            ),
            "plan": intent.as_dict(),
        }

    expected = intent.size_bytes
    partial = intent.destination + ".part"
    digest = hashlib.sha256()
    written = 0
    try:
        with _http_client(transport, follow_redirects=True) as client, client.stream("GET", url) as response:
            if response.status_code >= 400:
                return {"ok": False, "error": f"HTTP {response.status_code} downloading {url}"}
            with open(partial, "wb") as handle:
                for block in response.iter_bytes(CHUNK):
                    written += len(block)
                    if written > expected:
                        raise _Oversize()
                    digest.update(block)
                    handle.write(block)
    except _Oversize:
        _discard(partial)
        return {"ok": False, "error": f"the server sent more than the announced {expected} bytes; discarded"}
    except Exception as exc:  # httpx.HTTPError, OSError
        _discard(partial)
        return {"ok": False, "error": f"download failed after {written} bytes: {str(exc)[:120]}"}

    if written != expected:
        _discard(partial)
        return {
            "ok": False,
            "error": f"incomplete: expected {expected} bytes, received {written}",
        }

    actual = digest.hexdigest()
    if intent.sha256 and actual != intent.sha256:
        _discard(partial)
        return {
            "ok": False,
            "error": f"SHA256 mismatch: expected {intent.sha256}, received {actual}; discarded",
        }

    if filename.lower().endswith(".safetensors") and not _looks_like_safetensors(partial):
        _discard(partial)
        return {"ok": False, "error": "the downloaded file is not a valid safetensors file; discarded"}

    try:
        placed = place_new_file(partial, intent.destination)
    except OSError as exc:
        _discard(partial)
        return {"ok": False, "error": f"could not finalise the file: {exc}"}
    if not placed:
        _discard(partial)
        return {"ok": False, "error": f"{intent.destination} appeared during the download; left untouched"}

    return {
        "ok": True,
        "saved": intent.destination,
        "size_gb": round(written / 1024 ** 3, 2),
        "sha256": actual,
        "verified": "size and sha256" if intent.sha256 else "size only",
        "next_step": "refresh the module list in Forge, then select it for this preset",
    }


def place_new_file(partial: str, destination: str, *, rename_refuses_existing: bool = os.name == "nt") -> bool:
    """Move a finished download to `destination` unless something is there; False when it is.

    The check and the move are one operation, so a file that appears at the last moment is
    never overwritten: on Windows os.rename (MoveFileEx without MOVEFILE_REPLACE_EXISTING)
    fails when the destination exists; elsewhere a hard link does (link(2) never replaces),
    and the partial name is removed afterwards. Only a file system without hard links falls
    back to checking first, which leaves the old few-microsecond window.
    """
    if rename_refuses_existing:
        try:
            os.rename(partial, destination)
        except FileExistsError:
            return False
        return True
    try:
        os.link(partial, destination)
    except FileExistsError:
        return False
    except OSError:
        if os.path.exists(destination):
            return False
        os.replace(partial, destination)
        return True
    _discard(partial)
    return True


def _looks_like_safetensors(path: str) -> bool:
    """An 8-byte little-endian header length followed by a JSON object."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return False
            length = struct.unpack("<Q", raw)[0]
            if length == 0 or length > MAX_SAFETENSORS_HEADER:
                return False
            header = handle.read(length)
    except OSError:
        return False
    if len(header) < length:
        return False
    try:
        parsed = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return isinstance(parsed, dict)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
