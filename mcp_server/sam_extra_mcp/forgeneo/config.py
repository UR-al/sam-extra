# Vendored from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/config.py) into sam-extra (mcp_server/sam_extra_mcp/forgeneo/config.py).
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
# - `same_machine`: when Forge is on this PC (FORGE_URL host is a loopback address, or
#   SAM_EXTRA_MCP_SAME_MACHINE=1), `localise` returns absolute Forge paths unchanged. Upstream
#   returned None without FORGE_PATH_MAP, which made every sidecar, module and output-folder
#   lookup fail on the very setup the bridge defaults to. FORGE_PATH_MAP still wins when set.
#   The dataclass default stays False (upstream behaviour); `from_env` decides.
# - `auth` is excluded from repr(), and `describe()` reports only whether credentials are set,
#   so FORGE_AUTH never reaches a log line or a tool result. Credentials written into FORGE_URL
#   (http://user:password@host) are moved out of the URL into `auth` (FORGE_AUTH still wins),
#   because the URL itself is reported to the agent and appears in error messages.
# - is_loopback_url() accepts localhost and loopback IP literals only (upstream has no such
#   check; a name merely starting with "127." is not this machine).
# - `history_max_age` (FORGE_HISTORY_MAX_AGE, default 300 s): how old the history and LoRA
#   indexes may get before they rebuild themselves.
# - `max_pixels` (SAM_EXTRA_MCP_MAX_PIXELS, default 2048 x 2048 x 4): the most pixels one generate
#   call may ask Forge to render, all images and the hires pass included.
# - `from_env` takes an optional mapping (tests) and strips whitespace around FORGE_URL.

"""Runtime configuration, read from the environment.

The bridge never requires filesystem access: every setting below degrades to a
working default. When FORGE_PATH_MAP is set the server can translate the paths
reported by Forge into paths reachable from this machine, which lets tools
return file locations instead of megabytes of base64. When Forge runs on this
machine no mapping is needed: its paths are already ours.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit, urlunsplit

DEFAULT_URL = "http://127.0.0.1:7860"
DEFAULT_TIMEOUT = 600.0
DEFAULT_HISTORY_LIMIT = 600
DEFAULT_HISTORY_MAX_AGE = 300.0
# One generate call may render at most this many pixels in total (2048 x 2048 x 4, about 16.8 MP):
# enough for a 4096 x 4096 image or a batch of four 2048 x 2048 ones, not for a request that would
# take the GPU from whatever else runs on it.
DEFAULT_MAX_PIXELS = 2048 * 2048 * 4
ENV_MAX_PIXELS = "SAM_EXTRA_MCP_MAX_PIXELS"
ENV_SAME_MACHINE = "SAM_EXTRA_MCP_SAME_MACHINE"
LOOPBACK_HOSTS = frozenset({"localhost", "::1"})


def _parse_path_map(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse "REMOTE=LOCAL;REMOTE=LOCAL" into normalised prefix pairs.

    Remote prefixes are compared case-insensitively with forward slashes, since
    Forge reports Windows paths and we may be reading them over SMB.
    """
    pairs: list[tuple[str, str]] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        remote, local = entry.split("=", 1)
        remote = remote.strip().replace("\\", "/").rstrip("/")
        local = local.strip().replace("\\", "/").rstrip("/")
        if remote and local:
            pairs.append((remote.lower(), local))
    return tuple(pairs)


def is_loopback_url(url: str) -> bool:
    """Whether a URL points at this machine: localhost, or a loopback IP literal (127.0.0.0/8, ::1).

    Only literals count. A name merely starting with "127." (127.example.com) can resolve
    anywhere, and treating it as this PC would hand a remote Forge the local permissions.
    """
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if host in LOOPBACK_HOSTS:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or bool(mapped is not None and mapped.is_loopback)


def split_credentials(url: str) -> tuple[str, tuple[str, str] | None]:
    """Take "user:password@" out of a URL: (the URL without it, the decoded credentials or None).

    httpx would send them as Basic auth, but the URL is reported to the agent (capabilities,
    error messages), so they travel separately, like FORGE_AUTH, and the URL stays clean.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url, None
    if "@" not in parts.netloc:
        return url, None
    userinfo, _, host = parts.netloc.rpartition("@")
    user, _, password = userinfo.partition(":")
    credentials = (unquote(user), unquote(password)) if user or password else None
    return urlunsplit(parts._replace(netloc=host)), credentials


def is_absolute_path(path: str) -> bool:
    """POSIX root, UNC share or Windows drive path, in either separator style."""
    normalised = path.replace("\\", "/")
    return normalised.startswith("/") or (len(normalised) > 1 and normalised[1] == ":")


def parse_max_pixels(raw: str | None) -> int:
    """SAM_EXTRA_MCP_MAX_PIXELS: a positive pixel count ("16777216", "16_777_216", "1.6e7");
    anything else - unset, zero, negative, not a number - keeps the default."""
    text = (raw or "").strip().replace("_", "").replace(",", "")
    try:
        value = int(float(text)) if text else 0
    except (ValueError, OverflowError):
        value = 0
    return value if value > 0 else DEFAULT_MAX_PIXELS


def _env_flag(raw: str | None) -> bool | None:
    lowered = (raw or "").strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    return None


@dataclass(frozen=True)
class Config:
    url: str = DEFAULT_URL
    # Kept out of repr(): the credentials must never reach a log line or a tool result.
    auth: tuple[str, str] | None = field(default=None, repr=False)
    timeout: float = DEFAULT_TIMEOUT
    path_map: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    output_dir: str | None = None
    history_limit: int = DEFAULT_HISTORY_LIMIT
    history_max_age: float = DEFAULT_HISTORY_MAX_AGE
    same_machine: bool = False
    max_pixels: int = DEFAULT_MAX_PIXELS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        env = os.environ if env is None else env
        raw_auth = (env.get("FORGE_AUTH") or "").strip()
        auth = None
        if ":" in raw_auth:
            user, _, password = raw_auth.partition(":")
            auth = (user, password)

        raw_timeout = (env.get("FORGE_TIMEOUT") or "").strip()
        try:
            timeout = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT
        except ValueError:
            timeout = DEFAULT_TIMEOUT

        raw_limit = (env.get("FORGE_HISTORY_LIMIT") or "").strip()
        try:
            history_limit = int(raw_limit) if raw_limit else DEFAULT_HISTORY_LIMIT
        except ValueError:
            history_limit = DEFAULT_HISTORY_LIMIT

        raw_age = (env.get("FORGE_HISTORY_MAX_AGE") or "").strip()
        try:
            history_max_age = max(0.0, float(raw_age)) if raw_age else DEFAULT_HISTORY_MAX_AGE
        except ValueError:
            history_max_age = DEFAULT_HISTORY_MAX_AGE

        url, url_auth = split_credentials((env.get("FORGE_URL") or DEFAULT_URL).strip().rstrip("/"))
        # FORGE_AUTH wins, as httpx lets an explicit auth win over credentials in the URL.
        auth = auth or url_auth
        same_machine = _env_flag(env.get(ENV_SAME_MACHINE))
        if same_machine is None:
            same_machine = is_loopback_url(url)

        return cls(
            url=url,
            auth=auth,
            timeout=timeout,
            path_map=_parse_path_map(env.get("FORGE_PATH_MAP") or ""),
            output_dir=env.get("FORGE_OUTPUT_DIR") or None,
            history_limit=history_limit,
            history_max_age=history_max_age,
            same_machine=same_machine,
            max_pixels=parse_max_pixels(env.get(ENV_MAX_PIXELS)),
        )

    def localise(self, remote_path: str) -> str | None:
        """Translate a path reported by Forge into one reachable from here.

        Returns None when no mapping applies, which callers treat as "this file
        exists, but not for us" rather than as an error. On the same machine an
        absolute path needs no mapping and comes back unchanged (forward slashes).
        """
        if not remote_path:
            return None
        candidate = remote_path.replace("\\", "/")
        lowered = candidate.lower()
        for remote_prefix, local_prefix in self.path_map:
            if lowered.startswith(remote_prefix):
                return local_prefix + candidate[len(remote_prefix):]
        if self.same_machine and is_absolute_path(candidate):
            return candidate
        return None

    def describe(self) -> dict:
        """What the server is configured with, safe to hand to an agent."""
        return {
            "url": split_credentials(self.url)[0],
            "auth": "configured" if self.auth else "none",
            "timeout_seconds": self.timeout,
            "same_machine": self.same_machine,
            "path_map_entries": len(self.path_map),
            "output_dir_override": self.output_dir,
            "history_limit": self.history_limit,
            "history_max_age_seconds": self.history_max_age,
            "max_pixels_per_generate": self.max_pixels,
        }
