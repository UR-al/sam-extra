"""What the agent may change through this server: read from Forge's own Settings, every time.

Settings -> SAM Extra MCP (``scripts/mcp_settings.py``) stores four booleans in Forge's settings
file. This separate process reads that file (stat-cached) before every state-changing tool call
and refuses when the switch is off, whatever the agent passes - a ``confirm=True`` included.

Why the settings file rather than ``/sdapi/v1/options`` or a sam-extra HTTP route:

- ``/sdapi/v1/options`` answers through ``OptionsModel`` (``modules/api/models.py``), which is built
  once, when that module is first imported. An extension that imports ``modules.api`` while
  scripts load - sd-webui-api-payload-display does - freezes it before ``on_ui_settings``
  registers extension options, so ``sam3_mcp_*`` can be missing from the answer entirely.
- A sam-extra route would carry sam-extra's login guards (``extension_auth_dependencies``): under
  ``--gradio-auth`` it needs the browser's Gradio session cookie, which a separate process does
  not have, so the server could never read the policy there and would have to refuse everything.
- Forge rewrites the file whenever settings are applied (``modules/ui_settings.py``) and on each
  page load (``modules_forge/main_entry.on_preset_change``), so a change applies to the next tool
  call without restarting anything. The server and Forge run on the same PC by design.

Open without a switch, on purpose: everything that only reads, ``models action=refresh`` (Forge
rescans its checkpoint folders; nothing loaded or selected changes) and ``prompt_dialect`` with
``confirm`` (an answer kept in this server's own cache, nothing in Forge).

Fail-safe rules:

- settings file location unknown (no ``<Forge data>/extensions`` above this server and no
  ``SAM_EXTRA_MCP_FORGE_CONFIG``), or Forge on another machine: everything off;
- no file at that location: Forge has never saved settings, so it runs on the defaults - the
  defaults apply (generation on, the rest off), and the report says so;
- file unreadable or not a JSON object (after short retries: Forge rewrites it in place),
  nested too deeply to parse, or larger than ``MAX_SETTINGS_BYTES``: everything off;
- a key that is not a JSON boolean: that permission off;
- a key that is absent: its default, which is what Forge itself uses until the setting is saved.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

ALLOW_GENERATE = "sam3_mcp_allow_generate"
ALLOW_MODEL_SWITCH = "sam3_mcp_allow_model_switch"
ALLOW_INTERRUPT = "sam3_mcp_allow_interrupt"
ALLOW_DOWNLOAD = "sam3_mcp_allow_download"

SETTINGS_SECTION = ("sam3_mcp", "SAM Extra MCP")
WHERE_TO_CHANGE = "Forge -> Settings -> SAM Extra MCP (Apply settings; no restart needed)"


@dataclass(frozen=True)
class Permission:
    key: str
    default: bool
    action: str
    gates: tuple[str, ...]


PERMISSIONS: tuple[Permission, ...] = (
    Permission(
        ALLOW_GENERATE,
        True,
        "generate images (txt2img / img2img)",
        ("generate",),
    ),
    Permission(
        ALLOW_MODEL_SWITCH,
        False,
        "load another checkpoint, together with its preset, VAE and text encoders - an instance-wide change",
        ("models action=load",),
    ),
    Permission(
        ALLOW_INTERRUPT,
        False,
        "interrupt or skip the running generation, including one started from the web UI",
        ("progress action=interrupt", "progress action=skip"),
    ),
    Permission(
        ALLOW_DOWNLOAD,
        False,
        "download VAE / text encoder files into Forge's models folder (and ask the host for their size)",
        ("module_download confirm=True", "module_download label=... size check"),
    ),
)
BY_KEY: Mapping[str, Permission] = MappingProxyType({permission.key: permission for permission in PERMISSIONS})
DEFAULTS: Mapping[str, bool] = MappingProxyType({permission.key: permission.default for permission in PERMISSIONS})

STATE_SETTINGS = "settings file"
STATE_DEFAULTS = "defaults"
STATE_LOCKED = "locked"

# Forge's config.json is tens of kilobytes; anything near this size is not a settings file.
MAX_SETTINGS_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Policy:
    allowed: Mapping[str, bool]
    state: str
    source: str | None
    problems: tuple[str, ...] = field(default_factory=tuple)

    def allows(self, key: str) -> bool:
        return self.allowed.get(key) is True

    def as_dict(self) -> dict:
        return {
            "permissions": {permission.key: self.allows(permission.key) for permission in PERMISSIONS},
            "state": self.state,
            "source": self.source,
            "problems": list(self.problems),
            "where_to_change": WHERE_TO_CHANGE,
        }


def locked(source: str | None, problem: str) -> Policy:
    return Policy(MappingProxyType({key: False for key in DEFAULTS}), STATE_LOCKED, source, (problem,))


def defaults(source: str | None, problem: str) -> Policy:
    return Policy(DEFAULTS, STATE_DEFAULTS, source, (problem,))


def from_settings(data: Any, source: str | None = None) -> Policy:
    """Permissions from a parsed settings file (pure)."""
    if not isinstance(data, dict):
        return locked(source, "the settings file does not hold a JSON object; every permission is off")
    allowed: dict[str, bool] = {}
    problems: list[str] = []
    for permission in PERMISSIONS:
        if permission.key not in data:
            allowed[permission.key] = permission.default
            continue
        value = data[permission.key]
        if isinstance(value, bool):
            allowed[permission.key] = value
        else:
            allowed[permission.key] = False
            problems.append(f"{permission.key} is {value!r}, not true/false; treated as off")
    return Policy(MappingProxyType(allowed), STATE_SETTINGS, source, tuple(problems))


class PolicyReader:
    """Reads the permissions from Forge's settings file, re-reading it whenever it changes."""

    def __init__(
        self,
        settings_file: str | None,
        *,
        same_machine: bool = True,
        explicit: bool = False,
        attempts: int = 3,
        retry_delay: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._path = settings_file
        self._same_machine = same_machine
        self._explicit = explicit
        self._attempts = max(1, attempts)
        self._retry_delay = retry_delay
        self._sleep = sleep
        self._lock = threading.Lock()
        self._cache_key: tuple[int, int] | None = None
        self._cache: Policy | None = None

    @property
    def settings_file(self) -> str | None:
        return self._path

    def read(self) -> Policy:
        path = self._path
        if not path:
            return locked(
                None,
                "cannot locate Forge's settings file: this server is not inside <Forge>/extensions and "
                "SAM_EXTRA_MCP_FORGE_CONFIG is not set; every permission is off",
            )
        if not self._same_machine and not self._explicit:
            return locked(
                path,
                "FORGE_URL points at another machine, whose settings this server cannot read; every "
                "permission is off (set SAM_EXTRA_MCP_FORGE_CONFIG to that Forge's settings file to override)",
            )
        with self._lock:
            for attempt in range(self._attempts):
                try:
                    stat = os.stat(path)
                except FileNotFoundError:
                    return defaults(
                        path,
                        "no settings file yet (Forge has not saved its settings), so Forge and this server "
                        "use the defaults; if Forge runs with --ui-settings-file, set SAM_EXTRA_MCP_FORGE_CONFIG",
                    )
                except OSError as exc:
                    return locked(path, f"cannot read the settings file ({exc}); every permission is off")
                key = (stat.st_mtime_ns, stat.st_size)
                if key == self._cache_key and self._cache is not None:
                    return self._cache
                if stat.st_size > MAX_SETTINGS_BYTES:
                    return locked(
                        path,
                        f"the settings file is {stat.st_size} bytes, far more than Forge's settings ever take; "
                        "every permission is off",
                    )
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        # Bounded even if the file grows after the stat: a cut-off read fails to parse.
                        data = json.loads(handle.read(MAX_SETTINGS_BYTES + 1))
                except (ValueError, UnicodeDecodeError, OSError, RecursionError) as exc:
                    # Forge rewrites the file in place (open "w" + json.dump): a read can land mid-write.
                    if attempt + 1 < self._attempts:
                        self._sleep(self._retry_delay)
                        continue
                    return locked(path, f"the settings file could not be parsed ({exc}); every permission is off")
                policy = from_settings(data, path)
                self._cache_key = key
                self._cache = policy
                return policy
        return locked(path, "the settings file could not be read")  # pragma: no cover - loop always returns


def denial(policy: Policy, key: str) -> dict:
    """The tool result for an action the policy forbids."""
    permission = BY_KEY[key]
    reasons = " ".join(policy.problems)
    message = (
        f"not allowed: {permission.action}. The operator controls this in {WHERE_TO_CHANGE} "
        f"({key}); only the operator can change it - do not edit Forge's settings yourself, ask instead."
    )
    if policy.state != STATE_SETTINGS and reasons:
        message += f" Policy state: {policy.state} - {reasons}"
    return {
        "ok": False,
        "denied_by_policy": key,
        "error": message,
        "policy": policy.as_dict(),
    }
