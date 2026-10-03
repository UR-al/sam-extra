# Vendored from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/client.py) into sam-extra (mcp_server/sam_extra_mcp/forgeneo/client.py).
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
# - Route allow-list (ALLOWED_ROUTES): the client refuses, without touching the network, any
#   method/route the server does not use. /sdapi/v1/cmd-flags and /internal/sysinfo are named in
#   NEVER_REQUESTED as well: cmd-flags answers with vars(shared.cmd_opts), which carries the
#   --api-auth / --gradio-auth credentials in clear text. Redirects are not followed and a 3xx
#   answer is an error (upstream: follow_redirects=True), so the list covers every request sent.
# - Options writes only carry the keys a checkpoint switch needs (SETTABLE_OPTIONS); anything
#   else - the sam3_mcp_* permission keys above all - is refused before a request is made. The
#   check sits in the request path itself, so it holds for post() as well as set_options().
# - progress() asks Forge to skip the live preview image (skip_current_image) so a status call
#   does not put a base64 image into the agent's context.
# - Optional httpx transport (tests use httpx.MockTransport); requests may carry query params.
# - httpx is imported lazily: ApiResult and the rest of the core import without it.

"""HTTP client for the Forge Neo REST API.

Every call returns an explicit outcome instead of raising, because a bridge is
expected to survive a Forge instance that is booting, busy, or missing a route.
The instance probed during development had /sdapi/v1/cmd-flags returning 500
while every other route worked, so "route exists" and "route works" are treated
as different questions.

sam-extra only ever calls the routes in ALLOWED_ROUTES. /sdapi/v1/cmd-flags in
particular is never requested: Forge answers it with its whole command line,
credentials included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:  # the core (and its tests) work without httpx; only a live client needs it
    import httpx
except ImportError:  # pragma: no cover - httpx ships with the server's own environment
    httpx = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .config import Config

ALLOWED_ROUTES = frozenset(
    {
        ("GET", "/sdapi/v1/options"),
        ("GET", "/sdapi/v1/sd-models"),
        ("GET", "/sdapi/v1/loras"),
        ("GET", "/sdapi/v1/sd-modules"),
        ("GET", "/sdapi/v1/samplers"),
        ("GET", "/sdapi/v1/schedulers"),
        ("GET", "/sdapi/v1/progress"),
        ("GET", "/openapi.json"),
        ("POST", "/sdapi/v1/options"),
        ("POST", "/sdapi/v1/refresh-checkpoints"),
        ("POST", "/sdapi/v1/interrupt"),
        ("POST", "/sdapi/v1/skip"),
        ("POST", "/sdapi/v1/txt2img"),
        ("POST", "/sdapi/v1/img2img"),
    }
)
# Never requested, whatever the caller asks. cmd-flags returns vars(shared.cmd_opts) - the
# --api-auth and --gradio-auth credentials included - and sysinfo dumps settings and environment.
NEVER_REQUESTED = ("/sdapi/v1/cmd-flags", "/internal/sysinfo")
# The only options this server writes: a checkpoint switch and the preset/modules that go with it.
SETTABLE_OPTIONS = frozenset({"sd_model_checkpoint", "forge_preset", "forge_additional_modules"})
OPTIONS_ROUTE = "/sdapi/v1/options"


@dataclass(frozen=True)
class ApiResult:
    ok: bool
    data: Any = None
    error: str | None = None

    @property
    def value(self) -> Any:
        return self.data if self.ok else None


def _route(path: str) -> str:
    return path.split("?", 1)[0].split("#", 1)[0]


def route_allowed(method: str, path: str) -> bool:
    route = _route(path)
    if any(route.startswith(blocked) for blocked in NEVER_REQUESTED):
        return False
    return (method.upper(), route) in ALLOWED_ROUTES


def unsettable_options(values: Any) -> list[str]:
    """The keys of an options write this server refuses (all of them unless it is a JSON object)."""
    if not isinstance(values, dict):
        return ["<not an object>"]
    return sorted(str(key) for key in set(values) - SETTABLE_OPTIONS)


class ForgeClient:
    """Thin, fault-tolerant wrapper around the Forge REST API."""

    def __init__(self, config: Config, transport: Any = None) -> None:
        if httpx is None:  # pragma: no cover - see the import above
            raise RuntimeError("httpx is required to talk to Forge (it is a dependency of sam-extra-mcp)")
        self._config = config
        self._client = httpx.Client(
            base_url=config.url,
            timeout=config.timeout,
            auth=config.auth,
            # Redirects are not followed (upstream followed them): the route allow-list is checked
            # on the path asked for, so a redirect could otherwise lead anywhere on the server.
            follow_redirects=False,
            transport=transport,
        )

    @property
    def config(self) -> Config:
        return self._config

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, timeout: float | None = None, params: dict[str, Any] | None = None) -> ApiResult:
        return self._request("GET", path, timeout=timeout, params=params)

    def post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> ApiResult:
        return self._request("POST", path, json=payload, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> ApiResult:
        if not route_allowed(method, path):
            return ApiResult(False, error=f"refused: {method} {_route(path)} is not a route this server uses")
        if method.upper() == "POST" and _route(path) == OPTIONS_ROUTE:
            # Enforced here, not only in set_options(), so no caller of post() can write others.
            refused = unsettable_options(kwargs.get("json"))
            if refused:
                return ApiResult(
                    False,
                    error=f"refused: this server only sets {', '.join(sorted(SETTABLE_OPTIONS))}, not {', '.join(refused)}",
                )
        timeout = kwargs.pop("timeout", None)
        if kwargs.get("params") is None:
            kwargs.pop("params", None)
        try:
            response = self._client.request(
                method,
                path,
                timeout=timeout if timeout is not None else self._config.timeout,
                **kwargs,
            )
        except httpx.TimeoutException:
            return ApiResult(False, error=f"timeout calling {path}")
        except httpx.HTTPError as exc:
            return ApiResult(False, error=f"cannot reach {self._config.url}{path}: {exc}")

        if 300 <= response.status_code < 400:
            return ApiResult(
                False,
                error=(
                    f"refused: {path} answered with a redirect (HTTP {response.status_code}), which this "
                    "server does not follow; FORGE_URL must be Forge's own address"
                ),
            )
        if response.status_code >= 400:
            return ApiResult(False, error=f"HTTP {response.status_code} on {path}: {response.text[:200]}")

        try:
            return ApiResult(True, data=response.json())
        except ValueError:
            return ApiResult(False, error=f"non-JSON response from {path}")

    # -- convenience wrappers -------------------------------------------------

    def options(self) -> ApiResult:
        return self.get("/sdapi/v1/options")

    def set_options(self, values: dict[str, Any]) -> ApiResult:
        return self.post(OPTIONS_ROUTE, values)  # _request refuses any key outside SETTABLE_OPTIONS

    def checkpoints(self) -> ApiResult:
        return self.get("/sdapi/v1/sd-models")

    def loras(self) -> ApiResult:
        return self.get("/sdapi/v1/loras")

    def modules(self) -> ApiResult:
        return self.get("/sdapi/v1/sd-modules")

    def samplers(self) -> ApiResult:
        return self.get("/sdapi/v1/samplers")

    def schedulers(self) -> ApiResult:
        return self.get("/sdapi/v1/schedulers")

    def progress(self) -> ApiResult:
        return self.get("/sdapi/v1/progress", timeout=15.0, params={"skip_current_image": "true"})

    def interrupt(self) -> ApiResult:
        return self.post("/sdapi/v1/interrupt", {}, timeout=15.0)

    def skip(self) -> ApiResult:
        return self.post("/sdapi/v1/skip", {}, timeout=15.0)

    def refresh_checkpoints(self) -> ApiResult:
        return self.post("/sdapi/v1/refresh-checkpoints", {})

    def openapi(self) -> ApiResult:
        return self.get("/openapi.json")

    def txt2img(self, payload: dict[str, Any]) -> ApiResult:
        return self.post("/sdapi/v1/txt2img", payload)

    def img2img(self, payload: dict[str, Any]) -> ApiResult:
        return self.post("/sdapi/v1/img2img", payload)
