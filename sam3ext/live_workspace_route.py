"""Lightweight parent page for Live Workspaces.

The normal Forge root is a complete Gradio document. Using that document only
as a parent for more complete Forge iframe documents wastes one full UI build
before the active Workspace can even start. This extension-owned route serves
just the Live header/frame shell; every actual workspace still loads the
untouched Forge root in its own iframe.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


LIVE_WORKSPACE_PATH = "/sam3-live"
_ASSET_PATHS = (
    "style.css",
    "javascript/workspace_manager.js",
)


def _asset_version() -> str:
    root = Path(__file__).resolve().parents[1]
    mtimes = []
    for relative in _ASSET_PATHS:
        try:
            mtimes.append((root / relative).stat().st_mtime_ns)
        except OSError:
            continue
    return str(max(mtimes, default=0))


def live_workspace_html() -> str:
    version = escape(_asset_version(), quote=True)
    asset_root = "/file=extensions/forge_sam3_extension"
    return f"""<!doctype html>
<html lang="ko" data-sam3-standalone-live-shell="1">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark light">
  <title>Forge Neo · Live Workspaces</title>
  <link rel="stylesheet" href="{asset_root}/style.css?v={version}">
  <script defer src="{asset_root}/javascript/workspace_manager.js?v={version}"></script>
</head>
<body>
  <noscript>Live Workspaces를 사용하려면 JavaScript를 켜 주세요.</noscript>
</body>
</html>
"""


def register_live_workspace_route(app: Any) -> bool:
    """Register the extension-owned shell route once.

    Returns ``True`` only when a new route was added. Forge may invoke
    app-start callbacks again after Reload UI, so duplicate registration must
    be harmless.
    """

    for route in getattr(app, "routes", ()):
        if getattr(route, "path", None) == LIVE_WORKSPACE_PATH:
            return False

    from fastapi.responses import HTMLResponse

    async def live_workspace_shell() -> HTMLResponse:
        return HTMLResponse(
            live_workspace_html(),
            headers={
                "Cache-Control": "no-store, max-age=0",
                "Pragma": "no-cache",
            },
        )

    app.add_api_route(
        LIVE_WORKSPACE_PATH,
        live_workspace_shell,
        methods=["GET"],
        response_class=HTMLResponse,
        include_in_schema=False,
        name="sam3-live-workspaces",
    )
    return True
