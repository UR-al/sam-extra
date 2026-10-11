# sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다.
"""내장 Tag Autocomplete 시작 — 헬퍼 import 와 로더가 묻는 ``GET /sam3-tagcomplete/boot``.

``start_helper()`` 는 상류 ``scripts/tag_autocomplete_helper.py`` 를 Forge 의 스크립트 로더가 하던 것처럼 스크립트를
불러올 때마다 새 모듈로 실행한다(Forge 는 불러오기 전에 콜백을 비운다 — Reload UI 에서 두 번 걸리지 않음). 헬퍼는 import
될 때 상류와 똑같이: 임시 파일(``<forge>/tmp/tagAutocompletePath.txt``·``modelKeywordPath.txt``, ``tags/temp/*``)을 쓰고,
``on_model_loaded``(임베딩)·``on_ui_settings``(tac_* 57개)·``on_app_started``(``/tacapi/v1/*`` 15개)를 건다.
``shared_paths``·``model_keyword_support`` 는 상류의 ``scripts.shared_paths`` 처럼 ``sys.modules`` 에 남는다.

``/sam3-tagcomplete/boot`` 는 켜졌는지·이유·JS 파일 목록(Forge 가 확장 JS 에 쓰는 ``file=<경로>?<mtime>`` 꼴, 이름순)을
돌려준다. 앱마다 한 번 붙고, 다른 sam-extra 경로처럼 로그인 가드(``extension_auth_dependencies``)와 같은 출처 헤더를 받는다.
"""

# No `from __future__ import annotations` here: FastAPI resolves the route handler's annotations through the module
# globals, and Request / JSONResponse are imported inside register_route().
import importlib
import os
import sys
import traceback

from sam3ext.tagcomplete import UPSTREAM_LABEL, paths

HELPER_MODULE = "sam3ext.tagcomplete.tag_autocomplete_helper"
BOOT_PATH = "/sam3-tagcomplete/boot"

# code: "ok" (running) · "off" (switch) · "standalone" (another copy enabled) · "failed" · "not started"
STATE: dict = {"enabled": False, "code": "not started", "reason": "not started yet"}


def set_state(code: str, reason: str | None) -> None:
    STATE.update(enabled=code == "ok", code=code, reason=reason)


def _report(message: str) -> None:
    try:
        from modules import errors   # Forge
    except ImportError:
        print(message, file=sys.stderr)
        traceback.print_exc()
        return
    errors.report(message, exc_info=True)


def start_helper() -> str | None:
    """헬퍼를 새로 import 한다. 성공이면 None, 실패면 이유(콘솔에 traceback 을 남긴다)."""
    sys.modules.pop(HELPER_MODULE, None)
    try:
        importlib.import_module(HELPER_MODULE)
    except Exception as exc:  # noqa: BLE001 — one extension feature must not stop Forge's script loading
        sys.modules.pop(HELPER_MODULE, None)
        _report("[sam-extra] Tag Autocomplete: the built-in helper failed to start")
        return f"failed to start: {type(exc).__name__}: {exc}"
    return None


def _webpath(path: str) -> str:
    try:
        from modules.ui_gradio_extensions import webpath   # Forge: file=<path relative to cwd>?<mtime>
    except ImportError:
        cwd = os.path.abspath(os.getcwd())
        target = os.path.abspath(path)
        try:
            if os.path.commonpath([target, cwd]) == cwd:
                target = os.path.relpath(target, cwd)
        except ValueError:
            pass
        return f"file={target}?{os.path.getmtime(path)}"
    return webpath(path)


def script_files() -> list[str]:
    """``sam3ext/tagcomplete/javascript`` 의 .js — Forge 의 ``Extension.list_files`` 와 같은 거르기·순서(``sorted(os.listdir)``)."""
    folder = str(paths.JS_DIR)
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))
            if name.endswith(".js") and os.path.isfile(os.path.join(folder, name))]


def manifest() -> dict:
    enabled = bool(STATE.get("enabled"))
    return {
        "enabled": enabled,
        "reason": STATE.get("reason"),
        "upstream": UPSTREAM_LABEL,
        "scripts": [_webpath(path) for path in script_files()] if enabled else [],
    }


def register_route(_demo, app) -> bool:
    """``on_app_started`` — ``GET /sam3-tagcomplete/boot`` 를 앱마다 한 번 붙인다(내장이 쉬어도 붙는다: 로더가 이유를 받는다)."""
    for route in getattr(app, "routes", ()):
        if getattr(route, "path", None) == BOOT_PATH:
            return False

    from fastapi import Request
    from fastapi.responses import JSONResponse

    from sam3ext.notebook_store import extension_auth_dependencies, require_same_origin_header

    async def tagcomplete_boot(request: Request) -> JSONResponse:
        require_same_origin_header(request)
        return JSONResponse(manifest(), headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})

    app.add_api_route(BOOT_PATH, tagcomplete_boot, methods=["GET"], include_in_schema=False,
                      dependencies=extension_auth_dependencies(app))
    return True
