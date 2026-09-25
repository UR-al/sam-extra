from __future__ import annotations

import contextlib
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from sam3ext import lora_manager_core
from sam3ext.lora_manager_core import (
    _BRIDGE_JS,
    LORA_CONFIG_PATH,
    LORA_SPAWN_PATH,
    lora_config_data,
    register_lora_routes,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_HEADERS = {"X-SAM3-Notebook": "1"}
CONFIG_PAYLOAD = {"available": True, "replace": False, "port": 8765}
SPAWN_PAYLOAD = {
    "url": "http://127.0.0.1:8765/loras",
    "port": 8765,
    "status": "ready",
    "message": "",
}


@contextlib.contextmanager
def forge_api_auth(value, *, loaded=True, api=True, nowebui=False):
    """Forge started with ``--api-auth value``: its loaded ``modules.shared.cmd_opts`` carries it.

    ``api``/``nowebui``: whether Forge built its API at all (``--api`` / ``--nowebui``).
    ``loaded=False``: no Forge at all. Only this one ``sys.modules`` entry is swapped
    (the same stand-in as tests/test_notebook_store.py).
    """

    missing = object()
    saved = sys.modules.get("modules.shared", missing)
    if loaded:
        sys.modules["modules.shared"] = types.SimpleNamespace(
            cmd_opts=types.SimpleNamespace(api_auth=value, api=api, nowebui=nowebui))
    else:
        sys.modules.pop("modules.shared", None)
    try:
        yield
    finally:
        if saved is missing:
            sys.modules.pop("modules.shared", None)
        else:
            sys.modules["modules.shared"] = saved


def _gradio_login_app():
    """A host with Gradio's ``/login_check`` (``--gradio-auth``); the test header stands in for the cookie."""

    app = FastAPI()

    @app.get("/login_check")
    def login_check(x_test_auth: str | None = Header(default=None)):
        if x_test_auth != "ok":
            raise HTTPException(status_code=401, detail="Not authenticated")

    return app


def _get_both(client, **kwargs):
    return [client.get(path, **kwargs) for path in (LORA_CONFIG_PATH, LORA_SPAWN_PATH)]


def _statuses(client, **kwargs):
    return [response.status_code for response in _get_both(client, **kwargs)]


class LoraRouteTests(unittest.TestCase):
    def test_routes_registered_idempotently(self):
        app = FastAPI()
        self.assertTrue(register_lora_routes(app))
        self.assertFalse(register_lora_routes(app))

        paths = {route.path for route in app.routes}
        self.assertIn(LORA_CONFIG_PATH, paths)
        self.assertIn(LORA_SPAWN_PATH, paths)

        for route in app.routes:
            if route.path in (LORA_CONFIG_PATH, LORA_SPAWN_PATH):
                self.assertIn("GET", route.methods)

    def test_config_payload_shape(self):
        data = lora_config_data()
        self.assertIn("available", data)
        self.assertIn("replace", data)
        self.assertIn("port", data)
        # Without a webui/settings environment the config is safe defaults.
        self.assertIsInstance(data["available"], bool)
        self.assertIsInstance(data["replace"], bool)
        self.assertIsInstance(data["port"], int)

    def test_bridge_intercepts_single_and_bulk_sends(self):
        # The injected iframe bridge must catch BOTH the single-card context menu
        # and the multi-select bulk submenu, so nothing routes to ComfyUI.
        self.assertIn("context-menu-item[data-action]", _BRIDGE_JS)
        self.assertIn("#bulkContextMenu", _BRIDGE_JS)
        self.assertIn(".model-card.selected", _BRIDGE_JS)
        self.assertIn("stopImmediatePropagation", _BRIDGE_JS)
        # Replace vs append is carried on the message.
        self.assertIn("replace:", _BRIDGE_JS)

    def test_forge_side_handles_bulk_and_replace(self):
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(encoding="utf-8")
        # Insert helper honours the replace flag (strip existing <lora:...>).
        self.assertIn("function sam3InsertLora(text, replace)", lora)
        self.assertIn("<lora:[^>]*>", lora)
        self.assertIn("sam3InsertLora(d.text, !!d.replace)", lora)
        # The manager is injected directly into the one Forge page.
        self.assertIn("tryAllAndStop();", lora)
        self.assertNotIn("__sam3_live_workspace", lora)
        self.assertNotIn("inLiveChildFrame", lora)

    def test_dom_bootstrap_watchers_stop_after_injection(self):
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('myBtn.classList.remove("selected")', lora)
        self.assertIn("function stopBootstrapWatchers()", lora)
        self.assertIn("if (tryAll()) stopBootstrapWatchers()", lora)
        self.assertNotIn(
            "new MutationObserver(function () { tryAll(); })",
            lora,
        )

    def test_manage_tab_visibility_survives_a_tab_strip_rerender(self):
        """Tab switching must not depend on nodes captured at injection.

        Gradio 4.40 keeps non-selected TabItems mounted and only writes an
        inline display, so nothing but this code can hide the synthetic manager
        pane. Binding to captured buttons means any re-render of the tab strip
        silently drops the listeners and the manager stays on screen forever, so
        the listener is delegated to the container and the nodes are resolved on
        every click.
        """
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('container.addEventListener("click"', lora)
        self.assertIn("function liveButtons()", lora)
        self.assertIn("function livePanes()", lora)
        # No per-button listeners bound to the injection-time arrays.
        self.assertNotIn("allBtns.forEach(function (btn, idx)", lora)
        self.assertNotIn("var allPanes =", lora)

    def test_closing_manage_restores_the_previous_tab(self):
        """Gradio's state never changed while the manager was open.

        Its reactive block is therefore not dirty and it will not re-show the
        previous pane, so the restore is this code's job. Clearing the inline
        display instead would make a .tabitem fall back to display:block and
        show every pane at once.
        """
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("var savedNativeIndex = -1;", lora)
        self.assertNotIn('removeProperty("display")', lora)
        start = lora.index("function showManage(active)")
        end = lora.index("function selectTab(idx)", start)
        body = lora[start:end]
        # A real Gradio tab click wins; otherwise fall back to the saved tab.
        self.assertIn("selectedNow >= 0 ? selectedNow : savedNativeIndex", body)
        self.assertIn('pane.style.display = (index === target) ? "block" : "none"', body)

    def test_a_native_tab_click_closes_manage_without_waiting_for_gradio(self):
        """Clicking another tab must close the manager on that first click.

        Gradio flushes its tab switch asynchronously, and when it already
        considers the clicked tab selected it does not fire at all, so neither a
        deferred handler nor Gradio can be relied on. The click path closes
        synchronously and a container observer catches the async ordering.
        """
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(
            encoding="utf-8"
        )
        start = lora.index('container.addEventListener("click"')
        end = lora.index("var navWatcher", start)
        body = lora[start:end]
        # Our own button opens (deferred is fine); a native button closes now.
        self.assertIn("setTimeout(function () { showManage(true); }, 0);", body)
        self.assertIn("showManage(false);", body)
        self.assertNotIn(
            "showManage(btn === myBtn || btn.hasAttribute", body
        )
        # The observer is the ordering safety net and cannot loop.
        self.assertIn("var manageActive = false;", lora)
        self.assertIn("if (!manageActive) return;", lora)
        self.assertIn("navWatcher.observe(container,", lora)


class LoraRouteGuardTests(unittest.TestCase):
    """The config/spawn routes are guarded like the Notebook, memo and Tile & Repair routes."""

    def setUp(self):
        # No test may start a real LoRA Manager: the spawner is a mock, and the
        # real lora_spawn_data only runs wrapped so a refusal can prove that
        # nothing past the guard ran.
        self.get_or_spawn = self._patch("get_or_spawn", return_value=dict(SPAWN_PAYLOAD))
        self.spawn_data = self._patch(
            "lora_spawn_data", wraps=lora_manager_core.lora_spawn_data
        )
        self.config_data = self._patch("lora_config_data", return_value=dict(CONFIG_PAYLOAD))

    def _patch(self, name, **kwargs):
        patcher = mock.patch.object(lora_manager_core, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def assertNothingRan(self):
        self.config_data.assert_not_called()
        self.spawn_data.assert_not_called()
        self.get_or_spawn.assert_not_called()

    def test_without_the_same_origin_header_both_routes_answer_403_and_spawn_nothing(self):
        app = FastAPI()
        with forge_api_auth(None):
            register_lora_routes(app)
        with TestClient(app) as client:
            for headers in ({}, {"X-SAM3-Notebook": "0"}):
                with self.subTest(headers=headers):
                    for response in _get_both(client, headers=headers):
                        self.assertEqual(response.status_code, 403, response.text)
        self.assertNothingRan()

    def test_with_the_header_both_routes_answer_with_their_payload(self):
        app = FastAPI()
        with forge_api_auth(None):
            register_lora_routes(app)
        with TestClient(app) as client:
            config, spawn = _get_both(client, headers=NOTEBOOK_HEADERS)
        self.assertEqual((config.status_code, config.json()), (200, CONFIG_PAYLOAD))
        self.assertEqual((spawn.status_code, spawn.json()), (200, SPAWN_PAYLOAD))
        for response in (config, spawn):
            self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")
            self.assertEqual(response.headers["pragma"], "no-cache")
        self.config_data.assert_called_once_with()
        self.spawn_data.assert_called_once_with()
        self.get_or_spawn.assert_called_once()

    def _assert_forge_api_auth_guards_both_routes(self, *, api, nowebui):
        app = FastAPI()   # no Gradio /login_check: Forge's HTTP Basic guard only
        with forge_api_auth("user:secret,second:pass2", api=api, nowebui=nowebui):
            register_lora_routes(app)
        with TestClient(app) as client:
            for auth in (None, ("user", "wrong"), ("nobody", "secret"), ("second", "secret")):
                with self.subTest(auth=auth):
                    extra = {} if auth is None else {"auth": auth}
                    for response in _get_both(client, headers=NOTEBOOK_HEADERS, **extra):
                        self.assertEqual(response.status_code, 401, response.text)
                        self.assertEqual(response.headers["www-authenticate"], "Basic")
            # The right credentials still need the same-origin header.
            self.assertEqual(_statuses(client, auth=("user", "secret")), [403, 403])
            self.assertNothingRan()

            for auth in (("user", "secret"), ("second", "pass2")):
                with self.subTest(auth=auth):
                    config, spawn = _get_both(client, headers=NOTEBOOK_HEADERS, auth=auth)
                    self.assertEqual((config.status_code, config.json()), (200, CONFIG_PAYLOAD))
                    self.assertEqual((spawn.status_code, spawn.json()), (200, SPAWN_PAYLOAD))
        self.assertEqual(self.get_or_spawn.call_count, 2)

    def test_api_auth_under_forge_api_guards_both_routes_like_sdapi(self):
        self._assert_forge_api_auth_guards_both_routes(api=True, nowebui=False)

    def test_nowebui_api_auth_guards_both_routes_like_sdapi(self):
        self._assert_forge_api_auth_guards_both_routes(api=False, nowebui=True)

    def test_routes_reuse_the_gradio_login_guard(self):
        app = _gradio_login_app()
        with forge_api_auth(None):
            register_lora_routes(app)
        with TestClient(app) as client:
            self.assertEqual(_statuses(client, headers=NOTEBOOK_HEADERS), [401, 401])
            self.assertEqual(_statuses(client, headers={"X-Test-Auth": "ok"}), [403, 403])
            self.assertNothingRan()
            logged_in = {**NOTEBOOK_HEADERS, "X-Test-Auth": "ok"}
            self.assertEqual(_statuses(client, headers=logged_in), [200, 200])
        self.get_or_spawn.assert_called_once()

    def test_gradio_login_and_api_auth_are_both_required(self):
        app = _gradio_login_app()
        with forge_api_auth("user:secret"):
            register_lora_routes(app)
        logged_in = {**NOTEBOOK_HEADERS, "X-Test-Auth": "ok"}
        with TestClient(app) as client:
            self.assertEqual(_statuses(client, headers=logged_in), [401, 401])
            self.assertEqual(
                _statuses(client, headers=NOTEBOOK_HEADERS, auth=("user", "secret")), [401, 401])
            self.assertNothingRan()
            self.assertEqual(
                _statuses(client, headers=logged_in, auth=("user", "secret")), [200, 200])

    def test_api_auth_without_forge_api_needs_no_basic_credentials(self):
        # webui_worker builds Forge's Api only under --api: without it --api-auth guards no route.
        app = FastAPI()
        with forge_api_auth("user:secret", api=False, nowebui=False):
            register_lora_routes(app)
        with TestClient(app) as client:
            self.assertEqual(_statuses(client), [403, 403])
            self.assertNothingRan()
            self.assertEqual(_statuses(client, headers=NOTEBOOK_HEADERS), [200, 200])

    def test_outside_forge_only_the_header_is_needed(self):
        app = FastAPI()
        with forge_api_auth(None, loaded=False):
            register_lora_routes(app)
        with TestClient(app) as client:
            self.assertEqual(_statuses(client), [403, 403])
            self.assertEqual(_statuses(client, headers=NOTEBOOK_HEADERS), [200, 200])

    def test_explicit_auth_dependencies_replace_the_host_guards(self):
        def deny():
            raise HTTPException(status_code=401, detail="denied")

        denied = FastAPI()
        open_app = FastAPI()
        with forge_api_auth("user:secret"):
            register_lora_routes(denied, auth_dependencies=[Depends(deny)])
            register_lora_routes(open_app, auth_dependencies=[])
        with TestClient(denied) as client:
            self.assertEqual(
                _statuses(client, headers=NOTEBOOK_HEADERS, auth=("user", "secret")), [401, 401])
        self.assertNothingRan()
        with TestClient(open_app) as client:
            self.assertEqual(_statuses(client), [403, 403])
            self.assertEqual(_statuses(client, headers=NOTEBOOK_HEADERS), [200, 200])

    def test_reregistration_after_reload_ui_keeps_one_guarded_route_each(self):
        app = _gradio_login_app()
        with forge_api_auth(None):
            self.assertTrue(register_lora_routes(app))
            # Forge re-fires app_started after Reload UI.
            self.assertFalse(register_lora_routes(app))
        routes = sorted(
            (route.path, route.name, tuple(sorted(route.methods)), route.include_in_schema)
            for route in app.routes
            if route.path in (LORA_CONFIG_PATH, LORA_SPAWN_PATH)
        )
        self.assertEqual(routes, [
            (LORA_CONFIG_PATH, "sam3-lora-config", ("GET",), False),
            (LORA_SPAWN_PATH, "sam3-lora-spawn", ("GET",), False),
        ])
        with TestClient(app) as client:
            self.assertEqual(_statuses(client, headers=NOTEBOOK_HEADERS), [401, 401])
            self.assertEqual(_statuses(client, headers={"X-Test-Auth": "ok"}), [403, 403])
            self.assertNothingRan()
            logged_in = {**NOTEBOOK_HEADERS, "X-Test-Auth": "ok"}
            self.assertEqual(_statuses(client, headers=logged_in), [200, 200])


class LoraRouteDocsTests(unittest.TestCase):
    """What the notes and comments say about the config/spawn routes: who calls them and what they need."""

    def test_upgrade_notes_give_the_route_guard_its_own_bullet(self):
        # Both routes used to be open to anyone, so an external tool breaks on upgrade with or
        # without --api-auth. Inside the --api-auth bullet the header rule and the --gradio-auth
        # cookie are easy to skip, so they get an unconditional upgrade bullet of their own.
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        # The v0.30.0 notes only — a later release puts its own notes above them.
        release = changelog.index("## v0.30.0")
        release_end = changelog.find("\n## ", release + 1)
        release_end = len(changelog) if release_end < 0 else release_end
        start = changelog.index("**업그레이드할 때**", release, release_end)
        end = changelog.index("\n### ", start)
        bullets = [" ".join(bullet.split()) for bullet in re.split(r"\n(?=- )", changelog[start:end])[1:]]
        lora = [bullet for bullet in bullets if "/sam3-lora/config" in bullet and "/sam3-lora/spawn" in bullet]
        self.assertEqual(len(lora), 1, "one upgrade bullet names both LoRA Manager routes")
        (bullet,) = lora
        self.assertFalse(bullet.startswith("- Forge 를 `--api`"), bullet)
        for needle in ("`X-SAM3-Notebook: 1`", "403", "`--gradio-auth`", "401", "Manage"):
            self.assertIn(needle, bullet)
        header_bullets = [b for b in bullets if "X-SAM3-Notebook" in b]
        self.assertEqual(header_bullets, [bullet], "the header rule is not repeated inside a conditional bullet")

    def test_page_js_reaches_config_and_spawn_through_the_gradio_bridge_only(self):
        for path in sorted((ROOT / "javascript").glob("*.js")):
            with self.subTest(js=path.name):
                self.assertNotIn("/sam3-lora/", path.read_text(encoding="utf-8"))
        self.assertNotIn("/sam3-lora/", _BRIDGE_JS)
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(encoding="utf-8")
        self.assertIn('bridgeCall("sam3_lm_config_btn", "sam3_lm_config_out"', lora)
        self.assertIn('bridgeCall("sam3_lm_spawn_btn", "sam3_lm_spawn_out"', lora)


if __name__ == "__main__":
    unittest.main()
