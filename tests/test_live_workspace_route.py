from __future__ import annotations

import unittest

from fastapi import FastAPI

from sam3ext.live_workspace_route import (
    LIVE_WORKSPACE_PATH,
    live_workspace_html,
    register_live_workspace_route,
)


class LiveWorkspaceRouteTests(unittest.TestCase):
    def test_shell_html_is_lightweight_and_loads_extension_assets(self):
        html = live_workspace_html()

        self.assertIn('data-sam3-standalone-live-shell="1"', html)
        self.assertIn("/file=extensions/forge_sam3_extension/style.css?v=", html)
        self.assertIn(
            "/file=extensions/forge_sam3_extension/javascript/workspace_manager.js?v=",
            html,
        )
        self.assertNotIn("gradio-container", html)
        self.assertNotIn("/config", html)

    def test_route_registration_is_idempotent(self):
        app = FastAPI()

        self.assertTrue(register_live_workspace_route(app))
        self.assertFalse(register_live_workspace_route(app))
        matches = [route for route in app.routes if route.path == LIVE_WORKSPACE_PATH]
        self.assertEqual(len(matches), 1)
        self.assertIn("GET", matches[0].methods)


if __name__ == "__main__":
    unittest.main()
