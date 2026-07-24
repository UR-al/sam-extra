from __future__ import annotations

import unittest
from pathlib import Path

from fastapi import FastAPI

from sam3ext.lora_manager_core import (
    _BRIDGE_JS,
    LORA_CONFIG_PATH,
    LORA_SPAWN_PATH,
    lora_config_data,
    register_lora_routes,
)

ROOT = Path(__file__).resolve().parents[1]


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
        live = (ROOT / "javascript" / "live_workspaces.js").read_text(encoding="utf-8")
        # Insert helper honours the replace flag (strip existing <lora:...>).
        self.assertIn("function sam3InsertLora(text, replace)", lora)
        self.assertIn("<lora:[^>]*>", lora)
        self.assertIn("sam3InsertLora(d.text, !!d.replace)", lora)
        # Live shell forwards the replace flag to the active workspace.
        self.assertIn("replace: !!d.replace", live)


if __name__ == "__main__":
    unittest.main()
