from __future__ import annotations

import unittest

from fastapi import FastAPI

from sam3ext.lora_manager_core import (
    LORA_CONFIG_PATH,
    LORA_SPAWN_PATH,
    lora_config_data,
    register_lora_routes,
)


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


if __name__ == "__main__":
    unittest.main()
