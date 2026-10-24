"""sam-extra MCP server: sam-extra's light LoRA list (GET /sam3-lora/lite) where it is enough.

capabilities' LoRA count needs no safetensors header, so it asks the light route and falls back
to /sdapi/v1/loras on a Forge without it. The LoRA index keeps reading /sdapi/v1/loras (title,
base model, network dim and summed tag counts are not in the light list). Outputs are the same
either way.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _mcp_support import HAS_HTTPX, make_service

from sam_extra_mcp.forgeneo.client import LORA_LITE_ROUTE, route_allowed

LORAS = [
    {"name": "styleA", "alias": "styleA_v2", "path": "D:/LoRA/styleA/styleA.safetensors",
     "metadata": {"ss_output_name": "styleA_v2", "ss_base_model_version": "sdxl_base_v1-0",
                  "ss_network_dim": "32", "ss_tag_frequency": {"10_a": {"styleA": 9, "1girl": 4}}}},
    {"name": "turbo-accel", "alias": "turbo-accel", "path": "D:/LoRA/turbo.safetensors", "metadata": {}},
    {"name": "charB", "alias": "charB", "path": "D:/LoRA/charB.safetensors",
     "metadata": {"modelspec.title": "Char B"}},
]


class RouteListTests(unittest.TestCase):
    def test_the_light_route_is_allowed_for_get_only(self):
        self.assertEqual(LORA_LITE_ROUTE, "/sam3-lora/lite")
        self.assertTrue(route_allowed("GET", "/sam3-lora/lite"))
        self.assertTrue(route_allowed("GET", "/sam3-lora/lite?refresh=1"))
        self.assertFalse(route_allowed("POST", "/sam3-lora/lite"))
        self.assertFalse(route_allowed("GET", "/sam3-lora/spawn"))


@unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
class LoraLiteClientTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.service, self.fake, _tree = make_service(Path(self._tmp.name), settings={})
        self.addCleanup(self.service.close)
        self.fake.loras = [dict(item) for item in LORAS]

    def test_lora_list_reads_the_light_route_when_forge_serves_it(self):
        self.fake.lite_loras = True
        result = self.service.client.lora_list()
        self.assertTrue(result.ok, result.error)
        self.assertEqual([item["name"] for item in result.value], ["styleA", "turbo-accel", "charB"])
        self.assertNotIn("metadata", result.value[0])
        self.assertEqual(len(self.fake.calls("GET", LORA_LITE_ROUTE)), 1)
        self.assertEqual(self.fake.calls("GET", "/sdapi/v1/loras"), [])

    def test_lora_list_falls_back_to_sdapi_loras(self):
        cases = {
            "not served (older sam-extra)": None,
            "Forge without LoRA support": (503, {"detail": "sd_forge_lora is not loaded"}),
            "another version": (200, {"version": 2, "count": 0, "items": []}),
            "items not a list": (200, {"version": 1, "count": 0, "items": None}),
            "not an object": (200, [{"name": "x"}]),
        }
        for label, lite in cases.items():
            with self.subTest(label):
                self.fake.requests.clear()
                self.fake.lite_loras = lite
                result = self.service.client.lora_list()
                self.assertTrue(result.ok, result.error)
                self.assertEqual(result.value, LORAS)
                self.assertEqual([path for _method, path, _body in self.fake.requests],
                                 [LORA_LITE_ROUTE, "/sdapi/v1/loras"])

    def test_capabilities_are_the_same_with_and_without_the_light_route(self):
        without = self.service.capabilities()
        self.assertEqual(without["counts"]["loras"], 3)
        self.fake.requests.clear()
        self.fake.lite_loras = True
        with_lite = self.service.capabilities()
        self.assertEqual(with_lite, without)
        # The count came from the light route; nothing re-read the full list (the index is built).
        self.assertEqual(len(self.fake.calls("GET", LORA_LITE_ROUTE)), 1)
        self.assertEqual(self.fake.calls("GET", "/sdapi/v1/loras"), [])

    def test_an_unreachable_lora_list_is_reported_from_the_full_route(self):
        self.fake.lite_loras = (500, {"detail": "boom"})
        self.fake.loras = None  # /sdapi/v1/loras breaks too
        count = self.service.capabilities()["counts"]["loras"]
        # As before the light route existed: the error is the full route's, not the light one's.
        self.assertTrue(count.startswith("unavailable ("), count)
        self.assertIn("/sdapi/v1/loras", count)
        self.assertNotIn(LORA_LITE_ROUTE, count)

    def test_the_lora_index_still_reads_the_full_list(self):
        self.fake.lite_loras = True
        entry = self.service.loras(query="styleA", verbose=True)["results"][0]
        self.assertEqual(entry["base_model"], "sdxl_base_v1-0")
        self.assertEqual(entry["network_dim"], 32)
        self.assertEqual(entry["triggers"], ["styleA", "1girl"])
        self.assertEqual(len(self.fake.calls("GET", "/sdapi/v1/loras")), 1)


if __name__ == "__main__":
    unittest.main()
