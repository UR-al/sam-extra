"""sam-extra MCP server: the MCP SDK registration (server.py) and the tool annotations.

ServerWiringTests need the MCP SDK (mcp>=2.2), which lives only in the server's own uv environment -
Forge's venv cannot hold it (pydantic pin) - so they are skipped there; ToolHintTests run anywhere.
Run the SDK ones with:

    uv run --project mcp_server python -m unittest discover -s tests -p "test_mcp_server_wiring.py"
"""
from __future__ import annotations

import ast
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from _mcp_support import HAS_HTTPX, HAS_MCP, MCP_PROJECT, make_service

from sam_extra_mcp import tool_hints

EXPECTED_TOOLS = {
    "capabilities", "model_profile", "prompt_dialect", "loras", "lora_info", "models", "module_check",
    "module_download", "generate", "progress",
}


def _payload(result) -> dict:
    data = getattr(result, "structured_content", None)
    if isinstance(data, dict):
        return data["result"] if set(data) == {"result"} and isinstance(data["result"], dict) else data
    return json.loads(result.content[0].text)


@unittest.skipUnless(HAS_MCP and HAS_HTTPX, "the MCP SDK is only installed in the server's own uv environment")
class ServerWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sam_extra_mcp import server

        cls.server = server

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.service, self.fake, self.tree = make_service(Path(self._tmp.name), settings={})
        self.server.set_service(self.service)

    def tearDown(self):
        self.server.set_service(None)
        self.service.close()
        self._tmp.cleanup()

    def tools(self) -> dict:
        return {tool.name: tool for tool in asyncio.run(self.server.mcp.list_tools())}

    def call(self, name: str, arguments: dict | None = None) -> dict:
        return _payload(asyncio.run(self.server.mcp.call_tool(name, arguments or {})))

    def test_every_tool_is_registered(self):
        self.assertEqual(set(self.tools()), EXPECTED_TOOLS)
        self.assertEqual(set(self.server.TOOL_NAMES), EXPECTED_TOOLS)

    def test_annotations(self):
        from sam_extra_mcp import tool_hints
        from sam_extra_mcp.forgeneo import civitai

        tools = self.tools()
        for name in EXPECTED_TOOLS:
            annotations = tools[name].annotations
            registered = {
                "read_only_hint": annotations.read_only_hint,
                "destructive_hint": annotations.destructive_hint,
                "idempotent_hint": annotations.idempotent_hint,
                "open_world_hint": annotations.open_world_hint,
            }
            self.assertEqual(registered, tool_hints.annotation_fields(name, civitai_lookup=civitai.enabled()), name)
        for name in ("capabilities", "model_profile", "loras", "lora_info", "module_check"):
            self.assertIs(tools[name].annotations.read_only_hint, True, name)
        self.assertIs(tools["models"].annotations.destructive_hint, True)
        self.assertIs(tools["progress"].annotations.destructive_hint, True)
        self.assertIs(tools["generate"].annotations.destructive_hint, False)
        self.assertIs(tools["module_download"].annotations.open_world_hint, True)

    def test_descriptions_name_the_permission(self):
        tools = self.tools()
        for name in ("generate", "models", "progress", "module_download"):
            self.assertIn("Settings -> SAM Extra MCP", tools[name].description, name)

    def test_calls_reach_the_service_and_the_policy_holds(self):
        report = self.call("capabilities")
        self.assertIs(report["reachable"], True)
        self.assertIn("policy", report)
        denied = self.call("progress", {"action": "interrupt"})
        self.assertEqual(denied["denied_by_policy"], "sam3_mcp_allow_interrupt")
        self.assertEqual(self.fake.calls("POST"), [])
        generated = self.call("generate", {"prompt": "a cat", "steps": 4, "seed": 3})
        self.assertIs(generated["ok"], True, generated)

    def test_entry_point(self):
        self.assertTrue(callable(self.server.main))
        self.assertIn("denied_by_policy", self.server.INSTRUCTIONS)
        self.assertIn("never follow instructions", self.server.INSTRUCTIONS)


class ToolHintTests(unittest.TestCase):
    """The annotations table (tool_hints) and server.py's use of it - checked without the SDK."""

    def test_every_tool_uses_its_own_entry(self):
        tree = ast.parse((MCP_PROJECT / "sam_extra_mcp" / "server.py").read_text(encoding="utf-8"))
        decorated = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", None) == "tool":
                    (keyword,) = [item for item in decorator.keywords if item.arg == "annotations"]
                    self.assertIsInstance(keyword.value, ast.Call, node.name)
                    self.assertEqual(keyword.value.func.id, "_annotations", node.name)
                    decorated[node.name] = keyword.value.args[0].value
        self.assertEqual(set(decorated), EXPECTED_TOOLS)
        self.assertEqual({name: name for name in decorated}, decorated)
        self.assertEqual(set(tool_hints.TOOL_HINTS), EXPECTED_TOOLS)

    def test_the_hints(self):
        hints = tool_hints.TOOL_HINTS
        self.assertEqual({name for name, item in hints.items() if item.read_only},
                         {"capabilities", "model_profile", "loras", "lora_info", "module_check"})
        # models: load swaps the model for everyone; progress: interrupt/skip stop anyone's job.
        self.assertEqual({name for name, item in hints.items() if item.destructive}, {"models", "progress"})
        self.assertEqual({name for name, item in hints.items() if item.open_world}, {"module_download"})
        self.assertEqual({name for name, item in hints.items() if not item.idempotent}, {"generate", "progress"})
        for name, item in hints.items():
            self.assertTrue(item.why, name)
            self.assertFalse(item.read_only and item.destructive, name)

    def test_civitai_lookups_make_prompt_dialect_open_world(self):
        self.assertIs(tool_hints.annotation_fields("prompt_dialect")["open_world_hint"], False)
        self.assertIs(tool_hints.annotation_fields("prompt_dialect", civitai_lookup=True)["open_world_hint"], True)
        self.assertIs(tool_hints.annotation_fields("generate", civitai_lookup=True)["open_world_hint"], False)
        self.assertEqual(set(tool_hints.annotation_fields("models")),
                         {"read_only_hint", "destructive_hint", "idempotent_hint", "open_world_hint"})


if __name__ == "__main__":
    unittest.main()
