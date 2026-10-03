"""scripts/mcp_settings.py: the Settings -> SAM Extra MCP section the MCP server reads its policy from.

The script is loaded against stand-in Forge modules (as the other script tests do); its keys and
defaults must stay those ``sam_extra_mcp.policy`` enforces.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

from _mcp_support import ROOT

from sam_extra_mcp import policy

SCRIPT = ROOT / "scripts" / "mcp_settings.py"


class _OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None,
                 refresh=None, comment_before="", comment_after="", infotext=None, restrict_api=False, category_id=None):
        self.default = default
        self.label = label
        self.component = component
        self.section = section
        self.restrict_api = restrict_api
        self.comment = None

    def info(self, text):
        self.comment = text
        return self


class _OldOptionInfo(_OptionInfo):
    """An OptionInfo from before restrict_api existed."""

    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None):
        super().__init__(default, label, component, component_args, onchange, section)


def _load(option_info=_OptionInfo):
    registered: list = []
    added: dict = {}
    modules = types.ModuleType("modules")
    modules.script_callbacks = types.SimpleNamespace(on_ui_settings=registered.append)
    modules.shared = types.SimpleNamespace(
        OptionInfo=option_info,
        opts=types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info)),
    )
    gradio = types.ModuleType("gradio")
    gradio.Checkbox = object()
    saved = {name: sys.modules.get(name) for name in ("modules", "gradio")}
    sys.modules["modules"] = modules
    sys.modules["gradio"] = gradio
    try:
        spec = importlib.util.spec_from_file_location("_test_mcp_settings", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module, registered, added, gradio


class McpSettingsScriptTests(unittest.TestCase):
    def test_registers_through_on_ui_settings(self):
        module, registered, added, _ = _load()
        self.assertEqual(registered, [module.on_ui_settings])
        self.assertEqual(added, {}, "nothing is added before Forge asks")

    def test_the_four_switches(self):
        module, registered, added, gradio = _load()
        registered[0]()
        self.assertEqual(list(added), [
            "sam3_mcp_allow_generate", "sam3_mcp_allow_model_switch", "sam3_mcp_allow_interrupt", "sam3_mcp_allow_download",
        ])
        self.assertEqual({key: info.default for key, info in added.items()}, {
            "sam3_mcp_allow_generate": True,
            "sam3_mcp_allow_model_switch": False,
            "sam3_mcp_allow_interrupt": False,
            "sam3_mcp_allow_download": False,
        })
        for info in added.values():
            self.assertEqual(info.section, ("sam3_mcp", "SAM Extra MCP"))
            self.assertIs(info.component, gradio.Checkbox)
            self.assertIs(info.restrict_api, True)
            self.assertTrue(info.label.startswith("MCP: "))
            self.assertTrue(info.comment)

    def test_matches_what_the_server_enforces(self):
        module, registered, added, _ = _load()
        registered[0]()
        self.assertEqual({key: info.default for key, info in added.items()}, dict(policy.DEFAULTS))
        self.assertEqual(module.SECTION, policy.SETTINGS_SECTION)
        self.assertEqual(
            (module.OPT_ALLOW_GENERATE, module.OPT_ALLOW_MODEL_SWITCH, module.OPT_ALLOW_INTERRUPT, module.OPT_ALLOW_DOWNLOAD),
            (policy.ALLOW_GENERATE, policy.ALLOW_MODEL_SWITCH, policy.ALLOW_INTERRUPT, policy.ALLOW_DOWNLOAD),
        )

    def test_registration_command_names_this_installation(self):
        module, registered, added, _ = _load()
        registered[0]()
        expected_project = os.path.join(str(ROOT), "mcp_server")
        self.assertEqual(os.path.normcase(module.MCP_PROJECT), os.path.normcase(expected_project))
        command = module.registration_command()
        self.assertTrue(command.startswith("claude mcp add --scope user sam-extra -- uv run --project "))
        self.assertTrue(command.endswith(" sam-extra-mcp"))
        self.assertIn(command, added["sam3_mcp_allow_generate"].comment)
        self.assertIn("네트워크", added["sam3_mcp_allow_generate"].comment)

    def test_older_option_info_without_restrict_api(self):
        _, registered, added, _ = _load(_OldOptionInfo)
        registered[0]()
        self.assertEqual(len(added), 4)
        self.assertTrue(all(info.restrict_api is False for info in added.values()))

    def test_the_script_does_not_import_the_server(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import sam_extra_mcp", source)
        self.assertNotIn("from sam_extra_mcp", source)
        self.assertNotIn("import mcp", source)


if __name__ == "__main__":
    unittest.main()
