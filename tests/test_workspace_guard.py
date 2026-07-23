from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from sam3ext.workspace_guard import (
    guard_sampler_app_started_callbacks,
    prune_stale_sampler_load_targets,
)


ROOT = Path(__file__).resolve().parents[1]


class _Component:
    def __init__(self, component_id: int):
        self._id = component_id


class _Demo:
    def __init__(self, *components: _Component):
        self.blocks = {component._id: component for component in components}


class WorkspaceSamplerGuardTests(unittest.TestCase):
    def test_sampler_callbacks_prune_immediately_before_demo_load_registration(self):
        rk_current = (_Component(10), _Component(11))
        rk_stale = (_Component(110), _Component(111))
        tde_current = _Component(20)
        tde_stale = _Component(120)

        rk_module = types.ModuleType("_sam3_test_rk_sampler_wrapped")
        rk_module.__file__ = "C:/extensions/sd-webui-rk-sampler/scripts/rk_sampler.py"
        rk_module._rk_method_dropdowns = [rk_stale, rk_current]
        tde_module = types.ModuleType("_sam3_test_tde_sampler_wrapped")
        tde_module.__file__ = "C:/extensions/sd-webui-tde-sampler/scripts/tde_sampler.py"
        tde_module._tde_max_steps_sliders = [tde_stale, tde_current]

        observed = {}
        rk_module.observed = observed
        tde_module.observed = observed
        exec(
            "def callback(_demo, _app):\n"
            "    observed['rk'] = list(_rk_method_dropdowns)\n",
            rk_module.__dict__,
        )
        exec(
            "def callback(_demo, _app):\n"
            "    observed['tde'] = list(_tde_max_steps_sliders)\n",
            tde_module.__dict__,
        )

        rk_registered = types.SimpleNamespace(
            script="C:/extensions/sd-webui-rk-sampler/scripts/rk_sampler.py",
            callback=rk_module.callback,
        )
        tde_registered = types.SimpleNamespace(
            script="C:/extensions/sd-webui-tde-sampler/scripts/tde_sampler.py",
            callback=tde_module.callback,
        )
        callback_map = {
            "callbacks_app_started": [rk_registered, tde_registered],
        }

        installed = guard_sampler_app_started_callbacks(callback_map)
        installed_again = guard_sampler_app_started_callbacks(callback_map)
        demo = _Demo(*rk_current, tde_current)
        rk_registered.callback(demo, None)
        tde_registered.callback(demo, None)

        self.assertEqual(installed, 2)
        self.assertEqual(installed_again, 0)
        self.assertEqual(observed["rk"], [rk_current])
        self.assertEqual(observed["tde"], [tde_current])

    def test_forge_loaded_scripts_are_pruned_without_sys_modules_registration(self):
        rk_current = (_Component(10), _Component(11))
        rk_stale = (_Component(110), _Component(111))
        tde_current = _Component(20)
        tde_stale = _Component(120)

        rk_module = types.ModuleType("_sam3_test_rk_sampler_forge_loader")
        rk_module.__file__ = "C:/extensions/sd-webui-rk-sampler/scripts/rk_sampler.py"
        rk_module._rk_method_dropdowns = [rk_stale, rk_current]
        tde_module = types.ModuleType("_sam3_test_tde_sampler_forge_loader")
        tde_module.__file__ = "C:/extensions/sd-webui-tde-sampler/scripts/tde_sampler.py"
        tde_module._tde_max_steps_sliders = [tde_stale, tde_current]

        modules_package = types.ModuleType("modules")
        modules_package.__path__ = []
        script_loading = types.ModuleType("modules.script_loading")
        script_loading.loaded_scripts = {
            rk_module.__file__: rk_module,
            tde_module.__file__: tde_module,
        }
        modules_package.script_loading = script_loading
        previous_modules = sys.modules.get("modules")
        previous_script_loading = sys.modules.get("modules.script_loading")
        sys.modules["modules"] = modules_package
        sys.modules["modules.script_loading"] = script_loading
        try:
            removed = prune_stale_sampler_load_targets(
                _Demo(*rk_current, tde_current)
            )
        finally:
            if previous_modules is None:
                sys.modules.pop("modules", None)
            else:
                sys.modules["modules"] = previous_modules
            if previous_script_loading is None:
                sys.modules.pop("modules.script_loading", None)
            else:
                sys.modules["modules.script_loading"] = previous_script_loading

        self.assertNotIn(rk_module.__name__, sys.modules)
        self.assertNotIn(tde_module.__name__, sys.modules)
        self.assertEqual(removed, {"rk": 1, "tde": 1})
        self.assertEqual(rk_module._rk_method_dropdowns, [rk_current])
        self.assertEqual(tde_module._tde_max_steps_sliders, [tde_current])

    def test_stale_rk_and_tde_load_targets_are_pruned_in_place(self):
        rk_current = (_Component(10), _Component(11))
        rk_stale = (_Component(110), _Component(111))
        tde_current = _Component(20)
        tde_stale = _Component(120)

        rk_module = types.ModuleType("_sam3_test_rk_sampler")
        rk_module.__file__ = "C:/extensions/sd-webui-rk-sampler/scripts/rk_sampler.py"
        rk_module._rk_method_dropdowns = [rk_stale, rk_current]
        tde_module = types.ModuleType("_sam3_test_tde_sampler")
        tde_module.__file__ = "C:/extensions/sd-webui-tde-sampler/scripts/tde_sampler.py"
        tde_module._tde_max_steps_sliders = [tde_stale, tde_current]

        sys.modules[rk_module.__name__] = rk_module
        sys.modules[tde_module.__name__] = tde_module
        try:
            removed = prune_stale_sampler_load_targets(
                _Demo(*rk_current, tde_current)
            )
        finally:
            sys.modules.pop(rk_module.__name__, None)
            sys.modules.pop(tde_module.__name__, None)

        self.assertEqual(removed, {"rk": 1, "tde": 1})
        self.assertEqual(rk_module._rk_method_dropdowns, [rk_current])
        self.assertEqual(tde_module._tde_max_steps_sliders, [tde_current])

    def test_guard_callback_is_ordered_before_sampler_load_callbacks(self):
        metadata = (ROOT / "metadata.ini").read_text(encoding="utf-8")
        script = (ROOT / "scripts" / "!sam3.py").read_text(encoding="utf-8")

        self.assertIn("workspace-sampler-load-guard", metadata)
        self.assertIn("sd-webui-rk-sampler/rk_sampler.py/app_started", metadata)
        self.assertIn("sd-webui-tde-sampler/tde_sampler.py/app_started", metadata)
        self.assertIn("prune_stale_sampler_load_targets(demo)", script)


if __name__ == "__main__":
    unittest.main()
