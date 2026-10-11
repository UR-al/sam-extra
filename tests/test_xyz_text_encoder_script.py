"""XYZ ``[Modules] Text Encoder`` — the WebUI seam (scripts/modules_xyz_text_encoder.py) against a stub ``modules``.

Registration (appended once, replaced in place on Reload UI), the axis (label, type, cost, choices, confirm, labels),
``apply`` on xyz_grid's shallow cell copies, the always-on hook's ``before_process`` (a new ``override_settings`` per
cell, the restore flag, the key Forge's restore needs in ``opts.data``, a Checkpoint axis's checkpoint) and the empty
module list. Forge's own settings / reload / restore code runs in tests/test_xyz_text_encoder_origin.py.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import xyz_text_encoder as xte  # noqa: E402
from tests import _xyz_text_encoder_support as fx  # noqa: E402

SCRIPT = ROOT / "scripts" / "modules_xyz_text_encoder.py"


class AxisOption:
    """scripts/xyz_grid.py ``AxisOption`` (same signature)."""

    def __init__(self, label, type, apply, format_value=None, confirm=None, cost=0.0, choices=None, prepare=None):
        self.label, self.type, self.apply = label, type, apply
        self.format_value, self.confirm, self.cost = format_value, confirm, cost
        self.choices, self.prepare = choices, prepare


def new_xyz():
    return types.SimpleNamespace(AxisOption=AxisOption,
                                 axis_options=[AxisOption("Nothing", str, None), AxisOption("VAE", str, None)])


def load_script(registered=None, xyz=None):
    registered = registered if registered is not None else {}
    modules = types.ModuleType("modules")
    modules.__path__ = []

    class Script:
        pass

    data = []
    if xyz is not None:
        data.append(types.SimpleNamespace(script_class=type("XYZ", (), {"__module__": "xyz_grid.py"}), module=xyz))
    scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=data)
    callbacks = types.SimpleNamespace(on_before_ui=lambda fn: registered.setdefault("before_ui", []).append(fn))
    with mock.patch.dict(sys.modules, {"modules": modules, "modules.scripts": scripts,
                                       "modules.script_callbacks": callbacks}):
        spec = importlib.util.spec_from_file_location("_test_modules_xyz_text_encoder", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, scripts


class Forge:
    """The pieces of Forge the script reads when it runs, as ``sys.modules`` entries (``forge.installed()``)."""

    def __init__(self, models_path, *, modules=None, data=None, module_list=None, fill_on_refresh=None):
        self.module_list, self.module_map = ({}, {}) if module_list is not None else fx.write_models(models_path)
        if module_list is not None:
            self.module_list.update(module_list)
        self.fill_on_refresh = fill_on_refresh
        self.refreshes = 0
        self.ckpt_a = fx.CheckpointInfo("anima.safetensors", "aaaa")
        self.ckpt_b = fx.CheckpointInfo("other.safetensors", "bbbb")
        if data is None:
            data = {"sd_model_checkpoint": self.ckpt_a.title,
                    "forge_additional_modules": self.paths(*(modules or ("qwen_image_vae.safetensors",
                                                                         "qwen_3_06b_base.safetensors")))}
        self.opts = fx.FakeOpts(data, {"forge_additional_modules": [], "sd_modules_dropdown_use_short": True})
        self.loading = {"checkpoint_info": self.ckpt_a}
        self.models_path = str(models_path)

    def paths(self, *names):
        return sorted(self.module_list[name] for name in names)

    @contextlib.contextmanager
    def installed(self):
        forge = self
        modules = types.ModuleType("modules")
        modules.__path__ = []
        shared = types.ModuleType("modules.shared")
        shared.opts = self.opts
        shared.cmd_opts = types.SimpleNamespace(text_encoder_dirs=[])
        paths = types.ModuleType("modules.paths")
        paths.models_path = self.models_path
        sd_models = types.ModuleType("modules.sd_models")
        sd_models.model_data = types.SimpleNamespace(forge_loading_parameters=self.loading)
        sd_models.select_checkpoint = lambda: forge.ckpt_a
        modules.shared, modules.paths, modules.sd_models = shared, paths, sd_models
        modules_forge = types.ModuleType("modules_forge")
        modules_forge.__path__ = []
        main_entry = types.ModuleType("modules_forge.main_entry")
        main_entry.module_list, main_entry.module_map = self.module_list, self.module_map

        def refresh_models():
            forge.refreshes += 1
            if forge.fill_on_refresh:
                forge.module_list.update(forge.fill_on_refresh)
            return [], sorted(forge.module_list)

        main_entry.refresh_models = refresh_models
        modules_forge.main_entry = main_entry
        stubs = {"modules": modules, "modules.shared": shared, "modules.paths": paths, "modules.sd_models": sd_models,
                 "modules_forge": modules_forge, "modules_forge.main_entry": main_entry}
        with mock.patch.dict(sys.modules, stubs):
            yield


class P:
    """A processing object: xyz_grid copies it per cell (``copy(p)``)."""

    def __init__(self, override_settings=None):
        self.override_settings = {} if override_settings is None else override_settings
        self.override_settings_restore_afterwards = False
        self.styles = []


def registered_axis(module):
    xyz = new_xyz()
    module.register_xyz_axes(xyz)
    return xyz.axis_options[-1]


class RegistrationTests(unittest.TestCase):
    def test_appended_once_and_replaced_in_place_on_reload(self):
        xyz = new_xyz()
        registered = {}
        module, _ = load_script(registered, xyz)
        self.assertEqual(len(registered["before_ui"]), 1)
        registered["before_ui"][0]()
        labels = [option.label for option in xyz.axis_options]
        self.assertEqual(labels, ["Nothing", "VAE", "[Modules] Text Encoder"])
        xyz.axis_options.append(AxisOption("[Other] Later", str, None))
        registered["before_ui"][0]()
        self.assertEqual([option.label for option in xyz.axis_options],
                         ["Nothing", "VAE", "[Modules] Text Encoder", "[Other] Later"])
        # Reload UI: the file runs again; the kept XYZ grid gets the new module's axis at the same index
        again = {}
        reloaded, _ = load_script(again, xyz)
        again["before_ui"][0]()
        self.assertEqual([option.label for option in xyz.axis_options],
                         ["Nothing", "VAE", "[Modules] Text Encoder", "[Other] Later"])
        self.assertIs(xyz.axis_options[2].apply, reloaded._apply)
        self.assertIsNot(reloaded._apply, module._apply)

    def test_no_xyz_grid(self):
        registered = {}
        load_script(registered, None)
        registered["before_ui"][0]()   # nothing to register into, no error

    def test_axis(self):
        module, _ = load_script()
        axis = registered_axis(module)
        self.assertEqual(axis.label, "[Modules] Text Encoder")
        self.assertIs(axis.type, str)
        self.assertEqual(axis.cost, 0.9)
        self.assertIs(axis.apply, module._apply)
        self.assertIs(axis.confirm, module._confirm)
        self.assertIs(axis.format_value, module._format)
        self.assertIs(axis.choices, module._choices)
        self.assertFalse(hasattr(axis, "is_img2img"), "a plain AxisOption: both tabs")

    def test_file_sorts_before_negpip(self):
        self.assertLess(SCRIPT.name, "negpip.py")


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"
        self.module, self.scripts = load_script()
        self.hook = self.module.XyzTextEncoderHook()
        self.axis = registered_axis(self.module)
        xte.clear_cache()

    def tearDown(self):
        xte.clear_cache()
        self.tmp.cleanup()

    def test_always_visible_without_ui(self):
        self.assertIs(self.hook.show(False), self.scripts.AlwaysVisible)
        self.assertIs(self.hook.show(True), self.scripts.AlwaysVisible)
        self.assertIsNone(self.hook.ui(False))
        self.assertEqual(self.hook.title(), "XYZ Text Encoder (sam-extra)")

    def test_no_op_without_the_attribute(self):
        p = P({"x": 1})
        settings = p.override_settings
        with mock.patch.dict(sys.modules, {"modules.sd_models": None, "modules_forge": None}):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.hook.before_process(p)
        self.assertIs(p.override_settings, settings)
        self.assertEqual(settings, {"x": 1})
        self.assertFalse(p.override_settings_restore_afterwards)
        self.assertEqual(stderr.getvalue(), "")

    def test_cell_on_a_shallow_copy(self):
        forge = Forge(self.models)
        parent = P({"CLIP_stop_at_last_layers": 2})
        parent_settings = parent.override_settings
        xs = ["qwen_3_06b_base_full_fp32", "Use Forge current"]
        with forge.installed():
            self.axis.confirm(parent, xs)
            pc = copy.copy(parent)
            self.axis.apply(pc, xs[0], xs)
            self.assertFalse(hasattr(parent, xte.CELL_ATTR))
            self.hook.before_process(pc)
        expected = forge.paths("qwen_image_vae.safetensors", "qwen_3_06b_base_full_fp32.safetensors")
        self.assertEqual(pc.override_settings, {"CLIP_stop_at_last_layers": 2, "forge_additional_modules": expected})
        self.assertTrue(pc.override_settings_restore_afterwards)
        self.assertIs(parent.override_settings, parent_settings)
        self.assertEqual(parent_settings, {"CLIP_stop_at_last_layers": 2})
        self.assertFalse(parent.override_settings_restore_afterwards)
        self.assertEqual(forge.opts.saved, [])
        # "Use Forge current" and the selection's own file: nothing set
        for value in ("Use Forge current", "qwen_3_06b_base.safetensors"):
            with forge.installed():
                pc = copy.copy(parent)
                self.axis.apply(pc, value, xs)
                self.hook.before_process(pc)
            self.assertIs(pc.override_settings, parent_settings, value)
            self.assertFalse(pc.override_settings_restore_afterwards, value)

    def test_confirm_raises_value_error(self):
        forge = Forge(self.models)
        with forge.installed():
            for value, message in (("nope", "unknown file"), ("qwen_image_vae_HDR_fp32", "is a VAE"),
                                   ("qwen35_4b", "Qwen3.5"), ("qwen3vl_4b_bf16", "is a Qwen3-VL 4B encoder")):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                    self.axis.confirm(P(), ["Use Forge current", value])
            self.axis.confirm(P(), ["Use Forge current", "qwen_3_06b_base_full_fp32", "qwen_3_06b_base_fp16_uncensored"])
            with self.assertRaises(ValueError):
                self.axis.apply(P(), "nope", ["nope"])

    def test_choices_and_labels(self):
        forge = Forge(self.models)
        with forge.installed():
            self.assertEqual(self.axis.choices(), [
                "Use Forge current", "qwen_3_06b_base.safetensors", "qwen_3_06b_base_fp16_uncensored.safetensors",
                "qwen_3_06b_base_full_fp32.safetensors"])
            p = P()
            self.assertEqual(self.axis.format_value(p, self.axis, "qwen_3_06b_base_full_fp32.safetensors"),
                             "Text Encoder: qwen_3_06b_base_full_fp32")
            self.assertEqual(self.axis.format_value(p, self.axis, "Use Forge current"),
                             "Text Encoder: current (qwen_3_06b_base)")

    def test_api_selection_is_the_base(self):
        forge = Forge(self.models)
        parent = P({"forge_additional_modules": ["qwen_image_vae_HDR_fp32.safetensors", "qwen_3_06b_base.safetensors"]})
        with forge.installed():
            pc = copy.copy(parent)
            self.axis.apply(pc, "qwen_3_06b_base_full_fp32", [])
            self.hook.before_process(pc)
        self.assertEqual(pc.override_settings["forge_additional_modules"],
                         forge.paths("qwen_image_vae_HDR_fp32.safetensors", "qwen_3_06b_base_full_fp32.safetensors"))
        self.assertEqual(parent.override_settings["forge_additional_modules"],
                         ["qwen_image_vae_HDR_fp32.safetensors", "qwen_3_06b_base.safetensors"])

    def test_fresh_config_gets_the_key_forge_restores(self):
        forge = Forge(self.models, data={"sd_model_checkpoint": "anima.safetensors [aaaa]"})
        with forge.installed():
            pc = copy.copy(P())
            self.axis.apply(pc, "qwen_3_06b_base", [])
            self.hook.before_process(pc)
        self.assertEqual(pc.override_settings["forge_additional_modules"], forge.paths("qwen_3_06b_base.safetensors"))
        self.assertIn("forge_additional_modules", forge.opts.data)
        self.assertEqual(forge.opts.data["forge_additional_modules"], [])

    def test_checkpoint_axis_checkpoint_is_carried(self):
        forge = Forge(self.models)
        forge.loading["checkpoint_info"] = forge.ckpt_b
        with forge.installed():
            pc = copy.copy(P())
            self.axis.apply(pc, "qwen_3_06b_base_full_fp32", [])
            self.hook.before_process(pc)
        self.assertEqual(pc.override_settings["sd_model_checkpoint"], "other.safetensors")
        # an API caller's checkpoint wins
        with forge.installed():
            pc = copy.copy(P({"sd_model_checkpoint": "api.safetensors"}))
            self.axis.apply(pc, "qwen_3_06b_base_full_fp32", [])
            self.hook.before_process(pc)
        self.assertEqual(pc.override_settings["sd_model_checkpoint"], "api.safetensors")

    def test_hook_failure_is_reported_not_raised(self):
        p = P()
        setattr(p, xte.CELL_ATTR, {"value": "x", "modules": ["x"]})
        stderr = io.StringIO()
        with mock.patch.dict(sys.modules, {"modules.sd_models": None}), contextlib.redirect_stderr(stderr):
            self.hook.before_process(p)
        self.assertIn("NOT applied", stderr.getvalue())
        self.assertEqual(p.override_settings, {})


class EmptyListTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.module, _ = load_script()
        self.axis = registered_axis(self.module)

    def tearDown(self):
        self.tmp.cleanup()

    def test_refreshed_once_then_an_error(self):
        forge = Forge(self.tmp.name, module_list={}, data={})
        with forge.installed(), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "VAE / Text Encoder list is empty"):
                self.axis.confirm(P(), ["qwen_3_06b_base"])
            self.assertEqual(forge.refreshes, 1)
            self.assertEqual(self.axis.choices(), ["Use Forge current"])
            self.assertEqual(forge.refreshes, 2)

    def test_refresh_fills_the_list(self):
        listing, _ = fx.write_models(Path(self.tmp.name) / "models")
        forge = Forge(self.tmp.name, module_list={}, fill_on_refresh=listing,
                      data={"forge_additional_modules": [listing["qwen_3_06b_base.safetensors"]]})
        with forge.installed():
            self.axis.confirm(P(), ["qwen_3_06b_base_full_fp32"])
            self.axis.confirm(P(), ["qwen_3_06b_base_full_fp32"])
        self.assertEqual(forge.refreshes, 1)


if __name__ == "__main__":
    unittest.main()
