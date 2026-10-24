"""XYZ ``[Modules] Text Encoder`` — the WebUI seam (scripts/modules_xyz_text_encoder.py) against a stub ``modules``.

Registration (appended once, replaced in place on Reload UI), the axis (label, type, cost, choices, confirm, labels),
``apply`` on xyz_grid's shallow cell copies (also with the axis on two of X/Y/Z), the checkpoint the slot check reads,
the always-on hook (the cell's list in the loading parameters and the ``Module N`` list, Forge's settings and the
parent's ``override_settings`` untouched, both put back after the cell), Forge before 2.30-6 (no ``module_map``) and the
empty module list. Forge's own settings / reload / restore code runs in tests/test_xyz_text_encoder_origin.py.
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
from sam3ext import xyz_text_encoder_cell as cell_runtime  # noqa: E402
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


ANIMA_MODULES = ("qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors")
FLUX_MODULES = ("ae.safetensors", "clip_l.safetensors", "t5xxl_fp16.safetensors")


class Forge:
    """The pieces of Forge the script reads when it runs, as ``sys.modules`` entries (``forge.installed()``).

    ``forge_230_6=False``: the shape of Forge 2.29.2 / 2.30-2 - ``main_entry`` has no ``module_map`` and Settings has no
    'Show filenames without folder in the VAE / Text Encoder dropdown' (``sd_modules_dropdown_use_short``); both came
    with 2.30-6 (``ede60e1a``)."""

    def __init__(self, models_path, *, modules=None, data=None, module_list=None, fill_on_refresh=None,
                 checkpoint=None, forge_230_6=True):
        self.module_list, self.module_map = ({}, {}) if module_list is not None else fx.write_models(models_path)
        if module_list is not None:
            self.module_list.update(module_list)
        self.forge_230_6 = forge_230_6
        self.fill_on_refresh = fill_on_refresh
        self.refreshes = 0
        self.ckpt_a = checkpoint or fx.CheckpointInfo("anima.safetensors", "aaaa")
        self.ckpt_b = fx.CheckpointInfo("other.safetensors", "bbbb")
        if data is None:
            data = {"sd_model_checkpoint": self.ckpt_a.title,
                    "forge_additional_modules": self.paths(*(modules or ANIMA_MODULES))}
        defaults = {"forge_additional_modules": []}
        if forge_230_6:
            defaults["sd_modules_dropdown_use_short"] = True
        self.opts = fx.FakeOpts(data, defaults)
        self.loading = {"checkpoint_info": self.ckpt_a, "additional_modules": list(data.get("forge_additional_modules",
                                                                                           [])),
                        "unet_storage_dtype": None}
        self.models_path = str(models_path)
        self.processing = types.ModuleType("modules.processing")
        self.processing._overridden_modules = None
        self.sd_models = types.ModuleType("modules.sd_models")
        self.sd_models.model_data = types.SimpleNamespace(forge_loading_parameters=self.loading)
        self.sd_models.select_checkpoint = lambda: self.ckpt_a
        self.sd_models.checkpoint_aliases = {alias: info for info in (self.ckpt_a, self.ckpt_b) for alias in info.ids}
        self.state = types.SimpleNamespace(time_start=1.0, job="1 out of 2")

    def paths(self, *names):
        return sorted(self.module_list[name] for name in names)

    @property
    def params(self):
        return self.sd_models.model_data.forge_loading_parameters

    @contextlib.contextmanager
    def installed(self):
        forge = self
        modules = types.ModuleType("modules")
        modules.__path__ = []
        shared = types.ModuleType("modules.shared")
        shared.opts = self.opts
        shared.cmd_opts = types.SimpleNamespace(text_encoder_dirs=[])
        shared.state = self.state
        paths = types.ModuleType("modules.paths")
        paths.models_path = self.models_path
        modules.shared, modules.paths, modules.sd_models, modules.processing = (shared, paths, self.sd_models,
                                                                                self.processing)
        modules_forge = types.ModuleType("modules_forge")
        modules_forge.__path__ = []
        main_entry = types.ModuleType("modules_forge.main_entry")
        main_entry.module_list = self.module_list
        if self.forge_230_6:
            main_entry.module_map = self.module_map

        def refresh_models():
            forge.refreshes += 1
            if forge.fill_on_refresh:
                forge.module_list.update(forge.fill_on_refresh)
            return [], sorted(forge.module_list)

        def refresh_model_loading_parameters(*, refresh=True):
            if refresh:
                forge.sd_models.model_data.forge_loading_parameters = dict(
                    checkpoint_info=forge.sd_models.select_checkpoint(),
                    additional_modules=forge.opts.forge_additional_modules, unet_storage_dtype=None)

        main_entry.refresh_models = refresh_models
        main_entry.refresh_model_loading_parameters = refresh_model_loading_parameters
        modules_forge.main_entry = main_entry
        stubs = {"modules": modules, "modules.shared": shared, "modules.paths": paths,
                 "modules.sd_models": self.sd_models, "modules.processing": self.processing,
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


class ScriptCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"
        self.module, self.scripts = load_script()
        self.hook = self.module.XyzTextEncoderHook()
        self.axis = registered_axis(self.module)
        xte.clear_cache()
        cell_runtime._STATE = None

    def tearDown(self):
        cell_runtime._STATE = None
        xte.clear_cache()
        self.tmp.cleanup()

    def cell(self, forge, parent, *values):
        """xyz_grid's cell: a shallow copy, the axis applied once per value (X, Y, Z), the hook's before_process."""
        pc = copy.copy(parent)
        for value in values:
            self.axis.apply(pc, value, list(values))
        self.hook.before_process(pc)
        return pc


class HookTests(ScriptCase):
    def test_always_visible_without_ui(self):
        self.assertIs(self.hook.show(False), self.scripts.AlwaysVisible)
        self.assertIs(self.hook.show(True), self.scripts.AlwaysVisible)
        self.assertIsNone(self.hook.ui(False))
        self.assertEqual(self.hook.title(), "XYZ Text Encoder (sam-extra)")

    def test_no_op_without_the_attribute(self):
        p = P({"x": 1})
        settings = p.override_settings
        with mock.patch.dict(sys.modules, {"modules.sd_models": None, "modules_forge": None, "modules.processing": None}):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.hook.before_process(p)
                self.hook.process(p)
                self.hook.postprocess_image(p, None)
                self.hook.postprocess(p, None)
        self.assertIs(p.override_settings, settings)
        self.assertEqual(settings, {"x": 1})
        self.assertFalse(p.override_settings_restore_afterwards)
        self.assertEqual(stderr.getvalue(), "")

    def test_cell_goes_to_the_loader_not_to_the_settings(self):
        forge = Forge(self.models)
        parent = P({"CLIP_stop_at_last_layers": 2})
        parent_settings = parent.override_settings
        settings_before = copy.deepcopy(forge.opts.data)
        loading_before = forge.params
        xs = ["qwen_3_06b_base_full_fp32", "Use Forge current"]
        with forge.installed():
            self.axis.confirm(parent, xs)
            pc = copy.copy(parent)
            self.axis.apply(pc, xs[0], xs)
            self.assertFalse(hasattr(parent, xte.CELL_ATTR))
            self.hook.before_process(pc)
            expected = forge.paths("qwen_image_vae.safetensors", "qwen_3_06b_base_full_fp32.safetensors")
            self.assertEqual(forge.params["additional_modules"], expected)
            self.assertIs(forge.params["checkpoint_info"], forge.ckpt_a)
            self.assertEqual(forge.processing._overridden_modules, expected)
            # Forge's settings and the grid's override_settings are not touched
            self.assertEqual(forge.opts.data, settings_before)
            self.assertIs(pc.override_settings, parent_settings)
            self.assertEqual(parent_settings, {"CLIP_stop_at_last_layers": 2})
            self.assertFalse(pc.override_settings_restore_afterwards)
            self.hook.process(pc)
            self.hook.postprocess_image(pc, None)
            self.assertEqual(forge.processing._overridden_modules, expected)
            self.hook.postprocess(pc, None)
        self.assertIs(forge.params, loading_before)
        self.assertIsNone(forge.processing._overridden_modules)
        self.assertIsNone(cell_runtime.state())
        self.assertEqual(forge.opts.saved, [])
        # "Use Forge current" and the selection's own file: nothing set
        for value in ("Use Forge current", "None", "qwen_3_06b_base.safetensors"):
            with forge.installed():
                pc = self.cell(forge, parent, value)
            self.assertIs(forge.params, loading_before, value)
            self.assertIsNone(forge.processing._overridden_modules, value)
            self.assertIsNone(cell_runtime.state(), value)

    def test_a_fresh_config_gets_no_key(self):
        forge = Forge(self.models, data={"sd_model_checkpoint": "anima.safetensors [aaaa]"})
        with forge.installed():
            self.cell(forge, P(), "qwen_3_06b_base")
        self.assertEqual(forge.params["additional_modules"], forge.paths("qwen_3_06b_base.safetensors"))
        self.assertNotIn("forge_additional_modules", forge.opts.data)

    def test_confirm_raises_value_error(self):
        forge = Forge(self.models)
        with forge.installed():
            for value, message in (("nope", "unknown file"), ("qwen_image_vae_HDR_fp32", "is a VAE"),
                                   ("qwen35_4b", "Qwen3.5"), ("qwen3vl_4b_bf16", "is a Qwen3-VL 4B encoder"),
                                   ("Automatic", 'use "Use Forge current"')):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                    self.axis.confirm(P(), ["Use Forge current", value])
            self.axis.confirm(P(), ["Use Forge current", "None", "qwen_3_06b_base_full_fp32",
                                    "qwen_3_06b_base_fp16_uncensored"])
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

    def test_gguf_t5_in_the_selection(self):
        forge = Forge(self.models, modules=("ae.safetensors", "clip_l.safetensors", "t5-v1_1-xxl-encoder-Q8_0.gguf"))
        with forge.installed():
            choices = self.axis.choices()
            self.assertIn("t5xxl_fp16.safetensors", choices)
            self.assertIn("t5-v1_1-xxl-encoder-Q8_0.gguf", choices)
            self.axis.confirm(P(), ["Use Forge current", "t5xxl_fp16"])
            self.cell(forge, P(), "t5xxl_fp16")
        self.assertEqual(forge.params["additional_modules"],
                         forge.paths("ae.safetensors", "clip_l.safetensors", "t5xxl_fp16.safetensors"))

    def test_api_selection_is_the_base(self):
        forge = Forge(self.models)
        parent = P({"forge_additional_modules": ["qwen_image_vae_HDR_fp32.safetensors", "qwen_3_06b_base.safetensors"]})
        with forge.installed():
            pc = self.cell(forge, parent, "qwen_3_06b_base_full_fp32")
        self.assertEqual(forge.params["additional_modules"],
                         forge.paths("qwen_image_vae_HDR_fp32.safetensors", "qwen_3_06b_base_full_fp32.safetensors"))
        self.assertEqual(parent.override_settings["forge_additional_modules"],
                         ["qwen_image_vae_HDR_fp32.safetensors", "qwen_3_06b_base.safetensors"])
        self.assertIs(pc.override_settings, parent.override_settings)

    def test_checkpoint_axis_checkpoint_is_kept(self):
        forge = Forge(self.models)
        forge.loading["checkpoint_info"] = forge.ckpt_b     # what xyz_grid's apply_checkpoint left for the cell
        with forge.installed():
            self.cell(forge, P(), "qwen_3_06b_base_full_fp32")
        self.assertIs(forge.params["checkpoint_info"], forge.ckpt_b)
        self.assertNotIn("sd_model_checkpoint", P().override_settings)

    def test_loading_parameters_before_forge_set_them(self):
        # an API-only start may not have built them yet: Forge's own refresh builds them first
        forge = Forge(self.models)
        forge.sd_models.model_data.forge_loading_parameters = {}
        with forge.installed():
            self.cell(forge, P(), "qwen_3_06b_base_full_fp32")
        self.assertIs(forge.params["checkpoint_info"], forge.ckpt_a)
        self.assertEqual(forge.params["additional_modules"],
                         forge.paths("qwen_image_vae.safetensors", "qwen_3_06b_base_full_fp32.safetensors"))

    def test_hook_failure_is_reported_not_raised(self):
        p = P()
        setattr(p, xte.CELL_ATTR, {"value": "x", "modules": ["x"], "slots": frozenset()})
        stderr = io.StringIO()
        with mock.patch.dict(sys.modules, {"modules.sd_models": None}), contextlib.redirect_stderr(stderr):
            self.hook.before_process(p)
        self.assertIn("NOT applied", stderr.getvalue())
        self.assertEqual(p.override_settings, {})
        self.assertIsNone(cell_runtime.state())


class RefreshWrapperTests(ScriptCase):
    """``main_entry.refresh_model_loading_parameters`` wrapped once; it acts only while a cell runs in its task."""

    def test_wrapped_once_and_only_while_a_cell_runs(self):
        forge = Forge(self.models)
        with forge.installed():
            main_entry = sys.modules["modules_forge.main_entry"]
            original = main_entry.refresh_model_loading_parameters
            pc = self.cell(forge, P(), "qwen_3_06b_base_full_fp32")
            wrapped = main_entry.refresh_model_loading_parameters
            self.assertIs(wrapped.__wrapped__, original)
            cell_runtime.install(main_entry)
            self.assertIs(main_entry.refresh_model_loading_parameters, wrapped)
            cell_list = forge.paths("qwen_image_vae.safetensors", "qwen_3_06b_base_full_fp32.safetensors")
            # Forge rebuilds from the settings (the selection the cell started from): the cell's list stands in
            wrapped()
            self.assertEqual(forge.params["additional_modules"], cell_list)
            # "refresh=False" rebuilds nothing; a rebuild from other modules stays as Forge made it
            forge.sd_models.model_data.forge_loading_parameters = {"checkpoint_info": forge.ckpt_a,
                                                                   "additional_modules": list(forge.opts.data[
                                                                       "forge_additional_modules"])}
            wrapped(refresh=False)
            self.assertNotEqual(forge.params["additional_modules"], cell_list)
            forge.opts.data["forge_additional_modules"] = forge.paths("qwen_image_vae_HDR_fp32.safetensors")
            wrapped()
            self.assertEqual(forge.params["additional_modules"], forge.paths("qwen_image_vae_HDR_fp32.safetensors"))
            forge.opts.data["forge_additional_modules"] = forge.paths(*ANIMA_MODULES)
            # outside the task (the job ended) nothing is substituted
            forge.state.job = ""
            wrapped()
            self.assertEqual(forge.params["additional_modules"], forge.paths(*ANIMA_MODULES))
            forge.state.job = "1 out of 2"
            self.hook.postprocess(pc, None)
            self.assertIsNone(cell_runtime.state())
            wrapped()
            self.assertEqual(forge.params["additional_modules"], forge.paths(*ANIMA_MODULES))


class TwoAxesTests(ScriptCase):
    def setUp(self):
        super().setUp()
        self.forge = Forge(self.models, modules=FLUX_MODULES)

    def modules_of(self, *values):
        with self.forge.installed():
            pc = copy.copy(P())
            for value in values:
                self.axis.apply(pc, value, [])
        cell = getattr(pc, xte.CELL_ATTR)
        return None if cell is None else cell["modules"]

    def test_clip_on_one_axis_t5_on_the_other(self):
        f = self.forge
        self.assertEqual(self.modules_of("clip_l_other", "t5xxl_fp8"),
                         f.paths("ae.safetensors", "clip_l_other.safetensors", "t5xxl_fp8.safetensors"))
        # "Use Forge current" on one axis keeps the other's file
        self.assertEqual(self.modules_of("clip_l_other", "Use Forge current"),
                         f.paths("ae.safetensors", "clip_l_other.safetensors", "t5xxl_fp16.safetensors"))
        self.assertEqual(self.modules_of("Use Forge current", "t5xxl_fp8"),
                         f.paths("ae.safetensors", "clip_l.safetensors", "t5xxl_fp8.safetensors"))
        self.assertIsNone(self.modules_of("Use Forge current", "None"))
        with self.forge.installed():
            parent = P()
            self.axis.confirm(parent, ["clip_l", "clip_l_other"])
            self.axis.confirm(parent, ["Use Forge current", "t5xxl_fp8"])

    def test_an_empty_list_gets_one_encoder_per_axis(self):
        forge = Forge(self.models, data={"sd_model_checkpoint": "anima.safetensors [aaaa]",
                                         "forge_additional_modules": []})
        with forge.installed():
            parent = P()
            self.axis.confirm(parent, ["clip_l", "clip_l_other"])
            self.axis.confirm(parent, ["t5xxl_fp16", "t5xxl_fp8"])
            self.cell(forge, parent, "clip_l_other", "t5xxl_fp8")
        self.assertEqual(forge.params["additional_modules"],
                         forge.paths("clip_l_other.safetensors", "t5xxl_fp8.safetensors"))

    def test_the_cell_gets_both(self):
        with self.forge.installed():
            self.cell(self.forge, P(), "clip_l_other", "t5xxl_fp8")
        self.assertEqual(self.forge.params["additional_modules"],
                         self.forge.paths("ae.safetensors", "clip_l_other.safetensors", "t5xxl_fp8.safetensors"))

    def test_one_slot_on_two_axes_is_refused(self):
        with self.forge.installed():
            parent = P()
            self.axis.confirm(parent, ["t5xxl_fp16", "t5xxl_fp8"])
            with self.assertRaisesRegex(ValueError, "on two axes that both pick T5-XXL encoders"):
                self.axis.confirm(parent, ["Use Forge current", "clip_l_other + t5xxl_fp8"])
            # apply refuses it too (a caller that skips confirm)
            pc = copy.copy(P())
            self.axis.apply(pc, "t5xxl_fp8", [])
            with self.assertRaisesRegex(ValueError, "on two axes"):
                self.axis.apply(pc, "t5xxl_fp16", [])


class CellStartTests(ScriptCase):
    """What a cell's list is built on. Forge 2.31's stock VAE and Checkpoint axes write their values into the cell's
    ``override_settings`` - one dict for every cell - so while a cell's axes are applied it may hold the previous
    cell's. ``apply`` checks the values on the grid's start (``confirm`` keeps it on the grid's ``p``); the hook builds
    the cell's list when the cell starts, on what the cell has then."""

    VAE = "qwen_image_vae.safetensors"
    HDR = "qwen_image_vae_HDR_fp32.safetensors"
    FP32 = "qwen_3_06b_base_full_fp32.safetensors"

    def test_apply_never_reads_a_previous_cells_values(self):
        forge = Forge(self.models)
        parent = P()
        with forge.installed():
            self.axis.confirm(parent, [self.FP32])
            self.assertEqual(getattr(parent, xte.GRID_START_ATTR), {"modules": None, "checkpoint": forge.ckpt_a.title})
            # the previous cell's VAE list and checkpoint, left in the shared dict
            parent.override_settings.update({
                "forge_additional_modules": forge.paths(self.HDR, "qwen_3_06b_base.safetensors"),
                "sd_model_checkpoint": "other.safetensors"})
            pc = copy.copy(parent)
            self.axis.apply(pc, self.FP32, [self.FP32])
            cell = getattr(pc, xte.CELL_ATTR)
            self.assertEqual(cell["values"], (self.FP32,))
            self.assertEqual(cell["modules"], forge.paths(self.VAE, self.FP32), "the grid's start: the dropdown's")
            self.assertNotIn(xte.BUILT, cell)
            # the cell's own VAE list, written after the Text Encoder axis was applied: the hook builds on it
            parent.override_settings["forge_additional_modules"] = forge.paths(self.VAE, "qwen_3_06b_base.safetensors")
            self.hook.before_process(pc)
            self.assertTrue(cell[xte.BUILT])
            self.assertEqual(forge.params["additional_modules"], forge.paths(self.VAE, self.FP32))
            # a pass inside the cell (a copy of its p) is not built again
            parent.override_settings["forge_additional_modules"] = forge.paths(self.HDR, "qwen_3_06b_base.safetensors")
            self.hook.before_process(copy.copy(pc))
            self.assertEqual(forge.params["additional_modules"], forge.paths(self.VAE, self.FP32))
            self.hook.postprocess(pc, None)
        self.assertIsNone(cell_runtime.state())

    def test_the_hook_builds_on_the_cells_vae_list(self):
        forge = Forge(self.models)
        parent = P()
        with forge.installed():
            self.axis.confirm(parent, [self.FP32])
            for vae in (self.HDR, self.VAE, self.HDR):
                pc = copy.copy(parent)
                self.axis.apply(pc, self.FP32, [self.FP32])
                parent.override_settings["forge_additional_modules"] = forge.paths(vae, "qwen_3_06b_base.safetensors")
                self.hook.before_process(pc)
                self.assertEqual(forge.params["additional_modules"], forge.paths(vae, self.FP32), vae)
                self.assertEqual(forge.processing._overridden_modules, forge.paths(vae, self.FP32), vae)
                self.hook.postprocess(pc, None)
        self.assertIsNone(cell_runtime.state())

    def test_the_api_callers_start_is_kept_by_confirm(self):
        forge = Forge(self.models)
        parent = P({"forge_additional_modules": [self.HDR, "qwen_3_06b_base.safetensors"]})
        with forge.installed():
            self.axis.confirm(parent, [self.FP32])
            self.axis.confirm(parent, ["Use Forge current"])          # a second Text Encoder axis: the same start
            self.assertEqual(getattr(parent, xte.GRID_START_ATTR),
                             {"modules": [self.HDR, "qwen_3_06b_base.safetensors"], "checkpoint": forge.ckpt_a.title})
            parent.override_settings["forge_additional_modules"] = forge.paths(self.VAE, "qwen_3_06b_base.safetensors")
            pc = copy.copy(parent)
            self.axis.apply(pc, self.FP32, [self.FP32])
        self.assertEqual(getattr(pc, xte.CELL_ATTR)["modules"], forge.paths(self.HDR, self.FP32))

    def test_the_dropdowns_checkpoint_is_kept_by_confirm(self):
        """Without a checkpoint of the caller's - or with one Forge does not know, which process_images drops - confirm
        keeps the Checkpoint dropdown's (its title), read before the first cell: Forge 2.31 may leave a cell's
        checkpoint in the dropdown. A caller's checkpoint Forge knows is kept as it is; a dropdown that cannot be read
        leaves what the caller sent."""
        forge = Forge(self.models)
        with forge.installed():
            for settings in ({}, {"sd_model_checkpoint": "missing.safetensors"}):
                parent = P(dict(settings))
                self.axis.confirm(parent, [self.FP32])
                self.assertEqual(getattr(parent, xte.GRID_START_ATTR),
                                 {"modules": None, "checkpoint": forge.ckpt_a.title}, settings)
                self.assertEqual(parent.override_settings, settings, "the caller's override_settings are not changed")
            parent = P({"sd_model_checkpoint": forge.ckpt_b.name})
            self.axis.confirm(parent, [self.FP32])
            self.assertEqual(getattr(parent, xte.GRID_START_ATTR)["checkpoint"], forge.ckpt_b.name)
            forge.sd_models.select_checkpoint = lambda: None                  # no checkpoint at all
            parent = P()
            self.axis.confirm(parent, [self.FP32])
            self.assertIsNone(getattr(parent, xte.GRID_START_ATTR)["checkpoint"])

            def unreadable():
                raise FileNotFoundError("no checkpoints")

            forge.sd_models.select_checkpoint = unreadable
            parent = P({"sd_model_checkpoint": "missing.safetensors"})
            self.axis.confirm(parent, [self.FP32])
            self.assertEqual(getattr(parent, xte.GRID_START_ATTR)["checkpoint"], "missing.safetensors")

    def test_without_confirm_apply_reads_what_p_has(self):
        forge = Forge(self.models)
        parent = P({"forge_additional_modules": [self.HDR, "qwen_3_06b_base.safetensors"]})
        with forge.installed():
            pc = copy.copy(parent)
            self.axis.apply(pc, self.FP32, [self.FP32])
        self.assertEqual(getattr(pc, xte.CELL_ATTR)["modules"], forge.paths(self.HDR, self.FP32))
        self.assertFalse(hasattr(parent, xte.GRID_START_ATTR))


class CheckpointTests(ScriptCase):
    def forge_with(self, case, modules=()):
        path = fx.write_checkpoint(Path(self.tmp.name) / "Stable-diffusion", case)
        info = fx.CheckpointInfo(path.name, "cccc", str(path))
        data = {"sd_model_checkpoint": info.title,
                "forge_additional_modules": [str(self.models / "text_encoder" / name) for name in modules]}
        forge = Forge(self.models, data=data, checkpoint=info)
        return forge

    def test_sdxl_with_its_own_encoders(self):
        forge = self.forge_with("sdxl")
        with forge.installed():
            self.assertEqual(self.axis.choices(), ["Use Forge current", "clip_l.safetensors", "clip_l_other.safetensors"])
            self.axis.confirm(P(), ["Use Forge current", "clip_l_other"])
            with self.assertRaisesRegex(ValueError, 'checkpoint "sdxl.safetensors" has none built in'):
                self.axis.confirm(P(), ["t5xxl_fp8"])
            self.cell(forge, P(), "clip_l_other")
        self.assertEqual(forge.params["additional_modules"], forge.paths("clip_l_other.safetensors"))

    def test_flux_all_in_one_with_a_t5_override(self):
        forge = self.forge_with("flux_all_in_one", ("t5xxl_fp16.safetensors",))
        with forge.installed():
            self.assertIn("clip_l_other.safetensors", self.axis.choices())
            self.cell(forge, P(), "clip_l_other")
        self.assertEqual(forge.params["additional_modules"],
                         forge.paths("clip_l_other.safetensors", "t5xxl_fp16.safetensors"))

    def test_api_checkpoint_is_the_one_read(self):
        forge = self.forge_with("anima_unet_only", ("qwen_3_06b_base.safetensors",))
        sdxl = fx.write_checkpoint(Path(self.tmp.name) / "Stable-diffusion", "sdxl")
        info = fx.CheckpointInfo(sdxl.name, "dddd", str(sdxl))
        forge.sd_models.checkpoint_aliases.update({alias: info for alias in info.ids})
        with forge.installed():
            with self.assertRaisesRegex(ValueError, "CLIP-L encoder, which Forge does not load"):
                self.axis.confirm(P(), ["clip_l_other"])
            self.axis.confirm(P({"sd_model_checkpoint": info.title, "forge_additional_modules": []}), ["clip_l_other"])

    def test_the_cells_checkpoint_at_the_start_of_the_cell(self):
        """Forge 2.31's Checkpoint axis writes the cell's checkpoint into the shared override_settings - possibly after
        the Text Encoder axis was applied (then it holds the previous cell's). ``apply`` checks against the grid's
        start, so a previous cell's checkpoint never stops the grid; the hook builds the list for the cell's own
        checkpoint, and one that does not take the value gets the list ``apply`` checked (2.30's) and a console line."""
        forge = self.forge_with("flux_unet_only", ("clip_l.safetensors",))
        anima = fx.write_checkpoint(Path(self.tmp.name) / "Stable-diffusion", "anima_unet_only")
        info = fx.CheckpointInfo(anima.name, "eeee", str(anima))
        forge.sd_models.checkpoint_aliases.update({alias: info for alias in info.ids})
        parent = P()
        stderr = io.StringIO()
        with forge.installed(), contextlib.redirect_stderr(stderr):
            self.axis.confirm(parent, ["clip_l_other"])
            parent.override_settings["sd_model_checkpoint"] = info.name        # the previous cell's
            pc = copy.copy(parent)
            self.axis.apply(pc, "clip_l_other", ["clip_l_other"])              # no error
            self.hook.before_process(pc)                                       # the cell's own: Anima, no CLIP-L
            self.assertEqual(forge.params["additional_modules"], forge.paths("clip_l_other.safetensors"))
            self.hook.postprocess(pc, None)
            self.assertEqual(stderr.getvalue().count("[XYZ Text Encoder]"), 1)
            self.assertIn(f'which Forge does not load for the checkpoint "{anima.name}"', stderr.getvalue())
            self.assertIn(f'made as for the checkpoint "{forge.ckpt_a.name}"', stderr.getvalue())
            # the dropdown's checkpoint for the cell: no line
            parent.override_settings["sd_model_checkpoint"] = forge.ckpt_a.name
            pc = copy.copy(parent)
            self.axis.apply(pc, "clip_l_other", ["clip_l_other"])
            self.hook.before_process(pc)
            self.hook.postprocess(pc, None)
        self.assertEqual(stderr.getvalue().count("[XYZ Text Encoder]"), 1)
        self.assertIsNone(cell_runtime.state())

    def test_a_checkpoint_left_in_the_dropdown_is_not_the_grids(self):
        """Forge 2.31 may leave a cell's checkpoint in the Checkpoint dropdown for the next cells (no key in
        config.json - an API-only start - or an API caller's override_settings_restore_afterwards=False). ``apply``
        checks the values against the dropdown's checkpoint as ``confirm`` read it, so a grid mixing families does not
        stop at a later cell; the hook builds the list for the cell's own checkpoint, or ``apply``'s with a console
        line."""
        forge = self.forge_with("flux_unet_only", ("clip_l.safetensors",))
        anima = fx.write_checkpoint(Path(self.tmp.name) / "Stable-diffusion", "anima_unet_only")
        info = fx.CheckpointInfo(anima.name, "eeee", str(anima))
        forge.sd_models.checkpoint_aliases.update({alias: info for alias in info.ids})
        parent = P()
        stderr = io.StringIO()
        with forge.installed(), contextlib.redirect_stderr(stderr):
            self.axis.confirm(parent, ["clip_l_other"])
            self.assertEqual(getattr(parent, xte.GRID_START_ATTR)["checkpoint"], forge.ckpt_a.title)
            forge.sd_models.select_checkpoint = lambda: info     # an Anima cell's checkpoint, left in the dropdown
            for own in (forge.ckpt_a.name, info.name):           # the cell's own (2.31's Checkpoint axis writes it)
                pc = copy.copy(parent)
                self.axis.apply(pc, "clip_l_other", ["clip_l_other"])          # no error
                parent.override_settings["sd_model_checkpoint"] = own
                self.hook.before_process(pc)
                self.assertEqual(forge.params["additional_modules"], forge.paths("clip_l_other.safetensors"), own)
                self.hook.postprocess(pc, None)
        log = stderr.getvalue()
        self.assertEqual(log.count("[XYZ Text Encoder]"), 1, log)                 # the Anima cell
        self.assertIn(f'which Forge does not load for the checkpoint "{anima.name}"', log)
        self.assertIn(f'made as for the checkpoint "{forge.ckpt_a.name}"', log)
        self.assertIsNone(cell_runtime.state())

    def test_unreadable_checkpoint_is_unknown(self):
        forge = Forge(self.models, modules=ANIMA_MODULES)   # its checkpoint file does not exist
        with forge.installed():
            with self.assertRaisesRegex(ValueError, "the axis cannot read the checkpoint"):
                self.axis.confirm(P(), ["qwen3vl_4b_bf16"])


class ForgeVersionTests(ScriptCase):
    """``main_entry.module_map`` and the dropdown's short-name setting exist only from Forge 2.30-6; on 2.29.2 / 2.30-2
    the axis works the same and lists file names, as those versions' dropdown does."""

    FP32 = "qwen_3_06b_base_full_fp32.safetensors"

    def test_without_module_map(self):
        forge = Forge(self.models, forge_230_6=False)
        stderr = io.StringIO()
        with forge.installed(), contextlib.redirect_stderr(stderr):
            self.assertFalse(hasattr(sys.modules["modules_forge.main_entry"], "module_map"))
            with self.assertRaises(AttributeError):
                forge.opts.sd_modules_dropdown_use_short
            self.assertEqual(self.axis.choices(), [
                "Use Forge current", "qwen_3_06b_base.safetensors", "qwen_3_06b_base_fp16_uncensored.safetensors",
                self.FP32])
            self.axis.confirm(P(), ["Use Forge current", "qwen_3_06b_base_full_fp32", self.FP32, "anima/" + self.FP32])
            pc = self.cell(forge, P(), "qwen_3_06b_base_full_fp32")
            self.assertEqual(forge.params["additional_modules"], forge.paths("qwen_image_vae.safetensors", self.FP32))
            self.assertEqual(forge.processing._overridden_modules,
                             forge.paths("qwen_image_vae.safetensors", self.FP32))
            self.assertEqual(self.axis.format_value(pc, self.axis, self.FP32),
                             "Text Encoder: qwen_3_06b_base_full_fp32")
            self.hook.postprocess(pc, None)
        self.assertIsNone(cell_runtime.state())
        self.assertEqual(stderr.getvalue(), "")

    def test_relative_paths_from_module_map_on_2_30_6(self):
        forge = Forge(self.models)
        forge.module_map[self.FP32] = "anima/" + self.FP32
        forge.opts.data["sd_modules_dropdown_use_short"] = False
        with forge.installed():
            choices = self.axis.choices()
            self.assertIn("anima/" + self.FP32, choices)
            self.assertNotIn(self.FP32, choices)
            self.cell(forge, P(), "anima/" + self.FP32)
        self.assertEqual(forge.params["additional_modules"], forge.paths("qwen_image_vae.safetensors", self.FP32))


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
