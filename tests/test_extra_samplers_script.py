"""``scripts/anima_extra_samplers.py`` — registration at import, the accordion, per-request values, infotext
paste fields, XYZ axes and the layout lane.

The script is loaded with stand-ins for ``modules`` (scripts, script_callbacks and the sampler modules
``registry.register`` touches) and for ``k_diffusion.sampling``; no Forge is needed.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import importlib.util
import re
import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path

import gradio as gr


def _fixtures():
    name = "_extra_samplers_fixtures"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fx = _fixtures()
ROOT = fx.ROOT
SCRIPT = ROOT / "scripts" / "anima_extra_samplers.py"

from sam3ext import layout_lanes  # noqa: E402
from sam3ext.extra_samplers import params as sampler_params  # noqa: E402
from sam3ext.extra_samplers import registry  # noqa: E402

_AlwaysVisible = object()


class _Script:
    """``modules.scripts.Script`` stand-in with Forge's ``elem_id`` (modules/scripts.py:343-351)."""

    is_img2img = False

    def show(self, is_img2img):
        return _AlwaysVisible

    def elem_id(self, item_id):
        need_tabname = self.show(True) == self.show(False)
        tabkind = "img2img" if self.is_img2img else "txt2img"
        tabname = f"{tabkind}_" if need_tabname else ""
        title = re.sub(r"[^a-z_0-9]", "", re.sub(r"\s", "_", self.title().lower()))
        return f"script_{tabname}{title}_{item_id}"


_SamplerData = namedtuple("SamplerData", ["name", "constructor", "aliases", "options"])


def _load_script(preset=()):
    """Import the script once with stand-ins; returns (module, sd_samplers stand-in, before_ui callbacks)."""
    before_ui = []
    sd_samplers = types.ModuleType("modules.sd_samplers")
    sd_samplers.all_samplers = list(preset)

    def set_samplers():
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}

    def add_sampler(data):
        if data.name not in [x.name for x in sd_samplers.all_samplers]:
            sd_samplers.all_samplers.append(data)
            set_samplers()

    sd_samplers.add_sampler = add_sampler
    sd_samplers.set_samplers = set_samplers
    common = types.ModuleType("modules.sd_samplers_common")
    common.SamplerData = _SamplerData
    kdiffusion = types.ModuleType("modules.sd_samplers_kdiffusion")
    kdiffusion.KDiffusionSampler = type("KDiffusionSampler", (), {"__init__": lambda self, f, m, o=None: None})
    modules = types.ModuleType("modules")
    modules.__path__ = []
    modules.scripts = types.SimpleNamespace(Script=_Script, AlwaysVisible=_AlwaysVisible, scripts_data=[])
    modules.script_callbacks = types.SimpleNamespace(on_before_ui=before_ui.append)
    modules.sd_samplers, modules.sd_samplers_common, modules.sd_samplers_kdiffusion = sd_samplers, common, kdiffusion
    # Forge's UniPC module (UniPC bh2 is registered only when sample_unipc takes ``variant``)
    samplers_extra = types.ModuleType("modules.sd_samplers_extra")
    samplers_extra.sample_unipc = lambda model, x, sigmas, extra_args=None, callback=None, disable=False, variant="bh1": x
    modules.sd_samplers_extra = samplers_extra
    k_sampling = types.ModuleType("k_diffusion.sampling")
    for spec in registry.SPECS:
        for name in spec.requires:
            setattr(k_sampling, name, object())
    registry._LOGGED.clear()
    # Forge's vendored k_diffusion.deis (DEIS is registered only when get_deis_coeff_list exists)
    deis = types.ModuleType("k_diffusion.deis")
    deis.get_deis_coeff_list = lambda t_steps, max_order, N=10000, deis_mode="tab": []
    with fx.stub_modules({"modules": modules, "modules.sd_samplers_extra": samplers_extra}), \
            fx.installed_k_sampling(k_sampling), fx.installed_k_diffusion_deis(deis):
        spec = importlib.util.spec_from_file_location("_test_extra_samplers_script", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module, sd_samplers, before_ui


MOD, SD_SAMPLERS, BEFORE_UI = _load_script()


class RegistrationAtImportTests(unittest.TestCase):
    def test_import_registers_the_eighteen_samplers(self):
        self.assertEqual([x.name for x in SD_SAMPLERS.all_samplers],
                         ["ER SDE (Reverse-time)", "ER SDE (ODE)", "ER SDE (Tunable)", "DPM++ 4M SDE", "Euler Dy CFG++",
                          "Euler SMEA Dy CFG++", "Euler Dy", "Euler SMEA Dy", "DPM++ 2M SDE Heun", "DPM++ 2M (flow ODE)",
                          "DPM++ 2M Heun (flow ODE)", "DPM++ 3M (flow ODE)", "UniPC bh2", "CFG++ UD10 AB", "IPNDM",
                          "IPNDM_V", "DEIS", "Restart (flow)"])
        self.assertEqual(len(BEFORE_UI), 1)

    def test_the_help_names_every_sampler(self):
        for spec in registry.SPECS:
            self.assertIn(spec.label, MOD._HELP)

    def test_status_names_a_sampler_this_forge_cannot_run(self):
        report = registry.RegistrationReport(skipped_missing={"UniPC bh2": ["modules.sd_samplers_extra:sample_unipc(variant)"]})
        saved = registry._LAST_REPORT
        registry._LAST_REPORT = report
        try:
            status = MOD._status_markdown()
        finally:
            registry._LAST_REPORT = saved
        self.assertIn("`UniPC bh2`", status)
        self.assertIn("필요한 함수가 없어 건너뜀", status)

    def test_status_names_the_labels_another_extension_already_has(self):
        foreign = _SamplerData("ER SDE (Reverse-time)", lambda model: None, [], {})
        module, sd_samplers, _ = _load_script(preset=[foreign])
        self.assertIs(sd_samplers.all_samplers[0], foreign)
        self.assertIn("`ER SDE (Reverse-time)`", module._status_markdown())
        clean, _, _ = _load_script()
        self.assertEqual(clean._status_markdown(), "")


class ScriptFileTests(unittest.TestCase):
    def test_file_name_is_not_another_extensions(self):
        """Forge keys by the script's file name: the module name (``script_loading.load_module``), the
        ui-config keys (``customscript/<file>/…``), the startup timer, error reports, ADetailer's script
        filter (stem) and this extension's layout lane key. Panchovix/sd_forge_neo_extra_samplers already
        ships ``scripts/extra_samplers.py``, so the generic name is not used."""
        self.assertTrue(SCRIPT.is_file())
        self.assertFalse((ROOT / "scripts" / "extra_samplers.py").exists())
        self.assertEqual(layout_lanes.slot_key(str(SCRIPT)), "anima-extra-samplers")
        self.assertEqual(layout_lanes.lane_for(layout_lanes.slot_key(str(SCRIPT))), "anima")
        self.assertLess(SCRIPT.name, "negpip.py")   # NegPiP stays the last script (test_negpip_load_order)


class ScriptUiTests(unittest.TestCase):
    def _ui(self, is_img2img=False):
        script = MOD.ExtraSamplers()
        script.is_img2img = is_img2img
        with gr.Blocks():
            controls = script.ui(is_img2img)
        return script, controls

    def test_identity(self):
        script = MOD.ExtraSamplers()
        self.assertEqual(script.title(), "Extra Samplers")
        self.assertIs(script.show(False), _AlwaysVisible)
        self.assertIs(script.show(True), _AlwaysVisible)
        self.assertEqual(script.sorting_priority, -32)

    def test_the_two_sliders(self):
        script, (max_stage, eta, *window_row) = self._ui()
        self.assertIsInstance(max_stage, gr.Slider)
        self.assertEqual((max_stage.minimum, max_stage.maximum, max_stage.step, max_stage.value), (1, 3, 1, 3))
        self.assertEqual((eta.minimum, eta.maximum, eta.step, eta.value), (0.0, 10.0, 0.01, 1.0))
        self.assertEqual(max_stage.label, "ER SDE max stage")
        self.assertEqual(eta.label, "ER SDE eta")
        self.assertEqual(max_stage.elem_id, "script_txt2img_extra_samplers_er_sde_max_stage")
        self.assertEqual(eta.elem_id, "script_txt2img_extra_samplers_er_sde_eta")
        _, (img_stage, _img_eta, *_img_window) = self._ui(is_img2img=True)
        self.assertEqual(img_stage.elem_id, "script_img2img_extra_samplers_er_sde_max_stage")
        window, start, end = window_row
        self.assertEqual([(c, f) for c, f in script.infotext_fields],
                         [(max_stage, sampler_params.paste_max_stage), (eta, sampler_params.paste_eta),
                          (window, sampler_params.paste_noise_window), (start, sampler_params.paste_noise_start),
                          (end, sampler_params.paste_noise_end)])

    def test_the_noise_window_row(self):
        """Row 2 (v0.33.0): the checkbox and the two percentage sliders, in the order of ``params.ARG_NAMES``."""
        _script, controls = self._ui()
        self.assertEqual(len(controls), len(sampler_params.ARG_NAMES))
        self.assertEqual(sampler_params.ARG_NAMES, ("max_stage", "eta", "noise_window", "noise_start", "noise_end"))
        window, start, end = controls[2:]
        self.assertIsInstance(window, gr.Checkbox)
        self.assertIs(window.value, False)
        self.assertEqual(window.label, "ER SDE noise window")
        for slider, label, value in ((start, "ER SDE noise start", 0.2), (end, "ER SDE noise end", 0.8)):
            with self.subTest(label=label):
                self.assertIsInstance(slider, gr.Slider)
                self.assertEqual((slider.minimum, slider.maximum, slider.step, slider.value), (0.0, 1.0, 0.01, value))
                self.assertEqual(slider.label, label)
        self.assertEqual([c.elem_id for c in controls[2:]],
                         ["script_txt2img_extra_samplers_er_sde_noise_window",
                          "script_txt2img_extra_samplers_er_sde_noise_start",
                          "script_txt2img_extra_samplers_er_sde_noise_end"])
        _, img_controls = self._ui(is_img2img=True)
        self.assertEqual(img_controls[2].elem_id, "script_img2img_extra_samplers_er_sde_noise_window")

    def test_section_follows_the_layout_setting(self):
        script = MOD.ExtraSamplers()
        script.is_img2img = False
        self.assertEqual(script.section, layout_lanes.ANIMA_SECTION)
        script.is_img2img = True
        self.assertIsNone(script.section)

    def test_layout_lane_and_no_on_state(self):
        with gr.Blocks():
            with gr.Group() as group:
                script = MOD.ExtraSamplers()
                controls = script.ui(False)
            fake = types.SimpleNamespace(filename=str(SCRIPT), group=group,
                                         controls=controls, section=layout_lanes.ANIMA_SECTION, create_group=True)
            runner = types.SimpleNamespace(alwayson_scripts=[fake], selectable_scripts=[])
            self.assertEqual(layout_lanes.tag_runner(runner), 1)
        self.assertIn("sam3-slot--anima-extra-samplers", group.elem_classes)
        self.assertIn("sam3-lane--anima", group.elem_classes)
        self.assertNotIn("sam3-on-guess", group.elem_classes)
        for control in controls:
            self.assertNotIn("sam3-on", control.elem_classes or [])


class ProcessTests(unittest.TestCase):
    def setUp(self):
        sampler_params.reset_active()
        self.addCleanup(sampler_params.reset_active)

    def _process(self, *args, xyz=None):
        p = types.SimpleNamespace(extra_generation_params={})
        if xyz is not None:
            setattr(p, sampler_params.XYZ_ATTR, xyz)
        MOD.ExtraSamplers().process(p, *args)
        return getattr(p, sampler_params.P_ATTR)

    def test_values_are_stored_on_the_request(self):
        self.assertEqual(self._process(2, 0.4), sampler_params.ErSdeSettings(2, 0.4))
        self.assertEqual(sampler_params.active(), sampler_params.ErSdeSettings(2, 0.4))

    def test_missing_or_bad_api_arguments_fall_back_to_the_defaults(self):
        self.assertEqual(self._process(), sampler_params.ErSdeSettings())
        self.assertEqual(self._process("x", None), sampler_params.ErSdeSettings())
        self.assertEqual(self._process(7, -3), sampler_params.ErSdeSettings(3, 0.0))
        self.assertEqual(self._process(1.6, float("nan")), sampler_params.ErSdeSettings(2, 1.0))

    def test_xyz_values_override_the_accordion(self):
        self.assertEqual(self._process(3, 1.0, xyz={"max_stage": 1}), sampler_params.ErSdeSettings(1, 1.0))
        self.assertEqual(self._process(3, 1.0, xyz={"eta": "0.25"}), sampler_params.ErSdeSettings(3, 0.25))
        self.assertEqual(self._process(3, 1.0, False, 0.2, 0.8, xyz={"noise_window": (True, 0.1, 0.9)}),
                         sampler_params.ErSdeSettings(3, 1.0, True, 0.1, 0.9))
        self.assertEqual(self._process(3, 1.0, True, 0.3, 0.6, xyz={"noise_window": (False, 0.2, 0.8)}),
                         sampler_params.ErSdeSettings(3, 1.0, False, 0.2, 0.8))

    def test_argument_lists_of_every_length(self):
        """API ``args`` of 0, 2 (every v0.31/v0.32 request), 3, 5 and 7 entries: the missing ones default,
        the window stays off unless asked for, surplus entries are ignored."""
        S = sampler_params.ErSdeSettings
        for args, expected in [
            ((), S(3, 1.0, False, 0.2, 0.8)),
            ((2, 0.4), S(2, 0.4, False, 0.2, 0.8)),
            ((2, 0.4, True), S(2, 0.4, True, 0.2, 0.8)),
            ((2, 0.4, True, 0.1, 0.9), S(2, 0.4, True, 0.1, 0.9)),
            ((2, 0.4, True, 0.1, 0.9, "extra", 42), S(2, 0.4, True, 0.1, 0.9)),
        ]:
            with self.subTest(args=args):
                self.assertEqual(self._process(*args), expected)
                self.assertEqual(sampler_params.settings_from_args(list(args)), expected)
        self.assertEqual(sampler_params.ErSdeSettings(2, 0.4).as_dict(),
                         {"max_stage": 2, "eta": 0.4, "noise_window": False, "noise_start": 0.2, "noise_end": 0.8})

    def test_noise_window_coercion(self):
        for value, expected in [
            (True, True), (1, True), (1.0, True), ("true", True), ("ON", True), (" Yes ", True), ("1", True),
            (False, False), (0, False), (2, False), (0.5, False), ("false", False), ("off", False), ("", False),
            (None, False), ([True], False), ("2", False),
        ]:
            with self.subTest(value=value):
                self.assertIs(sampler_params.coerce_noise_window(value), expected)

    def test_the_window_argument_of_a_request_goes_through_that_coercion(self):
        """An API caller's ``"false"`` / ``"off"`` / ``2`` in the third slot leaves the window off (``bool()``
        would turn every non-empty text and every non-zero number on); the script and ``settings_from_args``
        read the slot the same way."""
        S = sampler_params.ErSdeSettings
        for value, enabled in [("false", False), ("off", False), (2, False), ("0", False), ("on", True), ("TRUE", True),
                               (1, True)]:
            with self.subTest(value=value):
                expected = S(3, 1.0, enabled, 0.2, 0.8)
                self.assertEqual(sampler_params.settings_from_args([3, 1.0, value]), expected)
                self.assertEqual(self._process(3, 1.0, value), expected)

    def test_noise_percent_coercion(self):
        for value, expected in [
            (0.3, 0.3), ("0.45", 0.45), (0, 0.0), (1, 1.0), (-0.5, 0.0), (1.7, 1.0),
            (float("nan"), 0.25), (float("inf"), 0.25), (None, 0.25), ("x", 0.25), (True, 0.25), (False, 0.25),
        ]:
            with self.subTest(value=value):
                self.assertEqual(sampler_params.coerce_noise_percent(value, 0.25), expected)
        # start >= end is kept as given (an empty window), never swapped
        self.assertEqual(self._process(3, 1.0, True, 0.9, 0.1), sampler_params.ErSdeSettings(3, 1.0, True, 0.9, 0.1))
        self.assertEqual(self._process(3, 1.0, "on", "bad", None), sampler_params.ErSdeSettings(3, 1.0, True, 0.2, 0.8))

    def test_an_api_dict_on_the_request_is_read_with_the_same_rules(self):
        p = types.SimpleNamespace()
        setattr(p, sampler_params.P_ATTR, {"max_stage": 2, "eta": "0.5", "noise_window": "yes", "noise_start": 0.1})
        self.assertEqual(sampler_params.resolve(p), sampler_params.ErSdeSettings(2, 0.5, True, 0.1, 0.8))
        setattr(p, sampler_params.P_ATTR, {"noise_window": 2, "noise_end": 7})
        self.assertEqual(sampler_params.resolve(p), sampler_params.ErSdeSettings(3, 1.0, False, 0.2, 1.0))


class XyzTests(unittest.TestCase):
    def _xyz_module(self):
        class AxisOption:
            def __init__(self, label, type, apply, format_value=None, confirm=None, cost=0.0, choices=None, prepare=None):
                self.label, self.type, self.apply, self.choices = label, type, apply, choices
                self.confirm = confirm

        module = types.SimpleNamespace(AxisOption=AxisOption, axis_options=[AxisOption("Nothing", str, None)])
        entry = types.SimpleNamespace(script_class=type("Script", (), {"__module__": "xyz_grid.py"}), module=module)
        return module, entry

    def test_axes_are_added_once(self):
        module, entry = self._xyz_module()
        MOD.scripts.scripts_data.append(entry)
        self.addCleanup(MOD.scripts.scripts_data.remove, entry)
        BEFORE_UI[0]()
        BEFORE_UI[0]()
        labels = [axis.label for axis in module.axis_options]
        self.assertEqual(labels, ["Nothing", "[Extra Samplers] ER SDE max stage", "[Extra Samplers] ER SDE eta",
                                  "[Extra Samplers] ER SDE noise window"])
        stage, eta, _window = module.axis_options[1:]
        self.assertIs(stage.type, int)
        self.assertEqual(stage.choices(), ["1", "2", "3"])
        self.assertIs(eta.type, float)
        self.assertIsNone(eta.choices)

        p = types.SimpleNamespace()
        stage.apply(p, 2, ["1", "2"])
        eta.apply(p, 0.75, [0.5, 0.75])
        self.assertEqual(getattr(p, sampler_params.XYZ_ATTR), {"max_stage": 2, "eta": 0.75})
        MOD.ExtraSamplers().process(p, 3, 1.0)
        self.assertEqual(getattr(p, sampler_params.P_ATTR), sampler_params.ErSdeSettings(2, 0.75))

    def test_the_noise_window_axis(self):
        module, entry = self._xyz_module()
        MOD.scripts.scripts_data.append(entry)
        self.addCleanup(MOD.scripts.scripts_data.remove, entry)
        BEFORE_UI[0]()
        axis = module.axis_options[-1]
        self.assertEqual(axis.label, "[Extra Samplers] ER SDE noise window")
        self.assertIs(axis.type, str)
        self.assertIsNone(axis.choices)
        axis.confirm(None, ["off", "0.1-0.9", " 0.2 - 0.8 ", "0.9-0.1", "0-1", "OFF"])   # all accepted
        for bad in ("on", "0.1", "0.2-1.5", "-0.1-0.5", "0.2-0.8-0.9", "a-b", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                axis.confirm(None, ["off", bad])
        for value, expected in [("0.1-0.9", (True, 0.1, 0.9)), ("off", (False, 0.2, 0.8)), ("0.7-0.3", (True, 0.7, 0.3))]:
            with self.subTest(value=value):
                p = types.SimpleNamespace()
                axis.apply(p, value, [value])
                self.assertEqual(getattr(p, sampler_params.XYZ_ATTR), {"noise_window": expected})
                MOD.ExtraSamplers().process(p, 3, 1.0, not expected[0], 0.5, 0.5)
                self.assertEqual(getattr(p, sampler_params.P_ATTR), sampler_params.ErSdeSettings(3, 1.0, *expected))

    def test_no_xyz_script_is_fine(self):
        BEFORE_UI[0]()   # scripts_data has no xyz_grid: nothing to do, nothing raised


class InfotextTests(unittest.TestCase):
    def test_written_keys_paste_back(self):
        for label, settings, expected in [
            ("ER SDE (Reverse-time)", sampler_params.ErSdeSettings(2, 0.35), {"ER SDE max stage": 2, "ER SDE eta": 0.35}),
            ("ER SDE (Reverse-time)", sampler_params.ErSdeSettings(3, 1.0), {}),
            ("ER SDE (ODE)", sampler_params.ErSdeSettings(1, 0.35), {"ER SDE max stage": 1}),
            ("Euler Dy CFG++", sampler_params.ErSdeSettings(1, 0.35), {}),
        ]:
            with self.subTest(label=label, settings=settings):
                p = types.SimpleNamespace(extra_generation_params={})
                sampler_params.record_infotext(p, label, settings)
                self.assertEqual(p.extra_generation_params, expected)
                pasted = {key: str(value) for key, value in expected.items()} | {"Sampler": label}
                stage = sampler_params.paste_max_stage(pasted)
                eta = sampler_params.paste_eta(pasted)
                self.assertEqual(stage, settings.max_stage if "ER SDE max stage" in expected else 3)
                self.assertEqual(eta, settings.eta if "ER SDE eta" in expected else 1.0)

    def test_eta_from_neo_extra_schedulers_infotexts(self):
        # aoleg/Neo_ExtraSchedulers takes η from Forge's global Eta (infotext "Eta").
        self.assertEqual(sampler_params.paste_eta({"Sampler": "ER SDE (Reverse-time)", "Eta": "0.5"}), 0.5)
        self.assertEqual(sampler_params.paste_eta({"Sampler": "Euler a", "Hires sampler": "ER SDE (Reverse-time)",
                                                   "Eta": "0.7"}), 0.7)
        self.assertEqual(sampler_params.paste_eta({"Sampler": "Euler a", "Eta": "0.5"}), 1.0)
        self.assertEqual(sampler_params.paste_eta({"Sampler": "ER SDE (Reverse-time)", "Eta": "0.5",
                                                   "ER SDE eta": "2"}), 2.0)
        self.assertEqual(sampler_params.paste_eta({"Sampler": "ER SDE (Reverse-time) Simple", "Eta": "0.4"}), 0.4)

    def test_our_eta_survives_forges_eta_in_the_same_infotext(self):
        """A Reverse-time pass at η 1 next to a pass that writes Forge's ``Eta`` (Euler a hires, Forge η
        0.6): without ``ER SDE eta`` the paste rule for aoleg's infotexts would restore 0.6."""
        for forge_eta, before, expected in [
            (0.6, {}, {"ER SDE eta": 1.0}),
            ("0.6", {}, {"ER SDE eta": 1.0}),
            (1.0, {"Eta": 0.6}, {"Eta": 0.6, "ER SDE eta": 1.0}),   # an earlier pass wrote it already
            (1.0, {}, {}),
            (None, {}, {}),
            (float("nan"), {}, {}),
        ]:
            with self.subTest(forge_eta=forge_eta, before=before):
                p = types.SimpleNamespace(extra_generation_params=dict(before))
                sampler_params.record_infotext(p, "ER SDE (Reverse-time)", sampler_params.ErSdeSettings(),
                                               forge_eta=forge_eta)
                self.assertEqual(p.extra_generation_params, expected)
                pasted = {key: str(value) for key, value in expected.items()}
                pasted.update({"Sampler": "ER SDE (Reverse-time)", "Hires sampler": "Euler a", "Eta": "0.6"})
                self.assertEqual(sampler_params.paste_eta(pasted), 1.0 if expected else 0.6)
        p = types.SimpleNamespace(extra_generation_params={})
        sampler_params.record_infotext(p, "ER SDE (ODE)", sampler_params.ErSdeSettings(), forge_eta=0.6)
        self.assertEqual(p.extra_generation_params, {})

    def test_paste_without_keys_resets_to_the_defaults(self):
        self.assertEqual(sampler_params.paste_max_stage({}), 3)
        self.assertEqual(sampler_params.paste_eta({}), 1.0)
        self.assertEqual(sampler_params.paste_max_stage({"ER SDE max stage": "9"}), 3)
        self.assertEqual(sampler_params.paste_eta({"ER SDE eta": "abc"}), 1.0)

    def test_xyz_float_noise_is_rounded_only_in_the_text(self):
        p = types.SimpleNamespace(extra_generation_params={})
        sampler_params.record_infotext(p, "ER SDE (Reverse-time)", sampler_params.ErSdeSettings(3, 0.1 + 0.2))
        self.assertEqual(p.extra_generation_params, {"ER SDE eta": 0.3})

    def test_tunable_writes_stage_eta_and_window_like_reverse_time(self):
        S = sampler_params.ErSdeSettings
        for label, settings, kwargs, expected in [
            ("ER SDE (Tunable)", S(2, 0.35), {}, {"ER SDE max stage": 2, "ER SDE eta": 0.35}),
            ("ER SDE (Tunable)", S(), {}, {}),
            ("ER SDE (Tunable)", S(), {"forge_eta": 0.6}, {"ER SDE eta": 1.0}),
            ("ER SDE (Tunable)", S(3, 1.0, True, 0.2, 0.8), {}, {"ER SDE noise window": "0.2-0.8"}),
            ("ER SDE (Reverse-time)", S(3, 2.0, True, 0.1, 0.9), {},
             {"ER SDE eta": 2.0, "ER SDE noise window": "0.1-0.9"}),
            ("ER SDE (Reverse-time)", S(3, 1.0, True, 0.9, 0.1), {}, {"ER SDE noise window": "0.9-0.1"}),   # empty
            ("ER SDE (Tunable)", S(3, 0.0, True, 0.2, 0.8), {}, {"ER SDE eta": 0.0}),   # η 0 = the ODE: no window
            ("ER SDE (Tunable)", S(3, 1.0, True, 0.2, 0.8), {"window_skipped": True}, {}),
            ("ER SDE (ODE)", S(1, 2.0, True, 0.2, 0.8), {}, {"ER SDE max stage": 1}),
            ("Euler Dy", S(1, 2.0, True, 0.2, 0.8), {}, {}),
        ]:
            with self.subTest(label=label, settings=settings, kwargs=kwargs):
                p = types.SimpleNamespace(extra_generation_params={})
                sampler_params.record_infotext(p, label, settings, **kwargs)
                self.assertEqual(p.extra_generation_params, expected)

    def test_the_window_pastes_back(self):
        for start, end in [(0.2, 0.8), (0.1, 0.9), (0.0, 1.0), (0.9, 0.1), (0.123456789, 0.5), (1e-06, 0.75)]:
            with self.subTest(start=start, end=end):
                p = types.SimpleNamespace(extra_generation_params={})
                settings = sampler_params.ErSdeSettings(3, 1.0, True, start, end)
                sampler_params.record_infotext(p, "ER SDE (Tunable)", settings)
                pasted = {key: str(value) for key, value in p.extra_generation_params.items()}
                self.assertIs(sampler_params.paste_noise_window(pasted), True)
                self.assertEqual(sampler_params.paste_noise_start(pasted), round(start, 6))
                self.assertEqual(sampler_params.paste_noise_end(pasted), round(end, 6))
        self.assertEqual(sampler_params.format_noise_window(0.2, 0.8), "0.2-0.8")
        self.assertEqual(sampler_params.format_noise_window(0.1 + 0.2, 1), "0.3-1")

    def test_window_paste_table(self):
        for params, expected in [
            ({}, (False, 0.2, 0.8)),
            ({"ER SDE noise window": "0.3-0.7"}, (True, 0.3, 0.7)),
            ({"ER SDE noise window": " 0.3 - 0.7 "}, (True, 0.3, 0.7)),
            ({"ER SDE noise window": "off"}, (False, 0.2, 0.8)),
            ({"ER SDE noise window": "0.3-1.7"}, (False, 0.2, 0.8)),
            ({"ER SDE noise window": "abc"}, (False, 0.2, 0.8)),
            ({"ER SDE noise window": "0.3"}, (False, 0.2, 0.8)),
            ({"ER SDE noise window": "1e-06-0.8"}, (True, 1e-06, 0.8)),
        ]:
            with self.subTest(params=params):
                got = (sampler_params.paste_noise_window(params), sampler_params.paste_noise_start(params),
                       sampler_params.paste_noise_end(params))
                self.assertEqual(got, expected)
        self.assertIsNone(sampler_params.parse_noise_window(None))
        self.assertEqual(sampler_params.parse_noise_window("OFF"), (False, 0.2, 0.8))

    def test_eta_paste_fallback_stays_reverse_time_only(self):
        """aoleg's infotexts carry Forge's ``Eta`` for Reverse-time; Tunable is this extension's own entry."""
        self.assertEqual(sampler_params.paste_eta({"Sampler": "ER SDE (Tunable)", "Eta": "0.5"}), 1.0)
        self.assertEqual(sampler_params.paste_eta({"Sampler": "ER SDE (Tunable)", "ER SDE eta": "0.5"}), 0.5)


if __name__ == "__main__":
    unittest.main()
