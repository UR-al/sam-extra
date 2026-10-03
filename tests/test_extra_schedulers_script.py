"""Extra Schedulers — the always-on script: accordion, per-generation state, infotext, paste, XYZ.

The script is loaded against a stub ``modules`` package (no WebUI). The infotext round trip uses
Forge's real ``quote``/``unquote`` and parameter regex (cut out of modules/infotext_utils.py) when a
Forge checkout is found.
"""
from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import _extra_schedulers_support as support  # noqa: E402
from sam3ext import layout_lanes as ll  # noqa: E402
from sam3ext import ui_extra_schedulers as ues  # noqa: E402
from sam3ext.extra_schedulers import registry  # noqa: E402
from sam3ext.extra_schedulers import schedulers as sch  # noqa: E402
from sam3ext.extra_schedulers import settings as st  # noqa: E402


class _ScriptCase(unittest.TestCase):
    """A freshly loaded script per test class, with the stub Forge modules at hand for ``process``."""

    @classmethod
    def setUpClass(cls):
        registry._LOGGED.clear()
        cls.sd_schedulers = support.stub_sd_schedulers()
        cls.module = support.load_script(sd_schedulers=cls.sd_schedulers)

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        self.script = self.module.ExtraSchedulers()

    def forge(self):
        """Stub Forge modules for code that imports them at call time (scheduler name lookup)."""
        stub = self.module._test_modules_stub
        return support.stub_modules({"modules": stub, "modules.sd_schedulers": stub.sd_schedulers,
                                     "modules.shared": stub.shared})

    def process(self, p, *args):
        with self.forge():
            self.script.process(p, *args)
        return st.active()


def _p(scheduler="custom", **kwargs):
    values = {"scheduler": scheduler, "enable_hr": False, "hr_scheduler": None, "extra_generation_params": {}}
    values.update(kwargs)
    return types.SimpleNamespace(**values)


class LoadTests(_ScriptCase):
    def test_loading_registers_the_schedulers(self):
        self.assertEqual(self.module._REPORT.added, list(registry.SCHEDULER_LABELS))
        labels = [s.label for s in self.sd_schedulers.schedulers]
        for label in registry.SCHEDULER_LABELS:
            self.assertIn(label, labels)
        self.assertEqual(
            self.module._test_stderr.strip(),
            "[Extra Schedulers (sam-extra)] added to Schedule type: Cosine, CosineExponential blend, Phi, Laplace, "
            "Karras Dynamic, custom",
        )

    def test_reloading_the_script_registers_nothing_twice(self):
        again = support.load_script(sd_schedulers=self.sd_schedulers)
        self.assertEqual(again._REPORT.added, [])
        self.assertEqual(again._REPORT.already, list(registry.SCHEDULER_LABELS))
        self.assertEqual(again._test_stderr, "")
        self.assertEqual([s.label for s in self.sd_schedulers.all_schedulers].count("Laplace"), 1)

    def test_script_identity(self):
        # The API's alwayson_scripts key; "Extra Schedulers" alone is aoleg/Neo_ExtraSchedulers' accordion name.
        self.assertEqual(self.script.title(), "Extra Schedulers (sam-extra)")
        self.assertNotEqual(self.script.title().lower(), "extra schedulers")
        self.assertIs(self.script.show(False), self.module._test_modules_stub.scripts.AlwaysVisible)
        self.assertIs(self.script.show(True), self.module._test_modules_stub.scripts.AlwaysVisible)
        self.assertEqual(self.script.sorting_priority, -34)

    def test_section_follows_the_layout_setting(self):
        with mock.patch.object(ll, "sections_enabled", return_value=True):
            self.script.is_img2img = False
            self.assertEqual(self.script.section, ll.ANIMA_SECTION)
            self.script.is_img2img = True
            self.assertIsNone(self.script.section)
        with mock.patch.object(ll, "sections_enabled", return_value=False):
            self.script.is_img2img = False
            self.assertIsNone(self.script.section)

    def test_lane_registry_places_it_with_the_anima_tuning_scripts(self):
        key = ll.slot_key("/forge/extensions/forge_sam3_extension/scripts/anima_extra_schedulers.py")
        self.assertEqual(key, "anima-extra-schedulers")
        self.assertEqual(ll.lane_for(key), "anima")
        self.assertTrue(ll.REGISTRY[key].on.none, "no enable checkbox to show")

    def test_before_ui_callback_is_registered(self):
        self.assertIn(("before_ui", self.module._on_before_ui), self.module._test_callbacks)


@unittest.skipUnless(support.forge_root(), "Forge checkout not found")
class ForgeScriptIdentityTests(_ScriptCase):
    """The title with Forge's own code: element ids (``Script.elem_id``) and the API's always-on lookup."""

    def test_element_ids_come_from_the_title(self):
        forge_elem_id = support.forge_function("modules/scripts.py", "Script.elem_id", {"re": re})
        for is_img2img, tab in ((False, "txt2img"), (True, "img2img")):
            self.script.is_img2img = is_img2img
            with self.subTest(tab=tab):
                self.assertEqual(forge_elem_id(self.script, "custom_mode"),
                                 f"script_{tab}_extra_schedulers_samextra_custom_mode")
                self.assertEqual(self.script.elem_id("custom_mode"), forge_elem_id(self.script, "custom_mode"),
                                 "the test stub's rule is Forge's")

    def test_the_api_finds_this_script_and_not_a_same_named_one(self):
        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)

        index = support.forge_function("modules/api/api.py", "script_name_to_index", {"HTTPException": HTTPException})
        fork = types.SimpleNamespace(title=lambda: "Extra Schedulers")   # aoleg/Neo_ExtraSchedulers' accordion name
        scripts = [fork, self.script]
        self.assertEqual(index("Extra Schedulers (sam-extra)", scripts), 1)
        self.assertEqual(index("extra schedulers (SAM-EXTRA)", scripts), 1)
        self.assertEqual(index("Extra Schedulers", scripts), 0)


class PasteResolutionTests(_ScriptCase):
    """With Forge's map at hand, a Schedule type written as a name or README alias counts as ours."""

    def test_laplace_by_name_gets_the_default_values(self):
        with self.forge():
            for params in ({"Schedule type": "laplace"}, {"Schedule type": "Laplace"},
                           {"Schedule type": "Karras", "Hires schedule type": "laplace"}):
                with self.subTest(params=params):
                    self.assertEqual((ues.paste_laplace_mu(params), ues.paste_laplace_beta(params)), (0.0, 0.5))
            self.assertIsNone(ues.paste_laplace_mu({"Schedule type": "karras dynamic"}))


class UiTests(_ScriptCase):
    def test_controls_defaults_and_paste_fields(self):
        with gr.Blocks():
            controls = self.script.ui(False)
        self.assertEqual(len(controls), len(ues.ARG_NAMES))
        mode, expression, sigmas, mu, beta = controls
        self.assertEqual(mode.value, "Expression")
        self.assertEqual(list(mode.choices), [("Expression", "Expression"), ("Sigma list", "Sigma list")])
        self.assertEqual(expression.value, st.DEFAULT_EXPRESSION)
        self.assertEqual(sigmas.value, st.DEFAULT_SIGMAS)
        self.assertEqual((mu.value, beta.value), (0.0, 0.5))
        self.assertEqual([c.elem_id for c in controls], [
            "script_txt2img_extra_schedulers_samextra_custom_mode",
            "script_txt2img_extra_schedulers_samextra_custom_expression",
            "script_txt2img_extra_schedulers_samextra_custom_sigmas",
            "script_txt2img_extra_schedulers_samextra_laplace_mu",
            "script_txt2img_extra_schedulers_samextra_laplace_beta",
        ])
        # Paste fields are bound to the components (and the infotext keys), not to the title.
        self.assertEqual([component for component, _paste in self.script.infotext_fields], list(controls))
        self.assertEqual([paste for _component, paste in self.script.infotext_fields],
                         [ues.paste_mode, ues.paste_expression, ues.paste_sigmas,
                          ues.paste_laplace_mu, ues.paste_laplace_beta])
        self.script.is_img2img = True
        with gr.Blocks():
            img2img_controls = self.script.ui(True)
        self.assertEqual(img2img_controls[0].elem_id, "script_img2img_extra_schedulers_samextra_custom_mode")

    def test_one_closed_accordion(self):
        with gr.Blocks() as blocks:
            self.script.ui(False)
        accordions = [block for block in blocks.blocks.values() if isinstance(block, gr.Accordion)]
        self.assertEqual(len(accordions), 1)
        self.assertEqual(accordions[0].label, "Extra Schedulers")   # readable; the title carries "(sam-extra)"
        self.assertEqual(accordions[0].elem_id, "script_txt2img_extra_schedulers_samextra_accordion")
        self.assertFalse(accordions[0].open)

    def test_the_default_ui_values_are_the_default_settings(self):
        with gr.Blocks():
            controls = self.script.ui(False)
        self.assertEqual(ues.coerce_args([c.value for c in controls]), st.DEFAULTS)


class ProcessTests(_ScriptCase):
    def test_positional_arguments_become_the_generation_state(self):
        active = self.process(_p(), "Sigma list", "  M * x + m ", "[3, 2, 1]", 1.5, 0.25)
        self.assertEqual(active, st.ExtraSchedulerSettings(
            custom_mode=st.MODE_SIGMAS, custom_expression="M * x + m", custom_sigmas="[3, 2, 1]",
            laplace_mu=1.5, laplace_beta=0.25))

    def test_dict_argument_and_missing_values(self):
        active = self.process(_p(), {"laplace_beta": 2, "custom_mode": "sigmas"})
        self.assertEqual(active, st.ExtraSchedulerSettings(custom_mode=st.MODE_SIGMAS, laplace_beta=2.0))
        self.assertEqual(self.process(_p()), st.DEFAULTS)

    def test_malformed_arguments_fall_back_without_raising(self):
        active = self.process(_p(), None, None, None, "nan", object())
        self.assertEqual(active, st.DEFAULTS)
        active = self.process(_p(), "???", 12, 34, "1e9", "-5")
        self.assertEqual(active.custom_mode, st.MODE_EXPRESSION)
        self.assertEqual((active.custom_expression, active.custom_sigmas), ("12", "34"))
        self.assertEqual((active.laplace_mu, active.laplace_beta), (st.LAPLACE_MU_MAX, st.LAPLACE_BETA_MIN))

    def test_state_is_replaced_every_generation(self):
        self.process(_p(), "Expression", "M * x + m", "", 3.0, 3.0)
        self.assertEqual(self.process(_p(), *[None] * 5), st.DEFAULTS)

    def test_state_is_set_even_when_the_infotext_fails(self):
        stderr = io.StringIO()
        with mock.patch.object(ues, "write_infotext", side_effect=RuntimeError("boom")), \
                contextlib.redirect_stderr(stderr):
            active = self.process(_p(), "Expression", "m + x", "", 0.0, 0.5)
        self.assertEqual(active.custom_expression, "m + x")
        self.assertIn("infotext not written: RuntimeError: boom", stderr.getvalue())

    def test_process_tolerates_a_bare_processing_object(self):
        p = types.SimpleNamespace()
        self.process(p, "Expression", "m + x", "", 0.0, 0.5)
        self.assertEqual(p.extra_generation_params, {})

    def test_xyz_overrides_win_over_the_ui_values(self):
        p = _p("Laplace")
        setattr(p, ues.XYZ_ATTR, {"laplace_mu": "2.5", "laplace_beta": 9.0, "custom_expression": "m * 2",
                                  "custom_mode": "expression", "unknown": 1})
        active = self.process(p, "Sigma list", "M", "[1, 0]", 0.0, 0.5)
        self.assertEqual((active.laplace_mu, active.laplace_beta), (2.5, 9.0))
        self.assertEqual((active.custom_mode, active.custom_expression), (st.MODE_EXPRESSION, "m * 2"))


class InfotextTests(_ScriptCase):
    def test_custom_expression(self):
        p = _p("custom")
        self.process(p, "Expression", " M *\n (m / M) ** x ", "[1, 0]", 1.0, 1.0)
        self.assertEqual(p.extra_generation_params, {"Custom scheduler expression": "M * (m / M) ** x"})

    def test_custom_sigma_list(self):
        p = _p("custom")
        self.process(p, "Sigma list", "m", "[1.0,  0.5,\n 0.0]", 0.0, 0.5)
        self.assertEqual(p.extra_generation_params, {"Custom scheduler sigmas": "[1.0, 0.5, 0.0]"})

    def test_laplace_writes_both_values(self):
        p = _p("Laplace")
        self.process(p, "Expression", "m", "", 0.0, 0.5)
        self.assertEqual(p.extra_generation_params, {"Laplace mu": "0.0", "Laplace beta": "0.5"})
        self.process(p, "Expression", "m", "", 1 / 3, 2.0)
        self.assertEqual(float(p.extra_generation_params["Laplace mu"]), 1 / 3, "round-trips exactly")

    def test_hires_pass_scheduler_counts(self):
        p = _p("Karras", enable_hr=True, hr_scheduler="Laplace")
        self.process(p, "Expression", "m", "", 0.0, 0.5)
        self.assertEqual(set(p.extra_generation_params), {"Laplace mu", "Laplace beta"})
        p = _p("custom", enable_hr=True, hr_scheduler="Use same scheduler")
        self.process(p, "Expression", "m", "", 0.0, 0.5)
        self.assertEqual(set(p.extra_generation_params), {"Custom scheduler expression"})
        p = _p("Karras", enable_hr=False, hr_scheduler="Laplace")   # hires off: its scheduler is not used
        self.process(p, "Expression", "m", "", 0.0, 0.5)
        self.assertEqual(p.extra_generation_params, {})

    def test_scheduler_names_are_resolved_to_labels(self):
        p = _p("laplace")    # the API accepts names too
        self.process(p, "Expression", "m", "", 0.0, 0.5)
        self.assertIn("Laplace mu", p.extra_generation_params)

    def test_other_schedulers_write_nothing_and_stale_keys_go(self):
        shared_params = {"Other": 1}
        cell_1 = _p("Laplace", extra_generation_params=shared_params)
        self.process(cell_1, "Expression", "m", "", 0.5, 0.5)
        self.assertIn("Laplace mu", shared_params)
        cell_2 = _p("Karras", extra_generation_params=shared_params)   # an XYZ cell sharing the dict
        self.process(cell_2, "Expression", "m", "", 0.5, 0.5)
        self.assertEqual(shared_params, {"Other": 1})


class ForeignSchedulerInfotextTests(unittest.TestCase):
    """A label another extension registered first (ours is skipped) runs that extension's scheduler with
    its own values, and a hidden label runs none — this accordion's values must not be recorded for them."""

    def _load(self, **kwargs):
        registry._LOGGED.clear()
        self.module = support.load_script(**kwargs)

    def process(self, p, *args):
        stub = self.module._test_modules_stub
        with support.stub_modules({"modules": stub, "modules.sd_schedulers": stub.sd_schedulers,
                                   "modules.shared": stub.shared}):
            self.module.ExtraSchedulers().process(p, *args)
        return p.extra_generation_params

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)

    def test_a_label_taken_by_another_extension_writes_nothing(self):
        foreign_module = support.stub_sd_schedulers()
        foreign = support.StubScheduler("laplace_other", "Laplace", lambda *a, **k: None)
        foreign_module.all_schedulers.append(foreign)
        foreign_module.schedulers.append(foreign)
        foreign_module.schedulers_map.update({"laplace_other": foreign, "Laplace": foreign})
        self._load(sd_schedulers=foreign_module)
        self.assertEqual(self.module._REPORT.skipped, ["Laplace"])
        for p in (_p("Laplace"), _p("laplace_other"), _p("Karras", enable_hr=True, hr_scheduler="Laplace")):
            with self.subTest(scheduler=p.scheduler, hires=p.hr_scheduler):
                self.assertEqual(self.process(p, "Expression", "m + x", "", 1.5, 2.0), {})
        p = _p("Laplace", enable_hr=True, hr_scheduler="custom")   # ours still records its own values
        self.assertEqual(self.process(p, "Expression", "m + x", "", 1.5, 2.0), {"Custom scheduler expression": "m + x"})

    def test_a_hidden_label_writes_nothing(self):
        self._load(sd_schedulers=support.stub_sd_schedulers(), hidden_schedulers=["custom"])
        self.assertEqual(self.module._REPORT.hidden, ["custom"])
        # Forge's get_sigmas finds no "custom" in schedulers_map and samples with the model's own sigmas.
        p = _p("Karras", enable_hr=True, hr_scheduler="custom")
        self.assertEqual(self.process(p, "Expression", "m + x", "", 0.0, 0.5), {})


class PasteTests(unittest.TestCase):
    def test_custom_expression(self):
        params = {"Schedule type": "custom", "Custom scheduler expression": "m + x"}
        self.assertEqual(ues.paste_mode(params), "Expression")
        self.assertEqual(ues.paste_expression(params), "m + x")
        self.assertIsNone(ues.paste_sigmas(params))

    def test_custom_sigmas(self):
        params = {"Schedule type": "custom", "Custom scheduler sigmas": "[1, 0.5, 0]"}
        self.assertEqual(ues.paste_mode(params), "Sigma list")
        self.assertEqual(ues.paste_sigmas(params), "[1, 0.5, 0]")
        self.assertIsNone(ues.paste_expression(params))

    def test_laplace_values_or_defaults(self):
        params = {"Schedule type": "Laplace", "Laplace mu": "1.5", "Laplace beta": "0.25"}
        self.assertEqual((ues.paste_laplace_mu(params), ues.paste_laplace_beta(params)), (1.5, 0.25))
        params = {"Schedule type": "Karras", "Hires schedule type": "Laplace"}
        self.assertEqual((ues.paste_laplace_mu(params), ues.paste_laplace_beta(params)), (0.0, 0.5))
        params = {"Schedule type": "Laplace", "Laplace mu": "99", "Laplace beta": "nan"}
        self.assertEqual((ues.paste_laplace_mu(params), ues.paste_laplace_beta(params)), (10.0, 0.5))

    def test_images_without_these_schedulers_leave_the_fields_alone(self):
        params = {"Schedule type": "Karras", "Hires schedule type": "Use same scheduler"}
        for paste in (ues.paste_mode, ues.paste_expression, ues.paste_sigmas, ues.paste_laplace_mu,
                      ues.paste_laplace_beta):
            self.assertIsNone(paste(params))

    def test_pasting_only_copies_text(self):
        evil = "__import__('os').system('echo pwned')"
        imported = []
        real_import = builtins.__import__

        def spy(name, *args, **kwargs):
            imported.append(name)
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=spy), \
                mock.patch.object(os, "system", side_effect=AssertionError("executed")):
            self.assertEqual(ues.paste_expression({"Custom scheduler expression": evil}), evil)
            self.assertEqual(ues.paste_mode({"Custom scheduler expression": evil}), "Expression")
        self.assertNotIn("os", imported)


@unittest.skipUnless(support.forge_root(), "Forge checkout not found")
class InfotextRoundTripTests(unittest.TestCase):
    """Written with Forge's quoting, read back with Forge's parameter regex — values survive."""

    @classmethod
    def setUpClass(cls):
        namespace = {"json": json}
        rel = "modules/infotext_utils.py"
        cls.quote = staticmethod(support.forge_function(rel, "quote", namespace))
        cls.unquote = staticmethod(support.forge_function(rel, "unquote", namespace))
        cls.re_param = re.compile(support.forge_constant(rel, "re_param_code"))

    def _round_trip(self, items):
        line = ", ".join(f"{key}: {self.quote(value)}" for key, value in
                         {"Steps": 20, "Sampler": "Euler", **items, "Seed": 1}.items())
        return {key: self.unquote(value) for key, value in self.re_param.findall(line)}

    def test_values_with_commas_colons_and_quotes_survive(self):
        current = st.ExtraSchedulerSettings(custom_expression='max(m, M * (1 - x) ** 2) + 0 * min(1, 2)',
                                            laplace_mu=-1.25, laplace_beta=0.1)
        items = ues.infotext_items(current, {"custom", "Laplace"})
        params = self._round_trip(items)
        self.assertEqual(ues.paste_expression(params), current.custom_expression)
        self.assertEqual(ues.paste_laplace_mu({**params, "Schedule type": "Laplace"}), -1.25)
        self.assertEqual(ues.paste_laplace_beta({**params, "Schedule type": "Laplace"}), 0.1)
        sigmas = st.ExtraSchedulerSettings(custom_mode=st.MODE_SIGMAS, custom_sigmas="[1.0, 0.6, 0.25, 0.1, 0.0]")
        params = self._round_trip(ues.infotext_items(sigmas, {"custom"}))
        self.assertEqual(ues.paste_sigmas(params), "[1.0, 0.6, 0.25, 0.1, 0.0]")
        self.assertEqual(ues.paste_mode(params), "Sigma list")
        odd = st.ExtraSchedulerSettings(custom_expression='m + x # "quoted", with: a colon')
        params = self._round_trip(ues.infotext_items(odd, {"custom"}))
        self.assertEqual(ues.paste_expression(params), odd.custom_expression)

    def test_keys_match_forges_key_pattern(self):
        for key in ues.INFOTEXT_KEYS:
            with self.subTest(key=key):
                params = self._round_trip({key: "1"})
                self.assertEqual(params.get(key), "1")


class CraftedInfotextTests(_ScriptCase):
    """A PNG's parameters can carry any text; using it must not run anything."""

    def test_malicious_expression_from_a_png_is_rejected_at_generation(self):
        params = {"Schedule type": "custom",
                  "Custom scheduler expression": "__import__('os').system('echo pwned') or M"}
        pasted = [paste(params) for paste in (ues.paste_mode, ues.paste_expression, ues.paste_sigmas,
                                              ues.paste_laplace_mu, ues.paste_laplace_beta)]
        self.process(_p("custom"), *pasted)
        imported = []
        real_import = builtins.__import__

        def spy(name, *args, **kwargs):
            imported.append(name)
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=spy), \
                mock.patch.object(os, "system", side_effect=AssertionError("executed")):
            with self.assertRaises(sch.CustomSchedulerError):
                sch.custom(n=10, sigma_min=0.03, sigma_max=14.6)
        self.assertNotIn("os", imported)

    def test_crafted_sigma_list_is_rejected(self):
        for text in ("[1, __import__('os')]", "[" + "9" * 400 + ", 1]", "[1e400, 1]", "[1, 0.5]" * 5000):
            with self.subTest(text=text[:30]):
                self.process(_p("custom"), "Sigma list", "m", text, 0.0, 0.5)
                with self.assertRaises(sch.CustomSchedulerError):
                    sch.custom(n=10, sigma_min=0.03, sigma_max=14.6)


class XyzTests(_ScriptCase):
    class _FakeAxisOption:
        def __init__(self, label, type, apply, format_value=None, confirm=None, cost=0.0, choices=None, prepare=None):
            self.label, self.type, self.apply, self.confirm, self.choices = label, type, apply, confirm, choices

    def _xyz_module(self):
        return types.SimpleNamespace(AxisOption=self._FakeAxisOption, axis_options=[self._FakeAxisOption("Nothing", str, None)])

    def test_axes(self):
        axes = self.module.make_xyz_axes(self._xyz_module())
        self.assertEqual([(a.label, a.type) for a in axes], [
            ("[Extra Schedulers (sam-extra)] Laplace mu", float),
            ("[Extra Schedulers (sam-extra)] Laplace beta", float),
            ("[Extra Schedulers (sam-extra)] Custom expression", str),
            ("[Extra Schedulers (sam-extra)] Custom sigma list", str),
        ])
        # Not mistaken for (and not hidden by) axes under the shorter "[Extra Schedulers]" prefix.
        self.assertFalse(any(a.label.startswith("[Extra Schedulers]") for a in axes))

    def test_axis_values_reach_the_generation(self):
        mu, beta, expression, sigmas = self.module.make_xyz_axes(self._xyz_module())
        p = _p("custom")
        mu.apply(p, 1.5, [1.5])
        beta.apply(p, 0.75, [0.75])
        sigmas.apply(p, "[3, 2, 1]", ["[3, 2, 1]"])
        active = self.process(p, "Expression", "m", "", 0.0, 0.5)
        self.assertEqual((active.laplace_mu, active.laplace_beta), (1.5, 0.75))
        self.assertEqual((active.custom_mode, active.custom_sigmas), (st.MODE_SIGMAS, "[3, 2, 1]"))
        self.assertEqual(p.extra_generation_params, {"Custom scheduler sigmas": "[3, 2, 1]"})
        expression.apply(p, "M * x + m", ["M * x + m"])   # the later-applied custom axis picks the mode
        active = self.process(p, "Sigma list", "m", "[1, 0]", 0.0, 0.5)
        self.assertEqual((active.custom_mode, active.custom_expression), (st.MODE_EXPRESSION, "M * x + m"))

    def test_confirm_rejects_bad_values_before_the_grid_runs(self):
        _mu, _beta, expression, sigmas = self.module.make_xyz_axes(self._xyz_module())
        expression.confirm(None, ["M * (m / M) ** x", "m + (M - m) * (1 - x) ** 3"])
        sigmas.confirm(None, ["[1.0, 0.5, 0.0]", "14.6 3 0.03"])
        with self.assertRaisesRegex(ValueError, "double quotes"):
            expression.confirm(None, ["max(m", "M)"])          # split at a comma
        with self.assertRaisesRegex(ValueError, "double quotes"):
            sigmas.confirm(None, ["[1.0", "0.5", "0.0]"])
        with self.assertRaises(ValueError):
            expression.confirm(None, ["__import__('os')"])

    def test_laplace_axes_check_the_slider_range_before_the_grid_runs(self):
        # Values reach confirm already converted by xyz_grid (opt.type = float); out of range or not
        # finite fails before any cell instead of being clamped silently under an unclamped label.
        mu, beta, _expression, _sigmas = self.module.make_xyz_axes(self._xyz_module())
        mu.confirm(None, [-10.0, -1.5, 0.0, 10.0])
        beta.confirm(None, [0.0, 0.5, 10.0])
        for axis, bad in ((mu, [0.0, 10.5]), (mu, [-11.0]), (mu, [float("nan")]), (mu, [float("inf")]),
                          (beta, [-0.1]), (beta, [10.01]), (beta, [float("nan")])):
            with self.subTest(axis=axis.label, values=bad):
                with self.assertRaisesRegex(ValueError, "outside"):
                    axis.confirm(None, bad)
        with self.assertRaisesRegex(ValueError, r"\[Extra Schedulers \(sam-extra\)\] Laplace mu: 12\.0 is outside -10 … 10"):
            mu.confirm(None, [12.0])

    def test_registration_finds_xyz_grid_once(self):
        xyz_module = self._xyz_module()
        script_data = types.SimpleNamespace(script_class=type("Script", (), {"__module__": "xyz_grid.py"}),
                                            module=xyz_module)
        self.module.scripts.scripts_data[:] = [script_data]
        self.addCleanup(self.module.scripts.scripts_data.clear)
        self.module._on_before_ui()
        self.module._on_before_ui()
        labels = [a.label for a in xyz_module.axis_options]
        self.assertEqual(labels.count("[Extra Schedulers (sam-extra)] Laplace mu"), 1)
        self.assertEqual(len(labels), 5)

    def test_axes_under_the_shorter_prefix_do_not_stop_ours(self):
        # Another extension's "[Extra Schedulers] …" axis (aoleg/Neo_ExtraSchedulers' names are unknown).
        xyz_module = self._xyz_module()
        xyz_module.axis_options.append(self._FakeAxisOption("[Extra Schedulers] Something", float, None))
        script_data = types.SimpleNamespace(script_class=type("Script", (), {"__module__": "xyz_grid.py"}),
                                            module=xyz_module)
        self.module.scripts.scripts_data[:] = [script_data]
        self.addCleanup(self.module.scripts.scripts_data.clear)
        self.module._on_before_ui()
        labels = [a.label for a in xyz_module.axis_options]
        self.assertIn("[Extra Schedulers (sam-extra)] Custom expression", labels)
        self.assertEqual(len(labels), 6)

    def test_missing_xyz_grid_is_fine(self):
        self.module.scripts.scripts_data[:] = []
        self.module._on_before_ui()


if __name__ == "__main__":
    unittest.main()
