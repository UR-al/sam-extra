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
from dataclasses import replace
from pathlib import Path
from unittest import mock

import gradio as gr
import torch

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


# The paste function of each script argument, in ARG_NAMES order.
PASTES = (ues.paste_mode, ues.paste_expression, ues.paste_sigmas, ues.paste_laplace_mu, ues.paste_laplace_beta,
          ues.paste_flow_cosmos_rho, ues.paste_flow_cosmos_sigma_max, ues.paste_flow_cosmos_sigma_min)
# A full v0.33.0 request (8 arguments) with every value away from its default.
FULL_ARGS = ("Sigma list", "  M * x + m ", "[3, 2, 1]", 1.5, 0.25, 5.5, 240.0, 0.006)
FULL_SETTINGS = st.ExtraSchedulerSettings(
    custom_mode=st.MODE_SIGMAS, custom_expression="M * x + m", custom_sigmas="[3, 2, 1]", laplace_mu=1.5,
    laplace_beta=0.25, flow_cosmos_rho=5.5, flow_cosmos_sigma_max=240.0, flow_cosmos_sigma_min=0.006)


class LoadTests(_ScriptCase):
    def test_loading_registers_the_schedulers(self):
        self.assertEqual(self.module._REPORT.added, list(registry.SCHEDULER_LABELS))
        labels = [s.label for s in self.sd_schedulers.schedulers]
        for label in registry.SCHEDULER_LABELS:
            self.assertIn(label, labels)
        self.assertEqual(
            self.module._test_stderr.strip(),
            "[Extra Schedulers (sam-extra)] added to Schedule type: Cosine, CosineExponential blend, Phi, Laplace, "
            "Karras Dynamic, custom, React Cosinusoidal DynSF, Flow Cosmos rho7, Flow Cosmos Dynamic",
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

    def test_flow_cosmos_by_name_gets_the_default_values(self):
        with self.forge():
            for params in ({"Schedule type": "flow_cosmos_rho7"}, {"Schedule type": "Flow Cosmos rho7"},
                           {"Schedule type": "Karras", "Hires schedule type": "flow_cosmos_rho7"}):
                with self.subTest(params=params):
                    self.assertEqual([paste(params) for paste in PASTES[5:]], [7.0, 80.0, 0.002])
            self.assertIsNone(ues.paste_flow_cosmos_rho({"Schedule type": "karras"}))

    def test_flow_cosmos_dynamic_by_name_gets_the_default_values(self):
        with self.forge():
            for params in ({"Schedule type": "flow_cosmos_dynamic"}, {"Schedule type": "Flow Cosmos Dynamic"},
                           {"Schedule type": "Karras", "Hires schedule type": "flow_cosmos_dynamic"}):
                with self.subTest(params=params):
                    self.assertEqual([paste(params) for paste in PASTES[5:]], [7.0, 80.0, 0.002])
            # Karras Dynamic reads Settings → rho, not these values
            self.assertIsNone(ues.paste_flow_cosmos_rho({"Schedule type": "karras_dynamic"}))


class UiTests(_ScriptCase):
    def test_controls_defaults_and_paste_fields(self):
        with gr.Blocks():
            controls = self.script.ui(False)
        self.assertEqual(len(controls), len(ues.ARG_NAMES))
        self.assertEqual(ues.ARG_NAMES, ("custom_mode", "custom_expression", "custom_sigmas", "laplace_mu",
                                         "laplace_beta", "flow_cosmos_rho", "flow_cosmos_sigma_max",
                                         "flow_cosmos_sigma_min"))
        mode, expression, sigmas, mu, beta, cosmos_rho, cosmos_max, cosmos_min = controls
        self.assertEqual(mode.value, "Expression")
        self.assertEqual(list(mode.choices), [("Expression", "Expression"), ("Sigma list", "Sigma list")])
        self.assertEqual(expression.value, st.DEFAULT_EXPRESSION)
        self.assertEqual(sigmas.value, st.DEFAULT_SIGMAS)
        self.assertEqual((mu.value, beta.value), (0.0, 0.5))
        self.assertEqual((cosmos_rho.value, cosmos_max.value, cosmos_min.value), (7.0, 80.0, 0.002))
        self.assertEqual([c.elem_id for c in controls], [
            "script_txt2img_extra_schedulers_samextra_custom_mode",
            "script_txt2img_extra_schedulers_samextra_custom_expression",
            "script_txt2img_extra_schedulers_samextra_custom_sigmas",
            "script_txt2img_extra_schedulers_samextra_laplace_mu",
            "script_txt2img_extra_schedulers_samextra_laplace_beta",
            "script_txt2img_extra_schedulers_samextra_flow_cosmos_rho",
            "script_txt2img_extra_schedulers_samextra_flow_cosmos_sigma_max",
            "script_txt2img_extra_schedulers_samextra_flow_cosmos_sigma_min",
        ])
        # Paste fields are bound to the components (and the infotext keys), not to the title.
        self.assertEqual([component for component, _paste in self.script.infotext_fields], list(controls))
        self.assertEqual([paste for _component, paste in self.script.infotext_fields], list(PASTES))
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

    def test_flow_cosmos_sliders_follow_the_laplace_ones(self):
        """One row of three sliders after the Laplace row, always shown like μ/β; ranges and steps from settings."""
        with gr.Blocks() as blocks:
            controls = self.script.ui(False)
        mu, beta, cosmos_rho, cosmos_max, cosmos_min = controls[3:]
        expected = {
            cosmos_rho: ("Flow Cosmos rho", 1.0, 15.0, 0.1, 7.0),
            cosmos_max: ("Flow Cosmos sigma max (σ̃)", 1.0, 1000.0, 0.1, 80.0),
            cosmos_min: ("Flow Cosmos sigma min (σ̃)", 0.0001, 1.0, 0.0001, 0.002),
        }
        for slider, (label, low, high, step, value) in expected.items():
            with self.subTest(label=label):
                self.assertIsInstance(slider, gr.Slider)
                self.assertEqual((slider.label, slider.minimum, slider.maximum, slider.step, slider.value),
                                 (label, low, high, step, value))
                self.assertTrue(slider.visible)
                self.assertTrue(low <= value <= high)
        self.assertEqual((mu.visible, beta.visible), (True, True))
        def components(block):   # Gradio wraps a row's inputs in a Form
            return [leaf for child in block.children
                    for leaf in (components(child) if hasattr(child, "children") else [child])]

        rows = [block for block in blocks.blocks.values() if isinstance(block, gr.Row)]
        self.assertEqual([components(row) for row in rows], [[mu, beta], [cosmos_rho, cosmos_max, cosmos_min]])
        # the slider ranges meet only at 1, where σ̃ min < σ̃ max fails (checked when the scheduler runs)
        self.assertEqual(st.FLOW_COSMOS_SIGMA_MIN_MAX, st.FLOW_COSMOS_SIGMA_MAX_MIN)


class FlowCosmosCopyTests(unittest.TestCase):
    """The accordion help and the README say what the code does (D19 review): the σ̃ range error is a flow-model one
    (eps/v models do not use σ̃ — tests/test_extra_schedulers_schedulers.py runs them with σ̃ 1 … 1), the narrow-range
    flat-step example names where the rule starts (841 steps, pinned there too), and no text refers to an earlier
    Flow Cosmos list or image without its keys — Flow Cosmos rho7 and its three values arrive together in v0.33.0."""

    README = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_the_help_scopes_the_sigma_range_error_to_flow_models(self):
        self.assertIn("flow 모델에서 σ̃ min 이 σ̃ max 보다 작지 않으면(둘 다 1) 생성이 오류로 멈춥니다(eps/v 는 σ̃ 를 쓰지 않아 "
                      "그대로 돎)", ues.HELP)
        self.assertEqual(ues.HELP.count("σ̃ max 보다 작지 않으면"), 1)

    def test_no_text_refers_to_an_earlier_flow_cosmos_list(self):
        for text, where in ((ues.HELP, "HELP"), (self.README, "README.md")):
            for phrase in ("전과 같은 목록", "v0.33.0 의 처음 목록", "이 키가 생기기 전"):
                with self.subTest(where=where, phrase=phrase):
                    self.assertNotIn(phrase, text)

    def test_the_readme_gives_where_the_narrow_range_flat_step_starts(self):
        # README Markdown escapes the range's tilde ("\\~" — a bare "a~b … c~d" pair renders struck through)
        self.assertIn("σ̃ 0.9999\\~1 은 841 스텝부터", self.README)
        self.assertNotIn("σ̃ 0.9999\\~1 은 1000 스텝", self.README)
        self.assertNotIn("1 은 1000 스텝부터", self.README)

    def test_the_help_and_the_readme_name_every_scheduler(self):
        """v0.33.0 (D20): nine schedulers, Flow Cosmos Dynamic with its own help line and README table row, credited
        as this extension's own composition."""
        self.assertEqual(len(registry.SCHEDULER_LABELS), 9)
        self.assertIn(" 9종이 추가됩니다", ues.HELP)
        self.assertIn("스케줄러 9개를 더합니다", self.README)
        for label in registry.SCHEDULER_LABELS:
            with self.subTest(label=label):
                self.assertIn(label, ues.HELP)
                self.assertEqual(self.README.count(f"| **{label}** |"), 1)
        self.assertIn("- **Flow Cosmos Dynamic** — Karras Dynamic 의 flow 모델판(이 확장이 두 항목을 합친 것)", ues.HELP)
        self.assertIn("| **Flow Cosmos Dynamic** | Karras Dynamic 의 flow 모델판 — 이 확장이 Karras Dynamic 과 Flow Cosmos "
                      "rho7 을 합친 것입니다", self.README)


class ProcessTests(_ScriptCase):
    def test_positional_arguments_become_the_generation_state(self):
        self.assertEqual(self.process(_p(), *FULL_ARGS), FULL_SETTINGS)

    def test_requests_with_0_to_9_arguments(self):
        """v0.33.0 grew from 5 to 8 arguments, the three new ones last: an older request (5 or fewer) runs Flow Cosmos
        with its defaults, missing values take the defaults, a 9th value is ignored."""
        defaults = (st.DEFAULTS.custom_mode, st.DEFAULTS.custom_expression, st.DEFAULTS.custom_sigmas,
                    st.DEFAULTS.laplace_mu, st.DEFAULTS.laplace_beta, st.DEFAULTS.flow_cosmos_rho,
                    st.DEFAULTS.flow_cosmos_sigma_max, st.DEFAULTS.flow_cosmos_sigma_min)
        full = tuple(getattr(FULL_SETTINGS, name) for name in ues.ARG_NAMES)
        for count in (0, 2, 3, 5, 8, 9):
            with self.subTest(arguments=count):
                args = (*FULL_ARGS, "ignored")[:count]
                active = self.process(_p(), *args)
                values = tuple(getattr(active, name) for name in ues.ARG_NAMES)
                given = min(count, len(ues.ARG_NAMES))
                self.assertEqual(values, full[:given] + defaults[given:])
        self.assertEqual(self.process(_p(), *FULL_ARGS[:5]).flow_cosmos_rho, 7.0)

    def test_dict_argument_and_missing_values(self):
        active = self.process(_p(), {"laplace_beta": 2, "custom_mode": "sigmas"})
        self.assertEqual(active, st.ExtraSchedulerSettings(custom_mode=st.MODE_SIGMAS, laplace_beta=2.0))
        self.assertEqual(self.process(_p()), st.DEFAULTS)
        active = self.process(_p(), {"flow_cosmos_sigma_min": 0.006, "flow_cosmos_rho": "4.5"})
        self.assertEqual(active, st.ExtraSchedulerSettings(flow_cosmos_rho=4.5, flow_cosmos_sigma_min=0.006))

    def test_malformed_arguments_fall_back_without_raising(self):
        active = self.process(_p(), None, None, None, "nan", object(), None, "inf", [1])
        self.assertEqual(active, st.DEFAULTS)
        active = self.process(_p(), "???", 12, 34, "1e9", "-5", True, "-inf", "x")
        self.assertEqual(active.custom_mode, st.MODE_EXPRESSION)
        self.assertEqual((active.custom_expression, active.custom_sigmas), ("12", "34"))
        self.assertEqual((active.laplace_mu, active.laplace_beta), (st.LAPLACE_MU_MAX, st.LAPLACE_BETA_MIN))
        self.assertEqual((active.flow_cosmos_rho, active.flow_cosmos_sigma_max, active.flow_cosmos_sigma_min),
                         (7.0, 80.0, 0.002))

    def test_a_huge_api_integer_reads_as_the_default_and_keeps_the_other_values(self):
        """JSON integers reach the script as Python ints, and ``float()`` of one past ~1.8e308 raises OverflowError
        (a 400-digit string or 1e400 gives inf instead). Such a value reads as the default — like inf or 1e400 — and
        the request's other values are kept instead of the whole accordion falling back to the defaults."""
        huge = 10 ** 400
        cases = {
            "flow_cosmos_rho": (*FULL_ARGS[:5], huge, 240.0, 0.006),
            "flow_cosmos_sigma_max": (*FULL_ARGS[:6], -huge, 0.006),
            "flow_cosmos_sigma_min": (*FULL_ARGS[:7], huge),
            "laplace_mu": (*FULL_ARGS[:3], huge, *FULL_ARGS[4:]),
            "laplace_beta": (*FULL_ARGS[:4], -huge, *FULL_ARGS[5:]),
        }
        for field, args in cases.items():
            with self.subTest(field=field):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    active = self.process(_p(), *args)
                expected = replace(FULL_SETTINGS, **{field: getattr(st.DEFAULTS, field)})
                self.assertEqual(active, expected)
                self.assertNotIn("bad arguments", stderr.getvalue())
        request = '["Sigma list", "M", "[3, 2, 1]", 1.5, 0.25, ' + "9" * 400 + ", 240, 0.006]"
        active = self.process(_p(), *json.loads(request))
        self.assertEqual(active, st.ExtraSchedulerSettings(
            custom_mode=st.MODE_SIGMAS, custom_expression="M", custom_sigmas="[3, 2, 1]", laplace_mu=1.5,
            laplace_beta=0.25, flow_cosmos_rho=7.0, flow_cosmos_sigma_max=240.0, flow_cosmos_sigma_min=0.006),
            "the JSON-decoded API request of the review")
        self.assertEqual(st.coerce({"laplace_mu": huge, "flow_cosmos_sigma_max": 300}),
                         st.ExtraSchedulerSettings(flow_cosmos_sigma_max=300.0))
        self.assertEqual(st.with_overrides(FULL_SETTINGS, {"flow_cosmos_rho": huge, "laplace_beta": 3}),
                         replace(FULL_SETTINGS, flow_cosmos_rho=7.0, laplace_beta=3.0))
        for raw in ("9" * 400, 1e400, "1e400"):
            with self.subTest(same_as=raw):
                self.assertEqual(st.coerce_flow_cosmos_rho(raw), st.coerce_flow_cosmos_rho(huge))

    def test_flow_cosmos_values_outside_the_sliders_are_clamped_like_laplaces(self):
        cases = {(0.0, 0.5, 5.0): (1.0, 1.0, 1.0), (-3, 1e9, 1e-9): (1.0, 1000.0, 0.0001),
                 (99, "2000", "0"): (15.0, 1000.0, 0.0001), ("0.5", 0, -1): (1.0, 1.0, 0.0001)}
        for raw, clamped in cases.items():
            with self.subTest(raw=raw):
                active = self.process(_p(), "Expression", "m", "", 0.0, 0.5, *raw)
                self.assertEqual((active.flow_cosmos_rho, active.flow_cosmos_sigma_max, active.flow_cosmos_sigma_min),
                                 clamped)

    def test_state_is_replaced_every_generation(self):
        self.process(_p(), "Expression", "M * x + m", "", 3.0, 3.0, 4.0, 100.0, 0.5)
        self.assertEqual(self.process(_p(), *[None] * 8), st.DEFAULTS)
        self.process(_p(), *FULL_ARGS)
        older = st.ExtraSchedulerSettings(custom_mode=st.MODE_SIGMAS, custom_expression="M * x + m",
                                          custom_sigmas="[3, 2, 1]", laplace_mu=1.5, laplace_beta=0.25)
        self.assertEqual(self.process(_p(), *FULL_ARGS[:5]), older,
                         "an older 5-argument request does not keep the previous generation's Flow Cosmos values")

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
                                  "custom_mode": "expression", "unknown": 1, "flow_cosmos_rho": 3.5,
                                  "flow_cosmos_sigma_max": "500", "flow_cosmos_sigma_min": 2.0})
        active = self.process(p, "Sigma list", "M", "[1, 0]", 0.0, 0.5, 7.0, 80.0, 0.002)
        self.assertEqual((active.laplace_mu, active.laplace_beta), (2.5, 9.0))
        self.assertEqual((active.custom_mode, active.custom_expression), (st.MODE_EXPRESSION, "m * 2"))
        self.assertEqual((active.flow_cosmos_rho, active.flow_cosmos_sigma_max, active.flow_cosmos_sigma_min),
                         (3.5, 500.0, 1.0), "through the same coercion (clamped)")


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
        cell_3 = _p("Flow Cosmos rho7", extra_generation_params=shared_params)
        self.process(cell_3, *FULL_ARGS)
        self.assertIn("Flow Cosmos rho", shared_params)
        self.process(_p("Laplace", extra_generation_params=shared_params), *FULL_ARGS)
        self.assertEqual(shared_params, {"Other": 1, "Laplace mu": "1.5", "Laplace beta": "0.25"})

    def test_flow_cosmos_writes_all_three_values(self):
        """Like Laplace: whenever Flow Cosmos rho7 runs (main or hires pass, by label or name) — also with the
        defaults, and also on an eps/v model, where only the rho is used (which model runs is not known here)."""
        p = _p("Flow Cosmos rho7")
        self.process(p, *FULL_ARGS)
        self.assertEqual(p.extra_generation_params, {"Flow Cosmos rho": "5.5", "Flow Cosmos sigma max": "240.0",
                                                     "Flow Cosmos sigma min": "0.006"})
        for p in (_p("flow_cosmos_rho7"), _p("Karras", enable_hr=True, hr_scheduler="Flow Cosmos rho7"),
                  _p("Flow Cosmos rho7", enable_hr=True, hr_scheduler="Use same scheduler")):
            with self.subTest(scheduler=p.scheduler, hires=p.hr_scheduler):
                self.process(p)
                self.assertEqual(p.extra_generation_params, {"Flow Cosmos rho": "7.0", "Flow Cosmos sigma max": "80.0",
                                                             "Flow Cosmos sigma min": "0.002"})
        p = _p("Karras", enable_hr=False, hr_scheduler="Flow Cosmos rho7")   # hires off: its scheduler is not used
        self.process(p, *FULL_ARGS)
        self.assertEqual(p.extra_generation_params, {})
        p = _p("Laplace", enable_hr=True, hr_scheduler="Flow Cosmos rho7")
        self.process(p, *FULL_ARGS)
        self.assertEqual(set(p.extra_generation_params), {"Laplace mu", "Laplace beta", "Flow Cosmos rho",
                                                          "Flow Cosmos sigma max", "Flow Cosmos sigma min"})
        p = _p("Flow Cosmos rho7")
        self.process(p, "Expression", "m", "", 0.0, 0.5, 22 / 7, 1000 / 7, 0.0001 * 3)
        self.assertEqual([float(p.extra_generation_params[key]) for key in ues.INFOTEXT_KEYS[4:]],
                         [22 / 7, 1000 / 7, 0.0001 * 3], "round-trips exactly")

    def test_flow_cosmos_dynamic_writes_the_same_three_values(self):
        """Flow Cosmos Dynamic reads the same three values, so the same keys are written whenever it runs (main or
        hires pass, by label or name, defaults too) — once when both Flow Cosmos schedulers run."""
        p = _p("Flow Cosmos Dynamic")
        self.process(p, *FULL_ARGS)
        self.assertEqual(p.extra_generation_params, {"Flow Cosmos rho": "5.5", "Flow Cosmos sigma max": "240.0",
                                                     "Flow Cosmos sigma min": "0.006"})
        for p in (_p("flow_cosmos_dynamic"), _p("Karras", enable_hr=True, hr_scheduler="Flow Cosmos Dynamic"),
                  _p("Flow Cosmos Dynamic", enable_hr=True, hr_scheduler="Use same scheduler"),
                  _p("Flow Cosmos rho7", enable_hr=True, hr_scheduler="flow_cosmos_dynamic")):
            with self.subTest(scheduler=p.scheduler, hires=p.hr_scheduler):
                self.process(p)
                self.assertEqual(p.extra_generation_params, {"Flow Cosmos rho": "7.0", "Flow Cosmos sigma max": "80.0",
                                                             "Flow Cosmos sigma min": "0.002"})
        p = _p("Karras", enable_hr=False, hr_scheduler="Flow Cosmos Dynamic")   # hires off: its scheduler is not used
        self.process(p, *FULL_ARGS)
        self.assertEqual(p.extra_generation_params, {})
        p = _p("Karras Dynamic", enable_hr=True, hr_scheduler="Laplace")   # Karras Dynamic reads none of them
        self.process(p, *FULL_ARGS)
        self.assertEqual(set(p.extra_generation_params), {"Laplace mu", "Laplace beta"})
        shared_params = {}   # an XYZ cell sharing the dict: the keys go when the next cell does not use them
        self.process(_p("Flow Cosmos Dynamic", extra_generation_params=shared_params), *FULL_ARGS)
        self.assertIn("Flow Cosmos rho", shared_params)
        self.process(_p("Karras Dynamic", extra_generation_params=shared_params), *FULL_ARGS)
        self.assertEqual(shared_params, {})


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

    def test_flow_cosmos_values_or_defaults(self):
        params = {"Schedule type": "Flow Cosmos rho7", "Flow Cosmos rho": "5.5", "Flow Cosmos sigma max": "240.0",
                  "Flow Cosmos sigma min": "0.006"}
        self.assertEqual([paste(params) for paste in PASTES[5:]], [5.5, 240.0, 0.006])
        # an image of v0.33.0 before these keys (or another tool's) used the defaults
        for params in ({"Schedule type": "Flow Cosmos rho7"},
                       {"Schedule type": "Karras", "Hires schedule type": "Flow Cosmos rho7"}):
            with self.subTest(params=params):
                self.assertEqual([paste(params) for paste in PASTES[5:]], [7.0, 80.0, 0.002])
                self.assertEqual([paste(params) for paste in PASTES[:5]], [None] * 5, "other fields stay")
        params = {"Schedule type": "Flow Cosmos rho7", "Flow Cosmos rho": "99", "Flow Cosmos sigma max": "nan",
                  "Flow Cosmos sigma min": "-1"}
        self.assertEqual([paste(params) for paste in PASTES[5:]], [15.0, 80.0, 0.0001], "coerced like the arguments")
        params = {"Schedule type": "Flow Cosmos rho7", "Flow Cosmos sigma min": "0.01"}   # one key only
        self.assertEqual([paste(params) for paste in PASTES[5:]], [7.0, 80.0, 0.01])

    def test_flow_cosmos_dynamic_values_or_defaults(self):
        """The same keys and rules as Flow Cosmos rho7 (the two schedulers share the values)."""
        params = {"Schedule type": "Flow Cosmos Dynamic", "Flow Cosmos rho": "5.5", "Flow Cosmos sigma max": "240.0",
                  "Flow Cosmos sigma min": "0.006"}
        self.assertEqual([paste(params) for paste in PASTES[5:]], [5.5, 240.0, 0.006])
        for params in ({"Schedule type": "Flow Cosmos Dynamic"},
                       {"Schedule type": "Karras", "Hires schedule type": "Flow Cosmos Dynamic"}):
            with self.subTest(params=params):
                self.assertEqual([paste(params) for paste in PASTES[5:]], [7.0, 80.0, 0.002])
                self.assertEqual([paste(params) for paste in PASTES[:5]], [None] * 5, "other fields stay")
        params = {"Schedule type": "Flow Cosmos Dynamic", "Flow Cosmos rho": "99", "Flow Cosmos sigma max": "nan",
                  "Flow Cosmos sigma min": "-1"}
        self.assertEqual([paste(params) for paste in PASTES[5:]], [15.0, 80.0, 0.0001], "coerced like the arguments")
        params = {"Schedule type": "Karras Dynamic"}   # Settings → rho, not these values
        self.assertEqual([paste(params) for paste in PASTES[5:]], [None] * 3)

    def test_images_without_these_schedulers_leave_the_fields_alone(self):
        params = {"Schedule type": "Karras", "Hires schedule type": "Use same scheduler"}
        for paste in PASTES:
            self.assertIsNone(paste(params))
        params = {"Schedule type": "Laplace"}
        self.assertEqual([paste(params) for paste in PASTES[5:]], [None] * 3)

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

    def test_flow_cosmos_values_survive(self):
        for values in ((7.0, 80.0, 0.002), (22 / 7, 1000 / 7, 0.0001 * 3), (15.0, 1.0, 0.0001), (1.0, 1000.0, 1.0)):
            with self.subTest(values=values):
                current = st.ExtraSchedulerSettings(flow_cosmos_rho=values[0], flow_cosmos_sigma_max=values[1],
                                                    flow_cosmos_sigma_min=values[2])
                params = self._round_trip(ues.infotext_items(current, {"Flow Cosmos rho7"}))
                self.assertEqual(set(params) - {"Steps", "Sampler", "Seed"},
                                 {"Flow Cosmos rho", "Flow Cosmos sigma max", "Flow Cosmos sigma min"})
                params["Schedule type"] = "Flow Cosmos rho7"
                self.assertEqual(tuple(paste(params) for paste in PASTES[5:]), values)

    def test_flow_cosmos_dynamic_values_survive(self):
        values = (22 / 7, 1000 / 7, 0.0001 * 3)
        current = st.ExtraSchedulerSettings(flow_cosmos_rho=values[0], flow_cosmos_sigma_max=values[1],
                                            flow_cosmos_sigma_min=values[2])
        written = ues.infotext_items(current, {"Flow Cosmos Dynamic"})
        self.assertEqual(written, ues.infotext_items(current, {"Flow Cosmos rho7"}))
        self.assertEqual(written, ues.infotext_items(current, {"Flow Cosmos rho7", "Flow Cosmos Dynamic"}))
        params = self._round_trip(written)
        params["Schedule type"] = "Flow Cosmos Dynamic"
        self.assertEqual(tuple(paste(params) for paste in PASTES[5:]), values)

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
        pasted = [paste(params) for paste in PASTES]
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
            # v0.33.0: appended, so the four above keep their places
            ("[Extra Schedulers (sam-extra)] Flow Cosmos rho", float),
            ("[Extra Schedulers (sam-extra)] Flow Cosmos sigma max", float),
            ("[Extra Schedulers (sam-extra)] Flow Cosmos sigma min", float),
        ])
        # Not mistaken for (and not hidden by) axes under the shorter "[Extra Schedulers]" prefix.
        self.assertFalse(any(a.label.startswith("[Extra Schedulers]") for a in axes))

    def test_flow_cosmos_axes_reach_the_generation_and_its_infotext(self):
        cosmos_rho, cosmos_max, cosmos_min = self.module.make_xyz_axes(self._xyz_module())[4:]
        p = _p("Flow Cosmos rho7")
        cosmos_rho.apply(p, 4.5, [4.5, 7.0])
        cosmos_max.apply(p, 240.0, [80.0, 240.0])
        cosmos_min.apply(p, 0.006, [0.002, 0.006])
        active = self.process(p, *FULL_ARGS[:5], 7.0, 80.0, 0.002)
        self.assertEqual((active.flow_cosmos_rho, active.flow_cosmos_sigma_max, active.flow_cosmos_sigma_min),
                         (4.5, 240.0, 0.006))
        self.assertEqual(p.extra_generation_params, {"Flow Cosmos rho": "4.5", "Flow Cosmos sigma max": "240.0",
                                                     "Flow Cosmos sigma min": "0.006"})
        # and the scheduler reads them (a flow model: the formula with these values)
        flow_model = types.SimpleNamespace(predictor=types.SimpleNamespace(prediction_type="const"))
        self.assertTrue(torch.equal(sch.flow_cosmos_rho7(12, 0.003, 1.0, inner_model=flow_model),
                                    sch._with_final_zero(sch._flow_cosmos_times(12, 4.5, 0.006, 240.0), "cpu")))

    def test_flow_cosmos_axes_reach_flow_cosmos_dynamic_too(self):
        cosmos_rho, cosmos_max, cosmos_min = self.module.make_xyz_axes(self._xyz_module())[4:]
        p = _p("Flow Cosmos Dynamic")
        cosmos_rho.apply(p, 4.5, [4.5, 7.0])
        cosmos_max.apply(p, 240.0, [80.0, 240.0])
        cosmos_min.apply(p, 0.006, [0.002, 0.006])
        active = self.process(p, *FULL_ARGS[:5], 7.0, 80.0, 0.002)
        self.assertEqual((active.flow_cosmos_rho, active.flow_cosmos_sigma_max, active.flow_cosmos_sigma_min),
                         (4.5, 240.0, 0.006))
        self.assertEqual(p.extra_generation_params, {"Flow Cosmos rho": "4.5", "Flow Cosmos sigma max": "240.0",
                                                     "Flow Cosmos sigma min": "0.006"})
        flow_model = types.SimpleNamespace(predictor=types.SimpleNamespace(prediction_type="const"))
        times, _bad = sch._flow_cosmos_dynamic_times(12, 4.5, 0.006, 240.0)
        self.assertTrue(torch.equal(sch.flow_cosmos_dynamic(12, 0.003, 1.0, inner_model=flow_model),
                                    sch._with_final_zero(times, "cpu")))

    def test_flow_cosmos_axes_check_the_slider_range_before_the_grid_runs(self):
        cosmos_rho, cosmos_max, cosmos_min = self.module.make_xyz_axes(self._xyz_module())[4:]
        cosmos_rho.confirm(None, [1.0, 3.5, 7.0, 15.0])
        cosmos_max.confirm(None, [1.0, 80.0, 240.0, 1000.0])
        cosmos_min.confirm(None, [0.0001, 0.002, 0.006, 1.0])
        for axis, bad in ((cosmos_rho, [7.0, 0.5]), (cosmos_rho, [15.5]), (cosmos_rho, [float("nan")]),
                          (cosmos_max, [0.5]), (cosmos_max, [1000.5]), (cosmos_max, [float("inf")]),
                          (cosmos_min, [0.0]), (cosmos_min, [0.00005]), (cosmos_min, [1.5]),
                          (cosmos_min, [float("nan")])):
            with self.subTest(axis=axis.label, values=bad):
                with self.assertRaisesRegex(ValueError, "outside"):
                    axis.confirm(None, bad)
        with self.assertRaisesRegex(ValueError, r"\[Extra Schedulers \(sam-extra\)\] Flow Cosmos sigma min: 2\.0 is "
                                                r"outside 0\.0001 … 1 \(the slider's range\)"):
            cosmos_min.confirm(None, [2.0])

    def test_axis_values_reach_the_generation(self):
        mu, beta, expression, sigmas = self.module.make_xyz_axes(self._xyz_module())[:4]
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
        _mu, _beta, expression, sigmas = self.module.make_xyz_axes(self._xyz_module())[:4]
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
        mu, beta, _expression, _sigmas = self.module.make_xyz_axes(self._xyz_module())[:4]
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
        self.assertEqual(len(labels), 1 + 7)   # "Nothing" + ours

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
        self.assertEqual(len(labels), 2 + 7)

    def test_missing_xyz_grid_is_fine(self):
        self.module.scripts.scripts_data[:] = []
        self.module._on_before_ui()


if __name__ == "__main__":
    unittest.main()
