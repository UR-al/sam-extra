"""``sam3ext.extra_samplers.substep_guard``: which requests the Dy/SMEA sub-steps cannot run on, and the
status infotext. The end-to-end runs (Forge's ``KDiffusionSampler.sample`` with Forge's real Spectrum
forecaster, Wan / PiD / extra-concat inputs) are in ``test_extra_samplers_forge_path.py``."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


def _fixtures():
    name = "_extra_samplers_fixtures"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fx = _fixtures()

from sam3ext.extra_samplers import params as sampler_params  # noqa: E402
from sam3ext.extra_samplers import registry, substep_guard  # noqa: E402


def _request(unet=None):
    return types.SimpleNamespace(
        extra_generation_params={},
        sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=unet)),
    )


def _dynamic_args(**values):
    dynamic_args = types.SimpleNamespace(ref_latents=[], concat_latent=None, lq_latent=[None, None], pid=False)
    dynamic_args.__dict__.update(values)
    return dynamic_args


def spectrum_unet_wrapper(model_function, kwargs):   # a stand-in with Spectrum's function name
    return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])


class SpectrumDetectionTests(unittest.TestCase):
    def test_forges_real_spectrum_wrapper_is_recognised(self):
        forecaster = fx.forge_spectrum_forecaster()
        unet = forecaster.SpectrumNode.patch(fx.UnetPatcherStandIn(), 20, *fx.SPECTRUM_DEFAULTS)
        wrapper = unet.model_options["model_function_wrapper"]
        self.assertEqual(wrapper.__module__, "lib_spectrum.forecaster")   # Forge: from lib_spectrum.forecaster import
        self.assertIn("spectrum_unet_wrapper", wrapper.__qualname__)
        self.assertTrue(substep_guard.spectrum_active(unet))

    def test_the_name_or_the_module_is_enough_and_other_wrappers_are_not_spectrum(self):
        self.assertTrue(substep_guard.spectrum_active(fx.UnetPatcherStandIn(
            {"model_function_wrapper": spectrum_unet_wrapper})))
        by_module = types.FunctionType(spectrum_unet_wrapper.__code__, {"__name__": "lib_spectrum.forecaster"}, "wrap")
        by_module.__qualname__ = "SpectrumNode.patch.<locals>.wrap"
        self.assertEqual(by_module.__module__, "lib_spectrum.forecaster")
        self.assertTrue(substep_guard.spectrum_active(fx.UnetPatcherStandIn({"model_function_wrapper": by_module})))

        def _model_wrapper(apply_model, w):   # e.g. Anima Guidance's PAG wrapper
            return apply_model(w["input"], w["timestep"], **w["c"])

        for options in ({"model_function_wrapper": _model_wrapper}, {"transformer_options": {}}, None):
            with self.subTest(options=options):
                self.assertFalse(substep_guard.spectrum_active(fx.UnetPatcherStandIn(options)))
        self.assertFalse(substep_guard.spectrum_active(None))


class BlockingReasonTests(unittest.TestCase):
    def test_nothing_blocks_an_ordinary_request(self):
        self.assertEqual(substep_guard.blocking_reasons(_request(fx.UnetPatcherStandIn()), _dynamic_args()), [])
        self.assertEqual(substep_guard.blocking_reasons(types.SimpleNamespace()), [])   # no Forge objects at all

    def test_each_reason(self):
        wan_i2v = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(in_dim=36, out_dim=16))
        wan_t2v = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(in_dim=16, out_dim=16))
        concat = torch.zeros(1, 20, 1, 8, 8)
        cases = [
            ("spectrum", fx.UnetPatcherStandIn({"model_function_wrapper": spectrum_unet_wrapper}), _dynamic_args(),
             [substep_guard.REASON_SPECTRUM]),
            ("wan i2v", fx.UnetPatcherStandIn(model=wan_i2v), _dynamic_args(concat_latent=concat),
             [substep_guard.REASON_WAN]),
            ("wan, model unknown", fx.UnetPatcherStandIn(), _dynamic_args(concat_latent=concat),
             [substep_guard.REASON_WAN]),
            ("wan t2v, leftover concat", fx.UnetPatcherStandIn(model=wan_t2v), _dynamic_args(concat_latent=concat), []),
            ("pid flag", fx.UnetPatcherStandIn(), _dynamic_args(pid=True), [substep_guard.REASON_PID]),
            ("pid lq latent", fx.UnetPatcherStandIn(), _dynamic_args(lq_latent=[torch.zeros(1, 4, 8, 8), None]),
             [substep_guard.REASON_PID]),
            ("extra concat", fx.UnetPatcherStandIn(extra_concat_condition=torch.zeros(1, 5, 8, 8)), _dynamic_args(),
             [substep_guard.REASON_EXTRA_CONCAT]),
            ("all", fx.UnetPatcherStandIn({"model_function_wrapper": spectrum_unet_wrapper}, model=wan_i2v,
                                          extra_concat_condition=torch.zeros(1)),
             _dynamic_args(concat_latent=concat, pid=True),
             [substep_guard.REASON_SPECTRUM, substep_guard.REASON_WAN, substep_guard.REASON_PID,
              substep_guard.REASON_EXTRA_CONCAT]),
        ]
        for name, unet, dynamic_args, expected in cases:
            with self.subTest(name):
                self.assertEqual(substep_guard.blocking_reasons(_request(unet), dynamic_args), expected)

    def test_forges_dynamic_args_are_read_when_none_is_given(self):
        backend = types.ModuleType("backend")
        backend.__path__ = []
        args_module = types.ModuleType("backend.args")
        args_module.dynamic_args = _dynamic_args(pid=True)
        with fx.stub_modules({"backend": backend, "backend.args": args_module}):
            self.assertEqual(substep_guard.blocking_reasons(_request(fx.UnetPatcherStandIn())),
                             [substep_guard.REASON_PID])


class StatusTests(unittest.TestCase):
    def test_the_status_infotext_collects_the_reasons_of_every_pass(self):
        p = _request()
        self.assertIsNone(substep_guard.record_status(p, []))
        self.assertEqual(p.extra_generation_params, {})
        substep_guard.record_status(p, [substep_guard.REASON_SPECTRUM])
        self.assertEqual(p.extra_generation_params,
                         {"Extra Samplers status": "dy sub-steps skipped (Spectrum)"})
        substep_guard.record_status(p, [substep_guard.REASON_SPECTRUM, substep_guard.REASON_PID])
        self.assertEqual(p.extra_generation_params["Extra Samplers status"],
                         "dy sub-steps skipped (Spectrum + PiD lq_latent)")
        # Forge's infotext quotes values with "," or ":" - the status has neither
        self.assertNotIn(",", p.extra_generation_params["Extra Samplers status"])
        self.assertNotIn(":", p.extra_generation_params["Extra Samplers status"])

    def test_a_request_without_infotext_params_is_tolerated(self):
        p = types.SimpleNamespace()
        self.assertEqual(substep_guard.record_status(p, [substep_guard.REASON_PID]), "dy sub-steps skipped (PiD lq_latent)")

    def test_every_reason_has_an_explanation(self):
        for reason in (substep_guard.REASON_SPECTRUM, substep_guard.REASON_WAN, substep_guard.REASON_PID,
                       substep_guard.REASON_EXTRA_CONCAT):
            self.assertNotEqual(substep_guard.explain(reason), reason)


class RuntimeKeywordTests(unittest.TestCase):
    def setUp(self):
        registry._LOGGED.clear()
        self.addCleanup(registry._LOGGED.clear)
        sampler_params.reset_active()
        self.addCleanup(sampler_params.reset_active)
        self.logged = []
        self.enterContext(mock.patch.object(registry, "_log", self.logged.append))
        self.specs = {spec.label: spec for spec in registry.SPECS}

    def test_the_dy_samplers_get_substeps_false_and_the_status(self):
        dy_labels = (registry.LABEL_EULER_DY_CFG_PP, registry.LABEL_EULER_SMEA_DY_CFG_PP,
                     registry.LABEL_EULER_DY, registry.LABEL_EULER_SMEA_DY)
        self.assertEqual(sorted(dy_labels), sorted(spec.label for spec in registry.SPECS if spec.kind == "dy"))
        for label in dy_labels:
            with self.subTest(label=label):
                p = _request(fx.UnetPatcherStandIn({"model_function_wrapper": spectrum_unet_wrapper}))
                kwargs = registry.runtime_kwargs(self.specs[label], p)
                self.assertIs(kwargs["substeps"], False)
                self.assertEqual(p.extra_generation_params, {"Extra Samplers status": "dy sub-steps skipped (Spectrum)"})
        self.assertEqual(len(self.logged), 4)   # once per sampler and reason
        for label, line in zip(dy_labels, self.logged):
            with self.subTest(label=label):
                self.assertTrue(line.startswith(label + ": sub-steps skipped"))
                steps = "plain CFG++ Euler steps" if self.specs[label].cfg_pp else "plain Euler steps"
                self.assertIn(f"it runs {steps} instead", line)
        registry.runtime_kwargs(self.specs[registry.LABEL_EULER_DY_CFG_PP],
                                _request(fx.UnetPatcherStandIn({"model_function_wrapper": spectrum_unet_wrapper})))
        self.assertEqual(len(self.logged), 4)

    def test_ordinary_requests_and_other_samplers_are_unchanged(self):
        kwargs = registry.runtime_kwargs(self.specs[registry.LABEL_EULER_DY_CFG_PP], _request(fx.UnetPatcherStandIn()))
        self.assertNotIn("substeps", kwargs)
        p = _request(fx.UnetPatcherStandIn({"model_function_wrapper": spectrum_unet_wrapper}))
        for label in (spec.label for spec in registry.SPECS if spec.kind != "dy"):
            self.assertNotIn("substeps", registry.runtime_kwargs(self.specs[label], p))
        self.assertNotIn("Extra Samplers status", p.extra_generation_params)
        self.assertEqual(self.logged, [])


if __name__ == "__main__":
    unittest.main()
