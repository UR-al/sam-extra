"""Anima SPEED + Extra Schedulers — every extra schedule through SPEED's two plans, and the error paths.

The schedules come from Forge's real ``KDiffusionSampler.get_sigmas`` (cut out of the checkout) with
this extension's schedulers registered the way the script registers them, for an Anima-like flow model
(``sigma_max`` 1, shift 3). Each one then samples through SPEED in transition mode (only the transition
step's sigma is patched to the aligned one) and respace mode (the remaining sigmas are re-spaced), with
Detail Daemon and Colorcraft following the published schedule:

* SPEED applies, or steps aside with a reason in ``Anima SPEED status`` (a schedule above 1 is not a
  flow schedule; one whose hand-off sigma is never reached has no transition) — never a crash;
* the list SPEED publishes during the run differs from Forge's exactly where the plan says, and
  Forge's list is put back afterwards;
* Colorcraft reads every step's own schedule value (on the patched list) and grades the whole run;
* ``KarrasDynamicError`` (a rising Karras Dynamic schedule at a low rho) and the custom scheduler's
  ``CustomSchedulerError`` stop the generation in ``get_sigmas`` — before SPEED's wrapper runs — with
  the scheduler's own message; SPEED has hijacked nothing and published nothing, and the next
  generation samples normally.

Nothing here depends on how Laplace treats the part of its curve above ``sigma_max`` on a flow model
or on whether a schedule may repeat a sigma: the checks hold for any falling schedule the scheduler
returns (a repeated sigma is only skipped by the per-step value check).
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import types
import unittest
from unittest import mock

import numpy as np
import torch

from sam3ext.colorcraft import engine, hook, spec
from sam3ext.extra_samplers import registry as sampler_registry
from sam3ext.extra_schedulers import registry, schedulers, settings as es_settings, sigma_list
from sam3ext.speed import schedule as speed_schedule
from sam3ext.speed import spectral
from tests import _extra_schedulers_support as ess
from tests import _integration_support as I

MODES = ("transition", "respace")
GRADE = I.Mod("advanced", {"start": 0.0, "end": 1.0, "advanced": True, "exponent": 1.0, "exposure": 0.4,
                           "temperature": 0.2})

# (label, settings for the schedulers that read the accordion)
SCHEDULES = (
    ("Cosine", {}),
    ("CosineExponential blend", {}),
    ("Phi", {}),
    ("Laplace", {}),
    ("Laplace", {"laplace_mu": -1.5, "laplace_beta": 0.8}),
    ("Karras Dynamic", {}),
    ("custom", {"custom_mode": es_settings.MODE_EXPRESSION, "custom_expression": "M * (m / M) ** x"}),
    ("custom", {"custom_mode": es_settings.MODE_EXPRESSION, "custom_expression": "m + (M - m) * (1 - x) ** 2"}),
    ("custom", {"custom_mode": es_settings.MODE_SIGMAS, "custom_sigmas": "[1.0, 0.6, 0.25, 0.1, 0.0]"}),
    # not 1.0 … 0.0, so taken as sigmas as they are (ten values: no interpolation)
    ("custom", {"custom_mode": es_settings.MODE_SIGMAS,
                "custom_sigmas": "0.95, 0.9, 0.75, 0.7, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01"}),
    ("React Cosinusoidal DynSF", {}),
    # the flow list (the model wrap below has a flow predictor): 80/81 … 0.002/1.002
    ("Flow Cosmos rho7", {}),
    # its accordion values (D19): ρ 5 on σ̃ 0.006 … 240 (Anima's shift 3 on the default range): 240/241 … 0.006/1.006
    ("Flow Cosmos rho7", {"flow_cosmos_rho": 5.0, "flow_cosmos_sigma_max": 240.0, "flow_cosmos_sigma_min": 0.006}),
    # Flow Cosmos Dynamic (D20): Karras Dynamic's per-step ρ on the same σ̃ range, from the same three values
    ("Flow Cosmos Dynamic", {}),
    ("Flow Cosmos Dynamic", {"flow_cosmos_rho": 5.0, "flow_cosmos_sigma_max": 240.0, "flow_cosmos_sigma_min": 0.006}),
)


def anima_model_sigmas():
    """Forge's discrete-flow model sigmas for shift 3 (``time_snr_shift(3, t)``, t = 1/1000 … 1)."""
    t = torch.linspace(1.0 / 1000, 1.0, 1000, dtype=torch.float64)
    return (3.0 * t / (1.0 + 2.0 * t)).to(torch.float32)


class ForgeSchedules:
    """``KDiffusionSampler.get_sigmas`` from the checkout, over a stub ``sd_schedulers`` holding Forge's
    three named defaults plus this extension's schedulers (``registry.register_schedulers``). The model wrap
    carries a flow predictor (``prediction_type`` "const"), which the two Flow Cosmos schedulers read
    (need_inner_model)."""

    def __init__(self):
        self.sd_schedulers = ess.stub_sd_schedulers()
        report = registry.register_schedulers(
            self.sd_schedulers, sd_samplers=types.SimpleNamespace(),
            sd_samplers_kdiffusion=types.SimpleNamespace(k_diffusion_scheduler={}), hidden_labels=(),
        )
        assert not report.skipped and not report.error, report
        self.opts = types.SimpleNamespace(always_discard_next_to_last_sigma=False, sigma_min=0, sigma_max=0, rho=0,
                                          beta_dist_alpha=0.6, beta_dist_beta=0.6)
        namespace = {"opts": self.opts, "sd_schedulers": self.sd_schedulers, "torch": torch,
                     "devices": types.SimpleNamespace(cpu=torch.device("cpu"))}
        self.get_sigmas = ess.forge_function("modules/sd_samplers_kdiffusion.py", "KDiffusionSampler.get_sigmas",
                                             namespace)
        self.interp = ess.forge_function("modules/sd_schedulers.py", "_loglinear_interp", {"np": np})

    def __call__(self, label, steps, request, *, options=None, values=None):
        es_settings.set_active(es_settings.coerce(values or {}))
        sampler = types.SimpleNamespace(config=types.SimpleNamespace(options=dict(options or {})),
                                        model_wrap=types.SimpleNamespace(
                                            sigmas=anima_model_sigmas(),
                                            predictor=types.SimpleNamespace(prediction_type="const")))
        request.scheduler = label
        request.hr_scheduler = None
        request.sampler_noise_scheduler_override = None
        with mock.patch.object(sigma_list, "forge_loglinear_interp", return_value=self.interp):
            return self.get_sigmas(sampler, request, steps)


class SpeedSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        I.require_forge()
        if ess.forge_root() is None:
            raise unittest.SkipTest("Forge's modules/sd_schedulers.py is not in the checkout")
        cls.scripts = I.Scripts()
        cls.h = I.Harness(cls.scripts)
        cls.ks = I.forge_k()
        cls.schedules = ForgeSchedules()

    def tearDown(self):
        self.scripts.teardown()
        es_settings.reset()

    def _generate(self, label, values, mode, *, func=None, options=None, brownian=False, colorcraft=True):
        request = self.h.request()
        sigmas = self.schedules(label, 10, request, options=options, values=values)
        hooks = {"speed": I.speed_args(mode=mode), "dd": I.dd_args()}
        if colorcraft:
            hooks["colorcraft"] = I.colorcraft_args(GRADE)
        result = self.h.generate(func or self.ks.sample_euler, request=request, sigmas=sigmas, hooks=hooks,
                                 brownian=brownian)
        return result, sigmas

    def _check_published(self, result, sigmas, mode):
        """What SPEED published at each model call, against its own plan."""
        plan = speed_schedule.build_plan(
            mode=mode, transform="dct", sigmas=sigmas, full_grid=(16, 16), scales=(0.5, 1.0),
            threshold="manual", manual_sigmas=(0.7,),
        )
        step = plan.transitions[0].step
        for call in result.calls:
            published = call.sampling_sigmas
            if call.shape[-1] == 8:
                self.assertIs(published, sigmas)                      # Forge's own list while coarse
                continue
            self.assertIsNot(published, sigmas)
            changed = (published != sigmas).nonzero().flatten().tolist()
            if mode == "transition":
                self.assertEqual(changed, [step])
                aligned = spectral.align_timestep(float(sigmas[step]), 2.0)     # scales 0.5 -> 1.0: r = 2
                self.assertAlmostEqual(float(published[step]), float(aligned), places=5)
            else:
                self.assertEqual(changed[0], step)
                self.assertTrue(set(changed) <= set(range(step, len(sigmas) - 1)))
        unet = result.request.sd_model.forge_objects.unet
        self.assertIs(unet.model_options["transformer_options"]["sampling_sigmas"], sigmas, "restored")

    def test_every_extra_schedule_through_both_plans(self):
        args = I.colorcraft_args(GRADE)
        entries = spec.build_chain(spec.config_from_args(args), False).entries
        expected_values = [float(v) for v in engine.build_entries(entries, 10)[0][0]]
        for label, values in SCHEDULES:
            for mode in MODES:
                with self.subTest(schedule=label, settings=values, mode=mode):
                    lookups = []
                    original = hook.sigma_to_value

                    def spy(sigma, walked, sched, _original=original):
                        value = _original(sigma, walked, sched)
                        lookups.append((float(sigma), [float(s) for s in walked], value))
                        return value

                    with mock.patch.object(hook, "sigma_to_value", side_effect=spy):
                        result, sigmas = self._generate(label, values, mode)
                    self.scripts.teardown()
                    self.assertEqual(len(sigmas), 11)
                    status = I.speed_status(result.request)
                    self.assertTrue(status.startswith(f"base: applied ({mode})"), f"{label}: {status}")
                    self.assertEqual(result.seen, list(range(10)))
                    self.assertEqual(tuple(result.out.shape), (1, 16, 1, 16, 16))
                    self.assertTrue(torch.isfinite(result.out).all())
                    self._check_published(result, sigmas, mode)
                    # Colorcraft: one lookup per step, on the list the sampler walks; a sigma that appears
                    # once in it reads its own step's value (a repeated one reads its first step's)
                    self.assertEqual(len(lookups), 10)
                    for step, (sigma, walked, value) in enumerate(lookups):
                        self.assertAlmostEqual(walked[step], sigma, places=5)
                        if sum(abs(s - sigma) < 1e-7 for s in walked[:-1]) == 1:
                            self.assertAlmostEqual(value, expected_values[step], places=9)
                    cc = I.colorcraft_status(result.request)
                    self.assertIn("10 evals", cc)
                    self.assertNotIn("other sampling run", cc)
                    self.assertNotIn("error", cc)

    def test_laplace_on_a_flow_model(self):
        """sigma_max 1 (Anima): whatever Laplace makes of its curve above 1, the schedule stays a flow
        schedule SPEED can segment."""
        request = self.h.request()
        sigmas = self.schedules("Laplace", 10, request)
        self.assertLessEqual(float(sigmas.max()), 1.0)
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertTrue(bool((sigmas[1:] <= sigmas[:-1]).all()))
        for mode in MODES:
            with self.subTest(mode=mode):
                result, _ = self._generate("Laplace", {}, mode)
                self.scripts.teardown()
                self.assertIn(f"applied ({mode})", I.speed_status(result.request))
                self.assertTrue(torch.isfinite(result.out).all())

    def test_dpmpp_4m_sde_discards_the_penultimate_sigma_before_speed(self):
        """Its sampler option ``discard_next_to_last_sigma`` (Forge drops the penultimate sigma in
        ``get_sigmas``): SPEED plans on the list the sampler gets."""
        spec4m = next(s for s in sampler_registry.SPECS if s.label == sampler_registry.LABEL_DPMPP_4M_SDE)
        for label in ("Karras Dynamic", "custom"):
            for mode in MODES:
                with self.subTest(schedule=label, mode=mode):
                    result, sigmas = self._generate(
                        label, {}, mode, func=spec4m.func, options=spec4m.options, brownian=True, colorcraft=False,
                    )
                    self.scripts.teardown()
                    self.assertEqual(len(sigmas), 11)
                    self.assertIn(f"applied ({mode})", I.speed_status(result.request))
                    self.assertTrue(torch.isfinite(result.out).all())

    # -- the stop paths --------------------------------------------------------------------------

    def _stopped_in_get_sigmas(self, label, values, error, rho=0):
        """Forge's ``sample``: the hooks have run (SPEED wrapped the sampler), then ``get_sigmas`` raises."""
        request = self.h.request()
        request.sampler = I.KDiffusionSampler(self.ks.sample_euler)
        self.schedules.opts.rho = rho
        try:
            with self.h.forge_environment(request) as ks:
                hijack = ks.torch
                x = request.rng.next()
                self.h.run_hooks(request, x=x, speed=I.speed_args(), dd=I.dd_args(),
                                 colorcraft=I.colorcraft_args(GRADE))
                self.assertTrue(getattr(request.sampler.func, "_sam3_speed_wrapped", False))
                with self.assertRaises(error) as caught:
                    self.schedules(label, 20, request, values=values)
                self.assertIs(ks.torch, hijack, "SPEED never swapped the noise hijack")
        finally:
            self.schedules.opts.rho = 0
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(I.speed_status(request), "base: pending - sampler wrapped, waiting for sampling")
        transformer = request.sd_model.forge_objects.unet.model_options["transformer_options"]
        self.assertNotIn("sampling_sigmas", transformer)
        self.scripts.teardown()
        # the next generation samples normally
        result, _ = self._generate("Cosine", {}, "transition")
        self.assertIn("applied (transition)", I.speed_status(result.request))
        return caught.exception

    def test_karras_dynamic_error_stops_before_speed(self):
        exc = self._stopped_in_get_sigmas("Karras Dynamic", {}, schedulers.KarrasDynamicError, rho=3.0)
        self.assertIn("Raise rho to about 4 or more", str(exc))

    def test_custom_expression_error_stops_before_speed(self):
        exc = self._stopped_in_get_sigmas(
            "custom", {"custom_mode": es_settings.MODE_EXPRESSION, "custom_expression": "M - 2 * M * x"},
            schedulers.CustomSchedulerError,
        )
        self.assertIn("custom scheduler's expression cannot be used", str(exc))
        exc = self._stopped_in_get_sigmas(       # the safe expression language refuses it
            "custom", {"custom_mode": es_settings.MODE_EXPRESSION, "custom_expression": "M * __import__('os')"},
            schedulers.CustomSchedulerError,
        )
        self.assertIn("custom scheduler's expression cannot be used", str(exc))
        exc = self._stopped_in_get_sigmas(
            "custom", {"custom_mode": es_settings.MODE_SIGMAS, "custom_sigmas": "1.0, nan, 0.0"},
            schedulers.CustomSchedulerError,
        )
        self.assertIn("sigma list cannot be used", str(exc))

    def test_schedules_speed_cannot_use_are_skipped_with_a_reason(self):
        cases = (
            # above 1: not a flow schedule
            ({"custom_mode": es_settings.MODE_EXPRESSION, "custom_expression": "1.5 * M * (m / M) ** x"},
             "is not a flow schedule"),
            # never at or below the hand-off sigma 0.7 before the last step
            ({"custom_mode": es_settings.MODE_SIGMAS, "custom_sigmas": "0.99, 0.95, 0.9, 0.85, 0.8, 0.0"},
             "is never reached in this schedule"),
            # starts below it: nothing to sample coarse
            ({"custom_mode": es_settings.MODE_EXPRESSION, "custom_expression": "0.6 * M * (m / M) ** x"},
             "no coarse steps"),
        )
        for values, reason in cases:
            for mode in MODES:
                with self.subTest(values=values, mode=mode):
                    result, sigmas = self._generate("custom", values, mode)
                    self.scripts.teardown()
                    status = I.speed_status(result.request)
                    self.assertIn("base: skipped", status)
                    self.assertIn(reason, status)
                    self.assertEqual({c.shape[-2:] for c in result.calls}, {(16, 16)})   # plain sampling
                    self.assertTrue(torch.isfinite(result.out).all())
                    self.assertIn("10 evals", I.colorcraft_status(result.request))


if __name__ == "__main__":
    unittest.main()
