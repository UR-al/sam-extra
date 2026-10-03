"""Colorcraft on the existing post-CFG stack: Optimal Scale, the guidance stack (PAG + DCW), Skimmed CFG,
Detail Daemon — and SPEED around them.

* **Order.** Forge runs ``process_before_every_sampling`` in script load order (scripts/ sorted), every
  script clones the UNet it finds and appends (Skimmed CFG prepends): Skimmed CFG, Optimal Scale, the
  guidance stack's ``_post_cfg``, Colorcraft — Colorcraft last, so it grades the final guided x0.
* **Optimal Scale's skip rules.** It only applies to an untouched linear CFG result. The guidance stack
  and Colorcraft come after it, so it applies on every step with them on; Skimmed CFG (prepended,
  rewrites the prediction pair) makes it step aside with its reason. Were Colorcraft before it, the
  graded x0 would read as a prior correction and Optimal Scale would never apply.
* **The pre-DD sigma.** Detail Daemon lowers the sigma handed to the model; Colorcraft looks its
  schedule up by the sampler's own sigma (its setting, default on), noted by Detail Daemon for that
  forward, so every step reads its own value — also at SPEED's aligned transition sigma.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import contextlib
import unittest
from unittest import mock

import torch

from sam3ext.colorcraft import engine, hook, spec
from tests import _integration_support as I

GRADE = I.Mod("advanced", {"start": 0.0, "end": 1.0, "advanced": True, "exponent": 1.0, "exposure": 0.4,
                           "contrast": 0.2, "temperature": 0.2})
PAG_DCW = {"anima_safe_pag_enable": True, "anima_safe_pag_blocks": "3", "anima_safe_pag_end": 1.0,
           "anima_guidance_dcw_enable": True}
OPTIMAL = [True, 0.5, 0.0, 1.0]
SKIM = [True, 3.0, False, False, 0.0, 1.0, 0.0]


def owner(fn):
    if hook.is_owned(fn):
        return "colorcraft"
    if getattr(fn, "_sam_extra_optimal_scale_owner", None):
        return "optimal-scale"
    if getattr(fn, "_sam_extra_post_cfg_owner", None) == "sam-extra/anima-skimmed-cfg":
        return "skimmed-cfg"
    if getattr(fn, "__name__", "") == "_post_cfg":
        return "guidance-stack"
    return getattr(fn, "__name__", repr(fn))


def expected_schedule(steps=10):
    entries = spec.build_chain(spec.config_from_args(I.colorcraft_args(GRADE)), False).entries
    return [float(v) for v in engine.build_entries(entries, steps)[0][0]]


@contextlib.contextmanager
def colorcraft_sigmas():
    """The sigma every Colorcraft schedule lookup uses."""
    seen = []
    original = hook.sigma_to_value

    def spy(sigma, sigmas, sched):
        value = original(sigma, sigmas, sched)
        seen.append((float(sigma), value))
        return value

    with mock.patch.object(hook, "sigma_to_value", side_effect=spy):
        yield seen


def pre_dd_option(value):
    """Colorcraft's Settings switch "look the schedule up by the pre-Detail-Daemon sigma"."""
    def option(name, default):
        return value if name == hook.OPT_PRE_DD else default

    return mock.patch.object(hook, "_option", side_effect=option)


class PostCfgStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        I.require_forge()
        cls.scripts = I.Scripts()
        cls.h = I.Harness(cls.scripts)
        cls.ks = I.forge_k()

    def tearDown(self):
        self.scripts.teardown()

    def _run(self, **hooks):
        return self.h.generate(self.ks.sample_euler, hooks=hooks)

    def test_colorcraft_is_the_last_post_cfg_function(self):
        cases = (
            (dict(optimal=OPTIMAL, dd=I.dd_args(), pag=self.scripts.pag_args(**PAG_DCW)),
             ["optimal-scale", "guidance-stack", "colorcraft"]),
            (dict(optimal=OPTIMAL, dd=I.dd_args(), pag=self.scripts.pag_args(**PAG_DCW), skim=SKIM,
                  speed=I.speed_args()),
             ["skimmed-cfg", "optimal-scale", "guidance-stack", "colorcraft"]),
        )
        for hooks, order in cases:
            with self.subTest(order=order):
                result = self._run(colorcraft=I.colorcraft_args(GRADE), **hooks)
                self.scripts.teardown()
                self.assertEqual([owner(fn) for fn in result.post_cfg], order)
                self.assertTrue(torch.isfinite(result.out).all())
                self.assertNotIn("error", I.colorcraft_status(result.request))

    def test_optimal_scale_applies_under_the_stack_and_colorcraft(self):
        result = self._run(optimal=OPTIMAL, dd=I.dd_args(), pag=self.scripts.pag_args(**PAG_DCW),
                           colorcraft=I.colorcraft_args(GRADE))
        params = result.request.extra_generation_params
        self.assertEqual(params["Anima Optimal Scale status"], "applied=10; skipped=0")
        self.assertIn("10 evals, 8 edited", I.colorcraft_status(result.request))
        self.assertIn("Anima DCW", params)

    def test_optimal_scale_steps_aside_for_skimmed_cfg(self):
        result = self._run(optimal=OPTIMAL, skim=SKIM, colorcraft=I.colorcraft_args(GRADE))
        status = result.request.extra_generation_params["Anima Optimal Scale status"]
        self.assertEqual(status, "applied=0; skipped=10; last_skip=Skimmed CFG modifies the prediction pair")
        self.assertIn("10 evals, 8 edited", I.colorcraft_status(result.request))

    def test_graded_x0_ahead_of_optimal_scale_would_read_as_a_prior_correction(self):
        """Why last matters: Colorcraft attached before Optimal Scale (not Forge's order) hands it a
        graded x0, and Optimal Scale skips every graded step."""
        every_step = I.Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.4})   # a plateau: all 10 graded
        request = self.h.request()
        request.sampler = I.KDiffusionSampler(self.ks.sample_euler)
        with self.h.forge_environment(request):
            x = request.rng.next()
            self.h.run_hooks(request, x=x, colorcraft=I.colorcraft_args(every_step))
            self.h.run_hooks(request, x=x, optimal=OPTIMAL)
            self.h.sample(request, self.ks.sample_euler, x, I.flow_sigmas(10))
        self.assertIn("10 evals, 10 edited", I.colorcraft_status(request))
        status = request.extra_generation_params["Anima Optimal Scale status"]
        self.assertEqual(status, "applied=0; skipped=10; last_skip=prior post-CFG correction (APG/PAG/DCW/etc.)")

    def test_colorcraft_reads_the_pre_dd_sigma(self):
        expected = expected_schedule()
        for pre_dd in (True, False):
            with self.subTest(pre_dd=pre_dd), pre_dd_option(pre_dd), colorcraft_sigmas() as lookups:
                result = self._run(optimal=OPTIMAL, dd=I.dd_args(amount=0.4), pag=self.scripts.pag_args(**PAG_DCW),
                                   colorcraft=I.colorcraft_args(GRADE))
                self.scripts.teardown()
                calls = result.calls
                self.assertTrue(any(abs(c.sigma - c.sigma_model) > 1e-4 for c in calls), "Detail Daemon scaled")
                self.assertEqual(len(lookups), len(calls))
                params = result.request.extra_generation_params
                self.assertEqual(params[hook.INFOTEXT_PRE_DD], str(pre_dd))
                for step, (call, (sigma, value)) in enumerate(zip(calls, lookups)):
                    self.assertAlmostEqual(sigma, call.sigma if pre_dd else call.sigma_model, places=6)
                    if pre_dd:
                        self.assertAlmostEqual(value, expected[step], places=12)

    def test_pre_dd_sigma_at_speeds_aligned_sigma(self):
        """SPEED's transition step runs at the aligned sigma; Detail Daemon scales it for the model and
        notes it; Colorcraft reads the note and finds it on the re-published list (its own step)."""
        expected = expected_schedule()
        for mode in ("transition", "respace"):
            with self.subTest(mode=mode), pre_dd_option(True), colorcraft_sigmas() as lookups:
                result = self._run(optimal=OPTIMAL, dd=I.dd_args(amount=0.4), pag=self.scripts.pag_args(**PAG_DCW),
                                   speed=I.speed_args(mode=mode), colorcraft=I.colorcraft_args(GRADE))
                self.scripts.teardown()
                self.assertIn(f"applied ({mode})", I.speed_status(result.request))
                first_full = next(i for i, c in enumerate(result.calls) if c.shape[-1] == 16)
                call = result.calls[first_full]
                self.assertGreater(abs(call.sigma - call.sigma_model), 1e-4)
                for step, (sigma, value) in enumerate(lookups):
                    self.assertAlmostEqual(sigma, result.calls[step].sigma, places=6)
                    self.assertAlmostEqual(value, expected[step], places=12)
                params = result.request.extra_generation_params
                self.assertEqual(params["Anima Optimal Scale status"], "applied=10; skipped=0")
                self.assertNotIn("other sampling run", I.colorcraft_status(result.request))


if __name__ == "__main__":
    unittest.main()
