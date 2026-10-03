"""Euler (SMEA) Dy CFG++ + Colorcraft — the sub-step evaluations at another resolution.

Euler Dy CFG++ evaluates the model a second time at steps 2-3 on a half-resolution latent (one pixel
of every 2x2 block) and adds that sub-step's change to the step; Euler SMEA Dy CFG++ re-steps the
whole latent at x1.25 at step 0 and the half-resolution pixels at step 1. Both mark those extra
evaluations with ``transformer_options["sam_extra_substep"]``. Neither upstream has the other feature
(Koishi-Star has no Colorcraft, Colorcraft has no Dy), so the behaviour is decided here:

**Colorcraft grades each sampler step once, at the step's own evaluation, and passes the sub-step
evaluations through ungraded** (counted in its status as "Dy/SMEA sub-steps passed through"). The
strength schedule is per step; a sub-step re-evaluates the same step, and grading it too gives the
pixels it moves that step's edit a second time — for Dy a 2x2-periodic pattern in the grade
(``test_one_grade_per_step_keeps_the_grade_uniform`` measures it). This mirrors the guidance stack,
which keeps its step-keyed state out of the sub-steps. Colorcraft never errors on the other grid either
way; the masks would simply be computed at that grid.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import contextlib
import unittest
from unittest import mock

import torch

from sam3ext.colorcraft import hook
from sam3ext.extra_samplers import euler_dy
from sam3ext.extra_samplers.common import SUBSTEP_MARKER
from sam3ext.extra_samplers.registry import LABEL_EULER_DY_CFG_PP, LABEL_EULER_SMEA_DY_CFG_PP
from tests import _colorcraft_support as ccs
from tests import _integration_support as I

Mod = I.Mod
GRADE = Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.5, "temperature": 0.3, "contrast": 0.2,
                         "vibrance": 0.3})
SAMPLERS = (
    (LABEL_EULER_DY_CFG_PP, euler_dy.sample_euler_dy_cfg_pp, ("dy", "dy")),
    (LABEL_EULER_SMEA_DY_CFG_PP, euler_dy.sample_euler_smea_dy_cfg_pp, ("smea", "dy")),
)


@contextlib.contextmanager
def graded_shapes():
    """The shape of every x0 Colorcraft's modifier chain actually runs on."""
    shapes = []
    original = hook.apply_chain

    def spy(built, work, values, *args, **kwargs):
        shapes.append(tuple(work.shape))
        return original(built, work, values, *args, **kwargs)

    with mock.patch.object(hook, "apply_chain", side_effect=spy):
        yield shapes


class DySubstepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        I.require_forge()
        cls.scripts = I.Scripts()
        cls.h = I.Harness(cls.scripts)

    def tearDown(self):
        self.scripts.teardown()

    def test_substeps_are_passed_through_and_every_step_is_graded(self):
        args = I.colorcraft_args(GRADE)
        for label, func, markers in SAMPLERS:
            with self.subTest(sampler=label), graded_shapes() as shapes:
                result = self.h.generate(func, hooks={"colorcraft": args},
                                         sampler_kwargs={"runtime": I.dy_runtime(label)})
                substeps = [call for call in result.calls if call.marker]
                self.assertEqual(tuple(call.marker for call in substeps), markers)
                self.assertTrue(all(call.shape[-2:] != (16, 16) for call in substeps))
                self.assertEqual(len(result.calls), 12)
                status = I.colorcraft_status(result.request)
                self.assertIn("10 evals, 10 edited", status)
                self.assertIn("Dy/SMEA sub-steps passed through x2", status)
                self.assertNotIn("error", status)
                self.assertEqual(len(shapes), 10)
                self.assertTrue(all(shape[-2:] == (16, 16) for shape in shapes), shapes)
                self.assertEqual(tuple(result.out.shape), (1, 16, 1, 16, 16))
                self.assertTrue(torch.isfinite(result.out).all())

    def test_with_speed_as_well(self):
        """All three: SPEED segments the Dy sampler (sub-steps at the pass's steps, on the coarse grid),
        Colorcraft grades every main step on both grids, follows the re-published sigmas and passes the
        sub-steps through."""
        args = I.colorcraft_args(GRADE)
        for label, func, _markers in SAMPLERS:
            for mode in ("transition", "respace"):
                with self.subTest(sampler=label, mode=mode), graded_shapes() as shapes:
                    result = self.h.generate(func, hooks={"speed": I.speed_args(mode=mode), "colorcraft": args},
                                             sampler_kwargs={"runtime": I.dy_runtime(label)})
                    self.scripts.teardown()
                    self.assertTrue(I.speed_status(result.request).startswith(f"base: applied ({mode})"))
                    status = I.colorcraft_status(result.request)
                    self.assertIn("10 evals, 10 edited", status)
                    self.assertIn("Dy/SMEA sub-steps passed through x2", status)
                    self.assertNotIn("other sampling run", status)
                    self.assertEqual([shape[-1] for shape in shapes], [8] * 6 + [16] * 4)
                    self.assertTrue(torch.isfinite(result.out).all())

    def test_a_marked_evaluation_returns_the_incoming_prediction(self):
        """The callback itself, with the options Forge's ``sampling_function`` builds for a sub-step
        (the UNet's options joined with the sampler's, which carry the marker)."""
        ccs.require_forge(self)
        p = ccs.make_p("krea2")
        callback = ccs.attach_ours(p, I.colorcraft_args(GRADE))
        unet = p.sd_model.forge_objects.unet
        unet.model_options["transformer_options"]["sampling_sigmas"] = ccs.flow_sigmas(10)
        sigma = float(ccs.flow_sigmas(10)[2])
        for kind in ("dy", "smea"):
            x0 = ccs.latent("krea2", 3, size=12)
            args = ccs.forge_args(x0, sigma, unet)
            args["model_options"] = I.forge_utils().join_dicts(
                unet.model_options, {"transformer_options": {SUBSTEP_MARKER: kind}})
            self.assertIs(callback(args), x0)
        x0 = ccs.latent("krea2", 3)
        self.assertIsNot(callback(ccs.forge_args(x0, sigma, unet)), x0)
        status = I.colorcraft_status(p)
        self.assertIn("1 evals, 1 edited", status)
        self.assertIn("Dy/SMEA sub-steps passed through x2", status)

    def test_the_debug_capture_is_the_steps_own_x0(self):
        """A capture at a Dy step keeps the main evaluation's full-size x0, not the sub-step's."""
        args = I.colorcraft_args(GRADE, debug=True, debug_step=2)
        result = self.h.generate(euler_dy.sample_euler_dy_cfg_pp, hooks={"colorcraft": args},
                                 sampler_kwargs={"runtime": I.dy_runtime(LABEL_EULER_DY_CFG_PP)})
        capture = getattr(result.request, hook.STATE_ATTR)["debug"]
        self.assertEqual(tuple(capture.latent.shape[-2:]), (16, 16))
        self.assertIn("debug latent captured at step 2", I.colorcraft_status(result.request))


class OneGradePerStepTests(unittest.TestCase):
    """Why the sub-steps are passed through, measured on a model with no spatial mixing.

    ``LinearVelocity`` is elementwise and linear and the grade is an exposure/temperature offset (the
    same vector added to every pixel's x0), so a grade applied once per step leaves the difference
    "graded − ungraded" exactly uniform over the image. Four steps, graded at step 2 only: Euler Dy's
    sub-step runs at step 2 (step 3 reaches sigma 0 and runs none), the latent has no grade before it,
    and the last step returns the x0 — a uniform scale — so the pattern after step 2 is what comes out.
    """

    STEPS = 4
    STEP_TWO = Mod("advanced", {"start": 2 / 3, "end": 2 / 3, "exposure": 0.8, "temperature": 0.5})

    @classmethod
    def setUpClass(cls):
        I.require_forge()
        cls.scripts = I.Scripts()
        cls.h = I.Harness(cls.scripts, dit=I.LinearVelocity())

    def tearDown(self):
        self.scripts.teardown()

    def _run(self, colorcraft):
        hooks = {"colorcraft": I.colorcraft_args(self.STEP_TWO)} if colorcraft else {}
        result = self.h.generate(euler_dy.sample_euler_dy_cfg_pp, steps=self.STEPS, hooks=hooks,
                                 sampler_kwargs={"runtime": I.dy_runtime(LABEL_EULER_DY_CFG_PP)})
        self.assertEqual([call.marker for call in result.calls], [None, None, None, "dy", None])
        return result

    @staticmethod
    def _phase_spread(delta):
        """Per channel, the largest difference between the four 2x2 phases' mean grade."""
        d = delta[:, :, 0]
        phases = torch.stack([d[..., i::2, j::2].mean(dim=(-1, -2)) for i in (0, 1) for j in (0, 1)])
        return float((phases.max(0).values - phases.min(0).values).abs().max())

    def test_one_grade_per_step_keeps_the_grade_uniform(self):
        plain = self._run(colorcraft=False).out
        graded = self._run(colorcraft=True)
        self.assertIn("Dy/SMEA sub-steps passed through x1", I.colorcraft_status(graded.request))
        delta = graded.out - plain
        magnitude = float(delta.abs().mean())
        self.assertGreater(magnitude, 1e-2, "the grade must show")
        self.assertLess(float((delta - delta.mean(dim=(-1, -2), keepdim=True)).abs().max()), 1e-5 * max(1.0, magnitude))
        self.assertLess(self._phase_spread(delta), 1e-5)

        # Grading the sub-step too: the pixels it moves get step 2's edit twice.
        self.scripts.teardown()
        with mock.patch.object(hook, "_extra_sampler_substep", return_value=False):
            doubled = self._run(colorcraft=True)
        delta2 = doubled.out - plain
        d = delta2[:, :, 0]
        odd = d[..., 1::2, 1::2].mean(dim=(-1, -2))
        even = d[..., 0::2, 0::2].mean(dim=(-1, -2))
        torch.testing.assert_close(odd - even, even, rtol=1e-3, atol=1e-5)   # odd = 2 x even
        self.assertGreater(self._phase_spread(delta2), 0.5 * magnitude)


if __name__ == "__main__":
    unittest.main()
