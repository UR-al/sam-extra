"""Anima SPEED + Colorcraft on one pass — Forge's real CFG, the Anima DiT, both scripts' hooks (CPU).

SPEED samples the first steps on a DCT-truncated coarse latent, expands it at the transition and
re-publishes the patched schedule as ``sampling_sigmas`` (transition mode: the aligned sigma of the
transition step; respace mode: the whole re-spaced tail). Colorcraft is the last post-CFG function and
looks its strength schedule up by sigma on that list. What the pair must do (``tests/_integration_support``
has the harness):

* grade the coarse x0 of the first segment and the full-size x0 after the expansion — no crash, no
  shape mismatch; masks are computed from the latent being graded, colour anchors broadcast over any
  grid, and the mask blur keeps its image-pixel size on the coarse grid (``hook.grid_downscale``);
* keep grading after the transition. Fixed here: Colorcraft keyed its run on the ``sampling_sigmas``
  object, took SPEED's re-published copy for another sampling run and passed the whole full-size tail
  through ungraded (status "other sampling run passed through"). SPEED now tags the copy
  (``sam3ext.guidance.sigmas.mark_republished``) and Colorcraft follows it;
* read every step's own schedule value — the transition step at its aligned sigma and the re-spaced
  tail included — by looking the sigma up on the list the sampler walks (the patched one, as Detail
  Daemon, DAVE, MG/HiGS and HiFlow already do);
* give a batch image equal to the same seed alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import contextlib
import types
import unittest
from unittest import mock

import torch

from sam3ext.colorcraft import engine, hook, masking, schedule, spec
from sam3ext.speed import forge_host as fh
from tests import _integration_support as I

Mod = I.Mod
# A ramp (exponent 1), so a lookup one step off reads a different value; contrast and colour shift
# need the VAE-encoded anchors.
RAMP = {"start": 0.0, "end": 1.0, "advanced": True, "exponent": 1.0}
GRADE = Mod("advanced", dict(RAMP, exposure=0.4, temperature=0.3, contrast=0.25, color_shift_amount=0.4,
                             color_shift_red=0.5, color_shift_blue=-0.2))
MASKED = Mod("advanced", dict(RAMP, exposure=0.5, tint=0.3), mask="M1")
MASK_LEAVES = {"M1": {"mask_axis": "exposure", "mask_mode": "highs", "blur": 24.0, "spread": 0.3}}


def expected_schedule(args, steps, hires=False):
    """The first chain entry's per-step values, as the hook builds them for ``steps`` steps."""
    return [float(v) for v in schedule_array(args, steps, hires)]


def schedule_array(args, steps=10, hires=False):
    entries = spec.build_chain(spec.config_from_args(args), hires).entries
    return engine.build_entries(entries, steps)[0][0]


def grid_run(full_hw):
    """What ``hook.grid_downscale`` reads off a ``hook.Run`` (krea2: 8 image pixels per latent pixel)."""
    return types.SimpleNamespace(downscale=8, full_hw=full_hw)


@contextlib.contextmanager
def recorded_lookups():
    """Every ``sigma_to_value`` Colorcraft makes: ``(sigma, walked list, value)`` per chain entry."""
    seen = []
    original = hook.sigma_to_value

    def spy(sigma, sigmas, sched):
        value = original(sigma, sigmas, sched)
        seen.append((float(sigma), [float(s) for s in sigmas], value))
        return value

    with mock.patch.object(hook, "sigma_to_value", side_effect=spy):
        yield seen


@contextlib.contextmanager
def recorded_masks():
    """Top-level mask resolutions: ``(pre shape, mask shape, vae_downscale_factor)``."""
    seen = []
    original = masking.resolve_mask_tensor
    depth = {"n": 0}

    def spy(mask_spec, pre, cur_basis, dev, factor):
        depth["n"] += 1
        try:
            mask = original(mask_spec, pre, cur_basis, dev, factor)
        finally:
            depth["n"] -= 1
        if depth["n"] == 0:
            seen.append((tuple(pre.shape), tuple(mask.shape), float(factor)))
        return mask

    with mock.patch.object(masking, "resolve_mask_tensor", side_effect=spy):
        yield seen


class SpeedColorcraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        I.require_forge()
        cls.scripts = I.Scripts()
        cls.h = I.Harness(cls.scripts)
        cls.ks = I.forge_k()

    def tearDown(self):
        self.scripts.teardown()

    def _run(self, mode, cc_args, *, seeds=(31,), speed_overrides=None, func=None):
        hooks = {"speed": I.speed_args(mode=mode, **(speed_overrides or {})), "colorcraft": cc_args}
        return self.h.generate(func or self.ks.sample_euler, seeds=seeds, hooks=hooks)

    def _assert_applied(self, result, mode):
        status = I.speed_status(result.request)
        self.assertTrue(status.startswith(f"base: applied ({mode})"), status)
        grids = [call.shape[-2:] for call in result.calls]
        self.assertEqual(grids, sorted(grids))
        self.assertEqual({grids[0], grids[-1]}, {(8, 8), (16, 16)})
        self.assertEqual(tuple(result.out.shape), (1, 16, 1, 16, 16))
        self.assertTrue(torch.isfinite(result.out).all())

    # -- grading on both grids, by the step's own schedule value --------------------------------

    def test_every_step_is_graded_on_both_grids(self):
        args = I.colorcraft_args(GRADE)
        expected = expected_schedule(args, 10)
        for mode in ("transition", "respace"):
            with self.subTest(mode=mode), recorded_lookups() as lookups:
                result = self._run(mode, args)
                self._assert_applied(result, mode)
                self.assertEqual(len(lookups), 10, "one lookup per step")
                for step, (sigma, _walked, value) in enumerate(lookups):
                    self.assertAlmostEqual(value, expected[step], places=12, msg=f"step {step} sigma {sigma}")
                status = I.colorcraft_status(result.request)
                edited = sum(1 for v in expected if v != 0)
                self.assertIn(f"10 evals, {edited} edited", status)
                self.assertNotIn("other sampling run", status)
                self.assertNotIn("error", status)

    def test_the_patched_sigmas_are_the_lookup_list(self):
        """The first full-size step runs at SPEED's aligned sigma; Colorcraft looks it up on the
        re-published list (exact: its own step) — on the unpatched list it would read another value."""
        args = I.colorcraft_args(GRADE)
        expected = expected_schedule(args, 10)
        original = I.flow_sigmas(10)
        for mode in ("transition", "respace"):
            with self.subTest(mode=mode), recorded_lookups() as lookups:
                result = self._run(mode, args)
                first_full = next(i for i, call in enumerate(result.calls) if call.shape[-1] == 16)
                sigma, walked, value = lookups[first_full]
                self.assertAlmostEqual(sigma, result.calls[first_full].sigma, places=6)
                self.assertNotAlmostEqual(sigma, float(original[first_full]), places=4)    # aligned
                self.assertAlmostEqual(walked[first_full], sigma, places=6)
                self.assertEqual(value, expected[first_full])
                stale = schedule.sigma_to_value(sigma, original, schedule_array(args))
                self.assertGreater(abs(stale - expected[first_full]), 0.05, "the test needs a ramp")
                if mode == "respace":
                    for step in range(first_full + 1, 10):
                        self.assertNotAlmostEqual(lookups[step][0], float(original[step]), places=4)
                        self.assertEqual(lookups[step][2], expected[step])
                # the original list is back after the run
                unet = result.request.sd_model.forge_objects.unet
                self.assertIs(unet.model_options["transformer_options"]["sampling_sigmas"], result.sigmas)

    def test_an_untagged_copy_is_still_another_run(self):
        """The mechanism: without SPEED's tag the copy is foreign (the fixed bug), with it the same run.
        A list another sampling run publishes stays foreign."""
        args = I.colorcraft_args(GRADE)
        with mock.patch.object(fh, "mark_republished", side_effect=lambda copy, source: copy):
            result = self._run("transition", args)
        self.assertIn("other sampling run passed through x4", I.colorcraft_status(result.request))
        self.scripts.teardown()
        result = self._run("transition", args)
        self.assertNotIn("other sampling run", I.colorcraft_status(result.request))

    # -- masks and anchors ------------------------------------------------------------------------

    def test_masks_and_anchors_follow_the_grid(self):
        args = I.colorcraft_args(MASKED, GRADE, masking=True, leaves=MASK_LEAVES)
        for mode in ("transition", "respace"):
            with self.subTest(mode=mode), recorded_masks() as masks:
                result = self._run(mode, args)
                self._assert_applied(result, mode)
                status = I.colorcraft_status(result.request)
                self.assertNotIn("error", status)
                self.assertNotIn("non-finite", status)
                self.assertTrue(masks)
                grids = set()
                for pre_shape, mask_shape, factor in masks:
                    self.assertEqual(mask_shape, (pre_shape[0], 1) + pre_shape[2:])
                    # one blur in image pixels: the coarse grid's latent pixel covers twice the image
                    self.assertEqual(factor, 8.0 * 16 / pre_shape[-1])
                    grids.add(pre_shape[-1])
                self.assertEqual(grids, {8, 16})
                # the VAE encoded the anchors once, before sampling
                self.assertEqual(result.request.sd_model.forge_objects.vae.calls, 2)

    def test_a_blur_that_fits_the_pass_grid_fits_the_coarse_grid(self):
        """Blur 24 px is 3 latent pixels on the 16x16 grid (radius 9). Read on the 8x8 coarse grid the
        same 3 latent pixels would need a reflect pad wider than the grid; in image pixels it is 1.5."""
        args = I.colorcraft_args(MASKED, masking=True, leaves=MASK_LEAVES)
        run = grid_run(full_hw=(16, 16))
        self.assertEqual(hook.grid_downscale(run, torch.zeros(1, 16, 8, 8)), 16.0)
        self.assertEqual(hook.grid_downscale(run, torch.zeros(1, 16, 16, 16)), 8)
        self.assertEqual(hook.grid_downscale(grid_run(full_hw=None), torch.zeros(1, 16, 8, 8)), 8)
        result = self._run("transition", args)
        status = I.colorcraft_status(result.request)
        expected = expected_schedule(args, 10)
        self.assertIn(f"10 evals, {sum(1 for v in expected if v != 0)} edited", status)
        self.assertNotIn("error", status)
        # without the grid (a caller that does not pass Forge's x) the coarse steps fail the blur and
        # are passed through, the full-size ones are graded: the old behaviour, never a crash
        self.scripts.teardown()
        with mock.patch.object(hook, "latent_grid", return_value=None), mock.patch.object(hook, "_log"):
            result = self._run("transition", args)
        status = I.colorcraft_status(result.request)
        self.assertRegex(status, r"error \w+ x\d+ \(input passed through\)")
        self.assertTrue(torch.isfinite(result.out).all())

    # -- batch, hires ------------------------------------------------------------------------------

    def test_batch_image_equals_the_same_seed_alone(self):
        args = I.colorcraft_args(MASKED, GRADE, masking=True, leaves=MASK_LEAVES)
        seeds = (31, 32)
        for mode in ("transition", "respace"):
            with self.subTest(mode=mode):
                batch = self._run(mode, args, seeds=seeds).out
                self.scripts.teardown()
                for index, seed in enumerate(seeds):
                    single = self._run(mode, args, seeds=(seed,)).out
                    self.scripts.teardown()
                    torch.testing.assert_close(batch[index:index + 1], single, atol=1e-4, rtol=1e-4)

    def test_hires_pass_grades_on_the_walked_slice(self):
        """Base pass plain, hires pass with SPEED (Apply to Hires pass) and a Colorcraft tab for both
        passes: the hires run walks ``sigmas[steps - t_enc - 1:]``; Colorcraft reads its schedule over
        that slice of the re-published list."""
        both = Mod("advanced", dict(RAMP, exposure=0.4, contrast=0.2, **{"pass": "Both"}))
        args = I.colorcraft_args(both)
        base = self.h.generate(self.ks.sample_euler, hooks={"colorcraft": args})
        with recorded_lookups() as lookups:
            hires = self.h.hires(base.request, self.ks.sample_euler, base.out, steps=8, denoise=0.6,
                                 hooks={"speed": I.speed_args(hires=True), "colorcraft": args})
        status = I.speed_status(hires.request)
        self.assertIn("hires: applied (transition)", status)
        self.assertEqual(tuple(hires.out.shape), (1, 16, 1, 24, 24))
        self.assertTrue(torch.isfinite(hires.out).all())
        walked_steps = len(hires.sigmas) - 1
        expected = expected_schedule(args, walked_steps, hires=True)
        self.assertEqual(len(lookups), walked_steps)
        for step, (sigma, walked, value) in enumerate(lookups):
            self.assertEqual(len(walked), walked_steps + 1)
            self.assertAlmostEqual(value, expected[step], places=12, msg=f"hires step {step}")
        cc = I.colorcraft_status(hires.request)
        self.assertIn("hires: krea2 (Wan21); mods I; 8 evals", cc)
        self.assertNotIn("other sampling run", cc)


if __name__ == "__main__":
    unittest.main()
