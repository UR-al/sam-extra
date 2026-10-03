# The preset and adaptive-delta cases are adapted from aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81
# tests/test_math.py:104-487 (TestPresets, TestAdaptiveDelta) — MIT License, Copyright (c) 2026 A. Izzuddin
# Al Faruq (the presets and adaptive delta are Oleg Afonin's work in that fork):
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Changes: rewritten against sam3ext.speed.schedule (``delta_threshold_plan`` + ``static_transitions``
# instead of ``_resolve_transitions``); the build_plan cases (no coarse steps, dropped stages, grids,
# respace) are new.
"""sam3ext.speed.schedule — presets, sigma*, and the transition / respace plans."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.speed import schedule  # noqa: E402
from sam3ext.speed.schedule import PlanError  # noqa: E402


def flux_schedule(steps: int, mu: float = 1.15) -> torch.Tensor:
    shift = math.exp(mu)
    ts = [1.0 - i / steps for i in range(steps)]
    return torch.tensor([shift * t / (shift * t + 1.0 - t) for t in ts] + [0.0])


def anima_schedule(steps: int, shift: float = 3.0) -> torch.Tensor:
    ts = [1.0 - i / steps for i in range(steps)]
    return torch.tensor([shift * t / (1.0 + (shift - 1.0) * t) for t in ts] + [0.0])


class PresetTests(unittest.TestCase):
    REFERENCE_SIGMA = {
        "flux": 0.9680, "wan21": 0.6941, "krea-2": 0.9000, "krea-2-raw": 0.8300,
        "z-image": 0.9400, "flux2": 0.9680, "anima": 0.9600,
    }
    REFERENCE_COARSE_SHARE = {"wan21": 0.7790, "krea-2": 0.2598, "krea-2-raw": 0.3928, "z-image": 0.1607, "anima": 0.1111}
    NEO_QUALITY_FLOOR = 0.9249

    def _sigma_star(self, name, scale=0.5, delta=0.01, latent=None):
        preset = schedule.PRESETS[name]
        latent = preset["ref_latent"] if latent is None else latent
        return schedule.delta_optimal_transitions([scale, 1.0], delta, preset["A"], preset["beta"], latent, latent)[0]

    def test_documented_transition_sigmas(self):
        for name, expected in self.REFERENCE_SIGMA.items():
            with self.subTest(preset=name):
                self.assertAlmostEqual(self._sigma_star(name), expected, places=4)

    def test_reference_coarse_shares(self):
        for name, expected in self.REFERENCE_COARSE_SHARE.items():
            with self.subTest(preset=name):
                share = schedule.reference_coarse_fraction(self._sigma_star(name), schedule.PRESETS[name]["ref_shift"])
                self.assertAlmostEqual(share, expected, places=4)

    def test_anima_clears_the_neo_floor_on_the_default_forge_path(self):
        self.assertGreater(self._sigma_star("anima") / 1.03, self.NEO_QUALITY_FLOOR)

    def test_measured_presets_have_no_ref_shift(self):
        for name in ("flux", "flux2"):
            self.assertNotIn("ref_shift", schedule.PRESETS[name])
        self.assertEqual(schedule.PRESETS["anima"]["ref_shift"], 3.0)
        self.assertEqual(schedule.PRESETS["anima"]["ref_latent"], 128)

    def test_preset_params(self):
        self.assertEqual(schedule.preset_params("anima"), (8664.998524, 2.422687, 128, 3.0))
        self.assertEqual(schedule.preset_params("custom", 1.0, 2.0), (1.0, 2.0, 128, None))
        self.assertEqual(schedule.preset_params("nope", 3.0, 4.0), (3.0, 4.0, 128, None))
        self.assertEqual((schedule.CUSTOM_DEFAULT_A, schedule.CUSTOM_DEFAULT_BETA), (203.615097, 1.915461))


class AdaptiveDeltaTests(unittest.TestCase):
    A, BETA = 203.615097, 1.915461
    REF_SHIFT = 3.158193

    def _coarse(self, latent, steps=40, mu=1.15, adaptive=False, scales=(0.5, 1.0), ref_shift=None, divisor=1.0):
        sigmas = flux_schedule(steps, mu)
        tp = schedule.delta_threshold_plan(steps, scales, 0.01, self.A, self.BETA, latent, latent, adaptive=adaptive,
                                           ref_latent=128, ref_shift=ref_shift, sigma_divisor=divisor)
        transitions = schedule.static_transitions(sigmas, scales, tp.values, tp.caps)
        return transitions[-1][0] if transitions else steps

    def test_coarse_share_grows_with_resolution_without_adaptive(self):
        self.assertGreater(self._coarse(256), self._coarse(128))

    def test_adaptive_pins_the_split(self):
        reference = self._coarse(128, adaptive=True)
        for latent in (192, 256, 384):
            self.assertEqual(self._coarse(latent, adaptive=True), reference)

    def test_adaptive_is_a_noop_at_and_below_the_reference(self):
        for latent in (64, 96, 128):
            self.assertEqual(self._coarse(latent, adaptive=True), self._coarse(latent))

    def test_cap_rescues_a_resolution_dependent_shift(self):
        self.assertEqual(self._coarse(384, 40, 6.70), 40)
        capped = self._coarse(384, 40, 6.70, adaptive=True)
        self.assertLessEqual(capped, 40 * schedule.MAX_COARSE_FRACTION)

    def test_reference_share_holds_for_spectrum_derived_presets(self):
        reference = self._coarse(128, adaptive=True, ref_shift=self.REF_SHIFT)
        for mu in (2.02, 3.23, 6.70):
            self.assertEqual(self._coarse(192, 40, mu, adaptive=True, ref_shift=self.REF_SHIFT), reference)

    def test_divisor(self):
        baseline = self._coarse(128, adaptive=True)
        self.assertEqual(self._coarse(128, adaptive=True, divisor=1.0), baseline)
        self.assertGreater(self._coarse(128, adaptive=True, divisor=1.03), baseline)
        self.assertLess(self._coarse(128, adaptive=True, divisor=0.9), baseline)

    def test_a_divisor_lifting_sigma_star_to_one_gets_no_cap(self):
        tp = schedule.delta_threshold_plan(32, (0.5, 1.0), 0.01, 8664.998524, 2.422687, 128, 128,
                                           adaptive=True, ref_shift=3.0, sigma_divisor=0.5)
        self.assertGreater(tp.values[0], 1.0)
        self.assertEqual(tp.caps, (None,))


class BuildPlanTests(unittest.TestCase):
    def plan(self, **kw):
        defaults = dict(mode="transition", transform="dct", sigmas=anima_schedule(20), full_grid=(32, 32),
                        scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.8,))
        defaults.update(kw)
        return schedule.build_plan(**defaults)

    def test_transition_plan(self):
        plan = self.plan()
        self.assertEqual(plan.first_grid, (16, 16))
        self.assertEqual(len(plan.transitions), 1)
        t = plan.transitions[0]
        sigmas = anima_schedule(20)
        expected_step = next(j for j in range(20) if float(sigmas[j]) <= 0.8)
        self.assertEqual((t.step, t.k, t.grid_from, t.grid_to), (expected_step, 0, (16, 16), (32, 32)))
        self.assertEqual(plan.coarse_steps, expected_step)
        self.assertEqual(plan.thresholds, (0.8,))

    def test_no_coarse_steps_when_the_pass_starts_below_the_transition(self):
        """img2img / hires: Forge samples sigmas[steps - t_enc - 1:], already below sigma*."""
        sliced = anima_schedule(20)[12:]
        with self.assertRaises(PlanError) as ctx:
            self.plan(sigmas=sliced)
        self.assertEqual(ctx.exception.kind, "skip")
        self.assertIn("no coarse steps", ctx.exception.reason)
        with self.assertRaises(PlanError):
            self.plan(sigmas=sliced, mode="respace")

    def test_a_pass_starting_between_two_thresholds_drops_the_first_stage(self):
        sigmas = anima_schedule(30)[3:]
        start = float(sigmas[0])
        plan = self.plan(sigmas=sigmas, scales=(0.25, 0.5, 1.0), manual_sigmas=(start + 0.01, 0.6), full_grid=(64, 64))
        self.assertEqual(plan.first_scale, 0.5)
        self.assertEqual(plan.first_grid, (32, 32))
        self.assertEqual([t.k for t in plan.transitions], [1])
        self.assertTrue(any("skipped" in note for note in plan.notes))

    def test_never_reached_is_refused(self):
        with self.assertRaises(PlanError) as ctx:
            self.plan(manual_sigmas=(0.01,))
        self.assertIn("never reached", ctx.exception.reason)
        with self.assertRaises(PlanError):
            self.plan(manual_sigmas=(0.01,), mode="respace")

    def test_divisor_pushing_sigma_star_past_one_means_no_coarse_steps(self):
        A, beta, ref_latent, ref_shift = schedule.preset_params("anima")
        with self.assertRaises(PlanError) as ctx:
            self.plan(threshold="neo_shift", A=A, beta=beta, ref_latent=ref_latent, ref_shift=ref_shift,
                      sigma_divisor=0.5, full_grid=(128, 128))
        self.assertIn("no coarse steps", ctx.exception.reason)

    def test_manual_validation(self):
        for kw in (dict(manual_sigmas=(0.9, 0.8)), dict(manual_sigmas=(1.0,)),
                   dict(scales=(0.25, 0.5, 1.0), manual_sigmas=(0.6, 0.8)),
                   dict(mode="bogus"), dict(transform="wavelet"), dict(threshold="auto")):
            with self.subTest(kw=kw):
                with self.assertRaises(PlanError) as ctx:
                    self.plan(**kw)
                self.assertEqual(ctx.exception.kind, "invalid")
        respace = self.plan(mode="respace", scales=(0.25, 0.5, 1.0), manual_sigmas=(0.7,), full_grid=(64, 64))
        self.assertEqual(respace.thresholds, (0.7, 0.7))          # sorryhyun: one sigma for every hand-off
        unordered = self.plan(mode="respace", scales=(0.25, 0.5, 1.0), manual_sigmas=(0.5, 0.7), full_grid=(64, 64))
        self.assertEqual(len(unordered.transitions), 2)

    def test_delta_validation(self):
        for delta in (0.0, 1.0, -0.1):
            with self.assertRaises(PlanError):
                self.plan(threshold="delta_optimal", delta=delta)

    def test_custom_spectrum_validation(self):
        """A <= 0 used to escape as ValueError / ZeroDivisionError (only PlanError is caught, so the whole
        Forge request failed) and a NaN A or beta ran with a NaN sigma*: all are settings errors now."""
        nan, inf = float("nan"), float("inf")
        for A, beta in ((-1.0, 1.9), (0.0, 1.9), (nan, 1.9), (inf, 1.9), (203.6, nan), (203.6, inf), (203.6, 1e6)):
            for threshold in ("delta_optimal", "neo_shift"):
                with self.subTest(A=A, beta=beta, threshold=threshold):
                    with self.assertRaises(PlanError) as ctx:
                        self.plan(threshold=threshold, A=A, beta=beta, full_grid=(128, 128))
                    self.assertEqual(ctx.exception.kind, "invalid")
        plan = self.plan(threshold="delta_optimal", A=203.6, beta=-3.0, full_grid=(128, 128))
        self.assertEqual(len(plan.transitions), 1)          # a negative beta is still a valid power law

    def test_grids(self):
        self.assertEqual(schedule.stage_grids("transition", [0.37, 1.0], 27, 33), [(10, 12), (27, 33)])
        self.assertEqual(schedule.stage_grids("respace", [0.5, 1.0], 125, 96), [(62, 48), (125, 96)])
        self.assertEqual(schedule.stage_grids("respace", [0.25, 0.5, 1.0], 128, 104), [(32, 26), (64, 52), (128, 104)])
        plan = self.plan(mode="respace", full_grid=(51, 37))
        self.assertEqual(plan.transitions[-1].grid_to, (51, 37))   # Forge does not pad odd latents
        self.assertEqual(plan.first_grid, (26, 18))

    def test_a_last_scale_validate_scales_accepts_is_the_full_grid(self):
        """validate_scales takes 1.0 within 1e-6; respace must not even-snap such a last stage."""
        for mode in ("transition", "respace"):
            with self.subTest(mode=mode):
                plan = self.plan(mode=mode, scales=(0.5, 0.9999999), full_grid=(125, 77))
                self.assertEqual(plan.transitions[-1].grid_to, (125, 77))
                self.assertEqual(plan.first_grid[0], 62)

    def test_dwt_needs_exact_doubling(self):
        self.assertEqual(self.plan(transform="dwt", full_grid=(50, 32)).first_grid, (25, 16))
        with self.assertRaises(PlanError) as ctx:
            self.plan(transform="dwt", full_grid=(51, 32))
        self.assertIn("dwt", ctx.exception.reason)
        with self.assertRaises(PlanError):
            self.plan(transform="dwt", scales=(0.37, 1.0))

    def test_dwt_refuses_a_ratio_other_than_two(self):
        """Official: 'DWT requires r=2 between consecutive scales' — also when rounding happens to double
        the grid (0.26 -> 0.5 on 8 rows: 2 -> 4 rows, r = 1.923)."""
        with self.assertRaises(PlanError) as ctx:
            self.plan(transform="dwt", scales=(0.26, 0.5, 1.0), manual_sigmas=(0.9, 0.6), full_grid=(8, 8))
        self.assertEqual(ctx.exception.kind, "invalid")
        self.assertIn("r = 1.9231", ctx.exception.reason)
        plan = self.plan(transform="dwt", scales=(0.25, 0.5, 1.0), manual_sigmas=(0.9, 0.6), full_grid=(8, 8))
        self.assertEqual([t.grid_to for t in plan.transitions], [(4, 4), (8, 8)])
        # a dropped leading stage does not expand, so only the ratios that run count
        sigmas = anima_schedule(30)[3:]
        start = float(sigmas[0])
        dropped = self.plan(transform="dwt", sigmas=sigmas, scales=(0.2, 0.5, 1.0), manual_sigmas=(start + 0.01, 0.6),
                            full_grid=(16, 16))
        self.assertEqual([t.k for t in dropped.transitions], [1])

    def test_hand_off_sigmas_are_the_ones_the_run_uses(self):
        sigmas = torch.linspace(1.0, 0.0, 21)
        plan = self.plan(sigmas=sigmas, scales=(0.25, 0.5, 1.0), manual_sigmas=(0.9, 0.6), full_grid=(64, 64))
        for t in plan.transitions:          # transition mode: the schedule value, aligned = sigma * kappa
            self.assertEqual(t.sigma, float(sigmas[t.step]))
            r = t.scale_to / t.scale_from
            self.assertAlmostEqual(t.aligned, t.sigma * r / (1.0 + (r - 1.0) * t.sigma), places=6)
        respace = self.plan(mode="respace", sigmas=sigmas, scales=(0.25, 0.5, 1.0), manual_sigmas=(0.9, 0.6),
                            full_grid=(64, 64))
        first, second = respace.transitions
        self.assertEqual((first.step, second.step), (2, 9))
        self.assertAlmostEqual(first.aligned, 1.8 / 1.9, places=6)
        self.assertAlmostEqual(second.sigma, float(sigmas[9]) * first.aligned / first.sigma, places=6)   # re-spaced
        self.assertNotAlmostEqual(second.sigma, float(sigmas[9]), places=3)

    def test_empty_coarse_grid(self):
        with self.assertRaises(PlanError) as ctx:
            self.plan(scales=(0.01, 1.0), full_grid=(16, 16))
        self.assertEqual(ctx.exception.kind, "invalid")

    def test_single_scale_is_nothing_to_do(self):
        with self.assertRaises(PlanError) as ctx:
            self.plan(scales=(1.0,))
        self.assertIn("no coarse stage", ctx.exception.reason)


class RespaceTransitionsTests(unittest.TestCase):
    def test_hand_computed_respacing(self):
        sigmas = torch.linspace(1.0, 0.0, 11)
        dropped, events, final = schedule.respace_transitions(sigmas, (0.5, 1.0), (0.55,))
        self.assertEqual(dropped, 0)
        step, k, old, new = events[0]
        self.assertEqual((step, k), (5, 0))
        self.assertAlmostEqual(old, 0.5, places=6)
        self.assertAlmostEqual(new, 2 * 0.5 / 1.5, places=6)
        expected_tail = new * (sigmas[6:].float() / float(sigmas[5]))
        torch.testing.assert_close(final[6:], expected_tail)
        torch.testing.assert_close(final[:5], sigmas[:5].float())

    def test_leading_stage_dropped_not_expanded(self):
        sigmas = torch.linspace(0.6, 0.0, 7)
        dropped, events, final = schedule.respace_transitions(sigmas, (0.25, 0.5, 1.0), (0.7, 0.3))
        self.assertEqual(dropped, 1)
        self.assertEqual([e[1] for e in events], [1])
        self.assertAlmostEqual(float(final[0]), 0.6, places=6)     # nothing was re-spaced at step 0

    def test_caps_fire_without_the_threshold(self):
        sigmas = torch.linspace(1.0, 0.0, 11)
        _, events, _ = schedule.respace_transitions(sigmas, (0.5, 1.0), (0.01,), caps=(3,))
        self.assertEqual(events[0][0], 3)


if __name__ == "__main__":
    unittest.main()
