"""Anima SPEED + Euler (SMEA) Dy CFG++ — a segmented run of a sampler whose sub-steps change resolution.

SPEED calls the selected sampler once per resolution segment (coarse grid, then full size after the
expansion). The Dy samplers place their extra evaluations by step index — Euler Dy CFG++ at steps 2-3
(half resolution), Euler SMEA Dy CFG++ at step 0 (x1.25) and step 1 (half). Fixed here: they counted
steps per call, so under SPEED the sub-steps ran again at the start of every segment — Euler Dy two
steps into the full-size tail, SMEA Dy a x1.25 step right on SPEED's freshly expanded latent. SPEED now
hands a sampler that declares ``sam_extra_step_offset`` (``sam3ext.speed.runner.STEP_OFFSET_KWARG``)
the first step of each segment, and the Dy samplers place their sub-steps by that pass index.

Decision: the pair works, neither steps aside. The sub-steps run at the same steps as without SPEED, on
the grid the pass is on there (an early Dy sub-step halves the coarse grid). Under Forge's Spectrum
Integrated both step aside, each with its own status. The ER SDE pair and DPM++ 4M SDE (no sub-steps)
run under SPEED like Forge's samplers; every combination gives a batch image equal to the seed alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import inspect
import unittest
from unittest import mock

import torch

from sam3ext.extra_samplers import euler_dy, registry, substep_guard
from sam3ext.extra_samplers.registry import LABEL_EULER_DY_CFG_PP, LABEL_EULER_SMEA_DY_CFG_PP
from sam3ext.speed import forge_host as fh
from sam3ext.speed import runner
from tests import _integration_support as I

DY = (
    (LABEL_EULER_DY_CFG_PP, euler_dy.sample_euler_dy_cfg_pp),
    (LABEL_EULER_SMEA_DY_CFG_PP, euler_dy.sample_euler_smea_dy_cfg_pp),
)
MODES = ("transition", "respace")


def substep_plan(calls):
    """``[(pass step, marker, grid)]`` of the sub-step calls (a sub-step belongs to the step whose own
    evaluation came just before it)."""
    plan, step = [], -1
    for call in calls:
        if call.marker is None:
            step += 1
        else:
            plan.append((step, call.marker, call.shape[-2:]))
    return plan


def spectrum_unet_wrapper(model_function, kwargs):
    """Named like Forge's Spectrum Integrated wrapper (lib_spectrum), which SPEED and the Dy guard detect."""
    return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])


class SpeedDyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        I.require_forge()
        cls.scripts = I.Scripts()
        cls.h = I.Harness(cls.scripts)

    def tearDown(self):
        self.scripts.teardown()

    def _run(self, label, func, *, mode=None, seeds=(31,), manual="0.7", prepare=None):
        hooks = {"speed": I.speed_args(mode=mode, manual_sigmas=manual)} if mode else {}
        result = self.h.generate(func, seeds=seeds, hooks=hooks, prepare=prepare,
                                 sampler_kwargs={"runtime": I.dy_runtime(label)})
        self.assertEqual(result.seen, list(range(10)))
        self.assertEqual(tuple(result.out.shape), (len(seeds), 16, 1, 16, 16))
        self.assertTrue(torch.isfinite(result.out).all())
        if mode:
            self.assertTrue(I.speed_status(result.request).startswith(f"base: applied ({mode})"),
                            I.speed_status(result.request))
        return result

    def test_substeps_run_at_the_same_pass_steps_as_without_speed(self):
        for label, func in DY:
            plain = substep_plan(self._run(label, func).calls)
            self.assertEqual([(s, m) for s, m, _ in plain],
                             [(2, "dy"), (3, "dy")] if label == LABEL_EULER_DY_CFG_PP else [(0, "smea"), (1, "dy")])
            for mode in MODES:
                with self.subTest(sampler=label, mode=mode):
                    result = self._run(label, func, mode=mode)
                    plan = substep_plan(result.calls)
                    self.assertEqual([(s, m) for s, m, _ in plan], [(s, m) for s, m, _ in plain])
                    # transition at step 6: every sub-step falls in the coarse 8x8 segment
                    for _step, marker, grid in plan:
                        self.assertEqual(grid, (4, 4) if marker == "dy" else (10, 10))

    def test_a_transition_inside_the_substep_window(self):
        """The full-size segment starts on a sub-step step: that sub-step runs at full size, once."""
        cases = (
            # Euler Dy: transition at step 3 (sigma 0.875 <= 0.9) — step 2 coarse, step 3 full
            (LABEL_EULER_DY_CFG_PP, euler_dy.sample_euler_dy_cfg_pp, "0.9", [(2, "dy", (4, 4)), (3, "dy", (8, 8))]),
            # SMEA Dy: transition at step 1 (sigma 0.964 <= 0.97) — x1.25 coarse, half full
            (LABEL_EULER_SMEA_DY_CFG_PP, euler_dy.sample_euler_smea_dy_cfg_pp, "0.97",
             [(0, "smea", (10, 10)), (1, "dy", (8, 8))]),
        )
        for label, func, manual, expected in cases:
            for mode in MODES:
                with self.subTest(sampler=label, mode=mode):
                    result = self._run(label, func, mode=mode, manual=manual)
                    self.assertEqual(substep_plan(result.calls), expected)

    def test_batch_image_equals_the_same_seed_alone(self):
        seeds = (31, 32)
        for label, func in DY:
            for mode in MODES:
                with self.subTest(sampler=label, mode=mode):
                    batch = self._run(label, func, mode=mode, seeds=seeds).out
                    for index, seed in enumerate(seeds):
                        single = self._run(label, func, mode=mode, seeds=(seed,)).out
                        torch.testing.assert_close(batch[index:index + 1], single, atol=1e-4, rtol=1e-4)

    def test_spectrum_integrated_makes_both_step_aside(self):
        def spectrum(request):
            request.sd_model.forge_objects.unet.model_options["model_function_wrapper"] = spectrum_unet_wrapper

        for label, func in DY:
            with self.subTest(sampler=label), mock.patch.object(registry, "_log"):
                result = self.h.generate(func, hooks={"speed": I.speed_args()}, prepare=spectrum,
                                         sampler_kwargs={"runtime": I.dy_runtime(label)})
                self.assertIn("skipped - Forge Spectrum Integrated is on", I.speed_status(result.request))
                params = result.request.extra_generation_params
                self.assertEqual(params[substep_guard.STATUS_KEY], "dy sub-steps skipped (Spectrum)")
                self.assertEqual([call.marker for call in result.calls], [None] * 10)
                self.assertEqual({call.shape[-2:] for call in result.calls}, {(16, 16)})


class SpeedExtraSamplersTests(unittest.TestCase):
    """The non-Dy extra samplers under SPEED (Forge's own per-image noise, Brownian for 4M SDE / 2M SDE Heun)."""

    @classmethod
    def setUpClass(cls):
        I.require_forge()
        cls.scripts = I.Scripts()
        cls.h = I.Harness(cls.scripts)

    def tearDown(self):
        self.scripts.teardown()

    def test_er_sde_and_dpmpp_4m_sde(self):
        """Every non-Dy entry: SPEED runs it in segments, except an entry SPEED refuses by function name
        (``forge_host.UNSUPPORTED_SAMPLERS`` — UniPC bh2, Forge's UniPC under another name): that one runs
        unsegmented at full size with the refusal in ``Anima SPEED status``."""
        for spec in registry.SPECS:
            if spec.kind == "dy":
                continue
            brownian = bool(spec.options.get("brownian_noise"))
            runtime = (lambda s: (lambda request: registry.runtime_kwargs(s, request)))(spec)
            refused = fh.UNSUPPORTED_SAMPLERS.get(spec.func.__name__)
            for mode in MODES:
                with self.subTest(sampler=spec.label, mode=mode):
                    outs = {}
                    for seeds in ((31, 32), (31,), (32,)):
                        result = self.h.generate(spec.func, seeds=seeds, brownian=brownian,
                                                 hooks={"speed": I.speed_args(mode=mode)},
                                                 sampler_kwargs={"runtime": runtime})
                        self.scripts.teardown()
                        self.assertTrue(torch.isfinite(result.out).all())
                        if refused is not None:
                            self.assertTrue(I.speed_status(result.request).startswith(f"base: skipped - {refused}"),
                                            I.speed_status(result.request))
                            self.assertEqual({c.shape[-2:] for c in result.calls}, {(16, 16)})
                            outs[seeds] = result.out
                            continue
                        self.assertTrue(I.speed_status(result.request).startswith(f"base: applied ({mode})"))
                        self.assertEqual(result.seen, list(range(10)))
                        self.assertEqual({c.shape[-2:] for c in result.calls}, {(8, 8), (16, 16)})
                        outs[seeds] = result.out
                    torch.testing.assert_close(outs[(31, 32)][:1], outs[(31,)], atol=1e-4, rtol=1e-4)
                    torch.testing.assert_close(outs[(31, 32)][1:], outs[(32,)], atol=1e-4, rtol=1e-4)


class StepOffsetContractTests(unittest.TestCase):
    def test_the_dy_samplers_declare_the_keyword_speed_passes(self):
        for func in (euler_dy.sample_euler_dy_cfg_pp, euler_dy.sample_euler_smea_dy_cfg_pp, euler_dy.euler_dy):
            self.assertIn(runner.STEP_OFFSET_KWARG, inspect.signature(func).parameters, func.__name__)
        # the samplers without step-indexed extras do not take it (SPEED then passes nothing)
        for spec in registry.SPECS:
            if spec.kind != "dy":
                self.assertFalse(runner._declares(spec.func, runner.STEP_OFFSET_KWARG), spec.label)

    def test_only_a_declared_keyword_is_passed(self):
        def declared(model, x, sigmas, extra_args=None, callback=None, disable=None, sam_extra_step_offset=0):
            return x

        def keyword_only(model, x, sigmas, *, sam_extra_step_offset=0, **kwargs):
            return x

        def catch_all(model, x, sigmas, **kwargs):
            return x

        self.assertTrue(runner._declares(declared, runner.STEP_OFFSET_KWARG))
        self.assertTrue(runner._declares(keyword_only, runner.STEP_OFFSET_KWARG))
        self.assertFalse(runner._declares(catch_all, runner.STEP_OFFSET_KWARG))
        self.assertFalse(runner._declares(object(), runner.STEP_OFFSET_KWARG))

    def test_each_segment_gets_its_first_pass_step(self):
        from sam3ext.speed import schedule

        seen = []

        def declared(model, x, sigmas, extra_args=None, callback=None, disable=None, sam_extra_step_offset=0):
            seen.append((sam_extra_step_offset, len(sigmas) - 1, tuple(x.shape[-2:])))
            return x

        sigmas = torch.linspace(1.0, 0.0, 13)
        for mode in MODES:
            with self.subTest(mode=mode):
                seen.clear()
                plan = schedule.build_plan(mode=mode, transform="dct", sigmas=sigmas, full_grid=(16, 16),
                                           scales=(0.25, 0.5, 1.0), threshold="manual", manual_sigmas=(0.8, 0.5))
                runner.run_speed(declared, None, torch.zeros(1, 4, 16, 16), sigmas,
                                 run=runner.SpeedRun(plan=plan, seeds=[1]))
                starts = [0] + [t.step for t in plan.transitions]
                self.assertEqual([offset for offset, _, _ in seen], starts)
                self.assertEqual(sum(n for _, n, _ in seen), 12)
                self.assertEqual([grid for _, _, grid in seen], [(4, 4), (8, 8), (16, 16)])


if __name__ == "__main__":
    unittest.main()
