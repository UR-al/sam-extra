# The segmentation cases at the top are adapted from aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81
# tests/test_forge_script.py:82-278 (RecordingSampler, TestCoreSegmentation) — MIT License, Copyright (c)
# 2026 A. Izzuddin Al Faruq (the Forge script and its tests are Oleg Afonin's work in that fork):
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
# Changes: rewritten against sam3ext.speed.runner (plans from schedule.build_plan, a host object);
# the respace, host, img2img and failure cases are new.
"""sam3ext.speed.runner — the segmented run, independent of Forge."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.speed import runner, schedule, spectral  # noqa: E402


class RecordingSampler:
    """Fake k-diffusion solver: records calls, invokes the callback per step, returns x unchanged."""

    def __init__(self):
        self.calls = []
        self.__name__ = "sample_fake_euler"

    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, noise_sampler=None):
        self.calls.append({
            "shape": tuple(x.shape), "sigmas": [float(s) for s in sigmas], "steps": len(sigmas) - 1,
            "noise_sampler": noise_sampler, "s_churn": s_churn, "x": x.clone(),
        })
        for i in range(len(sigmas) - 1):
            if callback is not None:
                callback({"i": i, "x": x, "sigma": sigmas[i], "denoised": x})
        return x


class RecordingHost(runner.SpeedHost):
    def __init__(self):
        self.published = []
        self.restored = 0
        self.coarse_requests = []
        self.scope_shapes = []

    def noise_scope(self, full_shape):
        self.scope_shapes.append(full_shape)
        return super().noise_scope(full_shape)

    def coarse_noise_sampler(self, template, x, stage, sigmas):
        self.coarse_requests.append((stage, tuple(x.shape)))
        return ("coarse", stage)

    def publish_sigmas(self, sigmas):
        self.published.append(sigmas.clone())

    def restore_sigmas(self):
        self.restored += 1


def anima_like(steps: int, shift: float = 3.0) -> torch.Tensor:
    """Discrete-flow (time_snr_shift) sigmas, uniform in t, ending at 0."""
    return torch.tensor([shift * t / (1.0 + (shift - 1.0) * t) for t in (1.0 - i / steps for i in range(steps))] + [0.0])


def manual_plan(sigmas, grid=(32, 32), scales=(0.5, 1.0), manual=(0.72,), mode="transition", transform="dct"):
    return schedule.build_plan(mode=mode, transform=transform, sigmas=sigmas, full_grid=grid, scales=scales,
                               threshold="manual", manual_sigmas=manual)


def run(plan, x, sigmas, seeds=None, **kw):
    seeds = seeds if seeds is not None else list(range(x.shape[0]))
    rec = kw.pop("sampler", None) or RecordingSampler()
    out, report = runner.run_speed(rec, None, x, sigmas, run=runner.SpeedRun(plan=plan, seeds=seeds, **kw.pop("run_kw", {})), **kw)
    return out, report, rec


class SegmentationTests(unittest.TestCase):
    def test_two_segments_shapes_steps_and_callback(self):
        sigmas = torch.linspace(1.0, 0.0, 21)
        seen = []
        out, report, rec = run(manual_plan(sigmas), torch.randn(1, 4, 32, 32), sigmas,
                               callback=lambda d: seen.append(d["i"]))
        self.assertEqual([c["shape"] for c in rec.calls], [(1, 4, 16, 16), (1, 4, 32, 32)])
        self.assertEqual(out.shape, (1, 4, 32, 32))
        self.assertEqual(rec.calls[0]["steps"] + rec.calls[1]["steps"], 20)
        self.assertEqual(seen, list(range(20)))
        t = rec.calls[0]["sigmas"][-1]
        self.assertAlmostEqual(rec.calls[1]["sigmas"][0], t * 2.0 / (1.0 + t), places=5)
        self.assertEqual(rec.calls[1]["sigmas"][1:], [float(s) for s in sigmas[rec.calls[0]["steps"] + 1:]])
        self.assertTrue(report.applied)
        self.assertEqual((report.coarse_steps, report.total_steps), (rec.calls[0]["steps"], 20))

    def test_5d_video_latent(self):
        sigmas = torch.linspace(1.0, 0.0, 11)
        out, _, rec = run(manual_plan(sigmas, grid=(16, 16), manual=(0.55,)), torch.randn(1, 4, 3, 16, 16), sigmas)
        self.assertEqual(rec.calls[0]["shape"], (1, 4, 3, 8, 8))
        self.assertEqual(out.shape, (1, 4, 3, 16, 16))

    def test_delta_optimal_flux_fit_transitions(self):
        sigmas = torch.linspace(1.0, 0.0, 21)
        plan = schedule.build_plan(mode="transition", transform="dct", sigmas=sigmas, full_grid=(64, 64), scales=(0.5, 1.0),
                                   threshold="delta_optimal", delta=0.01, A=203.615097, beta=1.915461, adaptive=False)
        out, _, rec = run(plan, torch.randn(1, 4, 64, 64), sigmas)
        self.assertEqual([c["shape"] for c in rec.calls], [(1, 4, 32, 32), (1, 4, 64, 64)])
        self.assertEqual(out.shape, (1, 4, 64, 64))

    def test_three_stages(self):
        sigmas = torch.linspace(1.0, 0.0, 31)
        out, report, rec = run(manual_plan(sigmas, grid=(64, 64), scales=(0.25, 0.5, 1.0), manual=(0.9, 0.6)),
                               torch.randn(2, 4, 64, 64), sigmas)
        self.assertEqual([c["shape"][-1] for c in rec.calls], [16, 32, 64])
        self.assertEqual([e["k"] for e in report.events], [0, 1])
        self.assertIn("16x16 -> 32x32 -> 64x64", report.summary(manual_plan(sigmas, grid=(64, 64), scales=(0.25, 0.5, 1.0), manual=(0.9, 0.6))))

    def test_respace_rescales_the_remaining_schedule(self):
        sigmas = torch.linspace(1.0, 0.0, 21)
        plan = manual_plan(sigmas, mode="respace", manual=(0.7,))
        _, report, rec = run(plan, torch.randn(1, 4, 32, 32), sigmas)
        coarse, full = rec.calls
        step = coarse["steps"]
        old = float(sigmas[step])
        new = 2 * old / (1 + old)
        self.assertAlmostEqual(full["sigmas"][0], new, places=6)
        expected = (new * (sigmas[step + 1:].float() / old)).tolist()      # sorryhyun's float32 ops
        self.assertEqual(full["sigmas"][1:], expected)
        self.assertEqual(report.events[0]["sigma"], old)

    def test_plan_hand_offs_are_exactly_the_runs(self):
        """The plan's per-transition sigmas (console plan line) equal what the run reports, bit for bit."""
        cases = [
            ("transition", torch.linspace(1.0, 0.0, 31), (0.9, 0.6)),
            ("respace", torch.linspace(1.0, 0.0, 31), (0.9, 0.6)),
            ("respace", anima_like(28), (0.7,)),
            ("transition", torch.linspace(1.0, 0.0, 5), (0.9, 0.8)),      # both fire at step 1: the second
        ]                                                                   # expands at the first's aligned sigma
        for mode, sigmas, manual in cases:
            with self.subTest(mode=mode, steps=len(sigmas) - 1, manual=manual):
                plan = manual_plan(sigmas, grid=(64, 64), scales=(0.25, 0.5, 1.0), manual=manual, mode=mode)
                _, report, _ = run(plan, torch.randn(1, 4, 64, 64), sigmas)
                self.assertEqual([(t.step, t.sigma, t.aligned) for t in plan.transitions],
                                 [(e["step"], e["sigma"], e["aligned"]) for e in report.events])
        same = manual_plan(torch.linspace(1.0, 0.0, 5), grid=(64, 64), scales=(0.25, 0.5, 1.0), manual=(0.9, 0.8))
        self.assertEqual(same.transitions[1].sigma, same.transitions[0].aligned)

    def test_extra_kwargs_reach_every_segment_noise_sampler_only_full(self):
        sigmas = torch.linspace(1.0, 0.0, 21)
        sentinel = object()
        host = RecordingHost()
        _, _, rec = run(manual_plan(sigmas, grid=(64, 64), scales=(0.25, 0.5, 1.0), manual=(0.9, 0.6)),
                        torch.randn(1, 4, 64, 64), sigmas, host=host,
                        sampler_kwargs={"s_churn": 0.25, "noise_sampler": sentinel})
        self.assertEqual([c["s_churn"] for c in rec.calls], [0.25, 0.25, 0.25])
        self.assertEqual([c["noise_sampler"] for c in rec.calls], [("coarse", 0), ("coarse", 1), sentinel])
        self.assertEqual(host.coarse_requests, [(0, (1, 4, 16, 16)), (1, (1, 4, 32, 32))])

    def test_host_publishes_after_each_transition_and_restores(self):
        sigmas = torch.linspace(1.0, 0.0, 31)
        host = RecordingHost()
        run(manual_plan(sigmas, grid=(64, 64), scales=(0.25, 0.5, 1.0), manual=(0.9, 0.6)), torch.randn(1, 4, 64, 64),
            sigmas, host=host)
        self.assertEqual(len(host.published), 2)
        self.assertEqual(host.restored, 1)
        self.assertEqual(host.scope_shapes, [(1, 4, 64, 64)])
        first = host.published[0]
        changed = (first != sigmas).nonzero().flatten().tolist()
        self.assertEqual(len(changed), 1)          # transition mode patches one sigma

    def test_restore_runs_on_errors_and_interrupts(self):
        class Interrupted(BaseException):
            pass

        for exc in (RuntimeError("boom"), Interrupted()):
            host = RecordingHost()

            def failing(model, x, sigmas, _exc=exc, **kwargs):
                if x.shape[-1] == 32:
                    raise _exc
                return x

            with self.subTest(exc=type(exc).__name__):
                sigmas = torch.linspace(1.0, 0.0, 11)
                with self.assertRaises(type(exc)):
                    runner.run_speed(failing, None, torch.randn(1, 4, 32, 32), sigmas,
                                     run=runner.SpeedRun(plan=manual_plan(sigmas, manual=(0.55,)), seeds=[0]), host=host)
                self.assertEqual(host.restored, 1)

    def test_input_is_never_modified(self):
        sigmas = torch.linspace(1.0, 0.0, 11)
        x = torch.randn(1, 4, 32, 32)
        before = x.clone()
        run(manual_plan(sigmas, manual=(0.55,)), x, sigmas)
        torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_mismatches_are_refused(self):
        sigmas = torch.linspace(1.0, 0.0, 11)
        plan = manual_plan(sigmas, manual=(0.55,))
        with self.assertRaises(ValueError):
            run(plan, torch.randn(1, 4, 30, 32), sigmas)
        with self.assertRaises(ValueError):
            run(plan, torch.randn(2, 4, 32, 32), sigmas, seeds=[1])
        with self.assertRaises(ValueError):
            run(plan, torch.randn(4, 32, 32), sigmas)


class SeedTests(unittest.TestCase):
    def test_batch_equals_single_seed_runs(self):
        sigmas = torch.linspace(1.0, 0.0, 15)
        x = torch.randn(3, 4, 1, 16, 16)
        for mode in ("transition", "respace"):
            for transform in ("dct", "fft", "dwt"):
                with self.subTest(mode=mode, transform=transform):
                    plan = manual_plan(sigmas, grid=(16, 16), manual=(0.6,), mode=mode, transform=transform)
                    out, _, _ = run(plan, x, sigmas, seeds=[11, 12, 13])
                    for b, seed in enumerate((11, 12, 13)):
                        single, _, _ = run(plan, x[b:b + 1], sigmas, seeds=[seed])
                        torch.testing.assert_close(out[b:b + 1], single, rtol=0, atol=0)

    def test_expansion_noise_depends_on_the_seed(self):
        sigmas = torch.linspace(1.0, 0.0, 15)
        x = torch.zeros(1, 4, 16, 16)
        plan = manual_plan(sigmas, grid=(16, 16), manual=(0.6,))
        a, _, _ = run(plan, x, sigmas, seeds=[1])
        b, _, _ = run(plan, x, sigmas, seeds=[2])
        c, _, _ = run(plan, x, sigmas, seeds=[1])
        self.assertFalse(torch.equal(a, b))
        torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_numpy_draw_order_of_5d_items(self):
        """Official 5-D handling: (C, T) images are visited as (T, C) channels."""
        draws = runner._numpy_draws(5, (2, 3), [(4, 4)])[0]
        import numpy as np

        rng = np.random.default_rng(5)
        expected = torch.empty(2, 3, 4, 4)
        for t in range(3):
            for c in range(2):
                expected[c, t] = torch.from_numpy(rng.standard_normal((4, 4)).astype(np.float32))
        torch.testing.assert_close(draws, expected, rtol=0, atol=0)


class Img2ImgInitTests(unittest.TestCase):
    def _coarse_input(self, init_latent, rescale, sigmas, x, seeds=(3,)):
        rec = RecordingSampler()
        plan = manual_plan(sigmas, grid=tuple(x.shape[-2:]), manual=(0.6,))
        runner.run_speed(rec, None, x, sigmas, run=runner.SpeedRun(plan=plan, seeds=list(seeds), init_latent=init_latent,
                                                                    img2img_rescale=rescale))
        return rec.calls[0]["x"].double()

    def test_txt2img_and_rescale_off_are_plain_truncation(self):
        x = torch.randn(1, 4, 32, 32)
        sigmas = torch.linspace(1.0, 0.0, 21)
        plain = spectral.dct_downscale(x, (16, 16)).to(x.dtype).double()
        torch.testing.assert_close(self._coarse_input(False, True, sigmas, x), plain)
        torch.testing.assert_close(self._coarse_input(True, True, sigmas, x), plain)     # sigma0 = 1: nothing to fix
        img2img = torch.linspace(1.0, 0.0, 21)[2:]
        torch.testing.assert_close(self._coarse_input(True, False, img2img, x), plain)

    def test_rescale_formula(self):
        x = torch.randn(2, 4, 32, 32)
        sigmas = torch.linspace(1.0, 0.0, 21)[2:]
        s0 = float(sigmas[0])
        got = self._coarse_input(True, True, sigmas, x, seeds=(3, 4))
        r = 2.0
        noise = torch.stack([
            torch.randn((4, 16, 16), generator=torch.Generator().manual_seed(seed + runner.INIT_NOISE_SEED_OFFSET))
            for seed in (3, 4)
        ]).double()
        expected = spectral.dct_downscale(x, (16, 16)) / r + s0 * math.sqrt(1 - 1 / r ** 2) * noise
        torch.testing.assert_close(got, expected.float().double(), atol=1e-6, rtol=1e-6)

    def test_rescale_restores_the_flow_form_statistically(self):
        """(1-s0) image + s0 noise on the full grid -> (1-s0) coarse image + s0 noise on the coarse grid."""
        torch.manual_seed(0)
        s0 = 0.9
        image = torch.linspace(-1.5, 1.5, 32).view(1, 1, 1, 32).expand(256, 4, 32, 32).contiguous()
        noise = torch.randn(256, 4, 32, 32)
        x = (1 - s0) * image + s0 * noise
        sigmas = torch.tensor([s0, 0.7, 0.5, 0.3, 0.0])
        y = spectral.amplitude_resize(image[:1], (16, 16))       # the coarse image at normal amplitude

        def image_gain(state):
            mean = state.mean(0, keepdim=True)
            return float((mean * y).sum() / (y * y).sum())

        coarse = self._coarse_input(True, True, sigmas, x, seeds=list(range(256)))
        self.assertAlmostEqual(image_gain(coarse), 1 - s0, delta=0.005)
        self.assertAlmostEqual(float((coarse - (1 - s0) * y).std()), s0, delta=0.01)
        upstream = self._coarse_input(True, False, sigmas, x, seeds=list(range(256)))
        self.assertAlmostEqual(image_gain(upstream), 2 * (1 - s0), delta=0.005)    # r = 2 times too strong
        self.assertAlmostEqual(float((upstream - 2 * (1 - s0) * y).std()), s0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
