"""``Restart (flow)`` (``sam3ext/extra_samplers/restart_flow.py``) — Restart sampling with the flow forward kernel.

Forge's real ``k_diffusion.sampling`` and ``modules/sd_samplers_extra.py`` (its built-in ``restart_sampler``,
A1111's code) are executed read-only from Forge's files; the built-in one is the oracle of the ε case only
(nothing of it is copied). What is pinned down:

* the plan: where the restart window lands on the scheduler's own grid (flow shift 3, ε Karras), when there
  is none, how many restarts;
* without a restart (fewer than 20 steps) the sampler is Forge's Heun bit for bit; on ε models with a restart
  it is Forge's Restart given the same window on a Karras grid (Forge rebuilds the restart sub-grid with
  float32 ``get_sigmas_karras`` — a sub-block of a Karras grid only up to rounding — hence the tolerance);
* the forward kernel moves the model marginal at σ_lo exactly to the one at σ_hi on flow, and Forge's VE
  kernel does not (negative control); end to end on the exact denoiser of Gaussian data the sampler ends in
  the data distribution;
* Forge's denoiser call counter and the callback index follow the grid step being taken.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

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

from sam3ext.extra_samplers import restart_flow  # noqa: E402

RF = restart_flow


def _levels(sigmas, flow):
    """s = σ/α computed here with numpy-free float maths (α = 1 − σ on flow)."""
    out = []
    for sigma in (float(v) for v in sigmas):
        alpha = (1.0 - sigma) if flow else 1.0
        out.append(math.inf if alpha <= 0 else sigma / alpha)
    return out


def _expected_plan(sigmas, flow):
    """The plan from its definition: argmin over the allowed indices (first index on ties)."""
    n = len(sigmas) - 1
    if n < 20:
        return None
    s = _levels(sigmas, flow)
    lo = sorted(range(1, n), key=lambda j: (abs(s[j] - 0.1), j))[0]
    finite = [j for j in range(lo) if math.isfinite(s[j])]
    if not finite:
        return None
    hi = sorted(finite, key=lambda j: (abs(s[j] - 2.0), j))[0]
    if hi >= lo or s[hi] <= s[lo]:
        return None
    return RF.RestartPlan(lo=lo, hi=hi, times=2 if n >= 36 else 1)


class PlanTests(unittest.TestCase):
    def test_the_constants(self):
        self.assertEqual((RF.RESTART_S_MIN, RF.RESTART_S_MAX, RF.RESTART_MIN_STEPS, RF.RESTART_TIMES,
                          RF.RESTART_TIMES_LONG, RF.RESTART_LONG_FROM), (0.1, 2.0, 20, 1, 2, 36))

    def test_the_window_on_animas_shift_3_grid(self):
        known = {28: RF.RestartPlan(27, 17, 1), 36: RF.RestartPlan(35, 22, 2), 50: RF.RestartPlan(48, 30, 2)}
        for steps, expected in known.items():
            with self.subTest(steps=steps):
                sigmas = fx.flow_sigmas(steps)
                plan = RF.restart_plan(sigmas, True)
                self.assertEqual(plan, expected)
                self.assertEqual(plan, _expected_plan(sigmas, True))
                # inside flow σ ∈ (0, 1): the window never reaches pure noise
                self.assertLess(float(sigmas[plan.hi]), 0.7)
                self.assertGreater(float(sigmas[plan.hi]), 0.6)
                self.assertTrue(0.07 < float(sigmas[plan.lo]) < 0.12)

    def test_the_window_on_an_eps_karras_grid(self):
        for steps in (20, 28, 36, 60):
            with self.subTest(steps=steps):
                sigmas = fx.eps_sigmas(steps)
                plan = RF.restart_plan(sigmas, False)
                self.assertEqual(plan, _expected_plan(sigmas, False))
                self.assertLess(abs(float(sigmas[plan.lo]) - 0.1), 0.05)
                self.assertLess(abs(float(sigmas[plan.hi]) - 2.0), 0.4)
                self.assertEqual(plan.times, 2 if steps >= 36 else 1)

    def test_no_restart(self):
        self.assertIsNone(RF.restart_plan(fx.flow_sigmas(19), True))
        self.assertIsNone(RF.restart_plan(fx.eps_sigmas(19), False))
        # lo = 1 next to σ = 1 (s infinite): nowhere to restart from
        sigmas = torch.cat([torch.tensor([1.0]), torch.linspace(0.09, 0.0, 20)]).double()
        self.assertEqual(len(sigmas) - 1, 20)
        self.assertIsNone(RF.restart_plan(sigmas, True))
        # a flat grid: s_hi == s_lo
        self.assertIsNone(RF.restart_plan(torch.cat([torch.full((20,), 0.5), torch.zeros(1)]).double(), False))

    def test_k_by_step_count(self):
        for steps in (20, 35, 36, 80):
            with self.subTest(steps=steps):
                self.assertEqual(RF.restart_plan(fx.flow_sigmas(steps), True).times, 2 if steps >= 36 else 1)


class KernelTests(unittest.TestCase):
    def test_the_flow_kernel_maps_the_model_marginal_exactly(self):
        model = fx.GaussianModel(fx.FlowSampling(3.0), mu=0.3, c=0.5)
        for sigma_lo, sigma_hi in ((0.1, 0.66), (0.0909, 0.6667), (0.3, 0.5), (0.05, 0.95)):
            with self.subTest(sigma_lo=sigma_lo, sigma_hi=sigma_hi):
                scale, std = RF.restart_kernel(sigma_lo, sigma_hi, True)
                mean_lo, sd_lo = model.marginal(sigma_lo)
                mean_hi, sd_hi = model.marginal(sigma_hi)
                self.assertAlmostEqual(scale * mean_lo, mean_hi, delta=1e-12 * abs(mean_hi))
                self.assertAlmostEqual((scale * sd_lo) ** 2 + std ** 2, sd_hi ** 2, delta=1e-12 * sd_hi ** 2)
                # per data point too: N(α_lo·x0, σ_lo²) → N(α_hi·x0, σ_hi²)
                self.assertAlmostEqual(scale * (1 - sigma_lo), 1 - sigma_hi, delta=1e-15)
                self.assertAlmostEqual((scale * sigma_lo) ** 2 + std ** 2, sigma_hi ** 2, delta=1e-15)

    def test_forges_ve_kernel_does_not_on_flow(self):
        """Negative control (VERIFIED §1.3): A1111's ``x + ε·√(σ_hi² − σ_lo²)`` keeps the signal at α_lo."""
        model = fx.GaussianModel(fx.FlowSampling(3.0), mu=0.3, c=0.5)
        mean_lo, sd_lo = model.marginal(0.1)
        mean_hi, sd_hi = model.marginal(0.66)
        ve_std = (0.66 ** 2 - 0.1 ** 2) ** 0.5
        self.assertGreater(abs(mean_lo - mean_hi), 0.1)
        self.assertGreater(abs(sd_lo ** 2 + ve_std ** 2 - sd_hi ** 2), 0.05)
        self.assertEqual(RF.restart_kernel(0.1, 2.0, False), (1.0, (2.0 ** 2 - 0.1 ** 2) ** 0.5))


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))


class HeunParityTests(_Base):
    def test_without_a_restart_it_is_forges_heun(self):
        for flow in (False, True):
            for dtype in (torch.float32, torch.float64):
                for steps in (1, 2, 5, 19):
                    with self.subTest(flow=flow, dtype=dtype, steps=steps):
                        sampling = fx.FlowSampling(3.0) if flow else fx.EpsSampling()
                        sigmas = fx.flow_sigmas(steps, dtype=dtype) if flow else fx.eps_sigmas(steps, dtype=dtype)
                        x = fx.seeded((2, 4, 6, 5), 0, dtype=dtype) * (1.0 if flow else float(sigmas[0]))
                        noise = fx.NoiseSequence(x)
                        ours = RF.sample_restart_flow(fx.ToyModel(sampling), x.clone(), sigmas, disable=True,
                                                      noise_sampler=noise)
                        heun = self.ks.sample_heun(fx.ToyModel(sampling), x.clone(), sigmas, disable=True)
                        self.assertTrue(torch.equal(ours, heun))
                        self.assertEqual(noise.calls, 0)

    def test_the_main_pass_is_heun_until_the_restart(self):
        sampling, sigmas = fx.FlowSampling(3.0), fx.flow_sigmas(28)
        x = fx.seeded((2, 4, 6, 5), 0)
        ours, heun = [], []
        RF.sample_restart_flow(fx.ToyModel(sampling), x.clone(), sigmas, disable=True, callback=ours.append,
                               noise_sampler=fx.NoiseSequence(x))
        self.ks.sample_heun(fx.ToyModel(sampling), x.clone(), sigmas, disable=True, callback=heun.append)
        plan = RF.restart_plan(sigmas, True)
        for a, b in zip(ours[:plan.lo], heun[:plan.lo]):
            self.assertTrue(torch.equal(a["x"], b["x"]) and torch.equal(a["denoised"], b["denoised"]))


class ForgeRestartParityTests(_Base):
    """ε models: Forge's built-in Restart (A1111's code) given the same window on a Karras main grid."""

    def test_eps_is_forges_restart_with_the_same_window(self):
        extra = fx.forge_sd_samplers_extra()
        for steps in (20, 28, 40):
            for s_noise in (1.0, 0.8):
                with self.subTest(steps=steps, s_noise=s_noise):
                    sigmas = self.ks.get_sigmas_karras(steps, 0.0292, 14.6146)   # Forge's float32 Karras grid
                    plan = RF.restart_plan(sigmas, False)
                    self.assertEqual(plan.times, 2 if steps >= 36 else 1)
                    restart_list = {float(sigmas[plan.lo]): [plan.lo - plan.hi + 1, plan.times, float(sigmas[plan.hi])]}
                    x = fx.seeded((2, 4, 6, 5), 1, dtype=torch.float32) * float(sigmas[0])
                    forge_noise = fx.NoiseSequence(x, seed=5)
                    with fx.origin_torch(self.ks, forge_noise):
                        forge = extra.restart_sampler(fx.ToyModel(fx.EpsSampling()), x.clone(), sigmas, disable=True,
                                                      s_noise=s_noise, restart_list=restart_list)
                    our_noise = fx.NoiseSequence(x, seed=5)
                    with fx.origin_torch(self.ks, our_noise):
                        ours = RF.sample_restart_flow(fx.ToyModel(fx.EpsSampling()), x.clone(), sigmas, disable=True,
                                                      s_noise=s_noise)   # the default noise sampler: randn_like
                    self.assertEqual(our_noise.calls, plan.times)
                    self.assertGreaterEqual(forge_noise.calls, plan.times)
                    for a, b in zip(our_noise.drawn, forge_noise.drawn):
                        self.assertTrue(torch.equal(a, b))
                    self.assertTrue(torch.isfinite(ours).all())
                    torch.testing.assert_close(ours, forge, rtol=1e-5, atol=1e-6)
                    plain = self.ks.sample_heun(fx.ToyModel(fx.EpsSampling()), x.clone(), sigmas, disable=True)
                    self.assertGreater(float((ours - plain).abs().max()), 1e-3)   # the restart did something


class MarginalTests(_Base):
    """The exact denoiser of Gaussian data on Anima's flow grid (20 000 particles, 4 standard errors).

    Every step of Heun is affine in ``x`` on this model, so plain Heun maps the exact start ``N(0, 1)`` at σ = 1
    to exactly ``N(b, A²)`` (two probes give ``b`` and ``A``). A restart whose forward kernel keeps the model's
    marginals lands on the same distribution up to Heun's error inside the window. ``N(b, A²)`` is not the
    data distribution ``N(μ, c²)`` at 28 steps: Forge's Heun ends with one Euler step into σ = 0 — from
    σ = 0.1 there — which returns the posterior mean, whose variance is c² minus the posterior variance (4.5 %
    low, more than 4 standard errors). At 40 steps (last σ 0.071) the data distribution itself is within reach."""

    PARTICLES = 20000

    def _heun_target(self, model, sigmas) -> tuple:
        probes = torch.tensor([0.0, 1.0], dtype=torch.float64).reshape(2, 1, 1, 1)
        ends = self.ks.sample_heun(model, probes, sigmas, disable=True).flatten()
        return float(ends[0]), float(ends[1] - ends[0])

    def _restart(self, sigmas, seed=21):
        model = fx.GaussianModel(fx.FlowSampling(3.0), mu=0.3, c=0.5)
        x = fx.seeded((self.PARTICLES, 1, 1, 1), seed)
        noise = fx.NoiseSequence(x, seed=seed + 1)
        out = RF.sample_restart_flow(model, x, sigmas, disable=True, noise_sampler=noise).flatten()
        self.assertEqual(noise.calls, RF.restart_plan(sigmas, True).times)
        return out

    def _assert_distribution(self, out, mean, std):
        n = out.numel()
        self.assertLess(abs(float(out.mean()) - mean), 4 * std / math.sqrt(n))
        self.assertLess(abs(float(out.var()) - std ** 2), 4 * std ** 2 * math.sqrt(2.0 / (n - 1)))

    def test_restart_flow_keeps_the_marginals_heun_follows(self):
        for steps in (28, 40):
            with self.subTest(steps=steps):
                sigmas = fx.flow_sigmas(steps)
                self.assertIsNotNone(RF.restart_plan(sigmas, True))
                b, a = self._heun_target(fx.GaussianModel(fx.FlowSampling(3.0), mu=0.3, c=0.5), sigmas)
                self._assert_distribution(self._restart(sigmas), b, abs(a))

    def test_restart_flow_ends_in_the_data_distribution(self):
        self._assert_distribution(self._restart(fx.flow_sigmas(40)), 0.3, 0.5)

    def test_the_ve_kernel_would_not(self):
        """Negative control: the same sampler with A1111's VE jump on flow misses both distributions."""
        sampling = fx.FlowSampling(3.0)
        sigmas = fx.flow_sigmas(28)
        plan = RF.restart_plan(sigmas, True)
        model = fx.GaussianModel(sampling, mu=0.3, c=0.5)
        x = fx.seeded((self.PARTICLES, 1, 1, 1), 21)
        noise = fx.NoiseSequence(x, seed=22)
        s_in = x.new_ones([x.shape[0]])
        for i in range(len(sigmas) - 1):
            x = RF._heun_step(model, x, sigmas, i, s_in, {}, None, self.ks.to_d)
            if i + 1 == plan.lo:
                x = x + noise(None, None) * (sigmas[plan.hi] ** 2 - sigmas[plan.lo] ** 2) ** 0.5
                for j in range(plan.hi, plan.lo):
                    x = RF._heun_step(model, x, sigmas, j, s_in, {}, None, self.ks.to_d)
        out = x.flatten()
        n = out.numel()
        b, a = self._heun_target(fx.GaussianModel(sampling, mu=0.3, c=0.5), sigmas)
        self.assertGreater(abs(float(out.var()) - a * a), 8 * a * a * math.sqrt(2.0 / (n - 1)))
        self.assertGreater(abs(float(out.var()) - 0.25), 8 * 0.25 * math.sqrt(2.0 / (n - 1)))


class StepCounterTests(_Base):
    def test_the_counter_and_the_callback_follow_the_grid_step(self):
        """``offset`` 7: a later SPEED segment, called with Forge's counter already at 14 (seven Heun steps)."""
        for offset in (0, 7):
            with self.subTest(offset=offset):
                sigmas = fx.flow_sigmas(28)
                plan = RF.restart_plan(sigmas, True)
                model = fx.GaussianModel(fx.FlowSampling(3.0))
                model.step = 2 * offset
                seen = []

                def record(info, model=model, seen=seen):
                    seen.append((info["i"], model.step))

                x = fx.seeded((1, 1, 2, 2), 0)
                RF.sample_restart_flow(model, x, sigmas, disable=True, callback=record,
                                       noise_sampler=fx.NoiseSequence(x))
                grid = list(range(plan.lo)) + list(range(plan.hi, plan.lo)) * plan.times + list(range(plan.lo, 28))
                self.assertEqual([i for i, _ in seen], grid)
                # the callback runs after the step's first evaluation: the counter is 2·(j + offset) + 1 there
                self.assertEqual([step for _, step in seen], [2 * (i + offset) + 1 for i in grid])
                self.assertEqual(model.step, 2 * (27 + offset) + 1)   # the last step (into σ = 0) is one call
                self.assertEqual(len(model.calls), 2 * 28 - 1 + 2 * plan.times * (plan.lo - plan.hi))

    def test_the_main_pass_leaves_the_counter_as_forge_counts(self):
        """Without a restart the counter is exactly what Forge's denoiser counts by itself (the sets are no-ops)."""
        sigmas = fx.flow_sigmas(12)
        counted = fx.GaussianModel(fx.FlowSampling(3.0))
        counted.step = 0
        steps_seen, forge_seen = [], []
        x = fx.seeded((1, 1, 2, 2), 0)
        RF.sample_restart_flow(counted, x, sigmas, disable=True, callback=lambda info: steps_seen.append(counted.step))
        heun_model = fx.GaussianModel(fx.FlowSampling(3.0))
        heun_model.step = 0
        self.ks.sample_heun(heun_model, x, sigmas, disable=True, callback=lambda info: forge_seen.append(heun_model.step))
        self.assertEqual(steps_seen, forge_seen)
        self.assertEqual(counted.step, heun_model.step)

    def test_speeds_step_offset_is_not_declared(self):
        """Only the samplers with step-placed extras (Dy) take SPEED's ``sam_extra_step_offset``; the counter base
        comes from Forge's denoiser itself."""
        import inspect

        self.assertNotIn("sam_extra_step_offset", inspect.signature(RF.sample_restart_flow).parameters)

    def test_a_model_without_an_int_counter_is_left_alone(self):
        sigmas = fx.flow_sigmas(28)
        for value in (None, True, 2.0):
            with self.subTest(value=value):
                model = fx.GaussianModel(fx.FlowSampling(3.0))
                model.step = value
                x = fx.seeded((1, 1, 2, 2), 0)
                RF.sample_restart_flow(model, x, sigmas, disable=True, noise_sampler=fx.NoiseSequence(x))
                self.assertIs(model.step, value)


if __name__ == "__main__":
    unittest.main()
