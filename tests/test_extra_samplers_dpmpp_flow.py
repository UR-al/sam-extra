"""``DPM++ 2M SDE Heun`` and the three "flow ODE" DPM++ entries (``sam3ext.extra_samplers.dpmpp_flow``).

The wrappers call Forge's own ``sample_dpmpp_2m_sde`` / ``sample_dpmpp_3m_sde`` (Forge Neo's real
``modules_forge/packages/k_diffusion/sampling.py``, ``fx.forge_k_sampling()``), so the first group of tests
compares them with those functions called directly — bit for bit, on ε and flow models, float32/float64,
4-D and 5-D latents. The second group shows what the entries solve: on ``fx.GaussianModel`` (the exact
denoiser of Gaussian data, whose probability-flow ODE has a closed form) the ODE entries converge to the
exact end point at second order or better, on a shifted flow grid (Anima's shift 3) and on an ε Karras grid.

Forge-path checks (Forge's ``KDiffusionSampler.sample``: "DPM++ 3M (flow ODE)" == "DPM++ 3M SDE" at Eta 0, the
Eta infotext) are in ``test_extra_samplers_forge_path.py``; the table entries in ``test_extra_samplers_registry.py``.
"""

from __future__ import annotations

import importlib.util
import inspect
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

from sam3ext.extra_samplers import dpmpp_flow  # noqa: E402

STEPS = (1, 2, 3, 6, 28)
SHAPES = ((2, 4, 5, 6), (1, 4, 2, 5, 6))   # 4-D and Anima-like 5-D (B, C, T, H, W)
DTYPES = (torch.float32, torch.float64)
ODE_ENTRIES = (
    # (wrapper, Forge function name, solver_type or None)
    (dpmpp_flow.sample_dpmpp_2m_flow_ode, "sample_dpmpp_2m_sde", "midpoint"),
    (dpmpp_flow.sample_dpmpp_2m_heun_flow_ode, "sample_dpmpp_2m_sde", "heun"),
    (dpmpp_flow.sample_dpmpp_3m_flow_ode, "sample_dpmpp_3m_sde", None),
)


def _sampling(flow: bool):
    return fx.FlowSampling() if flow else fx.EpsSampling()


def _sigmas(flow: bool, steps: int, dtype):
    return fx.flow_sigmas(steps, dtype=dtype) if flow else fx.eps_sigmas(steps, dtype=dtype)


def _latent(shape, flow: bool, sigmas, dtype, seed=0):
    x = fx.seeded(shape, seed, dtype=dtype)
    return x if flow else x * float(sigmas[0])


class _ForgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))


# ---------------------------------------------------------------------------
# The closed-form model the convergence tests rely on
# ---------------------------------------------------------------------------


class GaussianModelTests(unittest.TestCase):
    def test_the_closed_form_solves_the_models_probability_flow_ode(self):
        """d/dσ of ``ode_transport`` equals the ODE's velocity ``(x − D)/σ`` along the trajectory (central
        differences in float64) — on flow (α = 1 − σ) and ε (α = 1) models."""
        x0 = fx.seeded((3, 4, 5), 1)
        for flow, start in ((True, 0.97), (False, 9.0)):
            model = fx.GaussianModel(_sampling(flow))
            x_start = model.marginal(torch.tensor(start, dtype=torch.float64))[0] + x0 * float(
                model.marginal(torch.tensor(start, dtype=torch.float64))[1])
            for sigma in (0.9, 0.5, 0.2, 0.05) if flow else (5.0, 1.0, 0.3, 0.05):
                with self.subTest(flow=flow, sigma=sigma):
                    s = torch.tensor(sigma, dtype=torch.float64)
                    eps = 1e-6 * sigma
                    x = model.ode_transport(x_start, torch.tensor(start, dtype=torch.float64), s)
                    up = model.ode_transport(x_start, torch.tensor(start, dtype=torch.float64), s + eps)
                    down = model.ode_transport(x_start, torch.tensor(start, dtype=torch.float64), s - eps)
                    derivative = (up - down) / (2 * eps)
                    velocity = (x - model(x, s.reshape(1))) / s
                    torch.testing.assert_close(derivative, velocity, rtol=1e-7, atol=1e-8)

    def test_a_fine_euler_run_converges_to_the_closed_form_at_first_order(self):
        model = fx.GaussianModel(fx.FlowSampling())
        x = fx.seeded((2, 4, 6), 2)
        errors = []
        for steps in (512, 1024, 2048):
            sigmas = fx.time_snr_shift(3.0, torch.linspace(0.98, 0.05, steps + 1, dtype=torch.float64))
            xt = x.clone()
            for i in range(steps):
                xt = xt + (xt - model(xt, sigmas[i].reshape(1))) / sigmas[i] * (sigmas[i + 1] - sigmas[i])
            errors.append(float((xt - model.ode_transport(x, sigmas[0], sigmas[-1])).abs().max()))
        self.assertLess(errors[-1], 2e-3)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertAlmostEqual(coarse / fine, 2.0, delta=0.1)

    def test_the_marginal_is_the_data_distribution_carried_by_the_forward_process(self):
        for flow in (True, False):
            model = fx.GaussianModel(_sampling(flow))
            sigma = torch.tensor(0.4, dtype=torch.float64)
            alpha = 1 - sigma if flow else torch.tensor(1.0, dtype=torch.float64)
            mean, std = model.marginal(sigma)
            self.assertAlmostEqual(float(mean), float(alpha * model.mu), places=15)
            self.assertAlmostEqual(float(std ** 2), float(alpha ** 2 * model.c ** 2 + sigma ** 2), places=15)

    def test_the_step_counter_counts_calls_only_when_set(self):
        model = fx.GaussianModel(fx.FlowSampling())
        model(torch.zeros(1, 2), torch.tensor([0.5]))
        self.assertIsNone(model.step)
        model.step = 0
        model(torch.zeros(1, 2), torch.tensor([0.5]))
        self.assertEqual((model.step, len(model.calls)), (1, 2))


# ---------------------------------------------------------------------------
# The wrappers are Forge's functions
# ---------------------------------------------------------------------------


class ForgeFunctionTests(_ForgeBase):
    def test_dpmpp_2m_sde_heun_is_forges_2m_sde_with_the_heun_solver(self):
        for flow in (False, True):
            for dtype in DTYPES:
                for shape in SHAPES:
                    for steps in STEPS:
                        for eta in (0.0, 0.5, 1.0):
                            for s_noise in (0.0, 1.0):
                                sigmas = _sigmas(flow, steps, dtype)
                                x = _latent(shape, flow, sigmas, dtype)
                                with self.subTest(flow=flow, dtype=dtype, shape=shape, steps=steps, eta=eta,
                                                  s_noise=s_noise):
                                    ours_noise, forge_noise = fx.NoiseSequence(x, 7), fx.NoiseSequence(x, 7)
                                    seen, forge_seen = [], []
                                    ours = dpmpp_flow.sample_dpmpp_2m_sde_heun(
                                        fx.ToyModel(_sampling(flow)), x.clone(), sigmas, disable=True,
                                        callback=lambda d: seen.append((d["i"], d["x"])),
                                        eta=eta, s_noise=s_noise, noise_sampler=ours_noise)
                                    forge = self.ks.sample_dpmpp_2m_sde(
                                        fx.ToyModel(_sampling(flow)), x.clone(), sigmas, disable=True,
                                        callback=lambda d: forge_seen.append((d["i"], d["x"])),
                                        eta=eta, s_noise=s_noise, noise_sampler=forge_noise, solver_type="heun")
                                    self.assertTrue(torch.equal(ours, forge))
                                    self.assertEqual(ours.dtype, dtype)
                                    self.assertEqual(ours_noise.calls, forge_noise.calls)
                                    self.assertEqual([i for i, _x in seen], list(range(steps)))
                                    self.assertTrue(all(torch.equal(a, b) for (_i, a), (_j, b) in zip(seen, forge_seen)))

    def test_the_heun_solver_is_not_the_midpoint_one(self):
        sigmas = fx.flow_sigmas(6)
        x = fx.seeded((1, 4, 5, 5), 3)
        heun = dpmpp_flow.sample_dpmpp_2m_sde_heun(fx.ToyModel(fx.FlowSampling()), x.clone(), sigmas, disable=True,
                                                   eta=0.0)
        midpoint = self.ks.sample_dpmpp_2m_sde(fx.ToyModel(fx.FlowSampling()), x.clone(), sigmas, disable=True,
                                               eta=0.0, noise_sampler=fx.NoiseSequence(x))
        self.assertFalse(torch.equal(heun, midpoint))

    def test_the_flow_ode_entries_are_forges_sde_functions_at_eta_zero(self):
        """``ODE entry == Forge's SDE function at Eta 0`` — whatever noise sampler Forge would hand the SDE
        (its Brownian tree, or any other): at η = 0 the SDE functions never draw."""
        for wrapper, name, solver_type in ODE_ENTRIES:
            forge_function = getattr(self.ks, name)
            extra = {} if solver_type is None else {"solver_type": solver_type}
            for flow in (False, True):
                for dtype in DTYPES:
                    for shape in SHAPES:
                        for steps in STEPS:
                            sigmas = _sigmas(flow, steps, dtype)
                            x = _latent(shape, flow, sigmas, dtype, seed=steps)
                            with self.subTest(sampler=wrapper.__name__, flow=flow, dtype=dtype, shape=shape,
                                              steps=steps):
                                ours = wrapper(fx.ToyModel(_sampling(flow)), x.clone(), sigmas, disable=True)
                                never = fx.NoiseSequence(x)
                                forge = forge_function(fx.ToyModel(_sampling(flow)), x.clone(), sigmas, disable=True,
                                                       eta=0.0, s_noise=1.0, noise_sampler=never, **extra)
                                self.assertTrue(torch.equal(ours, forge))
                                self.assertEqual(never.calls, 0)
                                if steps >= 2:   # Forge's per-image Brownian tree, as KDiffusionSampler builds it
                                    positive = sigmas[sigmas > 0]
                                    tree = self.ks.BrownianTreeNoiseSampler(x, positive.min(), positive.max(),
                                                                            seed=[1] * x.shape[0])
                                    forge_tree = forge_function(fx.ToyModel(_sampling(flow)), x.clone(), sigmas,
                                                                disable=True, eta=0.0, s_noise=1.0,
                                                                noise_sampler=tree, **extra)
                                    self.assertTrue(torch.equal(ours, forge_tree))

    def test_the_flow_ode_entries_draw_nothing_and_build_no_brownian_tree(self):
        def refuse(*args, **kwargs):
            raise AssertionError("a Brownian tree was built")

        original = self.ks.BrownianTreeNoiseSampler
        self.ks.BrownianTreeNoiseSampler = refuse
        try:
            for wrapper, _name, _solver in ODE_ENTRIES:
                with self.subTest(sampler=wrapper.__name__):
                    x = fx.seeded((1, 4, 5, 5), 4)
                    out = wrapper(fx.ToyModel(fx.FlowSampling()), x, fx.flow_sigmas(8), disable=True)
                    self.assertTrue(torch.isfinite(out).all())
        finally:
            self.ks.BrownianTreeNoiseSampler = original
        with self.assertRaisesRegex(RuntimeError, "drew noise"):
            dpmpp_flow._no_noise(1.0, 0.5)

    def test_signatures(self):
        for wrapper, _name, _solver in ODE_ENTRIES:
            parameters = inspect.signature(wrapper).parameters
            self.assertEqual(list(parameters), ["model", "x", "sigmas", "extra_args", "callback", "disable"])
        heun = inspect.signature(dpmpp_flow.sample_dpmpp_2m_sde_heun).parameters
        self.assertEqual(list(heun), ["model", "x", "sigmas", "extra_args", "callback", "disable", "eta", "s_noise",
                                      "noise_sampler"])
        self.assertEqual((heun["eta"].default, heun["s_noise"].default), (1.0, 1.0))

    def test_on_eps_models_the_2m_entries_are_forges_dpmpp_2m_and_res_multistep(self):
        """λ = −log σ on ε models: the midpoint 2M ODE is Forge's ``DPM++ 2M`` and the Heun one Forge's
        ``Res Multistep`` (RES 2M), up to the order of the floating-point operations (float64 here)."""
        for steps in (2, 3, 6, 28):
            sigmas = fx.eps_sigmas(steps)
            x = fx.seeded((2, 4, 5, 5), 5) * float(sigmas[0])
            with self.subTest(steps=steps):
                ours = dpmpp_flow.sample_dpmpp_2m_flow_ode(fx.ToyModel(fx.EpsSampling()), x.clone(), sigmas,
                                                           disable=True)
                forge = self.ks.sample_dpmpp_2m(fx.ToyModel(fx.EpsSampling()), x.clone(), sigmas, disable=True)
                torch.testing.assert_close(ours, forge, rtol=1e-12, atol=1e-12)
                ours_heun = dpmpp_flow.sample_dpmpp_2m_heun_flow_ode(fx.ToyModel(fx.EpsSampling()), x.clone(),
                                                                     sigmas, disable=True)
                res = self.ks.sample_res_multistep(fx.ToyModel(fx.EpsSampling()), x.clone(), sigmas, disable=True,
                                                   noise_sampler=fx.NoiseSequence(x))
                torch.testing.assert_close(ours_heun, res, rtol=1e-12, atol=1e-12)

    def test_on_flow_models_they_step_in_the_flow_half_log_snr(self):
        """On a flow model they are *not* Forge's DPM++ 2M / Res Multistep (those step in −log σ)."""
        sigmas = fx.flow_sigmas(8)
        x = fx.seeded((1, 4, 5, 5), 6)
        ours = dpmpp_flow.sample_dpmpp_2m_flow_ode(fx.ToyModel(fx.FlowSampling()), x.clone(), sigmas, disable=True)
        ve = self.ks.sample_dpmpp_2m(fx.ToyModel(fx.FlowSampling()), x.clone(), sigmas, disable=True)
        self.assertGreater(float((ours - ve).abs().max()), 1e-3)


# ---------------------------------------------------------------------------
# What they solve
# ---------------------------------------------------------------------------


def _flow_grid(steps: int) -> torch.Tensor:
    """A shift-3 flow grid (Anima's shift) from σ 0.9933 down to 0.1364, ending above 0."""
    return fx.time_snr_shift(3.0, torch.linspace(0.98, 0.05, steps + 1, dtype=torch.float64))


def _eps_grid(steps: int) -> torch.Tensor:
    """A Karras ρ 7 grid from 14.6 down to 0.03, without the final 0."""
    return fx.eps_sigmas(steps)[:-1]


def _euler(model, x, sigmas):
    for i in range(len(sigmas) - 1):
        x = x + (x - model(x, sigmas[i].reshape(1))) / sigmas[i] * (sigmas[i + 1] - sigmas[i])
    return x


class ConvergenceTests(_ForgeBase):
    """Error against the closed-form end point of the probability-flow ODE (``GaussianModel``).

    The grids end above σ = 0 so the comparison is with the exact ODE solution, not with the last step's
    x0 prediction, and the flow grid starts below σ = 1 (Forge's ``offset_first_sigma_for_snr`` only moves a
    first σ of 1). Each doubling of the step count must cut the error by at least 3× (second order or better;
    Euler: about 2×). DPM-Solver++(3M) is third order per step, but its two start-up steps are of order 1 and
    2, which caps the global order at 2 on a fixed grid — it shows as a smaller error than 2M's at every count,
    not as a steeper slope."""

    COUNTS = (32, 64, 128)

    def _errors(self, flow: bool) -> dict:
        grid = _flow_grid if flow else _eps_grid
        errors = {"Euler": []}
        for wrapper, _name, _solver in ODE_ENTRIES:
            errors[wrapper.__name__] = []
        for steps in self.COUNTS:
            sigmas = grid(steps)
            model = fx.GaussianModel(_sampling(flow))
            mean, std = model.marginal(sigmas[0])
            x = mean + std * fx.seeded((2, 4, 5, 5), 9)   # a sample of the model's marginal at σ₀
            exact = model.ode_transport(x, sigmas[0], sigmas[-1])
            errors["Euler"].append(float((_euler(model, x.clone(), sigmas) - exact).abs().max()))
            for wrapper, _name, _solver in ODE_ENTRIES:
                out = wrapper(fx.GaussianModel(_sampling(flow)), x.clone(), sigmas, disable=True)
                errors[wrapper.__name__].append(float((out - exact).abs().max()))
        return errors

    def test_second_order_convergence_to_the_exact_flow_ode_solution(self):
        for flow in (True, False):
            errors = self._errors(flow)
            for name, values in errors.items():
                with self.subTest(flow=flow, sampler=name, errors=values):
                    ratios = [coarse / fine for coarse, fine in zip(values, values[1:])]
                    if name == "Euler":
                        self.assertTrue(all(1.8 < ratio < 2.2 for ratio in ratios), ratios)
                    else:
                        self.assertTrue(all(ratio >= 3.0 for ratio in ratios), ratios)
                        self.assertTrue(all(ours < euler for ours, euler in zip(values, errors["Euler"])))
            with self.subTest(flow=flow, check="3M below 2M"):
                three, two = errors["sample_dpmpp_3m_flow_ode"], errors["sample_dpmpp_2m_flow_ode"]
                self.assertTrue(all(a < b for a, b in zip(three, two)), (three, two))

    def test_the_last_step_to_zero_returns_the_models_x0(self):
        sigmas = fx.flow_sigmas(10)
        for wrapper, _name, _solver in ODE_ENTRIES:
            with self.subTest(sampler=wrapper.__name__):
                model = fx.GaussianModel(fx.FlowSampling())
                seen = []
                out = wrapper(model, fx.seeded((1, 4, 5, 5), 10), sigmas, disable=True,
                              callback=lambda d: seen.append(d["denoised"]))
                self.assertTrue(torch.equal(out, seen[-1]))
                self.assertEqual(len(model.calls), 10)


if __name__ == "__main__":
    unittest.main()
