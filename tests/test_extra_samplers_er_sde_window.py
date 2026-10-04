"""The ER SDE noise window (``sam3ext/extra_samplers/er_sde_window.py``) on Forge's own ``sample_er_sde``.

Forge's ``modules_forge/packages/k_diffusion/sampling.py`` and its predictors
(``backend/modules/k_prediction.py``: ``PredictionDiscreteFlow`` at Anima's shift 3, ``Prediction`` with
SDXL's ε schedule) are executed read-only from Forge's files. What is pinned down:

* a full window (start 0, end 1) is the unwindowed run, an empty one (start ≥ end) the ODE run with
  ``s_noise = 0`` — bit for bit, and the empty one draws nothing;
* a mixed window equals an independent float64 ER-SDE-Solver written here from the paper (stages 1-3,
  200-point quadrature, the scaler chosen per step), and draws exactly on the steps whose target σ lies
  in ``[percent_to_sigma(end), percent_to_sigma(start)]``;
* on the exact denoiser of Gaussian data the windowed SDE still ends in the data distribution
  (marginals kept);
* the mechanism's assumption — Forge (and ComfyUI) call ``callback`` before the first ``noise_scaler`` use
  of every step — is checked in the source, and the callback wrapper forwards every step unchanged.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import sys
import textwrap
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

from sam3ext.extra_samplers import er_sde, er_sde_window  # noqa: E402

_SAMPLING = "modules_forge/packages/k_diffusion/sampling.py"


class RecordingNoise:
    """A seeded noise sampler that records the ``(sigma, sigma_next)`` of every draw."""

    def __init__(self, like: torch.Tensor, seed: int = 7):
        self.inner = fx.NoiseSequence(like, seed=seed)
        self.sigmas: list = []

    def __call__(self, sigma, sigma_next):
        self.sigmas.append((float(sigma), float(sigma_next)))
        return self.inner(sigma, sigma_next)

    @property
    def calls(self) -> int:
        return len(self.sigmas)


def _flow_problem(steps=12, dtype=torch.float64, shape=(2, 4, 6, 5), seed=0):
    predictor = fx.forge_flow_predictor(3.0)
    sigmas = fx.flow_sigmas(steps, dtype=dtype)
    return predictor, sigmas, fx.seeded(shape, seed, dtype=dtype)


def _eps_problem(steps=12, dtype=torch.float64, shape=(2, 4, 6, 5), seed=0):
    predictor = fx.forge_eps_predictor()
    sigmas = fx.eps_sigmas(steps, sigma_max=14.614641189575195, sigma_min=0.029167158529162407, dtype=dtype)
    return predictor, sigmas, fx.seeded(shape, seed, dtype=dtype) * float(sigmas[0])


def _entry(kind):
    return {"tunable": er_sde.sample_er_sde_tunable, "reverse": er_sde.sample_er_sde_reverse_time}[kind]


class _ForgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))

    def _run(self, kind, problem, *, eta=1.0, stage=3, window=None, s_noise=1.0, seed=7, callback=None):
        predictor, sigmas, x = problem
        model = fx.ToyModel(predictor)
        noise = RecordingNoise(x, seed=seed)
        out = _entry(kind)(model, x.clone(), sigmas, extra_args={}, disable=True, callback=callback,
                           s_noise=s_noise, noise_sampler=noise, max_stage=stage, er_sde_eta=eta,
                           er_sde_window=window)
        return out, noise, model


class WindowBoundsTests(unittest.TestCase):
    def test_bounds_come_from_the_models_percent_to_sigma(self):
        flow = fx.forge_flow_predictor(3.0)
        hi, lo = er_sde_window.window_bounds(flow, 0.2, 0.8)
        self.assertAlmostEqual(hi, 0.923076923076923, places=12)   # Anima shift 3: σ 0.923 … 0.4286
        self.assertAlmostEqual(lo, 0.428571428571428, places=12)
        self.assertEqual(er_sde_window.window_bounds(flow, 0.0, 1.0), (1.0, 0.0))
        eps = fx.forge_eps_predictor()
        hi, lo = er_sde_window.window_bounds(eps, 0.0, 1.0)
        self.assertEqual((hi, lo), (999999999.9, 0.0))

    def test_a_predictor_without_percent_to_sigma_is_refused(self):
        with self.assertRaisesRegex(ValueError, "percent_to_sigma"):
            er_sde_window.window_bounds(fx.EpsSampling(), 0.2, 0.8)

    def test_steps_inside_and_the_empty_window(self):
        sigmas = torch.tensor([1.0, 0.95, 0.9, 0.6, 0.43, 0.42, 0.1, 0.0], dtype=torch.float64)
        self.assertEqual(er_sde_window.in_window_steps(sigmas, 0.2, 0.8, 0.923, 0.4286),
                         [False, True, True, True, False, False, False])
        self.assertEqual(er_sde_window.in_window_steps(sigmas, 0.8, 0.2, 0.923, 0.4286), [False] * 7)
        self.assertEqual(er_sde_window.in_window_steps(sigmas, 0.5, 0.5, 0.75, 0.75), [False] * 7)
        # the bounds are inclusive
        self.assertEqual(er_sde_window.in_window_steps(sigmas, 0.1, 0.9, 0.9, 0.43),
                         [False, True, True, True, False, False, False])

    def test_the_scaler_refuses_to_run_before_the_steps_callback(self):
        wrappers = er_sde_window.WindowedErSde([True], er_sde.er_sde_noise_scaler(1.0), er_sde.ode_noise_scaler,
                                               lambda s, t: None, torch.zeros(1))
        with self.assertRaises(RuntimeError):
            wrappers.scaler(torch.ones(1))
        with self.assertRaises(RuntimeError):
            wrappers.noise_sampler(1.0, 0.5)


class WindowParityTests(_ForgeBase):
    def test_a_full_window_is_the_unwindowed_run(self):
        for kind in ("tunable", "reverse"):
            for make in (_flow_problem, _eps_problem):
                for dtype in (torch.float32, torch.float64):
                    for stage in (1, 2, 3):
                        with self.subTest(kind=kind, problem=make.__name__, dtype=dtype, stage=stage):
                            problem = make(dtype=dtype)
                            plain, plain_noise, _ = self._run(kind, problem, eta=1.5, stage=stage)
                            full, noise, _ = self._run(kind, problem, eta=1.5, stage=stage, window=(0.0, 1.0))
                            self.assertTrue(torch.isfinite(full).all())
                            self.assertTrue(torch.equal(full, plain))
                            self.assertEqual(noise.sigmas, plain_noise.sigmas)
                            self.assertEqual(noise.calls, len(problem[1]) - 2)   # every step but the last

    def test_an_empty_window_is_the_ode_run_and_draws_nothing(self):
        for kind in ("tunable", "reverse"):
            for make in (_flow_problem, _eps_problem):
                for window in ((0.8, 0.2), (0.5, 0.5), (1.0, 0.0)):
                    with self.subTest(kind=kind, problem=make.__name__, window=window):
                        problem = make()
                        predictor, sigmas, x = problem
                        ode = self.ks.sample_er_sde(fx.ToyModel(predictor), x.clone(), sigmas, extra_args={},
                                                    disable=True, s_noise=0.0, noise_scaler=er_sde.ode_noise_scaler)
                        self.assertTrue(torch.equal(ode, er_sde.sample_er_sde_ode(fx.ToyModel(predictor), x.clone(),
                                                                                 sigmas, extra_args={}, disable=True)))
                        empty, noise, _ = self._run(kind, problem, eta=2.0, window=window)
                        self.assertTrue(torch.equal(empty, ode))
                        self.assertEqual(noise.calls, 0)

    def test_draws_happen_exactly_on_the_steps_inside_the_window(self):
        cases = (
            # (problem, window): the expected σ range comes from the formulas, not from the module
            (_flow_problem, (0.2, 0.8), (fx.time_snr_shift(3.0, 0.2), fx.time_snr_shift(3.0, 0.8))),
            (_flow_problem, (0.05, 0.5), (fx.time_snr_shift(3.0, 0.5), fx.time_snr_shift(3.0, 0.95))),
            (_eps_problem, (0.2, 0.8), None),
        )
        for make, window, flow_range in cases:
            for kind in ("tunable", "reverse"):
                with self.subTest(problem=make.__name__, window=window, kind=kind):
                    problem = make(steps=16)
                    predictor, sigmas, _x = problem
                    if flow_range is None:
                        lo, hi = predictor.percent_to_sigma(window[1]), predictor.percent_to_sigma(window[0])
                        self.assertAlmostEqual(hi, 5.0924, places=3)   # Forge's SDXL table
                        self.assertAlmostEqual(lo, 0.5712, places=3)
                    else:
                        lo, hi = flow_range
                    # Forge moves a flow schedule's first σ = 1 to percent_to_sigma(1e-4) (offset_first_sigma_for_snr)
                    first = float(predictor.percent_to_sigma(1e-4)) if float(sigmas[0]) >= 1.0 else float(sigmas[0])
                    starts = [first] + [float(s) for s in sigmas[1:]]
                    expected = [(starts[i], float(sigmas[i + 1])) for i in range(len(sigmas) - 2)
                                if lo <= float(sigmas[i + 1]) <= hi]
                    _out, noise, _ = self._run(kind, problem, eta=1.0, window=window)
                    self.assertGreater(len(expected), 2)
                    self.assertLess(len(expected), len(sigmas) - 2)
                    self.assertEqual([t for _, t in noise.sigmas], [t for _, t in expected])
                    self.assertEqual([s for s, _ in noise.sigmas], [s for s, _ in expected])
                    self.assertEqual(noise.calls, len(expected))

    def test_a_mixed_window_matches_the_papers_solver_in_float64(self):
        for kind in ("tunable", "reverse"):
            for make in (_flow_problem, _eps_problem):
                for stage in (1, 2, 3):
                    for eta, window in ((1.0, (0.2, 0.8)), (0.5, (0.0, 0.5)), (2.0, (0.3, 1.0))):
                        with self.subTest(kind=kind, problem=make.__name__, stage=stage, eta=eta, window=window):
                            problem = make(steps=14)
                            predictor, sigmas, x = problem
                            ours, noise, _ = self._run(kind, problem, eta=eta, stage=stage, window=window, s_noise=0.9)
                            reference, ref_draws = _paper_er_sde(
                                fx.ToyModel(predictor), x.clone(), sigmas, predictor, kind, eta, window, stage,
                                s_noise=0.9, noise=RecordingNoise(x, seed=7))
                            self.assertEqual(noise.calls, ref_draws)
                            self.assertGreater(ref_draws, 0)
                            self.assertEqual(ours.dtype, torch.float64)
                            # relative 1e-12 of the latent's scale (an elementwise rtol would fail on entries that
                            # happen to sit near 0, where 1e-16 rounding is a large fraction)
                            error = float((ours - reference).abs().max())
                            self.assertLessEqual(error, 1e-12 * float(reference.abs().max()))

    def test_the_callback_wrapper_forwards_every_step_unchanged(self):
        problem = _flow_problem()
        plain_seen = []
        self._run("tunable", problem, callback=plain_seen.append)
        for window in ((0.0, 1.0), (0.2, 0.8)):
            with self.subTest(window=window):
                seen = []
                self._run("tunable", problem, window=window, callback=seen.append)
                self.assertEqual([info["i"] for info in seen], list(range(len(problem[1]) - 1)))
                self.assertEqual([sorted(info) for info in seen], [sorted(info) for info in plain_seen])
                if window == (0.0, 1.0):   # the same run as without a window: the same callback contents
                    for a, b in zip(seen, plain_seen):
                        self.assertTrue(torch.equal(a["x"], b["x"]) and torch.equal(a["denoised"], b["denoised"]))
                        self.assertEqual(float(a["sigma"]), float(b["sigma"]))
        # without a real callback the wrapper still tracks the step (no error, finite result)
        out, _, _ = self._run("tunable", _flow_problem(), window=(0.2, 0.8), callback=None)
        self.assertTrue(torch.isfinite(out).all())

    def test_a_predictor_without_percent_to_sigma_raises_from_the_sampler(self):
        sigmas = fx.eps_sigmas(6)
        x = fx.seeded((1, 4, 4, 4), 0) * float(sigmas[0])
        for kind in ("tunable", "reverse"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "percent_to_sigma"):
                _entry(kind)(fx.ToyModel(fx.EpsSampling()), x, sigmas, disable=True, er_sde_window=(0.2, 0.8))

    def test_the_default_noise_is_forges_per_image_stream(self):
        """Without a noise sampler the windowed run draws from ``default_noise_sampler(x)`` (Forge's
        ``torch.randn_like`` through the module's ``torch`` — the per-image ``TorchHijack``) like the plain run."""
        problem = _flow_problem()
        predictor, sigmas, x = problem
        runs = []
        for window in (None, (0.0, 1.0)):
            torch.manual_seed(1234)
            runs.append(er_sde.sample_er_sde_tunable(fx.ToyModel(predictor), x.clone(), sigmas, disable=True,
                                                     er_sde_window=window))
        self.assertTrue(torch.equal(runs[0], runs[1]))


class WindowMarginalTests(_ForgeBase):
    """The exact denoiser of Gaussian data: every windowed SDE ends in ``N(μ, c²)`` (4 standard errors)."""

    PARTICLES = 20000
    STEPS = 100

    def test_the_windowed_sde_keeps_the_data_distribution(self):
        predictor = fx.forge_flow_predictor(3.0)
        sigmas = fx.flow_sigmas(self.STEPS, dtype=torch.float64)
        for kind in ("tunable", "reverse"):
            for eta in (0.5, 1.0, 2.0):
                with self.subTest(kind=kind, eta=eta):
                    model = fx.GaussianModel(predictor, mu=0.3, c=0.5)
                    x = fx.seeded((self.PARTICLES, 1, 1, 1), 11, dtype=torch.float64)
                    noise = fx.NoiseSequence(x, seed=12)
                    out = _entry(kind)(model, x, sigmas, extra_args={}, disable=True, noise_sampler=noise,
                                       er_sde_eta=eta, er_sde_window=(0.2, 0.8)).flatten()
                    self.assertGreater(noise.calls, 10)
                    n = out.numel()
                    mean, var = float(out.mean()), float(out.var())
                    self.assertLess(abs(mean - 0.3), 4 * 0.5 / math.sqrt(n))
                    self.assertLess(abs(var - 0.25), 4 * 0.25 * math.sqrt(2.0 / (n - 1)))


class SourceOrderTests(unittest.TestCase):
    """D9's assumption: in every step the callback runs before the first ``noise_scaler``/``noise_sampler`` use."""

    def _check(self, source: str):
        tree = ast.parse(textwrap.dedent(source))
        func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "sample_er_sde")
        loop = next(node for node in ast.walk(func) if isinstance(node, ast.For))

        def calls(name):
            return sorted(node.lineno for node in ast.walk(loop)
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name)

        callback, scaler, sampler, model = calls("callback"), calls("noise_scaler"), calls("noise_sampler"), calls("model")
        self.assertEqual(len(callback), 1)
        self.assertEqual(len(model), 1)
        self.assertTrue(scaler and sampler)
        self.assertLess(model[0], callback[0])
        self.assertLess(callback[0], min(scaler + sampler))
        # no scaler or noise use before the loop either (only the default-scaler definition)
        before = [node for node in ast.walk(func) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in ("noise_scaler", "noise_sampler") and node.lineno < loop.lineno]
        self.assertEqual(before, [])
        # the callback receives the step index under "i"
        call = next(node for node in ast.walk(loop) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "callback")
        keys = [key.value for key in call.args[0].keys]
        self.assertIn("i", keys)

    def test_forges_sample_er_sde(self):
        self._check(fx.forge_definition(_SAMPLING, "sample_er_sde"))

    def test_comfyuis_sample_er_sde(self):
        """ComfyUI's own copy (verbatim at 36c0b0a6 in tests/_origin_comfyui_er_sde.py) has the same order —
        what the app's node pack relies on to wrap it the same way."""
        origin = fx.load_comfy_er_sde_origin()
        self._check(inspect.getsource(origin.sample_er_sde))


def _paper_er_sde(model, x, sigmas, predictor, kind, eta, window, max_stage, *, s_noise, noise):
    """ER-SDE-Solver-1/2/3 (Cui et al., arXiv:2309.06169, eq. 18-20 with the official 200-point quadrature),
    written here independently in float64 with the noise scale function chosen per step: the SDE one on the
    steps whose target σ is inside the window, ``φ(λ) = λ`` (no noise) elsewhere. Returns ``(x, draws)``."""
    flow = predictor.prediction_type == "const"
    sig = [float(s) for s in sigmas]
    if flow and sig[0] >= 1.0:
        sig[0] = float(predictor.percent_to_sigma(1e-4))
    alpha = [(1.0 - s) if flow else 1.0 for s in sig]
    lam = [torch.tensor(s / a if a != 0 else math.inf, dtype=torch.float64) for s, a in zip(sig, alpha)]
    if kind == "tunable":
        def sde(t):
            return t * ((t ** 0.3).exp() + 10.0) ** eta
    else:
        def sde(t):
            return t ** (eta + 1)

    def ode(t):
        return t

    start, end = window
    hi, lo = float(predictor.percent_to_sigma(start)), float(predictor.percent_to_sigma(end))
    # The quadrature nodes are float32, as in the official solver's port in ComfyUI/Forge: the node indices are a
    # float32 ``arange`` and torch's type promotion keeps ``λ_t + k·Δ`` (a 0-dim float64 times a float32 vector)
    # in float32, so the two integrals carry float32 rounding. Everything else here is float64.
    points = torch.arange(200, dtype=torch.float32)
    s_in = x.new_ones([x.shape[0]])
    d_prev = d1_prev = None
    draws = 0
    for i in range(len(sig) - 1):
        denoised = model(x, torch.tensor(sig[i], dtype=torch.float64) * s_in)
        if sig[i + 1] == 0:
            x = denoised
            break
        inside = start < end and lo <= sig[i + 1] <= hi
        phi = sde if inside else ode
        l_s, l_t = lam[i], lam[i + 1]
        ratio = phi(l_t) / phi(l_s)
        x = (alpha[i + 1] / alpha[i]) * ratio * x + alpha[i + 1] * (1 - ratio) * denoised
        stage = min(max_stage, i + 1)
        if stage >= 2:
            width = (l_s - l_t) / 200.0
            pos = l_t + points * width
            integral_1 = torch.sum(1.0 / phi(pos)) * width
            d1 = (denoised - d_prev) / (l_s - lam[i - 1])
            x = x + alpha[i + 1] * ((l_t - l_s) + integral_1 * phi(l_t)) * d1
            if stage >= 3:
                integral_2 = torch.sum((pos - l_s) / phi(pos)) * width
                d2 = (d1 - d1_prev) / ((l_s - lam[i - 2]) / 2.0)
                x = x + alpha[i + 1] * ((l_t - l_s) ** 2 / 2.0 + integral_2 * phi(l_t)) * d2
            d1_prev = d1
        if inside and s_noise > 0:
            spread = (l_t ** 2 - l_s ** 2 * ratio ** 2).clamp(min=0).sqrt()
            x = x + alpha[i + 1] * noise(sig[i], sig[i + 1]) * s_noise * spread
            draws += 1
        d_prev = denoised
    return x, draws


if __name__ == "__main__":
    unittest.main()
