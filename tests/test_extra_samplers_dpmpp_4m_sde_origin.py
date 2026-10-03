"""Origin-parity tests for ``DPM++ 4M SDE`` (``sam3ext.extra_samplers.dpmpp_4m_sde``).

The oracle is Clybius' ``sample_clyb_4m_sde_momentumized`` at momentum 0 — the KSampler entry
``clyb_4m_sde_momentumized`` — loaded from the verbatim upstream file
``tests/_origin_clybius_extra_samplers.py`` (BSD-3-Clause, SHA-256 pinned below; Forge's
k-diffusion supplies the helpers it imports). Forge's side is Forge Neo's own
``sample_dpmpp_3m_sde`` from ``modules_forge/packages/k_diffusion/sampling.py``.

* Epsilon models: the port is Clybius' sampler (to float64 rounding — Forge's form multiplies by
  ``α_t = σ·e^{−log σ}`` and ``σ_t/σ_s·e^{−hη}`` where upstream writes 1 and ``e^{−h(1+η)}``).
* Rectified-flow models: the port equals Clybius' sampler run in epsilon-equivalent coordinates
  (``x/α``, ``σ/α`` with α = 1 − σ, the model called back in flow form) — which is what "Forge's
  3M SDE flow handling" means for a multistep DPM-Solver++ in half-log-SNR.
* Steps of order ≤ 3 are Forge's ``sample_dpmpp_3m_sde`` exactly (bit for bit, flow and epsilon).
"""

from __future__ import annotations

import importlib.util
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

from sam3ext.extra_samplers.dpmpp_4m_sde import sample_dpmpp_4m_sde  # noqa: E402

ORIGIN_FILE = "_origin_clybius_extra_samplers.py"
ORIGIN_MARKER = "# ---- upstream extra_samplers.py below (verbatim) ----\n"
# SHA-256 of Clybius/ComfyUI-Extra-Samplers@52eac1b7c847d2727e0ca93ca26d9ffd77029daa:extra_samplers.py
# (git blob d56d65e1ee5a768c863d203498c5cb970fe6b94d; LF upstream).
ORIGIN_SHA256 = "24c5d9e3853acfb57f115ad000190afb3d5eca8d098640007376eca7406234c3"

TOL = dict(rtol=1e-10, atol=1e-10)


class OriginCopyTests(unittest.TestCase):
    def test_origin_copy_is_upstream_verbatim(self):
        body = fx.origin_body(ORIGIN_FILE, ORIGIN_MARKER)
        self.assertEqual(fx.sha256(body), ORIGIN_SHA256)

    def test_the_registered_entry_is_the_momentum_free_sampler_with_discarded_penultimate_sigma(self):
        origin = fx.load_clybius_origin(fx.forge_k_sampling())
        self.assertIs(origin.extra_samplers["clyb_4m_sde_momentumized"], origin.sample_clyb_4m_sde)
        self.assertIn("clyb_4m_sde_momentumized", origin.discard_penultimate_sigma_samplers)
        import inspect

        self.assertEqual(inspect.signature(origin.sample_clyb_4m_sde).parameters["momentum"].default, 0.0)


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()
        cls.origin = fx.load_clybius_origin(cls.ks)

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))

    def _clybius(self, model, x, sigmas, eta, s_noise, noise, callback=None):
        return self.origin.sample_clyb_4m_sde_momentumized(
            model, x.clone(), sigmas, extra_args={}, callback=callback, disable=True,
            eta=eta, s_noise=s_noise, noise_sampler=noise, momentum=0.0,
        )

    def _ours(self, model, x, sigmas, eta, s_noise, noise, callback=None, extra_args=None):
        return sample_dpmpp_4m_sde(
            model, x.clone(), sigmas, extra_args={} if extra_args is None else extra_args,
            callback=callback, disable=True, eta=eta, s_noise=s_noise, noise_sampler=noise,
        )


class EpsilonParityTests(_Base):
    def test_matches_clybius_on_epsilon_models(self):
        sigmas = fx.eps_sigmas(10)
        x = fx.seeded((2, 4, 6, 5), 1) * float(sigmas[0])
        for eta, s_noise in ((1.0, 1.0), (0.6, 0.9), (0.0, 1.0), (1.0, 0.0)):
            with self.subTest(eta=eta, s_noise=s_noise):
                origin_noise, noise = fx.NoiseSequence(x, 7), fx.NoiseSequence(x, 7)
                origin = self._clybius(fx.ToyModel(fx.EpsSampling()), x, sigmas, eta, s_noise, origin_noise)
                ours = self._ours(fx.ToyModel(fx.EpsSampling()), x, sigmas, eta, s_noise, noise)
                torch.testing.assert_close(ours, origin, **TOL)
                if eta > 0 and s_noise > 0:
                    self.assertEqual(noise.calls, origin_noise.calls)
                else:
                    self.assertEqual(noise.calls, 0)   # Forge's rule: no draw when it would be scaled by 0

    def test_every_step_matches_clybius(self):
        sigmas = fx.eps_sigmas(8)
        x = fx.seeded((1, 4, 5, 5), 2) * float(sigmas[0])
        origin_steps, steps = [], []
        self._clybius(fx.ToyModel(fx.EpsSampling()), x, sigmas, 1.0, 1.0, fx.NoiseSequence(x, 3),
                      callback=lambda d: origin_steps.append(d["x"]))
        self._ours(fx.ToyModel(fx.EpsSampling()), x, sigmas, 1.0, 1.0, fx.NoiseSequence(x, 3),
                   callback=lambda d: steps.append(d["x"]))
        self.assertEqual(len(steps), len(sigmas) - 1)
        for i, (ours, origin) in enumerate(zip(steps, origin_steps)):
            with self.subTest(step=i):
                torch.testing.assert_close(ours, origin, **TOL)


class FlowParityTests(_Base):
    def _flow_vs_eps_coordinates(self, eta, s_noise, steps=9, shift=3.0):
        sampling = fx.FlowSampling(shift)
        sigmas = fx.flow_sigmas(steps, shift)
        self.assertEqual(float(sigmas[0]), 1.0)
        x = fx.seeded((2, 4, 6, 5), 4)
        flow_model = fx.ToyModel(sampling)
        steps_seen = []
        ours = self._ours(flow_model, x, sigmas, eta, s_noise, fx.NoiseSequence(x, 11),
                          callback=lambda d: steps_seen.append((d["x"], d["sigma"])))

        # Clybius in epsilon-equivalent coordinates: s = σ/α, x̃ = x/α, model x0(α·x̃, σ(s)).
        offset = self.ks.offset_first_sigma_for_snr(sigmas, sampling)
        self.assertLess(float(offset[0]), 1.0)
        eps_sigmas = offset / (1 - offset)

        def eps_model(xt, s, **kwargs):
            sigma = s / (1 + s)
            alpha = (1 - sigma).reshape(-1, *([1] * (xt.ndim - 1)))
            return flow_model(xt * alpha, sigma, **kwargs)

        origin_steps = []
        origin = self._clybius(eps_model, x / (1 - offset[0]), eps_sigmas, eta, s_noise, fx.NoiseSequence(x, 11),
                               callback=lambda d: origin_steps.append(d["x"]))
        return ours, origin, steps_seen, origin_steps, offset

    def test_matches_clybius_in_epsilon_equivalent_coordinates(self):
        for eta, s_noise in ((1.0, 1.0), (0.5, 1.0), (0.0, 1.0)):
            with self.subTest(eta=eta, s_noise=s_noise):
                ours, origin, steps, origin_steps, offset = self._flow_vs_eps_coordinates(eta, s_noise)
                torch.testing.assert_close(ours, origin, rtol=1e-9, atol=1e-9)
                for i, ((x_flow, sigma), x_eps) in enumerate(zip(steps, origin_steps)):
                    torch.testing.assert_close(x_flow, x_eps * (1 - offset[i]), rtol=1e-9, atol=1e-9,
                                               msg=f"step {i}")

    def test_flow_first_sigma_is_offset_like_forge(self):
        sampling = fx.FlowSampling(3.0)
        sigmas = fx.flow_sigmas(5)
        model = fx.ToyModel(sampling)
        self._ours(model, fx.seeded((1, 4, 4, 4), 5), sigmas, 1.0, 1.0, fx.NoiseSequence(torch.zeros(1, 4, 4, 4)))
        self.assertAlmostEqual(model.calls[0].sigma, float(self.ks.offset_first_sigma_for_snr(sigmas, sampling)[0]))
        self.assertLess(model.calls[0].sigma, 1.0)


class ForgeThreeMConventionTests(_Base):
    """Up to third order the 4M sampler is Forge's DPM++ 3M SDE term for term."""

    def _run_both(self, flow, sigmas, x, eta=1.0, s_noise=1.0):
        sampling = fx.FlowSampling(3.0) if flow else fx.EpsSampling()
        ours = self._ours(fx.ToyModel(sampling), x, sigmas, eta, s_noise, fx.NoiseSequence(x, 21))
        forge = self.ks.sample_dpmpp_3m_sde(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True,
                                            eta=eta, s_noise=s_noise, noise_sampler=fx.NoiseSequence(x, 21))
        return ours, forge

    def test_first_three_steps_are_forges_3m_sde(self):
        for flow in (False, True):
            full = fx.flow_sigmas(10) if flow else fx.eps_sigmas(10)
            x = fx.seeded((2, 4, 5, 6), 6) * (1.0 if flow else float(full[0]))
            for count in (1, 2, 3):   # orders 1, 2, 3 (no zero at the end: every step takes the SDE branch)
                with self.subTest(flow=flow, steps=count):
                    ours, forge = self._run_both(flow, full[: count + 1], x)
                    self.assertTrue(torch.equal(ours, forge))

    def test_fourth_order_step_is_where_they_part(self):
        for flow in (False, True):
            full = fx.flow_sigmas(10) if flow else fx.eps_sigmas(10)
            x = fx.seeded((2, 4, 5, 6), 6) * (1.0 if flow else float(full[0]))
            with self.subTest(flow=flow):
                ours, forge = self._run_both(flow, full[:5], x)
                self.assertFalse(torch.equal(ours, forge))
                self.assertTrue(torch.isfinite(ours).all())

    def test_last_step_returns_the_denoised_latent(self):
        sigmas = torch.tensor([0.7, 0.0], dtype=torch.float64)
        model = fx.ToyModel(fx.FlowSampling(3.0))
        x = fx.seeded((1, 4, 4, 4), 8)
        out = self._ours(model, x, sigmas, 1.0, 1.0, fx.NoiseSequence(x))
        self.assertTrue(torch.equal(out, fx.toy_x0(x, torch.tensor([0.7], dtype=torch.float64), 0.5)))
        self.assertIs(sample_dpmpp_4m_sde(model, x, sigmas[:1]), x)   # nothing to sample


class NoiseTests(_Base):
    def test_no_noise_drawn_without_eta_or_s_noise(self):
        sigmas = fx.eps_sigmas(6)
        x = fx.seeded((1, 4, 4, 4), 9) * float(sigmas[0])
        for eta, s_noise in ((0.0, 1.0), (1.0, 0.0)):
            noise = fx.NoiseSequence(x)
            self._ours(fx.ToyModel(fx.EpsSampling()), x, sigmas, eta, s_noise, noise)
            self.assertEqual(noise.calls, 0)

    def test_default_noise_is_a_seeded_cpu_brownian_tree_like_forges_3m_sde(self):
        # float32 like Forge's sigmas (torchsde keeps its tree in float32 on the CPU)
        sigmas = fx.eps_sigmas(6, dtype=torch.float32)
        x = fx.seeded((2, 4, 4, 4), 10, dtype=torch.float32) * float(sigmas[0])
        first = self._ours(fx.ToyModel(fx.EpsSampling()), x, sigmas, 1.0, 1.0, None, extra_args={"seed": 1234})
        again = self._ours(fx.ToyModel(fx.EpsSampling()), x, sigmas, 1.0, 1.0, None, extra_args={"seed": 1234})
        tree = self.ks.BrownianTreeNoiseSampler(x, sigmas[sigmas > 0].min(), sigmas.max(), seed=1234, cpu=True)
        explicit = self._ours(fx.ToyModel(fx.EpsSampling()), x, sigmas, 1.0, 1.0, tree)
        forge = self.ks.sample_dpmpp_3m_sde(fx.ToyModel(fx.EpsSampling()), x.clone(), sigmas[:4],
                                            extra_args={"seed": 1234}, disable=True)
        ours_three = self._ours(fx.ToyModel(fx.EpsSampling()), x, sigmas[:4], 1.0, 1.0, None, extra_args={"seed": 1234})
        self.assertTrue(torch.equal(first, again))
        self.assertTrue(torch.equal(first, explicit))
        self.assertTrue(torch.equal(ours_three, forge))


if __name__ == "__main__":
    unittest.main()
