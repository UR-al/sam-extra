"""Origin-parity tests for ``IPNDM`` / ``IPNDM_V`` / ``DEIS`` (``sam3ext/extra_samplers/ipndm_deis.py``).

The oracle is ComfyUI's own code, verbatim (``tests/_origin_comfyui_ipndm_deis.py``: ``sample_ipndm``,
``sample_ipndm_v``, ``sample_deis`` — zju-pi/diff-sampler under Apache 2, adapted by ComfyUI — and the whole
``deis.py`` at comfyanonymous/ComfyUI@387f98aa; SHA-256 of every block pinned below). The extension side runs
on Forge Neo's real ``k_diffusion.sampling`` and Forge's vendored ``k_diffusion/deis.py`` (both executed from
Forge's files). What must match: the sampled latent, bit for bit, on ε and flow models, 4-D and 5-D latents,
every order the samplers run; Forge's DEIS coefficients and ComfyUI's, coefficient for coefficient. Beyond
parity: they solve the flow ODE (convergence on the exact denoiser of Gaussian data) and the callback sees
the current latent.
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

from sam3ext.extra_samplers import ipndm_deis  # noqa: E402

ORIGIN_FILE = "_origin_comfyui_ipndm_deis.py"
# SHA-256 of each verbatim block (upstream comfyanonymous/ComfyUI@387f98aa lines, LF).
ORIGIN_BLOCK_SHA256 = {
    "comfy/k_diffusion/sampling.py:1173-1330": "d9a32e74c577669705a8437174cbdbcc298bea0897447588694e022967e054a9",
    "comfy/k_diffusion/deis.py:1-120": "8d4e5057a062b77ef2e33af8208a2d6a3c3fbb431315a4c37689a307d66bc807",
}
STEPS = (1, 2, 3, 4, 5, 8, 28)

ORIGIN = fx.load_comfy_ipndm_deis_origin()
_PAIRS = {
    "ipndm": (ipndm_deis.sample_ipndm, ORIGIN.sample_ipndm),
    "ipndm_v": (ipndm_deis.sample_ipndm_v, ORIGIN.sample_ipndm_v),
    "deis": (ipndm_deis.sample_deis, ORIGIN.sample_deis),
}


def _problem(flow: bool, steps: int, dtype, five_d: bool = False, seed: int = 0):
    sampling = fx.FlowSampling(shift=3.0) if flow else fx.EpsSampling()
    sigmas = fx.flow_sigmas(steps, dtype=dtype) if flow else fx.eps_sigmas(steps, dtype=dtype)
    shape = (1, 4, 2, 6, 5) if five_d else (2, 4, 6, 5)
    x = fx.seeded(shape, seed, dtype=dtype) * (1.0 if flow else float(sigmas[0]))
    return sampling, sigmas, x


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()
        cls.deis = fx.forge_deis()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))
        self.enterContext(fx.installed_k_diffusion_deis(self.deis))


class OriginCopyTests(_Base):
    def test_every_block_is_upstream_verbatim(self):
        blocks = fx.origin_blocks(ORIGIN_FILE)
        self.assertEqual(set(blocks), set(ORIGIN_BLOCK_SHA256))
        for key, digest in ORIGIN_BLOCK_SHA256.items():
            with self.subTest(block=key):
                self.assertEqual(fx.sha256(blocks[key]), digest)
        sampling_block = blocks["comfy/k_diffusion/sampling.py:1173-1330"]
        self.assertEqual(sampling_block.count("#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py\n"
                                              "#under Apache 2 license"), 3)

    def test_the_defaults_are_comfyuis(self):
        for name, (ours, origin) in _PAIRS.items():
            with self.subTest(sampler=name):
                ours_params = inspect.signature(ours).parameters
                origin_params = inspect.signature(origin).parameters
                self.assertEqual(ours_params["max_order"].default, origin_params["max_order"].default)
        self.assertEqual((ipndm_deis.IPNDM_MAX_ORDER, ipndm_deis.DEIS_MAX_ORDER, ipndm_deis.DEIS_MODE), (4, 3, "tab"))
        self.assertEqual(inspect.signature(ipndm_deis.sample_deis).parameters["deis_mode"].default, "tab")

    def test_forges_deis_coefficients_are_comfyuis(self):
        """Forge's vendored deis.py differs from ComfyUI's only in formatting and an unused lambda."""
        for sigmas in (fx.flow_sigmas(12), fx.flow_sigmas(12)[1:], fx.eps_sigmas(12), fx.flow_sigmas(9, dtype=torch.float32)):
            for max_order in (1, 2, 3, 4):
                with self.subTest(first=float(sigmas[0]), dtype=sigmas.dtype, max_order=max_order):
                    forge = self.deis.get_deis_coeff_list(sigmas, max_order, deis_mode="tab")
                    comfy = ORIGIN.get_deis_coeff_list(sigmas, max_order, deis_mode="tab")
                    self.assertEqual([len(c) for c in forge], [len(c) for c in comfy])
                    for a_list, b_list in zip(forge, comfy):
                        for a, b in zip(a_list, b_list):
                            self.assertTrue(torch.equal(a, b) or (torch.isnan(a) and torch.isnan(b)))


class OriginParityTests(_Base):
    def test_every_sampler_is_comfyuis_bit_for_bit(self):
        for name, (ours, origin) in _PAIRS.items():
            for flow in (False, True):
                for dtype in (torch.float32, torch.float64):
                    for five_d in (False, True):
                        for steps in STEPS:
                            for max_order in (2, 3, 4):
                                with self.subTest(sampler=name, flow=flow, dtype=dtype, five_d=five_d, steps=steps,
                                                  max_order=max_order):
                                    sampling, sigmas, x = _problem(flow, steps, dtype, five_d)
                                    expected = origin(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={},
                                                      disable=True, max_order=max_order)
                                    model = fx.ToyModel(sampling)
                                    got = ours(model, x.clone(), sigmas, extra_args={}, disable=True, max_order=max_order)
                                    self.assertTrue(torch.isfinite(got).all())
                                    self.assertTrue(torch.equal(got, expected))
                                    self.assertEqual(len(model.calls), steps)   # one model call per step

    def test_the_registered_defaults_are_comfyuis_defaults(self):
        for name, (ours, origin) in _PAIRS.items():
            for flow in (False, True):
                with self.subTest(sampler=name, flow=flow):
                    sampling, sigmas, x = _problem(flow, 28, torch.float32)
                    expected = origin(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True)
                    got = ours(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True)
                    self.assertTrue(torch.equal(got, expected))

    def test_order_one_is_euler_where_upstream_raises(self):
        """Upstream's history update indexes an empty list at ``max_order = 1``; here order 1 keeps no history
        and is Forge's Euler step (on a grid that does not end at 0, where upstream would denoise)."""
        sampling, sigmas, x = _problem(True, 8, torch.float64)
        for name, (ours, origin) in _PAIRS.items():
            with self.subTest(sampler=name):
                with self.assertRaises(IndexError):
                    origin(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True, max_order=1)
                euler = self.ks.sample_euler(fx.ToyModel(sampling), x.clone(), sigmas[:-1], extra_args={}, disable=True)
                got = ours(fx.ToyModel(sampling), x.clone(), sigmas[:-1], extra_args={}, disable=True, max_order=1)
                self.assertTrue(torch.equal(got, euler))
                # orders above 4 are clamped to 4 (upstream would skip the update)
                four = ours(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True, max_order=4)
                self.assertTrue(torch.equal(ours(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True,
                                                 max_order=9), four))

    def test_the_callback_sees_the_current_latent(self):
        for name, (ours, _origin) in _PAIRS.items():
            with self.subTest(sampler=name):
                sampling, sigmas, x = _problem(True, 6, torch.float64)
                inputs, seen = [], []

                class Recorder(fx.ToyModel):
                    def __call__(self, latent, sigma, **kwargs):
                        inputs.append(latent.clone())
                        return super().__call__(latent, sigma, **kwargs)

                ours(Recorder(sampling), x.clone(), sigmas, extra_args={}, disable=True, callback=seen.append)
                self.assertEqual([info["i"] for info in seen], list(range(6)))
                for info, latent in zip(seen, inputs):
                    self.assertTrue(torch.equal(info["x"], latent))
                self.assertFalse(torch.equal(seen[-1]["x"], x))


class FlowOdeTests(_Base):
    """On the exact denoiser of Gaussian data (closed-form ODE solution) on Anima's shift-3 grid."""

    def _error(self, fn, steps):
        sampling = fx.FlowSampling(shift=3.0)
        sigmas = fx.flow_sigmas(steps)
        model = fx.GaussianModel(sampling)
        x = fx.seeded((64, 1, 1, 1), 3)
        exact = model.ode_transport(x, sigmas[0], torch.tensor(0.0, dtype=torch.float64))
        return float((fn(model, x.clone(), sigmas, extra_args={}, disable=True) - exact).abs().max())

    def test_they_solve_the_flow_ode(self):
        errors = {name: [self._error(ours, n) for n in (16, 32, 64)] for name, (ours, _o) in _PAIRS.items()}
        euler = [self._error(self.ks.sample_euler, n) for n in (16, 32, 64)]
        for name, errs in errors.items():
            with self.subTest(sampler=name, errors=errs):
                self.assertLess(errs[1], errs[0] / 3)
                self.assertLess(errs[2], errs[1] / 3)
                for ours, plain in zip(errs, euler):
                    self.assertLess(ours, plain)
        # the variable-step coefficients suit the uneven (shifted) grid better than the fixed ones
        for v, fixed in zip(errors["ipndm_v"], errors["ipndm"]):
            self.assertLessEqual(v, fixed)


if __name__ == "__main__":
    unittest.main()
