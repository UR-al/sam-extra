"""Origin-parity tests for ``ER SDE (Tunable)`` — ComfyUI's ``ER-SDE`` solver type with η (40e46c71).

The oracle is ComfyUI's ``SamplerER_SDE`` node at the commit that introduced the η-extended scaler
``h(λ) = λ·(e^(λ^0.3) + 10)^η`` (``tests/_origin_comfyui_er_sde_eta.py``, GPL-3.0 like this extension,
SHA-256 pinned below): ``execute("ER-SDE", max_stage, eta, s_noise)`` builds the options of
``comfy.samplers.ksampler("er_sde", …)``. The extension side is ``sam3ext.extra_samplers.er_sde`` running
Forge Neo's own ``sample_er_sde`` (executed from Forge's file). What must match: the options (the ODE rule,
``s_noise``, ``max_stage``, the scaler's values) and the sampled latent, bit for bit, on ε and flow models;
at η = 1 the entry is Forge's built-in ``ER SDE`` (``noise_scaler=None``), at η = 0 ``ER SDE (ODE)``.
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

from sam3ext.extra_samplers import er_sde, params  # noqa: E402

ORIGIN_FILE = "_origin_comfyui_er_sde_eta.py"
# SHA-256 of the verbatim block (upstream comfyanonymous/ComfyUI@40e46c71 lines, LF). The same lines are
# pinned at 36c0b0a6 by test_extra_samplers_er_sde_origin.py (byte-identical, so the same digest).
ORIGIN_BLOCK_SHA256 = {
    "comfy_extras/nodes_custom_sampler.py:585-633": "a3c8e152f7de8543931743abbf267b6c79fedd42d9b5c02c65c186a9ed89bc08",
}
ETAS = (0.0, 0.37, 1.0, 2.5, 10.0)

ORIGIN = fx.load_comfy_er_sde_eta_origin()


def _node_options(solver_type, max_stage, eta, s_noise):
    output = ORIGIN.SamplerER_SDE.execute(solver_type, max_stage, eta, s_noise)
    (sampler,) = output.args
    assert sampler.sampler_name == "er_sde"
    return sampler.extra_options


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality (NaN-safe: a scaler that overflows in float32 gives the same NaN on both sides)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {torch.float32: torch.int32, torch.float64: torch.int64}[a.dtype]
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _problem(flow: bool, dtype, steps: int = 7, seed: int = 0, shape=(2, 4, 6, 5)):
    sampling = fx.FlowSampling(shift=3.0) if flow else fx.EpsSampling()
    sigmas = fx.flow_sigmas(steps, dtype=dtype) if flow else fx.eps_sigmas(steps, dtype=dtype)
    x = fx.seeded(shape, seed, dtype=dtype) * (1.0 if flow else float(sigmas[0]))
    return sampling, sigmas, x


class OriginCopyTests(unittest.TestCase):
    def test_the_block_is_upstream_verbatim(self):
        blocks = fx.origin_blocks(ORIGIN_FILE)
        self.assertEqual(set(blocks), set(ORIGIN_BLOCK_SHA256))
        for key, digest in ORIGIN_BLOCK_SHA256.items():
            with self.subTest(block=key):
                self.assertEqual(fx.sha256(blocks[key]), digest)

    def test_the_er_sde_solver_type_is_ours(self):
        schema = ORIGIN.SamplerER_SDE.define_schema()
        inputs = {item.name: item for item in schema.inputs}
        self.assertEqual(inputs["solver_type"].options, [er_sde.ER_SDE, er_sde.REVERSE_TIME, er_sde.ODE])
        eta = inputs["eta"]
        self.assertEqual((eta.default, eta.min, eta.max, eta.step),
                         (params.DEFAULT_ETA, params.ETA_MIN, params.ETA_MAX, params.ETA_STEP))

    def test_options_are_the_nodes(self):
        probe = torch.tensor([0.0, 1e-6, 0.05, 0.7, 1.0, 3.0, 40.0, 3.0e4], dtype=torch.float64)
        for solver_type in (er_sde.ER_SDE, er_sde.REVERSE_TIME, er_sde.ODE):
            for eta in ETAS:
                for s_noise in (0.0, 1.0, 0.97):
                    with self.subTest(solver=solver_type, eta=eta, s_noise=s_noise):
                        origin = _node_options(solver_type, 2, eta, s_noise)
                        ours = er_sde.er_sde_kwargs(solver_type, 2, eta, s_noise)
                        self.assertEqual(set(ours), set(origin))
                        self.assertEqual(ours["s_noise"], origin["s_noise"])
                        self.assertEqual(ours["max_stage"], origin["max_stage"])
                        for dtype in (torch.float64, torch.float32):
                            values = probe.to(dtype)
                            self.assertTrue(_bits_equal(ours["noise_scaler"](values), origin["noise_scaler"](values)))

    def test_the_tunable_scaler_at_eta_one_is_forges_default_scaler_bit_for_bit(self):
        """``t ** 1.0`` is exact, so ``λ·(e^(λ^0.3)+10)^1`` equals Forge's ``λ·(e^(λ^0.3)+10)`` on every float."""
        scaler = er_sde.er_sde_noise_scaler(1.0)
        for dtype in (torch.float32, torch.float64):
            values = torch.cat([torch.tensor([0.0, 1e-30, 1e-8]), torch.logspace(-6, 6, 4001)]).to(dtype)
            with self.subTest(dtype=dtype):
                self.assertTrue(_bits_equal(scaler(values), values * ((values ** 0.3).exp() + 10.0)))


class ForgeSolverParityTests(unittest.TestCase):
    """``sample_er_sde_tunable`` on Forge's ``sample_er_sde`` vs Forge's ``sample_er_sde`` with the node's options."""

    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))

    def _run(self, fn, flow, dtype, seed=0, **kwargs):
        sampling, sigmas, x = _problem(flow, dtype, seed=seed)
        model = fx.ToyModel(sampling)
        noise = fx.NoiseSequence(x, seed=100 + seed)
        out = fn(model, x.clone(), sigmas, extra_args={}, disable=True, noise_sampler=noise, **kwargs)
        return out, noise, model

    def test_tunable_matches_the_nodes_er_sde_options(self):
        for flow in (False, True):
            for dtype in (torch.float32, torch.float64):
                for stage in (1, 2, 3):
                    for eta in ETAS:
                        for s_noise in (0.0, 1.0):
                            with self.subTest(flow=flow, dtype=dtype, stage=stage, eta=eta, s_noise=s_noise):
                                options = _node_options(er_sde.ER_SDE, stage, eta, s_noise)
                                origin, origin_noise, origin_model = self._run(self.ks.sample_er_sde, flow, dtype,
                                                                               **options)
                                ours, noise, model = self._run(er_sde.sample_er_sde_tunable, flow, dtype,
                                                               s_noise=s_noise, max_stage=stage, er_sde_eta=eta)
                                self.assertTrue(_bits_equal(ours, origin))
                                self.assertEqual(noise.calls, origin_noise.calls)
                                self.assertEqual([c.sigma for c in model.calls], [c.sigma for c in origin_model.calls])
                                if eta > 0 and s_noise > 0:
                                    self.assertGreater(noise.calls, 0)
                                if eta in (0.37, 1.0) and dtype == torch.float64:
                                    self.assertTrue(torch.isfinite(ours).all())

    def test_eta_one_is_forges_builtin_er_sde(self):
        """Forge's "ER SDE" entry is ``sample_er_sde`` with its default scaler (``noise_scaler=None``)."""
        for flow in (False, True):
            for dtype in (torch.float32, torch.float64):
                for stage in (1, 2, 3):
                    with self.subTest(flow=flow, dtype=dtype, stage=stage):
                        builtin, builtin_noise, _ = self._run(self.ks.sample_er_sde, flow, dtype, max_stage=stage)
                        ours, noise, _ = self._run(er_sde.sample_er_sde_tunable, flow, dtype, max_stage=stage)
                        self.assertTrue(torch.isfinite(ours).all())
                        self.assertTrue(torch.equal(ours, builtin))
                        self.assertEqual(noise.calls, builtin_noise.calls)

    def test_eta_zero_is_the_ode_and_draws_nothing(self):
        for flow in (False, True):
            for stage in (1, 2, 3):
                with self.subTest(flow=flow, stage=stage):
                    ode, _, _ = self._run(er_sde.sample_er_sde_ode, flow, torch.float64, max_stage=stage)
                    ours, noise, _ = self._run(er_sde.sample_er_sde_tunable, flow, torch.float64,
                                               max_stage=stage, er_sde_eta=0.0)
                    self.assertTrue(torch.equal(ours, ode))
                    self.assertEqual(noise.calls, 0)
                    # … also with a window: η 0 ignores it (Forge's function is called plainly)
                    windowed, noise, _ = self._run(er_sde.sample_er_sde_tunable, flow, torch.float64,
                                                   max_stage=stage, er_sde_eta=0.0, er_sde_window=(0.2, 0.8))
                    self.assertTrue(torch.equal(windowed, ode))
                    self.assertEqual(noise.calls, 0)

    def test_eta_changes_the_result_and_values_are_clamped_like_the_node_inputs(self):
        base, _, _ = self._run(er_sde.sample_er_sde_tunable, True, torch.float64)
        self.assertFalse(torch.equal(base, self._run(er_sde.sample_er_sde_tunable, True, torch.float64,
                                                     er_sde_eta=0.5)[0]))
        clamped, _, _ = self._run(er_sde.sample_er_sde_tunable, False, torch.float64, max_stage=9, er_sde_eta=25.0)
        node_max, _, _ = self._run(self.ks.sample_er_sde, False, torch.float64,
                                   **_node_options(er_sde.ER_SDE, 3, 10.0, 1.0))
        self.assertTrue(_bits_equal(clamped, node_max))

    def test_the_tunable_entry_never_takes_forges_global_eta(self):
        import inspect

        parameters = inspect.signature(er_sde.sample_er_sde_tunable).parameters
        self.assertNotIn("eta", parameters)
        self.assertIn("er_sde_eta", parameters)
        self.assertIn("er_sde_window", parameters)
        self.assertEqual(parameters["er_sde_eta"].default, params.DEFAULT_ETA)
        self.assertIsNone(parameters["er_sde_window"].default)


if __name__ == "__main__":
    unittest.main()
