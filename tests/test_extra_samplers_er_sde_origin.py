"""Origin-parity tests for ``ER SDE (Reverse-time)`` and ``ER SDE (ODE)``.

The oracle is ComfyUI's own code, verbatim (``tests/_origin_comfyui_er_sde.py``, GPL-3.0 like this
extension, SHA-256 of every block pinned below): ``SamplerER_SDE.execute`` turns the node inputs
into the options of ``comfy.samplers.ksampler("er_sde", …)`` and ComfyUI's ``sample_er_sde`` runs
them. The extension side is ``sam3ext.extra_samplers.er_sde`` running Forge Neo's own
``sample_er_sde`` (``modules_forge/packages/k_diffusion/sampling.py``, executed from Forge's file).

What must match: the node's input ranges (the accordion sliders), the options it builds (the ODE
rule: ``solver_type == "ODE" or eta == 0`` → ``h(λ) = λ`` and ``s_noise = 0``), and the sampled
latent for every stage on epsilon and rectified-flow models (bit for bit: the two solvers are the
same code). Forge's built-in ``ER SDE`` is the node's ``ER-SDE`` at η = 1.
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

ORIGIN_FILE = "_origin_comfyui_er_sde.py"
# SHA-256 of each verbatim block (upstream comfyanonymous/ComfyUI@36c0b0a6 lines, LF).
ORIGIN_BLOCK_SHA256 = {
    "comfy/k_diffusion/sampling.py:78-88": "ef9b871f2a8a9acc43364d9e6cacf5ae2813dd5aa2d22863d6a7dbbf67d45e68",
    "comfy/k_diffusion/sampling.py:152-176": "5d1f8007b83393ce51a7ec40eb9ebfef838901200b6c8a63dcbbbe30d56500e8",
    "comfy/k_diffusion/sampling.py:1592-1656": "0f8db8e10b7d1b5311adaffe1ebfbdb41ced68c88fa31041eaa9e3a5c4d83543",
    "comfy_extras/nodes_custom_sampler.py:585-633": "a3c8e152f7de8543931743abbf267b6c79fedd42d9b5c02c65c186a9ed89bc08",
}

ORIGIN = fx.load_comfy_er_sde_origin()


class _ComfyFlowSampling(fx.FlowSampling, ORIGIN.CONST):
    """Flow model sampling for both hosts: ``prediction_type == "const"`` (Forge) and a CONST (ComfyUI)."""


def _node_options(solver_type, max_stage, eta, s_noise):
    output = ORIGIN.SamplerER_SDE.execute(solver_type, max_stage, eta, s_noise)
    (sampler,) = output.args
    assert sampler.sampler_name == "er_sde"
    return sampler.extra_options


def _problem(flow: bool, steps: int = 7, seed: int = 0):
    sampling = _ComfyFlowSampling(shift=3.0) if flow else fx.EpsSampling()
    sigmas = fx.flow_sigmas(steps) if flow else fx.eps_sigmas(steps)
    x = fx.seeded((2, 4, 6, 5), seed) * (1.0 if flow else float(sigmas[0]))
    return sampling, sigmas, x


class OriginCopyTests(unittest.TestCase):
    def test_every_block_is_upstream_verbatim(self):
        blocks = fx.origin_blocks(ORIGIN_FILE)
        self.assertEqual(set(blocks), set(ORIGIN_BLOCK_SHA256))
        for key, digest in ORIGIN_BLOCK_SHA256.items():
            with self.subTest(block=key):
                self.assertEqual(fx.sha256(blocks[key]), digest)

    def test_node_inputs_are_the_accordion_ranges(self):
        schema = ORIGIN.SamplerER_SDE.define_schema()
        inputs = {item.name: item for item in schema.inputs}
        self.assertEqual(inputs["solver_type"].options, ["ER-SDE", "Reverse-time SDE", "ODE"])
        self.assertIn(er_sde.REVERSE_TIME, inputs["solver_type"].options)
        self.assertIn(er_sde.ODE, inputs["solver_type"].options)
        stage = inputs["max_stage"]
        self.assertEqual((stage.default, stage.min, stage.max),
                         (params.DEFAULT_MAX_STAGE, params.MAX_STAGE_MIN, params.MAX_STAGE_MAX))
        eta = inputs["eta"]
        self.assertEqual((eta.default, eta.min, eta.max, eta.step),
                         (params.DEFAULT_ETA, params.ETA_MIN, params.ETA_MAX, params.ETA_STEP))

    def test_options_are_the_nodes(self):
        for solver_type, eta, s_noise in [
            (er_sde.REVERSE_TIME, 1.0, 1.0), (er_sde.REVERSE_TIME, 2.5, 0.97), (er_sde.REVERSE_TIME, 0.0, 1.0),
            (er_sde.ODE, 1.0, 1.0), (er_sde.ODE, 0.0, 0.5),
        ]:
            with self.subTest(solver=solver_type, eta=eta, s_noise=s_noise):
                origin = _node_options(solver_type, 2, eta, s_noise)
                ours = er_sde.er_sde_kwargs(solver_type, 2, eta, s_noise)
                self.assertEqual(set(ours), set(origin))
                self.assertEqual(ours["s_noise"], origin["s_noise"])
                self.assertEqual(ours["max_stage"], origin["max_stage"])
                probe = torch.tensor([0.05, 0.7, 3.0, 40.0], dtype=torch.float64)
                self.assertTrue(torch.equal(ours["noise_scaler"](probe), origin["noise_scaler"](probe)))

    def test_eta_zero_is_the_ode(self):
        kwargs = er_sde.er_sde_kwargs(er_sde.REVERSE_TIME, 3, 0.0, 1.0)
        self.assertEqual(kwargs["s_noise"], 0.0)
        self.assertIs(kwargs["noise_scaler"], er_sde.ode_noise_scaler)

    def test_unknown_solver_type_is_refused(self):
        with self.assertRaises(ValueError):
            er_sde.er_sde_kwargs("ER-SDE", 3, 1.0, 1.0)


class ForgeSolverParityTests(unittest.TestCase):
    """Our wrappers on Forge's ``sample_er_sde`` vs ComfyUI's node + ``sample_er_sde``."""

    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))

    def _origin(self, solver_type, max_stage, eta, s_noise, flow, seed=0):
        sampling, sigmas, x = _problem(flow, seed=seed)
        model = fx.ToyModel(sampling)
        noise = fx.NoiseSequence(x, seed=100 + seed)
        options = _node_options(solver_type, max_stage, eta, s_noise)
        out = ORIGIN.sample_er_sde(model, x.clone(), sigmas, extra_args={}, disable=True,
                                   noise_sampler=noise, **options)
        return out, noise, model

    def _ours(self, kind, max_stage, eta, s_noise, flow, seed=0):
        sampling, sigmas, x = _problem(flow, seed=seed)
        model = fx.ToyModel(sampling)
        noise = fx.NoiseSequence(x, seed=100 + seed)
        if kind == "reverse":
            out = er_sde.sample_er_sde_reverse_time(model, x.clone(), sigmas, extra_args={}, disable=True,
                                                    s_noise=s_noise, noise_sampler=noise,
                                                    max_stage=max_stage, er_sde_eta=eta)
        else:
            out = er_sde.sample_er_sde_ode(model, x.clone(), sigmas, extra_args={}, disable=True,
                                           noise_sampler=noise, max_stage=max_stage)
        return out, noise, model

    def test_reverse_time_matches_comfy(self):
        for flow in (False, True):
            for stage in (1, 2, 3):
                for eta, s_noise in ((1.0, 1.0), (2.5, 0.97), (0.35, 1.05)):
                    with self.subTest(flow=flow, stage=stage, eta=eta, s_noise=s_noise):
                        origin, origin_noise, origin_model = self._origin(er_sde.REVERSE_TIME, stage, eta, s_noise, flow)
                        ours, noise, model = self._ours("reverse", stage, eta, s_noise, flow)
                        self.assertTrue(torch.isfinite(ours).all())
                        self.assertTrue(torch.equal(ours, origin))
                        self.assertEqual(noise.calls, origin_noise.calls)
                        self.assertGreater(noise.calls, 0)
                        self.assertEqual([c.sigma for c in model.calls], [c.sigma for c in origin_model.calls])

    def test_ode_matches_comfy_and_never_draws_noise(self):
        for flow in (False, True):
            for stage in (1, 2, 3):
                with self.subTest(flow=flow, stage=stage):
                    # The node ignores eta and s_noise for the ODE.
                    origin, origin_noise, _ = self._origin(er_sde.ODE, stage, 4.0, 1.0, flow)
                    ours, noise, _ = self._ours("ode", stage, None, None, flow)
                    self.assertTrue(torch.equal(ours, origin))
                    self.assertEqual(noise.calls, 0)
                    self.assertEqual(origin_noise.calls, 0)

    def test_reverse_time_at_eta_zero_is_the_ode(self):
        for flow in (False, True):
            with self.subTest(flow=flow):
                ode, _, _ = self._ours("ode", 3, None, None, flow)
                reverse, noise, _ = self._ours("reverse", 3, 0.0, 1.0, flow)
                self.assertTrue(torch.equal(reverse, ode))
                self.assertEqual(noise.calls, 0)

    def test_stage_and_eta_change_the_result(self):
        base, _, _ = self._ours("reverse", 3, 1.0, 1.0, flow=True)
        self.assertFalse(torch.equal(base, self._ours("reverse", 1, 1.0, 1.0, flow=True)[0]))
        self.assertFalse(torch.equal(base, self._ours("reverse", 3, 2.0, 1.0, flow=True)[0]))

    def test_out_of_range_values_are_clamped_like_the_node_inputs(self):
        clamped, _, _ = self._ours("reverse", 9, 25.0, 1.0, flow=False)
        node_max, _, _ = self._origin(er_sde.REVERSE_TIME, 3, 10.0, 1.0, flow=False)
        self.assertTrue(torch.equal(clamped, node_max))
        low, _, _ = self._ours("ode", 0, None, None, flow=False)
        node_low, _, _ = self._origin(er_sde.ODE, 1, 1.0, 1.0, flow=False)
        self.assertTrue(torch.equal(low, node_low))

    def test_forge_builtin_er_sde_is_the_nodes_er_sde_at_eta_one(self):
        for flow in (False, True):
            with self.subTest(flow=flow):
                origin, _, _ = self._origin("ER-SDE", 3, 1.0, 1.0, flow)
                sampling, sigmas, x = _problem(flow)
                builtin = self.ks.sample_er_sde(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={},
                                                disable=True, noise_sampler=fx.NoiseSequence(x, seed=100))
                self.assertTrue(torch.equal(builtin, origin))

    def test_callbacks_and_final_step_follow_forge(self):
        sampling, sigmas, x = _problem(flow=True)
        seen = []
        out = er_sde.sample_er_sde_reverse_time(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={},
                                                callback=lambda d: seen.append(d["i"]), disable=True,
                                                noise_sampler=fx.NoiseSequence(x))
        self.assertEqual(seen, list(range(len(sigmas) - 1)))
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
