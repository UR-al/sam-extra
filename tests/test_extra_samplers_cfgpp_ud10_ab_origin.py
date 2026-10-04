"""Origin-parity tests for ``CFG++ UD10 AB`` (``sam3ext/extra_samplers/cfgpp_ud10_ab.py``).

The oracle is ComfyUI's own code, verbatim (``tests/_origin_comfyui_cfgpp_history.py``: ``_sample_cfgpp_history``,
``sample_cfgpp_ud10_ab`` and the helpers they call at comfyanonymous/ComfyUI@3ac5d794, GPL-3.0 like this
extension, SHA-256 of every block pinned below). The extension side runs on Forge Neo's real
``k_diffusion.sampling`` (executed from Forge's file).

Each host is emulated by a CFG stand-in that hands the post-CFG functions the arguments that host passes
(Forge ``backend/sampling/sampling_function.py`` ``sampling_function_inner``; ComfyUI
``comfy/samplers.py`` ``cfg_function``): when the CFG-1 optimisation skips the unconditional pass both
leave ``uncond_denoised`` at zeros, ComfyUI passes ``uncond=None`` and Forge the request's uncond
conditioning. The port must equal the origin bit for bit on both, at every CFG scale, including steps
that Forge runs at CFG 1 (Skip Early CFG / NGMS).
"""

from __future__ import annotations

import importlib.util
import inspect
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

from sam3ext.extra_samplers import cfgpp_ud10_ab  # noqa: E402

ORIGIN_FILE = "_origin_comfyui_cfgpp_history.py"
# SHA-256 of each verbatim block (upstream comfyanonymous/ComfyUI@3ac5d794 lines, LF).
ORIGIN_BLOCK_SHA256 = {
    "comfy/k_diffusion/utils.py:21-29": "816349d93f63ced7a88c4fc3967a6c1ba77bffca03536dea4b4e12604b98f5b5",
    "comfy/model_patcher.py:114-118": "bf2232a99b98ab85a8c9ba9d49ec86faa931a7e6a3c2c3cfd90cd9de98f06125",
    "comfy/k_diffusion/sampling.py:63-65": "79b063e3cb9561d840b552ddd71406e9204036cede1eafd025ea0d7b62426a01",
    "comfy/k_diffusion/sampling.py:152-176": "5d1f8007b83393ce51a7ec40eb9ebfef838901200b6c8a63dcbbbe30d56500e8",
    "comfy/k_diffusion/sampling.py:406-416": "0a46fc4d44ad669cd38df55ea73dfc2cd06de279392d702ba121fce96836ea73",
    "comfy/k_diffusion/sampling.py:419-484": "fabe5cd7d53c36151dd536434531f6bcc64ff9b052bc1839db719e0ea59abd4d",
}
_SAMPLING_FUNCTION = "backend/sampling/sampling_function.py"

ORIGIN = fx.load_comfy_cfgpp_history_origin()


class _ComfyFlowSampling(fx.FlowSampling, ORIGIN.CONST):
    """Flow model sampling for both hosts: ``prediction_type == "const"`` (Forge) and a CONST (ComfyUI)."""


class _CfgHost(fx.ToyModel):
    """A CFG denoiser stand-in with each host's post-CFG arguments.

    ``cond_scales``: one CFG scale per call (the last one repeats) — Forge runs a Skip Early CFG / NGMS
    step at 1.0. When the scale is 1 and ``disable_cfg1_optimization`` is not set, the unconditional pass is
    skipped: ``uncond_denoised`` stays zeros (both hosts) and ``uncond`` is None (ComfyUI) or the request's
    uncond conditioning (Forge)."""

    def __init__(self, sampling, host: str, cond_scales):
        super().__init__(sampling)
        self.host = host
        self.cond_scales = list(cond_scales)
        self.count = 0
        self.skipped = []

    def __call__(self, x, sigma, **extra_args):
        options = extra_args.get("model_options") or {}
        scale = self.cond_scales[min(self.count, len(self.cond_scales) - 1)]
        self.count += 1
        skipped = math.isclose(scale, 1.0) and options.get("disable_cfg1_optimization", False) == False  # noqa: E712
        self.skipped.append(skipped)
        cond = fx.toy_x0(x, sigma, 1.0)
        uncond = torch.zeros_like(cond) if skipped else fx.toy_x0(x, sigma, -1.0)
        denoised = uncond + (cond - uncond) * scale
        uncond_conditioning = ["uncond"]
        if self.host == "comfy" and skipped:
            uncond_conditioning = None
        for fn in options.get("sampler_post_cfg_function", []):
            denoised = fn({
                "denoised": denoised, "cond": ["cond"], "uncond": uncond_conditioning, "cond_scale": scale,
                "model": None, "uncond_denoised": uncond, "cond_denoised": cond, "sigma": sigma,
                "model_options": options, "input": x,
            })
        self.calls.append(fx.Call(shape=tuple(x.shape), sigma=float(sigma.reshape(-1)[0]), marker=None,
                                  init_latent=None, mask=None, image_cond=None, model_options=options))
        return denoised


def _problem(flow: bool, steps: int, dtype, seed: int = 0, shape=(2, 4, 6, 5)):
    sampling = _ComfyFlowSampling(shift=3.0) if flow else fx.EpsSampling()
    sigmas = fx.flow_sigmas(steps, dtype=dtype) if flow else fx.eps_sigmas(steps, dtype=dtype)
    x = fx.seeded(shape, seed, dtype=dtype) * (1.0 if flow else float(sigmas[0]))
    return sampling, sigmas, x


class OriginCopyTests(unittest.TestCase):
    def test_every_block_is_upstream_verbatim(self):
        blocks = fx.origin_blocks(ORIGIN_FILE)
        self.assertEqual(set(blocks), set(ORIGIN_BLOCK_SHA256))
        for key, digest in ORIGIN_BLOCK_SHA256.items():
            with self.subTest(block=key):
                self.assertEqual(fx.sha256(blocks[key]), digest)

    def test_the_ud10_weights_are_upstreams(self):
        source = inspect.getsource(ORIGIN.sample_cfgpp_ud10_ab)
        self.assertIn("history_weight=0.25", source)
        self.assertIn("zero_weight=1.0", source)
        self.assertIn("uncond_history_weight=0.1", source)
        self.assertEqual((cfgpp_ud10_ab.UD10_HISTORY_WEIGHT, cfgpp_ud10_ab.UD10_ZERO_WEIGHT,
                          cfgpp_ud10_ab.UD10_UNCOND_HISTORY_WEIGHT), (0.25, 1.0, 0.1))

    def test_forges_skip_predicate_is_the_one_the_port_mirrors(self):
        """The port's ``_uncond_was_skipped`` is Forge's own condition; Forge hands the post-CFG functions the
        request's ``uncond`` (never None) and zeros for a skipped unconditional prediction."""
        inner = fx.forge_definition(_SAMPLING_FUNCTION, "sampling_function_inner")
        self.assertIn('if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:',
                      inner)
        self.assertIn('"uncond": uncond, "cond_scale": cond_scale', inner)   # the original uncond, not uncond_
        self.assertIn('"uncond_denoised": uncond_pred', inner)
        batch = fx.forge_definition(_SAMPLING_FUNCTION, "calc_cond_uncond_batch")
        self.assertIn("out_uncond = torch.zeros_like(x_in)", batch)
        ours = inspect.getsource(cfgpp_ud10_ab._uncond_was_skipped)
        self.assertIn('math.isclose(args["cond_scale"], 1.0) and model_options.get("disable_cfg1_optimization", False) == False',
                      ours)


class OriginParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))

    def _pair(self, flow, steps, dtype, cond_scales, origin_fn, ours_fn, **kwargs):
        sampling, sigmas, x = _problem(flow, steps, dtype)
        comfy_host = _CfgHost(sampling, "comfy", cond_scales)
        forge_host = _CfgHost(sampling, "forge", cond_scales)
        origin = origin_fn(comfy_host, x.clone(), sigmas, extra_args={}, disable=True, **kwargs)
        seen = []
        ours = ours_fn(forge_host, x.clone(), sigmas, extra_args={}, disable=True, callback=seen.append, **kwargs)
        return origin, ours, comfy_host, forge_host, seen

    def test_ud10_ab_is_comfyuis_on_both_hosts(self):
        for flow in (False, True):
            for dtype in (torch.float32, torch.float64):
                for steps in (1, 2, 3, 6, 12):
                    for cfg in (1.0, 1.5, 2.0, 4.5):
                        with self.subTest(flow=flow, dtype=dtype, steps=steps, cfg=cfg):
                            origin, ours, comfy_host, forge_host, seen = self._pair(
                                flow, steps, dtype, [cfg], ORIGIN.sample_cfgpp_ud10_ab, cfgpp_ud10_ab.sample_cfgpp_ud10_ab)
                            self.assertTrue(torch.isfinite(ours).all())
                            self.assertTrue(torch.equal(ours, origin))
                            self.assertEqual([c.sigma for c in forge_host.calls], [c.sigma for c in comfy_host.calls])
                            self.assertEqual(forge_host.skipped, [cfg == 1.0] * steps)   # CFG 1 skips the uncond pass
                            self.assertEqual([info["i"] for info in seen], list(range(steps)))

    def test_skip_early_cfg_steps_use_the_conditional_prediction_like_comfyui(self):
        """Forge runs some steps at CFG 1 (Skip Early CFG, NGMS) with the uncond conditioning still in the
        arguments; the port treats them as ComfyUI treats a CFG-1 call."""
        for flow in (False, True):
            for scales in ([1.0, 1.0, 2.0], [2.0, 1.0, 2.0, 1.0, 1.5]):
                with self.subTest(flow=flow, scales=scales):
                    origin, ours, _comfy, forge_host, _seen = self._pair(
                        flow, 8, torch.float64, scales, ORIGIN.sample_cfgpp_ud10_ab, cfgpp_ud10_ab.sample_cfgpp_ud10_ab)
                    self.assertTrue(any(forge_host.skipped) and not all(forge_host.skipped))
                    self.assertTrue(torch.equal(ours, origin))

    def test_without_the_predicate_forge_would_use_its_zero_uncond(self):
        """Negative control: the plain upstream hook on Forge's arguments reads the zero ``uncond_denoised`` at CFG 1."""
        sampling, sigmas, x = _problem(True, 6, torch.float64)
        origin_on_forge = ORIGIN.sample_cfgpp_ud10_ab(_CfgHost(sampling, "forge", [1.0]), x.clone(), sigmas,
                                                      extra_args={}, disable=True)
        origin = ORIGIN.sample_cfgpp_ud10_ab(_CfgHost(sampling, "comfy", [1.0]), x.clone(), sigmas,
                                             extra_args={}, disable=True)
        self.assertFalse(torch.equal(origin_on_forge, origin))

    def test_the_history_sampler_with_other_weights(self):
        for flow in (False, True):
            for history_weight in (0.5, 0.0, 1.0):
                for zero_weight in (None, 0.5):
                    for zero_order in (1, 2):
                        for uncond_history_weight in (0.0, 0.3):
                            kwargs = dict(history_weight=history_weight, zero_weight=zero_weight,
                                          zero_order=zero_order, uncond_history_weight=uncond_history_weight)
                            with self.subTest(flow=flow, **kwargs):
                                origin, ours, *_ = self._pair(flow, 7, torch.float64, [1.7],
                                                              ORIGIN._sample_cfgpp_history,
                                                              cfgpp_ud10_ab.sample_cfgpp_history, **kwargs)
                                self.assertTrue(torch.equal(ours, origin))

    def test_the_hook_keeps_forges_cfg1_optimisation_and_the_callers_options(self):
        sampling, sigmas, x = _problem(True, 4, torch.float64)
        host = _CfgHost(sampling, "forge", [1.0])
        caller_options = {"transformer_options": {"k": 1}, "sampler_post_cfg_function": []}
        extra_args = {"model_options": caller_options}
        cfgpp_ud10_ab.sample_cfgpp_ud10_ab(host, x.clone(), sigmas, extra_args=extra_args, disable=True)
        used = host.calls[0].model_options
        self.assertNotIn("disable_cfg1_optimization", used)   # CFG 1 skips the uncond pass, as in ComfyUI
        self.assertEqual(len(used["sampler_post_cfg_function"]), 1)
        self.assertEqual(caller_options["sampler_post_cfg_function"], [])   # the caller's list is not appended to
        self.assertEqual(used["transformer_options"], {"k": 1})

    def test_a_flow_schedule_from_sigma_one_and_the_final_step(self):
        """σ0 = 1 on flow (α = 0) and the last step to σ = 0 (α = 1, the σ-zero extrapolation) stay finite."""
        sampling, sigmas, x = _problem(True, 5, torch.float32)
        self.assertEqual(float(sigmas[0]), 1.0)
        out = cfgpp_ud10_ab.sample_cfgpp_ud10_ab(_CfgHost(sampling, "forge", [2.0]), x, sigmas, disable=True)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
