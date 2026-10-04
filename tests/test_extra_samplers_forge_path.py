"""The extra samplers through Forge's own sampling path, end to end on a toy model.

``KDiffusionSampler.__init__``/``sample`` (modules/sd_samplers_kdiffusion.py) and ``Sampler.__init__``/
``initialize``/``launch_sampling``/``callback_state``/``create_noise_sampler``/``add_infotext`` and
``TorchHijack`` (modules/sd_samplers_common.py) are taken out of Forge's files with ``ast`` and run
unchanged; ``k_diffusion.sampling`` is Forge's real module. Only the host services around them are
stand-ins (state, shared, the UNet patcher, ``get_sigmas``, the image RNG with one generator per seed).

What this pins down:

* Forge calls each sampler with the parameters its signature declares (sigmas, Brownian noise for
  DPM++ 4M SDE, the ER SDE values, the Dy ``after_substep`` hook) and the result is finite;
* reproducibility per seed and batch: every image of a batch is identical to the same seed rendered
  alone — the noise goes through ``TorchHijack`` (one generator per image) or Forge's per-image
  Brownian trees, never through the global RNG;
* the Dy sub-steps leave Forge's preview/interrupt latent at full resolution;
* on requests the sub-steps cannot run on (``substep_guard``) — Forge's real Spectrum Integrated
  forecaster (executed read-only from its file), a Wan 2.2 I2V ``concat_latent``, PiD, an
  ``extra_concat_condition`` — the crash path is reproduced without the guard, and through Forge's
  ``initialize`` the samplers run their plain (CFG++) Euler steps with an ``Extra Samplers status`` infotext;
* the DPM++ entries of ``dpmpp_flow`` next to Forge's own entries: ``DPM++ 3M (flow ODE)`` is Forge's
  ``DPM++ 3M SDE`` at Eta 0 (and the 2M ones Forge's ``DPM++ 2M SDE``), ``DPM++ 2M SDE Heun`` is Forge's 2M SDE
  with the ``solver_type: heun`` option, and the ``Eta`` infotext follows; UniPC bh2 runs Forge's UniPC;
* ``ER SDE (Tunable)`` at η 1 is Forge's built-in ``ER SDE`` (also with a full noise window), a noise window
  writes its infotext and keeps every image of a batch the image of its seed; ``Restart (flow)`` restarts
  with the request's per-image noise and writes ``Sigma noise``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

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

from sam3ext.extra_samplers import common, euler_dy, restart_flow, substep_guard  # noqa: E402
from sam3ext.extra_samplers import params as sampler_params  # noqa: E402
from sam3ext.extra_samplers import registry  # noqa: E402

_COMMON = "modules/sd_samplers_common.py"
_KDIFFUSION = "modules/sd_samplers_kdiffusion.py"
SHAPE = (4, 1, 6, 6)   # one Anima-like latent (C, T, H, W); the batch is stacked in front


class _ImageRNG:
    """Forge ``ImageRNG`` without subseeds: one CPU generator per seed (modules/rng.py)."""

    def __init__(self, shape, seeds):
        self.shape = tuple(shape)
        self.generators = [torch.Generator().manual_seed(seed) for seed in seeds]

    def first(self):
        return self.next()

    def next(self):
        return torch.stack([torch.randn(self.shape, generator=g) for g in self.generators])


class _FlowPrediction(fx.FlowSampling):
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        # Forge AbstractPrediction.noise_scaling, const branch (backend/modules/k_prediction.py)
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return sigma * noise + (1.0 - sigma) * latent_image


class _Denoiser(fx.ToyModel):
    """``CFGDenoiserKDiffusion`` stand-in: the toy CFG model plus the fields Forge's sampler touches."""

    def __init__(self, sampler):
        super().__init__(_FlowPrediction(3.0), cond_scale=1.5)
        unet = types.SimpleNamespace(model_options={"transformer_options": {}})
        self.inner_model.inner_model = types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=unet))
        self.sampler = sampler
        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        self.step = 0
        self.p = None


def _forge_sampler_class(namespace):
    def method(path, cls, name):
        return textwrap.indent(fx.forge_definition(path, name, cls=cls), "    ")

    tree = ast.parse(fx.forge_source(_COMMON))
    hijack = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TorchHijack")
    exec(compile(ast.get_source_segment(fx.forge_source(_COMMON), hijack), "<forge TorchHijack>", "exec"), namespace)
    sampler_methods = ["__init__", "callback_state", "launch_sampling", "initialize", "create_noise_sampler", "add_infotext"]
    source = "class Sampler:\n" + "\n\n".join(method(_COMMON, "Sampler", m) for m in sampler_methods) + "\n"
    source += "\n\nclass KDiffusionSampler(Sampler):\n" + "\n\n".join(
        method(_KDIFFUSION, "KDiffusionSampler", m) for m in ("__init__", "sample")
    ) + "\n\n    def get_sigmas(self, p, steps):\n        return p.test_sigmas\n"
    exec(compile(source, "<forge samplers>", "exec"), namespace)
    return namespace["KDiffusionSampler"]


class _ForgePathBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()

    def setUp(self):
        registry._LOGGED.clear()
        registry._CLASS_CACHE.clear()
        sampler_params.reset_active()
        self.addCleanup(registry._CLASS_CACHE.clear)
        self.addCleanup(sampler_params.reset_active)
        self.stored = []
        self.state = types.SimpleNamespace(sampling_step=0, sampling_steps=0, preview_step=0, current_latent=None,
                                           interrupted=False, skipped=False, set_current_image=lambda: None)
        self.opts = types.SimpleNamespace(eta_ancestral=1.0, s_churn=0.0, s_tmin=0.0, s_tmax=0.0, s_noise=1.0,
                                          sgm_noise_multiplier=False, hide_samplers=[])
        namespace = {
            "inspect": inspect, "torch": torch, "opts": self.opts, "state": self.state,
            "shared": types.SimpleNamespace(total_tqdm=types.SimpleNamespace(update=lambda: None)),
            "k_diffusion": types.SimpleNamespace(sampling=self.ks),
            "sampler_extra_params": ast.literal_eval(
                next(n.value for n in ast.parse(fx.forge_source(_KDIFFUSION)).body
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "sampler_extra_params")),
            "CFGDenoiserKDiffusion": _Denoiser,
            "sampling_prepare": lambda unet, x=None: None, "sampling_cleanup": lambda unet: None,
            "InterruptedException": type("InterruptedException", (BaseException,), {}),
        }
        self.namespace = namespace   # tests may swap the denoiser class (looked up at construction)
        base = _forge_sampler_class(namespace)
        self.base = base
        self.addCleanup(setattr, self.ks, "torch", torch)   # Sampler.initialize hijacks it; Forge never restores
        # Forge's sample() asks for progress bars (disable=False); keep the test log quiet.
        trange = self.ks.trange
        self.ks.trange = lambda *args, disable=None, **kwargs: trange(*args, disable=True, **kwargs)
        self.addCleanup(setattr, self.ks, "trange", trange)
        uni_pc = fx.forge_uni_pc()   # Forge's UniPC has its own progress bar
        self.enterContext(mock.patch.object(
            uni_pc, "trange", lambda *args, disable=None, **kwargs: trange(*args, disable=True, **kwargs)))
        # torchsde's float32 interval-edge notice, also printed by Forge's own SDE samplers
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="Should have t", category=UserWarning)

        def store_latent(latent):
            self.state.current_latent = latent
            self.stored.append(tuple(latent.shape))

        sd_samplers = types.ModuleType("modules.sd_samplers")
        sd_samplers.all_samplers = []

        def add_sampler(data):
            sd_samplers.all_samplers.append(data)

        sd_samplers.add_sampler = add_sampler
        sd_samplers.set_samplers = lambda: None
        common = types.ModuleType("modules.sd_samplers_common")
        common.SamplerData = _forge_sampler_data()
        common.store_latent = store_latent
        kdiffusion = types.ModuleType("modules.sd_samplers_kdiffusion")
        kdiffusion.KDiffusionSampler = base
        modules = types.ModuleType("modules")
        modules.__path__ = []
        modules.sd_samplers, modules.sd_samplers_common, modules.sd_samplers_kdiffusion = sd_samplers, common, kdiffusion
        modules.sd_samplers_extra = fx.forge_sd_samplers_extra()   # Forge's UniPC (UniPC bh2)
        self.enterContext(fx.stub_modules({
            "modules": modules, "modules.sd_samplers": sd_samplers,
            "modules.sd_samplers_common": common, "modules.sd_samplers_kdiffusion": kdiffusion,
            "modules.sd_samplers_extra": modules.sd_samplers_extra,
        }))
        self.enterContext(fx.installed_k_sampling(self.ks))
        self.enterContext(fx.installed_k_diffusion_deis(fx.forge_deis()))   # Forge's vendored DEIS coefficients
        registry.register(log=lambda m: None)
        self.data = {d.name: d for d in sd_samplers.all_samplers}
        self.common = common

    def _forge_entry(self, label, funcname, options):
        """One of Forge's own k-diffusion entries (modules/sd_samplers_kdiffusion.py ``samplers_data_k_diffusion``):
        ``KDiffusionSampler(funcname, model)`` with the given options."""
        base = self.base
        return self.common.SamplerData(label, lambda model, funcname=funcname: base(funcname, model), [], options)

    def _request(self, seeds, sigmas, **settings):
        unet = types.SimpleNamespace(model_options={"transformer_options": {}})
        p = types.SimpleNamespace(
            steps=len(sigmas) - 1, cfg_scale=1.5, extra_generation_params={}, eta=None, s_min_uncond=0.0,
            s_churn=self.opts.s_churn, s_tmin=self.opts.s_tmin, s_tmax=float("inf"), s_noise=self.opts.s_noise,
            all_seeds=list(seeds), iteration=0, batch_size=len(seeds), rng=_ImageRNG(SHAPE, seeds),
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=unet)), test_sigmas=sigmas,
        )
        sampler_params.apply_to(p, sampler_params.ErSdeSettings(**settings))
        return p

    def _sample(self, label, p, data=None):
        data = self.data[label] if data is None else data
        sampler = data.constructor(object())
        sampler.config = data
        x = p.rng.first()
        samples = sampler.sample(p, x, conditioning=None, unconditional_conditioning=None)
        self.ks.torch = torch
        return samples, sampler

    def _generate(self, label, seeds, sigmas, **settings):
        p = self._request(seeds, sigmas, **settings)
        samples, sampler = self._sample(label, p)
        return samples, p, sampler


class ForgeSamplingPathTests(_ForgePathBase):
    def test_every_sampler_runs_through_forges_sample(self):
        sigmas = fx.flow_sigmas(8, dtype=torch.float32)
        self.assertEqual(sorted(self.data), sorted(spec.label for spec in registry.SPECS))   # all registered
        for label in (spec.label for spec in registry.SPECS):
            with self.subTest(label=label):
                samples, p, sampler = self._generate(label, [7], sigmas, max_stage=2, eta=0.5)
                self.assertEqual(tuple(samples.shape), (1, *SHAPE))
                self.assertTrue(torch.isfinite(samples).all())
                self.assertEqual(self.state.sampling_step, len(sigmas) - 2)   # callback saw every step
                expected = {
                    registry.LABEL_ER_SDE_REVERSE_TIME: {"ER SDE max stage": 2, "ER SDE eta": 0.5},
                    registry.LABEL_ER_SDE_ODE: {"ER SDE max stage": 2},
                    registry.LABEL_ER_SDE_TUNABLE: {"ER SDE max stage": 2, "ER SDE eta": 0.5},
                }.get(label, {})
                self.assertEqual(p.extra_generation_params, expected)

    def test_every_image_of_a_batch_is_the_image_of_its_seed(self):
        self.opts.s_churn = 1.0       # make the Dy samplers draw and use churn noise
        sigmas = fx.flow_sigmas(7, dtype=torch.float32)[1:]   # below 1 so the flow churn applies
        for label in registry.LABEL_ER_SDE_REVERSE_TIME, registry.LABEL_DPMPP_4M_SDE, \
                registry.LABEL_EULER_DY_CFG_PP, registry.LABEL_EULER_SMEA_DY_CFG_PP, \
                registry.LABEL_EULER_DY, registry.LABEL_EULER_SMEA_DY, registry.LABEL_DPMPP_2M_SDE_HEUN:
            with self.subTest(label=label):
                batch, p, _ = self._generate(label, [11, 22], sigmas)
                alone = [self._generate(label, [seed], sigmas)[0][0] for seed in (11, 22)]
                self.assertTrue(torch.equal(batch[0], alone[0]))
                self.assertTrue(torch.equal(batch[1], alone[1]))
                self.assertFalse(torch.equal(alone[0], alone[1]))
                if label not in (registry.LABEL_DPMPP_4M_SDE, registry.LABEL_DPMPP_2M_SDE_HEUN):
                    self.assertEqual(p.extra_generation_params.get("Sigma churn"),
                                     1.0 if "Dy" in label else None)
                else:
                    self.assertNotIn("Sigma churn", p.extra_generation_params)

    def test_the_deterministic_entries_keep_every_batch_image_bit_exact(self):
        """The noise-free entries never mix the images of a batch: with an elementwise model (no batched-matmul
        float noise) each batch image is bit for bit the image of its seed alone — UniPC bh2 up to Forge's UniPC's
        own reductions over the batch-shaped history (a float32 ULP or two, as Forge's ``UniPC``). The in-Forge
        integration test (``test_integration_samplers_guidance``) compares the same on Anima's DiT, where batch
        size changes the float rounding — IPNDM_V's variable-step coefficients amplify that noise most."""
        sigmas = fx.flow_sigmas(12, dtype=torch.float32)
        for label in (registry.LABEL_ER_SDE_ODE, registry.LABEL_DPMPP_2M_FLOW_ODE, registry.LABEL_DPMPP_2M_HEUN_FLOW_ODE,
                      registry.LABEL_DPMPP_3M_FLOW_ODE, registry.LABEL_UNIPC_BH2, registry.LABEL_IPNDM,
                      registry.LABEL_IPNDM_V, registry.LABEL_DEIS, registry.LABEL_CFGPP_UD10_AB):
            with self.subTest(label=label):
                batch, _, _ = self._generate(label, [11, 22, 33], sigmas)
                alone = [self._generate(label, [seed], sigmas)[0][0] for seed in (11, 22, 33)]
                for index in range(3):
                    if label == registry.LABEL_UNIPC_BH2:
                        torch.testing.assert_close(batch[index], alone[index], rtol=0, atol=6e-8)
                    else:
                        self.assertTrue(torch.equal(batch[index], alone[index]), index)
                self.assertFalse(torch.equal(alone[0], alone[1]))

    def test_churn_noise_comes_from_the_requests_generators(self):
        self.opts.s_churn = 1.0
        sigmas = fx.flow_sigmas(7, dtype=torch.float32)[1:]
        first, _, _ = self._generate(registry.LABEL_EULER_DY_CFG_PP, [5], sigmas)
        torch.manual_seed(0)
        second, _, _ = self._generate(registry.LABEL_EULER_DY_CFG_PP, [5], sigmas)
        torch.manual_seed(12345)                   # the global RNG must not matter
        third, _, _ = self._generate(registry.LABEL_EULER_DY_CFG_PP, [5], sigmas)
        self.assertTrue(torch.equal(first, second) and torch.equal(second, third))
        other, _, _ = self._generate(registry.LABEL_EULER_DY_CFG_PP, [6], sigmas)
        self.assertFalse(torch.equal(first, other))

    def test_dy_substeps_leave_the_preview_latent_at_full_resolution(self):
        sigmas = fx.flow_sigmas(8, dtype=torch.float32)
        for label in _DY_LABELS:
            with self.subTest(label=label):
                self.stored.clear()
                self._generate(label, [3], sigmas)
                self.assertEqual(self.stored, [(1, *SHAPE)] * 2)
                self.assertEqual(tuple(self.state.current_latent.shape), (1, *SHAPE))


def _forge_table_options(label: str) -> tuple:
    """``(funcname, options)`` of one of Forge's own entries, read from ``samplers_k_diffusion`` with ``ast``."""
    tree = ast.parse(fx.forge_source(_KDIFFUSION))
    table = next(n.value for n in tree.body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "samplers_k_diffusion")
    for entry in table.elts:
        name, funcname, _aliases, options = entry.elts
        if isinstance(name, ast.Constant) and name.value == label:
            function = funcname.value if isinstance(funcname, ast.Constant) else ast.unparse(funcname)
            return function, ast.literal_eval(options)
    raise LookupError(label)


class DpmppFlowPathTests(_ForgePathBase):
    """``dpmpp_flow`` through Forge's ``KDiffusionSampler.sample`` next to Forge's own entries."""

    def _forge(self, label, p, **extra_options):
        funcname, options = _forge_table_options(label)
        return self._sample(label, p, data=self._forge_entry(label, funcname, {**options, **extra_options}))[0]

    def test_the_flow_ode_entries_are_forges_sde_entries_at_eta_zero(self):
        """"DPM++ 3M (flow ODE)" is Forge's "DPM++ 3M SDE" with Eta 0 (its per-image Brownian tree never drawn
        from), the 2M ones Forge's "DPM++ 2M SDE" (and its ``solver_type: heun`` registration) — bit for bit;
        the ODE entries ignore Forge's Eta and write no ``Eta``, Forge's SDE entries write ``Eta: 0``."""
        cases = (
            (registry.LABEL_DPMPP_3M_FLOW_ODE, "DPM++ 3M SDE", {}),
            (registry.LABEL_DPMPP_2M_FLOW_ODE, "DPM++ 2M SDE", {}),
            (registry.LABEL_DPMPP_2M_HEUN_FLOW_ODE, "DPM++ 2M SDE", {"solver_type": "heun"}),
        )
        for flow_sigmas in (fx.flow_sigmas(9, dtype=torch.float32), fx.flow_sigmas(9, dtype=torch.float32)[1:]):
            for ours_label, forge_label, extra in cases:
                with self.subTest(label=ours_label, first_sigma=float(flow_sigmas[0])):
                    p = self._request([11, 22], flow_sigmas)
                    p.eta = 0.6                             # Forge's Eta: not ours
                    ours, _ = self._sample(ours_label, p)
                    self.assertEqual(p.extra_generation_params, {})
                    forge_p = self._request([11, 22], flow_sigmas)
                    forge_p.eta = 0.0
                    forge = self._forge(forge_label, forge_p, **extra)
                    self.assertEqual(forge_p.extra_generation_params, {"Eta": 0.0})
                    self.assertTrue(torch.equal(ours, forge))

    def test_the_3m_flow_ode_has_forges_3m_options_but_no_scheduler_hint(self):
        _funcname, options = _forge_table_options("DPM++ 3M SDE")
        ours = self.data[registry.LABEL_DPMPP_3M_FLOW_ODE].options
        self.assertEqual(ours["discard_next_to_last_sigma"], options["discard_next_to_last_sigma"])
        self.assertNotIn("scheduler", ours)
        self.assertNotIn("brownian_noise", ours)   # nothing is drawn: no tree is built for it

    def test_dpmpp_2m_sde_heun_is_forges_2m_sde_with_the_heun_solver_option(self):
        """The registration A1111/reForge/ComfyUI use — Forge's ``sample_dpmpp_2m_sde`` with the option
        ``solver_type: heun`` — gives the same image, and both write Forge's ``Eta``."""
        sigmas = fx.flow_sigmas(8, dtype=torch.float32)[1:]
        for eta in (0.6, None):
            with self.subTest(eta=eta):
                p = self._request([11, 22], sigmas)
                p.eta = eta
                ours, _ = self._sample(registry.LABEL_DPMPP_2M_SDE_HEUN, p)
                forge_p = self._request([11, 22], sigmas)
                forge_p.eta = eta
                forge = self._forge("DPM++ 2M SDE", forge_p, solver_type="heun")
                self.assertTrue(torch.equal(ours, forge))
                self.assertEqual(p.extra_generation_params, forge_p.extra_generation_params)
                self.assertEqual(p.extra_generation_params, {"Eta": 0.6} if eta is not None else {})

    def test_unipc_bh2_runs_forges_unipc_with_another_variant(self):
        sigmas = fx.flow_sigmas(8, dtype=torch.float32)
        ours, _ = self._sample(registry.LABEL_UNIPC_BH2, self._request([3], sigmas))
        function, options = _forge_table_options("UniPC")
        self.assertEqual(function, "sd_samplers_extra.sample_unipc")
        self.assertEqual(self.data[registry.LABEL_UNIPC_BH2].options, options)
        bh1 = self._sample("UniPC", self._request([3], sigmas),
                           data=self._forge_entry("UniPC", fx.forge_sd_samplers_extra().sample_unipc, options))[0]
        self.assertTrue(torch.isfinite(ours).all())
        self.assertFalse(torch.equal(ours, bh1))


class ErSdeTunablePathTests(_ForgePathBase):
    """``ER SDE (Tunable)`` and the noise window through Forge's ``KDiffusionSampler.sample``."""

    def _builtin(self, p):
        funcname, options = _forge_table_options("ER SDE")
        self.assertEqual(funcname, "sample_er_sde")
        return self._sample("ER SDE", p, data=self._forge_entry("ER SDE", funcname, options))[0]

    def test_tunable_at_eta_one_is_forges_builtin_er_sde(self):
        """Same seeds, same per-image noise (``TorchHijack``): bit for bit, with the window off or full."""
        for sigmas in (fx.flow_sigmas(9, dtype=torch.float32), fx.flow_sigmas(9, dtype=torch.float32)[1:]):
            for window in (False, True):
                with self.subTest(first_sigma=float(sigmas[0]), window=window):
                    p = self._request([11, 22], sigmas, noise_window=window, noise_start=0.0, noise_end=1.0)
                    ours, _ = self._sample(registry.LABEL_ER_SDE_TUNABLE, p)
                    builtin = self._builtin(self._request([11, 22], sigmas))
                    self.assertTrue(torch.equal(ours, builtin))
                    self.assertEqual(p.extra_generation_params, {"ER SDE noise window": "0-1"} if window else {})

    def test_a_noise_window_through_forge(self):
        sigmas = fx.flow_sigmas(12, dtype=torch.float32)
        for label in (registry.LABEL_ER_SDE_TUNABLE, registry.LABEL_ER_SDE_REVERSE_TIME):
            with self.subTest(label=label):
                settings = dict(eta=1.5, noise_window=True, noise_start=0.2, noise_end=0.8)
                batch, p, _ = self._generate(label, [11, 22], sigmas, **settings)
                self.assertTrue(torch.isfinite(batch).all())
                self.assertEqual(p.extra_generation_params, {"ER SDE eta": 1.5, "ER SDE noise window": "0.2-0.8"})
                alone = [self._generate(label, [seed], sigmas, **settings)[0][0] for seed in (11, 22)]
                self.assertTrue(torch.equal(batch[0], alone[0]) and torch.equal(batch[1], alone[1]))
                unwindowed, _, _ = self._generate(label, [11, 22], sigmas, eta=1.5)
                self.assertFalse(torch.equal(batch, unwindowed))


class RestartFlowPathTests(_ForgePathBase):
    """``Restart (flow)`` through Forge's ``KDiffusionSampler.sample`` (24 steps: the restart happens)."""

    def test_restart_flow_through_forge(self):
        sigmas = fx.flow_sigmas(24, dtype=torch.float32)
        self.assertIsNotNone(restart_flow.restart_plan(sigmas, True))
        batch, p, sampler = self._generate(registry.LABEL_RESTART_FLOW, [11, 22], sigmas)
        self.assertTrue(torch.isfinite(batch).all())
        self.assertEqual(p.extra_generation_params, {})
        self.assertEqual(self.state.sampling_step, len(sigmas) - 2)   # the last callback is the last grid step
        alone = [self._generate(registry.LABEL_RESTART_FLOW, [seed], sigmas)[0][0] for seed in (11, 22)]
        self.assertTrue(torch.equal(batch[0], alone[0]) and torch.equal(batch[1], alone[1]))
        self.assertFalse(torch.equal(alone[0], alone[1]))
        # the restart noise comes from the request's generators, not the global RNG
        torch.manual_seed(12345)
        again, _, _ = self._generate(registry.LABEL_RESTART_FLOW, [11, 22], sigmas)
        self.assertTrue(torch.equal(again, batch))
        # Forge's Heun on the same grid differs (the restart did something) and is the < 20-step behaviour
        heun_p = self._request([11, 22], sigmas)
        funcname, options = _forge_table_options("Heun")
        heun = self._sample("Heun", heun_p, data=self._forge_entry("Heun", funcname, options))[0]
        self.assertFalse(torch.equal(heun, batch))
        short = fx.flow_sigmas(12, dtype=torch.float32)
        ours_short, _, _ = self._generate(registry.LABEL_RESTART_FLOW, [11], short)
        heun_short = self._sample("Heun", self._request([11], short),
                                  data=self._forge_entry("Heun", funcname, options))[0]
        self.assertTrue(torch.equal(ours_short, heun_short))

    def test_sigma_noise_reaches_the_restart(self):
        self.opts.s_noise = 0.9
        sigmas = fx.flow_sigmas(24, dtype=torch.float32)
        noisy, p, _ = self._generate(registry.LABEL_RESTART_FLOW, [5], sigmas)
        self.assertEqual(p.extra_generation_params, {"Sigma noise": 0.9})
        self.opts.s_noise = 1.0
        plain, p, _ = self._generate(registry.LABEL_RESTART_FLOW, [5], sigmas)
        self.assertEqual(p.extra_generation_params, {})
        self.assertFalse(torch.equal(noisy, plain))


# ---------------------------------------------------------------------------
# Requests the Dy/SMEA sub-steps cannot run on (sam3ext/extra_samplers/substep_guard.py)
# ---------------------------------------------------------------------------


class _RoutedDenoiser(_Denoiser):
    """The toy CFG through the UNet path the guard is about (backend/sampling/sampling_function.py):
    one [cond, uncond] batch handed with ``c`` and ``cond_or_uncond`` to the request's
    ``model_function_wrapper`` — Spectrum Integrated's when it is on — like ``calc_cond_uncond_batch``;
    the output has to match the batch (Forge adds it into ``out_conds`` of the input's shape); a
    Wan 2.2 I2V-style DiT concatenates ``dynamic_args.concat_latent`` on the channel axis whenever the
    latent has fewer channels than its ``in_dim`` (backend/nn/wan.py ``forward``)."""

    def __call__(self, x, sigma, **extra_args):
        options = extra_args.get("model_options") or {}
        unet = self.p.sd_model.forge_objects.unet
        rows = x.shape[0]
        x_in, t_in = torch.cat([x, x]), torch.cat([sigma, sigma])

        def apply_model(xin, timestep, **c):
            in_dim = getattr(getattr(getattr(unet, "model", None), "diffusion_model", None), "in_dim", None)
            if isinstance(in_dim, int) and xin.shape[1] < in_dim:
                r = sys.modules["backend.args"].dynamic_args.concat_latent.to(xin)
                if xin.shape[0] == 2:   # batch_cond_uncond
                    r = torch.cat((r, r), dim=0)
                torch.cat((xin, r), dim=1)
            salt = c["salt"].to(xin).reshape((-1,) + (1,) * (xin.ndim - 1))
            return fx.toy_x0(xin, timestep, 0.3) + 0.1 * salt

        c = {
            "salt": torch.tensor([1.0] * rows + [-1.0] * rows),
            "transformer_options": dict(options.get("transformer_options") or {}),
        }
        wrapper = (unet.model_options or {}).get("model_function_wrapper")
        if wrapper is None:
            out = apply_model(x_in, t_in, **c)
        else:
            out = wrapper(apply_model, {"input": x_in, "timestep": t_in, "c": c, "cond_or_uncond": [0, 1]})
        if tuple(out.shape) != tuple(x_in.shape):
            raise RuntimeError(f"the UNet returned {tuple(out.shape)} for an input of {tuple(x_in.shape)}")
        cond, uncond = out[:rows], out[rows:]
        denoised = uncond + (cond - uncond) * self.cond_scale
        for fn in options.get("sampler_post_cfg_function", []):
            denoised = fn({
                "denoised": denoised, "cond": [], "uncond": [], "cond_scale": self.cond_scale, "model": None,
                "uncond_denoised": uncond, "cond_denoised": cond, "sigma": sigma, "model_options": options,
                "input": x,
            })
        self.calls.append(fx.Call(
            shape=tuple(x.shape), sigma=float(sigma.reshape(-1)[0]),
            marker=c["transformer_options"].get(common.SUBSTEP_MARKER),
            init_latent=None, mask=None, image_cond=None, model_options=options,
        ))
        return denoised


_DY_LABELS = (registry.LABEL_EULER_DY_CFG_PP, registry.LABEL_EULER_SMEA_DY_CFG_PP,
              registry.LABEL_EULER_DY, registry.LABEL_EULER_SMEA_DY)
_FUNCS = {
    registry.LABEL_EULER_DY_CFG_PP: euler_dy.sample_euler_dy_cfg_pp,
    registry.LABEL_EULER_SMEA_DY_CFG_PP: euler_dy.sample_euler_smea_dy_cfg_pp,
    registry.LABEL_EULER_DY: euler_dy.sample_euler_dy,
    registry.LABEL_EULER_SMEA_DY: euler_dy.sample_euler_smea_dy,
}
# Spectrum settings that make each sampler's sub-steps meet a forecast call: its defaults for Euler Dy
# (sub-steps are calls 3 and 5, the first forecast is call 6); warm-up 4 for Euler SMEA Dy (calls 1, 3 / 4).
# The plain entries call the model at the same points as their CFG++ twins.
_SPECTRUM_BREAKS = {
    registry.LABEL_EULER_DY_CFG_PP: fx.SPECTRUM_DEFAULTS,
    registry.LABEL_EULER_SMEA_DY_CFG_PP: fx.SPECTRUM_DEFAULTS[:5] + (4,) + fx.SPECTRUM_DEFAULTS[6:],
    registry.LABEL_EULER_DY: fx.SPECTRUM_DEFAULTS,
    registry.LABEL_EULER_SMEA_DY: fx.SPECTRUM_DEFAULTS[:5] + (4,) + fx.SPECTRUM_DEFAULTS[6:],
}


class SubstepGuardPathTests(_ForgePathBase):
    """Forge's own ``KDiffusionSampler.sample`` → ``initialize`` (registry → substep_guard) → the sampler,
    against Forge's real Spectrum forecaster (``extensions-builtin/sd_forge_spectrum/lib_spectrum/
    forecaster.py``, executed read-only from Forge's file) and Forge-shaped Wan / PiD / extra-concat inputs."""

    STEPS = 10

    def setUp(self):
        super().setUp()
        self.namespace["CFGDenoiserKDiffusion"] = _RoutedDenoiser
        self.forecaster = fx.forge_spectrum_forecaster()
        self.logged = []
        self.enterContext(mock.patch.object(registry, "_log", self.logged.append))
        self.sigmas = fx.flow_sigmas(self.STEPS, dtype=torch.float32)

    def _plain_request(self, **unet_fields):
        p = self._request([5], self.sigmas)
        p.sd_model.forge_objects.unet = fx.UnetPatcherStandIn(
            p.sd_model.forge_objects.unet.model_options, **unet_fields)
        return p

    def _spectrum_request(self, settings=fx.SPECTRUM_DEFAULTS):
        p = self._plain_request()
        p.sd_model.forge_objects.unet = self.forecaster.SpectrumNode.patch(
            p.sd_model.forge_objects.unet, p.steps, *settings)
        return p

    def _dynamic_args(self, **values):
        dynamic_args = types.SimpleNamespace(ref_latents=[], concat_latent=None, lq_latent=[None, None], pid=False)
        dynamic_args.__dict__.update(values)
        backend = types.ModuleType("backend")
        backend.__path__ = []
        args_module = types.ModuleType("backend.args")
        args_module.dynamic_args = dynamic_args
        self.enterContext(fx.stub_modules({"backend": backend, "backend.args": args_module}))
        return dynamic_args

    def _direct(self, label, p, substeps):
        """The sampler function on the routed denoiser, without Forge's initialize (no guard)."""
        model = _RoutedDenoiser(None)
        model.p = p
        x = p.rng.first()
        out = _FUNCS[label](model, x, self.sigmas, extra_args={"cond_scale": p.cfg_scale}, disable=True,
                            noise_sampler=fx.NoiseSequence(x), substeps=substeps)
        return out, model

    def _status(self, p):
        return p.extra_generation_params.get(substep_guard.STATUS_KEY)

    def test_the_spectrum_crash_path_is_real_without_the_guard(self):
        for label in _DY_LABELS:
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "the UNet returned"):
                    self._direct(label, self._spectrum_request(_SPECTRUM_BREAKS[label]), substeps=True)
                # the same request without Spectrum runs its sub-steps fine
                _, model = self._direct(label, self._plain_request(), substeps=True)
                self.assertEqual(len(model.calls), self.STEPS + 2)

    def test_spectrum_integrated_gets_the_plain_cfg_pp_steps(self):
        for label in _DY_LABELS:
            with self.subTest(label=label):
                p = self._spectrum_request(_SPECTRUM_BREAKS[label])
                samples, sampler = self._sample(label, p)          # no RuntimeError
                calls = sampler.model_wrap_cfg.calls
                self.assertEqual(len(calls), self.STEPS)
                self.assertTrue(all(call.marker is None and call.shape == (1, *SHAPE) for call in calls))
                self.assertEqual(self._status(p), "dy sub-steps skipped (Spectrum)")
                self.assertTrue(torch.isfinite(samples).all())
                plain, _ = self._direct(label, self._spectrum_request(_SPECTRUM_BREAKS[label]), substeps=False)
                self.assertTrue(torch.equal(samples, plain))

    def test_the_warning_is_logged_once_per_reason(self):
        for _ in range(2):
            self._sample(registry.LABEL_EULER_DY_CFG_PP, self._spectrum_request())
        spectrum = [line for line in self.logged if "Spectrum" in line]
        self.assertEqual(len(spectrum), 1)
        self.assertIn(registry.LABEL_EULER_DY_CFG_PP, spectrum[0])
        self.assertIn(substep_guard.STATUS_KEY, spectrum[0])
        self.assertIn("plain CFG++ Euler steps", spectrum[0])

    def test_the_plain_entries_say_they_run_plain_euler_steps(self):
        for label in (registry.LABEL_EULER_DY, registry.LABEL_EULER_SMEA_DY):
            with self.subTest(label=label):
                self._sample(label, self._spectrum_request(_SPECTRUM_BREAKS[label]))
                (line,) = [line for line in self.logged if line.startswith(label + ":")]
                self.assertIn("it runs plain Euler steps instead", line)
                self.assertNotIn("CFG++", line)

    def test_wan_i2v_concat_latent_skips_the_substeps(self):
        self._dynamic_args(concat_latent=torch.zeros(1, 2, *SHAPE[1:]))
        wan_i2v = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(in_dim=SHAPE[0] + 2, out_dim=SHAPE[0]))
        for label in _DY_LABELS:
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match"):
                    self._direct(label, self._plain_request(model=wan_i2v), substeps=True)
                p = self._plain_request(model=wan_i2v)
                _, sampler = self._sample(label, p)
                self.assertEqual(len(sampler.model_wrap_cfg.calls), self.STEPS)
                self.assertEqual(self._status(p), "dy sub-steps skipped (Wan I2V concat_latent)")

    def test_a_t2v_model_ignores_a_leftover_concat_latent(self):
        self._dynamic_args(concat_latent=torch.zeros(1, 2, *SHAPE[1:]))
        wan_t2v = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(in_dim=SHAPE[0], out_dim=SHAPE[0]))
        p = self._plain_request(model=wan_t2v)
        _, sampler = self._sample(registry.LABEL_EULER_DY_CFG_PP, p)
        self.assertEqual(len(sampler.model_wrap_cfg.calls), self.STEPS + 2)
        self.assertIsNone(self._status(p))

    def test_pid_and_extra_concat_condition_skip_the_substeps(self):
        cases = (
            ("PiD lq_latent", {"pid": True, "lq_latent": [torch.zeros(1, *SHAPE), None]}, {}),
            ("extra_concat_condition", {}, {"extra_concat_condition": torch.zeros(1, 5, *SHAPE[1:])}),
        )
        for reason, dynamic_values, unet_fields in cases:
            for label in _DY_LABELS:
                with self.subTest(reason=reason, label=label):
                    self._dynamic_args(**dynamic_values)    # the latest stand-in wins; all undone at the end
                    p = self._plain_request(**unet_fields)
                    _, sampler = self._sample(label, p)
                    self.assertEqual(len(sampler.model_wrap_cfg.calls), self.STEPS)
                    self.assertEqual(self._status(p), f"dy sub-steps skipped ({reason})")

    def test_the_status_collects_every_reason_of_the_request(self):
        dynamic_args = self._dynamic_args()
        p = self._spectrum_request()
        self._sample(registry.LABEL_EULER_SMEA_DY_CFG_PP, p)
        dynamic_args.pid = True                     # e.g. the hires pass of the same request
        self._sample(registry.LABEL_EULER_DY_CFG_PP, p)
        self.assertEqual(self._status(p), "dy sub-steps skipped (Spectrum + PiD lq_latent)")


def _forge_sampler_data():
    """Forge's ``SamplerData`` (namedtuple + ``total_steps``), from modules/sd_samplers_common.py."""
    from collections import namedtuple

    namespace = {"namedtuple": namedtuple}
    tree = ast.parse(fx.forge_source(_COMMON))
    value = next(n.value for n in tree.body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SamplerDataTuple")
    namespace["SamplerDataTuple"] = eval(ast.unparse(value), namespace)
    body = textwrap.indent(fx.forge_definition(_COMMON, "total_steps", cls="SamplerData"), "    ")
    source = "class SamplerData(SamplerDataTuple):" + chr(10) + body + chr(10)
    exec(compile(source, "<forge SamplerData>", "exec"), namespace)
    return namespace["SamplerData"]


if __name__ == "__main__":
    unittest.main()
