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
  ``initialize`` the samplers run their plain CFG++ steps with an ``Extra Samplers status`` infotext.
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

from sam3ext.extra_samplers import common, euler_dy, substep_guard  # noqa: E402
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
        self.addCleanup(setattr, self.ks, "torch", torch)   # Sampler.initialize hijacks it; Forge never restores
        # Forge's sample() asks for progress bars (disable=False); keep the test log quiet.
        trange = self.ks.trange
        self.ks.trange = lambda *args, disable=None, **kwargs: trange(*args, disable=True, **kwargs)
        self.addCleanup(setattr, self.ks, "trange", trange)
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
        self.enterContext(fx.stub_modules({
            "modules": modules, "modules.sd_samplers": sd_samplers,
            "modules.sd_samplers_common": common, "modules.sd_samplers_kdiffusion": kdiffusion,
        }))
        self.enterContext(fx.installed_k_sampling(self.ks))
        registry.register(log=lambda m: None)
        self.data = {d.name: d for d in sd_samplers.all_samplers}

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

    def _sample(self, label, p):
        sampler = self.data[label].constructor(object())
        sampler.config = self.data[label]
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
        for label in registry.LABEL_ER_SDE_REVERSE_TIME, registry.LABEL_ER_SDE_ODE, registry.LABEL_DPMPP_4M_SDE, \
                registry.LABEL_EULER_DY_CFG_PP, registry.LABEL_EULER_SMEA_DY_CFG_PP:
            with self.subTest(label=label):
                samples, p, sampler = self._generate(label, [7], sigmas, max_stage=2, eta=0.5)
                self.assertEqual(tuple(samples.shape), (1, *SHAPE))
                self.assertTrue(torch.isfinite(samples).all())
                self.assertEqual(self.state.sampling_step, len(sigmas) - 2)   # callback saw every step
                expected = {
                    registry.LABEL_ER_SDE_REVERSE_TIME: {"ER SDE max stage": 2, "ER SDE eta": 0.5},
                    registry.LABEL_ER_SDE_ODE: {"ER SDE max stage": 2},
                }.get(label, {})
                self.assertEqual(p.extra_generation_params, expected)

    def test_every_image_of_a_batch_is_the_image_of_its_seed(self):
        self.opts.s_churn = 1.0       # make the Dy samplers draw and use churn noise
        sigmas = fx.flow_sigmas(7, dtype=torch.float32)[1:]   # below 1 so the flow churn applies
        for label in registry.LABEL_ER_SDE_REVERSE_TIME, registry.LABEL_DPMPP_4M_SDE, \
                registry.LABEL_EULER_DY_CFG_PP, registry.LABEL_EULER_SMEA_DY_CFG_PP:
            with self.subTest(label=label):
                batch, p, _ = self._generate(label, [11, 22], sigmas)
                alone = [self._generate(label, [seed], sigmas)[0][0] for seed in (11, 22)]
                self.assertTrue(torch.equal(batch[0], alone[0]))
                self.assertTrue(torch.equal(batch[1], alone[1]))
                self.assertFalse(torch.equal(alone[0], alone[1]))
                if label != registry.LABEL_DPMPP_4M_SDE:
                    self.assertEqual(p.extra_generation_params.get("Sigma churn"),
                                     1.0 if "Dy" in label else None)

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
        for label in registry.LABEL_EULER_DY_CFG_PP, registry.LABEL_EULER_SMEA_DY_CFG_PP:
            with self.subTest(label=label):
                self.stored.clear()
                self._generate(label, [3], sigmas)
                self.assertEqual(self.stored, [(1, *SHAPE)] * 2)
                self.assertEqual(tuple(self.state.current_latent.shape), (1, *SHAPE))


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


_DY_LABELS = (registry.LABEL_EULER_DY_CFG_PP, registry.LABEL_EULER_SMEA_DY_CFG_PP)
_FUNCS = {
    registry.LABEL_EULER_DY_CFG_PP: euler_dy.sample_euler_dy_cfg_pp,
    registry.LABEL_EULER_SMEA_DY_CFG_PP: euler_dy.sample_euler_smea_dy_cfg_pp,
}
# Spectrum settings that make each sampler's sub-steps meet a forecast call: its defaults for Euler Dy
# (sub-steps are calls 3 and 5, the first forecast is call 6); warm-up 4 for Euler SMEA Dy (calls 1, 3 / 4).
_SPECTRUM_BREAKS = {
    registry.LABEL_EULER_DY_CFG_PP: fx.SPECTRUM_DEFAULTS,
    registry.LABEL_EULER_SMEA_DY_CFG_PP: fx.SPECTRUM_DEFAULTS[:5] + (4,) + fx.SPECTRUM_DEFAULTS[6:],
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
