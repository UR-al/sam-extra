# Adapted from aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 tests/test_forge_script.py:25-401
# (the stub harness: faked webui ``modules`` and ``k_diffusion``, RecordingSampler, FakeFullResHijack,
# _FakeAnimaModel, _fake_backend_args, make_p, run_hooks and the TestScriptHooks cases) — MIT License,
# Copyright (c) 2026 A. Izzuddin Al Faruq (the Forge script and its tests are Oleg Afonin's work in that fork):
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Changes: rewritten for scripts/anima_speed.py (process_before_every_sampling only, state on p, the
# guard reasons in the ``Anima SPEED status`` infotext); manual sigmas above 1 are refused (flow
# models only); reference latents of other models are now skipped too (aoleg's case asserted the
# opposite); the ControlNet / LLLite / SDXL / Spectrum / Wan I2V / PiD / no-coarse-step / sampler
# guards, the infotext paste, XYZ, settings, interrupt, fallback, console plan and batch-vs-single-seed
# cases are new.
"""scripts/anima_speed.py through its Forge hooks, with the webui stubbed (CPU only)."""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import re
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.speed import forge_host as fh  # noqa: E402

SCRIPT = ROOT / "scripts" / "anima_speed.py"


# ------------------------------------------------------------------------------------------------
# Stub environment
# ------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _stub_modules(stubs):
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _Registry:
    def __init__(self):
        self.settings = []
        self.before_ui = []


def _load_script():
    registry = _Registry()
    modules_stub = types.ModuleType("modules")

    class Script:
        is_img2img = False

    modules_stub.scripts = SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules_stub.script_callbacks = SimpleNamespace(
        on_ui_settings=registry.settings.append, on_before_ui=registry.before_ui.append,
    )
    options_stub = SimpleNamespace()
    modules_stub.shared = SimpleNamespace(opts=options_stub)
    with _stub_modules({"modules": modules_stub}):
        spec = importlib.util.spec_from_file_location("_test_anima_speed_script", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module, registry, modules_stub


SCRIPT_MODULE, REGISTRY, MODULES_STUB = _load_script()
MODULES_STUB.shared.opts.sam3_speed_log = False      # keep the test output readable


class KDiffusionSampler:
    """Named like Forge's sampler class; holds ``func`` like it."""

    def __init__(self, func):
        self.func = func


class RecordingSampler:
    """Fake k-diffusion solver: records calls, exercises randn_like, invokes the callback, returns x."""

    def __init__(self, use_randn=False):
        self.calls = []
        self.use_randn = use_randn
        self.__name__ = "sample_fake_euler"

    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, noise_sampler=None):
        self.calls.append({"shape": tuple(x.shape), "sigmas": [float(s) for s in sigmas], "noise_sampler": noise_sampler,
                           "s_churn": s_churn})
        for i in range(len(sigmas) - 1):
            if self.use_randn:
                kds = sys.modules["k_diffusion.sampling"]
                noise = kds.torch.randn_like(x)
                assert noise.shape == x.shape, f"randn_like shape {tuple(noise.shape)} != x {tuple(x.shape)}"
            if callback is not None:
                callback({"i": i, "x": x, "sigma": sigmas[i], "denoised": x})
        return x


class FakeFullResHijack:
    """Forge's TorchHijack: randn_like ignores the input shape and returns full-resolution noise."""

    def __init__(self, full_shape):
        self.full_shape = full_shape

    def __getattr__(self, item):
        if item == "randn_like":
            return self.randn_like
        return getattr(torch, item)

    def randn_like(self, x):
        return torch.zeros(self.full_shape)


class FakeImageRNG:
    """modules.rng.ImageRNG: one generator per image seed, ``next()`` stacks the per-seed draws."""

    def __init__(self, shape, seeds):
        self.shape = tuple(shape)
        self.generators = [torch.Generator().manual_seed(int(s)) for s in seeds]

    def next(self):
        return torch.stack([torch.randn(self.shape, generator=g) for g in self.generators])


class FakeTorchHijack:
    def __init__(self, rng):
        self.rng = rng

    def __getattr__(self, item):
        if item == "randn_like":
            return self.randn_like
        return getattr(torch, item)

    def randn_like(self, x):
        return self.rng.next()


def _forge_rng_module():
    module = types.ModuleType("modules.rng")
    module.randn_without_seed = lambda shape, generator=None: torch.randn(shape, generator=generator)
    return module


@contextlib.contextmanager
def fake_kdiffusion(torch_attr, brownian_cls=None):
    sampling = types.ModuleType("k_diffusion.sampling")
    sampling.torch = torch_attr
    if brownian_cls is not None:
        sampling.BrownianTreeNoiseSampler = brownian_cls
    package = types.ModuleType("k_diffusion")
    package.sampling = sampling
    with _stub_modules({"k_diffusion": package, "k_diffusion.sampling": sampling}):
        yield sampling


@contextlib.contextmanager
def fake_backend_args(ref_latents=(), **extra):
    args_stub = types.ModuleType("backend.args")
    args_stub.dynamic_args = SimpleNamespace(ref_latents=list(ref_latents), **extra)
    backend_stub = types.ModuleType("backend")
    backend_stub.args = args_stub
    with _stub_modules({"backend": backend_stub, "backend.args": args_stub}):
        yield args_stub


class Anima:
    """Stands in for Forge's Anima engine (the guard reads the class name)."""

    is_inpaint = False


class SDXL:
    is_inpaint = False


class Predictor:
    def __init__(self, prediction_type="const"):
        self.prediction_type = prediction_type

    def percent_to_sigma(self, percent):
        return 1.0 - percent


class Unet:
    def __init__(self, sigmas=None, prediction_type="const"):
        self.model_options = {"transformer_options": {"sampling_sigmas": sigmas}}
        self.controlnet_linked_list = None
        self.model = SimpleNamespace(predictor=Predictor(prediction_type))


class CFGDenoiser:
    """The ``model`` Forge hands to ``sampler.func`` (``model_wrap_cfg``)."""

    def __init__(self, p, prediction_type="const"):
        self.p = p
        self.mask = None
        self.inner_model = SimpleNamespace(predictor=Predictor(prediction_type))
        self.calls = []

    def __call__(self, x, sigma, **kwargs):
        self.calls.append((tuple(x.shape), float(sigma.flatten()[0])))
        s = sigma.reshape(sigma.shape + (1,) * (x.ndim - 1)).to(x.dtype)
        return torch.tanh(x) * (0.6 - 0.3 * s) + 0.05 * x.mean(dim=tuple(range(1, x.ndim)), keepdim=True)


def make_p(func, *, model_cls=Anima, sigmas=None, is_hr_pass=False, mask=None, seeds=(123,), img2img=False,
           prediction_type="const", rng=None):
    sd_model = model_cls()
    sd_model.forge_objects = SimpleNamespace(unet=Unet(sigmas, prediction_type))
    cls = type("StableDiffusionProcessingImg2Img" if img2img else "StableDiffusionProcessingTxt2Img", (), {})
    p = cls()
    p.sampler = KDiffusionSampler(func)
    p.sd_model = sd_model
    p.is_hr_pass = is_hr_pass
    p.mask = mask
    p.image_mask = None
    p.extra_generation_params = {}
    p.seeds = list(seeds)
    p.rng = rng
    return p


UI_DEFAULTS = dict(fh.DEFAULTS)
UI_DEFAULTS.update(enabled=True, threshold="manual", manual_sigmas="0.72")


def ui_args(**overrides):
    values = {**UI_DEFAULTS, **overrides}
    return [values[name] for name in fh.ARG_NAMES]


def run_hooks(p, **overrides):
    script = SCRIPT_MODULE.AnimaSpeed()
    script.process_before_every_sampling(p, *ui_args(**overrides))
    return script


def sample(p, x, sigmas, model=None, **kwargs):
    """What Forge does after the hooks: ``self.func(model_wrap_cfg, x, sigmas=..., ...)``."""
    model = model if model is not None else CFGDenoiser(p)
    seen = []
    out = p.sampler.func(model, x, sigmas=sigmas, extra_args={}, callback=lambda d: seen.append(d["i"]),
                         disable=True, **kwargs)
    return out, seen, model


def status(p):
    return p.extra_generation_params.get(fh.STATUS_KEY, "")


class ScriptTestCase(unittest.TestCase):
    def setUp(self):
        self._stack = contextlib.ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(fake_backend_args())


# ------------------------------------------------------------------------------------------------
# aoleg's script-hook cases, adapted
# ------------------------------------------------------------------------------------------------


class ScriptHookTests(ScriptTestCase):
    def test_wraps_sampler_and_runs(self):
        fake = RecordingSampler(use_randn=True)
        sigmas = torch.linspace(1.0, 0.0, 21)
        p = make_p(fake, sigmas=sigmas, seeds=(42,))
        full_shape = (1, 4, 32, 32)
        hijack = FakeFullResHijack(full_shape)
        with fake_kdiffusion(hijack) as kds:
            run_hooks(p)
            self.assertTrue(getattr(p.sampler.func, "_sam3_speed_wrapped", False))
            self.assertIn("s_churn", inspect.signature(p.sampler.func).parameters)   # functools.wraps
            self.assertIn(fh.KEY, p.extra_generation_params)
            self.assertIn("pending", status(p))
            sentinel = object()
            out, seen, _ = sample(p, torch.randn(*full_shape), sigmas, s_churn=0.25, noise_sampler=sentinel)
            self.assertIs(kds.torch, hijack)             # restored after the pass
        self.assertEqual([c["shape"] for c in fake.calls], [(1, 4, 16, 16), (1, 4, 32, 32)])
        self.assertEqual(out.shape, full_shape)
        self.assertEqual([c["s_churn"] for c in fake.calls], [0.25, 0.25])
        self.assertIs(fake.calls[1]["noise_sampler"], sentinel)
        self.assertEqual(seen, list(range(20)))
        self.assertTrue(status(p).startswith("base: applied (transition)"), status(p))

    def test_disabled_leaves_sampler_untouched_and_clears_stale_keys(self):
        fake = RecordingSampler()
        p = make_p(fake)
        p.extra_generation_params.update({fh.KEY: "old", fh.STATUS_KEY: "old"})
        run_hooks(p, enabled=False)
        self.assertIs(p.sampler.func, fake)
        self.assertNotIn(fh.KEY, p.extra_generation_params)
        self.assertNotIn(fh.STATUS_KEY, p.extra_generation_params)

    def test_inpaint_and_masks_are_skipped(self):
        sigmas = torch.linspace(1.0, 0.0, 11)
        for kind in ("mask", "image_mask", "inpaint_model", "denoiser_mask"):
            with self.subTest(kind=kind):
                fake = RecordingSampler()
                p = make_p(fake, sigmas=sigmas, mask=torch.ones(1) if kind == "mask" else None)
                if kind == "image_mask":
                    p.image_mask = object()
                if kind == "inpaint_model":
                    p.sd_model.is_inpaint = True
                run_hooks(p)
                model = CFGDenoiser(p)
                if kind == "denoiser_mask":
                    model.mask = torch.ones(1)
                out, _, _ = sample(p, torch.randn(1, 4, 16, 16), sigmas, model=model)
                self.assertEqual([c["shape"] for c in fake.calls], [(1, 4, 16, 16)])   # one plain call
                self.assertIn("inpainting / mask", status(p))

    def test_anima_reference_latents_are_skipped(self):
        fake = RecordingSampler()
        sigmas = torch.linspace(1.0, 0.0, 11)
        p = make_p(fake, sigmas=sigmas)
        with fake_backend_args(ref_latents=[torch.zeros(1, 16, 1, 128, 128)]):
            run_hooks(p)
            sample(p, torch.randn(1, 4, 16, 16), sigmas)
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("Anima reference latents", status(p))

    def test_anima_without_references_is_applied(self):
        fake = RecordingSampler()
        sigmas = torch.linspace(1.0, 0.0, 11)
        p = make_p(fake, sigmas=sigmas)
        with fake_backend_args(ref_latents=[]):
            run_hooks(p)
            sample(p, torch.randn(1, 4, 16, 16), sigmas)
        self.assertEqual(len(fake.calls), 2)

    def test_reference_latents_on_other_models_are_skipped(self):
        """Forge 2.29.2's Flux (Kontext, Flux.2 Klein), Qwen-Image-Edit, Krea 2 and Nunchaku DiTs append a
        non-empty dynamic_args.ref_latents to the image tokens at full size (aoleg guarded only Anima)."""
        sigmas = torch.linspace(1.0, 0.0, 11)
        for engine in ("Flux", "Flux2", "QwenImage", "Krea2"):
            with self.subTest(engine=engine):
                fake = RecordingSampler()
                p = make_p(fake, sigmas=sigmas, model_cls=type(engine, (), {"is_inpaint": False}))
                with fake_backend_args(ref_latents=[torch.zeros(1, 16, 128, 128)]):
                    run_hooks(p)
                    sample(p, torch.randn(1, 4, 16, 16), sigmas)
                self.assertEqual(len(fake.calls), 1)
                self.assertIn("skipped - reference latents are present", status(p))
                self.assertNotIn("Anima reference", status(p))
        fake = RecordingSampler()                       # an empty list (refs off) is no reason to skip
        p = make_p(fake, sigmas=sigmas, model_cls=type("Flux", (), {"is_inpaint": False}))
        with fake_backend_args(ref_latents=[]):
            run_hooks(p)
            sample(p, torch.randn(1, 4, 16, 16), sigmas)
        self.assertEqual(len(fake.calls), 2)

    def test_wan_i2v_concat_latent_is_skipped(self):
        """backend/nn/wan.py concatenates dynamic_args.concat_latent on the channel axis when the latent
        has fewer channels than the DiT's in_dim (I2V 36); a T2V DiT (in_dim = latent channels) does not."""
        sigmas = torch.linspace(1.0, 0.0, 11)
        for in_dim, skipped in ((36, True), (4, False)):
            with self.subTest(in_dim=in_dim):
                fake = RecordingSampler()
                p = make_p(fake, sigmas=sigmas, model_cls=type("Wan", (), {"is_inpaint": False}))
                p.sd_model.forge_objects.unet.model.diffusion_model = SimpleNamespace(in_dim=in_dim)
                with fake_backend_args(concat_latent=torch.zeros(1, 20, 3, 16, 16)):
                    run_hooks(p)
                    sample(p, torch.randn(1, 4, 3, 16, 16), sigmas)
                self.assertEqual(len(fake.calls), 1 if skipped else 2)
                self.assertEqual("concat_latent" in status(p), skipped, status(p))

    def test_pid_lq_latent_is_skipped(self):
        """backend/nn/pixeldit/pid.py reads dynamic_args.lq_latent[0] (the full-size input) every forward."""
        sigmas = torch.linspace(1.0, 0.0, 11)
        for extra in (dict(lq_latent=[torch.zeros(1, 16, 16, 16), torch.tensor([0.1])]), dict(pid=True, lq_latent=[None, None])):
            with self.subTest(extra=sorted(extra)):
                fake = RecordingSampler()
                p = make_p(fake, sigmas=sigmas, model_cls=type("PiD", (), {"is_inpaint": False}))
                with fake_backend_args(**extra):
                    run_hooks(p)
                    sample(p, torch.randn(1, 3, 16, 16), sigmas)
                self.assertEqual(len(fake.calls), 1)
                self.assertIn("PiD low-quality latent", status(p))
        fake = RecordingSampler()                       # other models: lq_latent is [None, None] after reset
        p = make_p(fake, sigmas=sigmas)
        with fake_backend_args(pid=False, lq_latent=[None, None], concat_latent=None):
            run_hooks(p)
            sample(p, torch.randn(1, 4, 16, 16), sigmas)
        self.assertEqual(len(fake.calls), 2)

    def test_hires_pass_needs_the_checkbox(self):
        fake = RecordingSampler()
        p = make_p(fake, is_hr_pass=True)
        run_hooks(p, hires=False)
        self.assertIs(p.sampler.func, fake)
        self.assertIn("hires: not applied - Apply to Hires pass is off", status(p))
        p2 = make_p(RecordingSampler(), is_hr_pass=True)
        run_hooks(p2, hires=True)
        self.assertTrue(getattr(p2.sampler.func, "_sam3_speed_wrapped", False))

    def test_no_double_wrap(self):
        fake = RecordingSampler()
        p = make_p(fake)
        script = run_hooks(p)
        first = p.sampler.func
        script.process_before_every_sampling(p, *ui_args())
        self.assertIsNot(p.sampler.func, first)
        self.assertIs(p.sampler.func._sam3_speed_base, fake)    # replaced, not stacked

    def test_invalid_settings_are_reported_and_leave_the_sampler(self):
        for overrides, text in (
            (dict(scales="0.5,0.9"), "last scale must equal 1.0"),
            (dict(manual_sigmas="3.5"), "(0, 1)"),            # flow sigmas only (aoleg allowed > 1 for eps)
            (dict(mode="sideways"), "mode must be one of"),
            (dict(threshold="delta_optimal", delta=1.5), "delta must be in (0, 1)"),
            (dict(preset="sd15"), "unknown spectrum preset"),
        ):
            with self.subTest(overrides=overrides):
                fake = RecordingSampler()
                p = make_p(fake)
                run_hooks(p, **overrides)
                self.assertIs(p.sampler.func, fake)
                self.assertIn("invalid settings", status(p))
                self.assertIn(text, status(p))

    def test_spectral_seed(self):
        p = make_p(RecordingSampler(), seeds=(42, 43))
        host = fh.ForgeSpeedHost(p, None, torch.linspace(1, 0, 5), fh.coerce_settings(ui_args(seed=999)), print)
        self.assertEqual(host.seeds(2), [999, 999])
        host = fh.ForgeSpeedHost(p, None, torch.linspace(1, 0, 5), fh.coerce_settings(ui_args()), print)
        self.assertEqual(host.seeds(2), [42, 43])
        self.assertEqual(host.seeds(3), [42, 43, 44])       # a batch the request does not describe


# ------------------------------------------------------------------------------------------------
# New guards
# ------------------------------------------------------------------------------------------------


class GuardTests(ScriptTestCase):
    sigmas = torch.linspace(1.0, 0.0, 11)

    def _run(self, p, model=None, x=None):
        run_hooks(p)
        x = torch.randn(1, 4, 16, 16) if x is None else x
        return sample(p, x, p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"],
                      model=model)

    def test_controlnet(self):
        fake = RecordingSampler()
        p = make_p(fake, sigmas=self.sigmas)
        p.sd_model.forge_objects.unet.controlnet_linked_list = object()
        self._run(p)
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("ControlNet is on", status(p))

    def test_controlnet_lllite_for_anima(self):
        fake = RecordingSampler()
        p = make_p(fake, sigmas=self.sigmas)
        with fake_backend_args(ACTIVE_LLLITE_DIT={object()}):
            self._run(p)
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("ControlNet-LLLite is on", status(p))

    def test_sdxl_and_other_non_flow_models(self):
        for kind in ("epsilon", "v_prediction"):
            with self.subTest(kind=kind):
                fake = RecordingSampler()
                p = make_p(fake, model_cls=SDXL, sigmas=torch.tensor([14.6, 7.0, 2.0, 0.5, 0.0]), prediction_type=kind)
                self._run(p, model=CFGDenoiser(p, prediction_type=kind))
                self.assertEqual(len(fake.calls), 1)
                self.assertIn(f"not a flow-matching model (prediction_type={kind})", status(p))
                self.assertIn("SDXL", status(p))

    def test_forge_spectrum_integrated_steps_aside(self):
        def spectrum_unet_wrapper(model_function, kwargs):  # same name as Forge's
            return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

        fake = RecordingSampler()
        p = make_p(fake, sigmas=self.sigmas)
        p.sd_model.forge_objects.unet.model_options["model_function_wrapper"] = spectrum_unet_wrapper
        self._run(p)
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("Forge Spectrum Integrated is on", status(p))
        # another wrapper (e.g. the Guidance suite's) does not count
        fake2 = RecordingSampler()
        p2 = make_p(fake2, sigmas=self.sigmas)
        p2.sd_model.forge_objects.unet.model_options["model_function_wrapper"] = lambda f, k: f(k["input"], k["timestep"])
        self._run(p2)
        self.assertEqual(len(fake2.calls), 2)

    def test_img2img_starting_below_the_transition_is_plain_sampling(self):
        fake = RecordingSampler()
        whole = torch.linspace(1.0, 0.0, 21)
        p = make_p(fake, sigmas=whole, img2img=True)
        run_hooks(p, manual_sigmas="0.72")
        x = torch.randn(1, 4, 16, 16)
        out, _, _ = sample(p, x, whole[8:])                 # starts at 0.6 <= 0.72
        self.assertEqual([c["shape"] for c in fake.calls], [(1, 4, 16, 16)])
        self.assertEqual(fake.calls[0]["sigmas"], [float(s) for s in whole[8:]])
        self.assertIs(out, x)
        self.assertIn("img2img: skipped - no coarse steps", status(p))

    def test_img2img_with_coarse_steps_records_the_rescale_setting(self):
        fake = RecordingSampler()
        whole = torch.linspace(1.0, 0.0, 21)
        p = make_p(fake, sigmas=whole, img2img=True)
        run_hooks(p, manual_sigmas="0.72")
        sample(p, torch.randn(1, 4, 16, 16), whole[2:])     # starts at 0.9 > 0.72
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(p.extra_generation_params[fh.INFOTEXT_IMG2IMG_RESCALE], "True")
        self.assertIn("init=rescaled", status(p))

    def test_console_plan_quotes_the_sigmas_the_run_uses(self):
        """respace re-spaces the schedule at each hand-off: the plan line must quote those sigmas (it
        printed the unpatched schedule's), the same ones the run reports in the status."""
        sigmas = torch.linspace(1.0, 0.0, 21)
        p = make_p(RecordingSampler(), sigmas=sigmas)
        settings = fh.coerce_settings(ui_args(mode="respace", scales="0.25,0.5,1.0", manual_sigmas="0.9,0.6"))
        lines = []
        self.assertIsNone(fh.attach(p, settings, "base", log=lines.append))
        sample(p, torch.randn(1, 4, 16, 16), sigmas)
        plan_line = next(line for line in lines if " plan - " in line)
        ran = re.findall(r"step (\d+) sigma ([0-9.]+)->([0-9.]+)", status(p))
        self.assertEqual(len(ran), 2)
        self.assertNotEqual(float(ran[1][1]), round(float(sigmas[int(ran[1][0])]), 4))   # re-spaced, not the input
        for step, old, new in ran:
            self.assertIn(f"step {step}: ", plan_line)
            self.assertIn(f"(sigma {old}->{new})", plan_line)

    def test_xyz_cells_sharing_generation_params_do_not_inherit_the_rescale_value(self):
        """Forge's xyz_grid runs every cell on copy(p): one extra_generation_params dict for all cells."""
        whole = torch.linspace(1.0, 0.0, 21)
        first = make_p(RecordingSampler(), sigmas=whole, img2img=True)
        run_hooks(first, manual_sigmas="0.72")
        sample(first, torch.randn(1, 4, 16, 16), whole[2:])           # coarse steps: value written
        self.assertIn(fh.INFOTEXT_IMG2IMG_RESCALE, first.extra_generation_params)
        second = make_p(RecordingSampler(), sigmas=whole, img2img=True)
        second.extra_generation_params = first.extra_generation_params
        run_hooks(second, manual_sigmas="0.5")
        sample(second, torch.randn(1, 4, 16, 16), whole[12:])          # starts at 0.4: no coarse step
        self.assertIn("no coarse steps", status(second))
        self.assertNotIn(fh.INFOTEXT_IMG2IMG_RESCALE, second.extra_generation_params)

    def test_unsupported_samplers(self):
        def restart_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return x

        def sample_unipc(model, x, sigmas, extra_args=None, callback=None, disable=False):
            return x

        for func in (restart_sampler, sample_unipc):
            with self.subTest(func=func.__name__):
                p = make_p(func)
                run_hooks(p)
                self.assertIs(p.sampler.func, func)
                self.assertIn("skipped", status(p))
        p = make_p(RecordingSampler())
        p.sampler = SimpleNamespace(func=RecordingSampler())   # DDIM/PLMS-style sampler class
        run_hooks(p)
        self.assertIn("is not a k-diffusion sampler", status(p))

    def test_no_schedule_solvers_pass_through(self):
        calls = []

        def sample_fake_fast(model, x, sigma_min=None, sigma_max=None, n=None, extra_args=None, callback=None, disable=None):
            calls.append((sigma_min, sigma_max, n))
            return x

        p = make_p(sample_fake_fast)
        run_hooks(p)
        p.sampler.func(CFGDenoiser(p), torch.randn(1, 4, 8, 8), extra_args={}, callback=None, disable=True,
                       sigma_min=0.1, sigma_max=1.0, n=5)
        self.assertEqual(calls, [(0.1, 1.0, 5)])
        self.assertIn("no sigma schedule", status(p))

    def test_wrapper_is_one_shot_and_ignores_other_requests(self):
        fake = RecordingSampler()
        sigmas = torch.linspace(1.0, 0.0, 11)
        p = make_p(fake, sigmas=sigmas)
        run_hooks(p)
        other = make_p(RecordingSampler(), sigmas=sigmas)
        sample(p, torch.randn(1, 4, 16, 16), sigmas, model=CFGDenoiser(other))   # e.g. img2img-hires-fix copy(p)
        self.assertEqual(len(fake.calls), 1)
        sample(p, torch.randn(1, 4, 16, 16), sigmas)
        self.assertEqual(len(fake.calls), 3)
        sample(p, torch.randn(1, 4, 16, 16), sigmas)                              # second run: plain
        self.assertEqual(len(fake.calls), 4)

    def test_failure_inside_speed_falls_back_to_plain_sampling(self):
        class Fragile(RecordingSampler):
            def __call__(self, model, x, sigmas, **kwargs):
                if x.shape[-1] != 16:
                    raise RuntimeError("coarse grid not supported by this extension")
                return super().__call__(model, x, sigmas, **kwargs)

        fake = Fragile()
        sigmas = torch.linspace(1.0, 0.0, 11)
        p = make_p(fake, sigmas=sigmas)
        run_hooks(p)
        x = torch.randn(1, 4, 16, 16)
        out, _, _ = sample(p, x, sigmas)
        self.assertIs(out, x)
        self.assertEqual(fake.calls[-1]["sigmas"], [float(s) for s in sigmas])
        self.assertIn("fell back to plain sampling after RuntimeError", status(p))
        self.assertIs(p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"], sigmas)

    def test_interrupt_during_a_coarse_segment_returns_a_full_size_latent(self):
        class InterruptedException(BaseException):
            pass

        state = SimpleNamespace(current_latent=None)

        def interrupted(model, x, sigmas, **kwargs):
            state.current_latent = torch.full_like(x, 0.25)          # Forge stores the denoised preview
            raise InterruptedException()

        sigmas = torch.linspace(1.0, 0.0, 11)
        p = make_p(interrupted, sigmas=sigmas)
        run_hooks(p)
        shared = SimpleNamespace(state=state)
        with _stub_modules({"modules.shared": shared}):
            with self.assertRaises(InterruptedException):
                sample(p, torch.randn(2, 4, 1, 16, 12), sigmas)
        self.assertEqual(tuple(state.current_latent.shape), (2, 4, 1, 16, 12))
        torch.testing.assert_close(state.current_latent, torch.full((2, 4, 1, 16, 12), 0.25))
        self.assertIn("interrupted", status(p))
        self.assertIs(p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"], sigmas)


class SamplingSigmasTests(ScriptTestCase):
    def test_patched_schedule_is_published_during_the_run_and_restored(self):
        whole = torch.linspace(1.0, 0.0, 21)
        seen = []

        def fake(model, x, sigmas, extra_args=None, callback=None, disable=None):
            seen.append(p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"].clone())
            return x

        for mode in ("transition", "respace"):
            with self.subTest(mode=mode):
                seen.clear()
                p = make_p(fake, sigmas=whole, img2img=True)
                run_hooks(p, mode=mode, manual_sigmas="0.7")
                sliced = whole[2:]                                   # img2img walks sigmas[offset:]
                sample(p, torch.randn(1, 4, 16, 16), sliced)
                coarse, full = seen
                self.assertTrue(torch.equal(coarse, whole))
                changed = (full != whole).nonzero().flatten().tolist()
                step = next(j for j in range(len(sliced)) if float(sliced[j]) <= 0.7)
                if mode == "transition":
                    self.assertEqual(changed, [2 + step])
                else:
                    self.assertEqual(changed[0], 2 + step)
                    self.assertGreater(len(changed), 1)
                self.assertIs(p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"], whole)


# ------------------------------------------------------------------------------------------------
# Per-seed noise through the Forge host
# ------------------------------------------------------------------------------------------------


class AncestralEuler:
    """Euler ancestral (flow) drawing its noise through ``k_diffusion.sampling.torch.randn_like``."""

    __name__ = "sample_fake_ancestral"

    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        kds = sys.modules["k_diffusion.sampling"]
        s_in = x.new_ones([x.shape[0]])
        for i in range(len(sigmas) - 1):
            denoised = model(x, sigmas[i] * s_in)
            if callback is not None:
                callback({"i": i, "x": x})
            if float(sigmas[i + 1]) == 0.0:
                x = denoised
                continue
            ratio = float(sigmas[i + 1] / sigmas[i]) ** 2
            x = ratio * x + (1 - ratio) * denoised + 0.3 * float(sigmas[i + 1]) * kds.torch.randn_like(x)
        return x


class PerSeedNoiseTests(ScriptTestCase):
    def _generate(self, seeds, sampler, *, mode, shape=(4, 1, 16, 16), transform="dct"):
        sigmas = torch.linspace(1.0, 0.0, 13)
        rng = FakeImageRNG(shape, seeds)
        p = make_p(sampler, sigmas=sigmas, seeds=seeds, rng=rng)
        with fake_kdiffusion(FakeTorchHijack(rng)), _stub_modules({"modules.rng": _forge_rng_module()}):
            run_hooks(p, mode=mode, manual_sigmas="0.6", transform=transform)
            x = rng.next()                                   # processing.py: x = self.rng.next()
            out, _, _ = sample(p, x, sigmas)
        self.assertIn("applied", status(p))
        return out

    def test_batch_image_equals_the_same_seed_alone(self):
        seeds = [11, 12, 13]
        for mode in ("transition", "respace"):
            for transform in ("dct", "fft"):
                with self.subTest(mode=mode, transform=transform):
                    batch = self._generate(seeds, AncestralEuler(), mode=mode, transform=transform)
                    for b, seed in enumerate(seeds):
                        single = self._generate([seed], AncestralEuler(), mode=mode, transform=transform)
                        torch.testing.assert_close(batch[b:b + 1], single, atol=1e-6, rtol=1e-6)

    def test_coarse_noise_comes_from_the_image_generators(self):
        sigmas = torch.linspace(1.0, 0.0, 13)
        rng = FakeImageRNG((4, 16, 16), [5])
        p = make_p(AncestralEuler(), sigmas=sigmas, seeds=[5], rng=rng)
        settings = fh.coerce_settings(ui_args())
        host = fh.ForgeSpeedHost(p, None, sigmas, settings, print)
        expected = torch.randn((4, 8, 8), generator=torch.Generator().manual_seed(5))
        with _stub_modules({"modules.rng": _forge_rng_module()}):
            got = host._coarse_randn(torch.zeros(1, 4, 8, 8))
        torch.testing.assert_close(got[0], expected, rtol=0, atol=0)

    def test_coarse_noise_without_forge_generators_is_still_per_seed(self):
        sigmas = torch.linspace(1.0, 0.0, 13)
        settings = fh.coerce_settings(ui_args())
        batch = fh.ForgeSpeedHost(make_p(None, sigmas=sigmas, seeds=[7, 8]), None, sigmas, settings, print)
        single = fh.ForgeSpeedHost(make_p(None, sigmas=sigmas, seeds=[8]), None, sigmas, settings, print)
        a = batch._coarse_randn(torch.zeros(2, 4, 8, 8))
        b = single._coarse_randn(torch.zeros(1, 4, 8, 8))
        torch.testing.assert_close(a[1:], b, rtol=0, atol=0)

    def test_coarse_brownian_sampler_per_seed(self):
        made = []

        class BrownianTreeNoiseSampler:
            def __init__(self, x, sigma_min, sigma_max, seed=None, transform=None, cpu=False):
                made.append((tuple(x.shape), float(sigma_min), float(sigma_max), list(seed)))

        whole = torch.linspace(1.0, 0.0, 11)
        p = make_p(None, sigmas=whole, seeds=[3, 4])
        host = fh.ForgeSpeedHost(p, None, whole, fh.coerce_settings(ui_args()), print)
        template = BrownianTreeNoiseSampler(torch.zeros(2, 4, 16, 16), 0.1, 1.0, seed=[3, 4])
        host.coarse_noise_sampler(template, torch.zeros(2, 4, 8, 8), 0, whole)
        host.coarse_noise_sampler(template, torch.zeros(2, 4, 12, 12), 1, whole)
        stride = fh.BROWNIAN_STAGE_SEED_STRIDE
        self.assertEqual(made[1:], [
            ((2, 4, 8, 8), float(whole[-2]), 1.0, [3 + stride, 4 + stride]),
            ((2, 4, 12, 12), float(whole[-2]), 1.0, [3 + 2 * stride, 4 + 2 * stride]),
        ])


def _load_forge_kdiffusion():
    root = FORGE_ROOT / "modules_forge" / "packages" / "k_diffusion"
    if not (root / "sampling.py").is_file():
        raise unittest.SkipTest("Forge k_diffusion package not found")
    package = types.ModuleType("k_diffusion")
    package.__path__ = [str(root)]
    backend = types.ModuleType("backend")
    backend.__path__ = []
    patcher = types.ModuleType("backend.patcher")
    patcher.__path__ = []
    base = types.ModuleType("backend.patcher.base")
    base.set_model_options_post_cfg_function = lambda options, fn, disable_cfg1_optimization=False: options
    stubs = {"k_diffusion": package, "backend": backend, "backend.patcher": patcher, "backend.patcher.base": base}
    with _stub_modules({**stubs, "k_diffusion.utils": None, "k_diffusion.sampling": None}):
        sys.modules.pop("k_diffusion.utils", None)
        sys.modules.pop("k_diffusion.sampling", None)
        spec = importlib.util.spec_from_file_location("k_diffusion.sampling", root / "sampling.py")
        sampling = importlib.util.module_from_spec(spec)
        sys.modules["k_diffusion.sampling"] = sampling
        assert spec.loader is not None
        spec.loader.exec_module(sampling)
    return package, sampling


class ForgeSamplerBatchTests(ScriptTestCase):
    """Forge's own k-diffusion samplers (modules_forge/packages/k_diffusion) under SPEED."""

    @classmethod
    def setUpClass(cls):
        cls.package, cls.kds = _load_forge_kdiffusion()

    def _generate(self, seeds, fn_name, *, mode, brownian=False, steps=12, shape=(4, 1, 16, 16)):
        kds = self.kds
        sigmas = torch.tensor([3.0 * t / (1.0 + 2.0 * t) for t in torch.linspace(1.0, 0.0, steps + 1).tolist()])
        rng = FakeImageRNG(shape, seeds)
        p = make_p(getattr(kds, fn_name), sigmas=sigmas, seeds=seeds, rng=rng)
        previous = kds.torch
        kds.torch = FakeTorchHijack(rng)               # Sampler.initialize: k_diffusion.sampling.torch = TorchHijack(p)
        try:
            with _stub_modules({"k_diffusion": self.package, "k_diffusion.sampling": kds,
                                "modules.rng": _forge_rng_module()}):
                run_hooks(p, mode=mode, manual_sigmas="0.65")
                x = rng.next()
                kwargs = {}
                if brownian:   # Sampler.create_noise_sampler: BrownianTreeNoiseSampler(x, min, max, seed=seeds)
                    kwargs["noise_sampler"] = kds.BrownianTreeNoiseSampler(x, sigmas[sigmas > 0].min(), sigmas.max(),
                                                                            seed=list(seeds))
                out, seen, _ = sample(p, x, sigmas, **kwargs)
        finally:
            kds.torch = previous
        self.assertIn("applied", status(p), status(p))
        self.assertEqual(seen, list(range(steps)))
        self.assertTrue(torch.isfinite(out).all())
        return out

    def test_samplers_batch_equals_single(self):
        for fn_name, brownian in (("sample_euler", False), ("sample_euler_ancestral", False),
                                  ("sample_er_sde", False), ("sample_dpmpp_2m", False),
                                  ("sample_dpmpp_2m_sde", True), ("sample_dpmpp_sde", True)):
            for mode in ("transition", "respace"):
                with self.subTest(sampler=fn_name, mode=mode):
                    seeds = [21, 22]
                    batch = self._generate(seeds, fn_name, mode=mode, brownian=brownian)
                    for b, seed in enumerate(seeds):
                        single = self._generate([seed], fn_name, mode=mode, brownian=brownian)
                        torch.testing.assert_close(batch[b:b + 1], single, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------------------------------------
# Infotext, paste, XYZ, settings, UI
# ------------------------------------------------------------------------------------------------


class InfotextAndUiTests(unittest.TestCase):
    def _ui(self, is_img2img=False):
        script = SCRIPT_MODULE.AnimaSpeed()
        with gr.Blocks():
            components = script.ui(is_img2img)
        return script, components

    def test_ui_order_defaults_and_ids(self):
        _, components = self._ui()
        self.assertEqual(len(components), len(fh.ARG_NAMES))
        values = [c.value for c in components]
        self.assertEqual(values, [fh.DEFAULTS[name] for name in fh.ARG_NAMES])
        ids = [c.elem_id for c in components]
        self.assertEqual(ids[0], "anima_speed_enable")
        self.assertIn("sam3-on", components[0].elem_classes)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(components[1].choices[0][0] if isinstance(components[1].choices[0], tuple) else components[1].choices[0], "transition")

    def test_defaults_are_off_and_the_official_mode(self):
        settings = fh.coerce_settings([fh.DEFAULTS[name] for name in fh.ARG_NAMES])
        self.assertFalse(settings.enabled)
        self.assertEqual((settings.mode, settings.threshold, settings.preset), ("transition", "neo_shift", "anima"))
        self.assertEqual(settings.scales, (0.5, 1.0))
        self.assertFalse(settings.hires)
        self.assertIsNone(settings.error)

    def test_paste_round_trip(self):
        script, components = self._ui()
        for overrides in (
            {},
            dict(mode="respace", threshold="manual", manual_sigmas="0.7", scales="0.5,1.0"),
            dict(preset="custom", spectrum_A=123.5, spectrum_beta=2.25, transform="fft", seed=77, hires=True),
            dict(threshold="delta_optimal", delta=0.02, adaptive=False, scales="0.25,0.5,1.0"),
        ):
            with self.subTest(overrides=overrides):
                settings = fh.coerce_settings(ui_args(**overrides))
                params = {fh.KEY: settings.summary()}
                pasted = {}
                for component, read in script.infotext_fields:
                    value = read(params)
                    if value is not None:
                        pasted[components.index(component)] = value
                args = ui_args(**overrides)
                for index, value in pasted.items():
                    args[index] = value
                self.assertEqual(fh.coerce_settings(args), settings)
        self.assertIs(dict(script.infotext_fields)[components[0]]({}), False)   # no key -> SPEED off

    def test_api_arguments(self):
        short = fh.coerce_settings([True, "respace"])                 # missing tail = UI defaults
        self.assertEqual((short.enabled, short.mode, short.threshold, short.preset), (True, "respace", "neo_shift", "anima"))
        named = fh.coerce_settings([{"enabled": True, "mode": "respace", "threshold": "manual", "manual": [0.7],
                                     "scales": [0.5, 1.0], "divisor": 1.1, "A": 1.0, "beta": 2.0}, False, "ignored"])
        self.assertTrue(named.enabled)
        self.assertEqual((named.mode, named.threshold, named.manual_sigmas, named.scales), ("respace", "manual", (0.7,), (0.5, 1.0)))
        self.assertEqual((named.sigma_divisor, named.spectrum_A, named.spectrum_beta), (1.1, 1.0, 2.0))
        self.assertIsNone(named.error)
        self.assertFalse(fh.coerce_settings([{"mode": "respace"}]).enabled)
        self.assertEqual(fh.coerce_settings([{"seed": "12"}]).seed, 12)

    def test_summary_format(self):
        settings = fh.coerce_settings(ui_args())
        self.assertEqual(
            settings.summary(),
            "mode=transition; threshold=manual; preset=anima; scales=0.5,1; delta=0.01; divisor=1.03; "
            "manual=0.72; adaptive=True; transform=dct; seed=-1; hires=False",
        )
        self.assertEqual(fh.parse_summary(settings.summary())["scales"], "0.5,1")

    def test_xyz_axes(self):
        class AxisOption:
            def __init__(self, label, type, apply, format_value=None, confirm=None, cost=0.0, choices=None, prepare=None):
                self.label, self.type, self.apply, self.choices = label, type, apply, choices

        xyz_module = SimpleNamespace(AxisOption=AxisOption, axis_options=[])
        MODULES_STUB.scripts.scripts_data[:] = [SimpleNamespace(script_class=type("X", (), {"__module__": "xyz_grid.py"}),
                                                                module=xyz_module)]
        try:
            for callback in REGISTRY.before_ui:
                callback()
            for callback in REGISTRY.before_ui:      # idempotent
                callback()
        finally:
            MODULES_STUB.scripts.scripts_data[:] = []
        labels = [a.label for a in xyz_module.axis_options]
        self.assertEqual(labels, [
            "[Anima SPEED] Enable", "[Anima SPEED] Mode", "[Anima SPEED] Manual sigma", "[Anima SPEED] Delta",
            "[Anima SPEED] Sigma divisor", "[Anima SPEED] Scale",
        ])
        p = SimpleNamespace()
        axes = {a.label.split("] ")[1]: a for a in xyz_module.axis_options}
        axes["Enable"].apply(p, "True", None)
        axes["Mode"].apply(p, "respace", None)
        axes["Manual sigma"].apply(p, 0.55, None)
        axes["Scale"].apply(p, 0.25, None)
        settings = fh.coerce_settings(ui_args(enabled=False, threshold="neo_shift"), p._anima_speed_xyz)
        self.assertTrue(settings.enabled)
        self.assertEqual((settings.mode, settings.threshold, settings.manual_sigmas, settings.scales),
                         ("respace", "manual", (0.55,), (0.25, 1.0)))
        q = SimpleNamespace()
        axes["Delta"].apply(q, 0.03, None)
        self.assertEqual(fh.coerce_settings(ui_args(), q._anima_speed_xyz).threshold, "neo_shift")
        r = SimpleNamespace()
        axes["Sigma divisor"].apply(r, 1.1, None)
        settings = fh.coerce_settings(ui_args(threshold="delta_optimal"), r._anima_speed_xyz)
        self.assertEqual((settings.threshold, settings.sigma_divisor), ("neo_shift", 1.1))

    def test_settings_registration(self):
        added = {}

        class OptionInfo:
            def __init__(self, default, label, component=None, component_args=None, section=None, infotext=None, **kw):
                self.default, self.label, self.section, self.infotext = default, label, section, infotext

            def info(self, text):
                self.info_text = text
                return self

        shared = SimpleNamespace(opts=SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info)),
                                 OptionInfo=OptionInfo)
        previous = SCRIPT_MODULE._shared
        SCRIPT_MODULE._shared = shared
        try:
            for callback in REGISTRY.settings:
                callback()
        finally:
            SCRIPT_MODULE._shared = previous
        self.assertEqual(set(added), {"sam3_speed_log", "sam3_speed_img2img_rescale"})
        self.assertTrue(all(info.default is True for info in added.values()))
        self.assertEqual({info.section for info in added.values()}, {("sam3_speed", "SAM Extra SPEED")})
        self.assertEqual(added["sam3_speed_img2img_rescale"].infotext, "Anima SPEED img2img rescale")
        self.assertIsNone(added["sam3_speed_log"].infotext)

    def test_title_and_lane(self):
        from sam3ext import layout_lanes

        self.assertEqual(SCRIPT_MODULE.AnimaSpeed().title(), "Anima SPEED")
        self.assertEqual(layout_lanes.slot_key(str(SCRIPT)), "anima-speed")
        self.assertEqual(layout_lanes.lane_for("anima-speed"), "anima")

    def test_batch_iterations_and_hires_status(self):
        with fake_backend_args():
            sigmas = torch.linspace(1.0, 0.0, 11)
            p = make_p(RecordingSampler(), sigmas=sigmas)
            run_hooks(p)
            sample(p, torch.randn(1, 4, 16, 16), sigmas)
            p.is_hr_pass = True
            p.sampler = KDiffusionSampler(RecordingSampler())
            run_hooks(p, hires=False)
            self.assertTrue(status(p).startswith("base: applied"))
            self.assertIn(" | hires: not applied", status(p))
            p.is_hr_pass = False                                  # next batch iteration
            p.sampler = KDiffusionSampler(RecordingSampler())
            run_hooks(p)
            self.assertEqual(status(p), "base: pending - sampler wrapped, waiting for sampling")


if __name__ == "__main__":
    unittest.main()
