"""Shared harness of the cross-feature tests (``tests/test_integration_*.py``); not a test module.

The features were written and tested one at a time; these tests run them together, on the CPU,
through the same hooks Forge calls. One generation pass is assembled from

* **Forge's own code** (read-only, from the checkout this extension lives in, like the other tests):
  ``backend/sampling/sampling_function.py`` ``sampling_function_inner`` (CFG and the post-CFG list),
  ``backend/utils.py`` ``join_dicts`` (how ``sampling_function`` merges the sampler's
  ``model_options`` into the UNet's) and ``deepcopy_`` (``ModelPatcher.clone``), the Anima DiT
  ``backend/nn/anima.py`` (small, random weights), ``modules_forge/packages/k_diffusion/sampling.py``
  (the samplers and the helpers the extra samplers call), the latent formats Colorcraft keys on and
  ``setup_img2img_steps``;
* **the extension's scripts**, each through its Forge hook and in Forge's load order (alphabetical in
  ``scripts/``): Anima Optimal Scale → Anima Detail Daemon → the guidance stack (``anima_safe_pag``)
  → Anima SPEED → Colorcraft, then ``sampler.func`` (SPEED's wrapper when it attached);
* host stand-ins for what Forge would run around them: ``Denoiser`` is ``CFGDenoiser.forward``
  (``on_cfg_denoiser`` callbacks — Detail Daemon's, then the stack's two — ``sampling_function``,
  ``store_latent``, ``step += 1``), ``sample`` is ``KDiffusionSampler.sample`` after the hooks
  (``initialize`` puts the per-image ``TorchHijack`` on ``k_diffusion.sampling``, then
  ``sampling_sigmas`` is published and ``func`` is called).
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.colorcraft import hook as cc_hook  # noqa: E402
from sam3ext.colorcraft import spec as cc_spec  # noqa: E402
from sam3ext.extra_samplers import registry as sampler_registry  # noqa: E402
from sam3ext.speed import forge_host as fh  # noqa: E402
from tests import _colorcraft_support as ccs  # noqa: E402
from tests import _extra_samplers_fixtures as esf  # noqa: E402
from tests.test_anima_safe_pag import (  # noqa: E402
    _load_forge_sampler,
    _load_pag_module,
    _load_real_forge_anima,
    _stub_modules,
)
from tests.test_anima_speed_script import (  # noqa: E402
    SCRIPT_MODULE as SPEED_SCRIPT,
    FakeImageRNG,
    FakeTorchHijack,
    _forge_rng_module,
    fake_backend_args,
)
from tests.test_skimmed_cfg import _load_skim_module  # noqa: E402
from tests.test_speed_guidance_stack import _load_detail_daemon  # noqa: E402

_REQUIRED = (
    "backend/sampling/condition.py", "backend/sampling/sampling_function.py", "backend/nn/anima.py",
    "backend/utils.py", "modules_forge/packages/k_diffusion/sampling.py",
    "modules_forge/packages/huggingface_guess/latent.py", "modules/sd_samplers_common.py",
)


def forge_available() -> bool:
    return all((FORGE / rel).is_file() for rel in _REQUIRED)


def require_forge(test=None):
    import unittest

    if not forge_available():
        raise unittest.SkipTest("the Forge checkout is not next to this extension (extensions/<ext>)")


# ------------------------------------------------------------------------------------------------
# Forge pieces (read-only)
# ------------------------------------------------------------------------------------------------

_CACHE: dict = {}


def forge_utils():
    """``join_dicts`` and ``deepcopy_`` from Forge's backend/utils.py (their bodies, nothing else)."""
    if "utils" not in _CACHE:
        namespace: dict = {}
        for name in ("join_dicts", "deepcopy_"):
            source = esf.forge_definition("backend/utils.py", name)
            exec(compile(source, "backend/utils.py", "exec"), namespace)  # noqa: S102
        _CACHE["utils"] = types.SimpleNamespace(join_dicts=namespace["join_dicts"], deepcopy_=namespace["deepcopy_"])
    return _CACHE["utils"]


def forge_sampler():
    """``(sampling_function module, Condition)`` — Forge's batching and CFG with host services stubbed."""
    if "sampler" not in _CACHE:
        module, condition, _memory = _load_forge_sampler()
        _CACHE["sampler"] = (module, condition)
    return _CACHE["sampler"]


def forge_k():
    """Forge Neo's k-diffusion ``sampling`` module (cached; tests put its ``torch`` back)."""
    return esf.forge_k_sampling()


def forge_modules_stub():
    """``modules`` with Forge's real ``setup_img2img_steps`` (img2img/hires offsets) and a recording
    ``store_latent`` (the Dy samplers' ``after_substep``)."""
    stub = ccs.forge_modules()
    stub.__path__ = []
    stored = []
    stub.sd_samplers_common.store_latent = stored.append
    stub.sd_samplers_common.stored = stored
    return stub


def wan21_latent_format():
    """Anima's latent format (Wan 2.1 VAE): Colorcraft's ``krea2`` basis family."""
    return ccs.latent_formats().Wan21()


# ------------------------------------------------------------------------------------------------
# Host stand-ins
# ------------------------------------------------------------------------------------------------


class FlowPredictor(esf.FlowSampling):
    """Forge ``PredictionDiscreteFlow`` stand-in: ``const`` with Anima's shift 3 ``percent_to_sigma``."""


class ForgeUnet:
    """Forge ``UnetPatcher``: ``clone`` deep-copies ``model_options`` with Forge's ``deepcopy_``
    (backend/patcher/base.py ``clone``); the ``set_model_*`` helpers are Forge's."""

    def __init__(self, dit, model_options=None, predictor=None):
        self.model = types.SimpleNamespace(diffusion_model=dit, predictor=predictor or FlowPredictor())
        self.model_options = model_options if model_options is not None else {"transformer_options": {}}
        self.controlnet_linked_list = None
        self.extra_concat_condition = None
        self.clones = 0

    def clone(self):
        self.clones += 1
        n = ForgeUnet(self.model.diffusion_model, forge_utils().deepcopy_(self.model_options), self.model.predictor)
        n.controlnet_linked_list = self.controlnet_linked_list
        n.extra_concat_condition = self.extra_concat_condition
        return n

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def set_model_sampler_pre_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_pre_cfg_function"] = self.model_options.get("sampler_pre_cfg_function", []) + [fn]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_post_cfg_function"] = self.model_options.get("sampler_post_cfg_function", []) + [fn]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True


class KModel:
    """Forge's ``KModel.apply_model`` for a const predictor: ``x0 = x - sigma * v`` (Anima)."""

    def __init__(self, dit, predictor, shapes):
        self.dit = dit
        self.predictor = predictor
        self.shapes = shapes

    @staticmethod
    def memory_required(shape):
        return 0

    def apply_model(self, x, t, c_crossattn=None, transformer_options=None, **kwargs):
        self.shapes.append(tuple(x.shape))
        with torch.no_grad():
            out = self.dit(x, t, context=c_crossattn, transformer_options=transformer_options or {})
        return x - out * t.reshape(t.shape + (1,) * (x.ndim - 1)).to(x.dtype)


class KDiffusionSampler:
    """Named like Forge's class (SPEED and the guidance stack check the name); ``func`` and
    ``model_wrap_cfg`` like it."""

    def __init__(self, func, model_wrap_cfg=None):
        self.func = func
        self.model_wrap_cfg = model_wrap_cfg


@dataclass
class Call:
    """One model call as the harness saw it."""

    shape: tuple
    sigma: float            # what the sampler asked for
    sigma_model: float      # after the on_cfg_denoiser callbacks (Detail Daemon)
    marker: object          # transformer_options["sam_extra_substep"]
    sampling_sigmas: object  # the list published at that moment
    out: torch.Tensor | None = None


class Denoiser:
    """Forge ``CFGDenoiser.forward`` (modules/sd_samplers_cfg_denoiser.py) for one request."""

    classic_ddim_eps_estimation = False

    def __init__(self, harness, request):
        self.h = harness
        self.p = request
        self.steps = int(request.steps)
        self.total_steps = int(request.steps)
        self.step = 0
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.inner_model = types.SimpleNamespace(predictor=harness.predictor)
        self.calls: list[Call] = []

    def __call__(self, x, sigma, **extra):
        h, request = self.h, self.p
        params = types.SimpleNamespace(
            x=x, image_cond=None, sigma=sigma, sampling_step=self.step, total_sampling_steps=self.steps,
            text_cond=None, text_uncond=None, denoiser=self,
        )
        before = float(sigma.flatten()[0])
        for callback in h.cfg_denoiser_callbacks():
            callback(params)
        unet = request.sd_model.forge_objects.unet
        extra_options = extra.get("model_options") or {}
        model_options = forge_utils().join_dicts(unet.model_options, extra_options)
        transformer = model_options.get("transformer_options") or {}
        cond = [{"model_conds": {"c_crossattn": h.Condition(h.ctx_cond)}, "strength": 1.0}]
        uncond = [{"model_conds": {"c_crossattn": h.Condition(h.ctx_uncond)}}]
        cond_scale = request.hr_cfg if getattr(request, "is_hr_pass", False) else request.cfg_scale
        with mock.patch.dict(sys.modules, {"backend.sampling.sampling_function": h.sampling}):
            out = h.sampling.sampling_function_inner(
                KModel(h.dit, h.predictor, h.shapes), params.x, params.sigma, uncond, cond, cond_scale,
                model_options,
            )
        self.calls.append(Call(
            shape=tuple(x.shape), sigma=before, sigma_model=float(params.sigma.flatten()[0]),
            marker=transformer.get("sam_extra_substep"), sampling_sigmas=transformer.get("sampling_sigmas"),
        ))
        self.step += 1
        return out


# ------------------------------------------------------------------------------------------------
# Scripts
# ------------------------------------------------------------------------------------------------


def _load_optimal_scale():
    modules = types.ModuleType("modules")
    modules.__path__ = []

    class Script:
        pass

    modules.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object())
    with _stub_modules({"modules": modules, "modules.scripts": modules.scripts}):
        spec_ = importlib.util.spec_from_file_location(
            "_integration_optimal_scale", ROOT / "scripts" / "anima_cfg_optimal_scale.py")
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
    return module


def _load_colorcraft_script():
    modules = types.ModuleType("modules")
    modules.__path__ = []

    class Script:
        pass

    modules.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules.script_callbacks = types.SimpleNamespace(on_before_ui=lambda fn: None, on_ui_settings=lambda fn: None)
    with _stub_modules({"modules": modules, "modules.scripts": modules.scripts,
                        "modules.script_callbacks": modules.script_callbacks}):
        spec_ = importlib.util.spec_from_file_location("_integration_colorcraft", ROOT / "scripts" / "colorcraft.py")
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
    return module


class Scripts:
    """The extension's scripts, loaded once per test class (their module state is global)."""

    def __init__(self):
        self.pag = _load_pag_module()
        self.dd = _load_detail_daemon()
        self.skim = _load_skim_module()
        self.optimal = _load_optimal_scale()
        self.colorcraft = _load_colorcraft_script()
        self.speed = SPEED_SCRIPT
        self.pag_script = self.pag.AnimaSafePAG()
        with gr.Blocks():
            inputs = self.pag_script.ui(False)
        self.pag_defaults = [c.value for c in inputs]
        self.pag_index = {c.elem_id: i for i, c in enumerate(inputs) if c.elem_id}

    def pag_args(self, **elem_values):
        args = list(self.pag_defaults)
        for elem_id, value in elem_values.items():
            args[self.pag_index[elem_id]] = value
        return args

    def teardown(self):
        self.pag._teardown_global_patches()
        self.dd._DD.update(on=False, node=None, offset_p=None)
        self.skim._SKIM.update(on=False, steps=0, warned=False, sigmas=None)
        cc_hook.clear_anchor_cache()


DD_DEFAULT_ARGS = [True, "Custom", 0.4, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, True, True, False]


def dd_args(amount=0.4, start=0.0, end=1.0, hires=False):
    args = list(DD_DEFAULT_ARGS)
    args[2], args[3], args[4], args[13] = amount, start, end, hires
    return args


def speed_args(**overrides):
    values = dict(fh.DEFAULTS, enabled=True, threshold="manual", manual_sigmas="0.7")
    values.update(overrides)
    return [values[name] for name in fh.ARG_NAMES]


def colorcraft_args(*mods, masking=False, leaves=None, combos=None, debug=False, debug_step=None):
    """Colorcraft's 579 positional arguments for ``mods`` (``_colorcraft_support.Mod``, tab I, II, …)."""
    config = ccs.scenario_config(ccs.Scenario("integration", list(mods), leaves or {}, combos or {}, masking))
    if debug:
        config.debug = True
        if debug_step is not None:
            config.debug_step = debug_step
    return cc_spec.config_to_args(config)


Mod = ccs.Mod


# ------------------------------------------------------------------------------------------------
# The generation
# ------------------------------------------------------------------------------------------------


def flow_sigmas(steps: int, shift: float = 3.0) -> torch.Tensor:
    """Anima-like float32 flow schedule ``3t / (1 + 2t)`` from 1 to 0."""
    return ccs.flow_sigmas(steps, shift)


@dataclass
class Pass:
    """What one sampling pass produced."""

    out: torch.Tensor
    calls: list
    shapes: list
    seen: list
    sigmas: torch.Tensor
    request: object
    denoiser: object
    post_cfg: list = field(default_factory=list)   # the post-CFG list sampling ran with


class Harness:
    """One request through the scripts and one sampler run (see the module docstring)."""

    def __init__(self, scripts: Scripts, *, channels=16, seed=1234, dit=None):
        require_forge()
        self.scripts = scripts
        self.sampling, self.Condition = forge_sampler()
        self.record: list = []
        if dit is None:
            anima = _load_real_forge_anima(self.record)
            torch.manual_seed(seed)
            dit = anima.Anima(
                in_channels=channels, out_channels=channels, patch_spatial=2, patch_temporal=1,
                model_channels=48, crossattn_emb_channels=32, adaln_lora_dim=16, num_blocks=6, num_heads=4,
            ).eval()
        self.dit = dit
        self.channels = channels
        self.predictor = FlowPredictor(3.0)
        g = torch.Generator().manual_seed(5)
        self.ctx_cond = torch.randn(1, 7, 32, generator=g)
        self.ctx_uncond = torch.randn(1, 7, 32, generator=g)
        self.shapes: list = []

    def cfg_denoiser_callbacks(self):
        s = self.scripts
        return (s.dd._denoiser_callback, s.pag._dave_run_callback, s.pag._hiflow_run_callback)

    # -- the request -------------------------------------------------------------------------
    def request(self, *, seeds=(31,), size=16, steps=10, cfg=4.0, img2img=False, vae=None):
        model = type("Anima", (), {"is_inpaint": False})()
        model.model_config = types.SimpleNamespace(latent_format=wan21_latent_format())
        model.forge_objects = types.SimpleNamespace(
            unet=ForgeUnet(self.dit, predictor=self.predictor),
            vae=vae if vae is not None else ccs.FakeVAE(channels=self.channels, latent_dim=3),
        )
        cls = type("StableDiffusionProcessingImg2Img" if img2img else "StableDiffusionProcessingTxt2Img", (), {})
        request = cls()
        request.sd_model = model
        request.extra_generation_params = {}
        request.steps = steps
        request.seeds = list(seeds)
        request.is_hr_pass = False
        request.enable_hr = False
        request.mask = None
        request.image_mask = None
        request.cfg_scale = cfg
        request.hr_cfg = cfg
        request.denoising_strength = 0.6
        request.hr_second_pass_steps = 0
        request.sampler_name = "Euler"
        request.rng = FakeImageRNG((self.channels, 1, size, size), seeds)
        request.sampler = KDiffusionSampler(None)
        return request

    # -- hooks ---------------------------------------------------------------------------------
    def run_hooks(self, request, *, x=None, noise=None, optimal=None, dd=None, pag=None, skim=None, speed=None,
                  colorcraft=None):
        """``process_before_every_sampling`` of each script given arguments, in Forge's load order
        (scripts/ sorted: anima_cfg_optimal_scale, anima_detail_daemon, anima_safe_pag, anima_skimmed_cfg,
        anima_speed, colorcraft), with the keyword arguments processing.py passes (``x``, ``noise``)."""
        s = self.scripts
        kw = {"x": x, "noise": x if noise is None else noise, "c": None, "uc": None}
        if optimal is not None:
            s.optimal.AnimaOptimalScale().process_before_every_sampling(request, *optimal, **kw)
        if dd is not None:
            with mock.patch.object(s.dd, "_log"):
                s.dd.AnimaDetailDaemon().process_before_every_sampling(request, *dd, **kw)
        if pag is not None:
            s.pag_script.process_before_every_sampling(request, *pag, **kw)
        if skim is not None:
            with mock.patch.object(s.skim, "_log"):
                s.skim.AnimaSkimmedCFG().process_before_every_sampling(request, *skim, **kw)
        if speed is not None:
            s.speed.AnimaSpeed().process_before_every_sampling(request, *speed, **kw)
        if colorcraft is not None:
            with mock.patch.object(cc_hook, "_log"):
                s.colorcraft.Colorcraft().process_before_every_sampling(request, *colorcraft, **kw)

    # -- sampling ------------------------------------------------------------------------------
    @contextlib.contextmanager
    def forge_environment(self, request):
        """``backend.args``, ``modules.rng`` / ``sd_samplers_common`` and Forge's k-diffusion module
        with the request's ``TorchHijack`` (``Sampler.initialize``), plus the two Forge modules extra samplers
        resolve when they run: ``k_diffusion.deis`` (DEIS) and ``modules.sd_samplers_extra`` (UniPC bh2), both
        executed read-only from Forge's files."""
        ks = forge_k()
        deis, samplers_extra = esf.forge_deis(), esf.forge_sd_samplers_extra()
        previous = ks.torch
        ks.torch = FakeTorchHijack(request.rng)
        stub = forge_modules_stub()
        try:
            with fake_backend_args(), esf.installed_k_sampling(ks), _stub_modules({
                "modules": stub, "modules.sd_samplers_common": stub.sd_samplers_common,
                "modules.rng": _forge_rng_module(),
            }), esf.installed_k_diffusion_deis(deis), esf.installed_sd_samplers_extra(samplers_extra):
                yield ks
        finally:
            ks.torch = previous

    def sample(self, request, func, x, sigmas, *, whole=None, **kwargs) -> Pass:
        """``KDiffusionSampler.sample`` from the point the hooks have run: publish ``sampling_sigmas``
        (``whole``, the full list, when the pass walks a slice of it) on the pass's UNet and call
        ``func`` (or SPEED's wrapper around it)."""
        if request.sampler.func is None:
            request.sampler.func = func
        denoiser = Denoiser(self, request)
        denoiser.steps = len(sigmas) - 1                    # launch_sampling(steps or t_enc + 1)
        denoiser.total_steps = denoiser.steps
        request.sampler.model_wrap_cfg = denoiser
        unet = request.sd_model.forge_objects.unet
        unet.model_options.setdefault("transformer_options", {})["sampling_sigmas"] = (
            sigmas if whole is None else whole
        )
        self.shapes = []
        seen = []
        out = request.sampler.func(
            denoiser, x, sigmas=sigmas, extra_args={}, callback=lambda d: seen.append(d["i"]), disable=True,
            **kwargs,
        )
        post_cfg = list(unet.model_options.get("sampler_post_cfg_function", []))
        return Pass(out=out, calls=denoiser.calls, shapes=list(self.shapes), seen=seen, sigmas=sigmas,
                    request=request, denoiser=denoiser, post_cfg=post_cfg)

    def generate(self, func, *, seeds=(31,), size=16, steps=10, sigmas=None, sampler_kwargs=None,
                 hooks=None, request=None, prepare=None, brownian=False) -> Pass:
        """A base (txt2img) pass: ``x = rng.next()``, the hooks, then the sampler. ``brownian``: hand the
        sampler Forge's per-image Brownian tree (sampler option ``brownian_noise``)."""
        request = request if request is not None else self.request(seeds=seeds, size=size, steps=steps)
        request.sampler = KDiffusionSampler(func)        # Forge creates the sampler before the hooks run
        sigmas = flow_sigmas(steps) if sigmas is None else sigmas
        with self.forge_environment(request):
            if prepare is not None:
                prepare(request)
            x = request.rng.next()                         # processing.py: x = self.rng.next(), then the hooks
            self.run_hooks(request, x=x, **(hooks or {}))
            kwargs = dict(sampler_kwargs or {})
            if callable(kwargs.get("runtime")):
                kwargs.update(kwargs.pop("runtime")(request))
            if brownian:
                kwargs["noise_sampler"] = forge_brownian(request, x, sigmas)
            return self.sample(request, func, x, sigmas, **kwargs)


    def finish(self, request):
        """``postprocess`` of the scripts that have one — what ends a generation in Forge."""
        s = self.scripts
        with mock.patch.object(s.pag, "_log"), mock.patch.object(s.skim, "_log"):
            s.pag_script.postprocess(request, None)
            s.dd.AnimaDetailDaemon().postprocess(request, None)
            s.skim.AnimaSkimmedCFG().postprocess(request, None)
            s.colorcraft.Colorcraft().postprocess(request, None)

    def hires(self, request, func, base_out, *, size=24, steps=8, denoise=0.6, hooks=None, sampler_kwargs=None,
              brownian=False) -> Pass:
        """The hires pass of ``request`` (processing.py ``sample_hr_pass`` → ``sample_img2img``): a new
        sampler and ``forge_objects`` reset before the hooks, the upscaled base latent noised to the
        first sigma of ``sigmas[steps - t_enc - 1:]`` (Forge's ``setup_img2img_steps``), the whole list
        published as ``sampling_sigmas``."""
        request.is_hr_pass = True
        request.hr_second_pass_steps = steps
        request.denoising_strength = denoise
        request.sampler = KDiffusionSampler(func)
        request.sd_model.forge_objects.unet = ForgeUnet(self.dit, predictor=self.predictor)
        request.rng = FakeImageRNG((self.channels, 1, size, size), request.seeds)
        with self.forge_environment(request):
            upscaled = torch.nn.functional.interpolate(
                base_out.squeeze(2), size=(size, size), mode="bilinear").unsqueeze(2)
            noise = request.rng.next()
            self.run_hooks(request, x=upscaled, noise=noise, **(hooks or {}))
            setup = ccs.setup_img2img_steps()
            total, t_enc = setup(request, steps)
            whole = flow_sigmas(total)
            sched = whole[total - t_enc - 1:]
            x = float(sched[0]) * noise + (1.0 - float(sched[0])) * upscaled
            kwargs = dict(sampler_kwargs or {})
            if callable(kwargs.get("runtime")):
                kwargs.update(kwargs.pop("runtime")(request))
            if brownian:
                kwargs["noise_sampler"] = forge_brownian(request, upscaled, whole)
            result = self.sample(request, func, x, sched, whole=whole, **kwargs)
        request.is_hr_pass = False
        return result


def forge_brownian(request, x, sigmas):
    """``Sampler.create_noise_sampler``: Forge's ``BrownianTreeNoiseSampler`` with one tree per image seed."""
    return forge_k().BrownianTreeNoiseSampler(x, sigmas[sigmas > 0].min(), sigmas.max(), seed=list(request.seeds))


class LinearVelocity:
    """An elementwise DiT stand-in, ``v = k * x`` (so ``x0 = (1 - k*sigma) * x``): no spatial mixing,
    linear — a uniform edit of x0 stays exactly uniform through a step unless a sampler treats some
    pixels differently."""

    def __init__(self, k=0.3):
        self.k = k

    def __call__(self, x, t, context=None, transformer_options=None, **kwargs):
        return self.k * x


def dy_runtime(label):
    """``ExtraKDiffusionSampler.initialize``'s additions for ``label`` (the registry's own code)."""
    spec = next(s for s in sampler_registry.SPECS if s.label == label)

    def runtime(request):
        return sampler_registry.runtime_kwargs(spec, request)

    return runtime


def colorcraft_status(request) -> str:
    return request.extra_generation_params.get(cc_spec.STATUS_KEY, "")


def speed_status(request) -> str:
    return request.extra_generation_params.get(fh.STATUS_KEY, "")
