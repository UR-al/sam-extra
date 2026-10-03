"""Origin-parity tests for CNS (Colored Noise Sampling) in the Anima Guidance suite.

The oracle is the upstream ComfyUI node itself, ``namemechan/comfyui-cns_sampler_patch@42278b13``
``cns_sampler_patch.py``, loaded verbatim from ``tests/_origin_cns_sampler_patch.py`` (GPL-3.0,
the same license as this extension; its SHA-256 is pinned below) with ``comfy.samplers`` /
``comfy.k_diffusion.sampling`` stubbed. The Forge side runs this script's real attach path
(``process_before_every_sampling``), its real noise-sampler patches and post-CFG hook, and a
sampler object that passes ``callback=self.callback_state`` like Forge
(modules/sd_samplers_kdiffusion.py:196, :250).

What must match upstream:

* ``color_noise_wavelet`` — every output element within 1e-6 (odd H/W, 5-D, strength < 1 with
  the plain lerp and no renormalisation, half-precision noise with an fp32 live latent).
* INPUT_TYPES — default/min/max/step of strength, gamma_power, gamma_scale (Forge sliders), the
  server-side clamps and the omitted-argument fallbacks.
* Runtime — the x_t each noise call is colored against: the initial x, then each step
  callback's ``info["x"]`` (upstream :315, :473-478, :567-572), also for a sampler that
  evaluates the model twice per step (dpmpp_2s_ancestral-like) and for Brownian noise.
* Scope — only the run of the SAMPLER the node wraps is colored (upstream :544-608); a nested
  sampler run of a p without this script (ADetailer's filtered img2img) keeps white noise.

Kept host differences (parity plan §5.4): the base noise is Forge's (seeded TorchHijack /
Brownian), the patch is a global install scoped to the attached sampler's ``launch_sampling``
instead of a SAMPLER wrapper, a coloring failure is logged once per generation instead of
every step.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.guidance import ui_config_migration as migration  # noqa: E402
from sam3ext.guidance.cns import color_noise_wavelet  # noqa: E402

ORIGIN_FILE = ROOT / "tests" / "_origin_cns_sampler_patch.py"
ORIGIN_MARKER = "# ---- upstream cns_sampler_patch.py below (verbatim) ----\n"
# SHA-256 of namemechan/comfyui-cns_sampler_patch@42278b138284f7a8685ef174af0a50fe03246dd0:
# cns_sampler_patch.py with CRLF normalised to LF (the raw upstream file is 30e17ee2…dfac2).
ORIGIN_SHA256 = "2eac2b16ea952cdafb1618eca0e8e05ef470494f2a1e2a3c7a28617ad284e352"


class _KSampler:
    """comfy.samplers.KSAMPLER as the upstream node uses it (sampler_function + options)."""

    def __init__(self, sampler_function, extra_options=None, inpaint_options=None):
        self.sampler_function = sampler_function
        self.extra_options = extra_options or {}
        self.inpaint_options = inpaint_options or {}

    def sample(self, model, x, sigmas, **kwargs):
        return self.sampler_function(model, x, sigmas, **kwargs, **self.extra_options)


def _comfy_stubs(kds=None):
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSAMPLER = _KSampler
    k_diffusion = types.ModuleType("comfy.k_diffusion")
    k_diffusion.__path__ = []
    comfy.samplers = samplers
    comfy.k_diffusion = k_diffusion
    modules = {
        "comfy": comfy, "comfy.samplers": samplers, "comfy.k_diffusion": k_diffusion,
    }
    if kds is not None:
        k_diffusion.sampling = kds
        modules["comfy.k_diffusion.sampling"] = kds
    return modules


def _load_origin():
    with mock.patch.dict(sys.modules, _comfy_stubs()):
        spec = importlib.util.spec_from_file_location("_origin_cns_sampler_patch", ORIGIN_FILE)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


def _load_test_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORIGIN = _load_origin()
_MIGRATION_TESTS = _load_test_module("_cns_origin_migration_tests", "test_ui_config_migration.py")
_BASE = _MIGRATION_TESTS._BASE

UPSTREAM_INPUTS = ORIGIN.CNSSamplerPatch.INPUT_TYPES()["required"]
PARAMS = ("strength", "gamma_power", "gamma_scale")
ELEM_IDS = {
    "enabled": "anima_guidance_cns_enable",
    "strength": "anima_guidance_cns_strength",
    "gamma_power": "anima_guidance_cns_gamma_power",
    "gamma_scale": "anima_guidance_cns_gamma_scale",
}
PREFIX = "customscript/anima_safe_pag.py"


# ---- deterministic inputs (no global RNG) -------------------------------------------

def _randn(shape, seed, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(dtype)


def _structured(shape, seed):
    """A 'latent' with a strong low-frequency image under noise, so the bands differ a lot."""
    *lead, height, width = shape
    ys = torch.linspace(-1.0, 1.0, height).view(height, 1)
    xs = torch.linspace(-1.0, 1.0, width).view(1, width)
    image = 3.0 * torch.cos(1.3 * xs) * (1.0 - ys * ys) + 1.5 * ys
    image = image.expand(*lead, height, width)
    return image + 0.2 * _randn(shape, seed)


def _max_diff(a, b):
    return float((a.float() - b.float()).abs().max())


class OriginCopyTests(unittest.TestCase):
    def test_upstream_copy_is_verbatim(self):
        text = ORIGIN_FILE.read_text(encoding="utf-8")
        self.assertIn(ORIGIN_MARKER, text)
        body = text.split(ORIGIN_MARKER, 1)[1]
        self.assertEqual(hashlib.sha256(body.encode("utf-8")).hexdigest(), ORIGIN_SHA256)


class ColorNoiseWaveletOriginTests(unittest.TestCase):
    """sam3ext/guidance/cns.py against upstream color_noise_wavelet (:169-288)."""

    def _both(self, noise, x_t, **kwargs):
        return (
            color_noise_wavelet(noise, x_t, **kwargs),
            ORIGIN.color_noise_wavelet(noise, x_t, **kwargs),
        )

    def test_strength_half_on_odd_hw_matches_upstream(self):
        # The plan's parity case: s = .5 on odd H/W (the crop-energy path, :274-282) —
        # the lerp is not renormalised afterwards (:284-288).
        for shape in ((1, 16, 7, 9), (2, 4, 5, 6), (1, 16, 13, 13)):
            noise = _randn(shape, 11)
            x_t = _structured(shape, 12)
            ours, upstream = self._both(noise, x_t, strength=0.5)
            with self.subTest(shape=shape):
                self.assertLessEqual(_max_diff(ours, upstream), 1e-6)
                # sensitive: the removed post-lerp renormalisation would restore this std
                renormalised = ours * (noise.std() / ours.std())
                self.assertGreater(_max_diff(renormalised, upstream), 1e-5)

    def test_parameter_grid_matches_upstream(self):
        shapes = ((1, 16, 8, 8), (1, 16, 7, 9), (2, 4, 9, 6), (1, 4, 2, 7, 9))
        for shape in shapes:
            noise = _randn(shape, 21)
            x_t = _structured(shape, 22)
            for strength in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
                for gamma_power in (0.1, 0.5, 2.0):
                    for gamma_scale in (0.1, 1.0, 2.0, 3.0, 25.0):
                        ours, upstream = self._both(
                            noise, x_t, strength=strength,
                            gamma_power=gamma_power, gamma_scale=gamma_scale,
                        )
                        with self.subTest(shape=shape, s=strength, gp=gamma_power, gs=gamma_scale):
                            self.assertEqual(ours.dtype, upstream.dtype)
                            self.assertLessEqual(_max_diff(ours, upstream), 1e-6)

    def test_signature_defaults_are_upstream(self):
        ours = color_noise_wavelet.__defaults__
        self.assertEqual(ours, ORIGIN.color_noise_wavelet.__defaults__)
        self.assertEqual(ours, (1.0, 0.5, 2.0))

    def test_strength_zero_returns_the_input_object(self):
        noise = _randn((1, 4, 7, 9), 31)
        x_t = _structured((1, 4, 7, 9), 32)
        self.assertIs(color_noise_wavelet(noise, x_t, strength=0.0), noise)
        self.assertIs(ORIGIN.color_noise_wavelet(noise, x_t, strength=0.0), noise)

    def test_half_precision_noise_with_fp32_live_latent(self):
        shape = (1, 16, 7, 9)
        x_t = _structured(shape, 42) * 40.0
        for dtype in (torch.float16, torch.bfloat16):
            noise = _randn(shape, 41, dtype)
            ours, upstream = self._both(noise, x_t, strength=0.5, gamma_scale=0.5)
            with self.subTest(dtype=dtype):
                self.assertEqual(ours.dtype, dtype)
                self.assertTrue(torch.equal(ours, upstream))


class _ForgeSampler:
    """What the script sees as ``p.sampler``: Forge runs ``self.func`` inside
    ``self.launch_sampling(steps, lambda: ...)`` and passes ``callback=self.callback_state``,
    both read when sampling starts (modules/sd_samplers_kdiffusion.py:194-196, :248-250)."""

    def __init__(self, func, model):
        self.func = func
        self.model = model
        self.callback_steps = []
        self.launched = 0

    def callback_state(self, d):
        self.callback_steps.append(d["i"])

    def launch_sampling(self, steps, func):
        # modules/sd_samplers_common.py:435-447 (state bookkeeping, then func())
        self.launched += 1
        return func()

    def run(self, x, sigmas, noise_sampler=None):
        kwargs = {} if noise_sampler is None else {"noise_sampler": noise_sampler}
        return self.launch_sampling(len(sigmas) - 1, lambda: self.func(
            self.model, x, sigmas, extra_args={}, disable=False,
            callback=self.callback_state, **kwargs,
        ))


class _SlottedSampler:
    """A sampler whose callback and launch_sampling cannot be wrapped (no instance
    ``__dict__``)."""

    __slots__ = ("func", "model", "callback_steps", "launched")

    __init__ = _ForgeSampler.__init__
    callback_state = _ForgeSampler.callback_state
    launch_sampling = _ForgeSampler.launch_sampling
    run = _ForgeSampler.run


SIGMAS = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
SHAPE = (1, 16, 7, 9)


def _make_kd_module(record, default_noise_sampler):
    """Ancestral ``sample_*`` stand-ins with k-diffusion's call order (model → callback →
    noise). ``default_noise_sampler`` is a module global like k-diffusion's."""
    kds = types.ModuleType("fake_kd_sampling")
    kds.default_noise_sampler = default_noise_sampler

    def sample_fake_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None,
                              eta=1.0, s_noise=1.0, noise_sampler=None):
        # euler_ancestral order (k_diffusion/sampling.py:187-209)
        extra_args = {} if extra_args is None else extra_args
        noise_sampler = kds.default_noise_sampler(x) if noise_sampler is None else noise_sampler
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            denoised = model(x, sigma, **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            if float(sigma_next) == 0.0:
                x = denoised
                continue
            noise = noise_sampler(sigma, sigma_next)
            record.append(noise.clone())
            ratio = float(sigma_next) / float(sigma)
            x = denoised + (x - denoised) * (ratio * 0.5) + noise * (float(sigma_next) * s_noise * 0.5)
        return x

    def sample_fake_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None,
                                 eta=1.0, s_noise=1.0, noise_sampler=None):
        # dpmpp_2s_ancestral order (k_diffusion/sampling.py:351-386): a second model
        # evaluation at the midpoint x_2 comes between the callback and the noise.
        extra_args = {} if extra_args is None else extra_args
        noise_sampler = kds.default_noise_sampler(x) if noise_sampler is None else noise_sampler
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            denoised = model(x, sigma, **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            if float(sigma_next) == 0.0:
                x = denoised
                continue
            x_2 = denoised + (x - denoised) * 0.6
            denoised_2 = model(x_2, sigma * 0.7, **extra_args)
            noise = noise_sampler(sigma, sigma_next)
            record.append(noise.clone())
            ratio = float(sigma_next) / float(sigma)
            x = denoised_2 + (x - denoised_2) * (ratio * 0.5) + noise * (float(sigma_next) * s_noise * 0.5)
        return x

    kds.sample_fake_ancestral = sample_fake_ancestral
    kds.sample_fake_2s_ancestral = sample_fake_2s_ancestral
    return kds


def _denoise(x, sigma):
    clean = _structured(SHAPE, 7) * 2.0
    s = float(sigma)
    return clean * (1.0 - 0.5 * s) + x * (0.25 * s)


def _base_noise_sampler(seed):
    """The host's base noise (Forge: seeded / Brownian; upstream: the one it is handed)."""
    counter = [0]

    def sample(sigma, sigma_next):
        counter[0] += 1
        return _randn(SHAPE, seed + counter[0])

    return sample


class _Unet:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, function):
        self.model_options["model_function_wrapper"] = function

    def set_model_sampler_pre_cfg_function(self, function, disable_cfg1_optimization=False):
        self.model_options.setdefault("sampler_pre_cfg_function", []).append(function)

    def set_model_sampler_post_cfg_function(self, function, disable_cfg1_optimization=False):
        self.model_options.setdefault("sampler_post_cfg_function", []).append(function)


class _BrownianLike:
    """Forge hands samplers a BrownianTreeNoiseSampler instance; its class ``__call__`` is
    what the script patches (_patched_brownian_call)."""


class CnsForgeRuntimeTests(unittest.TestCase):
    """Which x_t each noise call is colored against, on the real attach path."""

    CASE = {"strength": 0.75, "gamma_power": 0.5, "gamma_scale": 0.5}
    SEED = 300

    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    def setUp(self):
        _BASE.AnimaSafePagTests.setUp(self)
        self.pag._CNS.update(
            capture="post_cfg", callback_seen=False, scope="pass", live=False
        )

    def _args(self, **overrides):
        with gr.Blocks():
            inputs = self.pag.AnimaSafePAG().ui(False)
        index = {c.elem_id: i for i, c in enumerate(inputs)}
        args = [component.value for component in inputs]
        args[index[ELEM_IDS["enabled"]]] = True
        for name, value in {**self.CASE, **overrides}.items():
            args[index[ELEM_IDS[name]]] = value
        return args

    def _request(self, sampler):
        model = type("Anima", (), {})()
        model.forge_objects = types.SimpleNamespace(unet=_Unet())
        return types.SimpleNamespace(
            sd_model=model, extra_generation_params={}, steps=len(SIGMAS) - 1,
            cfg_scale=4.0, sampler=sampler,
        )

    def _upstream(self, kd_name):
        record = []
        kds = _make_kd_module(record, lambda x: (lambda s, sn: torch.zeros_like(x)))
        with mock.patch.dict(sys.modules, _comfy_stubs(kds)):
            (patched,) = ORIGIN.CNSSamplerPatch().patch(
                _KSampler(getattr(kds, kd_name)), **self.CASE
            )
            final = patched.sample(
                lambda x, sigma, **_: _denoise(x, sigma), _randn(SHAPE, 100), SIGMAS,
                extra_args={}, callback=None, disable=True,
                noise_sampler=_base_noise_sampler(self.SEED),
            )
        return record, final

    def _forge(self, kd_name, *, sampler_cls=_ForgeSampler, brownian=False):
        pag = self.pag
        record = []
        kds = _make_kd_module(record, pag._patched_default_noise_sampler)
        hooks = {}

        def model(x, sigma, **_extra):
            # Forge's sampling_function hands the post-CFG hook the evaluated x as "input".
            (post_cfg,) = hooks["unet"].model_options["sampler_post_cfg_function"]
            return post_cfg({"denoised": _denoise(x, sigma), "input": x, "sigma": sigma})

        sampler = sampler_cls(getattr(kds, kd_name), model) if sampler_cls else None
        request = self._request(sampler)
        base_seed = self.SEED
        brownian_base = _base_noise_sampler(base_seed)

        def base_factory(x):
            return _base_noise_sampler(base_seed)

        def base_brownian_call(_self, sigma, sigma_next):
            return brownian_base(sigma, sigma_next)

        brownian_cls = type("BrownianTreeNoiseSampler", (_BrownianLike,), {
            "__call__": pag._patched_brownian_call,
        })
        with mock.patch.object(pag, "_ensure_cns_noise_patched", return_value=True), \
                mock.patch.object(pag, "_ORIG_CNS_DEFAULT_FACTORY", base_factory), \
                mock.patch.object(pag, "_ORIG_CNS_BROWNIAN_CALL", base_brownian_call), \
                mock.patch.object(pag, "_log"):
            pag.AnimaSafePAG().process_before_every_sampling(request, *self._args())
            hooks["unet"] = request.sd_model.forge_objects.unet
            x0 = _randn(SHAPE, 100)
            noise_sampler = brownian_cls() if brownian else None
            if sampler is not None:
                final = request.sampler.run(x0, SIGMAS, noise_sampler)
            else:
                final = getattr(kds, kd_name)(
                    model, x0, SIGMAS, extra_args={}, callback=None,
                    **({"noise_sampler": noise_sampler} if brownian else {}),
                )
        return record, final, request

    def _assert_same_noise(self, forge, upstream):
        self.assertEqual(len(forge), len(upstream))
        self.assertGreater(len(upstream), 0)
        for step, (ours, theirs) in enumerate(zip(forge, upstream)):
            with self.subTest(step=step):
                self.assertLessEqual(_max_diff(ours, theirs), 1e-6)

    def test_callback_x_matches_upstream_for_euler_ancestral(self):
        upstream, upstream_final = self._upstream("sample_fake_ancestral")
        forge, final, request = self._forge("sample_fake_ancestral")
        self.assertEqual(self.pag._CNS["capture"], "callback")
        self._assert_same_noise(forge, upstream)
        self.assertLessEqual(_max_diff(final, upstream_final), 1e-6)
        self.assertEqual(request.sampler.callback_steps, list(range(len(SIGMAS) - 1)))
        self.assertEqual(self.pag._RUNTIME.cns_noise_calls, len(upstream))

    def test_second_model_evaluation_does_not_replace_the_step_x(self):
        """dpmpp_2s_ancestral / dpmpp_sde evaluate the midpoint before the noise: upstream
        still colors against the step's callback x, not the last post-CFG input."""
        upstream, upstream_final = self._upstream("sample_fake_2s_ancestral")
        forge, final, _request = self._forge("sample_fake_2s_ancestral")
        self._assert_same_noise(forge, upstream)
        self.assertLessEqual(_max_diff(final, upstream_final), 1e-6)

        # sensitive: the post-CFG fallback (the pre-parity source) colors against x_2.
        fallback, _final, _request = self._forge(
            "sample_fake_2s_ancestral", sampler_cls=_SlottedSampler
        )
        self.assertEqual(self.pag._CNS["capture"], "post_cfg")
        self.assertGreater(max(_max_diff(a, b) for a, b in zip(fallback, upstream)), 1e-4)

    def test_brownian_noise_is_colored_against_the_callback_x(self):
        upstream, _upstream_final = self._upstream("sample_fake_2s_ancestral")
        forge, _final, _request = self._forge("sample_fake_2s_ancestral", brownian=True)
        self._assert_same_noise(forge, upstream)

    def test_post_cfg_fallback_without_a_wrappable_callback(self):
        # One model evaluation per step: the fallback input is the step x, so it matches.
        upstream, _upstream_final = self._upstream("sample_fake_ancestral")
        for sampler_cls in (_SlottedSampler, None):
            with self.subTest(sampler=getattr(sampler_cls, "__name__", None)):
                self.setUp()
                forge, _final, _request = self._forge("sample_fake_ancestral", sampler_cls=sampler_cls)
                self.assertEqual(self.pag._CNS["capture"], "post_cfg")
                self.assertEqual(self.pag._CNS["scope"], "pass")
                self._assert_same_noise(forge, upstream)

    def test_initial_x_seeds_the_state_like_upstream(self):
        """make_cns_noise_sampler seeds x_current with the sampler's initial x (:315)."""
        pag = self.pag
        pag._CNS.update(on=True, capture="callback", callback_seen=False)
        x0 = _randn(SHAPE, 5)
        with mock.patch.object(pag, "_ORIG_CNS_DEFAULT_FACTORY", lambda x: _base_noise_sampler(9)):
            sample = pag._patched_default_noise_sampler(x0)
        self.assertTrue(torch.equal(pag._RUNTIME.cns_x_t, x0))
        self.assertIsNot(pag._RUNTIME.cns_x_t, x0)
        # a post-CFG input before the first callback does not replace the seed
        pag._capture_cns_post_cfg_input(x0 + 1.0)
        self.assertTrue(torch.equal(pag._RUNTIME.cns_x_t, x0))
        expected = ORIGIN.color_noise_wavelet(
            _base_noise_sampler(9)(None, None), x0,
            strength=pag._CNS["strength"], gamma_power=pag._CNS["gamma_power"],
            gamma_scale=pag._CNS["gamma_scale"],
        )
        self.assertTrue(torch.equal(sample(None, None), expected))

    def test_x_t_is_moved_to_the_noise_device_without_a_dtype_cast(self):
        pag = self.pag
        pag._CNS.update(on=True, strength=0.5, gamma_power=0.5, gamma_scale=0.5)
        x_t = _structured(SHAPE, 52) * 40.0
        pag._RUNTIME.cns_x_t = x_t
        noise = _randn(SHAPE, 51, torch.float16)
        out = pag._maybe_color_cns_noise(noise)
        expected = ORIGIN.color_noise_wavelet(noise, x_t, 0.5, 0.5, 0.5)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.equal(out, expected))

    def _nested_run(self, kd_name, *, brownian=False):
        """A sampler run of another p whose script list lacks this script, after this
        script's pass: ADetailer's inner img2img runs in ``postprocess_image`` (Forge
        modules/processing.py:1068) before ``postprocess`` (:1180) clears CNS, with only
        ``ad_script_names`` kept (aadetailer-neoforge scripts/!adetailer.py:618-645). Forge
        gives that p a new sampler object (modules/processing.py:1701) that
        ``process_before_every_sampling`` never saw.
        Returns (noise, white, initial x)."""
        pag = self.pag
        record = []
        kds = _make_kd_module(record, pag._patched_default_noise_sampler)
        sampler = _ForgeSampler(getattr(kds, kd_name), lambda x, sigma, **_: _denoise(x, sigma))
        seed = 777
        brownian_base = _base_noise_sampler(seed)
        brownian_cls = type("BrownianTreeNoiseSampler", (_BrownianLike,), {
            "__call__": pag._patched_brownian_call,
        })
        x0 = _randn(SHAPE, 555) * 0.7
        with mock.patch.object(pag, "_ORIG_CNS_DEFAULT_FACTORY",
                               lambda x: _base_noise_sampler(seed)),                 mock.patch.object(pag, "_ORIG_CNS_BROWNIAN_CALL",
                                  lambda _self, s, sn: brownian_base(s, sn)),                 mock.patch.object(pag, "_log"):
            sampler.run(x0, SIGMAS, brownian_cls() if brownian else None)
        reference = _base_noise_sampler(seed)
        white = [reference(None, None) for _ in record]
        return record, white, x0

    def test_nested_run_without_the_script_keeps_white_noise(self):
        """origin: namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:544-608
        — the node returns a new SAMPLER whose CNS noise sampler goes to that one call (path
        A) or whose kds patch lasts until a finally (path B); other sampler runs stay white."""
        pag = self.pag
        upstream, _upstream_final = self._upstream("sample_fake_ancestral")
        forge, _final, _request = self._forge("sample_fake_ancestral")
        self._assert_same_noise(forge, upstream)
        # CNS stays on until postprocess; only the attached sampler's run is live.
        self.assertTrue(pag._CNS["on"])
        self.assertEqual(pag._CNS["scope"], "sampler")
        self.assertFalse(pag._CNS["live"])
        calls = pag._RUNTIME.cns_noise_calls
        x_t = pag._RUNTIME.cns_x_t
        for kd_name, brownian in (
            ("sample_fake_ancestral", False),
            ("sample_fake_2s_ancestral", True),
        ):
            with self.subTest(kd_name=kd_name, brownian=brownian):
                noise, white, _x0 = self._nested_run(kd_name, brownian=brownian)
                self.assertEqual(len(noise), len(SIGMAS) - 2)
                for step, (ours, base) in enumerate(zip(noise, white)):
                    with self.subTest(step=step):
                        self.assertTrue(torch.equal(ours, base))
        self.assertEqual(pag._RUNTIME.cns_noise_calls, calls)
        # the nested sampler's initial x does not replace the state either
        self.assertIs(pag._RUNTIME.cns_x_t, x_t)

        # the next attached pass (hires / next batch) colors again like upstream
        forge, _final, _request = self._forge("sample_fake_ancestral")
        self._assert_same_noise(forge, upstream)

    def test_unscoped_gate_would_color_the_nested_run_against_its_frozen_x(self):
        """Sensitivity: with the pass-wide gate (CNS on only — the fallback, and the gate
        before the sampler scope) the nested run is colored against its initial x forever."""
        pag = self.pag
        self._forge("sample_fake_ancestral")
        pag._CNS["scope"] = "pass"
        noise, white, x0 = self._nested_run("sample_fake_ancestral")
        self.assertGreater(len(noise), 0)
        for step, (ours, base) in enumerate(zip(noise, white)):
            with self.subTest(step=step):
                frozen = ORIGIN.color_noise_wavelet(base, x0, **self.CASE)
                self.assertLessEqual(_max_diff(ours, frozen), 1e-6)
                self.assertGreater(_max_diff(ours, base), 1e-3)

    def test_sampler_scope_is_live_only_inside_launch_sampling(self):
        pag = self.pag
        sampler = _ForgeSampler(None, None)
        request = self._request(sampler)
        self.assertEqual(pag._install_cns_sampler_scope(request), "sampler")
        wrapper = vars(sampler)["launch_sampling"]
        self.assertEqual(pag._install_cns_sampler_scope(request), "sampler")
        self.assertIs(vars(sampler)["launch_sampling"], wrapper)

        pag._CNS.update(on=True, scope="sampler")
        self.assertFalse(pag._cns_active())
        seen = []
        self.assertEqual(wrapper(3, lambda: seen.append(pag._cns_active()) or "done"), "done")
        self.assertEqual(seen, [True])
        self.assertFalse(pag._cns_active())

        def interrupted():
            raise RuntimeError("stop")

        with self.assertRaises(RuntimeError):
            wrapper(3, interrupted)
        self.assertFalse(pag._CNS["live"])
        self.assertEqual(sampler.launched, 2)
        # nothing to wrap: the pass-wide fallback
        self.assertEqual(
            pag._install_cns_sampler_scope(self._request(_SlottedSampler(None, None))), "pass"
        )
        self.assertEqual(pag._install_cns_sampler_scope(self._request(None)), "pass")

    def test_capture_is_installed_only_while_cns_is_on_and_only_once(self):
        pag = self.pag
        kds = _make_kd_module([], pag._patched_default_noise_sampler)
        sampler = _ForgeSampler(kds.sample_fake_ancestral, None)
        request = self._request(sampler)
        with gr.Blocks():
            inputs = pag.AnimaSafePAG().ui(False)
        enabled = [c.elem_id for c in inputs].index(ELEM_IDS["enabled"])
        with mock.patch.object(pag, "_ensure_cns_noise_patched", return_value=True), \
                mock.patch.object(pag, "_log"):
            args = self._args()
            args[enabled] = False
            pag.AnimaSafePAG().process_before_every_sampling(request, *args)
            self.assertNotIn("callback_state", vars(sampler))
            self.assertNotIn("launch_sampling", vars(sampler))
            self.assertEqual(pag._CNS["scope"], "pass")

            args[enabled] = True
            pag.AnimaSafePAG().process_before_every_sampling(request, *args)
            wrapper = vars(sampler)["callback_state"]
            self.assertEqual(pag._install_cns_x_capture(request), "callback")
            self.assertIs(vars(sampler)["callback_state"], wrapper)
            self.assertIn("launch_sampling", vars(sampler))
            self.assertEqual(pag._CNS["scope"], "sampler")
        # the wrapper still runs Forge's callback (progress, stop_at) after recording
        wrapper({"x": torch.zeros(SHAPE), "i": 3})
        self.assertEqual(sampler.callback_steps, [3])
        self.assertTrue(pag._CNS["callback_seen"])


class CnsInputTypesOriginTests(unittest.TestCase):
    """Forge sliders, server clamps and fallbacks = upstream INPUT_TYPES (:391-439)."""

    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    def setUp(self):
        _BASE.AnimaSafePagTests.setUp(self)

    def _ui(self):
        with gr.Blocks():
            inputs = self.pag.AnimaSafePAG().ui(False)
        by_id = {c.elem_id: (i, c) for i, c in enumerate(inputs)}
        return inputs, by_id

    def _attach(self, args):
        model = type("Anima", (), {})()
        model.forge_objects = types.SimpleNamespace(unet=_Unet())
        request = types.SimpleNamespace(
            sd_model=model, extra_generation_params={}, steps=20, cfg_scale=4.0,
        )
        with mock.patch.object(self.pag, "_ensure_cns_noise_patched", return_value=True), \
                mock.patch.object(self.pag, "_log"):
            self.pag.AnimaSafePAG().process_before_every_sampling(request, *args)
        return request

    def test_sliders_are_the_upstream_inputs(self):
        _inputs, by_id = self._ui()
        for name in PARAMS:
            spec = UPSTREAM_INPUTS[name][1]
            _index, slider = by_id[ELEM_IDS[name]]
            with self.subTest(name=name):
                self.assertEqual(
                    (slider.value, slider.minimum, slider.maximum, slider.step),
                    (spec["default"], spec["min"], spec["max"], spec["step"]),
                )
        _index, scale = by_id[ELEM_IDS["gamma_scale"]]
        self.assertIn("기본 2.0 · Anima+cfg_pp 권장 3.0", scale.label)
        self.assertEqual(scale.label, migration.CNS_GAMMA_SCALE_LABEL)

    def test_server_clamps_to_the_upstream_bounds(self):
        inputs, by_id = self._ui()
        index = {name: by_id[ELEM_IDS[name]][0] for name in (*PARAMS, "enabled")}
        cases = [
            ({"strength": 1.7, "gamma_power": 0.05, "gamma_scale": 0.1}, (1.0, 0.1, 0.1)),
            ({"strength": -0.2, "gamma_power": 9.0, "gamma_scale": 30.0}, (0.0, 2.0, 25.0)),
            ({"strength": 0.35, "gamma_power": 0.15, "gamma_scale": 0.15}, (0.35, 0.15, 0.15)),
            # non-finite -> the upstream default
            ({"strength": float("nan"), "gamma_power": float("inf"),
              "gamma_scale": float("-inf")}, (1.0, 0.5, 2.0)),
        ]
        for values, expected in cases:
            args = [component.value for component in inputs]
            args[index["enabled"]] = True
            for name, value in values.items():
                args[index[name]] = value
            with self.subTest(values=values):
                self._attach(args)
                self.assertTrue(self.pag._CNS["on"])
                self.assertEqual(tuple(self.pag._CNS[name] for name in PARAMS), expected)

    def test_omitted_arguments_fall_back_to_the_upstream_defaults(self):
        inputs, by_id = self._ui()
        enabled = by_id[ELEM_IDS["enabled"]][0]
        args = [component.value for component in inputs[: enabled + 1]]
        args[enabled] = True
        request = self._attach(args)
        defaults = tuple(UPSTREAM_INPUTS[name][1]["default"] for name in PARAMS)
        self.assertEqual(tuple(self.pag._CNS[name] for name in PARAMS), defaults)
        self.assertEqual(
            request.extra_generation_params["Anima CNS Wavelet Noise"],
            "strength=1.0, gamma_power=0.5, gamma_scale=2.0",
        )


def _legacy_cns_tab(tab, *, strength=1.0, gamma_power=0.5, gamma_scale=3.0):
    """CNS keys the pre-parity script saved (forge_sam3_extension@3522928:
    scripts/anima_safe_pag.py:3253-3270) — the shape in this install's ui-config.json."""
    settings = {
        f"{PREFIX}/{tab}/Enable CNS-inspired Wavelet Noise/visible": True,
        f"{PREFIX}/{tab}/Enable CNS-inspired Wavelet Noise/value": False,
    }
    settings.update(_MIGRATION_TESTS._slider(tab, "CNS strength", strength, 0.0, 1.0, 0.01))
    settings.update(_MIGRATION_TESTS._slider(tab, "CNS gamma power", gamma_power, 0.05, 2.0, 0.05))
    settings.update(_MIGRATION_TESTS._slider(
        tab, migration.CNS_GAMMA_SCALE_LEGACY_LABEL, gamma_scale, 0.25, 25.0, 0.25,
    ))
    return settings


class CnsUiConfigMigrationTests(unittest.TestCase):
    """Forge's real UiLoadsave reapplies saved slider value/minimum/maximum/step per label
    (modules/ui_loadsave.py:37-110), so the old CNS bounds would survive the code change."""

    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    def setUp(self):
        _BASE.AnimaSafePagTests.setUp(self)
        if getattr(self, "_tmp", None) is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ui-config.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, settings):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=4, ensure_ascii=False)

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _forge_ui(self, tab):
        with gr.Blocks() as block:
            inputs = self.pag.AnimaSafePAG().ui(tab == "img2img")
        for component in inputs:  # modules/scripts.py:672-673
            component.custom_script_source = "anima_safe_pag.py"
        loadsave = _MIGRATION_TESTS._load_forge_ui_loadsave().UiLoadsave(str(self.path))  # skips without Forge
        loadsave.add_block(block, tab)
        by_id = {c.elem_id: c for c in inputs}
        return {name: by_id[ELEM_IDS[name]] for name in PARAMS}, loadsave

    def _fields(self, slider):
        return (slider.value, slider.minimum, slider.maximum, slider.step)

    def test_without_the_migration_the_old_bounds_win(self):
        self._write(_legacy_cns_tab("txt2img"))
        sliders, _loadsave = self._forge_ui("txt2img")
        self.assertEqual(sliders["strength"].step, 0.01)
        self.assertEqual(sliders["gamma_power"].minimum, 0.05)

    def test_old_defaults_reach_the_upstream_inputs(self):
        self._write({**_legacy_cns_tab("txt2img"), **_legacy_cns_tab("img2img")})
        changes = migration.migrate_ui_config_file(self.path)
        self.assertTrue(changes)
        saved = self._read()
        self.assertFalse([key for key in saved if migration.CNS_GAMMA_SCALE_LEGACY_LABEL in key])
        for tab in migration.TABS:
            sliders, _loadsave = self._forge_ui(tab)
            for name in PARAMS:
                spec = UPSTREAM_INPUTS[name][1]
                with self.subTest(tab=tab, name=name):
                    self.assertEqual(
                        self._fields(sliders[name]),
                        (spec["default"], spec["min"], spec["max"], spec["step"]),
                    )

    def test_values_the_user_set_stay(self):
        self._write(_legacy_cns_tab("txt2img", strength=0.6, gamma_power=0.05, gamma_scale=5.0))
        changes = migration.migrate_ui_config_file(self.path)
        self.assertTrue(any("5 -> " in line for line in changes), changes)
        sliders, _loadsave = self._forge_ui("txt2img")
        self.assertEqual(self._fields(sliders["strength"]), (0.6, 0.0, 1.0, 0.05))
        # below the upstream minimum: raised like the server clamp
        self.assertEqual(self._fields(sliders["gamma_power"]), (0.1, 0.1, 2.0, 0.05))
        self.assertEqual(self._fields(sliders["gamma_scale"]), (5.0, 0.1, 25.0, 0.1))

    def test_runs_once_and_fresh_installs_are_untouched(self):
        self._write(_legacy_cns_tab("img2img", gamma_scale=4.0))
        self.assertTrue(migration.migrate_ui_config_file(self.path))
        for tab in migration.TABS:
            _sliders, loadsave = self._forge_ui(tab)
            loadsave.dump_defaults()
        self.assertEqual(migration.migrate_ui_config_file(self.path), [])
        saved = self._read()
        label = migration.CNS_GAMMA_SCALE_LABEL
        self.assertEqual(saved[f"{PREFIX}/img2img/{label}/value"], 4.0)
        self.assertEqual(saved[f"{PREFIX}/txt2img/{label}/value"], 2.0)
        self.assertEqual(saved[f"{PREFIX}/img2img/CNS strength/step"], 0.05)

        # The user now saves 3.0 and a .05 step on purpose under the new label: left alone.
        saved[f"{PREFIX}/img2img/{label}/value"] = 3.0
        before = dict(saved)
        self.assertEqual(migration.migrate_ui_settings(saved), [])
        self.assertEqual(saved, before)


if __name__ == "__main__":
    unittest.main()
