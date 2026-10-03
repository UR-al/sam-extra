"""SPEED under the extension's guidance stack — a smoke test on Forge's real pieces, CPU only.

A request goes through the real hooks of scripts/anima_speed.py, scripts/anima_safe_pag.py (PAG with
its weak row, DCW, TSR, Momentum Guidance, HiGS, HiFlow recording and aligning) and
scripts/anima_detail_daemon.py (sigma scaling), then samples with a k-diffusion Euler loop whose
model call is Forge's ``CFGDenoiser.forward`` order: ``on_cfg_denoiser`` callbacks, then Forge's real
``sampling_function_inner`` (backend/sampling/sampling_function.py) over Forge's real Anima DiT
(backend/nn/anima.py, 6 small blocks). The latent changes size mid-run; every stage must survive it:
no exception, a finite full-size result, histories re-seeded at the new size, the HiFlow trajectory
holding only full-size records, Detail Daemon finding the patched sigmas in ``sampling_sigmas``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_anima_safe_pag import _load_forge_sampler, _load_pag_module, _load_real_forge_anima  # noqa: E402
from tests.test_anima_speed_script import (  # noqa: E402
    SCRIPT_MODULE as SPEED,
    FakeImageRNG,
    FakeTorchHijack,
    KDiffusionSampler,
    _forge_rng_module,
    _stub_modules,
    fake_backend_args,
    fake_kdiffusion,
)
from sam3ext.speed import forge_host as fh  # noqa: E402


def _load_detail_daemon():
    modules_stub = types.ModuleType("modules")

    class Script:
        is_img2img = False

    modules_stub.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules_stub.script_callbacks = types.SimpleNamespace(on_cfg_denoiser=lambda fn: None, on_before_ui=lambda fn: None)
    with _stub_modules({"modules": modules_stub}):
        spec = importlib.util.spec_from_file_location("_speed_stack_dd", ROOT / "scripts" / "anima_detail_daemon.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class FlowPredictor:
    prediction_type = "const"

    @staticmethod
    def percent_to_sigma(percent):
        return 1.0 - float(percent)


class ForgeUnet:
    """Forge ModelPatcher option handling (backend/patcher/base.py:65-69, 236-254)."""

    def __init__(self, dit, model_options=None):
        self.model = types.SimpleNamespace(diffusion_model=dit, predictor=FlowPredictor())
        self.model_options = model_options if model_options is not None else {"transformer_options": {}}
        self.controlnet_linked_list = None

    def clone(self):
        options = dict(self.model_options)
        options["transformer_options"] = dict(options.get("transformer_options", {}))
        return ForgeUnet(self.model.diffusion_model, options)

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def set_model_sampler_pre_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_pre_cfg_function"] = self.model_options.get("sampler_pre_cfg_function", []) + [fn]

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_post_cfg_function"] = self.model_options.get("sampler_post_cfg_function", []) + [fn]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True


class KModel:
    """Forge's KModel.apply_model for a const predictor: x0 = x - sigma * v (multiplier 1, like Anima)."""

    predictor = FlowPredictor()

    def __init__(self, dit, shapes):
        self.dit = dit
        self.shapes = shapes

    @staticmethod
    def memory_required(shape):
        return 0

    def apply_model(self, x, t, c_crossattn=None, transformer_options=None, **kwargs):
        self.shapes.append(tuple(x.shape))
        with torch.no_grad():
            out = self.dit(x, t, context=c_crossattn, transformer_options=transformer_options or {})
        return x - out * t.reshape(t.shape + (1,) * (x.ndim - 1)).to(x.dtype)


def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    # origin: modules_forge/packages/k_diffusion/sampling.py:160-184 (s_churn = 0)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = (x - denoised) / sigma_hat
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        x = x + d * (sigmas[i + 1] - sigma_hat)
    return x


def sample_euler_ancestral_rf(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0):
    # origin: modules_forge/packages/k_diffusion/sampling.py:212-238 (noise through the module's torch)
    kds = sys.modules["k_diffusion.sampling"]
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            x = (alpha_ip1 / alpha_down) * x + kds.torch.randn_like(x) * s_noise * renoise_coeff
    return x


class GuidanceStackUnderSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        forge = ROOT.parents[1]
        if not all((forge / path).is_file() for path in (
            "backend/sampling/condition.py", "backend/sampling/sampling_function.py", "backend/nn/anima.py",
        )):
            raise unittest.SkipTest("Forge backend sources not found next to the extension")
        cls.pag = _load_pag_module()
        cls.dd = _load_detail_daemon()
        cls.forge_sampler, cls.Condition, cls.memory = _load_forge_sampler()

    def setUp(self):
        self.record = []
        anima = _load_real_forge_anima(self.record)
        torch.manual_seed(1234)
        self.dit = anima.Anima(
            in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
            model_channels=48, crossattn_emb_channels=32, adaln_lora_dim=16, num_blocks=6, num_heads=4,
        ).eval()
        self.pag_script = self.pag.AnimaSafePAG()
        with gr.Blocks():
            inputs = self.pag_script.ui(False)
        self.pag_args = [c.value for c in inputs]
        self.pag_index = {c.elem_id: i for i, c in enumerate(inputs) if c.elem_id}
        g = torch.Generator().manual_seed(5)
        self.ctx_cond = torch.randn(1, 7, 32, generator=g)
        self.ctx_uncond = torch.randn(1, 7, 32, generator=g)

    def tearDown(self):
        self.pag._teardown_global_patches()

    # -- the request ---------------------------------------------------------------------------

    def _pag_on(self, **elem_values):
        args = list(self.pag_args)
        defaults = {
            "anima_safe_pag_enable": True, "anima_safe_pag_blocks": "3", "anima_safe_pag_end": 1.0,
            "anima_guidance_dcw_enable": True, "anima_guidance_tsr_enable": True,
            "anima_guidance_mg_enable": True, "anima_guidance_mg_min": 0.0, "anima_guidance_mg_max": 1.0,
            "anima_guidance_higs_enable": True, "anima_guidance_higs_t_min": 0.0,
            "anima_guidance_hiflow_enable": True,
        }
        defaults.update(elem_values)
        for elem_id, value in defaults.items():
            args[self.pag_index[elem_id]] = value
        return args

    def _request(self, func, seeds=(31,), shape=(4, 1, 16, 16)):
        model = type("Anima", (), {"is_inpaint": False})()
        model.forge_objects = types.SimpleNamespace(unet=ForgeUnet(self.dit))
        request = type("StableDiffusionProcessingTxt2Img", (), {})()
        request.sd_model = model
        request.extra_generation_params = {}
        request.steps = 10
        request.seeds = list(seeds)
        request.is_hr_pass = False
        request.enable_hr = True
        request.mask = None
        request.image_mask = None
        request.cfg_scale = 4.0
        request.sampler_name = "Euler"
        request.sampler = KDiffusionSampler(func)
        request.rng = FakeImageRNG(shape, seeds)
        return request

    def _denoiser(self, request, shapes, scaled):
        test = self
        forge_sampler, Condition = self.forge_sampler, self.Condition

        class Denoiser:
            """Forge CFGDenoiser.forward: callbacks, then sampling_function -> sampling_function_inner."""

            p = request
            steps = request.steps
            step = 0
            classic_ddim_eps_estimation = False
            mask = None
            inner_model = types.SimpleNamespace(predictor=FlowPredictor())

            def __call__(self, x, sigma, **extra):
                params = types.SimpleNamespace(x=x, image_cond=None, sigma=sigma, sampling_step=self.step,
                                               total_sampling_steps=request.steps, text_cond=None,
                                               text_uncond=None, denoiser=self)
                before = sigma.clone()
                test.dd._denoiser_callback(params)
                test.pag._dave_run_callback(params)
                test.pag._hiflow_run_callback(params)
                scaled.append((float(before[0]), float(params.sigma.flatten()[0])))   # every call
                unet = request.sd_model.forge_objects.unet
                cond = [{"model_conds": {"c_crossattn": Condition(test.ctx_cond)}, "strength": 1.0}]
                uncond = [{"model_conds": {"c_crossattn": Condition(test.ctx_uncond)}}]
                with mock.patch.dict(sys.modules, {"backend.sampling.sampling_function": forge_sampler}):
                    out = forge_sampler.sampling_function_inner(
                        KModel(test.dit, shapes), params.x, params.sigma, uncond, cond, request.cfg_scale,
                        unet.model_options,
                    )
                self.step += 1
                return out

        return Denoiser()

    def _sample(self, request, x, sigmas):
        """Forge KDiffusionSampler.sample after the hooks: publish sampling_sigmas, call func."""
        unet = request.sd_model.forge_objects.unet
        unet.model_options.setdefault("transformer_options", {})["sampling_sigmas"] = sigmas
        shapes, scaled, seen = [], [], []
        out = request.sampler.func(self._denoiser(request, shapes, scaled), x, sigmas=sigmas, extra_args={},
                                   callback=lambda d: seen.append(d["i"]), disable=True)
        self.assertIs(unet.model_options["transformer_options"]["sampling_sigmas"], sigmas, "restored")
        return out, shapes, scaled, seen

    def _run_base(self, mode, func=sample_euler):
        dd = self.dd
        request = self._request(func)
        with fake_backend_args(), fake_kdiffusion(FakeTorchHijack(request.rng)), \
                _stub_modules({"modules.rng": _forge_rng_module()}):
            self.pag_script.process_before_every_sampling(request, *self._pag_on())
            dd_args = [True, "Custom", 0.4, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, True, True, False]
            dd.AnimaDetailDaemon().process_before_every_sampling(request, *dd_args)
            speed_args = dict(fh.DEFAULTS, enabled=True, mode=mode, threshold="manual", manual_sigmas="0.7")
            SPEED.AnimaSpeed().process_before_every_sampling(request, *[speed_args[n] for n in fh.ARG_NAMES])
            sigmas = torch.tensor([3.0 * t / (1.0 + 2.0 * t) for t in torch.linspace(1.0, 0.0, 11).tolist()])
            x = request.rng.next()
            out, shapes, scaled, seen = self._sample(request, x, sigmas)
        return request, out, shapes, scaled, seen, sigmas

    # -- tests ---------------------------------------------------------------------------------

    def test_base_pass_with_every_stage_on(self):
        for mode in ("transition", "respace"):
            with self.subTest(mode=mode):
                pag, dd = self.pag, self.dd
                self.record.clear()
                request, out, shapes, scaled, seen, sigmas = self._run_base(mode)
                try:
                    status = request.extra_generation_params[fh.STATUS_KEY]
                    self.assertTrue(status.startswith(f"base: applied ({mode})"), status)
                    self.assertEqual(tuple(out.shape), (1, 4, 1, 16, 16))
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertEqual(seen, list(range(10)))
                    # the DiT ran on both grids, with PAG's weak row (3 rows = cond, uncond, weak)
                    grids = [s[-2:] for s in shapes]
                    self.assertIn((8, 8), grids)
                    self.assertIn((16, 16), grids)
                    self.assertEqual(grids, sorted(grids))
                    self.assertTrue(all(s[0] == 3 for s in shapes), shapes)
                    # every stage worked on both sizes (TSR leaves sigma = 1 alone: no log-SNR there)
                    self.assertEqual(pag._TSR["steps"], sum(1 for before, _ in scaled if before < 1.0))
                    self.assertGreater(pag._DCW["steps"], 0)
                    self.assertGreater(pag._RUNTIME.history.counters["mg"], 0)
                    self.assertGreater(pag._RUNTIME.history.counters["higs"], 0)
                    self.assertEqual(tuple(pag._RUNTIME.history.mg_m.shape), (1, 4, 1, 16, 16), "re-seeded at full size")
                    self.assertEqual(tuple(pag._RUNTIME.history.higs_g.shape), (1, 4, 1, 16, 16))
                    trajectory = pag._HIFLOW["trajectory"]
                    self.assertGreater(len(trajectory), 0)
                    self.assertEqual(trajectory.shape, (1, 4, 1, 16, 16), "coarse records were dropped")
                    self.assertTrue(any(after != before for before, after in scaled), "Detail Daemon scaled sigmas")
                finally:
                    self.pag_script.postprocess(request, None)
                    dd.AnimaDetailDaemon().postprocess(request, None)

    def test_detail_daemon_finds_the_patched_sigma(self):
        """The first full-size step runs at the aligned sigma; Detail Daemon looks it up in the
        published schedule (an exact match) instead of interpolating between neighbours."""
        dd = self.dd
        request, out, shapes, scaled, seen, sigmas = self._run_base("transition")
        try:
            first_full = next(i for i, s in enumerate(shapes) if s[-1] == 16)
            sigma_in, sigma_out = scaled[first_full]
            step = next(j for j in range(10) if float(sigmas[j]) <= 0.7)
            aligned = float(sigmas[step]) * 2.0 / (1.0 + float(sigmas[step]))
            self.assertAlmostEqual(sigma_in, aligned, places=5)
            schedule = torch.tensor(dd._make_schedule(10, 0.0, 1.0, 0.5, 0.4, 1.0, 0.0, 0.0, 0.0, True))
            expected = sigma_in * max(1e-06, 1.0 - float(schedule[step]) * 0.1 * 4.0)
            self.assertAlmostEqual(sigma_out, expected, places=5)
        finally:
            self.pag_script.postprocess(request, None)
            dd.AnimaDetailDaemon().postprocess(request, None)

    def test_hires_pass_hiflow_aligns_on_the_coarse_grid_too(self):
        pag, dd = self.pag, self.dd
        request, out, _shapes, _scaled, _seen, _sigmas = self._run_base("transition")
        try:
            request.is_hr_pass = True
            request.sampler = KDiffusionSampler(sample_euler_ancestral_rf)
            request.rng = FakeImageRNG((4, 1, 24, 24), request.seeds)
            with fake_backend_args(), fake_kdiffusion(FakeTorchHijack(request.rng)), \
                    _stub_modules({"modules.rng": _forge_rng_module()}):
                self.pag_script.process_before_every_sampling(request, *self._pag_on())
                self.assertTrue(pag._HIFLOW["applying"])
                dd_args = [True, "Custom", 0.4, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, True, True, False]
                dd.AnimaDetailDaemon().process_before_every_sampling(request, *dd_args)   # base pass only
                speed_args = dict(fh.DEFAULTS, enabled=True, threshold="manual", manual_sigmas="0.7", hires=True)
                SPEED.AnimaSpeed().process_before_every_sampling(request, *[speed_args[n] for n in fh.ARG_NAMES])
                hires_sigmas = torch.tensor([0.85, 0.75, 0.62, 0.45, 0.25, 0.0])
                upscaled = torch.nn.functional.interpolate(out.squeeze(2), size=(24, 24), mode="bilinear").unsqueeze(2)
                noise = request.rng.next()
                x = 0.85 * noise + 0.15 * upscaled
                hires_out, hires_shapes, _, hires_seen = self._sample(request, x, hires_sigmas)
            status = request.extra_generation_params[fh.STATUS_KEY]
            self.assertIn("hires: applied (transition)", status)
            self.assertIn("init=rescaled", status)
            self.assertEqual(tuple(hires_out.shape), (1, 4, 1, 24, 24))
            self.assertTrue(torch.isfinite(hires_out).all())
            self.assertIn((12, 12), [s[-2:] for s in hires_shapes])
            self.assertGreater(pag._HIFLOW["state"].counters["direction"], 0)
            self.assertEqual(hires_seen, list(range(5)))
        finally:
            self.pag_script.postprocess(request, None)
            dd.AnimaDetailDaemon().postprocess(request, None)


if __name__ == "__main__":
    unittest.main()
