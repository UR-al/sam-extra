from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_dd_module():
    """Load the Detail Daemon script without booting the full WebUI."""
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    modules_stub.scripts = types.SimpleNamespace(
        Script=Script,
        AlwaysVisible=object(),
        scripts_data=[],
    )
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_cfg_denoiser=lambda fn: None,
        on_before_ui=lambda fn: None,
    )

    old_modules = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_anima_detail_daemon", ROOT / "scripts" / "anima_detail_daemon.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_modules is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = old_modules


dd = _load_dd_module()


# 11 sampler steps whose step-5 sigma is 10.0 (Forge keeps the list in
# transformer_options["sampling_sigmas"], sd_samplers_kdiffusion.py:246).
SIGMAS_11 = torch.tensor([15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 0.0])
# 6 sampler steps whose step-2 sigma is 10.0 (Heun corrector of step 1 runs at it).
SIGMAS_6 = torch.tensor([12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 0.0])


def _processing(cfg_scale=4.0, is_hr_pass=False, hr_cfg=7.0, refiner_cfg=None,
                sampler_name="Euler", sampling_sigmas=SIGMAS_11):
    unet = types.SimpleNamespace(
        model_options={"transformer_options": {"sampling_sigmas": sampling_sigmas}},
    )
    return types.SimpleNamespace(
        cfg_scale=cfg_scale,
        is_hr_pass=is_hr_pass,
        hr_cfg=hr_cfg,
        refiner_cfg=refiner_cfg,
        sampler_name=sampler_name,
        extra_generation_params={},
        sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=unet)),
    )


def _script_args(enabled=True, preset="Custom", amount=0.1, start=0.2, end=0.8,
                 bias=0.5, exponent=1.0, start_offset=0.0, end_offset=0.0,
                 fade=0.0, multiplier=1.0, smooth=True, cfg_couple=True, hires=False):
    # Same order as AnimaDetailDaemon.ui() returns its components (14 args).
    return [enabled, preset, amount, start, end, bias, exponent,
            start_offset, end_offset, fade, multiplier, smooth, cfg_couple, hires]


def _denoiser_params(p, *, state_step, state_steps, model_calls_done,
                     model_calls_total, sampler_steps, refiner_pass=False,
                     classic=False):
    """Mirror Forge's CFGDenoiserParams / CFGDenoiser fields for one model call."""
    denoiser = types.SimpleNamespace(
        step=model_calls_done,
        total_steps=model_calls_total,
        steps=sampler_steps,
        p=p,
        _refiner_pass=refiner_pass,
        classic_ddim_eps_estimation=classic,
    )
    return types.SimpleNamespace(
        x=None,
        image_cond=None,
        sigma=torch.tensor([10.0]),
        sampling_step=state_step,
        total_sampling_steps=state_steps,
        text_cond=None,
        text_uncond=None,
        denoiser=denoiser,
    )


def _run(p, args, **params_kwargs):
    dd.AnimaDetailDaemon().process_before_every_sampling(p, *args)
    params = _denoiser_params(p, **params_kwargs)
    dd._denoiser_callback(params)
    return params.sigma.item()


# 11-step curve with the defaults (start 0.2, mid 0.5, end 0.8, smooth):
# schedule[5] = amount (peak), schedule[4] = 0.75 * amount.
PEAK_EULER_CALL = dict(state_step=5, state_steps=11, model_calls_done=5,
                       model_calls_total=11, sampler_steps=11)


class DetailDaemonStrengthTests(unittest.TestCase):
    def test_peak_step_uses_original_strength_scale(self):
        # muerrilla/ComfyUI: sigma * (1 - 0.1 * 0.1 * 4) = 10 * 0.96
        sigma = _run(_processing(cfg_scale=4.0), _script_args(amount=0.1), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 9.6, places=5)

    def test_old_cfg_couple_slot_is_ignored(self):
        # The originals always multiply by cfg_scale ("both" mode on Forge).
        sigma = _run(_processing(cfg_scale=4.0),
                     _script_args(amount=0.1, cfg_couple=False), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 9.6, places=5)

    def test_old_multiplier_slot_is_ignored(self):
        sigma = _run(_processing(cfg_scale=4.0),
                     _script_args(amount=0.1, multiplier=0.5), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 9.6, places=5)

    def test_disabled_leaves_sigma_untouched(self):
        sigma = _run(_processing(), _script_args(enabled=False), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 10.0, places=5)


class DetailDaemonStepTests(unittest.TestCase):
    def test_position_comes_from_the_call_sigma_not_the_lagging_state_step(self):
        # Forge updates state.sampling_step after the model call, so during
        # step 5 it still reads 4. Using it would pick schedule[4]: 10 * 0.97.
        # The ComfyUI node looks the call's sigma (10.0 = sigmas[5]) up instead.
        call = dict(PEAK_EULER_CALL, state_step=4)
        sigma = _run(_processing(cfg_scale=4.0), _script_args(amount=0.1), **call)
        self.assertAlmostEqual(sigma, 9.6, places=5)

    def test_second_order_sampler_reads_the_curve_at_the_call_sigma(self):
        # Heun-style: 6 sampler steps, Forge total_steps = 12 → 11 model calls.
        # Model call 3 is the corrector of sampler step 1, made at sigmas[2] = 10.
        # ComfyUI node: 6-entry curve looked up by sigma → schedule[2] = 0.1
        # → 10 * (1 - 0.1 * 0.1 * 4) = 9.6. (muerrilla's 11-entry call-count
        # curve gave schedule[3] = 0.025 → 9.9.)
        call = dict(state_step=1, state_steps=6, model_calls_done=3,
                    model_calls_total=12, sampler_steps=6)
        p = _processing(cfg_scale=4.0, sampling_sigmas=SIGMAS_6)
        sigma = _run(p, _script_args(amount=0.1), **call)
        self.assertAlmostEqual(sigma, 9.6, places=5)

    def test_timestep_samplers_use_the_model_call_count(self):
        # DDIM / PLMS (classic_ddim_eps_estimation) give no sigma list: the
        # position is muerrilla's call counter. Call 3 of an 11-call curve →
        # schedule[3] = 0.025 → 9.9, even with a stale sampling_sigmas around.
        call = dict(state_step=1, state_steps=6, model_calls_done=3,
                    model_calls_total=12, sampler_steps=6, classic=True)
        p = _processing(cfg_scale=4.0, sampling_sigmas=SIGMAS_6)
        sigma = _run(p, _script_args(amount=0.1), **call)
        self.assertAlmostEqual(sigma, 9.9, places=5)

    def test_sigma_scaled_in_place(self):
        # muerrilla ``params.sigma *= ...``: CFGDenoiser.forward's own ``sigma``
        # (NGMS check, MaskBlendArgs) is the same tensor and sees the change.
        p = _processing(cfg_scale=4.0)
        dd.AnimaDetailDaemon().process_before_every_sampling(p, *_script_args(amount=0.1))
        params = _denoiser_params(p, **PEAK_EULER_CALL)
        forward_sigma = params.sigma
        dd._denoiser_callback(params)
        self.assertIs(params.sigma, forward_sigma)
        self.assertAlmostEqual(forward_sigma.item(), 9.6, places=5)


class DetailDaemonCfgSourceTests(unittest.TestCase):
    # muerrilla captures ``self.cfg_scale = p.cfg_scale`` once and uses it on
    # every pass, so hr_cfg / refiner_cfg do not change the strength.
    def test_hires_pass_uses_base_cfg_scale(self):
        # Hires Pass daemon on the HiRes pass: 10 * (1 - 0.1 * 0.1 * 4), not hr_cfg 2
        p = _processing(cfg_scale=4.0, is_hr_pass=True, hr_cfg=2.0)
        sigma = _run(p, _script_args(amount=0.1, hires=True), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 9.6, places=5)

    def test_refiner_pass_uses_base_cfg_scale(self):
        p = _processing(cfg_scale=4.0, refiner_cfg=3.0)
        call = dict(PEAK_EULER_CALL, refiner_pass=True)
        sigma = _run(p, _script_args(amount=0.1), **call)
        self.assertAlmostEqual(sigma, 9.6, places=5)


class DetailDaemonPresetTests(unittest.TestCase):
    def test_old_preset_slot_is_ignored(self):
        # No presets in the originals: the amount slider is what runs.
        # 10 * (1 - 0.3 * 0.1 * 4)
        sigma = _run(_processing(cfg_scale=4.0),
                     _script_args(preset="Strong", amount=0.3), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 8.8, places=5)

    def test_preset_helpers_and_table_are_gone(self):
        for name in ("_PRESETS", "_preset_amount_update", "_amount_input_preset_update"):
            self.assertFalse(hasattr(dd, name), name)


class DetailDaemonInfotextTests(unittest.TestCase):
    def test_infotext_records_every_curve_parameter(self):
        p = _processing()
        args = _script_args(amount=0.3, exponent=1.5, start_offset=-0.1,
                            end_offset=0.2, fade=0.3, smooth=False, hires=True)
        dd.AnimaDetailDaemon().process_before_every_sampling(p, *args)
        text = p.extra_generation_params["Anima Detail Daemon"]
        for expected in ("amount=0.3", "exponent=1.5", "start_offset=-0.1",
                         "end_offset=0.2", "fade=0.3", "smooth=False", "hires=True"):
            self.assertIn(expected, text)
        # The ignored old slots are not recorded.
        self.assertNotIn("multiplier=", text)
        self.assertNotIn("cfg_couple=", text)


if __name__ == "__main__":
    unittest.main()
