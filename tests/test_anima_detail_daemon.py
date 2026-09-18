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


def _processing(cfg_scale=4.0, is_hr_pass=False, hr_cfg=7.0, refiner_cfg=None):
    return types.SimpleNamespace(
        cfg_scale=cfg_scale,
        is_hr_pass=is_hr_pass,
        hr_cfg=hr_cfg,
        refiner_cfg=refiner_cfg,
        extra_generation_params={},
    )


def _script_args(enabled=True, preset="Custom", amount=0.1, start=0.2, end=0.8,
                 bias=0.5, exponent=1.0, start_offset=0.0, end_offset=0.0,
                 fade=0.0, multiplier=1.0, smooth=True, cfg_couple=True):
    # Same order as AnimaDetailDaemon.ui() returns its components.
    return [enabled, preset, amount, start, end, bias, exponent,
            start_offset, end_offset, fade, multiplier, smooth, cfg_couple]


def _denoiser_params(p, *, state_step, state_steps, model_calls_done,
                     model_calls_total, sampler_steps, refiner_pass=False):
    """Mirror Forge's CFGDenoiserParams / CFGDenoiser fields for one model call."""
    denoiser = types.SimpleNamespace(
        step=model_calls_done,
        total_steps=model_calls_total,
        steps=sampler_steps,
        p=p,
        _refiner_pass=refiner_pass,
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

    def test_uncoupled_mode_ignores_cfg_scale(self):
        # 10 * (1 - 0.1 * 0.1 * 1)
        sigma = _run(_processing(cfg_scale=4.0),
                     _script_args(amount=0.1, cfg_couple=False), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 9.9, places=5)

    def test_disabled_leaves_sigma_untouched(self):
        sigma = _run(_processing(), _script_args(enabled=False), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 10.0, places=5)


class DetailDaemonStepTests(unittest.TestCase):
    def test_reads_model_call_count_not_lagging_state_step(self):
        # Forge updates state.sampling_step after the model call, so during
        # step 5 it still reads 4. Using it would pick schedule[4]: 10 * 0.97.
        call = dict(PEAK_EULER_CALL, state_step=4)
        sigma = _run(_processing(cfg_scale=4.0), _script_args(amount=0.1), **call)
        self.assertAlmostEqual(sigma, 9.6, places=5)

    def test_second_order_sampler_spreads_schedule_over_model_calls(self):
        # Heun-style: 6 sampler steps, Forge total_steps = 12 → 11 model calls.
        # Model call 3 is the second call of sampler step 1 (state step 1).
        # 11-entry curve: schedule[3] = 0.025 → 10 * (1 - 0.025 * 0.1 * 4) = 9.9
        # (6-entry curve by state step: 10.0; by call index: schedule[3] = 0.05 → 9.8)
        call = dict(state_step=1, state_steps=6, model_calls_done=3,
                    model_calls_total=12, sampler_steps=6)
        sigma = _run(_processing(cfg_scale=4.0), _script_args(amount=0.1), **call)
        self.assertAlmostEqual(sigma, 9.9, places=5)


class DetailDaemonCfgSourceTests(unittest.TestCase):
    def test_hires_pass_couples_to_hires_cfg(self):
        # 10 * (1 - 0.1 * 0.1 * 2)
        p = _processing(cfg_scale=4.0, is_hr_pass=True, hr_cfg=2.0)
        sigma = _run(p, _script_args(amount=0.1), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 9.8, places=5)

    def test_refiner_pass_couples_to_refiner_cfg(self):
        # 10 * (1 - 0.1 * 0.1 * 3)
        p = _processing(cfg_scale=4.0, refiner_cfg=3.0)
        call = dict(PEAK_EULER_CALL, refiner_pass=True)
        sigma = _run(p, _script_args(amount=0.1), **call)
        self.assertAlmostEqual(sigma, 9.7, places=5)


class DetailDaemonPresetTests(unittest.TestCase):
    def test_named_preset_does_not_override_amount_slider(self):
        # Slider 0.3 wins over "Strong": 10 * (1 - 0.3 * 0.1 * 4)
        sigma = _run(_processing(cfg_scale=4.0),
                     _script_args(preset="Strong", amount=0.3), **PEAK_EULER_CALL)
        self.assertAlmostEqual(sigma, 8.8, places=5)

    def test_choosing_named_preset_fills_amount_slider(self):
        update = dd._preset_amount_update("Strong")
        self.assertEqual(update.get("value"), 0.25)

    def test_choosing_custom_keeps_amount_slider(self):
        update = dd._preset_amount_update("Custom")
        self.assertNotIn("value", update)

    def test_moving_amount_slider_marks_preset_custom(self):
        update = dd._amount_input_preset_update(0.42)
        self.assertEqual(update.get("value"), "Custom")


class DetailDaemonInfotextTests(unittest.TestCase):
    def test_infotext_records_every_curve_parameter(self):
        p = _processing()
        args = _script_args(amount=0.3, exponent=1.5, start_offset=-0.1,
                            end_offset=0.2, fade=0.3, multiplier=0.5,
                            smooth=False, cfg_couple=False)
        dd.AnimaDetailDaemon().process_before_every_sampling(p, *args)
        text = p.extra_generation_params["Anima Detail Daemon"]
        for expected in ("amount=0.3", "exponent=1.5", "start_offset=-0.1",
                         "end_offset=0.2", "fade=0.3", "multiplier=0.5",
                         "smooth=False", "cfg_couple=False"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
