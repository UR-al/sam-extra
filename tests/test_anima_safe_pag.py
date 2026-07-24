from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_pag_module():
    """Load the extension script without booting the full WebUI."""
    modules_stub = types.ModuleType("modules")
    callbacks_stub = types.SimpleNamespace(on_before_ui=lambda fn: None)

    class Script:
        pass

    scripts_stub = types.SimpleNamespace(
        Script=Script,
        AlwaysVisible=object(),
        scripts_data=[],
    )
    modules_stub.script_callbacks = callbacks_stub
    modules_stub.scripts = scripts_stub

    old_modules = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_anima_safe_pag", ROOT / "scripts" / "anima_safe_pag.py"
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


class AnimaSafePagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def setUp(self):
        p = self.pag
        p._STATE.update(
            on=True,
            step_open=False,
            step=0,
            total=20,
            start=0.0,
            end=1.0,
            attn_method="pag",
            attn_scale=4.0,
            strength=0.75,
            legacy_attn=False,
            seg_sigma=100.0,
            head_spec="",
            attn_targets={0},
            slg_on=False,
            slg_scale=3.0,
            slg_targets=set(),
            rescale=0.0,
            rescale_mode="full",
            active=0,
            attn_raw=None,
            slg_raw=None,
            attn_b0=None,
            attn_b1=None,
            slg_b0=None,
            slg_b1=None,
            any_b0=None,
            attn_spatial_shape=None,
            attn_hook_hits=0,
            attn_hook_hits_total=0,
            attn_last_rel_delta=None,
            attn_diag_logged=False,
            attn_shape_warned=False,
            adg_skipped=False,
            apg_autooff_rescale=True,
            wrapper_calls=0,
            weak_steps=0,
            applied_steps=0,
            apg_steps=0,
            adg_skipped_steps=0,
            combined_calls=0,
            split_cond_calls=0,
            split_uncond_calls=0,
            control_blocked_calls=0,
            wrapper_fallbacks=0,
            delta_logged=False,
        )
        p._APG["on"] = False
        p._ADG["on"] = False
        p._CFG.update(
            smc_on=False,
            apg_on=False,
            cwm_on=False,
            mode="preserve",
            experimental_stack=False,
            steps=0,
            fit_error=None,
            effective_scale=None,
            external_cfg_detected=False,
            warned=False,
        )
        p._DCW.update(on=False, steps=0)
        p._DAVE.update(on=False, targets=set(), steps=0)
        p._CNS.update(on=False, warned=False)
        p._RUNTIME.reset_pass()

    def _fake_apply_model(self, x, timestep, **conditioning):
        p = self.pag
        out = conditioning["bias"].clone()
        if p._STATE["any_b0"] is not None:
            a0, a1 = p._STATE["attn_b0"], p._STATE["attn_b1"]
            out[a0:a1] += 2.0  # stand-in for a perturbed weak prediction
        return out

    def _post_cfg(self):
        return self.pag._post_cfg(
            {
                "denoised": torch.tensor([[7.5]]),
                "cond_denoised": torch.ones(1, 1),
                "uncond_denoised": torch.zeros(1, 1),
            }
        )

    def test_low_vram_split_cond_uncond_applies_pag(self):
        p = self.pag
        x = torch.zeros(1, 1)
        timestep = torch.ones(1)

        # Forge commonly runs uncond first, then cond, when both do not fit.
        p._model_wrapper(
            self._fake_apply_model,
            {
                "input": x,
                "timestep": timestep,
                "c": {"bias": torch.zeros(1, 1)},
                "cond_or_uncond": [1],
            },
        )
        p._model_wrapper(
            self._fake_apply_model,
            {
                "input": x,
                "timestep": timestep,
                "c": {"bias": torch.ones(1, 1)},
                "cond_or_uncond": [0],
            },
        )

        self.assertEqual(p._STATE["step"], 0)
        self.assertEqual(p._STATE["weak_steps"], 1)
        self.assertEqual(p._STATE["split_uncond_calls"], 1)
        self.assertEqual(p._STATE["split_cond_calls"], 1)
        self.assertEqual(p._STATE["combined_calls"], 0)
        self.assertEqual(self._post_cfg().item(), -0.5)
        self.assertEqual(p._STATE["applied_steps"], 1)

    def test_combined_cond_uncond_applies_pag(self):
        p = self.pag
        p._model_wrapper(
            self._fake_apply_model,
            {
                "input": torch.zeros(2, 1),
                "timestep": torch.ones(2),
                "c": {"bias": torch.tensor([[0.0], [1.0]])},
                "cond_or_uncond": [1, 0],
            },
        )

        self.assertEqual(p._STATE["wrapper_calls"], 1)
        self.assertEqual(p._STATE["weak_steps"], 1)
        self.assertEqual(p._STATE["combined_calls"], 1)
        self.assertEqual(self._post_cfg().item(), -0.5)

    def test_adaptive_guidance_counts_only_a_real_combined_batch_skip(self):
        p = self.pag
        p._STATE["on"] = False
        p._ADG.update(on=True, start=0.0, interval=0)

        out = p._model_wrapper(
            self._fake_apply_model,
            {
                "input": torch.zeros(2, 1),
                "timestep": torch.ones(2),
                "c": {"bias": torch.tensor([[0.0], [1.0]])},
                "cond_or_uncond": [1, 0],
            },
        )

        self.assertEqual(p._STATE["combined_calls"], 1)
        self.assertEqual(p._STATE["adg_skipped_steps"], 1)
        torch.testing.assert_close(out, torch.ones(2, 1))

    def test_cfg_one_cond_only_still_applies_pag(self):
        p = self.pag
        p._model_wrapper(
            self._fake_apply_model,
            {
                "input": torch.zeros(1, 1),
                "timestep": torch.ones(1),
                "c": {"bias": torch.ones(1, 1)},
                "cond_or_uncond": [0],
            },
        )

        self.assertEqual(p._STATE["weak_steps"], 1)
        self.assertEqual(self._post_cfg().item(), -0.5)

    def test_pag_strength_blends_value_path_and_changes_only_weak_rows(self):
        p = self.pag
        p._ORIG_ANIMA_ATTN_OP = lambda q, k, v, *args, **kwargs: torch.zeros(
            q.shape[0], q.shape[1], q.shape[2] * q.shape[3]
        )
        p._STATE.update(
            active=1,
            attn_b0=1,
            attn_b1=2,
            attn_method="pag",
            strength=0.5,
            attn_hook_hits=0,
        )
        q = torch.zeros(2, 4, 2, 3)
        value = torch.arange(q.numel(), dtype=torch.float32).reshape_as(q)

        out = p._patched_anima_attention_op(q, q, value)

        self.assertEqual(torch.count_nonzero(out[0]).item(), 0)
        torch.testing.assert_close(out[1], value.reshape(2, 4, 6)[1] * 0.5)
        self.assertEqual(p._STATE["attn_hook_hits"], 1)

        p._STATE["strength"] = 1.0
        out = p._patched_anima_attention_op(q, q, value)
        torch.testing.assert_close(out[1], value.reshape(2, 4, 6)[1])

        p._STATE["legacy_attn"] = True
        p._STATE["strength"] = 0.5
        p._STATE["attn_hook_hits"] = 0
        out = p._patched_anima_attention_op(q, q, value)
        torch.testing.assert_close(out[1], value.reshape(2, 4, 6)[1] * 0.5)
        self.assertEqual(p._STATE["attn_hook_hits"], 1)

        p._STATE["attn_method"] = "seg"
        p._STATE["strength"] = 1.0
        p._STATE["attn_hook_hits"] = 0
        out = p._patched_anima_attention_op(q, q, value)
        expected = value.reshape(2, 4, 6)[1].mean(0, keepdim=True).expand(4, 6)
        torch.testing.assert_close(out[1], expected)

    def test_pag_head_filter_changes_only_selected_attention_head(self):
        p = self.pag
        p._ORIG_ANIMA_ATTN_OP = lambda q, k, v, *args, **kwargs: torch.zeros(
            q.shape[0], q.shape[1], q.shape[2] * q.shape[3]
        )
        p._STATE.update(
            active=1,
            attn_b0=1,
            attn_b1=2,
            attn_method="pag",
            strength=1.0,
            legacy_attn=False,
            head_spec="1",
        )
        q = torch.zeros(2, 4, 2, 3)
        value = torch.ones_like(q)

        out = p._patched_anima_attention_op(q, q, value).reshape_as(value)

        self.assertEqual(torch.count_nonzero(out[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(out[1, :, 0, :]).item(), 0)
        torch.testing.assert_close(out[1, :, 1, :], value[1, :, 1, :])

    def test_official_seg_blurs_query_on_real_anima_hw_axes(self):
        p = self.pag
        captured = {}

        def fake_attention(q, k, v, *args, **kwargs):
            captured["q"] = q.clone()
            return q.reshape(q.shape[0], -1, q.shape[-2] * q.shape[-1])

        p._ORIG_ANIMA_ATTN_OP = fake_attention
        p._STATE.update(
            active=1,
            attn_b0=1,
            attn_b1=2,
            attn_method="seg",
            legacy_attn=False,
            seg_sigma=1.0,
            strength=1.0,
            head_spec="",
            attn_spatial_shape=(1, 5, 7),
            attn_hook_hits=0,
        )
        # Actual current Forge layout: [B,S,heads,dim].
        query = torch.zeros(2, 35, 1, 1)
        query[1, 2 * 7 + 3, 0, 0] = 1.0

        p._patched_anima_attention_op(query, query, query)

        self.assertTrue(torch.equal(captured["q"][0], query[0]))
        self.assertLess(captured["q"][1, 2 * 7 + 3, 0, 0].item(), 1.0)
        self.assertGreater(captured["q"][1, 2 * 7 + 2, 0, 0].item(), 0.0)
        self.assertGreater(captured["q"][1, 1 * 7 + 3, 0, 0].item(), 0.0)
        self.assertEqual(p._STATE["attn_hook_hits"], 1)

    def test_official_seg_infinite_sigma_produces_uniform_query(self):
        p = self.pag
        query = torch.arange(
            2 * 15, dtype=torch.float32
        ).reshape(2, 15, 1, 1)

        result = p._official_seg_query(
            query, 1, 2, 10000.0, spatial_shape=(1, 3, 5)
        )

        self.assertTrue(torch.equal(result[0], query[0]))
        expected = query[1].mean(dim=0, keepdim=True).expand_as(query[1])
        torch.testing.assert_close(result[1], expected)

    def test_official_seg_strength_and_head_filter_are_applied(self):
        p = self.pag
        query = torch.zeros(2, 15, 2, 1)
        query[1, 7, :, 0] = 1.0

        result = p._official_seg_query(
            query,
            1,
            2,
            10000.0,
            spatial_shape=(1, 3, 5),
            strength=0.5,
            head_spec="1",
        )

        self.assertTrue(torch.equal(result[1, :, 0], query[1, :, 0]))
        expected = torch.lerp(
            query[1, :, 1],
            query[1, :, 1].mean(dim=0, keepdim=True).expand_as(query[1, :, 1]),
            0.5,
        )
        torch.testing.assert_close(result[1, :, 1], expected)

    def test_zero_correction_with_rescale_keeps_cfg_base_bit_identical(self):
        p = self.pag
        cond = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
        base = torch.tensor([[8.0, -4.0, 2.0, -1.0]])
        p._STATE.update(
            attn_raw=cond.clone(),
            slg_raw=None,
            rescale=0.2,
            rescale_mode="full",
        )

        out = p._apply_perturbation({"cond_denoised": cond}, base)

        self.assertTrue(torch.equal(out, base))

    def test_default_post_cfg_fast_path_is_bitwise_identity(self):
        p = self.pag
        p._STATE["on"] = False
        denoised = torch.randn(1, 4, 3, 5)

        out = p._post_cfg({"denoised": denoised})

        self.assertIs(out, denoised)

    def test_disabled_pass_clears_stale_feature_modes_and_targets(self):
        p = self.pag
        p._STATE.update(
            on=True,
            attn_method="seg",
            attn_targets={18},
            slg_on=True,
            slg_targets={18},
        )
        p._APG["on"] = True
        p._ADG["on"] = True
        process = p.AnimaSafePAG()
        request = types.SimpleNamespace(extra_generation_params={})

        process.process_before_every_sampling(request, False)

        self.assertFalse(p._STATE["on"])
        self.assertIsNone(p._STATE["attn_method"])
        self.assertEqual(p._STATE["attn_targets"], set())
        self.assertFalse(p._STATE["slg_on"])
        self.assertEqual(p._STATE["slg_targets"], set())
        self.assertFalse(p._APG["on"])
        self.assertFalse(p._ADG["on"])

    def test_effective_cfg_recovery_retains_linear_edit_strength(self):
        p = self.pag
        cond = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
        uncond = torch.tensor([[-1.0, 0.5, 2.0, -2.0]])
        effective_scale = 3.25
        incoming = uncond + effective_scale * (cond - uncond)

        recovered, fit_error = p._recover_effective_cfg(
            {
                "cond_denoised": cond,
                "uncond_denoised": uncond,
            },
            incoming,
        )

        self.assertAlmostEqual(recovered, effective_scale, places=6)
        self.assertLess(fit_error, 1e-6)

    def test_neutral_apg_reduces_to_incoming_standard_cfg(self):
        p = self.pag
        cond = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
        uncond = torch.tensor([[-1.0, 0.5, 2.0, -2.0]])
        incoming = uncond + 4.0 * (cond - uncond)
        p._CFG.update(mode="apg", experimental_stack=False)
        p._APG.update(
            on=True,
            eta=1.0,
            norm_threshold=0.0,
            momentum=0.0,
            avg=None,
            last_sigma=None,
        )

        out = p._apply_cfg_base(
            {
                "cond_denoised": cond,
                "uncond_denoised": uncond,
                "sigma": torch.tensor([1.0]),
                "model_options": {},
            },
            incoming,
        )

        torch.testing.assert_close(out, incoming)

    def _cfg_base_args(self, cond, uncond):
        return {
            "cond_denoised": cond,
            "uncond_denoised": uncond,
            "sigma": torch.tensor([1.0]),
            "model_options": {},
        }

    def _cfg_base_fixture(self):
        """CWM runs a Haar transform, so these need real 4-D latent shapes."""
        p = self.pag
        cond = (torch.arange(64, dtype=torch.float32) / 10.0 - 3.0).reshape(
            1, 4, 4, 4
        )
        uncond = torch.flip(cond, dims=[-1]) * 0.5
        p._CFG.update(
            alpha_low=0.30, alpha_high=0.15, smc_lambda=6.0, smc_k=0.20,
        )
        return cond, uncond, uncond + 4.0 * (cond - uncond)

    def test_no_base_toggle_preserves_incoming(self):
        p = self.pag
        cond, uncond, incoming = self._cfg_base_fixture()

        out = p._apply_cfg_base(self._cfg_base_args(cond, uncond), incoming)

        torch.testing.assert_close(out, incoming)

    def test_legacy_radio_resolves_to_the_same_flags(self):
        p = self.pag
        p._CFG.update(mode="smc+cwm")
        self.assertEqual(p._cfg_base_flags(), (True, False, True))
        p._CFG.update(mode="preserve", experimental_stack=True)
        self.assertEqual(p._cfg_base_flags(), (True, True, True))
        p._CFG.update(experimental_stack=False, smc_on=True, cwm_on=True)
        self.assertEqual(p._cfg_base_flags(), (True, False, True))

    def test_smc_cwm_toggles_match_legacy_combined_mode(self):
        p = self.pag
        cond, uncond, incoming = self._cfg_base_fixture()
        args = self._cfg_base_args(cond, uncond)

        p._CFG.update(mode="smc+cwm")
        p.reset_cfg_state()
        legacy = p._apply_cfg_base(args, incoming)

        p._CFG.update(mode="preserve", smc_on=True, cwm_on=True)
        p.reset_cfg_state()
        toggled = p._apply_cfg_base(args, incoming)

        self.assertFalse(torch.allclose(toggled, incoming))
        torch.testing.assert_close(toggled, legacy)

    def test_three_toggles_match_the_experimental_stack(self):
        p = self.pag
        cond, uncond, incoming = self._cfg_base_fixture()
        args = self._cfg_base_args(cond, uncond)
        apg_neutral = dict(
            on=True, eta=1.0, norm_threshold=0.0, momentum=0.0,
            avg=None, last_sigma=None,
        )

        p._APG.update(**apg_neutral)
        p._CFG.update(experimental_stack=True)
        p.reset_cfg_state()
        stacked = p._apply_cfg_base(args, incoming)

        p._APG.update(**apg_neutral)
        p._CFG.update(
            experimental_stack=False, smc_on=True, apg_on=True, cwm_on=True,
        )
        p.reset_cfg_state()
        toggled = p._apply_cfg_base(args, incoming)

        torch.testing.assert_close(toggled, stacked)

    def test_apg_and_cwm_combine_without_the_legacy_stack(self):
        """The old radio could not express APG + CWM at once."""
        p = self.pag
        cond, uncond, incoming = self._cfg_base_fixture()
        args = self._cfg_base_args(cond, uncond)
        p._APG.update(
            on=True, eta=1.0, norm_threshold=0.0, momentum=0.0,
            avg=None, last_sigma=None,
        )

        p._CFG.update(apg_on=True)
        p.reset_cfg_state()
        apg_only = p._apply_cfg_base(args, incoming)

        p._APG.update(avg=None, last_sigma=None)
        p._CFG.update(apg_on=True, cwm_on=True)
        p.reset_cfg_state()
        apg_and_cwm = p._apply_cfg_base(args, incoming)

        # Neutral APG reduces to standard CFG; CWM must then reshape it.
        torch.testing.assert_close(apg_only, incoming)
        self.assertFalse(torch.allclose(apg_and_cwm, apg_only))

    def test_adaptive_skip_flushes_apg_and_smc_state(self):
        p = self.pag
        p._STATE["on"] = False
        p._ADG.update(on=True, start=0.0, interval=0)
        p._APG.update(
            on=True,
            avg=torch.ones(1, 1),
            last_sigma=1.0,
        )
        p._RUNTIME.smc_prev = torch.ones(1, 1)

        p._model_wrapper(
            self._fake_apply_model,
            {
                "input": torch.zeros(2, 1),
                "timestep": torch.ones(2),
                "c": {"bias": torch.tensor([[0.0], [1.0]])},
                "cond_or_uncond": [1, 0],
            },
        )

        self.assertIsNone(p._APG["avg"])
        self.assertIsNone(p._APG["last_sigma"])
        self.assertIsNone(p._RUNTIME.smc_prev)

    def test_authoritative_step_clock_ignores_multiple_wrapper_calls(self):
        p = self.pag
        old_shared = p.shared
        p.shared = types.SimpleNamespace(
            state=types.SimpleNamespace(sampling_step=7, sampling_steps=20)
        )
        try:
            for marker in ([1], [0]):
                p._model_wrapper(
                    self._fake_apply_model,
                    {
                        "input": torch.zeros(1, 1),
                        "timestep": torch.ones(1),
                        "c": {"bias": torch.ones(1, 1)},
                        "cond_or_uncond": marker,
                    },
                )
            self.assertEqual(p._STATE["step"], 7)
            self.assertAlmostEqual(p._pct_now(), 7 / 19)
        finally:
            p.shared = old_shared

    def test_dcw_failure_keeps_already_applied_perturbation(self):
        p = self.pag
        cond = torch.tensor([[2.0, -1.0]])
        weak = torch.tensor([[1.5, -0.5]])
        base = torch.tensor([[7.0, 3.0]])
        p._STATE.update(
            on=True,
            attn_raw=weak,
            attn_scale=2.0,
            rescale=0.0,
        )
        p._DCW["on"] = True
        original_apply_dcw = p.apply_dcw

        def fail_dcw(*_args, **_kwargs):
            raise ValueError("shape mismatch")

        p.apply_dcw = fail_dcw
        try:
            out = p._post_cfg(
                {
                    "denoised": base,
                    "cond_denoised": cond,
                    "uncond_denoised": torch.zeros_like(cond),
                    "input": torch.zeros_like(base),
                    "sigma": torch.tensor([1.0]),
                }
            )
        finally:
            p.apply_dcw = original_apply_dcw

        torch.testing.assert_close(out, base + 2.0 * (cond - weak))

    def test_rescale_scales_only_guidance_not_cfg_base(self):
        p = self.pag
        cond = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
        weak = torch.tensor([[0.5, -0.5, 1.0, -1.5]])
        base = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
        scale = 2.0
        rescale = 0.2
        p._STATE.update(
            attn_scale=scale,
            attn_raw=weak,
            slg_raw=None,
            rescale=rescale,
            rescale_mode="full",
        )

        raw_guidance = scale * (cond - weak)
        guided = base + raw_guidance
        dims = list(range(1, guided.ndim))
        std_cond = cond.std(dim=dims, keepdim=True).clamp_min(1e-6)
        std_guided = guided.std(dim=dims, keepdim=True).clamp_min(1e-6)
        factor = rescale * (std_cond / std_guided) + (1.0 - rescale)
        expected = base + raw_guidance * factor

        out = p._apply_perturbation({"cond_denoised": cond}, base)

        torch.testing.assert_close(out, expected)
        self.assertFalse(torch.equal(out, guided * factor))

    def test_partial_rescale_uses_conditional_prediction_for_std_source(self):
        p = self.pag
        cond = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
        weak = torch.tensor([[0.5, -0.5, 1.0, -1.5]])
        base = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
        scale = 2.0
        rescale = 0.2
        raw_guidance = scale * (cond - weak)
        guided = cond + raw_guidance
        dims = list(range(1, guided.ndim))
        factor = rescale * (
            cond.std(dim=dims, keepdim=True).clamp_min(1e-6)
            / guided.std(dim=dims, keepdim=True).clamp_min(1e-6)
        ) + (1.0 - rescale)
        p._STATE.update(
            attn_scale=scale,
            attn_raw=weak,
            slg_raw=None,
            rescale=rescale,
            rescale_mode="partial",
        )

        out = p._apply_perturbation({"cond_denoised": cond}, base)

        torch.testing.assert_close(out, base + raw_guidance * factor)

    def test_head_parser_empty_means_all_and_supports_ranges(self):
        self.assertEqual(self.pag._parse_attention_heads("", 4), {0, 1, 2, 3})
        self.assertEqual(self.pag._parse_attention_heads("0,2-4,99", 5), {0, 2, 3, 4})

    def test_guidance_ui_exposes_upstream_pag_controls_and_primary_sections(self):
        source = (ROOT / "scripts" / "anima_safe_pag.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("anima_safe_pag_official_strength", source)
        self.assertIn("anima_safe_pag_heads", source)
        self.assertIn("anima_safe_pag_rescale_mode", source)
        self.assertIn(
            "official_strength, head_indices, rescale_mode",
            source,
        )
        self.assertNotIn(
            'with gr.Accordion("DCW (post-CFG wavelet correction)"',
            source,
        )
        self.assertNotIn(
            'with gr.Accordion("DAVE (Anima diversity',
            source,
        )
        self.assertNotIn(
            'with gr.Accordion("CNS-inspired Wavelet Noise',
            source,
        )

    def test_guidance_ui_exposes_per_control_adjustment_hints(self):
        source = (ROOT / "scripts" / "anima_safe_pag.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "이미지가 찢어지거나 배경·구도가 과하게 변하면 이 값을 먼저 낮추세요.",
            source,
        )
        self.assertIn(
            "초반 구도·인물 배치가 무너지면 값을 올려 더 늦게 시작하세요.",
            source,
        )
        self.assertIn(
            "과채도·과대비면 올리고, 색이 탁하거나 대비가 눌리면 낮추거나 0으로 비교하세요.",
            source,
        )
        self.assertIn(
            "품질 손실이 보이면 올려 더 늦게 생략하고, 속도 우선이면 낮추세요.",
            source,
        )
        self.assertIn(
            "색 노이즈·거친 입자·구조 변형이 과하면 먼저 낮추세요.",
            source,
        )
        self.assertGreaterEqual(source.count("info="), 30)

    def test_empty_block_spec_uses_upstream_safe_default(self):
        self.assertEqual(self.pag._parse_blocks("", 28), {18})

    def test_extra_generation_param_cleanup_is_scoped_to_guidance_keys(self):
        p = types.SimpleNamespace(
            extra_generation_params={
                "Anima Perturbation Guidance": "PAG",
                "Anima APG": "APG",
                "Anima Adaptive Guidance": "AdaptiveG",
                "Steps": 20,
                "Unrelated Extension": "keep me",
            }
        )

        self.pag._clear_extra_generation_params(p)

        self.assertEqual(
            p.extra_generation_params,
            {
                "Steps": 20,
                "Unrelated Extension": "keep me",
            },
        )
if __name__ == "__main__":
    unittest.main()
