from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import gradio as gr
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


def _load_forge_sampler():
    """Execute Forge's real batching/area math with only host services stubbed."""
    forge = ROOT.parents[1]
    spec = importlib.util.spec_from_file_location(
        "_pag_test_conditions", forge / "backend" / "sampling" / "condition.py"
    )
    conditions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conditions)
    memory = types.SimpleNamespace(signal_empty_cache=False, get_free_memory=lambda _device: 100)
    backend = types.ModuleType("backend")
    backend.memory_management = memory
    backend.utils = types.SimpleNamespace()
    fake_args = types.ModuleType("backend.args")
    fake_args.args = types.SimpleNamespace(disable_gpu_warning=True)
    fake_args.dynamic_args = types.SimpleNamespace(context_handler=None)
    with mock.patch.dict(sys.modules, {
        "backend": backend, "backend.args": fake_args,
        "backend.sampling.condition": conditions,
    }):
        spec = importlib.util.spec_from_file_location(
            "_pag_test_sampler", forge / "backend" / "sampling" / "sampling_function.py"
        )
        sampler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sampler)
    return sampler, conditions.Condition, memory


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
            cond_raw=None,
            pert_oom_owner=None,
            pass_owner=None,
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
            cfg1_warned=False,
            base_skipped=False,
            fit_checked=False,
        )
        p._DCW.update(
            on=False,
            dcw_on=False,
            rdc_on=False,
            steps=0,
            dcw_steps=0,
            rdc_steps=0,
        )
        p._DAVE.update(on=False, targets=set(), steps=0)
        p._CNS.update(on=False, warned=False)
        p._MOD.update(
            on=False,
            targets=set(),
            block_modulations=None,
            typed={},
            hits=0,
            warned=False,
        )
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

    def test_weighted_and_regional_guidance_survives_real_forge_aggregation(self):
        sampler, Condition, memory = _load_forge_sampler()
        p = self.pag

        class Model:
            calls = 0
            @staticmethod
            def memory_required(shape):
                return shape[0]

            @staticmethod
            def apply_model(x, timestep, *, bias, transformer_options):
                Model.calls += 1
                out = bias.expand_as(x).clone()
                a0, a1 = p._STATE["attn_b0"], p._STATE["attn_b1"]
                if a0 is not None:
                    out[a0:a1] += bias[a0:a1] * 0.5 + 1.5
                return out

        for layout in ("full", "region", "mask"):
            for free_memory in (100, 0.5):
                with self.subTest(layout=layout, microbatch=free_memory < 1):
                    self.setUp()
                    Model.calls = 0
                    memory.get_free_memory = lambda _device: free_memory
                    x = torch.zeros(1, 1, 4, 8)
                    cond = [
                        {"model_conds": {"bias": Condition(torch.tensor([[[[1.]]]]))}, "strength": 1.0},
                        {"model_conds": {"bias": Condition(torch.tensor([[[[5.]]]]))}, "strength": 3.0},
                    ]
                    if layout == "region":
                        cond[1].update(area=(4, 4, 0, 4), mask=torch.ones(1, 4, 8))
                    elif layout == "mask":
                        mask = torch.zeros(1, 4, 8)
                        mask[..., 4:] = 0.5
                        cond[1].update(mask=mask, mask_strength=0.5)
                    uncond = [{"model_conds": {"bias": Condition(torch.zeros(1, 1, 1, 1))}}]
                    options = {
                        "model_function_wrapper": p._model_wrapper,
                        "sampler_post_cfg_function": [p._post_cfg],
                        "sampler_pre_cfg_function": [getattr(p, "_prepare_condition_aggregation", lambda *args: args)],
                    }
                    with mock.patch.dict(sys.modules, {"backend.sampling.sampling_function": sampler}):
                        actual = sampler.sampling_function_inner(
                            Model(), x, torch.ones(1), uncond, cond, 7.0, options,
                        )
                    # Official weighted CFG: edit_strength=4; aggregated
                    # cond=4, weak=7.5 => 4*7*4 + 4*(4-7.5) = 98.
                    expected = torch.full_like(x, 98.0)
                    if layout != "full":
                        # Left has only cond1, weak3: 1*7*4 + 4*(1-3)=20.
                        expected[..., :4] = 20.0
                    if layout == "mask":
                        # Right condition2 has 3 * .5 * .5 weight:
                        # cond=19/7, weak=39/7, result=28*19/7-80/7.
                        expected[..., 4:] = 452.0 / 7.0
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(Model.calls, 3 if free_memory < 1 else (2 if layout == "region" else 1))
                    self.assertNotIn("condition_aggregation", p._STATE)

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
        p._MOD.update(
            on=True,
            targets={0},
            block_modulations=torch.ones(1, 6),
            typed={(0, "cpu", "torch.float32"): torch.ones(6)},
            hits=3,
        )
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
        self.assertFalse(p._MOD["on"])
        self.assertEqual(p._MOD["targets"], set())
        self.assertIsNone(p._MOD["block_modulations"])
        self.assertEqual(p._MOD["typed"], {})

    def test_postprocess_releases_modulation_and_weak_tensors(self):
        p = self.pag
        p._STATE.update(
            attn_raw=torch.ones(1),
            slg_raw=torch.ones(1),
            cond_raw=torch.ones(1),
            requested_modulation=True,
        )
        p._MOD.update(
            on=True,
            targets={0},
            block_modulations=torch.ones(1, 6),
            typed={(0, "cpu", "torch.float32"): torch.ones(6)},
            hits=2,
        )

        p.AnimaSafePAG().postprocess(None, None)

        self.assertFalse(p._MOD["on"])
        self.assertEqual(p._MOD["targets"], set())
        self.assertIsNone(p._MOD["block_modulations"])
        self.assertEqual(p._MOD["typed"], {})
        self.assertEqual(p._MOD["hits"], 0)
        self.assertIsNone(p._STATE["attn_raw"])
        self.assertIsNone(p._STATE["slg_raw"])
        self.assertIsNone(p._STATE["cond_raw"])
        self.assertFalse(p._STATE["requested_modulation"])

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

    def _cfg_one_args(self, cond, uncond=None, cond_scale=1.0):
        """Forge runs no uncond at cond_scale=1, so the post-CFG hook sees an
        all-zero ``uncond_denoised`` and ``denoised == cond``."""
        args = self._cfg_base_args(
            cond, torch.zeros_like(cond) if uncond is None else uncond
        )
        if cond_scale is not None:
            args["cond_scale"] = cond_scale
        return args

    def test_cfg_one_skips_every_base_override_and_warns_once(self):
        """M16: at CFG 1 SMC/APG/CWM have no CFG error to work on. APG with
        eta=0 would otherwise zero the whole output; CWM/SMC are not neutral
        either. The incoming result must be kept and the skip logged once."""
        p = self.pag
        cond, _uncond, _incoming = self._cfg_base_fixture()
        incoming = cond.clone()  # uncond + 1.0 * (cond - uncond), uncond == 0
        apg_zero_eta = dict(
            on=True, eta=0.0, norm_threshold=15.0, momentum=0.0,
            avg=None, last_sigma=None,
        )
        for toggles in (
            dict(apg_on=True),
            dict(cwm_on=True),
            dict(smc_on=True),
            dict(smc_on=True, apg_on=True, cwm_on=True),
            dict(mode="apg"),
            dict(experimental_stack=True),
        ):
            with self.subTest(toggles=toggles):
                self.setUp()
                p._APG.update(**apg_zero_eta)
                p._CFG.update(**toggles)
                p.reset_cfg_state()
                with mock.patch.object(p, "_log") as log:
                    first = p._apply_cfg_base(self._cfg_one_args(cond), incoming)
                    second = p._apply_cfg_base(self._cfg_one_args(cond), incoming)
                torch.testing.assert_close(first, incoming)
                torch.testing.assert_close(second, incoming)
                self.assertEqual(p._CFG["steps"], 0)
                self.assertEqual(p._STATE["apg_steps"], 0)
                self.assertIsNone(p._RUNTIME.smc_prev)
                self.assertEqual(log.call_count, 1)
                self.assertIn("cond_scale", log.call_args.args[0])
                self.assertTrue(p._CFG["cfg1_warned"])

    def test_cfg_one_guard_triggers_on_scale_or_zero_uncond_alone(self):
        p = self.pag
        cond, uncond, incoming = self._cfg_base_fixture()
        p._CFG.update(cwm_on=True)

        # cond_scale=1 with a real uncond (disable_cfg1_optimization=True).
        p.reset_cfg_state()
        out = p._apply_cfg_base(self._cfg_one_args(cond, uncond), cond)
        torch.testing.assert_close(out, cond)

        # All-zero uncond without any cond_scale key (older/custom callers).
        p.reset_cfg_state()
        out = p._apply_cfg_base(
            self._cfg_one_args(cond, cond_scale=None), cond
        )
        torch.testing.assert_close(out, cond)

        # A real CFG>1 result with the cond_scale key still gets the override.
        p.reset_cfg_state()
        out = p._apply_cfg_base(
            self._cfg_one_args(cond, uncond, cond_scale=4.0), incoming
        )
        self.assertFalse(torch.allclose(out, incoming))
        self.assertEqual(p._CFG["steps"], 1)

    def test_cfg_one_post_cfg_keeps_incoming_with_apg_eta_zero(self):
        """Audit repro: APG eta=0 at CFG 1 drove |out|max to ~5e-7."""
        p = self.pag
        p._STATE["on"] = False
        torch.manual_seed(0)
        cond = torch.randn(1, 4, 8, 8)
        p._CFG.update(apg_on=True)
        p._APG.update(
            on=True, eta=0.0, norm_threshold=15.0, momentum=0.0,
            avg=None, last_sigma=None,
        )
        args = self._cfg_one_args(cond)
        args.update(denoised=cond.clone(), input=torch.randn_like(cond))

        out = p._post_cfg(args)

        torch.testing.assert_close(out, cond)
        self.assertEqual(p._CFG["steps"], 0)

    def test_cfg_one_base_guard_leaves_pag_untouched(self):
        """PAG/SEG/SLG never needed an uncond; CFG 1 + base toggle must still
        add the perturbation term exactly as with the toggle off."""
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
        p._CFG.update(apg_on=True, cwm_on=True)
        p._APG.update(
            on=True, eta=0.0, norm_threshold=15.0, momentum=0.0,
            avg=None, last_sigma=None,
        )

        self.assertEqual(self._post_cfg().item(), -0.5)
        self.assertEqual(p._STATE["applied_steps"], 1)
        self.assertEqual(p._CFG["steps"], 0)

    def test_cfg_one_warning_rearms_per_generation(self):
        p = self.pag
        p._CFG["cfg1_warned"] = True
        process = p.AnimaSafePAG()
        request = types.SimpleNamespace(extra_generation_params={})

        process.process_before_every_sampling(request, False)
        self.assertFalse(p._CFG["cfg1_warned"])

        p._CFG["cfg1_warned"] = True
        process.postprocess(None, None)
        self.assertFalse(p._CFG["cfg1_warned"])

    # --- 후속: attention 폴백 재시도 위치, CFG 1 rescale 자동 끄기, 경고 문구 ---

    def test_attention_op_fallback_retry_runs_after_exception_is_released(self):
        """재시도 forward 는 except 블록 밖에서 돌아야 한다. except 안이면 살아
        있는 예외의 traceback 이 실패한 호출의 활성값을 붙잡은 채 두 번째
        attention 이 돌아 VRAM 이 이중으로 든다. 결과(원본 op 출력)는 같다."""
        p = self.pag

        class Activation:
            pass

        for method, spatial in (("pag", None), ("seg", (1, 3, 3))):
            with self.subTest(method=method):
                self.setUp()
                seen = {}
                calls = []

                def flaky(q, k, v, *args, **kwargs):
                    calls.append(q)
                    if method == "pag" and len(calls) == 1:
                        activation = Activation()  # 실패한 op 의 중간 활성값
                        seen["ref"] = weakref.ref(activation)
                        raise RuntimeError("attention boom")
                    seen["exc"] = sys.exc_info()[0]
                    ref = seen.get("ref")
                    seen["alive"] = ref is not None and ref() is not None
                    return torch.full(
                        (q.shape[0], q.shape[1], q.shape[2] * q.shape[3]), 3.0
                    )

                p._ORIG_ANIMA_ATTN_OP = flaky
                # seg: spatial 이 seq(4) 와 안 맞아 _official_seg_query 가 먼저 실패.
                p._STATE.update(
                    active=1, attn_b0=1, attn_b1=2, attn_method=method,
                    legacy_attn=False, strength=1.0, head_spec="",
                    attn_spatial_shape=spatial, attn_hook_hits=0,
                )
                q = torch.zeros(2, 4, 2, 3)
                value = torch.ones_like(q)

                with mock.patch.object(p, "_log") as log:
                    out = p._patched_anima_attention_op(q, q, value)

                torch.testing.assert_close(out, torch.full((2, 4, 6), 3.0))
                self.assertIs(calls[-1], q)  # 재시도는 원래 query 로
                self.assertIsNone(seen["exc"])
                self.assertFalse(seen["alive"])
                self.assertEqual(p._STATE["attn_hook_hits"], 0)
                self.assertEqual(log.call_count, 1)
                self.assertIn("attention perturb skipped", log.call_args.args[0])

    def test_attention_op_failure_after_output_returns_output_without_retry(self):
        p = self.pag
        calls = []

        def op(q, k, v, *args, **kwargs):
            calls.append(q)
            return torch.full(
                (q.shape[0], q.shape[1], q.shape[2] * q.shape[3]), 5.0
            )

        p._ORIG_ANIMA_ATTN_OP = op
        p._STATE.update(
            active=1, attn_b0=1, attn_b1=2, attn_method="pag",
            legacy_attn=False, strength=1.0, head_spec="",
        )
        q = torch.zeros(2, 4, 2, 3)
        value = torch.ones_like(q)
        with mock.patch.object(
            p, "_head_index", side_effect=RuntimeError("late boom")
        ), mock.patch.object(p, "_log"):
            out = p._patched_anima_attention_op(q, q, value)

        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(out, torch.full((2, 4, 6), 5.0))

    def _apg_rescale_post_cfg(self, *, apg, cond_scale, rescale, keep_scale=True):
        """PAG rescale + (선택) APG 자동 끄기로 _post_cfg 한 번."""
        p = self.pag
        torch.manual_seed(0)
        cond = torch.randn(1, 4, 8, 8)
        weak = torch.randn_like(cond)
        if cond_scale == 1.0:
            uncond = torch.zeros_like(cond)  # Forge 는 CFG 1 에서 uncond 를 안 돌린다
        else:
            uncond = torch.randn_like(cond) * 0.3
        denoised = uncond + cond_scale * (cond - uncond)
        p._STATE.update(
            on=True, attn_scale=4.0, rescale=rescale, rescale_mode="full",
            apg_autooff_rescale=True, attn_raw=weak.clone(),
            cond_raw=cond.clone(), slg_raw=None, applied_steps=0,
        )
        if apg:
            p._CFG["apg_on"] = True
            p._APG.update(
                on=True, eta=0.0, norm_threshold=15.0, momentum=0.0,
                avg=None, last_sigma=None,
            )
        else:
            p._CFG["apg_on"] = False
            p._APG["on"] = False
        args = {
            "denoised": denoised.clone(),
            "cond_denoised": cond.clone(),
            "uncond_denoised": uncond.clone(),
            "sigma": torch.tensor([1.0]),
            "model_options": {},
            "input": torch.randn_like(cond),
        }
        if keep_scale:
            args["cond_scale"] = cond_scale
        return p._post_cfg(args)

    def test_cfg_one_apg_skip_does_not_auto_off_pag_rescale(self):
        """CFG 1 에서 APG 가 건너뛰어지면 rescale 자동 끄기도 적용하지 않는다:
        APG 를 켠 생성과 끈 생성이 비트 단위로 같아야 한다."""
        for keep_scale in (True, False):
            with self.subTest(cond_scale_key=keep_scale):
                self.setUp()
                with mock.patch.object(self.pag, "_log"):
                    with_apg = self._apg_rescale_post_cfg(
                        apg=True, cond_scale=1.0, rescale=0.7,
                        keep_scale=keep_scale,
                    )
                self.assertEqual(self.pag._CFG["steps"], 0)
                self.setUp()
                without_apg = self._apg_rescale_post_cfg(
                    apg=False, cond_scale=1.0, rescale=0.7,
                    keep_scale=keep_scale,
                )
                self.setUp()
                unscaled = self._apg_rescale_post_cfg(
                    apg=False, cond_scale=1.0, rescale=0.0,
                    keep_scale=keep_scale,
                )
                self.assertTrue(torch.equal(with_apg, without_apg))
                self.assertFalse(torch.allclose(with_apg, unscaled))

    def test_apg_auto_off_rescale_unchanged_above_cfg_one(self):
        """CFG>1 경로: APG 가 돌면 rescale 이 여전히 자동으로 꺼진다(비트 동일).
        앞선 CFG 1 스텝의 건너뜀 표시가 다음 스텝에 남지 않아야 한다."""
        p = self.pag
        with mock.patch.object(p, "_log"):
            self._apg_rescale_post_cfg(apg=True, cond_scale=1.0, rescale=0.7)
            rescaled = self._apg_rescale_post_cfg(
                apg=True, cond_scale=4.0, rescale=0.7
            )
            self.setUp()
            reference = self._apg_rescale_post_cfg(
                apg=True, cond_scale=4.0, rescale=0.0
            )
        self.assertTrue(torch.equal(rescaled, reference))
        self.setUp()
        with mock.patch.object(p, "_log"):
            no_apg = self._apg_rescale_post_cfg(
                apg=False, cond_scale=4.0, rescale=0.7
            )
            self.setUp()
            no_apg_unscaled = self._apg_rescale_post_cfg(
                apg=False, cond_scale=4.0, rescale=0.0
            )
        self.assertFalse(torch.allclose(no_apg, no_apg_unscaled))

    def test_cfg_one_warning_text_matches_disable_cfg1_optimization(self):
        p = self.pag
        cond, uncond, _incoming = self._cfg_base_fixture()

        def warning(args):
            p._CFG["cfg1_warned"] = False
            with mock.patch.object(p, "_log") as log:
                out = p._apply_cfg_base(args, cond)
            torch.testing.assert_close(out, cond)
            self.assertEqual(log.call_count, 1)
            return log.call_args.args[0]

        p._CFG.update(cwm_on=True)
        # 기본 Forge: CFG 1 에서 uncond 패스를 건너뛴다.
        default = warning(self._cfg_one_args(cond))
        self.assertIn("no uncond", default)
        self.assertNotIn("disable_cfg1_optimization", default)

        # disable_cfg1_optimization=True: uncond 는 계산됐지만 배율이 1 이다.
        args = self._cfg_one_args(cond, uncond)
        args["model_options"] = {"disable_cfg1_optimization": True}
        forced = warning(args)
        self.assertIn("cond_scale=1", forced)
        self.assertIn("disable_cfg1_optimization", forced)
        self.assertNotIn("no uncond", forced)
        self.assertNotIn("rescale", forced)

    def test_cfg_one_warning_mentions_kept_pag_rescale(self):
        p = self.pag
        cond, _uncond, _incoming = self._cfg_base_fixture()
        p._CFG.update(apg_on=True)
        p._APG.update(
            on=True, eta=0.0, norm_threshold=15.0, momentum=0.0,
            avg=None, last_sigma=None,
        )
        p._STATE.update(on=True, rescale=0.5, apg_autooff_rescale=True)
        with mock.patch.object(p, "_log") as log:
            p._apply_cfg_base(self._cfg_one_args(cond), cond)
        self.assertIn("rescale", log.call_args.args[0])

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
        p._DCW.update(on=True, dcw_on=True, rdc_on=False)
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

    def test_perturbation_ignores_upstream_rewrites_of_cond_denoised(self):
        """A prior post-CFG hook may rewrite Forge's prediction tensors.

        PAG must keep differencing the weak prediction against the cond it was
        derived from, otherwise the rewrite leaks a constant offset into
        scale*(cond - weak) and dilutes both scale and strength."""
        p = self.pag
        cond = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
        weak = torch.tensor([[0.5, -1.0, 0.25, 1.5]])
        base = torch.zeros_like(cond)
        p._STATE.update(
            attn_raw=weak, slg_raw=None, cond_raw=cond,
            attn_scale=4.0, rescale=0.0,
        )

        clean = p._apply_perturbation({"cond_denoised": cond}, base)
        # Emulate Skimmed CFG rewriting the shared prediction tensor.
        rewritten = cond + 0.75
        skimmed = p._apply_perturbation({"cond_denoised": rewritten}, base)

        torch.testing.assert_close(skimmed, clean)
        torch.testing.assert_close(clean, 4.0 * (cond - weak))

    def test_perturbation_falls_back_when_no_cond_was_captured(self):
        p = self.pag
        cond = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
        weak = torch.tensor([[0.5, -1.0, 0.25, 1.5]])
        base = torch.zeros_like(cond)
        p._STATE.update(
            attn_raw=weak, slg_raw=None, cond_raw=None,
            attn_scale=2.0, rescale=0.0,
        )

        out = p._apply_perturbation({"cond_denoised": cond}, base)

        torch.testing.assert_close(out, 2.0 * (cond - weak))

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
        self.assertIn("anima_mod_guidance_enable", source)
        self.assertIn("anima_mod_guidance_clip_model", source)
        self.assertIn(
            "mod_enabled, mod_clip_model, mod_weight",
            source,
        )
        self.assertIn("anima_guidance_smc_preset", source)
        self.assertIn("anima_guidance_smc_master_enable", source)
        self.assertIn("anima_guidance_rdc_enable", source)
        self.assertIn('"[Anima SMC] Preset"', source)
        self.assertIn('_arg(56, "Off")', source)
        self.assertIn(
            "mod_adapter_mode, mod_adapter_path,\n"
            "            # Appended after v0.20 to preserve all 56 older argument indexes.\n"
            "            smc_preset,",
            source,
        )
        self.assertLess(
            source.index('gr.Markdown("#### DCW — post-CFG'),
            source.index('gr.Markdown("#### CWM — CFG wavelet'),
        )
        self.assertLess(
            source.index('gr.Markdown("#### CWM — CFG wavelet'),
            source.index('gr.Markdown("#### SMC — sliding-mode'),
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

    def test_guidance_ui_builds_with_append_only_smc_and_rdc_arguments(self):
        with gr.Blocks():
            inputs = self.pag.AnimaSafePAG().ui(False)

        self.assertEqual(len(inputs), 62)
        self.assertEqual(inputs[26].elem_id, "anima_guidance_smc_lambda")
        self.assertEqual(inputs[26].maximum, 30.0)
        self.assertEqual(inputs[27].elem_id, "anima_guidance_smc_k")
        self.assertEqual(inputs[27].maximum, 5.0)
        self.assertEqual(inputs[42].elem_id, "anima_guidance_smc_enable")
        self.assertEqual(inputs[56].elem_id, "anima_guidance_smc_preset")
        self.assertEqual(inputs[56].value, "Auto")
        self.assertEqual(
            inputs[57].elem_id, "anima_guidance_smc_master_enable"
        )
        self.assertFalse(inputs[57].value)
        self.assertEqual(inputs[58].elem_id, "anima_guidance_rdc_enable")
        self.assertEqual(inputs[59].elem_id, "anima_guidance_rdc_tau")
        self.assertEqual(inputs[60].elem_id, "anima_guidance_rdc_alpha_ll")
        self.assertEqual(inputs[61].elem_id, "anima_guidance_rdc_alpha_hh")

    def test_xyz_axis_labels_are_append_only(self):
        """xyz_grid stores a chosen axis by its integer index in axis_options.

        Reordering the list therefore repoints every saved grid at a different
        control, exactly like shuffling the script-arg list would. The v0.20
        order must stay an unbroken prefix, with new axes appended.
        """
        registered = []

        class FakeAxisOption:
            def __init__(self, label, *args, **kwargs):
                self.label = label

        fake_xyz = types.SimpleNamespace(
            AxisOption=FakeAxisOption,
            axis_options=registered,
        )
        fake_entry = types.SimpleNamespace(
            script_class=types.SimpleNamespace(__module__="xyz_grid.py"),
            module=fake_xyz,
        )
        original = self.pag.scripts.scripts_data
        self.pag.scripts.scripts_data = [fake_entry]
        try:
            self.pag._make_pag_xyz_axis()
        finally:
            self.pag.scripts.scripts_data = original

        labels = [option.label for option in registered]
        expected_prefix = [
            "[Anima Pert] Enable",
            "[Anima Pert] Attn Method",
            "[Anima Pert] Attn Scale",
            "[Anima Pert] Perturbation Strength",
            "[Anima Pert] Legacy Soft/Approx",
            "[Anima Pert] Legacy Perturbation Strength",
            "[Anima Pert] SEG Blur Sigma",
            "[Anima Pert] Attn Block Indices",
            "[Anima Pert] Attn Head Indices",
            "[Anima Pert] SLG Enable",
            "[Anima Pert] SLG Scale",
            "[Anima Pert] SLG Block Indices",
            "[Anima Pert] Start Percent",
            "[Anima Pert] End Percent",
            "[Anima Pert] Rescale",
            "[Anima Pert] Rescale Mode",
            "[Anima APG] Enable",
            "[Anima APG] Eta",
            "[Anima APG] Norm Threshold",
            "[Anima APG] Momentum",
            "[Anima AdaptiveG] Enable",
            "[Anima AdaptiveG] Skip After",
            "[Anima AdaptiveG] Keep Every",
            "[Anima CFG] Base Mode",
            "[Anima CFG] Experimental Stack",
            "[Anima CWM] Enable",
            "[Anima CWM] Alpha Low",
            "[Anima CWM] Alpha High",
            "[Anima SMC] Enable",
            "[Anima SMC] Lambda",
            "[Anima SMC] K",
            "[Anima DCW] Enable",
            "[Anima DCW] Lambda Low",
            "[Anima DCW] Lambda High",
            "[Anima DAVE] Enable",
            "[Anima DAVE] Strength",
            "[Anima DAVE] Tau",
            "[Anima DAVE] Block Indices",
            "[Anima CNS] Enable",
            "[Anima CNS] Strength",
            "[Anima CNS] Gamma Power",
            "[Anima CNS] Gamma Scale",
            "[Anima Mod] Enable",
            "[Anima Mod] Direction Weight",
            "[Anima Mod] Start Block",
            "[Anima Mod] End Block",
        ]
        self.assertEqual(labels[:len(expected_prefix)], expected_prefix)
        self.assertEqual(
            labels[len(expected_prefix):],
            [
                "[Anima SMC] Preset",
                "[Anima RDC] Enable",
                "[Anima RDC] Tau",
                "[Anima RDC] Alpha LL",
                "[Anima RDC] Alpha HH",
            ],
        )

    def test_smc_auto_resolves_on_real_process_argument_path(self):
        class DummyUnet:
            def __init__(self):
                self.post_cfg = None

            def clone(self):
                return DummyUnet()

            def set_model_sampler_post_cfg_function(self, function):
                self.post_cfg = function

        Anima = type("Anima", (), {})
        model = Anima()
        model.forge_objects = types.SimpleNamespace(unet=DummyUnet())
        request = types.SimpleNamespace(
            sd_model=model,
            extra_generation_params={},
            steps=20,
        )
        process = self.pag.AnimaSafePAG()
        with gr.Blocks():
            inputs = process.ui(False)
        args = [component.value for component in inputs]
        args[56] = "Auto"
        args[57] = True

        process.process_before_every_sampling(request, *args)

        self.assertTrue(self.pag._CFG["smc_on"])
        self.assertEqual(self.pag._CFG["smc_preset"], "Auto")
        self.assertEqual(
            self.pag._CFG["smc_resolved_preset"], "Cosmos / Wan"
        )
        self.assertEqual(
            (self.pag._CFG["smc_lambda"], self.pag._CFG["smc_k"]),
            (6.0, 0.20),
        )
        self.assertIsNotNone(request.sd_model.forge_objects.unet.post_cfg)
        self.assertIn(
            "smc=Auto→Cosmos / Wan(6,0.2)",
            request.extra_generation_params["Anima CFG Orchestrator"],
        )

    def test_new_smc_master_toggle_can_keep_a_selected_preset_off(self):
        Anima = type("Anima", (), {})
        model = Anima()
        model.forge_objects = types.SimpleNamespace(unet=types.SimpleNamespace())
        request = types.SimpleNamespace(
            sd_model=model,
            extra_generation_params={},
            steps=20,
        )
        process = self.pag.AnimaSafePAG()
        with gr.Blocks():
            inputs = process.ui(False)
        args = [component.value for component in inputs]
        args[56] = "Auto"
        args[57] = False

        process.process_before_every_sampling(request, *args)

        self.assertFalse(self.pag._CFG["smc_on"])
        self.assertNotIn("Anima CFG Orchestrator", request.extra_generation_params)

    def test_rdc_can_run_without_instantaneous_dcw_correction(self):
        class DummyUnet:
            def __init__(self):
                self.post_cfg = None

            def clone(self):
                return DummyUnet()

            def set_model_sampler_post_cfg_function(self, function):
                self.post_cfg = function

        Anima = type("Anima", (), {})
        model = Anima()
        model.forge_objects = types.SimpleNamespace(unet=DummyUnet())
        request = types.SimpleNamespace(
            sd_model=model,
            extra_generation_params={},
            steps=20,
        )
        process = self.pag.AnimaSafePAG()
        with gr.Blocks():
            inputs = process.ui(False)
        args = [component.value for component in inputs]
        args[28] = False
        args[58] = True
        args[59] = 0.15
        args[60] = 0.04
        args[61] = 0.0

        process.process_before_every_sampling(request, *args)

        self.assertTrue(self.pag._DCW["on"])
        self.assertFalse(self.pag._DCW["dcw_on"])
        self.assertTrue(self.pag._DCW["rdc_on"])
        self.assertIsNotNone(request.sd_model.forge_objects.unet.post_cfg)
        self.assertIn("Anima RDC", request.extra_generation_params)
        self.assertNotIn("Anima DCW", request.extra_generation_params)

    def test_legacy_short_smc_call_keeps_historical_omitted_k_default(self):
        class DummyUnet:
            def clone(self):
                return DummyUnet()

            def set_model_sampler_post_cfg_function(self, function):
                self.post_cfg = function

        Anima = type("Anima", (), {})
        model = Anima()
        model.forge_objects = types.SimpleNamespace(unet=DummyUnet())
        request = types.SimpleNamespace(
            sd_model=model,
            extra_generation_params={},
            steps=20,
        )
        process = self.pag.AnimaSafePAG()
        with gr.Blocks():
            inputs = process.ui(False)
        # v0.20 and older callers can stop before index 27. Selecting the
        # legacy SMC radio must retain that contract's old k=0.20 fallback.
        args = [component.value for component in inputs[:27]]
        args[22] = "SMC"

        process.process_before_every_sampling(request, *args)

        self.assertTrue(self.pag._CFG["smc_on"])
        self.assertEqual(self.pag._CFG["smc_preset"], "Custom")
        self.assertEqual(
            (self.pag._CFG["smc_lambda"], self.pag._CFG["smc_k"]),
            (6.0, 0.20),
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
        self.assertIn(
            "w=0도 base CLIP modulation은 남습니다.",
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
                "Anima Modulation Guidance": "Mod",
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

    # ------------------------------------------------------------------
    # Bit-identical efficiency changes (2026-09-23 efficiency report)
    # ------------------------------------------------------------------

    def test_cfg_one_zero_scan_runs_only_when_cond_scale_is_unknown(self):
        """Forge always passes ``cond_scale`` to post-CFG hooks, so the
        ``torch.any(uncond != 0)`` scan (a GPU->CPU sync) is only a fallback
        for callers that omit or garble the key."""
        p = self.pag
        cond, uncond, _incoming = self._cfg_base_fixture()
        zeros = torch.zeros_like(cond)
        real_any = torch.any
        with mock.patch.object(p.torch, "any", wraps=real_any) as spy:
            self.assertIsNone(p._cfg_base_skip_reason(
                self._cfg_one_args(cond, uncond, cond_scale=4.0)
            ))
            self.assertIn("cond_scale", p._cfg_base_skip_reason(
                self._cfg_one_args(cond, zeros, cond_scale=1.0)
            ))
            self.assertEqual(spy.call_count, 0)

            self.assertIn("all zero", p._cfg_base_skip_reason(
                self._cfg_one_args(cond, zeros, cond_scale=None)
            ))
            self.assertIn("all zero", p._cfg_base_skip_reason(
                self._cfg_one_args(cond, zeros, cond_scale="not-a-number")
            ))
            self.assertIsNone(p._cfg_base_skip_reason(
                self._cfg_one_args(cond, uncond, cond_scale=None)
            ))
            self.assertEqual(spy.call_count, 3)

    def _run_cfg_base_twice(self, toggles, diagnostics):
        p = self.pag
        self.setUp()
        cond, uncond, incoming = self._cfg_base_fixture()
        p._CFG.update(**toggles)
        p._APG.update(
            on=bool(toggles.get("apg_on")), eta=0.4, norm_threshold=2.0,
            momentum=0.0, avg=None, last_sigma=None,
        )
        p.reset_cfg_state()
        args = self._cfg_one_args(cond, uncond, cond_scale=4.0)
        with mock.patch.object(p, "guidance_diagnostics_enabled",
                               return_value=diagnostics), \
                mock.patch.object(p, "_recover_effective_cfg",
                                  wraps=p._recover_effective_cfg) as spy:
            first = p._apply_cfg_base(args, incoming)
            second = p._apply_cfg_base(args, incoming)
        fit_flags = [call.kwargs.get("with_fit_error", True)
                     for call in spy.call_args_list]
        return first, second, fit_flags

    def test_fit_error_is_measured_only_for_diagnostics_or_first_step(self):
        """fit_error only feeds a log line and the [VERIFY] summary, so after
        the first evaluation of a pass it is skipped unless diagnostics are on;
        the CFG base result itself must stay bit-identical."""
        for toggles in (
            dict(apg_on=True), dict(cwm_on=True), dict(smc_on=True),
            dict(smc_on=True, apg_on=True, cwm_on=True),
        ):
            with self.subTest(toggles=toggles):
                on_first, on_second, on_flags = self._run_cfg_base_twice(
                    toggles, diagnostics=True
                )
                off_first, off_second, off_flags = self._run_cfg_base_twice(
                    toggles, diagnostics=False
                )
                self.assertTrue(torch.equal(on_first, off_first))
                self.assertTrue(torch.equal(on_second, off_second))
                self.assertEqual(on_flags, [True, True])
                self.assertEqual(off_flags, [True, False])
                # First-step value survives for the (diagnostics-only) summary.
                self.assertIsNotNone(self.pag._CFG["fit_error"])

    def test_nonlinear_incoming_cfg_is_still_warned_on_the_first_step(self):
        p = self.pag
        cond, uncond, incoming = self._cfg_base_fixture()
        p._CFG.update(cwm_on=True)
        nonlinear = incoming + torch.randn_like(incoming)
        with mock.patch.object(p, "guidance_diagnostics_enabled",
                               return_value=False), \
                mock.patch.object(p, "_log") as log:
            p._apply_cfg_base(
                self._cfg_one_args(cond, uncond, cond_scale=4.0), nonlinear
            )
        self.assertTrue(p._CFG["warned"])
        self.assertIn("fit_error=", log.call_args.args[0])

        # An external sampler_cfg_function warns without needing fit_error.
        self.setUp()
        cond, uncond, incoming = self._cfg_base_fixture()
        p._CFG.update(cwm_on=True, fit_checked=True)
        args = self._cfg_one_args(cond, uncond, cond_scale=4.0)
        args["model_options"] = {"sampler_cfg_function": object()}
        with mock.patch.object(p, "guidance_diagnostics_enabled",
                               return_value=False), \
                mock.patch.object(p, "_log") as log:
            p._apply_cfg_base(args, incoming)
        self.assertTrue(p._CFG["warned"])
        self.assertIn("fit_error=?", log.call_args.args[0])

    def test_fit_check_rearms_per_pass_and_generation(self):
        p = self.pag
        p._CFG["fit_checked"] = True
        process = p.AnimaSafePAG()
        process.process_before_every_sampling(
            types.SimpleNamespace(extra_generation_params={}), False
        )
        self.assertFalse(p._CFG["fit_checked"])
        p._CFG["fit_checked"] = True
        process.postprocess(None, None)
        self.assertFalse(p._CFG["fit_checked"])

    def _pag_wrapper_step(self, apply_model=None, batch_bias=(0.0, 1.0)):
        return self.pag._model_wrapper(
            apply_model or self._fake_apply_model,
            {
                "input": torch.zeros(len(batch_bias), 1),
                "timestep": torch.ones(len(batch_bias)),
                "c": {"bias": torch.tensor([[b] for b in batch_bias])},
                "cond_or_uncond": [1, 0][: len(batch_bias)],
            },
        )

    def test_rel_delta_is_measured_only_for_diagnostics_or_first_log(self):
        p = self.pag
        norm = torch.linalg.vector_norm
        for diagnostics, expected_calls in ((False, 2), (True, 6)):
            with self.subTest(diagnostics=diagnostics):
                self.setUp()
                outs = []
                with mock.patch.object(
                    p, "guidance_diagnostics_enabled",
                    return_value=diagnostics,
                ), mock.patch.object(
                    p.torch.linalg, "vector_norm", wraps=norm,
                ) as spy, mock.patch.object(p, "_log"):
                    for _step in range(3):
                        p._STATE["step_open"] = False
                        outs.append(self._pag_wrapper_step())
                # Two norms (numerator, denominator) per measured step.
                self.assertEqual(spy.call_count, expected_calls)
                self.assertTrue(p._STATE["attn_diag_logged"])
                self.assertIsNotNone(p._STATE["attn_last_rel_delta"])
                for out in outs:
                    self.assertTrue(torch.equal(out, torch.tensor([[0.0], [1.0]])))
                self.assertTrue(torch.equal(
                    p._STATE["attn_raw"], torch.tensor([[3.0]])
                ))

    def test_wrapper_index_tensors_are_cached_per_layout(self):
        p = self.pag
        p._INDEX_TENSOR_CACHE.clear()
        seen = []

        def apply_model(x, timestep, **c):
            seen.append(x.shape[0])
            return self._fake_apply_model(x, timestep, **c)

        expected = torch.tensor([[0.0], [1.0]])
        calls = [
            {
                "input": torch.zeros(2, 1),
                "timestep": torch.ones(2),
                "c": {"bias": expected.clone()},
                "cond_or_uncond": [1, 0],
            }
            for _step in range(3)
        ]
        outs = []
        with mock.patch.object(p.torch, "tensor", wraps=torch.tensor) as spy:
            for w in calls:
                p._STATE["step_open"] = False
                outs.append(p._model_wrapper(apply_model, w))
        for out in outs:
            self.assertTrue(torch.equal(out, expected))
        # cond index [1] and uncond index [0]: built once, then reused.
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(seen, [3, 3, 3])
        a = p._index_tensor([1], torch.device("cpu"))
        self.assertIs(a, p._index_tensor([1], torch.device("cpu")))
        self.assertEqual(a.dtype, torch.long)
        self.assertEqual(a.tolist(), [1])
        p.AnimaSafePAG().postprocess(None, None)
        self.assertEqual(p._INDEX_TENSOR_CACHE, {})

    def _raising_apply_model(self, exc_factory, witness):
        """Fail only the enlarged (weak-row) forward, like a real OOM would."""
        class Activation:
            pass

        def apply_model(x, timestep, **c):
            if x.shape[0] > 2:
                activation = Activation()  # stand-in for a failed forward's
                witness.append(weakref.ref(activation))  # activations
                raise exc_factory()
            # The fallback forward must not run while the failed forward's
            # traceback (and so its activations) is still alive.
            witness.append(("fallback", sys.exc_info()[0],
                            witness[0]() if witness else None))
            return self._fake_apply_model(x, timestep, **c)

        return apply_model

    def test_oom_fallback_runs_outside_except_and_disables_perturbation(self):
        p = self.pag
        witness = []
        memory = types.SimpleNamespace(soft_empty_cache=mock.Mock())
        backend = types.ModuleType("backend")
        backend.memory_management = memory
        request = types.SimpleNamespace()
        p._STATE["pass_owner"] = lambda: request
        with mock.patch.dict(sys.modules, {
            "backend": backend, "backend.memory_management": memory,
        }), mock.patch.object(p, "_log") as log:
            out = self._pag_wrapper_step(self._raising_apply_model(
                lambda: torch.OutOfMemoryError("CUDA out of memory."), witness,
            ))

        self.assertTrue(torch.equal(out, torch.tensor([[0.0], [1.0]])))
        _label, active_exc, activation = witness[1]
        self.assertIsNone(active_exc)
        self.assertIsNone(activation)
        memory.soft_empty_cache.assert_called_once()
        self.assertFalse(p._STATE["on"])
        self.assertEqual(p._STATE["wrapper_fallbacks"], 1)
        self.assertIsNone(p._STATE["attn_raw"])
        self.assertIsNone(p._STATE["any_b0"])
        self.assertIn("OutOfMemoryError", " ".join(
            call.args[0] for call in log.call_args_list
        ))
        self.assertTrue(p._perturbation_oom_blocked(request))
        self.assertFalse(p._perturbation_oom_blocked(types.SimpleNamespace()))

        # The rest of the generation runs the plain batch only.
        seen = []

        def plain(x, timestep, **c):
            seen.append(x.shape[0])
            return self._fake_apply_model(x, timestep, **c)

        p._STATE["step_open"] = False
        self._pag_wrapper_step(plain)
        self.assertEqual(seen, [2])

        p.AnimaSafePAG().postprocess(None, None)
        self.assertFalse(p._perturbation_oom_blocked(request))

    def test_non_oom_fallback_keeps_perturbation_for_the_next_step(self):
        p = self.pag
        witness = []
        memory = types.SimpleNamespace(soft_empty_cache=mock.Mock())
        backend = types.ModuleType("backend")
        backend.memory_management = memory
        with mock.patch.dict(sys.modules, {
            "backend": backend, "backend.memory_management": memory,
        }), mock.patch.object(p, "_log"):
            out = self._pag_wrapper_step(self._raising_apply_model(
                lambda: ValueError("shape mismatch"), witness,
            ))

        self.assertTrue(torch.equal(out, torch.tensor([[0.0], [1.0]])))
        _label, active_exc, activation = witness[1]
        self.assertIsNone(active_exc)
        self.assertIsNone(activation)
        memory.soft_empty_cache.assert_not_called()
        self.assertTrue(p._STATE["on"])
        self.assertEqual(p._STATE["wrapper_fallbacks"], 1)

    def test_runtime_error_with_oom_message_counts_as_oom(self):
        p = self.pag
        self.assertTrue(p._is_out_of_memory(torch.OutOfMemoryError("x")))
        self.assertTrue(p._is_out_of_memory(
            RuntimeError("HIP out of memory. Tried to allocate 2.00 GiB")
        ))
        self.assertFalse(p._is_out_of_memory(RuntimeError("shape mismatch")))

    def test_oom_block_is_scoped_to_the_same_processing_object(self):
        p = self.pag

        class Request:
            pass

        first, second = Request(), Request()
        p._STATE["pass_owner"] = p._owner_ref(first)
        p._STATE["pert_oom_owner"] = p._STATE["pass_owner"]
        self.assertTrue(p._perturbation_oom_blocked(first))
        self.assertFalse(p._perturbation_oom_blocked(second))
        # Objects without weakref support fall back to "this pass only".
        self.assertIsNone(p._owner_ref(types.SimpleNamespace()))
        p._STATE["pert_oom_owner"] = None

    def test_pag_head_selection_is_bit_identical_to_legacy_indexing(self):
        p = self.pag
        g = torch.Generator().manual_seed(3)
        cases = 0
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for method, legacy in (("pag", False), ("seg", True)):
                for head_spec in ("", "0-3", "1-2", "2", "0,2", "1,3"):
                    for strength in (0.3, 0.5, 0.75, 1.0):
                        for layout in ("contiguous", "strided"):
                            with self.subTest(dtype=dtype, method=method,
                                              heads=head_spec,
                                              strength=strength,
                                              layout=layout):
                                b, s, n, d = 3, 10, 4, 16
                                if layout == "strided":
                                    value = torch.randn(
                                        b, n, s, d, generator=g
                                    ).to(dtype).transpose(1, 2)
                                    out = torch.randn(
                                        b, n * d, s, generator=g
                                    ).to(dtype).transpose(1, 2)
                                    self.assertFalse(out.is_contiguous())
                                else:
                                    value = torch.randn(
                                        b, s, n, d, generator=g
                                    ).to(dtype)
                                    out = torch.randn(
                                        b, s, n * d, generator=g
                                    ).to(dtype)
                                p._ORIG_ANIMA_ATTN_OP = (
                                    lambda q, k, v, *a, _o=out, **kw: _o
                                )
                                p._STATE.update(
                                    active=1, attn_b0=1, attn_b1=3,
                                    attn_method=method, legacy_attn=legacy,
                                    strength=strength, head_spec=head_spec,
                                )
                                new = p._patched_anima_attention_op(
                                    value, value, value
                                )
                                heads = sorted(p._parse_attention_heads(
                                    head_spec, n
                                ))
                                old = _legacy_pag_value_blend(
                                    out, value, 1, 3, heads, strength, method
                                )
                                self.assertTrue(torch.equal(new, old))
                                self.assertEqual(new.dtype, old.dtype)
                                cases += 1
        self.assertEqual(cases, 3 * 2 * 6 * 4 * 2)

    def test_official_seg_head_selection_is_bit_identical_to_legacy(self):
        p = self.pag
        g = torch.Generator().manual_seed(5)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for head_spec in ("", "0-2", "1-3", "3", "0,3"):
                for strength in (0.3, 0.5, 1.0):
                    for sigma in (1.5, 10000.0):
                        with self.subTest(dtype=dtype, heads=head_spec,
                                          strength=strength, sigma=sigma):
                            query = torch.randn(
                                3, 1 * 5 * 6, 4, 8, generator=g
                            ).to(dtype)
                            new = p._official_seg_query(
                                query, 1, 3, sigma, (1, 5, 6), strength,
                                head_spec,
                            )
                            old = _legacy_official_seg_query(
                                p, query, 1, 3, sigma, (1, 5, 6), strength,
                                head_spec,
                            )
                            self.assertTrue(torch.equal(new, old))

    def test_head_index_is_a_cached_device_tensor_not_a_host_list(self):
        """A Python list index is re-uploaded to the GPU on every indexing op;
        the cached LongTensor carries the same values on the same device."""
        p = self.pag
        p._INDEX_TENSOR_CACHE.clear()
        idx = p._head_index([0, 2, 3], torch.device("cpu"))
        self.assertIs(idx, p._head_index([0, 2, 3], torch.device("cpu")))
        self.assertEqual(idx.dtype, torch.long)
        self.assertEqual(idx.tolist(), [0, 2, 3])
        p._INDEX_TENSOR_CACHE.clear()


FORGE_ROOT = ROOT.parents[1]
NEGPIP_ANIMA = FORGE_ROOT / "extensions" / "sd-forge-negpip" / "lib_negpip" / "anima.py"


@contextlib.contextmanager
def _stub_modules(stubs):
    """``stubs`` 이름만 잠시 바꾼다. ``mock.patch.dict(sys.modules)`` 는 블록 안에서 처음
    import 된 모듈(torch._dynamo·einops·torchvision 등)까지 지워 재import 를 깨뜨린다."""
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


def _attention_stub(record):
    def attention_function(q, k, v, heads, skip_reshape=True, transformer_options=None):
        record.append(int(q.shape[0]))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        b, h, s, d = out.shape
        return out.transpose(1, 2).reshape(b, s, h * d)
    return attention_function


def _load_real_forge_anima(record):
    """Forge 의 실제 ``backend/nn/anima.py``(Block·Anima forward)를 GPU 없이 불러온다.

    무거운 backend 서비스(attention 백엔드, manual cast, 커스텀 커널)만 CPU 대역으로
    바꾼다. 행마다 독립인 연산이라는 성질은 그대로다."""
    path = FORGE_ROOT / "backend" / "nn" / "anima.py"
    if not path.is_file():
        raise unittest.SkipTest("Forge backend/nn/anima.py 가 없다")

    def rms_rope_split_half(q, k, rope, q_scale, k_scale, eps):
        def rms(t, scale):
            t32 = t.float()
            t32 = t32 * torch.rsqrt(t32.pow(2).mean(-1, keepdim=True) + eps)
            return (t32 * scale.float()).to(t.dtype)
        return rms(q, q_scale), rms(k, k_scale)

    backend = types.ModuleType("backend")
    backend.__path__ = []
    stubs = {
        "backend": backend,
        "backend.args": types.SimpleNamespace(
            dynamic_args=types.SimpleNamespace(ref_latents=[]),
        ),
        "backend.attention": types.SimpleNamespace(
            attention_function=_attention_stub(record),
        ),
        "backend.memory_management": types.SimpleNamespace(
            is_device_mps=lambda device: False,
        ),
        "backend.operations": types.SimpleNamespace(
            main_stream_worker=lambda *a, **k: contextlib.nullcontext(),
            scaled_dot_product_attention=torch.nn.functional.scaled_dot_product_attention,
            weights_manual_cast=lambda module, x: (module.weight, None, None),
        ),
        "backend.quant_ops": types.SimpleNamespace(
            ck=types.SimpleNamespace(rms_rope_split_half=rms_rope_split_half),
        ),
        "backend.utils": types.SimpleNamespace(pad_to_patch_size=lambda x, size: x),
    }
    with _stub_modules(stubs):
        spec = importlib.util.spec_from_file_location("_pag_test_forge_anima", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _load_real_negpip(anima_module):
    """sd-forge-negpip 의 실제 Anima 훅(DiT forward·cross-attn forward)."""
    if not NEGPIP_ANIMA.is_file():
        raise unittest.SkipTest("sd-forge-negpip 확장이 없다")
    backend = types.ModuleType("backend")
    backend.__path__ = []
    nn_pkg = types.ModuleType("backend.nn")
    nn_pkg.__path__ = []
    nn_pkg.anima = anima_module
    sampling_pkg = types.ModuleType("backend.sampling")
    sampling_pkg.__path__ = []
    sampling_pkg.condition = types.SimpleNamespace()
    sampling_pkg.sampling_function = types.SimpleNamespace()
    modules_stub = types.ModuleType("modules")
    modules_stub.shared = types.SimpleNamespace()
    stubs = {
        "backend": backend,
        "backend.nn": nn_pkg,
        "backend.nn.anima": anima_module,
        "backend.sampling": sampling_pkg,
        "modules": modules_stub,
    }
    with _stub_modules(stubs):
        spec = importlib.util.spec_from_file_location("_pag_test_negpip", NEGPIP_ANIMA)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class PrefixDedupRealBlockTests(unittest.TestCase):
    """앞쪽 블록 중복 제거(효율 보고서 [2])를 Forge 의 실제 Anima Block 으로 확인한다.

    같은 입력으로 설정을 끈 경로(예전 전체 배치)와 켠 경로를 돌려 반환 행·weak 예측이
    ``torch.allclose`` 로 같은지, 첫 target 이전 블록이 원래 배치 행만 돌았는지 본다."""

    ATOL = 1e-5
    RTOL = 1e-5

    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()
        cls.max_diff = 0.0

    @classmethod
    def tearDownClass(cls):
        cls.pag._teardown_global_patches()
        print(
            f"\n[prefix dedup] max |on-off| over real-block cases = {cls.max_diff:.3e}",
            file=sys.stderr,
        )

    def setUp(self):
        AnimaSafePagTests.setUp(self)
        p = self.pag
        p._STATE.update(
            prefix_dedup=True, seg_separable=True,
            dedup_blocks=0, dedup_fallbacks=0, dedup_warned=False,
            dedup_disabled=False, dedup_until=None, dedup_src=None, dedup_next=0,
        )
        self.record = []
        self.anima = _load_real_forge_anima(self.record)
        torch.manual_seed(1234)
        self.dit = self.anima.Anima(
            in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
            model_channels=48, crossattn_emb_channels=32, adaln_lora_dim=16,
            num_blocks=6, num_heads=4,
        ).eval()
        self.negpip = None

    def tearDown(self):
        self.pag._teardown_global_patches()
        if self.negpip is not None:
            self.negpip._hook_forwards(True)
            self.negpip._hook_dit_forward(self.dit, True)

    def _inputs(self, batch=2, seed=7, negpip=False):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(batch, 4, 1, 8, 10, generator=g)
        ts = torch.rand(batch, generator=g)
        context = torch.randn(batch, 5, 32, generator=g)
        c = {"c_crossattn": context, "transformer_options": {"cond_or_uncond": []}}
        if negpip:
            mask = torch.ones(batch, 5, 1)
            mask[:, 1] = -1.0
            c["c_negpip_mask"] = mask
        return x, ts, c

    def _apply_model(self, x, ts, **c):
        extra = {}
        if "c_negpip_mask" in c:
            extra["c_negpip_mask"] = c["c_negpip_mask"]
        with torch.no_grad():
            return self.dit(
                x, ts, context=c["c_crossattn"],
                transformer_options=c.get("transformer_options", {}), **extra,
            )

    def _run(self, dedup, cou, *, batch=2, negpip=False, seed=7):
        p = self.pag
        p._STATE.update(
            prefix_dedup=dedup, dedup_blocks=0, dedup_fallbacks=0,
            dedup_warned=False, dedup_disabled=False,
            step_open=False, attn_raw=None, slg_raw=None, cond_raw=None,
        )
        self.record.clear()
        x, ts, c = self._inputs(batch=batch, seed=seed, negpip=negpip)
        out = p._model_wrapper(
            self._apply_model,
            {"input": x, "timestep": ts, "c": c, "cond_or_uncond": cou},
        )
        return {
            "out": out,
            "attn": p._STATE["attn_raw"],
            "slg": p._STATE["slg_raw"],
            "cond": p._STATE["cond_raw"],
            "blocks": p._STATE["dedup_blocks"],
            "fallbacks": p._STATE["dedup_fallbacks"],
            "record": list(self.record),
        }

    def _assert_close(self, on, off):
        for key in ("out", "attn", "slg"):
            if off[key] is None:
                self.assertIsNone(on[key], key)
                continue
            diff = float((on[key].float() - off[key].float()).abs().max())
            type(self).max_diff = max(type(self).max_diff, diff)
            self.assertTrue(
                torch.allclose(on[key], off[key], atol=self.ATOL, rtol=self.RTOL),
                f"{key}: max diff {diff:.3e}",
            )

    def _patch(self):
        self.assertEqual(self.pag._ensure_patched(self.dit), 6)

    def test_prefix_blocks_run_original_rows_and_match_full_batch(self):
        p = self.pag
        self._patch()
        cases = [
            # (설명, 갱신, cond_or_uncond, batch, 첫 target)
            ("pag", dict(attn_method="pag", attn_targets={3}), [0, 1], 2, 3),
            ("pag-uncond-first", dict(attn_method="pag", attn_targets={2, 4}), [1, 0], 2, 2),
            ("pag-batch2", dict(attn_method="pag", attn_targets={3}), [1, 0], 4, 3),
            ("seg", dict(attn_method="seg", attn_targets={2}, seg_sigma=1.0), [0, 1], 2, 2),
            ("seg+slg", dict(attn_method="seg", attn_targets={4}, seg_sigma=1.0,
                             slg_on=True, slg_targets={3}), [0, 1], 2, 3),
            ("slg-only", dict(attn_method=None, attn_targets=set(),
                              slg_on=True, slg_targets={3}), [0, 1], 2, 3),
            ("cond-only", dict(attn_method="pag", attn_targets={3}), [0], 1, 3),
        ]
        for name, update, cou, batch, first in cases:
            with self.subTest(case=name):
                p._STATE.update(
                    attn_method="pag", attn_targets={0}, slg_on=False,
                    slg_targets=set(), seg_sigma=100.0,
                )
                p._STATE.update(update)
                off = self._run(False, cou, batch=batch)
                on = self._run(True, cou, batch=batch)
                self._assert_close(on, off)
                # weak 예측이 실제로 cond 와 다른(교란이 걸린) 비교인지 확인한다.
                for key in ("attn", "slg"):
                    if on[key] is not None:
                        self.assertFalse(torch.allclose(on[key], on["cond"]), key)
                self.assertEqual(off["blocks"], 0)
                self.assertEqual(on["blocks"], first)
                self.assertEqual(on["fallbacks"], 0)
                n_cond = batch // len(cou) * cou.count(0)
                n_weak = n_cond * (int(p._STATE["attn_method"] is not None)
                                   + int(p._STATE["slg_on"]))
                # 블록마다 self·cross attention 두 번. 앞쪽 first 블록은 원래 행만.
                self.assertEqual(off["record"], [batch + n_weak] * 12)
                self.assertEqual(
                    on["record"],
                    [batch] * (2 * first) + [batch + n_weak] * (12 - 2 * first),
                )
                # 표식은 forward 가 끝나면 지워진다.
                self.assertIsNone(p._STATE["dedup_until"])
                self.assertIsNone(p._STATE["dedup_src"])

    def test_modulation_and_dave_in_prefix_blocks_match_full_batch(self):
        p = self.pag
        self._patch()
        p._STATE.update(attn_method="pag", attn_targets={4})
        g = torch.Generator().manual_seed(3)
        p._MOD.update(
            on=True, targets={0, 2, 4},
            block_modulations=torch.randn(6, 3 * 48, generator=g) * 0.1,
            typed={}, hits=0, warned=False,
        )
        p._DAVE.update(on=True, strength=0.3, tau=0.0, targets={1, 3, 5}, steps=0)
        try:
            off = self._run(False, [0, 1])
            on = self._run(True, [0, 1])
        finally:
            p._MOD.update(on=False, targets=set(), block_modulations=None, typed={})
            p._DAVE.update(on=False, targets=set(), steps=0)
        self._assert_close(on, off)
        self.assertEqual(on["blocks"], 4)
        self.assertEqual(on["fallbacks"], 0)

    def test_real_negpip_hooks_are_sliced_with_the_rows(self):
        p = self.pag
        self.negpip = _load_real_negpip(self.anima)
        self.negpip._hook_dit_forward(self.dit, False)
        self.negpip._hook_forwards(False)
        self._patch()
        p._STATE.update(attn_method="pag", attn_targets={3})
        off = self._run(False, [0, 1], negpip=True)
        on = self._run(True, [0, 1], negpip=True)
        self._assert_close(on, off)
        # 마스크를 x 와 같이 자르지 않으면 NegPiP cross-attn 이 행 수로 실패해 폴백한다.
        self.assertEqual(on["fallbacks"], 0)
        self.assertEqual(on["blocks"], 3)
        # NegPiP 의 음수 토큰이 실제로 결과를 바꾸는 경로인지(무의미한 비교가 아닌지).
        plain = self._run(True, [0, 1], negpip=False)
        self.assertFalse(torch.allclose(plain["out"], on["out"]))

    def test_unknown_extended_transformer_option_falls_back_to_full_batch(self):
        p = self.pag
        self._patch()
        p._STATE.update(attn_method="pag", attn_targets={3})
        native = type(self.dit).forward
        dit = self.dit

        def foreign_forward(x, timesteps, context, *args, **kwargs):
            # 다른 확장이 확장 배치 행 수의 텐서를 transformer_options 에 싣는 경우.
            options = dict(kwargs.get("transformer_options") or {})
            options["foreign_rows"] = torch.zeros(context.shape[0], 1)
            kwargs["transformer_options"] = options
            return native(dit, x, timesteps, context, *args, **kwargs)

        dit.forward = foreign_forward
        try:
            off = self._run(False, [0, 1])
            on = self._run(True, [0, 1])
        finally:
            del dit.forward
        for key in ("out", "attn"):
            self.assertTrue(torch.equal(on[key], off[key]), key)
        self.assertEqual(on["blocks"], 0)
        self.assertEqual(on["fallbacks"], 1)
        self.assertEqual(on["record"], off["record"])


class PrefixDedupUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def setUp(self):
        AnimaSafePagTests.setUp(self)
        self.pag._STATE.update(
            prefix_dedup=True, dedup_blocks=0, dedup_fallbacks=0,
            dedup_warned=False, dedup_disabled=False,
        )
        self.pag._clear_markers()

    def _layout(self, batch=2, src=(0,), until=3):
        p = self.pag
        p._STATE.update(
            any_b0=batch, dedup_src=torch.tensor(src), dedup_next=0,
            dedup_until=until,
        )

    def test_block_call_slices_known_rows_and_keeps_broadcasts(self):
        p = self.pag
        self._layout()
        x = torch.randn(3, 1, 2, 2, 4)
        mask = torch.ones(3, 5, 1)
        options = {"negpip_mask": mask, "sigmas": torch.ones(2), "cond_or_uncond": [0, 1]}
        plan = p._dedup_block_call(
            (x, torch.randn(3, 1, 4), torch.randn(3, 5, 8)),
            {
                "rope_emb_L_1_1_D": torch.randn(1, 4, 1, 1, 2),
                "adaln_lora_B_T_3D": torch.randn(3, 1, 12),
                "extra_per_block_pos_emb": None,
                "transformer_options": options,
            },
        )
        self.assertIsInstance(plan, tuple)
        args, kwargs = plan
        self.assertEqual([a.shape[0] for a in args], [2, 2, 2])
        self.assertEqual(kwargs["rope_emb_L_1_1_D"].shape[0], 1)
        self.assertEqual(kwargs["adaln_lora_B_T_3D"].shape[0], 2)
        self.assertIsNone(kwargs["extra_per_block_pos_emb"])
        self.assertEqual(kwargs["transformer_options"]["negpip_mask"].shape[0], 2)
        self.assertIs(kwargs["transformer_options"]["sigmas"], options["sigmas"])
        # 공유 transformer_options 는 그대로다.
        self.assertIs(options["negpip_mask"], mask)

    def test_block_call_rejects_unknown_or_mismatched_inputs(self):
        p = self.pag
        self._layout()
        x = torch.randn(3, 1, 2, 2, 4)
        emb = torch.randn(3, 1, 4)
        ctx = torch.randn(3, 5, 8)
        cases = {
            "unknown kwarg": ((x, emb, ctx), {"ip_tokens": torch.randn(3, 2, 8)}),
            "row mismatch": ((x, emb, torch.randn(2, 5, 8)), {}),
            "foreign option": ((x, emb, ctx), {"transformer_options": {"ipa": torch.zeros(3, 1)}}),
            "non-tensor arg": ((x, emb, ctx, "rope"), {}),
            "x rows": ((torch.randn(2, 1, 2, 2, 4), emb, ctx), {}),
        }
        for name, (args, kwargs) in cases.items():
            with self.subTest(case=name):
                self.assertIsInstance(p._dedup_block_call(args, kwargs), str)

    def test_plan_requires_contiguous_blocks_from_zero(self):
        p = self.pag
        self._layout(until=3)
        args = (torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 2))
        # 블록 1 이 먼저 오면(0 이 감싸지지 않음 등) 이번 forward 는 끝이다.
        self.assertIsNone(p._dedup_plan(1, args, {}))
        self.assertIsNone(p._STATE["dedup_until"])
        self._layout(until=3)
        self.assertIsNotNone(p._dedup_plan(0, args, {}))
        # 첫 target 에 닿으면 끝.
        p._STATE["dedup_next"] = 3
        self.assertIsNone(p._dedup_plan(3, args, {}))
        self.assertIsNone(p._STATE["dedup_until"])

    def test_block_failure_on_sliced_rows_falls_back_and_disables_for_generation(self):
        p = self.pag
        p._STATE.update(attn_targets=set(), slg_targets=set())
        calls = []

        def orig_forward(x, emb, ctx):
            calls.append(x.shape[0])
            if x.shape[0] != 3:
                raise ValueError("needs the enlarged batch")
            return x * 2.0

        wrapped = p._make_block_wrapper(0, orig_forward)
        self._layout()
        x = torch.arange(6.0).reshape(3, 2)
        out = wrapped(x, torch.ones(3, 2), torch.ones(3, 2))
        self.assertTrue(torch.equal(out, x * 2.0))
        self.assertEqual(calls, [2, 3])
        self.assertTrue(p._STATE["dedup_disabled"])
        self.assertEqual(p._STATE["dedup_fallbacks"], 1)
        self.assertIsNone(p._STATE["dedup_until"])

    def test_block_oom_on_sliced_rows_propagates(self):
        p = self.pag
        oom = getattr(torch, "OutOfMemoryError", None) or RuntimeError

        def orig_forward(x, emb, ctx):
            raise oom("CUDA out of memory")

        wrapped = p._make_block_wrapper(0, orig_forward)
        self._layout()
        with self.assertRaises(oom):
            wrapped(torch.ones(3, 2), torch.ones(3, 2), torch.ones(3, 2))

    def test_output_copies_cond_rows_into_weak_slots(self):
        p = self.pag
        p._STATE.update(attn_targets=set(), slg_targets=set())
        wrapped = p._make_block_wrapper(0, lambda x, emb, ctx: x + emb)
        # 배치 [uncond, cond] + weak(cond 사본) → 원본 행 1 이 weak 자리로 간다.
        self._layout(batch=2, src=(1,))
        x = torch.tensor([[1.0], [2.0], [2.0]])
        out = wrapped(x, torch.ones(3, 1), torch.ones(3, 1))
        self.assertTrue(torch.equal(out, torch.tensor([[2.0], [3.0], [3.0]])))
        self.assertEqual(p._STATE["dedup_blocks"], 1)
        self.assertEqual(p._STATE["dedup_next"], 1)

    def test_setting_off_or_first_target_zero_keeps_full_batch(self):
        p = self.pag
        seen = []

        def apply_model(x, ts, **c):
            seen.append(p._STATE["dedup_until"])
            return c["bias"].clone()

        w = {
            "input": torch.zeros(2, 1), "timestep": torch.ones(2),
            "c": {"bias": torch.tensor([[0.0], [1.0]])}, "cond_or_uncond": [1, 0],
        }
        for dedup, targets, expected in (
            (True, {18}, 18), (False, {18}, None), (True, {0, 5}, None),
        ):
            p._STATE.update(prefix_dedup=dedup, attn_targets=targets, step_open=False)
            p._model_wrapper(apply_model, w)
            self.assertEqual(seen[-1], expected)
        p._STATE.update(prefix_dedup=True, attn_targets={18}, step_open=False,
                        dedup_disabled=True)
        p._model_wrapper(apply_model, w)
        self.assertIsNone(seen[-1])


class SegSeparableBlurTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def test_separable_blur_matches_2d_kernel(self):
        p = self.pag
        g = torch.Generator().manual_seed(11)
        worst = {}
        for dtype, atol in ((torch.float64, 1e-12), (torch.float32, 1e-5),
                            (torch.bfloat16, 2e-2), (torch.float16, 2e-3)):
            for shape in ((2, 6, 9, 13), (1, 4, 16, 16), (3, 2, 5, 7)):
                for sigma in (0.8, 1.5, 3.0, 100.0):
                    with self.subTest(dtype=dtype, shape=shape, sigma=sigma):
                        img = torch.randn(*shape, generator=g).to(dtype)
                        old = p._gaussian_blur_2d(img, sigma, separable=False)
                        new = p._gaussian_blur_2d(img, sigma, separable=True)
                        self.assertEqual(new.shape, old.shape)
                        self.assertEqual(new.dtype, old.dtype)
                        diff = float((new.double() - old.double()).abs().max())
                        worst[dtype] = max(worst.get(dtype, 0.0), diff)
                        self.assertTrue(torch.allclose(new, old, atol=atol, rtol=0))
        print(
            "\n[SEG separable] max |separable-2D|: "
            + ", ".join(f"{str(k).split('.')[-1]}={v:.3e}" for k, v in worst.items()),
            file=sys.stderr,
        )

    def test_default_is_separable_and_off_keeps_legacy_2d(self):
        p = self.pag
        img = torch.randn(1, 3, 9, 11, generator=torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(p._gaussian_blur_2d(img, 2.0),
                                    p._gaussian_blur_2d(img, 2.0, separable=True)))
        # 끈 경로는 예전 식 그대로(2D 외적 커널)다.
        k = 9
        half = (k - 1) * 0.5
        xs = torch.linspace(-half, half, steps=k)
        pdf = torch.exp(-0.5 * (xs / 2.0).pow(2))
        k1 = pdf / pdf.sum()
        k2 = torch.mm(k1[:, None], k1[None, :]).expand(3, 1, k, k)
        legacy = torch.nn.functional.conv2d(
            torch.nn.functional.pad(img, [k // 2] * 4, mode="reflect"), k2, groups=3,
        )
        self.assertTrue(torch.equal(p._gaussian_blur_2d(img, 2.0, separable=False), legacy))

    def test_attention_op_follows_the_setting(self):
        p = self.pag
        query = torch.randn(2, 1 * 5 * 6, 2, 4, generator=torch.Generator().manual_seed(4))
        seen = []
        original = mock.Mock(side_effect=lambda q, k, v, *a, **kw: (
            seen.append(q.clone()) or q.reshape(q.shape[0], q.shape[1], -1)
        ))
        with mock.patch.object(p, "_ORIG_ANIMA_ATTN_OP", original):
            for flag in (True, False):
                p._STATE.update(
                    active=1, attn_b0=1, attn_b1=2, attn_method="seg",
                    legacy_attn=False, seg_sigma=1.5, strength=1.0, head_spec="",
                    attn_spatial_shape=(1, 5, 6), seg_separable=flag,
                    attn_hook_hits=0, attn_hook_hits_total=0,
                )
                p._patched_anima_attention_op(query, query, query)
        p._STATE.update(active=0, attn_b0=None, attn_b1=None, seg_separable=True)
        for flag, q in zip((True, False), seen):
            expected = p._official_seg_query(
                query, 1, 2, 1.5, (1, 5, 6), 1.0, "", separable=flag,
            )
            self.assertTrue(torch.equal(q, expected))


class GuidanceSettingsInfotextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def _process(self, method="PAG", seg_sigma=100.0, legacy=False, opts=None):
        p = self.pag
        record = []
        anima = _load_real_forge_anima(record)
        torch.manual_seed(0)
        dit = anima.Anima(
            in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
            model_channels=48, crossattn_emb_channels=32, adaln_lora_dim=16,
            num_blocks=6, num_heads=4,
        )

        class DummyUnet:
            def __init__(self):
                self.model = types.SimpleNamespace(diffusion_model=dit)
                self.model_options = {}

            def clone(self):
                clone = DummyUnet()
                return clone

            def set_model_unet_function_wrapper(self, fn):
                self.model_options["model_function_wrapper"] = fn

            def set_model_sampler_pre_cfg_function(self, fn):
                pass

            def set_model_sampler_post_cfg_function(self, fn):
                pass

        Anima = type("Anima", (), {})
        model = Anima()
        model.forge_objects = types.SimpleNamespace(unet=DummyUnet())
        request = types.SimpleNamespace(sd_model=model, extra_generation_params={}, steps=20)
        process = p.AnimaSafePAG()
        with gr.Blocks():
            inputs = process.ui(False)
        args = [component.value for component in inputs]
        args[0] = True
        args[1] = method
        args[4] = "3"
        args[20] = legacy
        args[21] = seg_sigma
        shared_stub = types.SimpleNamespace(opts=types.SimpleNamespace(**(opts or {})))
        try:
            with mock.patch.object(p, "shared", shared_stub):
                process.process_before_every_sampling(request, *args)
            return dict(request.extra_generation_params), dict(p._STATE)
        finally:
            process.postprocess(request, None)
            p._teardown_global_patches()

    def test_infotext_records_both_settings_for_reproduction(self):
        p = self.pag
        params, state = self._process("SEG", seg_sigma=2.0)
        self.assertEqual(params[p.INFOTEXT_PREFIX_DEDUP], "True")
        self.assertEqual(params[p.INFOTEXT_SEG_SEPARABLE], "True")
        self.assertTrue(state["prefix_dedup"] and state["seg_separable"])

        params, state = self._process("SEG", seg_sigma=2.0, opts={
            p.OPT_PREFIX_DEDUP: False, p.OPT_SEG_SEPARABLE: False,
        })
        self.assertEqual(params[p.INFOTEXT_PREFIX_DEDUP], "False")
        self.assertEqual(params[p.INFOTEXT_SEG_SEPARABLE], "False")
        self.assertFalse(state["prefix_dedup"] or state["seg_separable"])

    def test_seg_blur_key_only_when_blur_is_used(self):
        p = self.pag
        for method, sigma, legacy in (("PAG", 100.0, False), ("SEG", 10000.0, False),
                                      ("SEG", 100.0, True)):
            with self.subTest(method=method, sigma=sigma, legacy=legacy):
                params, _ = self._process(method, seg_sigma=sigma, legacy=legacy)
                self.assertIn(p.INFOTEXT_PREFIX_DEDUP, params)
                self.assertNotIn(p.INFOTEXT_SEG_SEPARABLE, params)

    def test_infotext_keys_are_cleared_between_xyz_cells(self):
        p = self.pag
        request = types.SimpleNamespace(extra_generation_params={
            p.INFOTEXT_PREFIX_DEDUP: "True", p.INFOTEXT_SEG_SEPARABLE: "False",
            "Steps": 20,
        })
        p._clear_extra_generation_params(request)
        self.assertEqual(request.extra_generation_params, {"Steps": 20})

    def test_settings_register_under_sam_extra_guidance_with_infotext_names(self):
        p = self.pag
        added = {}

        class OptionInfo:
            def __init__(self, default, label, component=None, *, section=None, infotext=None):
                self.default, self.section, self.infotext = default, section, infotext

            def info(self, _text):
                return self

        shared_stub = types.SimpleNamespace(
            OptionInfo=OptionInfo,
            opts=types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info)),
        )
        with mock.patch.object(p, "shared", shared_stub):
            p._on_ui_settings()
        self.assertEqual(set(added), {p.OPT_PREFIX_DEDUP, p.OPT_SEG_SEPARABLE})
        for key, infotext in ((p.OPT_PREFIX_DEDUP, p.INFOTEXT_PREFIX_DEDUP),
                              (p.OPT_SEG_SEPARABLE, p.INFOTEXT_SEG_SEPARABLE)):
            self.assertIs(added[key].default, True)
            self.assertEqual(added[key].section, ("sam3_guidance", "SAM Extra Guidance"))
            self.assertEqual(added[key].infotext, infotext)


def _legacy_pag_value_blend(out, value, a0, a1, heads, strength, method):
    """Pre-change value-path blend of ``_patched_anima_attention_op`` (list
    indexing of the head axis), kept verbatim as a bit-exact oracle."""
    out_heads = out.reshape_as(value)
    result_heads = out_heads.clone()
    if method == "pag":
        target = value[a0:a1]
    else:
        target = value[a0:a1].mean(dim=1, keepdim=True).expand_as(value[a0:a1])
    weak = result_heads[a0:a1]
    weak[..., heads, :] = torch.lerp(
        out_heads[a0:a1, :, heads, :],
        target[..., heads, :],
        strength,
    )
    result_heads[a0:a1] = weak
    return result_heads.reshape_as(out)


def _legacy_official_seg_query(p, query, a0, a1, sigma, spatial_shape,
                               strength, head_spec):
    """Pre-change 4-D branch of ``_official_seg_query``, verbatim."""
    weak = query[a0:a1]
    t, h, w = (int(v) for v in spatial_shape)
    b, seq, n, d = weak.shape
    weak_6d = weak.reshape(b, t, h, w, n, d)
    spatial = weak_6d.permute(0, 1, 4, 5, 2, 3).reshape(b * t, n * d, h, w)
    if sigma > 9999.0:
        spatial = spatial.mean(dim=(-2, -1), keepdim=True).expand_as(spatial)
    else:
        spatial = p._gaussian_blur_2d(spatial, sigma)
    perturbed = (
        spatial.reshape(b, t, n, d, h, w)
        .permute(0, 1, 4, 5, 2, 3)
        .reshape(b, seq, n, d)
    )
    heads = sorted(p._parse_attention_heads(head_spec, int(weak.shape[-2])))
    if not heads:
        return query
    strength = min(1.0, max(0.0, float(strength)))
    blended = weak.clone()
    blended[..., heads, :] = torch.lerp(
        weak[..., heads, :],
        perturbed[..., heads, :],
        strength,
    )
    result = query.clone()
    result[a0:a1] = blended
    return result


if __name__ == "__main__":
    unittest.main()
