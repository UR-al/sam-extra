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
            attn_targets={0},
            slg_on=False,
            slg_scale=3.0,
            slg_targets=set(),
            auto_decay=False,
            rescale=0.0,
            active=0,
            attn_raw=None,
            slg_raw=None,
            attn_b0=None,
            attn_b1=None,
            slg_b0=None,
            slg_b1=None,
            any_b0=None,
            apg_autooff_rescale=True,
            wrapper_calls=0,
            weak_steps=0,
            applied_steps=0,
            delta_logged=False,
        )
        p._APG["on"] = False
        p._ADG["on"] = False

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

        self.assertEqual(p._STATE["step"], 1)
        self.assertEqual(p._STATE["weak_steps"], 1)
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
        self.assertEqual(self._post_cfg().item(), -0.5)

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

    def test_current_anima_attention_layout_changes_only_weak_rows(self):
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
        )
        q = torch.zeros(2, 4, 2, 3)
        value = torch.arange(q.numel(), dtype=torch.float32).reshape_as(q)

        out = p._patched_anima_attention_op(q, q, value)

        self.assertEqual(torch.count_nonzero(out[0]).item(), 0)
        torch.testing.assert_close(out[1], value.reshape(2, 4, 6)[1] * 0.5)

        p._STATE["attn_method"] = "seg"
        p._STATE["strength"] = 1.0
        out = p._patched_anima_attention_op(q, q, value)
        expected = value.reshape(2, 4, 6)[1].mean(0, keepdim=True).expand(4, 6)
        torch.testing.assert_close(out[1], expected)

    def test_zero_correction_with_rescale_keeps_cfg_base_bit_identical(self):
        p = self.pag
        cond = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
        base = torch.tensor([[8.0, -4.0, 2.0, -1.0]])
        p._STATE.update(
            attn_raw=cond.clone(),
            slg_raw=None,
            rescale=0.2,
        )

        out = p._apply_perturbation({"cond_denoised": cond}, base)

        self.assertTrue(torch.equal(out, base))

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
