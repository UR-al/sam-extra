"""Regression tests for combining several guidance techniques at once.

The user-facing question these lock in: can PAG/SEG, DCW, CWM, SMC and Skimmed
CFG all be enabled in the same generation without one silently disabling
another? They compose as:

  * Skimmed CFG runs in its own explicitly-prepended post-CFG hook and rewrites
    ``cond_denoised``/``uncond_denoised`` in place.
  * The Anima Safe PAG suite runs its single post-CFG orchestrator afterwards
    despite Forge currently ignoring ``sorting_priority`` for
    ``process_before_every_sampling``: CFG base (SMC -> APG -> CWM) ->
    PAG/SEG/SLG perturbation -> DCW, all in denoised (x0) space.

The composition therefore depends on (a) Skimmed CFG's hook running *before* the
Safe PAG hook, and (b) each Safe PAG stage actually contributing. Both are
asserted here so a future reorder or refactor can't reintroduce a silent
mutual-exclusion.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(module_file: str, test_name: str):
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    modules_stub.script_callbacks = types.SimpleNamespace(on_before_ui=lambda fn: None)
    modules_stub.scripts = types.SimpleNamespace(
        Script=Script, AlwaysVisible=object(), scripts_data=[]
    )
    modules_stub.shared = types.SimpleNamespace(
        state=types.SimpleNamespace(sampling_step=0, sampling_steps=20)
    )

    old = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        spec = importlib.util.spec_from_file_location(
            test_name, ROOT / "scripts" / module_file
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = old


class GuidanceCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load("anima_safe_pag.py", "_test_compose_pag")
        cls.skim = _load("anima_skimmed_cfg.py", "_test_compose_skim")

    def test_skimmed_registration_forces_real_hook_order(self):
        """Reproduce Forge's actual alphabetic registration order.

        Safe PAG appends first because anima_safe_pag.py loads before
        anima_skimmed_cfg.py. Skimmed must then prepend itself; comparing the
        two sorting_priority integers would miss Forge's duplicate runner
        method and is not an execution-order test.
        """

        class DummyUnet:
            def __init__(self):
                self.model_options = {}

            def set_model_sampler_post_cfg_function(self, function):
                current = self.model_options.get(
                    "sampler_post_cfg_function", []
                )
                self.model_options["sampler_post_cfg_function"] = [
                    *current,
                    function,
                ]

        unet = DummyUnet()
        unet.set_model_sampler_post_cfg_function(self.pag._post_cfg)
        self.skim._prepend_post_cfg_function(unet)

        self.assertEqual(
            unet.model_options["sampler_post_cfg_function"],
            [self.skim._post_cfg, self.pag._post_cfg],
        )

    def _base_args(self) -> dict:
        torch.manual_seed(0)
        cond = torch.randn(1, 4, 8, 8)
        uncond = torch.randn(1, 4, 8, 8)
        denoised = uncond + 7.0 * (cond - uncond)  # linear CFG @ ~7
        return {
            "denoised": denoised,
            "cond_denoised": cond,
            "uncond_denoised": uncond,
            "input": torch.randn(1, 4, 8, 8),
            "sigma": torch.tensor([1.0]),
        }

    def _configure(self, *, cwm: bool, smc: bool, pert: bool, dcw: bool) -> None:
        p = self.pag
        p._RUNTIME.reset_pass()
        p._RUNTIME.smc_prev = None
        p._APG["on"] = False
        p._CFG.update(
            mode="preserve", experimental_stack=False, warned=True,
            external_cfg_detected=False, steps=0,
            smc_on=smc, apg_on=False, cwm_on=cwm,
        )
        p._DCW.update(
            on=dcw,
            dcw_on=dcw,
            rdc_on=False,
            lambda_low=0.10,
            lambda_high=0.02,
            steps=0,
            dcw_steps=0,
            rdc_steps=0,
        )
        p._STATE.update(
            on=pert, adg_skipped=False, attn_scale=3.0, slg_scale=3.0,
            slg_raw=None, rescale=0.0, apg_autooff_rescale=True,
            delta_logged=True, applied_steps=0,
        )

    def test_all_stages_contribute_together(self):
        """CFG base + PAG perturbation + DCW each change the output when stacked
        — i.e. none is silently dropped by the presence of the others."""
        p = self.pag
        args = self._base_args()
        weak = args["cond_denoised"] - 0.5  # a distinct weak prediction

        self._configure(cwm=True, smc=True, pert=False, dcw=False)
        p._STATE["attn_raw"] = None
        base_only = p._post_cfg(dict(args)).clone()

        self._configure(cwm=True, smc=True, pert=True, dcw=False)
        p._STATE["attn_raw"] = weak.clone()
        base_pert = p._post_cfg(dict(args)).clone()

        self._configure(cwm=True, smc=True, pert=True, dcw=True)
        p._STATE["attn_raw"] = weak.clone()
        base_pert_dcw = p._post_cfg(dict(args)).clone()

        self.assertFalse(
            torch.allclose(base_only, base_pert),
            "perturbation contributed nothing on top of the CFG base",
        )
        self.assertFalse(
            torch.allclose(base_pert, base_pert_dcw),
            "DCW contributed nothing on top of base+perturbation",
        )

    def test_smc_and_cwm_both_applied_when_combined(self):
        """With both toggles on, the CFG base reflects SMC *and* CWM — neither
        silently overrides the other."""
        p = self.pag
        args = self._base_args()

        self._configure(cwm=True, smc=False, pert=False, dcw=False)
        cwm_only = p._apply_cfg_base(args, args["denoised"]).clone()

        self._configure(cwm=False, smc=True, pert=False, dcw=False)
        smc_only = p._apply_cfg_base(args, args["denoised"]).clone()

        self._configure(cwm=True, smc=True, pert=False, dcw=False)
        both = p._apply_cfg_base(args, args["denoised"]).clone()

        self.assertFalse(
            torch.allclose(both, cwm_only),
            "SMC contribution lost when combined with CWM",
        )
        self.assertFalse(
            torch.allclose(both, smc_only),
            "CWM contribution lost when combined with SMC",
        )

    def _run_skim_pag_chain(self, order, attn_scale):
        torch.manual_seed(20260724)
        cond_original = torch.randn(1, 4, 8, 8)
        uncond_original = torch.randn(1, 4, 8, 8)
        live_input = torch.randn(1, 4, 8, 8)
        weak = cond_original * 0.85 + 0.05
        cond = cond_original.clone()
        uncond = uncond_original.clone()
        result = uncond + 7.0 * (cond - uncond)

        self.skim._SKIM.update(
            on=True,
            start=0.0,
            end=1.0,
            skimming_cfg=3.0,
            full_skim_negative=False,
            disable_flipping_filter=False,
            flip_at=0.0,
            warned=True,
            steps=0,
        )
        self._configure(cwm=False, smc=False, pert=True, dcw=False)
        self.pag._STATE.update(
            cond_raw=cond_original.clone(),
            attn_raw=weak.clone(),
            attn_scale=float(attn_scale),
            slg_raw=None,
        )

        for function in order:
            result = function(
                {
                    "denoised": result,
                    "cond_denoised": cond,
                    "uncond_denoised": uncond,
                    "input": live_input,
                    "sigma": torch.tensor([1.0]),
                    "cond_scale": 7.0,
                    "model_options": {},
                }
            )
        return result

    def test_prepend_restores_pag_scale_response_with_skimmed_enabled(self):
        fixed_order = [self.skim._post_cfg, self.pag._post_cfg]
        scale_zero = self._run_skim_pag_chain(fixed_order, 0.0)
        scale_eight = self._run_skim_pag_chain(fixed_order, 8.0)

        self.assertFalse(torch.equal(scale_zero, scale_eight))
        self.assertGreater(
            float(torch.sqrt(torch.mean((scale_eight - scale_zero) ** 2))),
            1e-6,
        )

    def test_old_append_order_erases_pag_scale_response(self):
        old_order = [self.pag._post_cfg, self.skim._post_cfg]
        scale_zero = self._run_skim_pag_chain(old_order, 0.0)
        scale_eight = self._run_skim_pag_chain(old_order, 8.0)

        self.assertTrue(torch.equal(scale_zero, scale_eight))


if __name__ == "__main__":
    unittest.main()
