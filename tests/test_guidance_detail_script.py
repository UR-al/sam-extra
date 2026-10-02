"""The v0.30 detail suite through the real script path (scripts/anima_safe_pag.py).

ui() → process_before_every_sampling → model wrapper / post-CFG, on Forge's real Anima DiT
(``backend/nn/anima.py``, 6 small blocks on the CPU — tests.test_anima_safe_pag helpers).
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import gradio as gr
import torch

from tests.test_anima_safe_pag import _load_pag_module, _load_real_forge_anima

# ui() return positions appended after the 62-argument prefix (append-only contract)
APPENDED = {
    62: ("anima_safe_pag_slg_mode", "Fixed"),
    63: ("anima_safe_pag_s2_scale", 0.25),
    64: ("anima_safe_pag_s2_ratio", 0.05),
    65: ("anima_safe_pag_s2_blocks", ""),
    66: ("anima_safe_pag_s2_start", 0.10),
    67: ("anima_safe_pag_s2_end", 0.90),
    68: ("anima_guidance_smc_mode", "Unit-L2"),
    69: ("anima_guidance_smc_adaptive_alpha", 0.2),
    70: ("anima_guidance_smc_adaptive_lambda", 5.0),
}


class _Unet:
    def __init__(self, dit=None):
        self.model = types.SimpleNamespace(diffusion_model=dit)
        self.model_options = {}

    def clone(self):
        clone = _Unet(self.model.diffusion_model)
        clone.model_options = dict(self.model_options)
        return clone

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def set_model_sampler_pre_cfg_function(self, fn):
        self.model_options["pre_cfg"] = fn

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["post_cfg"] = fn


class DetailSuiteScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def setUp(self):
        self.record = []
        self.anima = _load_real_forge_anima(self.record)
        torch.manual_seed(1234)
        self.dit = self.anima.Anima(
            in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
            model_channels=48, crossattn_emb_channels=32, adaln_lora_dim=16,
            num_blocks=6, num_heads=4,
        ).eval()
        self.process = self.pag.AnimaSafePAG()
        with gr.Blocks():
            self.inputs = self.process.ui(False)
        self.args = [component.value for component in self.inputs]

    def tearDown(self):
        self.pag._teardown_global_patches()

    def _request(self, **extra):
        Anima = type("Anima", (), {})
        model = Anima()
        model.forge_objects = types.SimpleNamespace(unet=_Unet(self.dit))
        return types.SimpleNamespace(
            sd_model=model, extra_generation_params={}, steps=20, seeds=[4321], **extra,
        )

    def _attach(self, request):
        self.process.process_before_every_sampling(request, *self.args)
        return request

    def test_appended_arguments_keep_the_prefix(self):
        for index, (elem_id, value) in APPENDED.items():
            with self.subTest(index=index):
                self.assertEqual(self.inputs[index].elem_id, elem_id)
                self.assertEqual(self.inputs[index].value, value)
        self.assertEqual(self.inputs[61].elem_id, "anima_guidance_rdc_alpha_hh")

    def test_defaults_leave_every_new_stage_off(self):
        request = self._request()
        self.args[0] = True   # PAG on, everything new at its default
        self.args[4] = "3"    # the test DiT has 6 blocks (default 18 is out of range)
        self._attach(request)
        p = self.pag
        self.assertFalse(p._S2["on"])
        self.assertEqual(p._CFG["smc_mode"], "Unit-L2")
        self.assertNotIn("S2", request.extra_generation_params["Anima Perturbation Guidance"])
        self.process.postprocess(request, None)

    def test_s2_draws_a_fresh_block_set_per_evaluation(self):
        p = self.pag
        self.args[0] = True
        self.args[1] = "None"   # SLG/S² only
        self.args[5] = True     # Enable SLG
        self.args[62] = "Stochastic (S²)"
        self.args[63] = 0.5
        self.args[64] = 0.2     # 0.2·5 eligible = 1 block per evaluation
        self.args[66] = 0.0     # window from the first step (the test has no Forge step counter)
        request = self._attach(self._request())
        try:
            self.assertTrue(p._S2["on"])
            self.assertEqual(p._S2["eligible"], {1, 2, 3, 4, 5})
            self.assertEqual(p._STATE["slg_scale"], 0.5)
            text = request.extra_generation_params["Anima Perturbation Guidance"]
            self.assertIn("S2 scale=0.5 ratio=0.2 drop=1/5 eligible=1-5 window=0.00-0.90 seed=4321:base", text)
            self.assertNotIn("SLG scale=", text)

            drawn = []
            for evaluation in range(8):
                g = torch.Generator().manual_seed(evaluation)
                x = torch.randn(2, 4, 1, 8, 10, generator=g)
                ts = torch.full((2,), 0.5)
                c = {"c_crossattn": torch.randn(2, 5, 32, generator=g),
                     "transformer_options": {"cond_or_uncond": []}}

                def apply_model(xx, tt, **cc):
                    with torch.no_grad():
                        return self.dit(xx, tt, context=cc["c_crossattn"],
                                        transformer_options=cc.get("transformer_options", {}))

                out = p._model_wrapper(apply_model, {"input": x, "timestep": ts, "c": c,
                                                     "cond_or_uncond": [0, 1]})
                self.assertEqual(out.shape[0], 2)
                weak, cond = p._STATE["slg_raw"], p._STATE["cond_raw"]
                self.assertIsNotNone(weak)
                self.assertFalse(torch.equal(weak, cond), "the dropped block changes the weak row")
                drawn.append(tuple(sorted(p._STATE["slg_targets"])))
                self.assertEqual(len(drawn[-1]), 1)
                p._RUNTIME.close_step()
            self.assertEqual(p._S2["draws"], 8)
            self.assertGreater(len(set(drawn)), 1, "a fresh mask per evaluation")
            # same seed and pass → same sequence (reproducible infotext)
            expected = [
                tuple(sorted(p.s2_guidance.draw_blocks({1, 2, 3, 4, 5}, 0.2, 4321, "base", i)))
                for i in range(8)
            ]
            self.assertEqual(drawn, expected)
        finally:
            self.process.postprocess(request, None)
        self.assertFalse(p._S2["on"])

    def test_s2_outside_its_window_adds_no_weak_rows(self):
        p = self.pag
        self.args[0] = True
        self.args[1] = "None"
        self.args[5] = True
        self.args[62] = "Stochastic (S²)"
        request = self._attach(self._request())   # default window 0.10-0.90, step 0 → outside
        try:
            x = torch.randn(2, 4, 1, 8, 10)
            seen = []

            def apply_model(xx, tt, **cc):
                seen.append(xx.shape[0])
                return xx

            p._model_wrapper(apply_model, {"input": x, "timestep": torch.full((2,), 0.9),
                                           "c": {}, "cond_or_uncond": [0, 1]})
            self.assertEqual(seen, [2], "no weak row outside the S² window")
            self.assertIsNone(p._STATE["slg_raw"])
        finally:
            self.process.postprocess(request, None)

    def test_adaptive_smc_runs_through_the_post_cfg_orchestrator(self):
        p = self.pag
        self.args[57] = True            # SMC master
        self.args[56] = "Auto"
        self.args[68] = "Adaptive sign"
        self.args[69] = 0.3
        self.args[70] = 4.0
        request = self._attach(self._request())
        try:
            self.assertEqual(p._CFG["smc_mode"], "Adaptive sign")
            self.assertIn(
                "smc=Adaptive sign(alpha=0.3,lambda=4)",
                request.extra_generation_params["Anima CFG Orchestrator"],
            )
            torch.manual_seed(3)
            cond, uncond = torch.randn(1, 4, 1, 8, 8), torch.randn(1, 4, 1, 8, 8)
            sigma = torch.tensor([0.6])
            args = {
                "denoised": uncond + 4.0 * (cond - uncond),
                "cond_denoised": cond, "uncond_denoised": uncond,
                "input": torch.randn(1, 4, 1, 8, 8), "sigma": sigma, "cond_scale": 4.0,
                "model_options": {},
            }
            got = p._post_cfg(args)
            corrected, _ = p.apply_smc_adaptive(cond - uncond, 0.6, None, 0.3, 4.0)
            torch.testing.assert_close(got, uncond + 4.0 * corrected)
            self.assertIsInstance(p._RUNTIME.smc_prev, dict)
        finally:
            self.process.postprocess(request, None)

    def test_short_api_argument_lists_keep_old_behaviour(self):
        # a 62-argument caller (the UR_IV app before this build) gets Fixed SLG and unit-L2 SMC
        p = self.pag
        args = list(self.args[:62])
        args[0] = True
        args[4] = args[7] = "3"   # 6-block test DiT
        args[5] = True
        args[57] = True
        args[56] = "Auto"
        request = self._request()
        self.process.process_before_every_sampling(request, *args)
        try:
            self.assertFalse(p._S2["on"])
            self.assertEqual(p._CFG["smc_mode"], "Unit-L2")
            self.assertIn("SLG scale=3.0 skip=[3]", request.extra_generation_params["Anima Perturbation Guidance"])
        finally:
            self.process.postprocess(request, None)

    def test_xyz_overrides_reach_the_new_knobs(self):
        p = self.pag
        self.args[0] = True
        self.args[5] = True
        request = self._request()
        request._anima_safe_pag_xyz = {
            "slg_mode": "Stochastic (S²)", "s2_scale": "0.75", "s2_ratio": "0.4",
            "s2_blocks": "2-4", "s2_start": "0", "s2_end": "1",
            "smc_mode": "Adaptive sign", "smc_adaptive_alpha": "0.1",
            "smc_adaptive_lambda": "3",
        }
        self.args[57] = True
        self.args[56] = "Auto"
        with mock.patch.object(p, "_s2_in_range", return_value=True):
            self._attach(request)
        try:
            self.assertTrue(p._S2["on"])
            self.assertEqual(p._S2["eligible"], {2, 3, 4})
            self.assertEqual((p._S2["scale"], p._S2["ratio"]), (0.75, 0.4))
            self.assertEqual((p._S2["start"], p._S2["end"]), (0.0, 1.0))
            self.assertEqual(p._CFG["smc_mode"], "Adaptive sign")
            self.assertEqual(
                (p._CFG["smc_adaptive_alpha"], p._CFG["smc_adaptive_lambda"]), (0.1, 3.0)
            )
        finally:
            self.process.postprocess(request, None)


class _FlowModel:
    """KModel stand-in for post-CFG args: a const (rectified-flow) predictor."""

    predictor = types.SimpleNamespace(prediction_type="const",
                                      percent_to_sigma=lambda percent: 1.0 - percent)


class DetailStageScriptTests(unittest.TestCase):
    """TSR / Momentum / HiGS / HiFlow through process_before_every_sampling and _post_cfg."""

    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def setUp(self):
        self.process = self.pag.AnimaSafePAG()
        with gr.Blocks():
            inputs = self.process.ui(False)
        self.args = [component.value for component in inputs]
        self.unet = _Unet(None)
        self.unet.model = types.SimpleNamespace(diffusion_model=None, predictor=_FlowModel.predictor)

    def tearDown(self):
        self.pag._teardown_global_patches()

    def _request(self, **extra):
        Anima = type("Anima", (), {})
        model = Anima()
        model.forge_objects = types.SimpleNamespace(unet=self.unet)
        return types.SimpleNamespace(sd_model=model, extra_generation_params={}, steps=20, seeds=[1], **extra)

    def _post(self, sigma, *, seed=0, schedule=None, cfg=4.0):
        g = torch.Generator().manual_seed(seed)
        cond, uncond = torch.randn(1, 4, 1, 8, 8, generator=g), torch.randn(1, 4, 1, 8, 8, generator=g)
        options = {"transformer_options": {"sampling_sigmas": schedule}} if schedule is not None else {}
        args = {
            "denoised": uncond + cfg * (cond - uncond), "cond_denoised": cond, "uncond_denoised": uncond,
            "input": torch.randn(1, 4, 1, 8, 8, generator=g), "sigma": torch.tensor([sigma]),
            "cond_scale": cfg, "model": _FlowModel(), "model_options": options,
        }
        return args, self.pag._post_cfg(dict(args))

    def test_new_stages_parse_record_and_reach_post_cfg(self):
        a = self.args
        a[71], a[72], a[73] = True, 0.9, 2.0                       # TSR
        a[74], a[75], a[76], a[77], a[78], a[79] = True, 0.7, 0.5, True, 0.0, 1.0   # MG
        a[80], a[81], a[82] = True, 1.5, 0.25                      # HiGS
        request = self._request()
        self.process.process_before_every_sampling(request, *a)
        p = self.pag
        try:
            self.assertTrue(p._TSR["on"] and p._HIST["mg_on"] and p._HIST["higs_on"])
            self.assertIs(p._DETAIL["flow"], True)
            params = request.extra_generation_params
            self.assertEqual(params["Anima TSR"], "k=0.9, sigma=2.0")
            self.assertEqual(params["Anima Momentum Guidance"],
                             "alpha=0.7, beta=0.5, normalize=True, window=0.00-1.00")
            self.assertEqual(params["Anima HiGS"], "w=1.5, eta=0.25, alpha=0.75, cutoff=0.05, t=0.40-1.00")
            schedule = torch.tensor([1.0, 0.8, 0.6, 0.3, 0.2, 0.0])
            outs = [self._post(s, seed=i, schedule=schedule) for i, s in enumerate([0.8, 0.6, 0.3])]
            for args, out in outs:
                self.assertFalse(torch.equal(out, args["denoised"]), "TSR changes every evaluation")
            self.assertEqual(p._TSR["steps"], 3)
            self.assertEqual(p._RUNTIME.history.counters["mg"], 2, "MG: first evaluation only seeds")
            self.assertEqual(p._RUNTIME.history.counters["higs"], 1, "HiGS: seeds, then σ 0.3 < t_min → w 0")
            # a midpoint (off the schedule) leaves the history alone
            before = p._RUNTIME.history.mg_m.clone()
            self._post(0.5, seed=9, schedule=schedule)
            torch.testing.assert_close(p._RUNTIME.history.mg_m, before)
        finally:
            self.process.postprocess(request, None)
        self.assertFalse(p._TSR["on"] or p._HIST["mg_on"] or p._HIST["higs_on"])

    def test_tsr_alone_matches_the_module(self):
        self.args[71], self.args[72], self.args[73] = True, 0.93, 3.0
        request = self._request()
        self.process.process_before_every_sampling(request, *self.args)
        try:
            args, out = self._post(0.5)
            expected = self.pag.tsr_guidance.apply_tsr(args["denoised"], args["input"], args["sigma"],
                                                       k=0.93, tsr_sigma=3.0, flow=True)
            torch.testing.assert_close(out, expected)
        finally:
            self.process.postprocess(request, None)

    def test_diagnostics_report_a_detail_stage_on_alone(self):
        # Before 2026-10-02 the summary printed only when a pre-v0.30 feature was requested, so
        # a TSR/MG/HiGS/HiFlow-only run logged nothing (found on the first real Forge run).
        self.args[71] = True                                   # TSR only
        request = self._request()
        self.process.process_before_every_sampling(request, *self.args)
        with mock.patch.object(self.pag, "guidance_diagnostics_enabled", return_value=True), \
                mock.patch.object(self.pag, "_log") as log:
            self._post(0.5)
            self.process.postprocess(request, None)
        lines = [str(call.args[0]) for call in log.call_args_list if call.args]
        self.assertTrue(any(line.startswith("[VERIFY] detail: TSR=APPLIED(1 evals)") for line in lines), lines)

    def test_mg_higs_eval_counts_restart_every_generation(self):
        # 2026-10-03 GPU run: the [VERIFY] MG/HiGS counts added up over one Forge session (MG 9, 18, 27 …)
        # because reset_pass kept the history's counters. Two identical generations report the same counts.
        self.args[74], self.args[80] = True, True              # MG, HiGS with their defaults
        schedule = torch.tensor([1.0, 0.8, 0.6, 0.3, 0.2, 0.0])
        reports = []
        for _ in range(2):
            request = self._request()
            self.process.process_before_every_sampling(request, *self.args)
            with mock.patch.object(self.pag, "guidance_diagnostics_enabled", return_value=True), \
                    mock.patch.object(self.pag, "_log") as log:
                for i, sigma in enumerate([0.8, 0.6, 0.3]):
                    self._post(sigma, seed=i, schedule=schedule)
                self.process.postprocess(request, None)
            reports.append(next(str(call.args[0]) for call in log.call_args_list
                                if call.args and str(call.args[0]).startswith("[VERIFY] detail:")))
        self.assertEqual(reports[0], reports[1])
        self.assertIn("MG=APPLIED(2 evals)", reports[1])           # the first evaluation only seeds
        self.assertIn("HiGS=APPLIED(1 evals)", reports[1])         # seeds, applies at 0.6, σ 0.3 < t_min

    def test_hiflow_records_the_base_pass_and_aligns_its_hires_pass(self):
        p = self.pag
        self.args[87] = True
        request = self._request(enable_hr=True, is_hr_pass=False)
        self.process.process_before_every_sampling(request, *self.args)
        try:
            self.assertTrue(p._HIFLOW["recording"])
            self.assertNotIn("Anima HiFlow", request.extra_generation_params)
            for i, sigma in enumerate([1.0, 0.75, 0.5, 0.25]):
                p._hiflow_run_callback(types.SimpleNamespace(denoiser=types.SimpleNamespace(p=request)))
                self._post(sigma, seed=i)
            self.assertEqual(len(p._HIFLOW["trajectory"]), 4)
            # an inner run of another request (ADetailer) does not record
            p._hiflow_run_callback(types.SimpleNamespace(denoiser=types.SimpleNamespace(p=object())))
            self._post(0.6, seed=7)
            self.assertEqual(len(p._HIFLOW["trajectory"]), 4)

            request.is_hr_pass = True
            self.process.process_before_every_sampling(request, *self.args)
            self.assertTrue(p._HIFLOW["applying"])
            self.assertFalse(p._HIFLOW["recording"])
            self.assertEqual(request.extra_generation_params["Anima HiFlow"],
                             "alpha=1.0, beta=0.5, cutoff=0.2, reference=4 sigmas")
            hires_schedule = torch.tensor([0.6, 0.4, 0.2, 0.0])
            p._hiflow_run_callback(types.SimpleNamespace(denoiser=types.SimpleNamespace(p=request)))
            args, out = self._post(0.6, seed=11, schedule=hires_schedule)
            self.assertFalse(torch.equal(out, args["denoised"]))
            reference = p._HIFLOW["trajectory"].at(0.6)
            expected = p.hiflow_guidance.apply_hiflow(
                args["denoised"], 0.6, reference, p.hiflow_guidance.HiFlowState(),
                alpha=1.0, beta=0.5, cutoff=0.2, weight=1.0, step_start=True)
            torch.testing.assert_close(out, expected)
            self._post(0.4, seed=12, schedule=hires_schedule)
            self.assertEqual(p._HIFLOW["state"].counters["acceleration"], 1)
            # another request's run during the hires pass is left alone
            p._hiflow_run_callback(types.SimpleNamespace(denoiser=types.SimpleNamespace(p=object())))
            args, out = self._post(0.4, seed=13, schedule=hires_schedule)
            torch.testing.assert_close(out, args["denoised"])
        finally:
            self.process.postprocess(request, None)
        self.assertEqual(len(p._HIFLOW["trajectory"]), 0, "postprocess drops the trajectory")

    def test_hiflow_without_hires_fix_does_nothing(self):
        p = self.pag
        self.args[87] = True
        request = self._request(enable_hr=False, is_hr_pass=False)
        self.process.process_before_every_sampling(request, *self.args)
        try:
            self.assertFalse(p._HIFLOW["recording"] or p._HIFLOW["applying"])
            self.assertIsNone(request.sd_model.forge_objects.unet.model_options.get("post_cfg"),
                              "nothing else on → no hook attached")
        finally:
            self.process.postprocess(request, None)


if __name__ == "__main__":
    unittest.main()
