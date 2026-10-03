"""Euler (SMEA) Dy CFG++ sub-step evaluations and the Anima Guidance detail stages.

The Dy/SMEA sub-steps call the CFG denoiser a second time in a step, at half or ×1.25 resolution, with
``transformer_options["sam_extra_substep"]`` set (``sam3ext.extra_samplers.common.SUBSTEP_MARKER``).
The step-keyed detail stages of ``scripts/anima_safe_pag.py`` ignore those evaluations instead of
restarting on the shape change: the HiFlow base-pass trajectory keeps its full-resolution records
(``Trajectory.record`` clears the store on a new shape), HiFlow alignment and the Momentum/HiGS history
leave them alone, and stateless TSR still rescales them. Same harness as
``tests/test_guidance_detail_script.py`` (the real script's ``ui`` → ``process_before_every_sampling``
→ ``_post_cfg``).
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import types
import unittest

import gradio as gr
import torch

from sam3ext.extra_samplers.common import SUBSTEP_MARKER
from sam3ext.guidance.cwm_smc import compose_cfg
from sam3ext.guidance.dcw import apply_dcw
from tests.test_anima_safe_pag import _load_pag_module

# ui() positions (tests/test_guidance_detail_script.py)
TSR_ON, MG_ON, HIGS_ON, HIFLOW_ON = 71, 74, 80, 87

FULL = (1, 4, 1, 8, 8)
HALF = (1, 4, 1, 4, 4)


class _FlowModel:
    predictor = types.SimpleNamespace(prediction_type="const", percent_to_sigma=lambda percent: 1.0 - percent)


class _Unet:
    def __init__(self):
        self.model = types.SimpleNamespace(diffusion_model=None, predictor=_FlowModel.predictor)
        self.model_options = {}

    def clone(self):
        clone = _Unet()
        clone.model_options = dict(self.model_options)
        return clone

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def set_model_sampler_pre_cfg_function(self, fn):
        self.model_options["pre_cfg"] = fn

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["post_cfg"] = fn


class SubstepDetailStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def setUp(self):
        self.process = self.pag.AnimaSafePAG()
        with gr.Blocks():
            inputs = self.process.ui(False)
        self.args = [component.value for component in inputs]
        self.unet = _Unet()

    def tearDown(self):
        self.pag._teardown_global_patches()

    def _request(self, **extra):
        model = type("Anima", (), {})()
        model.forge_objects = types.SimpleNamespace(unet=self.unet)
        return types.SimpleNamespace(sd_model=model, extra_generation_params={}, steps=20, seeds=[1], **extra)

    def _post(self, sigma, *, seed=0, shape=FULL, substep=None, schedule=None):
        g = torch.Generator().manual_seed(seed)
        cond, uncond = torch.randn(shape, generator=g), torch.randn(shape, generator=g)
        transformer_options = {}
        if schedule is not None:
            transformer_options["sampling_sigmas"] = schedule
        if substep is not None:
            transformer_options[SUBSTEP_MARKER] = substep
        args = {
            "denoised": uncond + 4.0 * (cond - uncond), "cond_denoised": cond, "uncond_denoised": uncond,
            "input": torch.randn(shape, generator=g), "sigma": torch.tensor([sigma]), "cond_scale": 4.0,
            "model": _FlowModel(), "model_options": {"transformer_options": transformer_options},
        }
        return args, self.pag._post_cfg(dict(args))

    def _base_pass_with_hiflow(self):
        self.args[HIFLOW_ON] = True
        request = self._request(enable_hr=True, is_hr_pass=False)
        self.process.process_before_every_sampling(request, *self.args)
        self.addCleanup(self.process.postprocess, request, None)
        return request

    def _run_callback(self, request):
        self.pag._hiflow_run_callback(types.SimpleNamespace(denoiser=types.SimpleNamespace(p=request)))

    def test_the_marker_is_read_from_model_options(self):
        self.assertTrue(self.pag._is_extra_sampler_substep(
            {"model_options": {"transformer_options": {SUBSTEP_MARKER: "dy"}}}))
        self.assertFalse(self.pag._is_extra_sampler_substep({"model_options": {"transformer_options": {}}}))
        self.assertFalse(self.pag._is_extra_sampler_substep({}))

    def test_hiflow_base_trajectory_keeps_its_records_through_substeps(self):
        p = self.pag
        request = self._base_pass_with_hiflow()
        self._run_callback(request)
        for i, sigma in enumerate([1.0, 0.75, 0.5]):
            self._post(sigma, seed=i)
            if sigma == 0.75:   # Euler Dy's steps 2-3 / SMEA Dy's steps 0-1: an extra evaluation in the step
                self._post(sigma, seed=10 + i, shape=HALF, substep="dy")
        self._post(0.25, seed=3)
        self.assertEqual(len(p._HIFLOW["trajectory"]), 4)
        self.assertEqual(p._HIFLOW["trajectory"].shape, FULL)

    def test_an_unmarked_resolution_change_still_restarts_the_trajectory(self):
        """What the marker prevents: Trajectory.record starts over on a new shape."""
        p = self.pag
        request = self._base_pass_with_hiflow()
        self._run_callback(request)
        for i, sigma in enumerate([1.0, 0.75, 0.5]):
            self._post(sigma, seed=i)
        self._post(0.5, seed=9, shape=HALF)
        self.assertEqual(len(p._HIFLOW["trajectory"]), 1)

    def test_hiflow_alignment_skips_substeps_in_the_hires_pass(self):
        p = self.pag
        request = self._base_pass_with_hiflow()
        self._run_callback(request)
        for i, sigma in enumerate([1.0, 0.75, 0.5, 0.25]):
            self._post(sigma, seed=i)
        request.is_hr_pass = True
        self.process.process_before_every_sampling(request, *self.args)
        self.assertTrue(p._HIFLOW["applying"])
        self._run_callback(request)
        schedule = torch.tensor([0.6, 0.4, 0.2, 0.0])
        args, out = self._post(0.6, seed=20, schedule=schedule)
        self.assertFalse(torch.equal(out, args["denoised"]))
        state = p._HIFLOW["state"]
        before = (dict(state.counters), state.prev_sigma)
        args, out = self._post(0.6, seed=21, shape=HALF, substep="smea", schedule=schedule)
        torch.testing.assert_close(out, args["denoised"])
        self.assertEqual((dict(state.counters), state.prev_sigma), before)

    def test_momentum_and_higs_history_ignore_substeps(self):
        p = self.pag
        self.args[MG_ON], self.args[HIGS_ON] = True, True
        request = self._request()
        self.process.process_before_every_sampling(request, *self.args)
        self.addCleanup(self.process.postprocess, request, None)
        schedule = torch.tensor([1.0, 0.8, 0.6, 0.3, 0.2, 0.0])
        self._post(0.8, seed=1, schedule=schedule)
        self._post(0.6, seed=2, schedule=schedule)
        history = p._RUNTIME.history
        snapshot = (history.mg_m.clone(), dict(history.counters))
        args, out = self._post(0.6, seed=3, shape=HALF, substep="dy", schedule=schedule)
        torch.testing.assert_close(out, args["denoised"])
        torch.testing.assert_close(history.mg_m, snapshot[0])
        self.assertEqual(dict(history.counters), snapshot[1])
        args, out = self._post(0.3, seed=4, schedule=schedule)      # the next step carries on
        self.assertEqual(history.counters["mg"], snapshot[1]["mg"] + 1)

    def test_tsr_still_rescales_a_substep(self):
        self.args[TSR_ON] = True
        request = self._request()
        self.process.process_before_every_sampling(request, *self.args)
        self.addCleanup(self.process.postprocess, request, None)
        args, out = self._post(0.5, seed=5, shape=HALF, substep="dy")
        expected = self.pag.tsr_guidance.apply_tsr(args["denoised"], args["input"], args["sigma"],
                                                   k=float(self.pag._TSR["k"]), tsr_sigma=float(self.pag._TSR["sigma"]),
                                                   flow=True)
        torch.testing.assert_close(out, expected)
        self.assertFalse(torch.equal(out, args["denoised"]))


# ---------------------------------------------------------------------------
# SMC e_prev, APG momentum, RDC moving average
# ---------------------------------------------------------------------------

_STATEFUL_CONFIGS = (
    ("SMC unit-L2", dict(smc="unit")),
    ("SMC adaptive", dict(smc="adaptive")),
    ("APG momentum", dict(apg=True)),
    ("SMC + APG momentum", dict(smc="unit", apg=True)),
    ("RDC", dict(rdc=True)),
    ("DCW + RDC", dict(rdc=True, dcw=True)),
    ("all of them", dict(smc="adaptive", apg=True, rdc=True, dcw=True)),
)
SCALE = 4.0


class SubstepCfgStateTests(unittest.TestCase):
    """The step-keyed CFG state of ``_post_cfg`` — SMC ``e_prev`` (``_RUNTIME.smc_prev``), APG momentum
    (``_APG["avg"]``/``["last_sigma"]``) and the RDC moving average (``_RUNTIME.rdc_state``) — is read
    but neither updated nor reset by a marked Dy/SMEA sub-step; without the marker nothing changes."""

    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def tearDown(self):
        self.pag._teardown_global_patches()

    def _configure(self, *, smc=None, apg=False, rdc=False, dcw=False):
        p = self.pag
        p._RUNTIME.reset_pass()
        p._CFG.update(
            mode="preserve", experimental_stack=False, warned=True, external_cfg_detected=False, steps=0,
            fit_checked=True, smc_on=smc is not None, apg_on=apg, cwm_on=False, alpha_low=0.0, alpha_high=0.0,
            smc_lambda=6.0, smc_k=0.1, smc_mode=p.SMC_MODE_ADAPTIVE if smc == "adaptive" else p.SMC_MODE_UNIT,
        )
        p._APG.update(on=apg, eta=0.5, norm_threshold=0.0, momentum=0.6 if apg else 0.0, avg=None, last_sigma=None)
        p._DCW.update(
            on=rdc or dcw, dcw_on=dcw, rdc_on=rdc, lambda_low=0.1, lambda_high=0.02,
            rdc_tau=0.5, rdc_alpha_ll=0.3, rdc_alpha_hh=0.2, steps=0, dcw_steps=0, rdc_steps=0,
        )
        p._STATE.update(on=False, adg_skipped=False)
        p._ADG.update(on=False)

    @staticmethod
    def _args(sigma, seed, shape=FULL, substep=None):
        g = torch.Generator().manual_seed(seed)
        cond, uncond, x = (torch.randn(shape, generator=g) for _ in range(3))
        transformer_options = {SUBSTEP_MARKER: substep} if substep else {}
        return {
            "denoised": uncond + SCALE * (cond - uncond), "cond_denoised": cond, "uncond_denoised": uncond,
            "input": x, "sigma": torch.tensor([sigma]), "cond_scale": SCALE, "model": _FlowModel(),
            "model_options": {"transformer_options": transformer_options},
        }

    def _post(self, sigma, seed, shape=FULL, substep=None):
        return self.pag._post_cfg(self._args(sigma, seed, shape, substep))

    def _state(self):
        p = self.pag
        return {
            "smc": p._RUNTIME.smc_prev, "apg": p._APG["avg"], "apg_sigma": p._APG["last_sigma"],
            "rdc": dict(p._RUNTIME.rdc_state),
        }

    def _assert_untouched(self, before, after):
        self.assertIs(after["smc"], before["smc"])
        self.assertIs(after["apg"], before["apg"])
        self.assertEqual(after["apg_sigma"], before["apg_sigma"])
        self.assertEqual(set(after["rdc"]), set(before["rdc"]))
        for key, value in before["rdc"].items():
            self.assertIs(after["rdc"][key], value, key)

    def test_a_marked_substep_leaves_no_trace(self):
        for name, config in _STATEFUL_CONFIGS:
            with self.subTest(name):
                self._configure(**config)
                plain = [self._post(0.9, 1), self._post(0.7, 2), self._post(0.5, 3)]
                plain_state = self._state()
                self.assertTrue(
                    plain_state["smc"] is not None or plain_state["apg"] is not None or plain_state["rdc"],
                    "the configuration keeps no state - nothing to test",
                )

                self._configure(**config)
                mixed = [self._post(0.9, 1)]
                before = self._state()
                self._post(0.9, 11, HALF, "smea")          # SMEA Dy: ×1.25 / half sub-steps
                self._assert_untouched(before, self._state())
                mixed.append(self._post(0.7, 2))
                before = self._state()
                self._post(0.7, 12, HALF, "dy")
                self._assert_untouched(before, self._state())
                mixed.append(self._post(0.5, 3))

                for step, (a, b) in enumerate(zip(plain, mixed)):
                    self.assertTrue(torch.equal(a, b), f"step {step}")
                final = self._state()
                if torch.is_tensor(plain_state["apg"]):
                    self.assertTrue(torch.equal(final["apg"], plain_state["apg"]))
                self.assertEqual(set(final["rdc"]), set(plain_state["rdc"]))

    def test_an_adaptive_guidance_substep_does_not_reset_apg_momentum(self):
        p = self.pag
        self._configure(apg=True)
        self._post(0.9, 1)
        momentum = p._APG["avg"]
        self.assertIsNotNone(momentum)
        # post-CFG: a cond-only (ADG-skipped) evaluation
        p._STATE["adg_skipped"] = True
        self._post(0.9, 11, HALF, "dy")
        self.assertIs(p._APG["avg"], momentum)
        p._STATE["adg_skipped"] = True
        self._post(0.7, 2)
        self.assertIsNone(p._APG["avg"])                   # the unmarked cond-only step resets, as before

        # the UNet wrapper's ADG path (``_model_wrapper_inner``)
        p._APG["avg"] = momentum
        p._ADG.update(on=True, start=0.0, interval=0)
        p._STATE.update(step=5, total=10)

        def wrapper_call(substep):
            transformer_options = {SUBSTEP_MARKER: substep} if substep else {}
            w = {"input": torch.randn(2, 4, 1, 4, 4), "timestep": torch.tensor([0.5, 0.5]),
                 "c": {"transformer_options": transformer_options}, "cond_or_uncond": [0, 1]}
            out = p._model_wrapper_inner(lambda x, t, **c: x * 0.5, w)
            p._RUNTIME.close_step()
            return out

        wrapper_call("dy")
        self.assertTrue(p._STATE["adg_skipped_steps"] >= 1)
        self.assertIs(p._APG["avg"], momentum)
        wrapper_call(None)
        self.assertIsNone(p._APG["avg"])

    def test_without_the_marker_the_state_chain_is_the_documented_one(self):
        """No marker: the stages compute and store exactly what their library functions return."""
        p = self.pag
        plan = ((0.9, 1), (0.7, 2), (0.5, 3))

        self._configure(smc="unit")
        previous = None
        for sigma, seed in plan:
            args = self._args(sigma, seed)
            out = p._post_cfg(dict(args))
            expected, previous = compose_cfg(
                cond=args["cond_denoised"].float(), uncond=args["uncond_denoised"].float(), sigma=args["sigma"],
                effective_scale=SCALE, mode="smc", alpha_low=0.0, alpha_high=0.0, smc_lambda=6.0, smc_k=0.1,
                smc_previous=previous, smc_mode=p.SMC_MODE_UNIT, smc_sigma=None,
                smc_alpha=float(p._CFG["smc_adaptive_alpha"]),
            )
            self.assertTrue(torch.equal(out, expected))
            self.assertTrue(torch.equal(p._RUNTIME.smc_prev, previous))

        self._configure(rdc=True)
        reference = {}
        for sigma, seed in plan:
            args = self._args(sigma, seed)
            out = p._post_cfg(dict(args))
            expected = apply_dcw(args["denoised"].float(), args["input"], args["sigma"], 0.0, 0.0,
                                 rdc_tau=0.5, rdc_alpha_ll=0.3, rdc_alpha_hh=0.2, rdc_state=reference)
            self.assertTrue(torch.equal(out, expected))
            self.assertEqual(set(p._RUNTIME.rdc_state), set(reference))
            for key, value in reference.items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(p._RUNTIME.rdc_state[key], value), key)
                else:
                    self.assertEqual(p._RUNTIME.rdc_state[key], value, key)

        self._configure(apg=True)
        average = None
        for sigma, seed in plan:
            args = self._args(sigma, seed)
            p._post_cfg(dict(args))
            guidance = args["cond_denoised"].float() - args["uncond_denoised"].float()
            average = guidance if average is None else 0.6 * average + guidance
            torch.testing.assert_close(p._APG["avg"], average, rtol=0, atol=1e-6)
            self.assertEqual(p._APG["last_sigma"], float(args["sigma"][0]))

    def test_an_unmarked_resolution_change_still_restarts_the_state(self):
        p = self.pag
        self._configure(smc="adaptive", apg=True, rdc=True, dcw=True)
        self._post(0.9, 1)
        before = self._state()
        self._post(0.9, 11, HALF)                          # e.g. another sampler's resolution change
        after = self._state()
        self.assertIsNot(after["smc"], before["smc"])
        self.assertEqual(tuple(after["smc"]["e"].shape), HALF)
        self.assertEqual(tuple(after["apg"].shape), HALF)
        self.assertTrue(any(torch.is_tensor(v) and v is not before["rdc"].get(k) for k, v in after["rdc"].items()))


if __name__ == "__main__":
    unittest.main()
