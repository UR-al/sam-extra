"""The extra samplers under the guidance stack, base pass + hires pass, on Forge's real CFG path.

Every entry of ``registry.SPECS`` (ER SDE, DPM++ 4M SDE with Forge's per-image Brownian tree, the Dy family, the
flow ODEs, UniPC bh2, CFG++ UD10 AB, IPNDM/IPNDM_V/DEIS, Restart (flow)), each with PAG, adaptive SMC, DCW + RDC, APG momentum, TSR, Momentum Guidance, HiGS
and HiFlow (the base pass of a hires request records its trajectory, the hires pass aligns to it), with
Detail Daemon scaling the sigmas:

* no crash, finite results, the step-keyed state at the pass's own size — the Dy/SMEA sub-steps
  (marked ``sam_extra_substep``) never enter the HiFlow trajectory, the Momentum/HiGS history, SMC's
  error, APG's momentum or RDC's average;
* no state leaks: a generation gives the same image whether or not another one ran before it;
* a batch image equals the same seed alone with every per-image stage. Adaptive SMC is the one
  deliberate exception — it computes one gain over the whole batch like sorryhyun's original
  (``sam3ext.guidance.cwm_smc.apply_smc_adaptive``), so it is checked for leaks, not for batch parity.

HiFlow keeps its trajectory in float16; the batch comparison stores it in float32 so the float noise of
a batched matmul is not rounded into a visible difference (that is storage precision, not coupling).
IPNDM_V's variable-step weights (about 3.0, -5.5, 5.6, -2.2 late on a shift-3 grid, against AB4's fixed
2.29, -2.46, 1.54, -0.38) amplify that float noise about 5x more than IPNDM's, so its bound is 5x wider
(measured 1.34e-4 in 1 of 4096 values). Coupling is ruled out exactly elsewhere: with an elementwise model
every deterministic entry, IPNDM_V included, keeps each batch image bit-exact
(``test_extra_samplers_forge_path...test_the_deterministic_entries_keep_every_batch_image_bit_exact``).
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import unittest

import torch

from sam3ext.extra_samplers import registry
from tests import _integration_support as I

PER_IMAGE = {
    "anima_safe_pag_enable": True, "anima_safe_pag_blocks": "3", "anima_safe_pag_end": 1.0,
    "anima_guidance_dcw_enable": True, "anima_guidance_rdc_tau": 0.5,
    "anima_safe_pag_apg_enable": True, "anima_safe_pag_apg_momentum": 0.5,
    "anima_guidance_tsr_enable": True,
    "anima_guidance_mg_enable": True, "anima_guidance_mg_min": 0.0, "anima_guidance_mg_max": 1.0,
    "anima_guidance_higs_enable": True, "anima_guidance_higs_t_min": 0.0,
    "anima_guidance_hiflow_enable": True,
}
ADAPTIVE_SMC = {"anima_guidance_smc_master_enable": True, "anima_guidance_smc_mode": "Adaptive sign"}
UNIT_SMC = {"anima_guidance_smc_master_enable": True, "anima_guidance_smc_mode": "Unit-L2"}
# batch-vs-alone float-noise bound per label (module docstring); every other entry keeps 1e-4
BATCH_NOISE_BOUND = {registry.LABEL_IPNDM_V: 5e-4}


def runtime(spec):
    return lambda request: registry.runtime_kwargs(spec, request)


class SamplersUnderTheStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        I.require_forge()
        cls.scripts = I.Scripts()
        cls.h = I.Harness(cls.scripts)

    def tearDown(self):
        self.scripts.teardown()

    def _generation(self, spec, *, seeds=(31,), stack=None):
        """Base pass (hires enabled: HiFlow records) and its hires pass; then ``postprocess``."""
        pag = self.scripts.pag_args(**(stack or {**PER_IMAGE, **ADAPTIVE_SMC}))
        brownian = bool(spec.options.get("brownian_noise"))
        kwargs = {"runtime": runtime(spec)}

        def hires_on(request):
            request.enable_hr = True

        base = self.h.generate(spec.func, seeds=seeds, prepare=hires_on, brownian=brownian,
                               hooks={"pag": pag, "dd": I.dd_args()}, sampler_kwargs=kwargs)
        trajectory = self.scripts.pag._HIFLOW["trajectory"]
        recorded = (len(trajectory), trajectory.shape)
        hires = self.h.hires(base.request, spec.func, base.out, brownian=brownian,
                             hooks={"pag": pag, "dd": I.dd_args(hires=True)}, sampler_kwargs=kwargs)
        state = self._state()
        self.h.finish(base.request)
        return base, hires, recorded, state

    def _state(self):
        pag = self.scripts.pag
        runtime_state = pag._RUNTIME
        rdc = {k: tuple(v.shape) for k, v in (runtime_state.rdc_state or {}).items() if torch.is_tensor(v)}
        smc = runtime_state.smc_prev
        if isinstance(smc, dict):          # adaptive SMC keeps {"e": error, "sigma": ...}
            smc = smc.get("e")

        def shape(value):
            return tuple(value.shape) if torch.is_tensor(value) else None

        return {
            "hiflow": dict(pag._HIFLOW["state"].counters),
            "mg": shape(runtime_state.history.mg_m),
            "higs": shape(runtime_state.history.higs_g),
            "smc": shape(smc),
            "apg": shape(pag._APG["avg"]),
            "rdc": rdc,
        }

    def test_base_and_hires_with_every_stage(self):
        for spec in registry.SPECS:
            with self.subTest(sampler=spec.label):
                base, hires, recorded, state = self._generation(spec)
                self.scripts.teardown()
                self.assertTrue(torch.isfinite(base.out).all())
                self.assertTrue(torch.isfinite(hires.out).all())
                self.assertEqual(tuple(hires.out.shape), (1, 16, 1, 24, 24))
                substeps = [c for c in base.calls + hires.calls if c.marker]
                self.assertEqual(len(substeps), 4 if spec.kind == "dy" else 0)
                # HiFlow: one full-size record per base step, none from a sub-step
                self.assertEqual(recorded, (10, (1, 16, 1, 16, 16)))
                self.assertGreater(state["hiflow"]["direction"], 0)
                full = (1, 16, 1, 24, 24)
                for name in ("mg", "higs", "smc", "apg"):
                    self.assertEqual(state[name], full, name)
                self.assertTrue(state["rdc"])
                self.assertTrue(all(shape[0] == 1 and shape[-1] == 12 for shape in state["rdc"].values()), state["rdc"])

    def test_no_state_leaks_between_generations(self):
        for spec in registry.SPECS:
            with self.subTest(sampler=spec.label):
                _, alone, _, _ = self._generation(spec, seeds=(77,))
                self.scripts.teardown()
                self._generation(spec, seeds=(31,))                       # another generation first
                _, after, _, _ = self._generation(spec, seeds=(77,))
                self.scripts.teardown()
                torch.testing.assert_close(after.out, alone.out, atol=0, rtol=0)

    def test_batch_image_equals_the_same_seed_alone(self):
        trajectory = self.scripts.pag._HIFLOW["trajectory"]
        previous = trajectory.storage_dtype
        trajectory.storage_dtype = torch.float32
        try:
            for spec in registry.SPECS:
                with self.subTest(sampler=spec.label):
                    stack = {**PER_IMAGE, **UNIT_SMC}
                    base, hires, _, _ = self._generation(spec, seeds=(31, 32), stack=stack)
                    self.scripts.teardown()
                    bound = BATCH_NOISE_BOUND.get(spec.label, 1e-4)
                    for index, seed in enumerate((31, 32)):
                        single_base, single_hires, _, _ = self._generation(spec, seeds=(seed,), stack=stack)
                        self.scripts.teardown()
                        torch.testing.assert_close(base.out[index:index + 1], single_base.out, atol=bound, rtol=bound)
                        torch.testing.assert_close(hires.out[index:index + 1], single_hires.out, atol=bound, rtol=bound)
        finally:
            trajectory.storage_dtype = previous

    def test_adaptive_smc_gain_is_one_for_the_batch(self):
        """By design (the original's single gain over the whole tensor), not a leak: excluded above."""
        spec = next(s for s in registry.SPECS if s.label == registry.LABEL_EULER_DY_CFG_PP)
        stack = {**PER_IMAGE, **ADAPTIVE_SMC}
        batch, _, _, _ = self._generation(spec, seeds=(31, 32), stack=stack)
        self.scripts.teardown()
        single, _, _, _ = self._generation(spec, seeds=(31,), stack=stack)
        self.assertGreater(float((batch.out[:1] - single.out).abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
