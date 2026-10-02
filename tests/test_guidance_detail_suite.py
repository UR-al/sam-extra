"""Math tests for the v0.30 detail suite: S²-Guidance masks, adaptive SMC, TSR, MG/HiGS, HiFlow.

Each stage is checked against an independent reference written from the paper or the origin code
(quoted in the stage module's docstring), plus the identities that keep a disabled stage a no-op.
"""

from __future__ import annotations

import math
import random
import unittest

import torch

from sam3ext.guidance import hiflow, history, s2
from sam3ext.guidance.cwm_smc import (
    SMC_MODE_ADAPTIVE,
    SMC_MODE_UNIT,
    apply_smc_adaptive,
    apply_smc_error,
    compose_cfg,
    normalize_smc_mode,
)
from sam3ext.guidance.sigmas import half_log_snr, is_flow_model, sampler_sigma, schedule_index
from sam3ext.guidance.trajectory import Trajectory
from sam3ext.guidance.tsr import apply_tsr, rescale_factors


class S2MaskTests(unittest.TestCase):
    def test_count_scales_with_depth_and_keeps_one_block(self):
        # default ratio 0.05 → 1 / 2 / 3 blocks on Anima's 28 / 40 / 52-block models (block 0 excluded)
        for blocks, expected in ((28, 1), (40, 2), (52, 3)):
            with self.subTest(blocks=blocks):
                eligible = s2.default_eligible(blocks)
                self.assertNotIn(0, eligible)
                self.assertEqual(len(eligible), blocks - 1)
                self.assertEqual(s2.count_for(s2.DEFAULT_RATIO, len(eligible)), expected)
        self.assertEqual(s2.count_for(0.001, 27), 1, "at least one block while ratio > 0")
        self.assertEqual(s2.count_for(0.0, 27), 0)
        self.assertEqual(s2.count_for(0.5, 0), 0)
        self.assertEqual(s2.count_for(1.0, 5), 5)

    def test_draw_is_reproducible_and_changes_per_evaluation(self):
        eligible = s2.default_eligible(28)
        first = [s2.draw_blocks(eligible, 0.1, 1234, "base", draw) for draw in range(20)]
        again = [s2.draw_blocks(eligible, 0.1, 1234, "base", draw) for draw in range(20)]
        self.assertEqual(first, again)
        self.assertGreater(len({tuple(sorted(m)) for m in first}), 5, "a fresh mask per evaluation")
        for mask in first:
            self.assertEqual(len(mask), s2.count_for(0.1, 27))
            self.assertTrue(mask <= eligible)
        other_seed = [s2.draw_blocks(eligible, 0.1, 1235, "base", draw) for draw in range(20)]
        hires = [s2.draw_blocks(eligible, 0.1, 1234, "hires", draw) for draw in range(20)]
        self.assertNotEqual(first, other_seed)
        self.assertNotEqual(first, hires)

    def test_draw_matches_a_string_seeded_random_sample(self):
        # the documented rule: random.Random(str key).sample(sorted pool, count)
        eligible = {3, 9, 1, 27, 14}
        expected = set(random.Random("s2:42:base:7").sample([1, 3, 9, 14, 27], 2))
        self.assertEqual(s2.draw_blocks(eligible, 0.4, 42, "base", 7), expected)


def _sorryhyun_v_space(states, x, x0_c, x0_u, sigma, w, alpha, lam):
    """Reference: sorryhyun SMCCFGState.combine in velocity space + the ComfyUI x0<->v seam.

    origin (MIT): sorryhyun/anima_lora library/inference/corrections/smc_cfg.py and
    sorryhyun/ComfyUI-Spectrum-KSampler smc_cfg.py — v = (x - x0)/σ, e = v_c - v_u, e_prev raw,
    s = (e - e_prev) + λ·e_prev, k = α·mean|e|, v̂ = v_u + w·(e - k·sign(s)), x0̂ = x - σ·v̂.
    """
    v_c = (x - x0_c) / sigma
    v_u = (x - x0_u) / sigma
    e = v_c - v_u
    e_prev = e if states.get("e") is None or states["e"].shape != e.shape else states["e"]
    s = (e - e_prev) + lam * e_prev
    k = alpha * e.abs().mean().clamp_min(1e-12)
    states["e"] = e.detach()
    v_hat = v_u + w * (e - k * torch.sign(s))
    return x - sigma * v_hat


class AdaptiveSmcTests(unittest.TestCase):
    def test_matches_the_velocity_space_reference_across_steps(self):
        torch.manual_seed(0)
        shape = (2, 16, 1, 12, 10)
        w, alpha, lam = 4.5, 0.2, 5.0
        ref_state, ours = {}, None
        for sigma in torch.linspace(1.0, 0.05, 12, dtype=torch.float64).tolist():
            x = torch.randn(shape, dtype=torch.float64)
            x0_c = torch.randn(shape, dtype=torch.float64) * 0.5
            x0_u = x0_c - 0.05 * torch.randn(shape, dtype=torch.float64)
            expected = _sorryhyun_v_space(ref_state, x, x0_c, x0_u, sigma, w, alpha, lam)
            corrected, ours = apply_smc_adaptive(x0_c - x0_u, sigma, ours, alpha, lam)
            got = x0_u + w * corrected.to(torch.float64)
            # float32 inside the controller (like the other SMC path)
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

    def test_compose_cfg_routes_the_adaptive_controller(self):
        torch.manual_seed(1)
        cond, uncond = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        unit, _ = compose_cfg(cond, uncond, torch.tensor([0.5]), 4.0, "smc", 0.0, 0.0, 6.0, 0.1, None)
        adaptive, state = compose_cfg(
            cond, uncond, torch.tensor([0.5]), 4.0, "smc", 0.0, 0.0, 5.0, 0.1, None,
            smc_mode=SMC_MODE_ADAPTIVE, smc_sigma=0.5, smc_alpha=0.2,
        )
        manual, _ = apply_smc_adaptive(cond - uncond, 0.5, None, 0.2, 5.0)
        torch.testing.assert_close(adaptive, uncond + 4.0 * manual)
        self.assertFalse(torch.allclose(unit, adaptive))
        self.assertIsInstance(state, dict)
        self.assertEqual(state["sigma"], 0.5)

    def test_first_step_and_alpha_zero(self):
        error = torch.randn(1, 4, 6, 6)
        # first call: e_prev = e → s = λ·e → the correction opposes e element-wise
        corrected, state = apply_smc_adaptive(error, 0.7, None, 0.2, 5.0)
        expected = error - 0.2 * error.abs().mean() * torch.sign(error)
        torch.testing.assert_close(corrected, expected)
        torch.testing.assert_close(state["e"], error)
        same, _ = apply_smc_adaptive(error, 0.7, state, 0.0, 5.0)
        self.assertIs(same, error)

    def test_stored_error_is_rescaled_by_the_sigma_ratio(self):
        e_prev = torch.ones(1, 1, 2, 2)
        e_now = torch.full((1, 1, 2, 2), 0.4)
        state = {"e": e_prev, "sigma": 0.8}
        # x0-space e_prev' = e_prev·σ_t/σ_prev = 0.5 → s = (0.4 − 0.5) + 5·0.5 = 2.4 > 0
        corrected, new_state = apply_smc_adaptive(e_now, 0.4, state, 0.5, 5.0)
        torch.testing.assert_close(corrected, e_now - 0.5 * 0.4)
        self.assertEqual(new_state["sigma"], 0.4)
        torch.testing.assert_close(new_state["e"], e_now, msg="the uncorrected error is stored")

    def test_unknown_sigma_or_shape_restarts(self):
        error = torch.randn(1, 2, 4, 4)
        stale = {"e": torch.randn(1, 2, 2, 2), "sigma": 0.5}
        fresh, _ = apply_smc_adaptive(error, 0.4, None, 0.2, 5.0)
        for previous, sigma in ((stale, 0.4), ({"e": error * 3, "sigma": 0.5}, None)):
            with self.subTest(sigma=sigma):
                got, _ = apply_smc_adaptive(error, sigma, previous, 0.2, 5.0)
                torch.testing.assert_close(got, fresh)

    def test_mode_names_and_unit_state_guard(self):
        self.assertEqual(normalize_smc_mode("Adaptive sign"), SMC_MODE_ADAPTIVE)
        self.assertEqual(normalize_smc_mode("adaptive"), SMC_MODE_ADAPTIVE)
        self.assertEqual(normalize_smc_mode(None), SMC_MODE_UNIT)
        self.assertEqual(normalize_smc_mode("whatever"), SMC_MODE_UNIT)
        # a dict state left by the adaptive controller does not break the unit-L2 one
        error = torch.randn(1, 2, 4, 4)
        a, _ = apply_smc_error(error, {"e": error, "sigma": 0.5}, 6.0, 0.1)
        b, _ = apply_smc_error(error, None, 6.0, 0.1)
        torch.testing.assert_close(a, b)


class SigmaHelperTests(unittest.TestCase):
    def test_half_log_snr_and_model_kind(self):
        self.assertAlmostEqual(half_log_snr(0.25, True), math.log(3.0))
        self.assertAlmostEqual(half_log_snr(0.5, True), 0.0)
        self.assertAlmostEqual(half_log_snr(2.0, False), -math.log(2.0))
        flow = type("M", (), {"predictor": type("P", (), {"prediction_type": "const"})()})()
        eps = type("M", (), {"predictor": type("P", (), {"prediction_type": "epsilon"})()})()
        self.assertIs(is_flow_model(flow), True)
        self.assertIs(is_flow_model(eps), False)
        self.assertIsNone(is_flow_model(object()))

    def test_schedule_index_uses_daves_isclose_rule(self):
        schedule = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        self.assertEqual(schedule_index(schedule, 0.5), 2)
        self.assertEqual(schedule_index(schedule, 0.5 + 3e-5), 2)
        self.assertIsNone(schedule_index(schedule, 0.6), "midpoint evaluations are off the schedule")
        self.assertIsNone(schedule_index(None, 0.5))
        self.assertEqual(schedule_index([1.0, 0.5], 1.0), 0)

    def test_sampler_sigma_prefers_the_pre_detail_daemon_note(self):
        from sam3ext.guidance import dave_gate

        args = {"sigma": torch.tensor([0.45])}
        try:
            dave_gate.note_pre_dd_sigma(torch.tensor([0.5]), torch.tensor([0.45]))
            self.assertAlmostEqual(sampler_sigma(args), 0.5)
            dave_gate.note_pre_dd_sigma(torch.tensor([0.5]), torch.tensor([0.3]))
            self.assertAlmostEqual(sampler_sigma(args), 0.45, places=6)
        finally:
            dave_gate.note_pre_dd_sigma(None, None)
        self.assertAlmostEqual(sampler_sigma(args), 0.45, places=6)


class TrajectoryTests(unittest.TestCase):
    def test_interpolates_linearly_in_sigma_and_clamps(self):
        traj = Trajectory(storage_dtype=torch.float32)
        traj.record(1.0, torch.full((1, 2, 2), 10.0))
        traj.record(0.5, torch.full((1, 2, 2), 4.0))
        traj.record(0.0, torch.full((1, 2, 2), 0.0))
        self.assertEqual(traj.sigmas, (0.0, 0.5, 1.0))
        torch.testing.assert_close(traj.at(0.75), torch.full((1, 2, 2), 7.0))
        torch.testing.assert_close(traj.at(0.25), torch.full((1, 2, 2), 2.0))
        torch.testing.assert_close(traj.at(1.5), torch.full((1, 2, 2), 10.0))
        torch.testing.assert_close(traj.at(-1.0), torch.full((1, 2, 2), 0.0))

    def test_rerecord_replaces_and_new_shape_restarts(self):
        traj = Trajectory()
        traj.record(0.5, torch.zeros(1, 2))
        traj.record(0.5, torch.ones(1, 2))
        self.assertEqual(len(traj), 1)
        torch.testing.assert_close(traj.at(0.5), torch.ones(1, 2))
        self.assertEqual(traj.at(0.5).dtype, torch.float32)
        traj.record(0.25, torch.ones(1, 3))
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj.shape, (1, 3))
        traj.clear()
        self.assertIsNone(traj.at(0.5))


def _tsr_reference(denoised, x, sigmas, k, tsr_sigma, flow):
    """ComfyUI nodes_eps.py TemporalScoreRescaling, one row at a time (its scalar-sigma form)."""
    rows = []
    for row, sigma in enumerate(sigmas):
        sigma = torch.tensor(float(sigma), dtype=torch.float64)
        d, xx = denoised[row].double(), x[row].double()
        if k == 1 or sigma == 0:
            rows.append(d)
            continue
        hls = torch.log((1 - sigma) / sigma) if flow else -torch.log(sigma)   # sigma_to_half_log_snr
        snr = (2 * hls).exp()
        if snr == 0:
            rows.append(d)
            continue
        variance = tsr_sigma ** 2
        r = (snr * variance + 1) / (snr * variance / k + 1)
        alpha = sigma * hls.exp()
        rows.append(torch.lerp(xx / alpha, d, r))
    return torch.stack(rows)


class TsrTests(unittest.TestCase):
    def test_matches_the_comfyui_formula_per_row(self):
        torch.manual_seed(0)
        for flow, sigmas in ((True, [0.95, 0.5, 0.1]), (False, [14.6, 1.0, 0.05])):
            with self.subTest(flow=flow):
                denoised = torch.randn(3, 4, 1, 6, 5)
                x = torch.randn(3, 4, 1, 6, 5)
                got = apply_tsr(denoised, x, torch.tensor(sigmas), k=0.93, tsr_sigma=3.0, flow=flow)
                expected = _tsr_reference(denoised, x, sigmas, 0.93, 3.0, flow).float()
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

    def test_rescale_factor_values(self):
        # flow σ → r for (k, tsr_sigma) — the paper's r_t = (ησ²+1)/(ησ²/k+1), η = ((1-σ)/σ)²
        for sigma, k, ts, r in ((0.5, 0.95, 1.0, 0.97436), (0.25, 0.95, 1.0, 0.95477),
                                (0.75, 0.93, 3.0, 0.96373), (0.1, 0.93, 3.0, 0.93009)):
            with self.subTest(sigma=sigma, k=k):
                got, alpha = rescale_factors(sigma, k, ts, True)
                self.assertAlmostEqual(got, r, places=5)
                self.assertAlmostEqual(alpha, 1.0 - sigma)
        self.assertEqual(rescale_factors(0.5, 1.0, 1.0, True), (1.0, 1.0))
        self.assertEqual(rescale_factors(1.0, 0.9, 1.0, True), (1.0, 1.0), "flow σ = 1: α = 0, no rescale")
        self.assertEqual(rescale_factors(0.0, 0.9, 1.0, False), (1.0, 1.0))

    def test_identity_cases(self):
        denoised, x = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
        self.assertIs(apply_tsr(denoised, x, torch.tensor([0.5, 0.5]), k=1.0, tsr_sigma=1.0, flow=True), denoised)
        self.assertIs(apply_tsr(denoised, x, torch.tensor([1.0, 0.0]), k=0.9, tsr_sigma=1.0, flow=True), denoised)
        mixed = apply_tsr(denoised, x, torch.tensor([0.0, 0.5]), k=0.9, tsr_sigma=1.0, flow=True)
        torch.testing.assert_close(mixed[0], denoised[0])
        self.assertFalse(torch.allclose(mixed[1], denoised[1]))

    def test_lower_k_keeps_more_of_the_noise_late(self):
        # x0' = x/α + r·(D − x/α): r < 1 moves x0 toward x/α (the noisy input) — more residual texture
        denoised, x = torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)
        out = apply_tsr(denoised, x, torch.tensor([0.2]), k=0.9, tsr_sigma=1.0, flow=True)
        r, alpha = rescale_factors(0.2, 0.9, 1.0, True)
        torch.testing.assert_close(out, torch.full((1, 1, 2, 2), (1 - r) / alpha))
        self.assertTrue(bool((out > 0).all()))


class MomentumGuidanceTests(unittest.TestCase):
    def _run(self, sigmas, alpha, beta, normalize=False, window=None):
        torch.manual_seed(5)
        state = history.HistoryState()
        x = torch.randn(1, 4, 1, 6, 6)
        outputs, velocities = [], []
        for sigma in sigmas:
            denoised = torch.randn(1, 4, 1, 6, 6)
            role = history.step_role(state, sigma, True)
            active = True if window is None else window[0] <= sigma <= window[1]
            out = history.apply_mg(denoised, x, sigma, state, alpha=alpha, beta=beta,
                                   normalize=normalize, active=active, role=role)
            outputs.append((denoised, out, x))
            velocities.append((denoised - x) / sigma)
            x = x + (sigmas[min(len(sigmas) - 1, len(outputs))] - sigma) * (x - out) / sigma
        return outputs, velocities, state

    def test_first_step_seeds_and_later_steps_follow_eq_13(self):
        sigmas = [1.0, 0.8, 0.6, 0.4, 0.2]
        alpha, beta = 0.5, 0.6
        outputs, velocities, _state = self._run(sigmas, alpha, beta)
        self.assertTrue(torch.equal(outputs[0][1], outputs[0][0]), "m₀ = v₀ → no effect on step one")
        m = velocities[0]
        for i in range(1, len(sigmas)):
            denoised, out, _x = outputs[i]
            v = velocities[i]
            torch.testing.assert_close(out, denoised + alpha * sigmas[i] * (v - m), rtol=1e-5, atol=1e-5)
            m = (1 - beta) * v + beta * m   # the EMA takes the unmodified velocity

    def test_forge_euler_step_reproduces_the_paper_update(self):
        # D̃ returned to Forge's Euler (x' = x + (σ' − σ)·(x − D̃)/σ) == Z + Δt·[v + α(v − m)], t = 1 − σ
        sigma, sigma_next, alpha = 0.7, 0.5, 0.8
        torch.manual_seed(2)
        x, denoised, m = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)
        state = history.HistoryState(mg_m=m.clone(), last_sigma=0.9)
        role = history.step_role(state, sigma, True)
        out = history.apply_mg(denoised, x, sigma, state, alpha=alpha, beta=0.6,
                               normalize=False, active=True, role=role)
        forge = x + (sigma_next - sigma) * (x - out) / sigma
        v = (denoised - x) / sigma
        paper = x + (sigma - sigma_next) * (v + alpha * (v - m))
        torch.testing.assert_close(forge, paper, rtol=1e-5, atol=1e-5)

    def test_alpha_zero_and_inactive_window_only_update_the_history(self):
        outputs, velocities, state = self._run([1.0, 0.5, 0.25], 0.0, 0.5)
        for denoised, out, _x in outputs:
            self.assertTrue(torch.equal(out, denoised))
        expected = (1 - 0.5) * velocities[2] + 0.5 * ((1 - 0.5) * velocities[1] + 0.5 * velocities[0])
        torch.testing.assert_close(state.mg_m, expected)
        outputs, _v, _s = self._run([1.0, 0.5, 0.25], 0.7, 0.5, window=(0.3, 0.9))
        self.assertTrue(torch.equal(outputs[2][1], outputs[2][0]), "σ 0.25 is outside 0.3-0.9")
        self.assertFalse(torch.equal(outputs[1][1], outputs[1][0]))

    def test_normalize_rescales_the_momentum_to_the_velocity_norm(self):
        x = torch.zeros(1, 1, 2, 2)
        state = history.HistoryState(mg_m=torch.full((1, 1, 2, 2), 10.0), last_sigma=1.0)
        denoised = torch.full((1, 1, 2, 2), 0.5)   # v = (D − x)/σ = 1
        role = history.step_role(state, 0.5, True)
        out = history.apply_mg(denoised, x, 0.5, state, alpha=1.0, beta=0.0,
                               normalize=True, active=True, role=role)
        torch.testing.assert_close(out, denoised, msg="‖m‖ rescaled to ‖v‖ → m = v → no push")

    def test_step_roles(self):
        state = history.HistoryState()
        self.assertEqual(history.step_role(state, 0.8, True), history.ROLE_NEW)
        self.assertEqual(history.step_role(state, 0.8, True), history.ROLE_REPEAT)
        self.assertEqual(history.step_role(state, 0.7, False), history.ROLE_SKIP, "midpoint")
        self.assertEqual(history.step_role(state, 0.7, None), history.ROLE_NEW, "no schedule published")
        state.mg_m = torch.ones(1)
        self.assertEqual(history.step_role(state, 0.95, True), history.ROLE_NEW)
        self.assertIsNone(state.mg_m, "a rising sigma starts a new run")
        self.assertEqual(history.step_role(state, 0.0, True), history.ROLE_SKIP)


def _dct_reference(t):
    """Orthonormal 2-D DCT-II by the explicit sum (independent of the module's matrices)."""
    h, w = t.shape[-2:]
    out = torch.zeros_like(t, dtype=torch.float64)
    src = t.double()
    for k in range(h):
        for l in range(w):
            ck = math.sqrt(1 / h) if k == 0 else math.sqrt(2 / h)
            cl = math.sqrt(1 / w) if l == 0 else math.sqrt(2 / w)
            total = 0.0
            for i in range(h):
                for j in range(w):
                    total = total + src[..., i, j] * math.cos(math.pi * (2 * i + 1) * k / (2 * h)) \
                        * math.cos(math.pi * (2 * j + 1) * l / (2 * w))
            out[..., k, l] = ck * cl * total
    return out


class HigsTests(unittest.TestCase):
    def test_dct_matrix_matches_the_explicit_sum(self):
        torch.manual_seed(0)
        t = torch.randn(2, 3, 4, 5)
        c_h = history._dct_matrix(4, t.device, torch.float64)
        c_w = history._dct_matrix(5, t.device, torch.float64)
        torch.testing.assert_close(c_h @ t.double() @ c_w.T, _dct_reference(t))
        torch.testing.assert_close(c_h @ c_h.T, torch.eye(4, dtype=torch.float64))

    def test_highpass_keeps_detail_and_drops_most_of_dc(self):
        flat = torch.full((1, 1, 8, 8), 2.0)
        out = history.dct_highpass(flat, cutoff=0.05)
        keep = 1 / (1 + math.exp(50 * 0.05))   # sigmoid(50·(0 − 0.05)) ≈ 0.0759
        torch.testing.assert_close(out, flat * keep, rtol=1e-5, atol=1e-6)
        checker = torch.tensor([[(-1.0) ** (i + j) for j in range(8)] for i in range(8)]).view(1, 1, 8, 8)
        torch.testing.assert_close(history.dct_highpass(checker, cutoff=0.05), checker, rtol=1e-4, atol=1e-4)
        noise = torch.randn(1, 2, 6, 7)
        torch.testing.assert_close(history.dct_highpass(noise, cutoff=-10.0), noise, rtol=1e-5, atol=1e-5)

    def test_weight_schedule(self):
        self.assertEqual(history.higs_weight(0.4, 1.75, 0.4, 1.0), 0.0)
        self.assertAlmostEqual(history.higs_weight(1.0, 1.75, 0.4, 1.0), 1.75)
        self.assertAlmostEqual(history.higs_weight(0.55, 2.0, 0.4, 1.0), 2.0 * math.sqrt(0.25))
        self.assertEqual(history.higs_weight(1.0, 1.75, 0.4, 0.95), 0.0, "above t_max")
        self.assertAlmostEqual(history.noise_level(0.6, True), 0.6)
        self.assertAlmostEqual(history.noise_level(3.0, False), 0.75)

    def _reference(self, denoised, g, weight, eta, cutoff, t):
        diff = (denoised - g).double()
        ref = denoised.double()
        dims = tuple(range(1, diff.ndim))
        par = (diff * ref).sum(dims, keepdim=True) / (ref * ref).sum(dims, keepdim=True) * ref
        diff = (diff - par + eta * par)
        h, w = diff.shape[-2:]
        c_h, c_w = history._dct_matrix(h, "cpu", torch.float64), history._dct_matrix(w, "cpu", torch.float64)
        u = torch.arange(h, dtype=torch.float64) / h
        v = torch.arange(w, dtype=torch.float64) / w
        mask = torch.sigmoid(50 * (torch.sqrt(u[:, None] ** 2 + v[None, :] ** 2) - cutoff))
        filtered = c_h.T @ ((c_h @ diff @ c_w.T) * mask) @ c_w
        return (ref + history.higs_weight(t, weight, 0.4, 1.0) * filtered).float()

    def test_seed_then_paper_update(self):
        torch.manual_seed(9)
        state = history.HistoryState()
        alpha, weight, eta, cutoff = 0.75, 1.75, 0.3, 0.05
        sigmas = [0.95, 0.8, 0.6, 0.35]
        g = None
        for i, sigma in enumerate(sigmas):
            denoised = torch.randn(2, 4, 1, 6, 6)
            role = history.step_role(state, sigma, True)
            out = history.apply_higs(denoised, sigma, state, weight=weight, eta=eta, alpha=alpha,
                                     cutoff=cutoff, t_min=0.4, t_max=1.0, flow=True, role=role)
            if i == 0:
                self.assertTrue(torch.equal(out, denoised), "the first evaluation only seeds")
                g = alpha * denoised
                continue
            expected = self._reference(denoised, g, weight, eta, cutoff, sigma)
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
            g = alpha * denoised + (1 - alpha) * g
            torch.testing.assert_close(state.higs_g, g)
        self.assertEqual(state.counters["higs"], 2, "σ 0.35 is below t_min 0.4 → weight 0")

    def test_eta_zero_removes_the_component_along_d(self):
        torch.manual_seed(4)
        denoised = torch.randn(1, 3, 1, 8, 8)
        g = torch.randn(1, 3, 1, 8, 8)
        state = history.HistoryState(higs_g=g.clone(), last_sigma=1.0)
        out = history.apply_higs(denoised, 0.9, state, weight=1.0, eta=0.0, alpha=0.5, cutoff=-10.0,
                                 t_min=0.0, t_max=1.0, flow=True, role=history.ROLE_REPEAT)
        push = (out - denoised).double()
        self.assertLess(abs(float((push * denoised.double()).sum())), 1e-3)
        torch.testing.assert_close(state.higs_g, g, msg="ROLE_REPEAT does not update")


def _origin_butterworth_lowpass(x, d_s):
    """Bujiazi/HiFlow utils.py (Apache-2.0): loop-built Butterworth mask (n=4) + shifted FFT."""
    h_, w_ = x.shape[-2:]
    mask = torch.zeros(h_, w_)
    for h in range(h_):
        for w in range(w_):
            d_square = ((2 * h / h_ - 1) ** 2 + (2 * w / w_ - 1) ** 2)
            mask[h, w] = 1 / (1 + (d_square / d_s ** 2) ** 4)
    x_freq = torch.fft.fftshift(torch.fft.fft2(x.float()))
    return torch.fft.ifft2(torch.fft.ifftshift(x_freq * mask)).real


class HiFlowTests(unittest.TestCase):
    def test_lowpass_matches_the_origin_filter(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 8, 12)
        torch.testing.assert_close(hiflow.butterworth_lowpass(x, 0.2), _origin_butterworth_lowpass(x, 0.2),
                                   rtol=1e-5, atol=1e-5)
        mask = hiflow.butterworth_mask(8, 12, 0.2)
        self.assertAlmostEqual(float(mask[4, 6]), 1.0, msg="centre (DC after fftshift) passes")
        torch.testing.assert_close(hiflow.butterworth_lowpass(torch.full((1, 1, 8, 8), 3.0), 0.2),
                                   torch.full((1, 1, 8, 8), 3.0))

    def test_resize_latent_bicubic_4d_and_5d(self):
        torch.manual_seed(1)
        lr = torch.randn(1, 16, 1, 6, 8)
        hr = hiflow.resize_latent(lr, (12, 16))
        self.assertEqual(tuple(hr.shape), (1, 16, 1, 12, 16))
        expected = torch.nn.functional.interpolate(lr[:, :, 0], size=(12, 16), mode="bicubic",
                                                   align_corners=False).unsqueeze(2)
        torch.testing.assert_close(hr, expected)
        torch.testing.assert_close(hiflow.resize_latent(hr, (12, 16)), hr, msg="same size passes through")

    def test_step_weight(self):
        self.assertEqual(hiflow.step_weight(0, 16), 1.0)
        self.assertAlmostEqual(hiflow.step_weight(4, 16), 0.75)
        self.assertAlmostEqual(hiflow.step_weight(15, 16), 1 / 16)
        self.assertEqual(hiflow.step_weight(None, 16), 1.0)

    def test_x0_replacement_reproduces_the_origin_euler_loop(self):
        """Origin pipeline (velocity space, Euler) vs this module's x0 replacement + Forge Euler."""
        torch.manual_seed(3)
        shape = (1, 4, 8, 8)
        sigmas = [0.8, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0]
        steps = len(sigmas) - 1
        a_base, b_base, cutoff = 1.0, 0.5, 0.2
        model_x0 = [torch.randn(shape) for _ in range(steps)]     # the model's raw x0 at each step
        refs = [torch.randn(shape) for _ in range(steps)]         # upsampled low-res x0 at each step
        x_origin = x_ours = torch.randn(shape)
        state = hiflow.HiFlowState()
        prev_high = prev_ref_v = None
        for i in range(steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            alpha = a_base * (steps - i) / steps
            beta = b_base * (steps - i) / steps
            # --- origin: flux_pipeline_hiflow.py (direction on x0, acceleration on velocity)
            pred_x0 = model_x0[i] + alpha * (_origin_butterworth_lowpass(refs[i], cutoff)
                                             - _origin_butterworth_lowpass(model_x0[i], cutoff))
            v = (x_origin - pred_x0) / sigma
            v_ref = (x_origin - refs[i]) / sigma
            if prev_high is not None:
                v = v + beta * (prev_high + v_ref - prev_ref_v - v)
            prev_high, prev_ref_v = v, v_ref
            x_origin = x_origin + (sigma_next - sigma) * v
            # --- ours: return O, Forge's Euler forms d = (x − O)/σ
            out = hiflow.apply_hiflow(model_x0[i], sigma, refs[i], state, alpha=a_base, beta=b_base,
                                      cutoff=cutoff, weight=hiflow.step_weight(i, steps), step_start=True)
            x_ours = x_ours + (sigma_next - sigma) * (x_ours - out) / sigma
            torch.testing.assert_close(x_ours, x_origin, rtol=1e-4, atol=1e-4)
        self.assertEqual(state.counters["acceleration"], steps - 1)

    def test_midpoint_gets_direction_only_and_keeps_the_state(self):
        torch.manual_seed(4)
        state = hiflow.HiFlowState()
        d0, r0 = torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)
        o0 = hiflow.apply_hiflow(d0, 0.8, r0, state, alpha=1.0, beta=0.5, cutoff=0.2, weight=1.0,
                                 step_start=True)
        snapshot = (state.prev_sigma, state.prev_out.clone())
        mid = hiflow.apply_hiflow(torch.randn(1, 2, 8, 8), 0.75, r0, state, alpha=1.0, beta=0.5,
                                  cutoff=0.2, weight=1.0, step_start=False)
        self.assertEqual(state.prev_sigma, snapshot[0])
        torch.testing.assert_close(state.prev_out, snapshot[1])
        self.assertEqual(mid.shape, o0.shape)
        same = hiflow.apply_hiflow(d0, 0.8, r0, hiflow.HiFlowState(), alpha=0.0, beta=0.0, cutoff=0.2,
                                   weight=1.0, step_start=True)
        torch.testing.assert_close(same, d0, msg="α = β = 0 is the identity")


if __name__ == "__main__":
    unittest.main()
