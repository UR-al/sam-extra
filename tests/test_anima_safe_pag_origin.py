"""Origin-parity tests for the Anima Safe PAG window, scale range and index parsing.

The oracle is the upstream ComfyUI node itself: its sigma-window and index
helpers are copied below verbatim (MIT, iljung1106/comfyui-anima-safe-pag),
together with the ComfyUI and Forge code that decides which sigmas a sampler
visits. Each copy carries an ``origin:`` comment with the repository, commit,
file and lines it was taken from.

Upstream converts ``start_percent``/``end_percent`` with the model's
``percent_to_sigma`` once and tests every model call's sigma against that
window, both ends inclusive. The extension must place PAG on exactly the same
calls: on Anima (flow, shift 3) the default window 0.0-0.7 is sigma 1.0-0.5625,
which covers 15/20/22 steps of a 20/28/30-step simple schedule and 9 of 20 steps
at img2img denoise 0.5 — not the step fraction's 14/19/21 and 14/20.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.guidance import sigma_window  # noqa: E402


def _load_base_tests():
    """The PAG test module's loaders (script without WebUI, real Forge Anima on CPU).

    Loaded under a private name so its TestCase classes are not collected twice.
    """
    spec = importlib.util.spec_from_file_location(
        "_pag_origin_base_tests", ROOT / "tests" / "test_anima_safe_pag.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_tests()


# ---------------------------------------------------------------------------
# Upstream oracle — MIT License, Copyright (c) 2026 (comfyui-anima-safe-pag;
# full notice in THIRD_PARTY_NOTICES.md). Verbatim.
# origin: iljung1106/comfyui-anima-safe-pag@905b0107d1f924fc6acbcac3b6a879b566ff671c:__init__.py:9-64
# ---------------------------------------------------------------------------


def _sigma_to_float(sigma):
    if torch.is_tensor(sigma):
        return float(sigma.flatten()[0].item())
    return float(sigma)


def _sigma_active(sigma, sigma_start, sigma_end):
    sigma_start = _sigma_to_float(sigma_start)
    sigma_end = _sigma_to_float(sigma_end)
    if sigma_start < sigma_end:
        sigma_start, sigma_end = sigma_end, sigma_start

    value = _sigma_to_float(sigma)
    return sigma_end <= value <= sigma_start


def _percent_range_to_sigmas(model, start_percent, end_percent):
    start_percent = max(0.0, min(1.0, float(start_percent)))
    end_percent = max(0.0, min(1.0, float(end_percent)))
    if start_percent > end_percent:
        start_percent, end_percent = end_percent, start_percent

    model_sampling = model.get_model_object("model_sampling")
    sigma_start = model_sampling.percent_to_sigma(start_percent)
    sigma_end = model_sampling.percent_to_sigma(end_percent)
    return sigma_start, sigma_end, start_percent, end_percent


def _parse_indices(text, max_index):
    values = set()
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            pieces = [p.strip() for p in part.split("-", 1)]
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise RuntimeError(f"Invalid block range '{part}'. Use values like 18 or 18,20,22.")
            start, end = int(pieces[0]), int(pieces[1])
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(part))

    indices = sorted(i for i in values if 0 <= i <= max_index)
    if not indices:
        raise RuntimeError(f"No valid block indices found. Valid range is 0 to {max_index}.")
    return indices


def _parse_optional_indices(text, max_index):
    if text is None or not str(text).strip():
        return None
    return _parse_indices(text, max_index)


# origin: iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:201 (scale input)
_ORIGIN_SCALE = {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}


# ---------------------------------------------------------------------------
# Host code: Anima's flow sampling and the schedules the sampler visits.
# ---------------------------------------------------------------------------


def _time_snr_shift(alpha, t):
    # origin: Comfy-Org/ComfyUI@387f98aa2822f684b8597959a52a467d88cc4806:comfy/model_sampling.py:289-292
    # (same body in Haoming02/sd-webui-forge-classic@e33f40e4:backend/modules/k_prediction.py:32-35)
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class _FlowPredictor:
    """Anima flow sampling, shift 3 — ComfyUI ``ModelSamplingDiscreteFlow`` /
    Forge ``PredictionDiscreteFlow`` (the object at ``KModel.predictor``)."""

    def __init__(self, shift=3.0, multiplier=1.0, timesteps=1000):
        self.shift = shift
        self.multiplier = multiplier
        # origin: Comfy-Org/ComfyUI@387f98aa:comfy/model_sampling.py:307-311 (set_parameters)
        # (= Haoming02/sd-webui-forge-classic@e33f40e4:backend/modules/k_prediction.py:190-194)
        self.sigmas = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)

    def sigma(self, timestep):
        # origin: Comfy-Org/ComfyUI@387f98aa:comfy/model_sampling.py:327-328
        return _time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        # origin: Comfy-Org/ComfyUI@387f98aa:comfy/model_sampling.py:331-336
        # (= Haoming02/sd-webui-forge-classic@e33f40e4:backend/modules/k_prediction.py:212-217)
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return _time_snr_shift(self.shift, 1.0 - percent)


class _ComfyModel:
    """What upstream's ``_percent_range_to_sigmas`` asks for (ModelPatcher)."""

    def __init__(self, model_sampling):
        self._model_sampling = model_sampling

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self._model_sampling


def _simple_sigmas(model_sampling, steps):
    # origin: Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:645-652 (simple_scheduler)
    # (= Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_schedulers.py:49-55)
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def _comfy_denoise_sigmas(model_sampling, steps, denoise):
    # origin: Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:1431-1441 (KSampler.set_steps)
    # Forge img2img with "img2img_fix_steps" gives the same slice
    # (Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_samplers_common.py:44-47,
    #  modules/sd_samplers_kdiffusion.py:145-148).
    new_steps = int(steps / denoise)
    sigmas = _simple_sigmas(model_sampling, new_steps)
    return sigmas[-(steps + 1):]


def _forge_img2img_sigmas(model_sampling, steps, denoise):
    # origin: Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_samplers_common.py:48-50
    # (default, img2img_fix_steps off) + modules/sd_samplers_kdiffusion.py:145-148
    t_enc = int(min(denoise, 0.999) * steps)
    sigmas = _simple_sigmas(model_sampling, steps)
    return sigmas[steps - t_enc - 1:]


def _karras_sigmas(model_sampling, steps, rho=7.0):
    # origin: Haoming02/sd-webui-forge-classic@e33f40e4:modules_forge/packages/k_diffusion/sampling.py:24-30
    # (get_sigmas_karras), sigma_min/max = the model's sigmas[0]/[-1]
    # (modules/sd_samplers_kdiffusion.py:94, 100).
    sigma_min, sigma_max = float(model_sampling.sigmas[0]), float(model_sampling.sigmas[-1])
    ramp = torch.linspace(0, 1, steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def _unet_with(predictor):
    return types.SimpleNamespace(model=types.SimpleNamespace(predictor=predictor))


# ---------------------------------------------------------------------------
# 1) sam3ext.guidance.sigma_window == upstream helpers
# ---------------------------------------------------------------------------


class SigmaWindowOriginTests(unittest.TestCase):
    PERCENTS = (-0.5, 0.0, 0.001, 0.2, 0.3, 0.5, 0.7, 0.8, 0.999, 1.0, 1.5)

    def test_percent_range_to_sigmas_matches_upstream(self):
        flow = _FlowPredictor()
        for start in self.PERCENTS:
            for end in self.PERCENTS:
                with self.subTest(start=start, end=end):
                    self.assertEqual(
                        sigma_window.percent_range_to_sigmas(flow.percent_to_sigma, start, end),
                        _percent_range_to_sigmas(_ComfyModel(flow), start, end),
                    )

    def test_sigma_active_matches_upstream_including_bounds_and_reversal(self):
        flow = _FlowPredictor()
        table = [float(s) for s in flow.sigmas[::37]] + [0.0, 0.5625, 0.75, 1.0, 1.25]
        bounds = [
            (1.0, 0.5625), (0.5625, 1.0), (0.75, 0.75), (0.9, 0.2), (0.2, 0.9),
            (999999999.9, 0.0), (torch.tensor(0.8), torch.tensor([0.3])),
        ]
        for hi, lo in bounds:
            for value in table:
                for sigma in (value, torch.full((2,), value), torch.tensor([value, 5.0])):
                    with self.subTest(hi=hi, lo=lo, sigma=sigma):
                        self.assertIs(
                            sigma_window.sigma_active(sigma, hi, lo),
                            _sigma_active(sigma, hi, lo),
                        )

    def test_anima_default_window(self):
        """end 0.7 on Anima (shift 3) is sigma 0.5625; start 0 is sigma 1."""
        flow = _FlowPredictor()
        hi, lo, start, end = sigma_window.percent_range_to_sigmas(flow.percent_to_sigma, 0.0, 0.7)
        self.assertEqual((hi, start, end), (1.0, 0.0, 0.7))
        self.assertAlmostEqual(lo, 0.5625, places=12)
        # Reversed percents give the same window (upstream :28-29).
        self.assertEqual(
            sigma_window.percent_range_to_sigmas(flow.percent_to_sigma, 0.7, 0.0),
            (hi, lo, start, end),
        )

    def test_module_stays_pure(self):
        source = (ROOT / "sam3ext" / "guidance" / "sigma_window.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertIn("comfyui-anima-safe-pag@905b0107", source)


# ---------------------------------------------------------------------------
# 2) Gate tables — Anima flow shift 3
# ---------------------------------------------------------------------------


class SigmaGateTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()
        cls.flow = _FlowPredictor()

    def setUp(self):
        p = self.pag
        saved = {key: p._STATE.get(key) for key in ("sigma_hi", "sigma_lo", "start", "end", "step", "total")}
        self.addCleanup(p._STATE.update, saved)

    def _attach(self, start, end):
        p = self.pag
        sigma_hi, sigma_lo = p._pag_sigma_window(_unet_with(self.flow), start, end)
        p._STATE.update(sigma_hi=sigma_hi, sigma_lo=sigma_lo,
                        start=min(start, end), end=max(start, end))

    def _extension_active(self, sigmas):
        return [bool(self.pag._pag_in_range(torch.full((2,), float(s)))) for s in sigmas]

    def _origin_active(self, sigmas, start, end):
        sigma_start, sigma_end, _s, _e = _percent_range_to_sigmas(_ComfyModel(self.flow), start, end)
        return [_sigma_active(torch.full((2,), float(s)), sigma_start, sigma_end) for s in sigmas]

    def test_txt2img_simple_20_28_30_steps(self):
        for steps, expected in ((20, 15), (28, 20), (30, 22)):
            sigmas = _simple_sigmas(self.flow, steps)[:-1]
            with self.subTest(steps=steps):
                self._attach(0.0, 0.7)
                origin = self._origin_active(sigmas, 0.0, 0.7)
                self.assertEqual(self._extension_active(sigmas), origin)
                self.assertEqual(sum(origin), expected)
                # A contiguous prefix: the window closes once, at sigma 0.5625.
                self.assertEqual(origin, [True] * expected + [False] * (steps - expected))

    def test_img2img_denoise_follows_sigma_not_step_count(self):
        for denoise, expected in ((0.5, 9), (0.35, 3)):
            sigmas = _comfy_denoise_sigmas(self.flow, 20, denoise)[:-1]
            with self.subTest(denoise=denoise):
                self.assertEqual(len(sigmas), 20)
                self._attach(0.0, 0.7)
                origin = self._origin_active(sigmas, 0.0, 0.7)
                self.assertEqual(self._extension_active(sigmas), origin)
                self.assertEqual(sum(origin), expected)
        # Forge's default img2img slice (fewer steps) is placed by sigma too.
        sigmas = _forge_img2img_sigmas(self.flow, 20, 0.5)[:-1]
        self._attach(0.0, 0.7)
        self.assertEqual(self._extension_active(sigmas), self._origin_active(sigmas, 0.0, 0.7))

    def test_windows_and_reversed_percents(self):
        sigmas = _simple_sigmas(self.flow, 30)[:-1]
        for start, end in ((0.2, 0.8), (0.8, 0.2), (0.0, 1.0), (0.5, 0.5), (-1.0, 2.0), (0.3, 0.0)):
            with self.subTest(start=start, end=end):
                self._attach(start, end)
                self.assertEqual(self._extension_active(sigmas),
                                 self._origin_active(sigmas, start, end))

    def test_second_order_midpoint_sigmas_are_gated_by_value(self):
        """Extra model calls between steps (Heun/DPM2) use their own sigma, like upstream."""
        steps = _simple_sigmas(self.flow, 20)
        mids = [float((a * b).sqrt()) for a, b in zip(steps[:-2], steps[1:-1])]
        self._attach(0.0, 0.7)
        self.assertEqual(self._extension_active(mids), self._origin_active(mids, 0.0, 0.7))

    def test_without_predictor_the_step_fraction_is_the_fallback(self):
        p = self.pag
        self.assertEqual(p._pag_sigma_window(_unet_with(None), 0.0, 0.7), (None, None))
        self.assertEqual(p._pag_sigma_window(types.SimpleNamespace(), 0.0, 0.7), (None, None))
        p._STATE.update(sigma_hi=None, sigma_lo=None, start=0.0, end=0.7)
        old_shared = p.shared
        try:
            active = []
            for step in range(20):
                p.shared = types.SimpleNamespace(
                    state=types.SimpleNamespace(sampling_step=step, sampling_steps=20))
                active.append(p._pag_in_range(torch.ones(2)))
        finally:
            p.shared = old_shared
        self.assertEqual(sum(active), 14)


# ---------------------------------------------------------------------------
# 3) The hooks use the gate with the sigma of the call
# ---------------------------------------------------------------------------


class SigmaGateHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    def setUp(self):
        # Same neutral state as the main PAG tests.
        _BASE.AnimaSafePagTests.setUp(self)
        p = self.pag
        sigma_hi, sigma_lo = p._pag_sigma_window(_unet_with(_FlowPredictor()), 0.0, 0.7)
        p._STATE.update(sigma_hi=sigma_hi, sigma_lo=sigma_lo, start=0.0, end=0.7,
                        guard_params=None)
        self.addCleanup(p._STATE.update, {"sigma_hi": None, "sigma_lo": None, "guard_params": None})

    def _call(self, sigma, c=None):
        p = self.pag
        seen = []

        def apply_model(x, timestep, **conditioning):
            seen.append(int(x.shape[0]))
            return conditioning["bias"].clone()

        p._STATE["step_open"] = False
        conditioning = {"bias": torch.tensor([[0.0], [1.0]])}
        conditioning.update(c or {})
        p._model_wrapper(apply_model, {
            "input": torch.zeros(2, 1),
            "timestep": torch.full((2,), float(sigma)),
            "c": conditioning,
            "cond_or_uncond": [1, 0],
        })
        return seen[0]

    def test_wrapper_appends_weak_rows_exactly_on_upstream_steps(self):
        sigmas = _simple_sigmas(_FlowPredictor(), 20)[:-1]
        batches = [self._call(s) for s in sigmas]
        self.assertEqual(batches, [3] * 15 + [2] * 5)
        self.assertEqual(self.pag._STATE["weak_steps"], 15)

    def test_condition_aggregation_is_gated_by_the_same_sigma(self):
        p = self.pag
        prepared = types.SimpleNamespace(area=(1, 1, 0, 0), mult=1.0)
        backend = types.ModuleType("backend")
        backend.__path__ = []
        sampling = types.ModuleType("backend.sampling")
        sampling.__path__ = []
        function = types.ModuleType("backend.sampling.sampling_function")
        function.get_area_and_mult = lambda item, x, timestep: prepared
        cond = [{"model_conds": {}, "area": (1, 1, 0, 0)}, {"model_conds": {}}]
        with mock.patch.dict(sys.modules, {
            "backend": backend, "backend.sampling": sampling,
            "backend.sampling.sampling_function": function,
        }):
            for sigma, attached in ((0.8, True), (0.5625, True), (0.5, False)):
                with self.subTest(sigma=sigma):
                    out = p._prepare_condition_aggregation(
                        None, cond, None, torch.zeros(1, 1), torch.full((1,), sigma), {},
                    )
                    self.assertEqual("condition_aggregation" in p._STATE, attached)
                    self.assertEqual(out[1] is cond, not attached)
        p._STATE.pop("condition_aggregation", None)

    def test_controlnet_guard_is_kept_and_written_to_the_infotext(self):
        p = self.pag
        params = {"Steps": 20}
        p._STATE["guard_params"] = params
        control = {"control": {"output": []}}
        self.assertEqual(self._call(0.5, control), 2)  # outside the window: no guard needed
        self.assertNotIn(p.INFOTEXT_CONTROLNET_GUARD, params)
        self.assertEqual(p._STATE["control_blocked_calls"], 0)
        self.assertEqual(self._call(0.9, control), 2)  # inside: guard skips the weak rows
        self.assertEqual(p._STATE["control_blocked_calls"], 1)
        self.assertEqual(params[p.INFOTEXT_CONTROLNET_GUARD], p.INFOTEXT_CONTROLNET_GUARD_VALUE)
        self.assertIn(p.INFOTEXT_CONTROLNET_GUARD, p._EXTRA_GENERATION_PARAM_KEYS)
        self.assertEqual(self._call(0.9), 3)  # no ControlNet: PAG runs


# ---------------------------------------------------------------------------
# 4) Index parsing == upstream (reversed ranges swap)
# ---------------------------------------------------------------------------


class IndexParsingOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    VALID = ("18", "18,20,22", "14-27", "20-18", "27-14", " 18 , 20-18 ", "0-99",
             "18,99", "5-5", "3,1-2,9-7")

    def test_blocks_match_upstream(self):
        for spec in self.VALID:
            with self.subTest(spec=spec):
                self.assertEqual(self.pag._parse_blocks(spec, 28), set(_parse_indices(spec, 27)))
        self.assertEqual(self.pag._parse_blocks("20-18", 28), {18, 19, 20})

    def test_heads_match_upstream(self):
        for spec in self.VALID + ("7-4",):
            with self.subTest(spec=spec):
                self.assertEqual(self.pag._parse_attention_heads(spec, 32),
                                 set(_parse_optional_indices(spec, 31)))
        self.assertEqual(self.pag._parse_attention_heads("7-4", 16), {4, 5, 6, 7})
        # Empty = every head (upstream returns None and patches all heads).
        self.assertIsNone(_parse_optional_indices("", 15))
        self.assertEqual(self.pag._parse_attention_heads("", 16), set(range(16)))

    def test_invalid_text_does_not_fail_the_generation(self):
        """Host policy (kept): upstream raises, the extension falls back."""
        for spec in ("abc", "99", "-3"):
            with self.subTest(spec=spec):
                with self.assertRaises((RuntimeError, ValueError)):
                    _parse_indices(spec, 27)
                self.assertEqual(self.pag._parse_blocks(spec, 28), set())
        self.assertEqual(self.pag._parse_blocks("", 28), {18})


# ---------------------------------------------------------------------------
# 5) Attach: window from predictor, scale range 0..100
# ---------------------------------------------------------------------------


class AttachOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    def _process(self, predictor, params=None, is_hr_pass=False, **overrides):
        """Run process_before_every_sampling on a CPU Anima; return (infotext, state).

        ``params`` is the pass's starting ``p.extra_generation_params`` (what an
        earlier pass left), ``is_hr_pass`` Forge's hires flag.
        """
        p = self.pag
        anima = _BASE._load_real_forge_anima([])
        torch.manual_seed(0)
        dit = anima.Anima(
            in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
            model_channels=48, crossattn_emb_channels=32, adaln_lora_dim=16,
            num_blocks=6, num_heads=4,
        )

        class DummyUnet:
            def __init__(self):
                self.model = types.SimpleNamespace(diffusion_model=dit, predictor=predictor)
                self.model_options = {}

            def clone(self):
                return DummyUnet()

            def set_model_unet_function_wrapper(self, fn):
                self.model_options["model_function_wrapper"] = fn

            def set_model_sampler_pre_cfg_function(self, fn):
                pass

            def set_model_sampler_post_cfg_function(self, fn):
                pass

        model = type("Anima", (), {})()
        model.forge_objects = types.SimpleNamespace(unet=DummyUnet())
        request = types.SimpleNamespace(
            sd_model=model, extra_generation_params=dict(params or {}), steps=20,
            is_hr_pass=is_hr_pass,
        )
        xyz = overrides.pop("xyz", None)
        if xyz is not None:
            request._anima_safe_pag_xyz = xyz
        process = p.AnimaSafePAG()
        with gr.Blocks():
            inputs = process.ui(False)
        args = [component.value for component in inputs]
        args[0] = True
        args[4] = "3"
        for index, value in overrides.items():
            args[int(index.lstrip("a"))] = value
        try:
            with mock.patch.object(p, "shared", types.SimpleNamespace(opts=types.SimpleNamespace())):
                process.process_before_every_sampling(request, *args)
            return dict(request.extra_generation_params), dict(p._STATE), request
        finally:
            process.postprocess(request, None)
            p._teardown_global_patches()

    def test_window_is_stored_from_predictor_at_attach(self):
        p = self.pag
        params, state, request = self._process(_FlowPredictor())
        self.assertEqual(state["sigma_hi"], 1.0)
        self.assertAlmostEqual(state["sigma_lo"], 0.5625, places=12)
        self.assertIn("pag_sigma_window=1.0000-0.5625", params["Anima Perturbation Guidance"])
        self.assertIs(state["guard_params"], request.extra_generation_params)
        # postprocess drops the per-pass window and infotext reference.
        self.assertIsNone(p._STATE["sigma_hi"])
        self.assertIsNone(p._STATE["guard_params"])

    def test_reversed_percents_give_the_same_window(self):
        _params, state, _request = self._process(_FlowPredictor(), a8=0.7, a9=0.0)
        self.assertEqual(state["sigma_hi"], 1.0)
        self.assertAlmostEqual(state["sigma_lo"], 0.5625, places=12)

    def test_without_predictor_the_infotext_says_step_fraction(self):
        params, state, _request = self._process(None)
        self.assertIsNone(state["sigma_hi"])
        self.assertIn("pag_sigma_window=step-fraction", params["Anima Perturbation Guidance"])

    def test_scale_range_is_upstream_0_to_100(self):
        with gr.Blocks():
            inputs = self.pag.AnimaSafePAG().ui(False)
        self.assertEqual(inputs[2].elem_id, "anima_safe_pag_scale")
        self.assertEqual(
            (inputs[2].value, inputs[2].minimum, inputs[2].maximum, inputs[2].step),
            (_ORIGIN_SCALE["default"], _ORIGIN_SCALE["min"], _ORIGIN_SCALE["max"], _ORIGIN_SCALE["step"]),
        )
        for value, expected in ((50.0, 50.0), (100.0, 100.0), (150.0, 100.0)):
            with self.subTest(api_scale=value):
                _params, state, _request = self._process(_FlowPredictor(), a2=value)
                self.assertEqual(state["attn_scale"], expected)
        _params, state, _request = self._process(_FlowPredictor(), xyz={"scale": 40.0})
        self.assertEqual(state["attn_scale"], 40.0)


# ---------------------------------------------------------------------------
# Wrapper driver shared by the pass-level tests below
# ---------------------------------------------------------------------------


def _run_pass(pag, sigmas):
    """Call the model wrapper once per step of ``sigmas`` ([cond, uncond] batch).

    Returns ``(batches, adg_steps)``: the batch each call sent to the model
    (2 = no weak row, +1 per PAG/SEG and SLG row, 1 = ADG cond-only) and the
    steps Adaptive Guidance skipped. ``shared.state`` carries the step, as in
    Forge, so the step-fraction gates (ADG, SEG/SLG) see it.
    """
    batches, adg_steps = [], []
    for step, sigma in enumerate(sigmas):
        skipped = pag._STATE["adg_skipped_steps"]
        batches.append(_call_at(pag, step, len(sigmas), sigma))
        if pag._STATE["adg_skipped_steps"] > skipped:
            adg_steps.append(step)
    return batches, adg_steps


def _call_at(pag, step, total, sigma):
    """One wrapper call at ``step`` of ``total``; returns the model batch size."""
    seen = []

    def apply_model(x, timestep, **conditioning):
        seen.append(int(x.shape[0]))
        return conditioning["bias"].clone()

    pag._STATE["step_open"] = False
    shared = types.SimpleNamespace(
        state=types.SimpleNamespace(sampling_step=step, sampling_steps=total))
    with mock.patch.object(pag, "shared", shared):
        pag._model_wrapper(apply_model, {
            "input": torch.zeros(2, 1),
            "timestep": torch.full((2,), float(sigma)),
            "c": {"bias": torch.tensor([[0.0], [1.0]])},
            "cond_or_uncond": [1, 0],
        })
    return seen[0]


# ---------------------------------------------------------------------------
# 6) Adaptive Guidance (suite feature) with PAG's sigma window
# ---------------------------------------------------------------------------


class AdaptiveGuidanceWindowTests(unittest.TestCase):
    """ADG's start is a step fraction; PAG's window is upstream's sigma window.

    Adaptive Guidance is not in the upstream node. On the steps it skips, the
    wrapper runs cond-only before the PAG gate, so PAG never runs there. The PAG
    window itself stays the requested one (origin :25-34): ADG's step fraction
    used as a sigma percent moved PAG's end on non-uniform schedules, and an
    ADG start before the PAG start was swapped into a window nobody asked for.
    """

    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()
        cls.flow = _FlowPredictor()

    _process = AttachOriginTests._process

    def setUp(self):
        _BASE.AnimaSafePagTests.setUp(self)
        p = self.pag
        self.addCleanup(p._STATE.update, {"sigma_hi": None, "sigma_lo": None,
                                          "guard_params": None, "start": 0.0, "end": 1.0})
        self.addCleanup(p._ADG.update, {"on": False, "start": 0.5, "interval": 0})

    def _attach_and_run(self, sigmas, adg_start, **overrides):
        """Attach with ADG on (keep-every 0), then run one pass of the wrapper."""
        params, state, _request = self._process(
            self.flow, a17=adg_start is not None,
            a18=0.5 if adg_start is None else adg_start, a19=0, **overrides)
        p = self.pag
        _BASE.AnimaSafePagTests.setUp(self)  # the attach's postprocess cleared the state
        p._STATE.update({key: state[key] for key in ("sigma_hi", "sigma_lo", "start", "end")})
        p._ADG.update(on=adg_start is not None, start=adg_start or 0.0, interval=0)
        batches, adg_steps = _run_pass(p, sigmas)
        pag_steps = [step for step, batch in enumerate(batches) if batch == 3]
        return params, state, pag_steps, adg_steps

    def _origin_pag_steps(self, sigmas, start, end, adg_steps):
        sigma_start, sigma_end, _s, _e = _percent_range_to_sigmas(_ComfyModel(self.flow), start, end)
        return [step for step, sigma in enumerate(sigmas)
                if _sigma_active(float(sigma), sigma_start, sigma_end) and step not in adg_steps]

    def test_adg_starting_before_pag_keeps_the_requested_window(self):
        """PAG 0.5-0.9, ADG from 0.3 — the window is not swapped to sigma 0.875-0.75."""
        sigmas = _karras_sigmas(self.flow, 20)[:-1]
        params, state, pag_steps, adg_steps = self._attach_and_run(
            sigmas, 0.3, a8=0.5, a9=0.9)
        origin = _percent_range_to_sigmas(_ComfyModel(self.flow), 0.5, 0.9)
        self.assertEqual((state["sigma_hi"], state["sigma_lo"]), (origin[0], origin[1]))
        self.assertIn("pag_sigma_window=0.7500-0.2500", params["Anima Perturbation Guidance"])
        # The step-fraction range (SEG/SLG, no-predictor fallback) keeps the old cut.
        self.assertEqual((state["start"], state["end"]), (0.5, 0.3))
        self.assertEqual(state["range_mode"], "continuous-cut-by-adaptive")

        self.assertEqual(adg_steps, list(range(6, 20)))
        self.assertEqual(pag_steps, self._origin_pag_steps(sigmas, 0.5, 0.9, adg_steps))
        self.assertEqual(pag_steps, [2, 3, 4, 5])
        # Karras step 1 (sigma 0.81) is outside the requested window: never PAG.
        self.assertGreater(float(sigmas[1]), origin[0])
        self.assertNotIn(1, pag_steps)

        # The simple schedule reaches sigma 0.75 only after ADG took over: no PAG,
        # as with the old step-fraction gate.
        simple = _simple_sigmas(self.flow, 20)[:-1]
        _params, _state, pag_steps, adg_steps = self._attach_and_run(
            simple, 0.3, a8=0.5, a9=0.9)
        self.assertEqual(pag_steps, [])
        self.assertEqual(self._origin_pag_steps(simple, 0.5, 0.9, adg_steps), [])

    def test_adg_does_not_move_pag_end_on_karras(self):
        """Default PAG 0-0.7, ADG from 0.5: PAG keeps upstream's steps [0, 1, 2]."""
        sigmas = _karras_sigmas(self.flow, 20)[:-1]
        _params, state_off, pag_off, adg_off = self._attach_and_run(sigmas, None)
        _params, state_on, pag_on, adg_on = self._attach_and_run(sigmas, 0.5)
        self.assertEqual(adg_off, [])
        self.assertEqual(adg_on, list(range(10, 20)))
        origin = _percent_range_to_sigmas(_ComfyModel(self.flow), 0.0, 0.7)
        for state in (state_off, state_on):
            self.assertEqual((state["sigma_hi"], state["sigma_lo"]), (origin[0], origin[1]))
        self.assertEqual(pag_off, self._origin_pag_steps(sigmas, 0.0, 0.7, []))
        self.assertEqual(pag_on, pag_off)
        self.assertEqual(pag_on, [0, 1, 2])

    def test_adg_skip_still_ends_pag_where_it_starts(self):
        """The suite's ADG cut (plan §2.2 #9, kept) holds through the ADG branch."""
        sigmas = _simple_sigmas(self.flow, 20)[:-1]
        _params, _state, pag_steps, adg_steps = self._attach_and_run(sigmas, 0.3)
        self.assertEqual(adg_steps, list(range(6, 20)))
        self.assertEqual(pag_steps, list(range(0, 6)))
        self.assertFalse(set(pag_steps) & set(adg_steps))


# ---------------------------------------------------------------------------
# 7) SEG and SLG keep their step-fraction gate (not in the safe-pag original)
# ---------------------------------------------------------------------------


class SegSlgGateTests(unittest.TestCase):
    """Only PAG follows upstream's sigma window.

    SEG and SLG are not part of iljung1106/comfyui-anima-safe-pag (parity plan
    §0.3 item 9, out of scope), so they keep the step-fraction gate they had —
    the ComfyUI pack keeps its own SEG/SLG gate for the same reason.
    """

    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()
        cls.flow = _FlowPredictor()

    _process = AttachOriginTests._process

    def setUp(self):
        _BASE.AnimaSafePagTests.setUp(self)
        p = self.pag
        sigma_hi, sigma_lo = p._pag_sigma_window(_unet_with(self.flow), 0.0, 0.7)
        p._STATE.update(sigma_hi=sigma_hi, sigma_lo=sigma_lo, start=0.0, end=0.7)
        self.addCleanup(p._STATE.update, {"sigma_hi": None, "sigma_lo": None, "start": 0.0,
                                          "end": 1.0, "slg_on": False, "slg_targets": set(),
                                          "attn_method": "pag"})
        self.sigmas = _simple_sigmas(self.flow, 20)[:-1]
        # Step fraction step/19 <= 0.7: steps 0-13. PAG's sigma window: steps 0-14.
        self.step_steps = list(range(14))

    def test_pag_and_slg_rows_each_keep_their_window(self):
        p = self.pag
        p._STATE.update(slg_on=True, slg_scale=3.0, slg_targets={0})
        batches, _adg = _run_pass(p, self.sigmas)
        self.assertEqual(batches, [4] * 14 + [3] + [2] * 5)
        # Step 14: PAG's row only (sigma 0.5625, the inclusive lower bound;
        # step fraction 14/19 > 0.7).
        self.assertEqual(_call_at(p, 14, 20, self.sigmas[14]), 3)
        self.assertIsNotNone(p._STATE["attn_raw"])
        self.assertIsNone(p._STATE["slg_raw"])

    def test_seg_rows_follow_the_step_fraction(self):
        p = self.pag
        p._STATE.update(attn_method="seg")
        batches, _adg = _run_pass(p, self.sigmas)
        self.assertEqual([step for step, batch in enumerate(batches) if batch == 3], self.step_steps)

    def test_slg_only_follows_the_step_fraction(self):
        p = self.pag
        p._STATE.update(attn_method=None, slg_on=True, slg_scale=3.0, slg_targets={0})
        batches, _adg = _run_pass(p, self.sigmas)
        self.assertEqual([step for step, batch in enumerate(batches) if batch == 3], self.step_steps)

    def test_attach_stores_a_sigma_window_only_for_pag(self):
        for label, overrides in (("SEG", {"a1": "SEG"}),
                                 ("SLG only", {"a1": "None", "a5": True, "a7": "3"})):
            with self.subTest(label):
                params, state, _request = self._process(self.flow, **overrides)
                self.assertIsNone(state["sigma_hi"])
                self.assertNotIn("pag_sigma_window", params["Anima Perturbation Guidance"])
        params, state, _request = self._process(self.flow, a5=True, a7="3")  # PAG + SLG
        self.assertEqual(state["sigma_hi"], 1.0)
        self.assertIn("pag_sigma_window=1.0000-0.5625", params["Anima Perturbation Guidance"])


# ---------------------------------------------------------------------------
# 8) The ControlNet guard note survives an unblocked hires pass
# ---------------------------------------------------------------------------


class ControlNetGuardHiresTests(unittest.TestCase):
    """Forge ControlNet "Low res only" blocks PAG in the base pass only.

    The hires pass runs process_before_every_sampling again on the same ``p``
    (Haoming02/sd-webui-forge-classic@e33f40e4:modules/processing.py:1453,
    1546); its clean-up must not drop the base pass's guard note.
    """

    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    _process = AttachOriginTests._process

    def test_clear_keeps_only_the_guard_on_the_hires_pass(self):
        p = self.pag
        guard = p.INFOTEXT_CONTROLNET_GUARD
        for hires in (False, True, None):
            with self.subTest(hires=hires):
                params = {key: "x" for key in p._EXTRA_GENERATION_PARAM_KEYS}
                params["Steps"] = 20
                request = types.SimpleNamespace(extra_generation_params=params)
                if hires is not None:
                    request.is_hr_pass = hires
                p._clear_extra_generation_params(request)
                expected = {"Steps": 20}
                if hires:
                    expected[guard] = "x"
                self.assertEqual(params, expected)

    def test_base_pass_guard_survives_an_unblocked_hires_pass(self):
        p = self.pag
        blocked_base = {
            "Steps": 20,
            p.INFOTEXT_CONTROLNET_GUARD: p.INFOTEXT_CONTROLNET_GUARD_VALUE,
            "Anima Perturbation Guidance": "stale base-pass value",
        }
        params, _state, _request = self._process(
            _FlowPredictor(), params=blocked_base, is_hr_pass=True)
        self.assertEqual(params[p.INFOTEXT_CONTROLNET_GUARD], p.INFOTEXT_CONTROLNET_GUARD_VALUE)
        self.assertIn("pag_sigma_window=", params["Anima Perturbation Guidance"])
        # The next base pass (next batch or XYZ cell) starts clean.
        params, _state, _request = self._process(_FlowPredictor(), params=blocked_base)
        self.assertNotIn(p.INFOTEXT_CONTROLNET_GUARD, params)


if __name__ == "__main__":
    unittest.main()
