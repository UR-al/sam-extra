"""Anima Optimal Scale (scripts/anima_cfg_optimal_scale.py) — CFG-Zero*'s optimized scale only.

The x0-space residual is checked against an independent velocity-space reference of the paper's
optimized-scale CFG; then the skip rules, the per-pass status infotext and hook ownership (foreign
callbacks kept, only this script's own callback removed). zero-init is not part of the feature.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "anima_cfg_optimal_scale.py"


def _load():
    modules = types.ModuleType("modules")
    modules.__path__ = []

    class Script:
        pass

    modules.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object())
    saved = {name: sys.modules.get(name) for name in ("modules", "modules.scripts")}
    sys.modules["modules"] = modules
    sys.modules["modules.scripts"] = modules.scripts
    try:
        spec = importlib.util.spec_from_file_location("_test_anima_optimal_scale", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _Predictor:
    prediction_type = "const"

    @staticmethod
    def percent_to_sigma(percent):
        return 1.0 - float(percent)


def _args(scale=4.0, sigma=0.6, seed=0, batch=2, **overrides):
    g = torch.Generator().manual_seed(seed)
    shape = (batch, 4, 1, 6, 6)
    x = torch.randn(shape, generator=g)
    cond = torch.randn(shape, generator=g)
    uncond = cond * 0.7 + 0.1 * torch.randn(shape, generator=g)
    args = {
        "denoised": uncond + (cond - uncond) * scale,
        "cond_denoised": cond, "uncond_denoised": uncond, "input": x,
        "sigma": torch.full((batch,), sigma), "cond_scale": scale,
        "cond": [{"strength": 1.0}], "uncond": [{"strength": 1.0}],
        "model": types.SimpleNamespace(predictor=_Predictor()), "model_options": {},
    }
    args.update(overrides)
    return args


class OptimalScaleMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_blend_one_is_the_optimized_scale_cfg_in_velocity_space(self):
        args = _args()
        x, cond, uncond = args["input"].double(), args["cond_denoised"].double(), args["uncond_denoised"].double()
        sigma, w = 0.6, 4.0
        v_c, v_u = (x - cond) / sigma, (x - uncond) / sigma          # x0 = x − σ·v
        dims = tuple(range(1, x.ndim))
        s = (v_c * v_u).sum(dims, keepdim=True) / ((v_u * v_u).sum(dims, keepdim=True) + 1e-8)
        v = s * v_u + w * (v_c - s * v_u)                           # CFG-Zero* optimized scale
        expected = (x - sigma * v).float()
        got = self.mod._optimal_scale_residual(args["denoised"], args["input"], args["cond_denoised"],
                                               args["uncond_denoised"], w, 1.0, args["sigma"])
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)

    def test_blend_is_linear_and_zero_is_the_same_object(self):
        args = _args()
        call = lambda blend: self.mod._optimal_scale_residual(  # noqa: E731
            args["denoised"], args["input"], args["cond_denoised"], args["uncond_denoised"], 4.0, blend,
            args["sigma"])
        full, quarter = call(1.0), call(0.25)
        torch.testing.assert_close(quarter, args["denoised"] + 0.25 * (full - args["denoised"]),
                                   rtol=1e-5, atol=1e-5)
        self.assertIs(call(0.0), args["denoised"])


class OptimalScaleCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def _callback(self, blend=1.0, start=0.0, end=1.0):
        params = {}
        return self.mod._make_callback(params, blend, start, end), params

    def test_applies_and_counts(self):
        callback, params = self._callback()
        args = _args()
        out = callback(args)
        self.assertIsNot(out, args["denoised"])
        self.assertEqual(params[self.mod.STATUS_KEY], "applied=1; skipped=0")

    def test_skip_rules_return_the_incoming_object(self):
        skimmed = lambda a: a["denoised"]  # noqa: E731
        skimmed._sam_extra_post_cfg_owner = "sam-extra/anima-skimmed-cfg"
        cases = {
            "custom CFG function": _args(model_options={"sampler_cfg_function": lambda a: a}),
            "Skimmed CFG modifies the prediction pair": _args(
                model_options={"sampler_post_cfg_function": [skimmed]}),
            "CFG <= 1 or skipped negative step": _args(scale=1.0),
            "negative not evaluated or identical predictions": _args(uncond=None),
            "non-const prediction": _args(model=types.SimpleNamespace(
                predictor=types.SimpleNamespace(prediction_type="epsilon"))),
            # window start 0 / end 0.1 → σ 1.0..0.9 on this flow predictor; σ 0.5 is outside
            "outside sigma window": _args(sigma=0.5),
            "prediction shape mismatch": _args(input=torch.zeros(2, 4, 1, 5, 5)),
        }
        mismatch = _args()
        mismatch["sigma"] = torch.full((3,), 0.6)
        cases["sigma batch mismatch"] = mismatch
        for reason, args in cases.items():
            with self.subTest(reason=reason):
                callback, params = self._callback(start=0.0, end=0.1 if reason == "outside sigma window" else 1.0)
                out = callback(args)
                self.assertIs(out, args["denoised"])
                self.assertTrue(params[self.mod.STATUS_KEY].endswith("last_skip=" + reason),
                                params[self.mod.STATUS_KEY])

    def test_prior_correction_is_left_alone(self):
        callback, params = self._callback()
        args = _args()
        args["denoised"] = args["denoised"] + 0.01      # an earlier post-CFG changed the result
        self.assertIs(callback(args), args["denoised"])
        self.assertIn("prior post-CFG correction", params[self.mod.STATUS_KEY])

    def test_identical_predictions_skip(self):
        callback, params = self._callback()
        args = _args()
        args["uncond_denoised"] = args["cond_denoised"]
        args["denoised"] = args["cond_denoised"].clone()
        self.assertIs(callback(args), args["denoised"])


class _Unet:
    def __init__(self, callbacks=()):
        self.model_options = {"sampler_post_cfg_function": list(callbacks)}
        self.clones = 0

    def clone(self):
        clone = _Unet(self.model_options.get("sampler_post_cfg_function", []))
        clone.model_options = dict(self.model_options)
        self.clones += 1
        return clone


class OptimalScaleHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def _p(self, unet, name="Anima"):
        model = type(name, (), {})()
        model.forge_objects = types.SimpleNamespace(unet=unet)
        return types.SimpleNamespace(sd_model=model, extra_generation_params={})

    def test_on_appends_after_foreign_callbacks_and_records(self):
        foreign = lambda a: a["denoised"]  # noqa: E731
        p = self._p(_Unet([foreign]))
        script = self.mod.AnimaOptimalScale()
        script.process_before_every_sampling(p, True, 0.5, 0.0, 1.0)
        callbacks = p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"]
        self.assertIs(callbacks[0], foreign)
        self.assertEqual(getattr(callbacks[1], "_sam_extra_optimal_scale_owner"), self.mod.OWNER)
        self.assertEqual(p.extra_generation_params[self.mod.KEY],
                         "blend=0.5; start=0; end=1; zero_init=omitted")
        self.assertEqual(p.extra_generation_params[self.mod.STATUS_KEY], "pending model evaluation")
        # a second attach on the same (reused) UNet keeps one own callback
        script.process_before_every_sampling(p, True, 0.5, 0.0, 1.0)
        callbacks = p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"]
        owned = [fn for fn in callbacks if getattr(fn, "_sam_extra_optimal_scale_owner", None) == self.mod.OWNER]
        self.assertEqual(len(owned), 1)
        self.assertIs(callbacks[0], foreign)

    def test_off_removes_only_its_own_callback(self):
        foreign = lambda a: a["denoised"]  # noqa: E731
        p = self._p(_Unet([foreign]))
        script = self.mod.AnimaOptimalScale()
        script.process_before_every_sampling(p, True, 0.5, 0.0, 1.0)
        patched = p.sd_model.forge_objects.unet
        script.process_before_every_sampling(p, False, 0.5, 0.0, 1.0)
        unet = p.sd_model.forge_objects.unet
        self.assertIsNot(unet, patched, "the patched UNet is cloned, not edited in place")
        self.assertEqual(unet.model_options["sampler_post_cfg_function"], [foreign])
        self.assertNotIn(self.mod.KEY, p.extra_generation_params)
        self.assertNotIn(self.mod.STATUS_KEY, p.extra_generation_params)

    def test_off_and_zero_blend_do_not_touch_a_clean_unet(self):
        for args in ((False, 0.5, 0.0, 1.0), (True, 0.0, 0.0, 1.0), (True, 0.5, 0.6, 0.4)):
            with self.subTest(args=args):
                unet = _Unet()
                p = self._p(unet)
                self.mod.AnimaOptimalScale().process_before_every_sampling(p, *args)
                self.assertIs(p.sd_model.forge_objects.unet, unet)
                self.assertEqual(unet.clones, 0)
                self.assertEqual(p.extra_generation_params, {})

    def test_other_models_are_left_alone(self):
        unet = _Unet()
        p = self._p(unet, name="StableDiffusionXL")
        self.mod.AnimaOptimalScale().process_before_every_sampling(p, True, 0.5, 0.0, 1.0)
        self.assertIs(p.sd_model.forge_objects.unet, unet)
        self.assertEqual(p.extra_generation_params, {})

    def test_title_and_zero_init_note(self):
        script = self.mod.AnimaOptimalScale()
        self.assertEqual(script.title(), "Anima Optimal Scale")
        self.assertIn("zero-init", SCRIPT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
