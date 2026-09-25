from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_skim_module():
    """Load the extension script without booting the full WebUI."""
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    modules_stub.scripts = types.SimpleNamespace(
        Script=Script,
        AlwaysVisible=object(),
        scripts_data=[],
    )
    modules_stub.shared = types.SimpleNamespace(
        state=types.SimpleNamespace(sampling_step=0, sampling_steps=20)
    )
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_before_ui=lambda fn: None
    )

    old_modules = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_anima_skimmed_cfg", ROOT / "scripts" / "anima_skimmed_cfg.py"
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


def _reference_skim(x_orig, cond, uncond, cond_scale, skimming_scale,
                    disable_flipping_filter=False):
    """Upstream Extraltodeus/Skimmed_CFG maths, transcribed for comparison."""
    cond = cond.clone()
    denoised = x_orig - (
        (x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    )
    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        deviation_influence = denoised.sign() == (denoised - x_orig).sign()
        outer_influence = (
            matching_pred_signs & matching_diff_after & deviation_influence
        )
    low_cfg_denoised_outer = x_orig - (
        (x_orig - uncond)
        + skimming_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (
        difference[outer_influence] / cond_scale
    )
    return cond


def _legacy_skim_predictions(x, target, reference, scale, skimming_scale,
                             flip_filter):
    """The pre-``torch.where`` implementation, kept verbatim as a bit-exact
    oracle: a host ``.any()`` check plus three boolean-mask gathers/scatter,
    each of which synchronises the GPU."""
    if abs(float(scale)) < 1e-6:
        return target

    denoised = reference + scale * (target - reference)
    matching_pred_signs = (target - reference).sign() == target.sign()
    matching_diff_after = target.sign() == denoised.sign()
    outer_influence = matching_pred_signs & matching_diff_after
    if not flip_filter:
        outer_influence &= denoised.sign() == (denoised - x).sign()

    if not bool(outer_influence.any()):
        return target

    low_scale_denoised = reference + skimming_scale * (target - reference)
    correction = (denoised - low_scale_denoised) / scale
    skimmed = target.clone()
    skimmed[outer_influence] = target[outer_influence] - correction[outer_influence]
    return skimmed


class SkimmedCFGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skim = _load_skim_module()

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(1, 4, 8, 8)
        self.cond = torch.randn(1, 4, 8, 8)
        self.uncond = torch.randn(1, 4, 8, 8)
        self.skim._SKIM.update(
            on=False, skimming_cfg=7.0, full_skim_negative=False,
            disable_flipping_filter=False, start=0.0, end=1.0, flip_at=0.0,
            steps=0, warned=False,
        )

    def _args(self, cond_scale=8.0):
        denoised = self.uncond + cond_scale * (self.cond - self.uncond)
        return {
            "denoised": denoised,
            "input": self.x,
            "cond_denoised": self.cond,
            "uncond_denoised": self.uncond,
            "cond_scale": cond_scale,
        }

    def test_matches_upstream_reference_maths(self):
        for flip in (False, True):
            with self.subTest(disable_flipping_filter=flip):
                out = self.skim._skim_predictions(
                    self.x, self.cond, self.uncond, 8.0, 7.0, flip
                )
                expected = _reference_skim(
                    self.x, self.cond, self.uncond, 8.0, 7.0, flip
                )
                torch.testing.assert_close(out, expected)

    def test_where_skim_is_bit_identical_to_legacy_mask_indexing(self):
        """The ``torch.where`` rewrite must reproduce the old boolean-mask
        scatter bit for bit (torch.equal, not a tolerance), including the
        CFG-1 shortcut and a mask that selects nothing."""
        shapes = ((1, 4, 8, 8), (2, 16, 1, 12, 10))
        cases = 0
        for seed in range(6):
            for shape in shapes:
                g = torch.Generator().manual_seed(seed)
                x = torch.randn(shape, generator=g)
                target = torch.randn(shape, generator=g)
                reference = torch.randn(shape, generator=g)
                for scale in (8.0, 7.0, 4.5, 1.5, 0.5, -2.0, 0.0):
                    for skimming in (7.0, 3.0, 0.0, 12.0):
                        for flip in (False, True):
                            with self.subTest(seed=seed, shape=shape,
                                              scale=scale, skim=skimming,
                                              flip=flip):
                                new = self.skim._skim_predictions(
                                    x, target, reference, scale, skimming,
                                    flip,
                                )
                                old = _legacy_skim_predictions(
                                    x, target, reference, scale, skimming,
                                    flip,
                                )
                                self.assertTrue(torch.equal(new, old))
                                self.assertEqual(new.dtype, old.dtype)
                                cases += 1
        self.assertGreater(cases, 600)

        # A mask that selects nothing: target == reference makes the guidance
        # direction zero, so ``matching_pred_signs`` is False everywhere.
        same = torch.randn(1, 4, 8, 8)
        new = self.skim._skim_predictions(self.x, same, same.clone(), 8.0, 3.0, False)
        self.assertTrue(torch.equal(new, same))

    def test_post_cfg_is_bit_identical_to_legacy_implementation(self):
        """Whole hook, including the in-place publish into Forge's cond/uncond
        tensors, against the legacy helper, for every storage dtype."""
        original = self.skim._skim_predictions
        configs = (
            dict(skimming_cfg=7.0),
            dict(skimming_cfg=3.0),
            dict(skimming_cfg=-1.0),
            dict(skimming_cfg=3.0, full_skim_negative=True),
            dict(skimming_cfg=2.0, disable_flipping_filter=True),
            dict(skimming_cfg=2.0, flip_at=0.5),
        )
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for config in configs:
                for cond_scale in (8.0, 4.0, 1.5):
                    with self.subTest(dtype=dtype, config=config,
                                      cond_scale=cond_scale):
                        g = torch.Generator().manual_seed(11)
                        x = torch.randn(2, 16, 1, 16, 16, generator=g).to(dtype)
                        cond = torch.randn(2, 16, 1, 16, 16, generator=g).to(dtype)
                        uncond = torch.randn(2, 16, 1, 16, 16, generator=g).to(dtype)

                        def run(helper):
                            self.skim._SKIM.update(
                                on=True, skimming_cfg=7.0,
                                full_skim_negative=False,
                                disable_flipping_filter=False,
                                start=0.0, end=1.0, flip_at=0.0,
                                steps=0, warned=False,
                            )
                            self.skim._SKIM.update(**config)
                            c, u = cond.clone(), uncond.clone()
                            args = {
                                "denoised": (u + cond_scale * (c - u)),
                                "input": x,
                                "cond_denoised": c,
                                "uncond_denoised": u,
                                "cond_scale": cond_scale,
                            }
                            self.skim._skim_predictions = helper
                            try:
                                out = self.skim._post_cfg(args)
                            finally:
                                self.skim._skim_predictions = original
                            return out, c, u

                        new_out, new_c, new_u = run(original)
                        old_out, old_c, old_u = run(_legacy_skim_predictions)
                        self.assertTrue(torch.equal(new_out, old_out))
                        self.assertTrue(torch.equal(new_c, old_c))
                        self.assertTrue(torch.equal(new_u, old_u))
                        self.assertEqual(new_out.dtype, dtype)

    def test_skim_never_reads_the_mask_back_to_the_host(self):
        """No ``.any()``/``.item()``/boolean-mask indexing: each of those is a
        GPU->CPU synchronisation in the per-step post-CFG hook."""
        getitem = torch.Tensor.__getitem__
        setitem = torch.Tensor.__setitem__

        def is_mask(key):
            return torch.is_tensor(key) and key.dtype == torch.bool

        def guarded_getitem(tensor, key):
            if is_mask(key):
                raise AssertionError("boolean-mask gather syncs the GPU")
            return getitem(tensor, key)

        def guarded_setitem(tensor, key, value):
            if is_mask(key):
                raise AssertionError("boolean-mask scatter syncs the GPU")
            return setitem(tensor, key, value)

        def host_sync(*_args, **_kwargs):
            raise AssertionError("host sync")

        with mock.patch.object(torch.Tensor, "__getitem__", guarded_getitem), \
                mock.patch.object(torch.Tensor, "__setitem__", guarded_setitem), \
                mock.patch.object(torch.Tensor, "any", host_sync), \
                mock.patch.object(torch.Tensor, "item", host_sync), \
                mock.patch.object(torch.Tensor, "__bool__", host_sync):
            out = self.skim._skim_predictions(
                self.x, self.cond, self.uncond, 8.0, 3.0, False
            )
        expected = _legacy_skim_predictions(
            self.x, self.cond, self.uncond, 8.0, 3.0, False
        )
        self.assertTrue(torch.equal(out, expected))

    def test_predictions_are_not_mutated_in_place(self):
        cond_before = self.cond.clone()
        self.skim._skim_predictions(
            self.x, self.cond, self.uncond, 8.0, 7.0, False
        )
        torch.testing.assert_close(self.cond, cond_before)

    def test_equal_scales_leave_predictions_untouched(self):
        """skimming_cfg == cond_scale has nothing to pull back."""
        out = self.skim._skim_predictions(
            self.x, self.cond, self.uncond, 8.0, 8.0, False
        )
        torch.testing.assert_close(out, self.cond)

    def test_disabled_hook_preserves_incoming(self):
        args = self._args()
        out = self.skim._post_cfg(args)
        self.assertIs(out, args["denoised"])

    def test_enabled_hook_changes_the_result(self):
        self.skim._SKIM.update(on=True)
        args = self._args()
        out = self.skim._post_cfg(args)
        self.assertEqual(out.shape, args["denoised"].shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.allclose(out, args["denoised"]))
        self.assertEqual(self.skim._SKIM["steps"], 1)

    def test_cfg_scale_one_is_skipped(self):
        self.skim._SKIM.update(on=True)
        args = self._args(cond_scale=1.0)
        out = self.skim._post_cfg(args)
        self.assertIs(out, args["denoised"])

    def test_percent_range_gates_the_hook(self):
        self.skim._SKIM.update(on=True, start=0.5, end=1.0)
        args = self._args()
        # shared.state stub reports step 0 of 20 -> 0%
        self.assertIs(self.skim._post_cfg(args), args["denoised"])

    def test_negative_skimming_cfg_follows_the_live_scale(self):
        self.skim._SKIM.update(on=True, skimming_cfg=-1.0)
        args = self._args(cond_scale=8.0)
        out = self.skim._post_cfg(args)
        self.assertTrue(torch.isfinite(out).all())

    def test_full_skim_negative_runs(self):
        self.skim._SKIM.update(on=True, full_skim_negative=True)
        args = self._args()
        out = self.skim._post_cfg(args)
        self.assertTrue(torch.isfinite(out).all())

    def test_skim_is_published_to_downstream_guidance(self):
        """Upstream is a pre-CFG node, so later guidance must see the skim."""
        # skimming_cfg 3 < cond_scale - 1, so BOTH predictions get skimmed;
        # at the default 7 with CFG 8 the positive is skimmed at an identical
        # scale and only the negative changes.
        self.skim._SKIM.update(on=True, skimming_cfg=3.0)
        args = self._args()
        cond_before = args["cond_denoised"].clone()
        uncond_before = args["uncond_denoised"].clone()

        out = self.skim._post_cfg(args)

        # Forge reuses these tensors for every later post-CFG consumer.
        self.assertFalse(torch.allclose(args["cond_denoised"], cond_before))
        self.assertFalse(torch.allclose(args["uncond_denoised"], uncond_before))
        # A downstream consumer rebuilding the base from the published
        # predictions must land on our result, not on the unskimmed one.
        rebuilt = args["uncond_denoised"] + 8.0 * (
            args["cond_denoised"] - args["uncond_denoised"]
        )
        torch.testing.assert_close(rebuilt, out)

    def test_gated_out_steps_publish_nothing(self):
        self.skim._SKIM.update(on=True, start=0.5, end=1.0)
        args = self._args()
        cond_before = args["cond_denoised"].clone()

        self.skim._post_cfg(args)

        torch.testing.assert_close(args["cond_denoised"], cond_before)

    def test_missing_uncond_preserves_incoming(self):
        self.skim._SKIM.update(on=True)
        args = self._args()
        args["uncond_denoised"] = torch.zeros(1, 4, 4, 4)  # shape mismatch
        self.assertIs(self.skim._post_cfg(args), args["denoised"])

    def test_prepend_is_first_deduplicated_and_preserves_other_callbacks(self):
        calls = []

        def other_a(args):
            calls.append("a")
            return args["denoised"]

        def stale_skim(args):
            calls.append("stale")
            return args["denoised"]

        stale_skim._sam_extra_post_cfg_owner = self.skim._POST_CFG_OWNER

        def other_b(args):
            calls.append("b")
            return args["denoised"]

        unet = types.SimpleNamespace(
            model_options={
                "unrelated": 7,
                "sampler_post_cfg_function": [other_a, stale_skim, other_b],
            }
        )

        self.skim._prepend_post_cfg_function(unet)
        callbacks = unet.model_options["sampler_post_cfg_function"]

        self.assertIs(callbacks[0], self.skim._post_cfg)
        self.assertEqual(callbacks[1:], [other_a, other_b])
        self.assertEqual(
            sum(
                getattr(fn, "_sam_extra_post_cfg_owner", None)
                == self.skim._POST_CFG_OWNER
                for fn in callbacks
            ),
            1,
        )
        self.assertEqual(unet.model_options["unrelated"], 7)

    def test_logging_survives_a_legacy_console_encoding(self):
        """A log line must never be able to abort a generation.

        Several messages carry non-ASCII characters (em dash, check mark) and
        _log is called from inside the post-CFG hook, so on a Windows console
        using a legacy code page print() raising UnicodeEncodeError would kill
        the sampler run. Reproduce that stdout and assert the hook still returns.
        """
        import io

        class LegacyStream(io.TextIOBase):
            encoding = "cp949"

            def write(self, text):  # noqa: D401 - stream protocol
                text.encode("cp949")  # raises on characters cp949 lacks
                return len(text)

        self.skim._SKIM.update(on=True, warned=False)
        args = self._args(cond_scale=1.0)
        original = sys.stdout
        sys.stdout = LegacyStream()
        try:
            # Directly prove the message is unencodable on this stream...
            with self.assertRaises(UnicodeEncodeError):
                sys.stdout.write("em dash — here")
            # ...yet the hook completes and preserves the incoming result.
            out = self.skim._post_cfg(args)
        finally:
            sys.stdout = original

        self.assertIs(out, args["denoised"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
