"""SAM3 changes from the 2026-10-02 review (docs/review_proposals_20261002/sam3).

- 01: Target and Exclude empty + a manual mask → the drawn mask is used without loading SAM3
  (no device check, no checkpoint resolution), same mask contract as the old fallback.
- 05 (= 02 + 03): Refine PNG metadata (detection/mask settings, pre-blur mask SHA-256) and the
  list-compatible ``RefineResults`` outcome facts with their UI messages.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

core = importlib.import_module("sam3ext.core")
from sam3ext import ui_refine  # noqa: E402
from tests.test_sam3_core_cache import _inpaint_core  # noqa: E402


def _image(w=8, h=6, value=100):
    return Image.fromarray(np.full((h, w, 3), value, dtype=np.uint8), mode="RGB")


class ManualMaskShortcutTests(unittest.TestCase):
    def _run(self, mask, prompt="", exclude="", image=None):
        return core.run_sam3_on_pil(
            image or _image(), prompt, 0.4, "sam3.pt", "cuda", exclude_prompt=exclude, user_mask=mask,
        )

    def test_manual_mask_skips_device_and_checkpoint(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:4, 2:6] = True
        with mock.patch.object(core, "_resolve_device", side_effect=AssertionError("device")), \
                mock.patch.object(core, "resolve_checkpoint_path", side_effect=AssertionError("checkpoint")):
            result = self._run(mask)
        self.assertEqual((result.device, result.checkpoint), ("manual", "not used (manual mask)"))
        self.assertEqual((result.boxes, result.scores), ([], []))
        np.testing.assert_array_equal(np.asarray(result.mask) > 127, mask)
        self.assertEqual(result.mask.mode, "L")
        self.assertEqual(len(result.masks), 1)
        np.testing.assert_array_equal(np.asarray(result.masks[0]), np.asarray(result.mask))
        overlay = np.asarray(result.overlay)
        expected = (np.array([100, 100, 100]) * 0.35 + np.array([30, 210, 255]) * 0.65).astype(np.uint8)
        np.testing.assert_array_equal(overlay[2, 3], expected)   # same tint as the old fallback
        np.testing.assert_array_equal(overlay[0, 0], [100, 100, 100])

    def test_mask_is_resized_nearest_to_the_image(self):
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 0] = True
        result = self._run(mask)
        got = np.asarray(result.mask) > 127
        self.assertEqual(got.shape, (6, 8))
        self.assertTrue(got[:2, :2].all())
        self.assertFalse(got[2:, :].any())

    def test_empty_manual_mask_gives_no_pass(self):
        result = self._run(np.zeros((6, 8), dtype=bool))
        self.assertEqual(result.masks, [])
        self.assertFalse(np.asarray(result.mask).any())

    def test_text_or_exclude_keeps_the_detection_path(self):
        mask = np.ones((6, 8), dtype=bool)
        for prompt, exclude in (("face", ""), ("", "eyes"), ("  face ", "  ")):
            with self.subTest(prompt=prompt, exclude=exclude), \
                    mock.patch.object(core, "_resolve_device", side_effect=RuntimeError("detection path")):
                with self.assertRaisesRegex(RuntimeError, "detection path"):
                    self._run(mask, prompt=prompt, exclude=exclude)

    def test_whitespace_target_counts_as_empty(self):
        with mock.patch.object(core, "_resolve_device", side_effect=AssertionError("device")):
            result = self._run(np.ones((6, 8), dtype=bool), prompt="   ", exclude=" ")
        self.assertEqual(result.device, "manual")

    def test_bad_mask_rank_is_an_error(self):
        with self.assertRaises(ValueError):
            self._run(np.zeros((2, 6, 8), dtype=bool))


class RefineOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inpaint_core = _inpaint_core()

    def _results(self, **facts):
        results = self.inpaint_core.RefineResults(detected_masks=facts.pop("detected", 2))
        for key, value in facts.items():
            setattr(results, key, value)
        return results

    def test_finish_reasons(self):
        cases = (
            ({"interrupted": True}, [], "interrupted"),
            ({}, [("img", "info")], "ok"),
            ({"failed_passes": 1}, [("img", "info")], "partial"),
            ({"empty_passes": 1}, [("img", "info")], "partial"),
            ({"failed_passes": 2, "attempted_passes": 2}, [], "passes_failed"),
            ({}, [], "no_images"),
        )
        for facts, items, reason in cases:
            with self.subTest(reason=reason):
                results = self._results(**facts)
                results.extend(items)
                self.assertEqual(results.finish().reason, reason)
        none = self.inpaint_core.RefineResults(reason="empty_mask")
        self.assertEqual(none.finish().reason, "empty_mask")

    def test_still_a_list_for_callers(self):
        results = self.inpaint_core.RefineResults()
        self.assertIsInstance(results, list)
        self.assertFalse(results)
        results.append(("img", "info"))
        self.assertEqual(len(results), 1)
        self.assertEqual(list(results), [("img", "info")])

    def test_ui_messages(self):
        make = self.inpaint_core.RefineResults
        for reason, needle in (("empty_mask", "검출된 마스크가 없습니다"), ("interrupted", "중단"),
                               ("runner_unavailable", "준비되지 않았습니다"), ("no_images", "반환하지 않았습니다")):
            with self.subTest(reason=reason):
                self.assertIn(needle, ui_refine._refine_empty_message(make(reason=reason)))
        failed = make(reason="passes_failed")
        failed.failed_passes, failed.attempted_passes = 2, 3
        self.assertIn("시도 3개 중 2개", ui_refine._refine_empty_message(failed))
        self.assertIn("콘솔을 확인하세요", ui_refine._refine_empty_message([]), "plain-list callers")
        ok = make()
        ok.extend([("a", "i"), ("b", "i")])
        self.assertIn("결과 2개를 추가했습니다.", ui_refine._refine_success_message(ok))
        self.assertIn("#383", ui_refine._refine_success_message(ok))
        ok.failed_passes, ok.interrupted = 1, True
        message = ui_refine._refine_success_message(ok)
        self.assertIn("오류 1개", message)
        self.assertIn("작업 중단", message)
        self.assertIn("#c80", message)

    def test_messages_carry_no_raw_exception_text(self):
        failed = self.inpaint_core.RefineResults(reason="passes_failed")
        failed.error = "<script>alert(1)</script>"
        self.assertNotIn("<script>", ui_refine._refine_empty_message(failed))


class RefineMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inpaint_core = _inpaint_core()

    def test_settings_and_pre_blur_mask_digest(self):
        mask = Image.fromarray((np.arange(48).reshape(6, 8) % 2 * 255).astype(np.uint8), mode="L")
        result = types.SimpleNamespace(checkpoint="C:/models/sam3.pt", device="cuda")
        args = {"sam3_prompt": "hair", "sam3_exclude_prompt": "face", "sam3_threshold": 0.35,
                "sam3_checkpoint": "sam3.pt", "sam3_mask_mode": "Individual", "sam3_mask_dilation": 4,
                "sam3_mask_hull": True, "sam3_mask_outline_px": 2, "sam3_mask_blur": 6,
                "sam3_mask_invert": False, "sam3_inpaint_only_masked": True,
                "sam3_inpaint_only_masked_padding": 48, "sam3_inpainting_fill": "original"}
        params = self.inpaint_core._refine_generation_params(args, result, mask, None, 2)
        expected = hashlib.sha256(b"8x6\0" + mask.tobytes()).hexdigest()
        self.assertEqual(params["SAM3 Refine Mask SHA256"], expected)
        self.assertEqual(params["SAM3 Refine Mask Dimensions"], "8x6")
        self.assertEqual(params["SAM3 Refine Mask Stage"], "pre-blur/pre-invert")
        self.assertEqual(params["SAM3 Refine Target"], "hair")
        self.assertEqual(params["SAM3 Refine Exclude"], "face")
        self.assertEqual(params["SAM3 Refine Pass"], 2)
        self.assertEqual(params["SAM3 Refine Checkpoint Used"], "C:/models/sam3.pt")
        self.assertEqual(params["SAM3 Refine Masked Content"], 1, "original → Forge fill 1")
        self.assertIs(params["SAM3 Refine Manual Mask"], False)
        self.assertNotIn("SAM3 Refine Manual Mask SHA256", params)

    def test_manual_mask_digest(self):
        mask = Image.new("L", (4, 2), 255)
        manual = np.zeros((2, 4), dtype=bool)
        manual[0, 1] = True
        params = self.inpaint_core._refine_generation_params(
            {}, types.SimpleNamespace(checkpoint="not used (manual mask)", device="manual"), mask, manual, 1)
        manual_bytes = (manual.astype(np.uint8) * 255).tobytes()
        self.assertEqual(params["SAM3 Refine Manual Mask SHA256"],
                         hashlib.sha256(b"4x2\0" + manual_bytes).hexdigest())
        self.assertEqual(params["SAM3 Refine Checkpoint Used"], "not used (manual mask)")
        self.assertEqual(params["SAM3 Refine Device"], "manual")


if __name__ == "__main__":
    unittest.main()
