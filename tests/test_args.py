from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.args import Sam3Args  # noqa: E402


class Sam3ArgsValidationTests(unittest.TestCase):
    def test_device_passthrough_and_fallback(self):
        for good in ("auto", "cpu", "cuda", "CUDA", "cuda:1"):
            self.assertIn(
                Sam3Args(sam3_device=good).sam3_device,
                {"auto", "cpu", "cuda", "cuda:1"},
            )
        # Unknown / malformed device strings fall back to auto.
        self.assertEqual(Sam3Args(sam3_device="gpu0").sam3_device, "auto")
        self.assertEqual(Sam3Args(sam3_device="cuda:x").sam3_device, "auto")
        self.assertEqual(Sam3Args(sam3_device="").sam3_device, "auto")

    def test_seed_is_clamped(self):
        self.assertEqual(Sam3Args(sam3_seed=-1).sam3_seed, -1)
        self.assertEqual(Sam3Args(sam3_seed=-99).sam3_seed, -1)
        self.assertEqual(Sam3Args(sam3_seed=12345).sam3_seed, 12345)
        self.assertEqual(Sam3Args(sam3_seed=2 ** 40).sam3_seed, 2 ** 32 - 1)

    def test_inpaint_size_snaps_to_multiple_of_8(self):
        self.assertEqual(Sam3Args(sam3_inpaint_width=513).sam3_inpaint_width, 512)
        self.assertEqual(Sam3Args(sam3_inpaint_height=100).sam3_inpaint_height, 96)
        # Below the floor snaps up to 64.
        self.assertEqual(Sam3Args(sam3_inpaint_width=10).sam3_inpaint_width, 64)
        self.assertEqual(Sam3Args(sam3_inpaint_width=512).sam3_inpaint_width, 512)

    def test_transposed_cn_guidance_window_is_swapped(self):
        args = Sam3Args(sam3_cn_guidance_start=0.8, sam3_cn_guidance_end=0.2)
        self.assertEqual(args.sam3_cn_guidance_start, 0.2)
        self.assertEqual(args.sam3_cn_guidance_end, 0.8)
        # A normal window is left untouched.
        ok = Sam3Args(sam3_cn_guidance_start=0.1, sam3_cn_guidance_end=0.9)
        self.assertEqual((ok.sam3_cn_guidance_start, ok.sam3_cn_guidance_end), (0.1, 0.9))

    def test_defaults_still_valid(self):
        args = Sam3Args()
        self.assertEqual(args.sam3_device, "auto")
        self.assertEqual(args.sam3_inpaint_width, 512)
        self.assertEqual(args.sam3_cn_guidance_start, 0.0)


class Sam3ArgsCoercionTests(unittest.TestCase):
    """감사 M4: 범위를 벗어난 수치·표기가 다른 Literal 은 예외(→ SAM3 무음 비활성화) 대신 클램프/정규화한다."""

    def test_out_of_range_numbers_are_clamped(self):
        self.assertEqual(Sam3Args(sam3_threshold=1.5).sam3_threshold, 1.0)
        self.assertEqual(Sam3Args(sam3_threshold=-0.2).sam3_threshold, 0.0)
        self.assertEqual(Sam3Args(sam3_mask_blur=-1).sam3_mask_blur, 0)
        self.assertEqual(Sam3Args(sam3_mask_dilation=-5).sam3_mask_dilation, 0)
        self.assertEqual(Sam3Args(sam3_mask_outline_px=-1).sam3_mask_outline_px, 0)
        self.assertEqual(Sam3Args(sam3_denoising_strength=2).sam3_denoising_strength, 1.0)
        self.assertEqual(Sam3Args(sam3_inpaint_only_masked_padding=-8).sam3_inpaint_only_masked_padding, 0)
        self.assertEqual(Sam3Args(sam3_steps=0).sam3_steps, 1)
        self.assertEqual(Sam3Args(sam3_cfg_scale=-3).sam3_cfg_scale, 0.0)
        self.assertEqual(Sam3Args(sam3_noise_multiplier=5).sam3_noise_multiplier, 2.0)
        self.assertEqual(Sam3Args(sam3_cn_weight=-1).sam3_cn_weight, 0.0)
        self.assertEqual(Sam3Args(sam3_cn_guidance_end=1.7).sam3_cn_guidance_end, 1.0)
        self.assertEqual(Sam3Args(sam3_cn_processor_res=-64).sam3_cn_processor_res, 0)
        # 0 은 PositiveInt 검사 전에 8 의 배수(최소 64)로 맞춘다.
        self.assertEqual(Sam3Args(sam3_inpaint_width=0).sam3_inpaint_width, 64)
        # 범위 안의 값은 그대로.
        self.assertEqual(Sam3Args(sam3_threshold=0.55).sam3_threshold, 0.55)
        self.assertEqual(Sam3Args(sam3_mask_blur=12).sam3_mask_blur, 12)

    def test_unparseable_numbers_fall_back_to_the_default(self):
        self.assertEqual(Sam3Args(sam3_threshold="abc").sam3_threshold, 0.4)
        self.assertEqual(Sam3Args(sam3_threshold=float("nan")).sam3_threshold, 0.4)
        self.assertEqual(Sam3Args(sam3_mask_blur=None).sam3_mask_blur, 4)
        self.assertEqual(Sam3Args(sam3_steps="").sam3_steps, 28)
        # 숫자 문자열(XYZ 축·API)은 여전히 받아들인다.
        self.assertEqual(Sam3Args(sam3_threshold="0.7").sam3_threshold, 0.7)
        self.assertEqual(Sam3Args(sam3_mask_blur="6").sam3_mask_blur, 6)

    def test_literal_values_are_normalised(self):
        self.assertEqual(Sam3Args(sam3_inpainting_fill="Original").sam3_inpainting_fill, "original")
        self.assertEqual(Sam3Args(sam3_inpainting_fill=" Latent Noise ").sam3_inpainting_fill, "latent noise")
        self.assertEqual(Sam3Args(sam3_inpainting_fill="bogus").sam3_inpainting_fill, "original")
        self.assertEqual(Sam3Args(sam3_inpainting_fill=None).sam3_inpainting_fill, "original")
        self.assertEqual(Sam3Args(sam3_mode="inpaint").sam3_mode, "Inpaint")
        self.assertEqual(Sam3Args(sam3_mode="mask only").sam3_mode, "Mask only")
        self.assertEqual(Sam3Args(sam3_mask_mode="combined").sam3_mask_mode, "Combined")
        self.assertEqual(Sam3Args(sam3_cn_control_mode="???").sam3_cn_control_mode, "Balanced")
        self.assertEqual(Sam3Args(sam3_cn_resize_mode="just resize").sam3_cn_resize_mode, "Just Resize")

    def test_unknown_keys_still_raise(self):
        # 오타 키는 진짜 오류다 — 무시하지 않는다(extra=forbid). 호출자가 로그로 알린다(tests/test_sam3_script.py ProcessFallbackTests).
        with self.assertRaises(Exception):
            Sam3Args(bogus_key=1)

    def test_unknown_literal_value_is_reported_on_stderr(self):
        # 후속: 모르는 값을 기본값으로 바꿀 때 조용히 넘어가지 않고 한 줄 남긴다(값은 그대로 기본값).
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            fill = Sam3Args(sam3_inpainting_fill="bogus").sam3_inpainting_fill
        self.assertEqual(fill, "original")
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("sam3_inpainting_fill", lines[0])
        self.assertIn("'bogus'", lines[0])
        self.assertIn("'original'", lines[0])

    def test_known_or_missing_literal_values_stay_quiet(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            Sam3Args(sam3_inpainting_fill="Original", sam3_mode="inpaint", sam3_cn_resize_mode="just resize")
            Sam3Args(sam3_inpainting_fill=None)   # 값이 없음 — 모르는 값이 아니다
            Sam3Args()
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
