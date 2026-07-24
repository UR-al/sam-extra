from __future__ import annotations

import unittest

from sam3ext.args import Sam3Args


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


if __name__ == "__main__":
    unittest.main()
