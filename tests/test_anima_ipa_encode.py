"""참조 이미지 → SigLIP2 입력/토큰. 진짜 인코더 없이 기하와 배선만 본다."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.anima_ipa.encode import (  # noqa: E402
    ReferenceEncoder,
    letterbox,
    to_pixel_values,
)


class LetterboxTests(unittest.TestCase):
    def test_result_is_square_at_the_requested_size(self):
        out = letterbox(Image.new("RGB", (100, 50), "red"), 512)
        self.assertEqual(out.size, (512, 512))

    def test_aspect_ratio_is_kept_and_the_image_is_centred(self):
        out = letterbox(Image.new("RGB", (100, 50), "red"), 100)
        # 가로 100 세로 50 → 100x50 으로 그려지고 위아래 25px 이 검정
        self.assertEqual(out.getpixel((50, 50)), (255, 0, 0))
        self.assertEqual(out.getpixel((50, 5)), (0, 0, 0))
        self.assertEqual(out.getpixel((50, 95)), (0, 0, 0))

    def test_rgba_input_is_flattened(self):
        out = letterbox(Image.new("RGBA", (10, 10), (255, 0, 0, 0)), 10)
        self.assertEqual(out.mode, "RGB")

    def test_a_very_thin_image_still_produces_at_least_one_pixel(self):
        out = letterbox(Image.new("RGB", (1, 400), "red"), 64)
        self.assertEqual(out.size, (64, 64))


class PixelValueTests(unittest.TestCase):
    def test_shape_and_normalisation(self):
        # 세로가 짧은 입력이라 위아래에 검정 패딩이 생긴다.
        values = to_pixel_values(Image.new("RGB", (64, 32), (255, 255, 255)), 64)
        self.assertEqual(tuple(values.shape), (1, 3, 64, 64))
        self.assertEqual(values.dtype, torch.float32)
        self.assertAlmostEqual(float(values.max()), 1.0, places=5)    # 흰색 → +1
        self.assertAlmostEqual(float(values.min()), -1.0, places=5)   # 패딩 검정 → -1

    def test_a_square_input_has_no_padding_at_all(self):
        values = to_pixel_values(Image.new("RGB", (32, 32), (255, 255, 255)), 64)
        self.assertAlmostEqual(float(values.min()), 1.0, places=5)


class _FakeVision:
    """SiglipVisionModel 대역. 부른 인자와 장치 이동을 기록한다."""

    def __init__(self, tokens=8, dim=768, layers=3):
        self.tokens, self.dim, self.layers = tokens, dim, layers
        self.calls = []
        self.devices = []

    def to(self, device):
        self.devices.append(str(device))
        return self

    def __call__(self, pixel_values, **kwargs):
        self.calls.append(kwargs)
        last = torch.ones(1, self.tokens, self.dim)
        hidden = [
            torch.full((1, self.tokens, self.dim), float(i)) for i in range(self.layers)
        ]
        return type("Out", (), {"last_hidden_state": last, "hidden_states": hidden})()


class EncoderTests(unittest.TestCase):
    def test_last_layer_is_used_by_default(self):
        vision = _FakeVision()
        tokens = ReferenceEncoder(vision).encode(Image.new("RGB", (64, 64)), size=64)
        self.assertEqual(tuple(tokens.shape), (1, 8, 768))
        self.assertTrue(torch.allclose(tokens, torch.ones_like(tokens)))
        self.assertTrue(vision.calls[0]["interpolate_pos_encoding"])

    def test_an_explicit_layer_asks_for_hidden_states(self):
        vision = _FakeVision()
        tokens = ReferenceEncoder(vision).encode(
            Image.new("RGB", (64, 64)), size=64, layer=1
        )
        self.assertTrue(vision.calls[0]["output_hidden_states"])
        self.assertTrue(torch.allclose(tokens, torch.ones_like(tokens)))

    def test_optional_modules_run_in_order(self):
        vision = _FakeVision()
        order = []

        def norm(x):
            order.append("norm")
            return x + 1

        def compressor(x):
            order.append("compressor")
            return x[:, :2, :]

        def self_attn(x):
            order.append("self_attn")
            return x * 2

        encoder = ReferenceEncoder(
            vision, siglip_norm=norm, compressor=compressor, self_attn=self_attn
        )
        tokens = encoder.encode(Image.new("RGB", (64, 64)), size=64, layer=0)
        self.assertEqual(order, ["norm", "compressor", "self_attn"])
        self.assertEqual(tuple(tokens.shape), (1, 2, 768))

    def test_the_encoder_goes_back_to_the_cpu_after_encoding(self):
        """GPU 에 남겨 두면 Forge 가 Anima 를 부분 로드로 밀어낸다(TIPO 와 같은 이유)."""
        vision = _FakeVision()
        ReferenceEncoder(vision).encode(
            Image.new("RGB", (64, 64)), size=64, device="cpu"
        )
        self.assertEqual(vision.devices[-1], "cpu")

    def test_the_encoder_comes_home_even_when_the_forward_blows_up(self):
        class _Boom(_FakeVision):
            def __call__(self, pixel_values, **kwargs):
                raise RuntimeError("boom")

        vision = _Boom()
        with self.assertRaises(RuntimeError):
            ReferenceEncoder(vision).encode(Image.new("RGB", (64, 64)), size=64)
        self.assertEqual(vision.devices[-1], "cpu")


if __name__ == "__main__":
    unittest.main()
