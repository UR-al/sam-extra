"""Anima VAE DeGrid — 잔차 산술(모드·부호·강도·마지막 clamp), 타일(ComfyUI tiled_scale 과 같음), 16 배수가 아닌 타일의
반사 패딩, OOM 재시도, PIL 변환, 이미지를 내는 NAFNet 거절, NAFNet 파일 찾기, spandrel 로더, 런타임(장치·Forge 메모리 관리,
생성 탭(inference_mode)에서 처음 불러 장치로 옮기는 경로). GPU 없이(CPU) 돈다.

대조 오라클
- 잔차 적용: ComfyUI-NAFNet-Residual(DraconicDragon, Apache-2.0 — LICENSE 에 저작권자 이름 없음, NOTICE 없음)
  commit e15460d3724c70d428e333518b58eb7ba8903d76 ``patch.py`` 48-62줄 ``_apply_residual_mode`` 를 아래에 그대로 옮겼다.
  노드(``nafnet_node.py`` 132-149줄)는 같은 식 뒤에 ``torch.clamp(result, 0, 1)``.
- 타일: Forge ``backend/patcher/vae.py`` 의 ``tiled_scale_multidim``(ComfyUI v0.3.64 ``comfy/utils.py`` 를 옮긴 것 — 노드 팩이
  부르는 ``comfy.utils.tiled_scale`` 과 같은 함수)을 Forge 가 확장 옆에 있을 때만 AST 로 꺼내 쓴다(CI 에서는 건너뜀).
- Forge 의 ``torch.load``·``safetensors.torch.load_file`` 감싸개(실패하면 str 파일 경로를 ``.corrupted`` 로 바꿈):
  ``modules_forge/patch_basic.py`` 의 ``build_loaded`` 를 Forge 가 옆에 있으면 AST 로 꺼내 쓰고, 없으면(CI) 같은 동작의 대역.
  실제 ``models/ESRGAN`` 목록(읽기만)은 SAM3_RUN_FORGE_INTEGRATION_TESTS=1 일 때만.
- 실제 가중치(qwenVAEDegridNafnet_v11 — 잔차 검사는 있으면 Anzhc 파인튜닝 NAFNet-QwenVAE-DeGrid 로도)는
  SAM3_RUN_FORGE_INTEGRATION_TESTS=1 이고 파일이 있을 때만(CPU, 약 30 초). 실제 Anima 이미지 조각은 거기에 더해
  SAM3_DEGRID_ANIMA_IMAGES(PNG 파일·폴더, os.pathsep 구분)를 줄 때만 — 생성 이미지는 저장소에 넣지 않는다.
"""
from __future__ import annotations

import ast
import contextlib
import functools
import io
import itertools
import logging
import math
import os
import sys
import tempfile
import time
import types
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import vae_degrid as vd  # noqa: E402
from sam3ext import vae_degrid_models as vdm  # noqa: E402
from sam3ext import vae_degrid_runtime as vdr  # noqa: E402

FORGE_ROOT = ROOT.parents[1]
FORGE_VAE_PY = FORGE_ROOT / "backend" / "patcher" / "vae.py"
REAL_MODEL = Path(os.environ.get(
    "SAM3_DEGRID_MODEL", str(FORGE_ROOT / "models" / "ESRGAN" / "qwenVAEDegridNafnet_v11.safetensors")))
# 다른 DeGrid 파인튜닝(Anzhc 의 NAFNet-QwenVAE-DeGrid) — 있으면 잔차 검사 실제 가중치 테스트를 이것으로도 돈다.
REAL_MODEL_ALT = Path(os.environ.get(
    "SAM3_DEGRID_MODEL_ALT", str(FORGE_ROOT / "models" / "ESRGAN" / "NAFNet-QwenVAE-DeGrid.safetensors")))
# 모델마다 다르게 재 둔 값(저대비 1px 체커 폭주 여부)을 확인할 때 쓰는 기본 파일 이름.
REAL_MODEL_V11_STEM = "qwenVAEDegridNafnet_v11"
REAL_MODEL_ANZHC_STEM = "NAFNet-QwenVAE-DeGrid"

try:
    import spandrel  # noqa: F401
    HAVE_SPANDREL = True
except Exception:  # CI 는 spandrel 이 없다(torchvision 을 끌고 와서 requirements-dev 에 넣지 않음)
    HAVE_SPANDREL = False


# origin: DraconicDragon/ComfyUI-NAFNet-Residual@e15460d patch.py 48-62 (Apache-2.0) — 그대로
def _apply_residual_mode(image, delta, mode: str):
    if mode == "full":
        return image + delta

    if mode == "dark_pixels":
        # Positive residuals correct dark pixels.
        return image + delta.clamp_min(0)

    if mode == "bright_pixels":
        # Negative residuals correct bright pixels.
        return image + delta.clamp_max(0)

    # Disabled: return the raw model output exactly as Spandrel normally
    # would before its own output clamp.
    return delta


_PACK_MODE = {vd.MODE_FULL: "full", vd.MODE_DARK: "dark_pixels", vd.MODE_BRIGHT: "bright_pixels"}


def _forge_tiled_scale():
    """Forge backend/patcher/vae.py 의 tiled_scale(ComfyUI v0.3.64) — 파일을 import 하지 않고(backend 가 CUDA 를 조회)
    함수 정의만 AST 로 꺼낸다."""
    tree = ast.parse(FORGE_VAE_PY.read_text(encoding="utf-8"))
    namespace = {"itertools": itertools, "torch": torch, "math": math}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("tiled_scale_multidim", "tiled_scale")]
    exec(compile(ast.Module(nodes, []), str(FORGE_VAE_PY), "exec"), namespace)
    return namespace["tiled_scale"]


class ResidualModeTests(unittest.TestCase):
    def setUp(self):
        # 픽셀마다 부호가 다른 잔차: -0.2, -0.01, 0, +0.01, +0.3
        self.image = torch.full((1, 3, 1, 5), 0.5)
        self.delta = torch.tensor([-0.2, -0.01, 0.0, 0.01, 0.3]).view(1, 1, 1, 5).expand(1, 3, 1, 5).clone()

    def test_modes_pick_the_sign(self):
        full = vd.apply_residual(self.image, self.delta, vd.MODE_FULL)
        dark = vd.apply_residual(self.image, self.delta, vd.MODE_DARK)
        bright = vd.apply_residual(self.image, self.delta, vd.MODE_BRIGHT)
        torch.testing.assert_close(full[0, 0, 0], torch.tensor([0.3, 0.49, 0.5, 0.51, 0.8]))
        # Dark Pixels Mainly: 양의 잔차(어두운 점을 밝힘)만 — 음수 자리는 그대로
        torch.testing.assert_close(dark[0, 0, 0], torch.tensor([0.5, 0.5, 0.5, 0.51, 0.8]))
        # Bright Pixels Mainly: 음의 잔차(밝은 점을 어둡게)만
        torch.testing.assert_close(bright[0, 0, 0], torch.tensor([0.3, 0.49, 0.5, 0.5, 0.5]))
        # 두 반쪽을 더하면 전체
        torch.testing.assert_close(dark + bright - self.image, full)

    def test_matches_node_pack_bit_for_bit(self):
        g = torch.Generator().manual_seed(0)
        image = torch.rand((1, 3, 17, 23), generator=g)
        delta = (torch.rand((1, 3, 17, 23), generator=g) - 0.5) * 0.4
        for mode, pack in _PACK_MODE.items():
            with self.subTest(mode=mode):
                ours = vd.finalize(vd.apply_residual(image, delta, mode, 1.0))
                theirs = torch.clamp(_apply_residual_mode(image, delta, pack), min=0.0, max=1.0)
                self.assertTrue(torch.equal(ours, theirs))

    def test_strength_scales_the_residual_before_the_final_clamp(self):
        for strength, expected in ((0.0, 0.5), (0.5, 0.35), (1.0, 0.2), (1.5, 0.05)):
            out = vd.apply_residual(torch.tensor([0.5]), torch.tensor([-0.3]), vd.MODE_FULL, strength)
            torch.testing.assert_close(out, torch.tensor([expected]))
        # 강도는 자른 결과를 섞는 것이 아니다: 0.9 + 0.5·0.3 = 1.05 → 1.0.
        # (자른 뒤 섞으면 0.9 + 0.5·(1.0 − 0.9) = 0.95 가 된다)
        out = vd.finalize(vd.apply_residual(torch.tensor([0.9]), torch.tensor([0.3]), vd.MODE_FULL, 0.5))
        torch.testing.assert_close(out, torch.tensor([1.0]))

    def test_only_the_final_image_is_clamped(self):
        # 잔차가 [0,1] 밖이어도(음수) 자르지 않는다 — ComfyUI 코어 경로(spandrel __call__ 의 clamp_(0,1))와 다른 점
        image = torch.tensor([0.5, 0.95, 0.02])
        delta = torch.tensor([-0.2, 0.1, -0.05])
        raw = vd.apply_residual(image, delta, vd.MODE_FULL)
        torch.testing.assert_close(raw, torch.tensor([0.3, 1.05, -0.03]))   # 중간 합은 자르지 않음
        torch.testing.assert_close(vd.finalize(raw), torch.tensor([0.3, 1.0, 0.0]))
        # 잔차를 먼저 [0,1] 로 자른 코어 경로는 Dark Pixels Mainly 와 같다(모델 카드)
        core = torch.clamp(image + delta.clamp(0, 1), 0, 1)
        torch.testing.assert_close(core, vd.finalize(vd.apply_residual(image, delta, vd.MODE_DARK)))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            vd.apply_residual(self.image, self.delta, "sideways")

    def test_normalize_mode_accepts_keys_labels_and_pack_names(self):
        cases = {
            "full": vd.MODE_FULL, "Full": vd.MODE_FULL, "Full (전체)": vd.MODE_FULL,
            "dark": vd.MODE_DARK, "Dark Pixels Mainly": vd.MODE_DARK, "dark_pixels": vd.MODE_DARK,
            "Dark Pixels Mainly (어두운 점 위주)": vd.MODE_DARK,
            "bright": vd.MODE_BRIGHT, "bright_pixels": vd.MODE_BRIGHT, "BRIGHT PIXELS MAINLY": vd.MODE_BRIGHT,
        }
        for value, key in cases.items():
            with self.subTest(value=value):
                self.assertEqual(vd.normalize_mode(value), key)
        for value in (None, "", "disabled", "sideways"):
            self.assertIsNone(vd.normalize_mode(value))

    def test_strength_and_tile_coercion(self):
        self.assertEqual(vd.coerce_strength("0.75"), 0.75)
        self.assertEqual(vd.coerce_strength(9), 1.5)
        self.assertEqual(vd.coerce_strength(-1), 0.0)
        self.assertEqual(vd.coerce_strength("x"), 1.0)
        self.assertEqual(vd.coerce_strength(float("nan")), 1.0)
        self.assertEqual(vd.coerce_tile(0), 0)
        self.assertEqual(vd.coerce_tile(-5), 0)
        self.assertEqual(vd.coerce_tile(64), 128)
        self.assertEqual(vd.coerce_tile("512"), 512)
        self.assertEqual(vd.coerce_tile(99999), 4096)
        self.assertEqual(vd.coerce_tile(None), 512)


class ResidualSanityTests(unittest.TestCase):
    """잔차가 아니라 이미지를 내는 NAFNet(예: SIDD width 32 노이즈 제거 — 키·구성이 DeGrid 와 같아 목록에 나온다)을 거른다."""

    def setUp(self):
        self.g = torch.Generator().manual_seed(7)
        self.x = torch.rand((1, 3, 64, 64), generator=self.g)

    def test_degrid_like_residual_is_accepted(self):
        # 실제 v1.1 은 |평균| 0.1~0.6/255, 최대 수십/255
        raw = (torch.rand(self.x.shape, generator=self.g) - 0.5) * (4 / 255)
        raw[..., 10, 10] = 40 / 255
        self.assertFalse(vd.output_looks_like_image(self.x, raw))

    def test_image_output_is_refused(self):
        noise = (torch.rand(self.x.shape, generator=self.g) - 0.5) * 0.02
        self.assertTrue(vd.output_looks_like_image(self.x, self.x + noise))

    def test_dark_image_output_is_refused_by_correlation(self):
        dark = self.x * 0.06                      # 평균 약 7.6/255 — |평균| 기준(25/255) 아래
        self.assertTrue(vd.output_looks_like_image(dark, dark * 0.98 + 0.001))

    def test_near_black_and_flat_images_are_not_misjudged(self):
        black = torch.full((1, 3, 32, 32), 1 / 255)
        self.assertFalse(vd.output_looks_like_image(black, torch.full_like(black, 1 / 255)))
        flat = torch.full((1, 3, 32, 32), 0.5)     # 분산 0 — 상관을 정할 수 없으면 잔차로 본다
        self.assertFalse(vd.output_looks_like_image(flat, torch.full_like(flat, 5 / 255)))

    @staticmethod
    def _screentone(size=64):
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        dots = ((yy % 4) < 2) & ((xx % 4) < 2)                      # 4px 망점(2x2 검은 점) — 화면 전체
        return (~dots).float().expand(1, 3, size, size).contiguous()

    @staticmethod
    def _checker(size=64):
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        return ((yy + xx) % 2).float().expand(1, 3, size, size).contiguous()

    def test_large_residual_that_opposes_the_input_is_accepted(self):
        # 실제 DeGrid 가 잔 스크린톤·1px 체커에 내는 잔차: 크지만(25/255 초과) 입력과 반대로 움직여 무늬를 누른다
        # (CPU 실측 v1.1: 4px 스크린톤 |평균| 31.3/255 r -0.92, 1px 체커 56.7/255 r -0.47)
        tone = self._screentone()
        noise = torch.randn(tone.shape, generator=self.g)
        raw = -0.33 * (tone - tone.mean()) + 0.06 * noise
        check = vd.check_residual(tone, raw)
        self.assertGreater(check.mean_abs, vd.IMAGE_LIKE_ABS_MEAN)
        self.assertLess(check.correlation, -0.8)
        self.assertFalse(check.looks_like_image)
        self.assertFalse(vd.output_looks_like_image(tone, raw))

        checker = self._checker()
        raw = 0.08 - 0.2 * (checker - 0.5) + 0.19 * torch.randn(checker.shape, generator=self.g)
        check = vd.check_residual(checker, raw)
        self.assertGreater(check.mean_abs, vd.IMAGE_LIKE_ABS_MEAN)
        self.assertLess(check.correlation, -0.3)
        self.assertFalse(vd.output_looks_like_image(checker, raw))

    def test_large_uncorrelated_residual_is_accepted(self):
        raw = (torch.rand(self.x.shape, generator=self.g) - 0.5) * 0.5   # |평균| 약 32/255, 상관 약 0
        self.assertGreater(float(raw.abs().mean()), vd.IMAGE_LIKE_ABS_MEAN)
        self.assertFalse(vd.output_looks_like_image(self.x, raw))

    def test_image_outputs_on_screentone_and_checker_are_refused(self):
        for name, pattern in (("screentone", self._screentone()), ("checker", self._checker())):
            with self.subTest(name=name):
                self.assertTrue(vd.output_looks_like_image(pattern, pattern.clone()))              # 복원 이미지 그대로
                softened = 0.7 * pattern + 0.3 * float(pattern.mean())                          # 대비만 줄인 이미지
                self.assertTrue(vd.output_looks_like_image(pattern, softened))

    def test_large_output_that_loosely_follows_the_input_is_refused(self):
        # r 0.5~0.9 — 작은 출력이면 잔차로 보지만(날카롭게 하는 잔차는 r 이 양수일 수 있다), 크면 이미지다
        out = self.x + 0.45 * torch.randn(self.x.shape, generator=self.g)
        check = vd.check_residual(self.x, out)
        self.assertGreater(check.mean_abs, vd.IMAGE_LIKE_ABS_MEAN)
        self.assertTrue(0.5 < check.correlation < 0.9, check.correlation)
        self.assertTrue(check.looks_like_image)
        small = 0.04 * (self.x - 0.5) + 0.012 * torch.randn(self.x.shape, generator=self.g)  # 비슷한 상관, |평균| 약 3/255
        check = vd.check_residual(self.x, small)
        self.assertTrue(vd.IMAGE_LIKE_MIN_ABS_MEAN < check.mean_abs < vd.IMAGE_LIKE_ABS_MEAN, check.mean_abs)
        self.assertTrue(0.5 < check.correlation < 0.9, check.correlation)
        self.assertFalse(check.looks_like_image)

    def test_single_colour_input_with_a_large_output_is_refused(self):
        # 한 색 입력은 상관을 정할 수 없다 — 이미지를 내는 모델이면 출력이 그 색 그대로다
        flat = torch.full((1, 3, 32, 32), 128 / 255)
        check = vd.check_residual(flat, flat.clone())
        self.assertTrue(check.input_flat)
        self.assertIsNone(check.correlation)
        self.assertTrue(check.looks_like_image)
        self.assertFalse(vd.output_looks_like_image(flat, torch.full_like(flat, 5 / 255)))
        red = torch.zeros((1, 3, 32, 32))
        red[:, 0] = 1.0                            # 채널마다 값이 달라 상관을 정할 수 있다
        self.assertTrue(vd.output_looks_like_image(red, red.clone()))

    def test_constant_output_does_not_follow_the_input(self):
        self.assertFalse(vd.output_looks_like_image(self.x, torch.full_like(self.x, 40 / 255)))
        check = vd.check_residual(self.x, torch.full_like(self.x, 40 / 255))
        self.assertLess(check.dc_ratio, vd.IMAGE_LIKE_DC_RATIO)       # 40/255 ÷ 입력 평균 약 0.5 = 0.31
        self.assertFalse(check.follows_input_mean)

    # ── 잔 결이 입력 분산의 대부분일 때: 이미지 모델의 상관은 낮지만 밝기(DC)는 입력 그대로 ──

    def test_image_model_on_fine_grain_is_refused_by_brightness(self):
        # 회색 바탕 200±4 입자 — 흐림·median 은 입자를 지워 r 이 0.3 아래로 떨어진다(dc59409 규칙은 적용해 모든 픽셀이 255)
        grain = _grain_image_tensor(128)
        for name, model in (("gaussian s2", _BlurImageModel(2.0)), ("median3", _Median3ImageModel())):
            with self.subTest(model=name):
                check = vd.check_residual(grain, model(grain))
                self.assertGreater(check.mean_abs, vd.IMAGE_LIKE_ABS_MEAN)
                self.assertLess(check.correlation, vd.IMAGE_LIKE_LARGE_CORRELATION)   # 상관 규칙만으로는 못 거른다
                self.assertAlmostEqual(check.dc_ratio, 1.0, places=2)
                self.assertAlmostEqual(check.signed_mean, check.mean_abs, places=6)  # 이미지 — 음수가 없다
                self.assertTrue(check.follows_input_mean)
                self.assertTrue(check.looks_like_image)
                self.assertTrue(check.blew_up)        # |평균| 200/255 도 폭주 문턱 위 — 런타임은 이미지 판정을 먼저 알린다

    def test_image_model_on_full_screentone_is_refused_by_brightness(self):
        tone = self._screentone(128)
        for name, model in (("gaussian s2", _BlurImageModel(2.0)), ("median3", _Median3ImageModel())):
            with self.subTest(model=name):
                check = vd.check_residual(tone, model(tone))
                self.assertLess(check.correlation, vd.IMAGE_LIKE_LARGE_CORRELATION)
                self.assertTrue(check.follows_input_mean)
                self.assertTrue(check.looks_like_image)

    def test_output_of_the_input_average_colour_is_refused(self):
        # 한 값뿐인 출력(상관을 정할 수 없음)이라도 그 값이 입력 평균 밝기면 이미지(아주 강한 흐림)
        self.assertAlmostEqual(float(self.x.mean()), 0.5, places=2)
        out = torch.full_like(self.x, 0.5)            # 정확히 한 값 — 상관의 분모가 0
        check = vd.check_residual(self.x, out)
        self.assertIsNone(check.correlation)
        self.assertFalse(check.input_flat)
        self.assertAlmostEqual(check.dc_ratio, 1.0, places=1)
        self.assertTrue(check.looks_like_image)
        # 채널마다 그 채널 평균이면 상관은 정해지지만 작다(채널 사이 차이뿐) — 밝기로 거른다
        per_channel = self.x.mean(dim=(2, 3), keepdim=True).expand_as(self.x).clone()
        check = vd.check_residual(self.x, per_channel)
        self.assertLess(check.correlation, vd.IMAGE_LIKE_LARGE_CORRELATION)
        self.assertAlmostEqual(check.dc_ratio, 1.0, places=5)
        self.assertTrue(check.looks_like_image)

    def test_residual_that_swings_both_ways_is_not_judged_by_brightness(self):
        # 실제 1px 가로줄 폭주(v1.1: |평균| 362/255, 평균 +89/255, 투영 비 +0.70, r -0.07)처럼 평균은 크고 입력 밝기 쪽이지만
        # 양쪽으로 크게 흔들리는 잔차는 이미지로 보지 않는다(부호 쏠림 0.25) — 폭주로만 건너뛴다.
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        stripes = (yy % 2).float().expand(1, 3, 64, 64).contiguous()
        swing = torch.where(xx % 2 == 0, 1.0, -1.0).expand(1, 3, 64, 64)
        raw = 89 / 255 + (362 / 255) * swing
        check = vd.check_residual(stripes, raw)
        self.assertAlmostEqual(check.correlation, 0.0, places=5)
        self.assertGreater(check.signed_mean, vd.IMAGE_LIKE_DC_MEAN)
        self.assertGreater(check.dc_ratio, vd.IMAGE_LIKE_DC_RATIO)
        self.assertLess(abs(check.signed_mean), vd.IMAGE_LIKE_DC_SIGN * check.mean_abs)
        self.assertFalse(check.follows_input_mean)
        self.assertFalse(check.looks_like_image)
        self.assertTrue(check.blew_up)

    def test_dark_checker_residual_is_not_an_image(self):
        # Anzhc 가 어두운 1px 체커(0/40)에서 내는 잔차 모양 — |평균| 64.6, 부호 쏠림 0.54, 투영 비 1.7, 입력과 반대 방향.
        # 부호 쏠림 문턱이 0.5 이면 '이미지'로 잘못 거절됐다(0.8 로 올림).
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        dark = ((yy + xx) % 2 == 1).float().expand(1, 3, 64, 64).contiguous() * (40 / 255)
        raw = torch.where(dark > 0, -30.0 / 255, 99.2 / 255)
        check = vd.check_residual(dark, raw)
        self.assertGreater(check.signed_mean, vd.IMAGE_LIKE_DC_MEAN)
        self.assertGreater(check.dc_ratio, vd.IMAGE_LIKE_DC_RATIO)
        self.assertGreater(abs(check.signed_mean), 0.5 * check.mean_abs)
        self.assertFalse(check.looks_like_image)
        self.assertFalse(check.blew_up)

    def test_blow_up_ceiling(self):
        sign = torch.where(torch.rand(self.x.shape, generator=self.g) < 0.5, 1.0, -1.0)
        below = vd.check_residual(self.x, sign * (99 / 255))
        above = vd.check_residual(self.x, sign * (101 / 255))
        self.assertFalse(below.blew_up)
        self.assertTrue(above.blew_up)
        self.assertFalse(above.looks_like_image)                 # 폭주는 이미지 판정과 따로
        # 실측 예시: 1px 체커 잔차 62.8/255 는 적용, 118/138 1px 체커의 폭주 183.1/255 는 건너뜀(깨끗한 틈은 아님 — 어림 문턱)
        self.assertLess(62.8 / 255, vd.RESIDUAL_BLOWUP_ABS_MEAN)
        self.assertLess(vd.RESIDUAL_BLOWUP_ABS_MEAN, 183.1 / 255)
        self.assertFalse(vd.check_residual(self.x, sign * (1 / 255)).blew_up)   # 2/255 이하는 보지 않음


def _grain_image_tensor(size=128, level=200.0, sigma=4.0, seed=3):
    """회색 바탕 200±4 입자(8비트로 반올림, 세 채널 같음) — (1, 3, H, W)."""
    g = np.random.default_rng(seed)
    gray = np.clip(np.round(level + g.normal(0, sigma, (size, size))), 0, 255) / 255.0
    return torch.from_numpy(gray).float().expand(1, 3, size, size).contiguous()


def _grain_image(size=128):
    arr = (_grain_image_tensor(size)[0, 0].numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


class _BlurImageModel(torch.nn.Module):
    """이미지를 내는 '노이즈 제거' 대역 — 가우시안 흐림(σ, 반사 패딩)."""

    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, x):
        r = int(math.ceil(3 * self.sigma))
        k = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype) ** 2 / (2 * self.sigma ** 2))
        k = k / k.sum()
        c = x.shape[1]
        y = torch.nn.functional.pad(x, (r, r, r, r), mode="reflect")
        y = torch.nn.functional.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
        return torch.nn.functional.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


class _Median3ImageModel(torch.nn.Module):
    """이미지를 내는 '노이즈 제거' 대역 — 3x3 median(반사 패딩)."""

    def forward(self, x):
        y = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        return y.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*x.shape, 9).median(-1).values


class _BlowUpModel(torch.nn.Module):
    """잔차 폭주 대역 — 입력과 상관없이 ±2(= ±510/255)로 흔들리는 잔차(학습에 없던 무늬에서 실제 DeGrid 가 내는 것처럼)."""

    def forward(self, x):
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        return torch.where((yy + xx) % 2 == 0, 2.0, -2.0).to(x).expand_as(x).clone()


def _pointwise(t):
    return t * 0.3 - 0.1


def _box3(t):
    return torch.nn.functional.avg_pool2d(t, 3, 1, 1, count_include_pad=True)


class TilingTests(unittest.TestCase):
    def test_tile_starts_follow_comfyui(self):
        # range(0, size - overlap, tile - overlap)
        self.assertEqual(vd.tile_starts(512, 512, 32), [0])
        self.assertEqual(vd.tile_starts(768, 512, 32), [0, 480])
        self.assertEqual(vd.tile_starts(1216, 512, 32), [0, 480, 960])
        self.assertEqual(vd.tile_starts(1856, 512, 32), [0, 480, 960, 1440])

    def test_pointwise_function_tiles_exactly(self):
        g = torch.Generator().manual_seed(1)
        x = torch.rand((1, 3, 300, 700), generator=g)
        whole = vd.tiled_residual(x, _pointwise, tile=0)
        tiled = vd.tiled_residual(x, _pointwise, tile=128, overlap=32)
        torch.testing.assert_close(tiled, whole, rtol=0, atol=1e-6)

    def test_seams_of_a_local_filter_are_feathered(self):
        # 3x3 필터는 타일 가장자리 한 줄이 틀린다(0 패딩). feather 가중치 1/32 로 섞여 최대 오차가 1% 남짓.
        g = torch.Generator().manual_seed(2)
        x = torch.rand((1, 3, 256, 384), generator=g)
        whole = vd.tiled_residual(x, _box3, tile=0)
        tiled = vd.tiled_residual(x, _box3, tile=128, overlap=32)
        err = (tiled - whole).abs()
        self.assertLess(float(err.max()), 0.02)
        self.assertLess(float(err.mean()), 1e-3)

    def test_small_image_or_no_tiling_calls_once(self):
        calls = []

        def fn(t):
            calls.append(tuple(t.shape))
            return t

        x = torch.rand(1, 3, 100, 90)
        vd.tiled_residual(x, fn, tile=512)
        vd.tiled_residual(x, fn, tile=0)
        self.assertEqual(calls, [(1, 3, 100, 90)] * 2)

    @unittest.skipUnless(FORGE_VAE_PY.is_file(), "Forge 의 backend/patcher/vae.py 가 확장 옆에 없음")
    def test_matches_forge_comfyui_tiled_scale_bit_for_bit(self):
        tiled_scale = _forge_tiled_scale()

        def nonlocal_fn(t):   # 타일 전체 평균을 쓰는 함수(NAFNet SCA 처럼) — 같은 타일로 불러야만 같다
            return t * 0.5 + t.mean() - 0.25

        g = torch.Generator().manual_seed(3)
        for shape, tile in (((1, 3, 300, 700), 128), ((1, 3, 1216 // 4, 1856 // 4), 160), ((1, 3, 100, 90), 512)):
            with self.subTest(shape=shape, tile=tile):
                x = torch.rand(shape, generator=g)
                ours = vd.tiled_residual(x, nonlocal_fn, tile=tile, overlap=32)
                theirs = tiled_scale(x, nonlocal_fn, tile_x=tile, tile_y=tile, overlap=32, upscale_amount=1,
                                     out_channels=3, output_device="cpu")
                self.assertTrue(torch.equal(ours, theirs))


class _ZeroPaddingNafnetLike(torch.nn.Module):
    """spandrel NAFNet 처럼 안에서 ``padder_size`` 배수로 0 을 채우고(``check_image_size``) 끝에서 자르는 국소 잔차 모델.
    잔차 = 3x3 평균(가장자리 복제) − 입력 — 평평한 면에서는 0 이고, 이미지와 0 채움 사이 경계에서만 크다."""

    padder_size = 16

    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape[-2:]))
        h, w = x.shape[-2:]
        x = torch.nn.functional.pad(x, (0, (-w) % self.padder_size, 0, (-h) % self.padder_size))
        mean = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
        return (mean - x)[:, :, :h, :w]


class TilePaddingTests(unittest.TestCase):
    """16 배수가 아닌 크기 — spandrel NAFNet 은 안에서 0 으로 채워 오른쪽·아래 가장자리에 큰 잔차가 생긴다(실제 v1.1, 250x190:
    오른쪽 아래 56/255). 타일마다 반사 패딩으로 배수를 맞추고 자른다(저자 infer.py 와 같은 반사). 16 배수면 그대로 부른다."""

    def test_pad_to_multiple_reflects_right_and_bottom(self):
        x = torch.rand(1, 3, 190, 250)
        padded = vd.pad_to_multiple(x, 16)
        self.assertEqual(tuple(padded.shape), (1, 3, 192, 256))
        self.assertTrue(torch.equal(padded[..., :190, :250], x))
        # 반사: 가장자리 줄을 축으로 접는다(가장자리 줄은 되풀이하지 않음)
        self.assertTrue(torch.equal(padded[..., 190:192, :250], x[..., 187:189, :].flip(-2)))
        self.assertTrue(torch.equal(padded[..., :190, 250:256], x[..., 243:249].flip(-1)))

    def test_multiples_are_untouched_and_tiny_inputs_replicate(self):
        x = torch.rand(1, 3, 192, 256)
        self.assertIs(vd.pad_to_multiple(x, 16), x)
        self.assertIs(vd.pad_to_multiple(x, 1), x)
        tiny = torch.rand(1, 3, 5, 7)             # 채울 칸(11·9)이 길이 이상이면 반사할 수 없다 → 가장자리 복제
        padded = vd.pad_to_multiple(tiny, 16)
        self.assertEqual(tuple(padded.shape), (1, 3, 16, 16))
        self.assertTrue(torch.equal(padded[..., :5, :7], tiny))
        self.assertTrue(torch.equal(padded[..., 15, 6], tiny[..., 4, 6]))

    def test_odd_size_gets_no_edge_residual(self):
        model = _ZeroPaddingNafnetLike()
        flat = torch.full((1, 3, 190, 250), 0.5)
        self.assertGreater(float(model(flat).abs().max()), 0.1)    # 0 채움 경계(고치기 전 동작)
        fn, _ = vdr.make_residual_fn(model, torch.device("cpu"), False)
        self.assertLess(float(fn(flat).abs().max()), 1e-6)
        self.assertEqual(model.shapes[-1], (192, 256))

    def test_multiple_of_16_calls_the_model_unchanged(self):
        # 노드 팩 동등성: 16 배수면 모델을 그대로 부른다
        model = _ZeroPaddingNafnetLike()
        x = torch.rand(1, 3, 192, 256)
        fn, _ = vdr.make_residual_fn(model, torch.device("cpu"), False)
        self.assertTrue(torch.equal(fn(x), model(x)))
        self.assertEqual(model.shapes, [(192, 256), (192, 256)])

    def test_standard_anima_sizes_tile_into_multiples_of_16(self):
        # 표준 Anima 크기(와 1.5 배 hires)는 타일 512·256·128 의 모든 조각이 16 배수 — 패딩이 끼지 않아 노드와 같다
        for height, width in ((1856, 1216), (1216, 832), (1024, 1024), (1536, 1536), (2784, 1824)):
            for tile in (512, 256, 128):
                sizes = {min(tile, height - top) for top in vd.tile_starts(height, tile, vd.TILE_OVERLAP)}
                sizes |= {min(tile, width - left) for left in vd.tile_starts(width, tile, vd.TILE_OVERLAP)}
                with self.subTest(size=(width, height), tile=tile):
                    self.assertTrue(all(size % 16 == 0 for size in sizes), sizes)

    def test_runtime_keeps_a_flat_odd_sized_image_flat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            path.write_bytes(b"x")
            rt = vdr.DegridRuntime(loader=lambda p: _ZeroPaddingNafnetLike(), forge_memory=lambda: None,
                                   logger=lambda m: None)
            image = Image.new("RGB", (250, 190), (128, 128, 128))
            for tile in (0, 128):
                with self.subTest(tile=tile):
                    out = rt.run(image, vdm.ModelEntry("m", str(path)), tile=tile, device="cpu")
                    self.assertEqual(out.image.tobytes(), image.tobytes())


class _Oom(RuntimeError):
    pass


class OomRetryTests(unittest.TestCase):
    def _fn(self, limit):
        def fn(t):
            if t.shape[-1] * t.shape[-2] > limit:
                raise _Oom("CUDA out of memory. Tried to allocate 2.00 GiB")
            return t * 0.1
        return fn

    def test_halves_the_tile_until_it_fits(self):
        x = torch.rand(1, 3, 300, 300)
        retries = []
        out, used = vd.tiled_residual_with_oom_retry(
            x, self._fn(128 * 128), tile=512, on_retry=lambda t, e: retries.append(t))
        self.assertEqual(used, 128)
        self.assertEqual(retries, [256, 128])
        torch.testing.assert_close(out, vd.tiled_residual(x, self._fn(10 ** 9), tile=128))

    def test_untiled_request_starts_from_half_the_long_side(self):
        x = torch.rand(1, 3, 300, 520)
        retries = []
        _out, used = vd.tiled_residual_with_oom_retry(
            x, self._fn(260 * 260), tile=0, on_retry=lambda t, e: retries.append(t))
        self.assertEqual(retries, [260])
        self.assertEqual(used, 260)

    def test_gives_up_below_128(self):
        with self.assertRaises(_Oom):
            vd.tiled_residual_with_oom_retry(torch.rand(1, 3, 300, 300), self._fn(10), tile=512)

    def test_other_errors_are_not_retried(self):
        retries = []

        def boom(t):
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            vd.tiled_residual_with_oom_retry(torch.rand(1, 3, 64, 64), boom, tile=512,
                                             on_retry=lambda t, e: retries.append(t))
        self.assertEqual(retries, [])


class PilConversionTests(unittest.TestCase):
    def test_every_8bit_level_round_trips(self):
        levels = np.arange(256, dtype=np.uint8)
        array = np.stack([np.tile(levels, (4, 1)), np.tile(levels[::-1], (4, 1)), np.full((4, 256), 7, np.uint8)], -1)
        image = Image.fromarray(array, "RGB")
        x, alpha = vd.pil_to_tensor(image)
        self.assertIsNone(alpha)
        self.assertEqual(tuple(x.shape), (1, 3, 4, 256))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(vd.tensor_to_pil(x).tobytes(), image.tobytes())

    def test_alpha_is_kept(self):
        image = Image.new("RGBA", (8, 6), (10, 200, 30, 77))
        x, alpha = vd.pil_to_tensor(image)
        out = vd.tensor_to_pil(x, alpha)
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.getpixel((3, 3)), (10, 200, 30, 77))

    def test_grayscale_becomes_rgb(self):
        x, alpha = vd.pil_to_tensor(Image.new("L", (5, 5), 128))
        self.assertEqual(tuple(x.shape), (1, 3, 5, 5))
        self.assertEqual(vd.tensor_to_pil(x).mode, "RGB")


def _fake_nafnet_state(width=4):
    """spandrel NAFNet 감지 키를 모두 가진 작은 state dict(값은 아무거나 — 판정은 키만 본다)."""
    state = {key: torch.zeros(1) for key in vdm.NAFNET_KEYS}
    state["intro.weight"] = torch.zeros(width, 3, 3, 3)
    return state


def _save_safetensors(path, state):
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in state.items()}, str(path))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "ESRGAN").mkdir()
        (self.root / "DeGrid").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_nafnet_files_are_listed(self):
        esrgan, degrid = self.root / "ESRGAN", self.root / "DeGrid"
        _save_safetensors(esrgan / "qwenVAEDegridNafnet_v11.safetensors", _fake_nafnet_state())
        _save_safetensors(esrgan / "4x-UltraSharpV2.safetensors",
                          {"model.0.weight": torch.zeros(4, 3, 3, 3), "model.1.sub.0.RDB1.conv1.0.weight": torch.zeros(1)})
        torch.save({"model.0.weight": torch.zeros(1)}, esrgan / "4x_foolhardy_Remacri.pth")
        # basicsr 학습 체크포인트({"params": sd}, 접두사 module.)
        torch.save({"params": {f"module.{k}": v for k, v in _fake_nafnet_state().items()}}, degrid / "net_g_30000_ft1.pth")
        (esrgan / "INFO.txt").write_text("not a model", encoding="utf-8")
        (esrgan / "broken.safetensors").write_bytes(b"\x00\x01")
        entries = vdm.discover(self.root)
        self.assertEqual([e.name for e in entries], ["qwenVAEDegridNafnet_v11", "net_g_30000_ft1"])
        self.assertEqual(Path(entries[0].path).parent.name, "ESRGAN")
        self.assertEqual(Path(entries[1].path).parent.name, "DeGrid")

    def test_header_only_detection(self):
        # 텐서 데이터를 잘라낸 파일도 헤더만으로 판정된다(텐서를 읽지 않는다는 뜻)
        path = self.root / "ESRGAN" / "big.safetensors"
        _save_safetensors(path, _fake_nafnet_state())
        data = path.read_bytes()
        (length,) = __import__("struct").unpack("<Q", data[:8])
        path.write_bytes(data[: 8 + length])
        self.assertTrue(vdm.is_nafnet_file(path))

    def test_declared_version_goes_first_and_is_the_auto_choice(self):
        # v1.1 은 modelspec.version "1.1" 을 적어 두었다. 버전을 적지 않은 NAFNet(여기서는 개발 PC 의 다른 파인튜닝
        # 'NAFNet-finetune-v2' 메타데이터)은 이름순이면 앞이지만 선언된 버전이 있는 파일 뒤로 간다
        from safetensors.torch import save_file

        esrgan = self.root / "ESRGAN"
        save_file(_fake_nafnet_state(), str(esrgan / "NAFNet-QwenVAE-DeGrid.safetensors"),
                  metadata={"modelspec.title": "NAFNet-finetune-v2"})
        save_file(_fake_nafnet_state(), str(esrgan / "qwenVAEDegridNafnet_v11.safetensors"),
                  metadata={"modelspec.version": "1.1", "modelspec.architecture": "NAFNet-small"})
        entries = vdm.discover(self.root)
        self.assertEqual([(e.name, e.version) for e in entries],
                         [("qwenVAEDegridNafnet_v11", (1, 1)), ("NAFNet-QwenVAE-DeGrid", ())])
        self.assertEqual(vdm.resolve("", entries).name, "qwenVAEDegridNafnet_v11")
        self.assertEqual(vdm.parse_version("v1.10"), (1, 10))
        self.assertEqual(vdm.parse_version(None), ())

    def test_same_stem_in_both_folders_gets_a_folder_prefix(self):
        _save_safetensors(self.root / "ESRGAN" / "degrid.safetensors", _fake_nafnet_state())
        _save_safetensors(self.root / "DeGrid" / "degrid.safetensors", _fake_nafnet_state())
        self.assertEqual([e.name for e in vdm.discover(self.root)], ["ESRGAN/degrid", "DeGrid/degrid"])

    def test_classification_follows_file_changes(self):
        path = self.root / "ESRGAN" / "x.safetensors"
        _save_safetensors(path, {"model.0.weight": torch.zeros(1)})
        self.assertFalse(vdm.is_nafnet_file(path))
        _save_safetensors(path, _fake_nafnet_state(width=8))   # 크기가 달라져 캐시 키가 바뀐다
        os.utime(path, ns=(time.time_ns(), time.time_ns() + 10_000_000))
        self.assertTrue(vdm.is_nafnet_file(path))

    def test_missing_folders_are_fine(self):
        self.assertEqual(vdm.discover(self.root / "nowhere"), [])

    def test_forge_esrgan_path_option_is_followed(self):
        modules_stub = types.ModuleType("modules")
        modules_stub.paths = types.SimpleNamespace(models_path=str(self.root))
        modules_stub.shared = types.SimpleNamespace(
            cmd_opts=types.SimpleNamespace(esrgan_models_path=str(self.root / "custom")))
        saved = sys.modules.get("modules")
        sys.modules["modules"] = modules_stub
        try:
            dirs = vdm.model_dirs()
        finally:
            if saved is None:
                sys.modules.pop("modules", None)
            else:
                sys.modules["modules"] = saved
        self.assertEqual(dirs, [self.root / "custom", self.root / "DeGrid"])
        # 폴더를 직접 준 경우(테스트·도구)는 그대로
        self.assertEqual(vdm.model_dirs(self.root), [self.root / "ESRGAN", self.root / "DeGrid"])

    def test_resolve(self):
        entries = [vdm.ModelEntry("qwenVAEDegridNafnet_v11", "/m/ESRGAN/qwenVAEDegridNafnet_v11.safetensors"),
                   vdm.ModelEntry("NAFNet-QwenVAE-DeGrid", "/m/ESRGAN/NAFNet-QwenVAE-DeGrid.safetensors")]
        self.assertIs(vdm.resolve("NAFNet-QwenVAE-DeGrid", entries), entries[1])
        self.assertIs(vdm.resolve("nafnet-qwenvae-degrid", entries), entries[1])
        self.assertIs(vdm.resolve("qwenVAEDegridNafnet_v11.safetensors", entries), entries[0])
        for auto in ("", None, "None", "auto"):
            self.assertIs(vdm.resolve(auto, entries), entries[0])
        self.assertIsNone(vdm.resolve("4x-UltraSharp", entries))
        self.assertIsNone(vdm.resolve("", []))


def _tiny_nafnet():
    from spandrel.architectures.NAFNet import NAFNet

    torch.manual_seed(0)
    model = NAFNet(img_channel=3, width=8, middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("beta", "gamma")):
                param.uniform_(-0.5, 0.5)   # 0 이면 NAFBlock 이 항등이라 SCA 까지 타지 않는다
    return model.eval()


@unittest.skipUnless(HAVE_SPANDREL, "spandrel 없음(Forge venv 에는 있음)")
class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_a_nafnet_and_the_detection_keys_match_spandrel(self):
        path = self.root / "tiny.safetensors"
        _save_safetensors(path, _tiny_nafnet().state_dict())
        self.assertTrue(vdm.is_nafnet_file(path))
        model = vdm.load_nafnet(path)
        self.assertEqual(type(model).__name__, "NAFNet")
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.training)
        # 우리 키 목록이 spandrel 의 감지 조건과 같다
        from spandrel.architectures.NAFNet import NAFNetArch

        self.assertTrue(NAFNetArch().detect(_fake_nafnet_state()))
        missing_one = _fake_nafnet_state()
        missing_one.pop(vdm.NAFNET_KEYS[-1])
        self.assertFalse(NAFNetArch().detect(missing_one))
        self.assertFalse(vdm.is_nafnet_keys(missing_one))

    def test_basicsr_params_wrapper(self):
        path = self.root / "net_g.pth"
        torch.save({"params": _tiny_nafnet().state_dict()}, path)
        self.assertTrue(vdm.is_nafnet_file(path))
        self.assertEqual(type(vdm.load_nafnet(path)).__name__, "NAFNet")

    def test_other_architectures_are_refused(self):
        from spandrel.architectures.Compact import Compact

        path = self.root / "compact.safetensors"
        _save_safetensors(path, Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2).state_dict())
        self.assertFalse(vdm.is_nafnet_file(path))
        with self.assertRaises(ValueError):
            vdm.load_nafnet(path)

    def test_default_loader_inside_inference_mode_survives_a_device_move(self):
        # 생성 탭 첫 사용(Forge 가 inference_mode 안에서 부름) → GPU 로 옮김(대역: dtype 이동, 같은 param.data= 경로)
        path = self.root / "tiny.safetensors"
        _save_safetensors(path, _tiny_nafnet().state_dict())
        rt = vdr.DegridRuntime(forge_memory=lambda: None, logger=lambda m: None)
        with torch.inference_mode():
            model = rt.model_for(path)
        self.assertFalse(any(p.is_inference() for p in model.parameters()))
        vdr.DegridRuntime._move(model, torch.float64)
        with torch.inference_mode():
            model(torch.rand(1, 3, 16, 16, dtype=torch.float64))

    def test_pad_multiple_follows_the_model(self):
        model = _tiny_nafnet()                    # 인코더 2단 → spandrel padder_size 4
        self.assertEqual(vdr.model_pad_multiple(model), 2 ** len(model.encoders))
        fn, _ = vdr.make_residual_fn(model, torch.device("cpu"), False)
        with torch.inference_mode():
            x = torch.rand(1, 3, 20, 28)          # 4 의 배수 — spandrel 도 여기도 채우지 않는다
            self.assertTrue(torch.equal(fn(x), model(x)))
            y = torch.rand(1, 3, 21, 27)          # spandrel 은 0 으로, 여기는 반사로 채운다
            self.assertFalse(torch.equal(fn(y), model(y)))


# ---------------------------------------------------------------------------
# .pth 안전 — 찾을 때는 torch.load 를 부르지 않고, 불러올 때는 Forge 가 감싸기 전 로더에 str 을 넘기지 않는다
# ---------------------------------------------------------------------------

FORGE_PATCH_BASIC_PY = FORGE_ROOT / "modules_forge" / "patch_basic.py"
REAL_MODELS_DIR = FORGE_ROOT / "models"


def _simulated_build_loaded(module, loader_name):
    """Forge 밖(CI)용 ``modules_forge/patch_basic.build_loaded`` 대역 — 같은 동작만: 원래 로더를 ``<name>_origin`` 에 두고,
    감싼 로더가 실패하면 str 인자 중 파일인 것을 ``<파일>.corrupted`` 로 바꾼 뒤 BufferError."""
    origin_name = f"{loader_name}_origin"
    if not hasattr(module, origin_name):
        setattr(module, origin_name, getattr(module, loader_name))
    original = getattr(module, origin_name)

    def loader(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception:
            for path in list(args) + list(kwargs.values()):
                if isinstance(path, str) and os.path.isfile(path):
                    os.replace(path, f"{path}.corrupted")
            raise BufferError("Failed to load model...") from None

    setattr(module, loader_name, loader)


def _forge_build_loaded():
    """Forge 가 옆에 있으면 실제 ``build_loaded``(import 하지 않고 — gradio·modules.errors 를 끌고 옴 — AST 로 꺼냄), 없으면 대역."""
    if not FORGE_PATCH_BASIC_PY.is_file():
        return _simulated_build_loaded
    tree = ast.parse(FORGE_PATCH_BASIC_PY.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_loaded"]
    if len(nodes) != 1:
        return _simulated_build_loaded
    namespace = {"os": os, "warnings": warnings, "wraps": functools.wraps, "display": lambda error, task: None}
    exec(compile(ast.Module(nodes, []), str(FORGE_PATCH_BASIC_PY), "exec"), namespace)
    return namespace["build_loaded"]


class _ForgePatchedLoaders:
    """``torch.load``·``safetensors.torch.load_file`` 을 Forge 처럼 감싼다(원래 로더는 ``*_origin``). 감싼 로더가 불린 것을
    ``wrapped_calls`` 에 남기고, 끝나면 둘 다 되돌린다."""

    def __enter__(self):
        import safetensors.torch as st

        self.wrapped_calls = []
        self._stack = contextlib.ExitStack()
        build_loaded = _forge_build_loaded()
        for module, name in ((torch, "load"), (st, "load_file")):
            stand_in = types.ModuleType(f"{module.__name__}_stand_in")
            setattr(stand_in, name, getattr(module, f"{name}_origin", None) or getattr(module, name))
            build_loaded(stand_in, name)

            def recorded(*args, _wrapped_loader=getattr(stand_in, name), _loader_name=name, **kwargs):
                self.wrapped_calls.append((_loader_name, args, kwargs))
                return _wrapped_loader(*args, **kwargs)

            self._stack.enter_context(mock.patch.object(module, name, recorded))
            self._stack.enter_context(
                mock.patch.object(module, f"{name}_origin", getattr(stand_in, f"{name}_origin"), create=True))
        return self

    def __exit__(self, *exc):
        self._stack.close()
        return False


@contextlib.contextmanager
def _forbid_loaders():
    """찾기 동안 가중치 로더(감싼 것·원래 것 모두)가 불리면 기록하고 실패시킨다. 불린 목록을 돌려준다."""
    import safetensors.torch as st

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("weights loader called during discovery")

    with contextlib.ExitStack() as stack:
        for module, name in ((torch, "load"), (torch, "load_origin"), (st, "load_file"), (st, "load_file_origin")):
            stack.enter_context(mock.patch.object(module, name, forbidden, create=True))
        yield calls


_PICKLED_CALLS: list = []


def _record_pickled_call(tag):
    _PICKLED_CALLS.append(tag)
    return tag


class _PicklePayload:
    """되살리면 ``_record_pickled_call`` 을 부르는 객체 — 키 읽기가 pickle 속 전역을 부르지 않는지 보는 표식."""

    def __reduce__(self):
        return (_record_pickled_call, ("called",))


class TorchFileDiscoveryTests(unittest.TestCase):
    """.pth/.pt 찾기는 torch.load 없이 zip 의 data.pkl 만 — 옛 형식 pickle 은 열지 않는다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.esrgan = self.root / "ESRGAN"
        self.esrgan.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _assert_untouched(self, *paths):
        for path in paths:
            self.assertTrue(Path(path).is_file(), path)
        self.assertEqual(sorted(p.name for p in self.root.rglob("*.corrupted")), [])

    def test_legacy_pickle_is_skipped_without_any_load(self):
        # 옛 형식(zip 아님, 예: 4x-UltraSharp.pth) — 키가 NAFNet 이어도 열지 않는다(DeGrid 는 safetensors·새 형식 pth 로 배포).
        # 고치기 전에는 torch.load(mmap=True) 가 'mmap can only be used with files saved with ...' 로 실패하고 통째로 다시 읽었다.
        legacy = self.esrgan / "4x-UltraSharp.pth"
        torch.save(_fake_nafnet_state(), legacy, _use_new_zipfile_serialization=False)
        self.assertFalse(vdm.is_torch_zip(legacy))
        with _forbid_loaders() as calls, self.assertLogs(vdm._LOG, "DEBUG") as logs:
            self.assertEqual(vdm.discover(self.root), [])
            self.assertEqual(vdm.discover(self.root), [])   # 두 번째는 판정 캐시 — 로그도 한 줄
            self.assertIsNone(vdm.read_torch_state_skeleton(legacy))
        self.assertEqual(calls, [])
        self.assertEqual([(r.levelno, "4x-UltraSharp.pth" in r.getMessage()) for r in logs.records], [(logging.DEBUG, True)])
        self._assert_untouched(legacy)

    def test_zip_pth_with_nafnet_keys_is_found_without_any_load(self):
        degrid = self.root / "DeGrid"
        degrid.mkdir()
        torch.save(_fake_nafnet_state(), self.esrgan / "degrid_new_format.pth")
        # basicsr 학습 체크포인트({"params": sd}, 접두사 module.)도 .pt 로
        torch.save({"params": {f"module.{k}": v for k, v in _fake_nafnet_state().items()}}, degrid / "net_g_ft.pt")
        with _forbid_loaders() as calls:
            entries = vdm.discover(self.root)
            keys = vdm.read_torch_keys(self.esrgan / "degrid_new_format.pth")
        self.assertEqual(calls, [])
        self.assertEqual([e.name for e in entries], ["degrid_new_format", "net_g_ft"])
        self.assertEqual(sorted(keys), sorted(vdm.NAFNET_KEYS))

    def test_zip_pth_with_other_keys_is_skipped(self):
        other = self.esrgan / "4x_foolhardy_Remacri.pth"
        torch.save({"model.0.weight": torch.zeros(4, 3, 3, 3), "model.1.sub.0.RDB1.conv1.0.weight": torch.zeros(1)}, other)
        self.assertTrue(vdm.is_torch_zip(other))
        with _forbid_loaders() as calls:
            self.assertEqual(vdm.discover(self.root), [])
            self.assertEqual(vdm.read_torch_keys(other), ["model.0.weight", "model.1.sub.0.RDB1.conv1.0.weight"])
        self.assertEqual(calls, [])
        self._assert_untouched(other)

    def test_key_reader_never_calls_pickled_globals(self):
        path = self.esrgan / "payload.pth"
        torch.save({"state_dict": _fake_nafnet_state(), "extra": _PicklePayload()}, path)
        _PICKLED_CALLS.clear()
        with _forbid_loaders() as calls:
            self.assertTrue(vdm.is_nafnet_file(path))
        self.assertEqual((calls, _PICKLED_CALLS), ([], []))
        # 대조: 같은 파일을 torch.load(weights_only=False)로 풀면 그 전역이 실제로 불린다
        torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(_PICKLED_CALLS, ["called"])
        _PICKLED_CALLS.clear()

    def test_broken_or_foreign_zip_is_not_a_model(self):
        truncated = self.esrgan / "truncated.pth"
        torch.save(_fake_nafnet_state(), truncated)
        data = truncated.read_bytes()
        truncated.write_bytes(data[: len(data) // 2])        # 앞은 PK 지만 zip 끝 레코드가 없다
        foreign = self.esrgan / "archive.pt"
        with zipfile.ZipFile(foreign, "w") as archive:
            archive.writestr("readme.txt", "not a checkpoint")
        with _forbid_loaders() as calls:
            self.assertEqual(vdm.discover(self.root), [])
        self.assertEqual(calls, [])
        self._assert_untouched(truncated, foreign)


@unittest.skipUnless(HAVE_SPANDREL, "spandrel 없음(Forge venv 에는 있음)")
class ForgePatchedLoaderTests(unittest.TestCase):
    """Forge 가 감싼 ``torch.load``·``load_file``(실패하면 str 파일 경로를 ``.corrupted`` 로 바꿈)이 찾기·불러오기에서 불리지 않는다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_harness_renames_like_forge(self):
        # 대조: 고치기 전 찾기처럼 감싼 로더에 str 경로·mmap 으로 옛 형식을 주면 Forge 는 사용자의 파일 이름을 바꾼다
        legacy = self.root / "4x-UltraSharp.pth"
        torch.save({"model.0.weight": torch.zeros(1)}, legacy, _use_new_zipfile_serialization=False)
        with _ForgePatchedLoaders() as forge, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BufferError):
                torch.load(str(legacy), map_location="cpu", weights_only=True, mmap=True)
        self.assertEqual(len(forge.wrapped_calls), 1)
        self.assertFalse(legacy.exists())
        self.assertTrue(Path(f"{legacy}.corrupted").is_file())

    def test_discovery_and_loading_never_reach_the_wrapped_loaders(self):
        esrgan = self.root / "ESRGAN"
        esrgan.mkdir()
        legacy = esrgan / "4x-UltraSharp.pth"
        torch.save({"model.0.weight": torch.zeros(1)}, legacy, _use_new_zipfile_serialization=False)
        zip_pth = esrgan / "degrid.pth"
        torch.save(_tiny_nafnet().state_dict(), zip_pth)
        st_file = esrgan / "degrid_st.safetensors"
        _save_safetensors(st_file, _tiny_nafnet().state_dict())
        broken_pth = esrgan / "broken.pth"
        broken_pth.write_bytes(b"\x80\x02not a pickle")
        # 헤더는 NAFNet(목록에 나옴)이지만 텐서 데이터가 잘린 safetensors — 불러오기가 실패한다
        broken_st = esrgan / "broken.safetensors"
        _save_safetensors(broken_st, _tiny_nafnet().state_dict())
        data = broken_st.read_bytes()
        (length,) = __import__("struct").unpack("<Q", data[:8])
        broken_st.write_bytes(data[: 8 + length])
        files = [legacy, zip_pth, st_file, broken_pth, broken_st]

        out = io.StringIO()
        with _ForgePatchedLoaders() as forge, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            self.assertEqual([e.name for e in vdm.discover(self.root)], ["broken", "degrid", "degrid_st"])
            for path in (zip_pth, str(zip_pth), st_file, str(st_file)):
                self.assertEqual(type(vdm.load_nafnet(path)).__name__, "NAFNet")
            rt = vdr.DegridRuntime(forge_memory=lambda: None, logger=lambda m: None)
            self.assertEqual(type(rt.model_for(str(zip_pth))).__name__, "NAFNet")
            self.assertIsInstance(vdm.torch_load(str(legacy)), dict)   # 옛 형식도 원래 로더로(mmap 없이)는 읽힌다
            for failing in (lambda: vdm.load_nafnet(str(broken_st)), lambda: vdm.torch_load(str(broken_pth))):
                with self.assertRaises(Exception) as ctx:
                    failing()
                self.assertNotIsInstance(ctx.exception, BufferError)   # 감싼 로더가 아니라 원래 로더의 오류
        self.assertEqual(forge.wrapped_calls, [])
        self.assertEqual(out.getvalue(), "")
        for path in files:
            self.assertTrue(path.is_file(), path)
        self.assertEqual(sorted(p.name for p in self.root.rglob("*.corrupted")), [])

    def test_without_an_original_loader_no_str_reaches_the_loader(self):
        # Forge 밖(원래 로더 없음)에서는 그 로더를 쓰되 str 인자는 넘기지 않는다(경로 Path, map_location torch.device)
        import safetensors.torch as st

        seen = []

        def spy(*args, **kwargs):
            seen.append([type(v).__name__ for v in list(args) + list(kwargs.values()) if isinstance(v, str)])
            raise RuntimeError("spy")

        target = str(self.root / "missing.pth")
        with mock.patch.object(torch, "load", spy), mock.patch.object(torch, "load_origin", None, create=True), \
                mock.patch.object(st, "load_file", spy), mock.patch.object(st, "load_file_origin", None, create=True):
            for load in (vdm.torch_load, vdm.safetensors_load):
                with self.assertRaises(RuntimeError):
                    load(target)
        self.assertEqual(seen, [[], []])
        origin = lambda *a, **k: None  # noqa: E731
        self.assertIs(vdm.unpatched_loader(types.SimpleNamespace(load=spy, load_origin=origin), "load"), origin)
        self.assertIs(vdm.unpatched_loader(types.SimpleNamespace(load=spy), "load"), spy)


@unittest.skipUnless(
    os.environ.get("SAM3_RUN_FORGE_INTEGRATION_TESTS") == "1" and (REAL_MODELS_DIR / "ESRGAN").is_dir(),
    "실제 models/ESRGAN 목록(읽기만) — SAM3_RUN_FORGE_INTEGRATION_TESTS=1",
)
class RealModelFolderTests(unittest.TestCase):
    def test_listing_reads_only_headers_and_changes_nothing(self):
        def snapshot():
            return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                          for folder in vdm.model_dirs(REAL_MODELS_DIR) if folder.is_dir() for p in folder.iterdir())

        before = snapshot()
        with vdm._CACHE_LOCK:
            vdm._CLASSIFY_CACHE.clear()
        out = io.StringIO()
        with _forbid_loaders() as calls, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out), \
                self.assertNoLogs(level="WARNING"):
            entries = vdm.discover(REAL_MODELS_DIR)
        self.assertEqual(calls, [])
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(snapshot(), before)
        paths = [Path(e.path) for e in entries]
        for path in paths:
            self.assertTrue(path.suffix.lower() == ".safetensors" or vdm.is_torch_zip(path), path)
            self.assertTrue(vdm.is_nafnet_file(path), path)
        for known in (REAL_MODEL, REAL_MODEL_ALT):
            if known.is_file() and known.parent in vdm.model_dirs(REAL_MODELS_DIR):
                self.assertIn(known, paths)


def _stub_residual(x):
    """점마다 정해지는 잔차 0.05·cos(8πx) — 부호가 섞이고(|평균| 약 8/255) 입력과 상관이 없어 잔차 검사를 지난다.
    (x 에 비례하는 잔차는 이미지처럼 보여 ``output_looks_like_image`` 가 거른다.)"""
    return torch.cos(x * (8 * math.pi)) * 0.05


class _ShiftModel(torch.nn.Module):
    """잔차 = ``_stub_residual`` (국소·결정적). 호출 크기를 기록한다."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape))
        return _stub_residual(x) + self.p


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m.safetensors"
        self.path.write_bytes(b"x")
        self.loaded = []
        self.model = _ShiftModel()
        self.logs = []

        def loader(path):
            self.loaded.append(path)
            return self.model

        self.rt = vdr.DegridRuntime(loader=loader, forge_memory=lambda: None, logger=self.logs.append)
        self.entry = vdm.ModelEntry("m", str(self.path))

    def tearDown(self):
        self.tmp.cleanup()

    def _image(self, mode="RGB"):
        g = np.random.default_rng(0)
        return Image.fromarray(g.integers(0, 256, (70, 90, 3), dtype=np.uint8), "RGB").convert(mode)

    def test_cpu_run_matches_the_formula(self):
        image = self._image()
        out = self.rt.run(image, self.entry, mode="full", strength=1.0, tile=0, device="cpu", precision="fp32",
                          keep_loaded=False)
        x, _ = vd.pil_to_tensor(image)
        expected = vd.tensor_to_pil(torch.clamp(x + _stub_residual(x), 0, 1))
        self.assertEqual(out.image.tobytes(), expected.tobytes())
        self.assertEqual((out.device, out.precision, out.tile_used), ("cpu", "fp32", 0))
        self.assertGreater(out.stats["changed_pixels"], 0.5)
        self.assertIn("Full", out.summary())

    def test_modes_and_strength_through_the_runtime(self):
        image = self._image()
        x, _ = vd.pil_to_tensor(image)
        delta = _stub_residual(x)
        for mode in vd.MODES:
            for strength in (0.5, 1.5):
                with self.subTest(mode=mode, strength=strength):
                    out = self.rt.run(image, self.entry, mode=vd.MODE_LABELS[mode], strength=strength, tile=512,
                                      device="cpu")
                    expected = vd.tensor_to_pil(vd.finalize(vd.apply_residual(x, delta, mode, strength)))
                    self.assertEqual(out.image.tobytes(), expected.tobytes())

    def test_strength_zero_returns_the_image_without_loading(self):
        image = self._image()
        out = self.rt.run(image, self.entry, strength=0, device="cpu")
        self.assertTrue(out.skipped)
        self.assertEqual(out.image.tobytes(), image.tobytes())
        self.assertEqual(self.loaded, [])

    def test_alpha_survives(self):
        image = self._image("RGBA")
        image.putalpha(123)
        out = self.rt.run(image, self.entry, device="cpu")
        self.assertEqual(out.image.mode, "RGBA")
        self.assertEqual(out.image.getchannel("A").getextrema(), (123, 123))

    def test_model_is_cached_until_the_file_changes(self):
        image = self._image()
        self.rt.run(image, self.entry, device="cpu")
        self.rt.run(image, self.entry, device="cpu")
        self.assertEqual(len(self.loaded), 1)
        self.path.write_bytes(b"xy")
        self.rt.run(image, self.entry, device="cpu")
        self.assertEqual(len(self.loaded), 2)

    def test_tiles_are_used(self):
        self.rt.run(Image.new("RGB", (300, 200)), self.entry, tile=128, device="cpu")
        self.assertTrue(all(h <= 128 and w <= 128 for _b, _c, h, w in self.model.shapes))
        self.assertGreater(len(self.model.shapes), 4)

    def test_fp16_overflow_tile_is_redone_in_fp32(self):
        class Overflowing(torch.nn.Module):
            def forward(self, x):
                if torch.is_autocast_enabled("cpu"):
                    return torch.full_like(x, float("inf"))
                return x * 0 + 0.01

        fn, retiles = vdr.make_residual_fn(Overflowing(), torch.device("cpu"), True)
        with torch.inference_mode():
            out = fn(torch.rand(1, 3, 8, 8))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertEqual(retiles[0], 1)
        fn32, retiles32 = vdr.make_residual_fn(Overflowing(), torch.device("cpu"), False)
        with torch.inference_mode():
            torch.testing.assert_close(fn32(torch.rand(1, 3, 4, 4)), torch.full((1, 3, 4, 4), 0.01))
        self.assertEqual(retiles32[0], 0)

    def test_release_policy(self):
        self.assertTrue(vdr.release_after_run(False, "cuda"))
        self.assertFalse(vdr.release_after_run(True, "cuda"))
        self.assertTrue(vdr.release_after_run(True, "cpu"))

    def test_memory_estimate(self):
        self.assertEqual(vdr.memory_estimate(1856, 1216, 512), vdr.BYTES_PER_PIXEL * 512 * 512)
        self.assertEqual(vdr.memory_estimate(1856, 1216, 0), vdr.BYTES_PER_PIXEL * 1856 * 1216)
        self.assertEqual(vdr.memory_estimate(300, 200, 512), vdr.BYTES_PER_PIXEL * 300 * 200)

    def test_cpu_option_never_asks_forge(self):
        self.assertEqual(vdr.choose_device("cpu"), torch.device("cpu"))
        fake = types.SimpleNamespace(get_torch_device=lambda: torch.device("cpu"))
        saved = sys.modules.get("backend.memory_management")
        sys.modules["backend.memory_management"] = fake
        try:
            self.assertEqual(vdr.choose_device("auto"), torch.device("cpu"))
        finally:
            if saved is None:
                sys.modules.pop("backend.memory_management", None)
            else:
                sys.modules["backend.memory_management"] = saved

    def test_image_output_model_is_refused_and_the_image_is_not_changed(self):
        class Identity(torch.nn.Module):      # 복원 이미지를 내는 일반 NAFNet 대역
            def forward(self, x):
                return x.clone()

        rt = vdr.DegridRuntime(loader=lambda path: Identity(), forge_memory=lambda: None, logger=self.logs.append)
        with self.assertRaises(vdr.NotResidualModelError) as caught:
            rt.run(self._image(), self.entry, device="cpu")
        message = str(caught.exception)
        self.assertIn("not a DeGrid residual model", message)
        self.assertIn("output does not look like a residual", message)
        self.assertIn("correlation with the input +1.00", message)
        self.assertNotIn("outputs an image", message)

    def test_large_opposing_residual_is_applied_through_the_runtime(self):
        class Flatten(torch.nn.Module):     # 잔 무늬를 누르는 DeGrid 대역: 잔차 = -0.4·(x - 5x5 평균) — 크고 입력과 반대로
            def forward(self, x):           # (4px 스크린톤에서 약 42/255 — 실제 v1.1 31.3, Anzhc 46.8. 0.4 없이는 106/255 로 폭주 문턱 위)
                return -0.4 * (x - torch.nn.functional.avg_pool2d(x, 5, 1, 2, count_include_pad=False))

        yy, xx = np.mgrid[:64, :64]
        tone = Image.fromarray((255 * ~(((yy % 4) < 2) & ((xx % 4) < 2))).astype(np.uint8)).convert("RGB")
        rt = vdr.DegridRuntime(loader=lambda path: Flatten(), forge_memory=lambda: None, logger=self.logs.append)
        out = rt.run(tone, self.entry, tile=0, device="cpu")
        self.assertGreater(out.stats["abs_mean_255"], 25.0)
        self.assertLess(out.stats["abs_mean_255"], vd.RESIDUAL_BLOWUP_ABS_MEAN * 255)
        self.assertGreater(out.stats["changed_pixels"], 0.9)

    def test_image_models_on_grain_and_screentone_are_refused_through_the_runtime(self):
        # dc59409 은 둘 다 적용했다 — 회색 입자는 모든 픽셀이 255, 스크린톤은 PSNR 6~9 dB
        yy, xx = np.mgrid[:128, :128]
        tone = Image.fromarray((255 * ~(((yy % 4) < 2) & ((xx % 4) < 2))).astype(np.uint8)).convert("RGB")
        for model_name, model in (("gaussian s2", _BlurImageModel(2.0)), ("median3", _Median3ImageModel())):
            for image_name, image in (("gray grain 200+-4", _grain_image()), ("screentone 4px", tone)):
                with self.subTest(model=model_name, image=image_name):
                    before = image.tobytes()
                    rt = vdr.DegridRuntime(loader=lambda path, m=model: m, forge_memory=lambda: None,
                                           logger=self.logs.append)
                    with torch.inference_mode(), self.assertRaises(vdr.NotResidualModelError) as caught:
                        rt.run(image, self.entry, tile=512, device="cpu")
                    self.assertIn("follows the input brightness", str(caught.exception))
                    self.assertEqual(image.tobytes(), before)

    def test_blown_up_residual_skips_the_image(self):
        rt = vdr.DegridRuntime(loader=lambda path: _BlowUpModel(), forge_memory=lambda: None, logger=self.logs.append)
        image = self._image()
        before = image.tobytes()
        with self.assertRaises(vdr.ResidualBlowUpError) as caught:
            rt.run(image, self.entry, tile=0, device="cpu")
        message = str(caught.exception)
        self.assertTrue(message.startswith("output blew up (mean |residual| 510.0/255 > 100/255"), message)
        self.assertIsInstance(caught.exception, vdr.DegridSkipError)
        self.assertNotIsInstance(caught.exception, vdr.NotResidualModelError)
        self.assertEqual(vdr.failure_reason(caught.exception), message)
        self.assertEqual(image.tobytes(), before)

    def test_failure_reason_names_only_unexpected_errors(self):
        self.assertEqual(vdr.failure_reason(vdr.NotResidualModelError("not a DeGrid residual model: m")),
                         "not a DeGrid residual model: m")
        self.assertEqual(vdr.failure_reason(vdr.ResidualBlowUpError("output blew up (x)")), "output blew up (x)")
        self.assertEqual(vdr.failure_reason(RuntimeError("CUDA out of memory")), "RuntimeError: CUDA out of memory")


class _ConvResidual(torch.nn.Module):
    """작은 3x3 합성곱 잔차(|평균| 1/255 미만). 로더처럼 불릴 때 새로 만들어, 파라미터가 그 문맥(inference_mode 인지)을
    그대로 가진다 — inference 텐서 합성곱 가중치를 옮긴 뒤 부르면 'Inference tensors do not track version counter.'"""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.mul_(0.005)
            self.conv.bias.zero_()
        self.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class _RoundTripRuntime(vdr.DegridRuntime):
    """GPU 경로 대역(CPU): 계산 장치로 올렸다 내리는 것을 dtype 왕복으로 한다 — CPU↔CUDA 이동과 같은
    ``Module._apply`` 의 ``param.data =`` 경로다(dense CPU·CUDA 텐서는 얕은 복사 호환이라 자리에서 바꾼다)."""

    def _acquire(self, model, device, memory_required):
        self._move(model, torch.float64)
        self._move(model, torch.float32)


class InferenceModeLoadTests(unittest.TestCase):
    """Forge 는 ``postprocess_image_after_composite`` 를 ``torch.inference_mode()`` 안에서 부른다(``modules/processing.py``
    process_images_inner). 생성 탭에서 처음 불러온 모델도 일반 텐서여야 GPU 로 옮긴 뒤에도 계산할 수 있다 — 아니면 그 뒤
    모든 이미지가(캐시가 파일 기준이라 Extras·CPU 도) Forge 를 다시 켤 때까지 실패한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m.safetensors"
        self.path.write_bytes(b"x")
        self.entry = vdm.ModelEntry("m", str(self.path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_loader_runs_outside_inference_mode_without_grad(self):
        seen = []

        def loader(path):
            seen.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
            return _ConvResidual()

        rt = vdr.DegridRuntime(loader=loader, forge_memory=lambda: None, logger=lambda m: None)
        with torch.inference_mode():
            model = rt.model_for(self.path)
        self.assertEqual(seen, [(False, False)])
        self.assertFalse(any(p.is_inference() for p in model.parameters()))
        vdr.DegridRuntime._move(model, torch.float64)
        with torch.inference_mode():
            model(torch.rand(1, 3, 16, 16, dtype=torch.float64))

    def test_generation_tab_first_use_then_every_later_image(self):
        rt = _RoundTripRuntime(loader=lambda path: _ConvResidual(), forge_memory=lambda: None, logger=lambda m: None)
        image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (40, 48, 3), dtype=np.uint8), "RGB")
        outs = []
        for label in ("txt2img 1", "txt2img 2", "txt2img 3"):
            with self.subTest(image=label), torch.inference_mode():
                outs.append(rt.run(image, self.entry, tile=0, device="cpu").image.tobytes())
        extras = rt.run(image, self.entry, tile=0, device="cpu")   # Extras(inference_mode 밖)
        self.assertEqual(outs, [extras.image.tobytes()] * 3)


class _FakeLoaded:
    def __init__(self, patcher):
        self.model = patcher
        self.unloads = 0

    def model_unload(self):
        self.unloads += 1
        self.model.detach()


class _FakePatcher:
    made = []

    def __init__(self, model, load_device, offload_device):
        self.model, self.load_device, self.offload_device = model, load_device, offload_device
        self.detached = 0
        _FakePatcher.made.append(self)

    def detach(self, unpatch_all=True):
        self.detached += 1


class _FakeMemory:
    def __init__(self):
        self.current_loaded_models = []
        self.calls = []
        self.emptied = 0
        self.vram_state = types.SimpleNamespace(name="NORMAL_VRAM")

    def load_models_gpu(self, models, memory_required=0, force_full_load=False):
        self.calls.append((list(models), memory_required, force_full_load))
        for model in models:
            if not any(entry.model is model for entry in self.current_loaded_models):
                self.current_loaded_models.insert(0, _FakeLoaded(model))

    def soft_empty_cache(self):
        self.emptied += 1


class _DeviceFakingRuntime(vdr.DegridRuntime):
    """GPU 없이 GPU 경로를 본다 — 실제 장치 이동 대신 '어느 장치에 있다' 는 표시만 바꾼다."""

    def __init__(self, memory, **kw):
        super().__init__(forge_memory=lambda: (memory, _FakePatcher), **kw)
        self.where = "cpu"
        self.moves = []
        self.forge_places = True

    def _off_device(self, model, device):
        return torch.device(device).type != self.where

    def _move(self, model, device):
        self.moves.append(torch.device(device).type)
        self.where = torch.device(device).type


class ForgeMemoryTests(unittest.TestCase):
    def setUp(self):
        _FakePatcher.made.clear()
        self.mm = _FakeMemory()
        self.logs = []
        self.rt = _DeviceFakingRuntime(self.mm, logger=self.logs.append)
        self.model = _ShiftModel()
        self.rt._model = self.model
        self.cuda = torch.device("cuda")

        original = self.mm.load_models_gpu

        def load_and_place(models, memory_required=0, force_full_load=False):
            original(models, memory_required=memory_required, force_full_load=force_full_load)
            if self.rt.forge_places:
                self.rt.where = "cuda"

        self.mm.load_models_gpu = load_and_place

    def test_gpu_goes_through_load_models_gpu_with_full_load(self):
        self.rt._acquire(self.model, self.cuda, 123.0)
        self.assertEqual(len(self.mm.calls), 1)
        models, memory_required, full = self.mm.calls[0]
        self.assertEqual(memory_required, 123.0)
        self.assertTrue(full)
        patcher = models[0]
        self.assertIs(patcher.model, self.model)
        self.assertEqual((patcher.load_device, patcher.offload_device), (self.cuda, torch.device("cpu")))
        self.assertEqual(self.rt.moves, [])      # Forge 가 올렸으므로 직접 옮기지 않는다
        self.rt._acquire(self.model, self.cuda, 123.0)
        self.assertEqual(len(_FakePatcher.made), 1)   # 같은 패처를 다시 쓴다

    def test_release_takes_it_out_of_forge_and_empties_the_cache(self):
        self.rt._acquire(self.model, self.cuda, 1.0)
        entry = self.mm.current_loaded_models[0]

        def unload():
            entry.unloads += 1
            self.rt.where = "cpu"        # detach → offload_device

        entry.model_unload = unload
        self.assertTrue(self.rt.release())
        self.assertEqual(self.mm.current_loaded_models, [])
        self.assertEqual(entry.unloads, 1)
        self.assertEqual(self.mm.emptied, 1)
        self.assertFalse(self.rt.release())      # 두 번째는 할 일 없음

    def test_partial_load_is_finished_by_hand_and_logged_once(self):
        self.rt.forge_places = False     # --novram 처럼 Forge 가 다 올리지 않음
        self.rt._acquire(self.model, self.cuda, 1.0)
        self.assertEqual(self.rt.moves, ["cuda"])
        self.rt.where = "cpu"
        self.rt._acquire(self.model, self.cuda, 1.0)
        self.assertEqual(len([m for m in self.logs if "직접 옮깁니다" in m]), 1)

    def test_cpu_run_after_a_kept_gpu_model_brings_it_back(self):
        self.rt._acquire(self.model, self.cuda, 1.0)
        entry = self.mm.current_loaded_models[0]
        entry.model_unload = lambda: setattr(self.rt, "where", "cpu")
        self.rt._acquire(self.model, torch.device("cpu"), 1.0)
        self.assertEqual(self.rt.where, "cpu")
        self.assertEqual(self.mm.current_loaded_models, [])


def _forge_built_model():
    """Forge 가 만든 모델 대역 — Forge 는 UNet·TE·VAE 를 ``@torch.inference_mode()`` 안에서 만든다(``backend/loader.py``
    forge_loader, ``modules/sd_models.py`` forge_model_reload). 그래서 파라미터가 inference 텐서다."""
    with torch.inference_mode():
        return torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), torch.nn.LayerNorm(6))


def _dtype_round_trip(model):
    """장치 이동 대역 — CPU↔CUDA 와 같은 ``Module._apply`` 의 ``param.data =`` 경로(dtype 왕복)."""
    model.to(torch.float64)
    model.to(torch.float32)


class _MovingLoaded(_FakeLoaded):
    def model_unload(self):
        self.unloads += 1
        _dtype_round_trip(self.model.model)        # detach → offload_device(CPU) 로


class _EvictingMemory(_FakeMemory):
    """``load_models_gpu`` 대역: 자리가 모자라 Forge 모델을 (일부) 내리고(``free_memory`` → ``model_unload``) 요청한 모델을
    올린다 — 둘 다 부른 문맥에서. 부를 때의 inference_mode 여부를 기록한다."""

    def __init__(self, forge_model):
        super().__init__()
        self.forge_model = forge_model
        self.inference_mode_seen = []

    def load_models_gpu(self, models, memory_required=0, force_full_load=False):
        self.inference_mode_seen.append(torch.is_inference_mode_enabled())
        self.calls.append((list(models), memory_required, force_full_load))
        self.forge_model.to(torch.float64)          # free_memory: Forge 모델을 내림
        for patcher in models:
            if not any(entry.model is patcher for entry in self.current_loaded_models):
                _dtype_round_trip(patcher.model)     # 요청한 모델을 올림
                self.current_loaded_models.insert(0, _MovingLoaded(patcher))


class _PlacedRuntime(vdr.DegridRuntime):
    """Forge 가 모델을 (가짜) GPU 에 다 올렸다고 본다 — 직접 옮기지 않는다."""

    @staticmethod
    def _off_device(model, device):
        return False


class AcquireInferenceModeTests(unittest.TestCase):
    """``load_models_gpu`` 는 부른 문맥 그대로 부른다(``scripts/anima_vae_2x.py`` ``_load_decoder`` 와 같음). VRAM 이 모자라면
    Forge 가 자리를 내려고 자기 UNet·TE·VAE 를 (일부) 내리는데, 이 모델들은 inference_mode 안에서 만들어져 inference_mode
    밖에서 옮기면 Forge 가 다시 올릴 때까지 'Inference tensors do not track version counter.' 로 계산이 죽는다. DeGrid 모델은
    ``model_for`` 가 inference_mode 밖에서 만들어 어느 문맥에서 옮겨도 계산되므로 감쌀 이유가 없다."""

    def setUp(self):
        _FakePatcher.made.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m.safetensors"
        self.path.write_bytes(b"x")
        self.forge_model = _forge_built_model()
        self.mm = _EvictingMemory(self.forge_model)
        self.rt = _PlacedRuntime(loader=lambda path: _ConvResidual(), forge_memory=lambda: (self.mm, _FakePatcher),
                                 logger=lambda m: None)
        self.cuda = torch.device("cuda")

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_models_gpu_runs_in_the_callers_context(self):
        model = self.rt.model_for(self.path)
        with torch.inference_mode():                 # 생성 탭(postprocess_image_after_composite)
            self.rt._acquire(model, self.cuda, 1.0)
        self.rt.release()
        self.rt._acquire(model, self.cuda, 1.0)      # Extras
        self.assertEqual(self.mm.inference_mode_seen, [True, False])

    def test_forge_model_moved_to_make_room_keeps_working(self):
        with torch.inference_mode():
            model = self.rt.model_for(self.path)
            self.rt._acquire(model, self.cuda, 1.0)
            # 다음 생성의 Forge 모델 계산 — 예전(inference_mode(False) 로 감쌈)에는 여기서 RuntimeError
            out = self.forge_model(torch.rand(1, 3, 8, 8, dtype=torch.float64))
        self.assertEqual(tuple(out.shape), (1, 3, 6, 6))

    def test_degrid_model_survives_moves_inside_and_outside_inference_mode(self):
        x = torch.rand(1, 3, 16, 16)
        with torch.inference_mode():                 # 생성 탭에서 처음 불러옴
            model = self.rt.model_for(self.path)
        with torch.inference_mode():
            reference = model(x).clone()
        outs = []
        for label, inside in (("txt2img", True), ("txt2img", True), ("Extras", False), ("txt2img", True)):
            with self.subTest(step=label):
                with torch.inference_mode(inside):
                    self.rt._acquire(model, self.cuda, 1.0)     # 부른 문맥에서 올림
                with torch.inference_mode():
                    outs.append(model(x).clone())
                with torch.inference_mode(inside):
                    self.assertTrue(self.rt.release())          # 내림(inference_mode 밖)
        # 남겨 둔(keep_loaded) 모델을 생성 탭에서 올린 뒤 Forge 가 inference_mode 밖에서 옮기는 경우
        with torch.inference_mode():
            self.rt._acquire(model, self.cuda, 1.0)
        self.assertTrue(any(p.is_inference() for p in model.parameters()))
        _dtype_round_trip(model)
        with torch.inference_mode():
            outs.append(model(x).clone())
        vdr.DegridRuntime._move(model, torch.float64)            # 부분 로드를 직접 마무리하는 경로
        with torch.inference_mode():
            outs.append(model(x.double()).float().clone())
        for out in outs:
            torch.testing.assert_close(out, reference)
        self.assertFalse(any(p.is_inference() for p in model.parameters()))


def _synthetic_illustration(size=256, seed=0):
    """VAE 를 거친 적 없는 깨끗한 그림(단색 면·그라데이션·안티에일리어싱 선)."""
    rng = np.random.default_rng(seed)
    big = size * 4
    img = Image.new("RGB", (big, big), (250, 244, 240))
    draw = ImageDraw.Draw(img)
    for y in range(big // 2):
        c = int(150 + 80 * y / (big / 2))
        draw.line([(0, y), (big, y)], fill=(c - 60, c - 20, 255))
    for _ in range(10):
        x0, y0 = (int(v) for v in rng.integers(0, big, 2))
        w, h = (int(v) for v in rng.integers(big // 8, big // 3, 2))
        draw.ellipse([x0, y0, x0 + w, y0 + h], fill=(230, 60, 80), outline=(30, 20, 80), width=10)
    for _ in range(12):
        pts = [tuple(int(v) for v in rng.integers(0, big, 2)) for _ in range(3)]
        draw.line(pts, fill=(30, 20, 80), width=8, joint="curve")
    return img.filter(ImageFilter.GaussianBlur(1.2)).resize((size, size), Image.LANCZOS)


@unittest.skipUnless(
    HAVE_SPANDREL and os.environ.get("SAM3_RUN_FORGE_INTEGRATION_TESTS") == "1" and REAL_MODEL.is_file(),
    "실제 DeGrid 가중치 — SAM3_RUN_FORGE_INTEGRATION_TESTS=1 (파일: SAM3_DEGRID_MODEL 또는 models/ESRGAN 기본 경로)",
)
class RealWeightsTests(unittest.TestCase):
    """실제 v1.1 가중치(117 MB, CPU fp32). 잔차가 작고 결과가 원본과 미세하게만 다르다."""

    @classmethod
    def setUpClass(cls):
        cls.model = vdm.load_nafnet(REAL_MODEL)
        cls.rt = vdr.DegridRuntime(loader=lambda path: cls.model, forge_memory=lambda: None, logger=lambda m: None)
        cls.entry = vdm.ModelEntry(REAL_MODEL.stem, str(REAL_MODEL))

    def _delta(self, x, tile):
        with torch.inference_mode():
            return vd.tiled_residual(x, lambda t: self.model(t.float()), tile=tile)

    def test_file_is_found_as_nafnet(self):
        self.assertTrue(vdm.is_nafnet_file(REAL_MODEL))
        self.assertEqual(self.model.hyperparameters["width"], 32)
        self.assertEqual(self.model.hyperparameters["enc_blk_nums"], [2, 2, 4, 8])

    def test_output_is_a_small_residual_not_an_image(self):
        image = _synthetic_illustration()
        x, _ = vd.pil_to_tensor(image)
        delta = self._delta(x, 0)
        # CPU 실측(이 그림): |평균| 0.14/255 · 최대 4/255
        self.assertLess(float(delta.abs().mean()) * 255, 1.0)
        self.assertLess(float(delta.abs().max()) * 255, 16.0)
        # 이미지라면 입력과 거의 같아 상관이 1 에 가깝다
        corr = float(np.corrcoef(delta.flatten().numpy(), x.flatten().numpy())[0, 1])
        self.assertLess(abs(corr), 0.5)

    def test_result_differs_only_subtly(self):
        image = _synthetic_illustration()
        out = self.rt.run(image, self.entry, mode="full", strength=1.0, tile=512, device="cpu")
        a = np.asarray(image, np.int16)
        b = np.asarray(out.image, np.int16)
        self.assertLess(float(np.abs(a - b).mean()), 0.5)          # 실측 0.05 단계
        self.assertLess(int(np.abs(a - b).max()), 16)               # 실측 4
        # 깨끗한(VAE 를 거치지 않은) 그림은 거의 그대로 둔다(학습의 항등 배치) — CPU 실측 60.8 dB
        mse = float(((a - b).astype(np.float64) ** 2).mean())
        self.assertGreater(10 * math.log10(255 ** 2 / max(mse, 1e-12)), 50.0)

    def test_tiled_close_to_untiled(self):
        x, _ = vd.pil_to_tensor(_synthetic_illustration(384, seed=1))
        whole = self._delta(x, 0)
        tiled = self._delta(x, 256)
        diff = (whole - tiled).abs() * 255
        self.assertLess(float(diff.mean()), 0.25)     # 실측 0.03/255
        self.assertLess(float(diff.max()), 4.0)       # 실측 0.4/255 (Anima 생성 이미지 768² 에서는 최대 2/255)

    def test_generation_tab_first_use_survives_a_device_round_trip(self):
        # 기본 로더(spandrel)로 inference_mode 안에서 처음 불러 GPU 대역 왕복 — 두 번째 이미지까지 같은 결과
        rt = _RoundTripRuntime(forge_memory=lambda: None, logger=lambda m: None)
        image = _synthetic_illustration(128)
        with torch.inference_mode():
            first = rt.run(image, self.entry, tile=0, device="cpu")
        with torch.inference_mode():
            second = rt.run(image, self.entry, tile=0, device="cpu")
        self.assertEqual(first.image.tobytes(), second.image.tobytes())
        self.assertFalse(any(p.is_inference() for p in rt.model_for(self.entry.path).parameters()))

    def test_odd_size_edges_are_not_amplified(self):
        # 16 배수가 아닌 크기: 모델을 바로 부르면(안에서 0 채움) 오른쪽·아래 가장자리 잔차가 커진다 — 반사 패딩으로 안쪽과 비슷하게
        x, _ = vd.pil_to_tensor(_synthetic_illustration(256, seed=3).crop((3, 5, 253, 195)))   # 250x190
        with torch.inference_mode():
            raw = self.model(x)[0].abs().amax(0) * 255
            fn, _ = vdr.make_residual_fn(self.model, torch.device("cpu"), False)
            padded = fn(x)[0].abs().amax(0) * 255
        edge = torch.zeros_like(raw, dtype=torch.bool)
        edge[-4:, :] = True
        edge[:, -4:] = True
        self.assertGreater(float(raw[edge].max()), 3 * float(padded[edge].max()))
        self.assertLess(float(padded[edge].max()), max(4.0, 2 * float(padded[~edge].max())))

    def test_real_model_output_passes_the_residual_check(self):
        x, _ = vd.pil_to_tensor(_synthetic_illustration(256, seed=4))
        with torch.inference_mode():
            self.assertFalse(vd.output_looks_like_image(x, self.model(x)))

    @unittest.skipUnless(FORGE_VAE_PY.is_file(), "Forge 의 backend/patcher/vae.py 가 확장 옆에 없음")
    def test_real_model_tiles_like_comfyui(self):
        x, _ = vd.pil_to_tensor(_synthetic_illustration(384, seed=2))
        tiled_scale = _forge_tiled_scale()
        with torch.inference_mode():
            ours = vd.tiled_residual(x, lambda t: self.model(t.float()), tile=256)
            theirs = tiled_scale(x, lambda t: self.model(t.float()), tile_x=256, tile_y=256, overlap=32,
                                 upscale_amount=1, out_channels=3, output_device="cpu")
        self.assertTrue(torch.equal(ours, theirs))


def _screentone_image(size=512):
    """화면을 채운 4px 망점(2x2 검은 점) — 만화 스크린톤."""
    yy, xx = np.mgrid[:size, :size]
    return Image.fromarray((255 * ~(((yy % 4) < 2) & ((xx % 4) < 2))).astype(np.uint8)).convert("RGB")


def _checker_image(size=512):
    yy, xx = np.mgrid[:size, :size]
    return Image.fromarray((((yy + xx) % 2) * 255).astype(np.uint8)).convert("RGB")


def _pattern_image(fn, size=512):
    """화면을 채운 무늬 ``fn(yy, xx) → 0..255`` (회색, RGB)."""
    yy, xx = np.mgrid[:size, :size]
    return Image.fromarray(np.clip(fn(yy, xx), 0, 255).astype(np.uint8)).convert("RGB")


def _anima_crops(limit=8, size=512):
    """``SAM3_DEGRID_ANIMA_IMAGES``(PNG 파일·폴더, os.pathsep 구분)의 앞 ``limit`` 장 — 가로 가운데·세로 1/3 의 size² 조각."""
    files = []
    for item in filter(None, os.environ.get("SAM3_DEGRID_ANIMA_IMAGES", "").split(os.pathsep)):
        path = Path(item)
        files.extend(sorted(path.glob("*.png")) if path.is_dir() else [path] if path.is_file() else [])
    crops = []
    for path in files[:limit]:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            half = size // 2
            cx, cy = min(max(w // 2, half), w - half), min(max(h // 3, half), h - half)
            crops.append((path.name, im.crop((cx - half, cy - half, cx + half, cy + half))))
    return crops


class _RestoredImage(torch.nn.Module):
    """같은 DeGrid 망을 '복원 이미지' 를 내게 감싼 것(``x + 잔차``) — 이미지를 내는 일반 NAFNet 대역."""

    def __init__(self, residual_model):
        super().__init__()
        self.residual_model = residual_model

    def forward(self, x):
        return x + self.residual_model(x)


@unittest.skipUnless(
    HAVE_SPANDREL and os.environ.get("SAM3_RUN_FORGE_INTEGRATION_TESTS") == "1" and REAL_MODEL.is_file(),
    "실제 DeGrid 가중치 — SAM3_RUN_FORGE_INTEGRATION_TESTS=1 (파일: SAM3_DEGRID_MODEL 또는 models/ESRGAN 기본 경로)",
)
class RealWeightsResidualCheckTests(unittest.TestCase):
    """실제 DeGrid 가중치(v1.1 + 있으면 다른 파인튜닝)로 잔차 검사: 잔 스크린톤·1px 체커에서는 잔차가 25/255 를 넘게
    커지지만 입력과 반대로 움직이므로 적용하고(예전에는 크기만 보고 거절해 DeGrid 없이 저장), 같은 망이 복원 이미지를
    내면 어떤 입력에서도 거절한다. 화면을 채운 1px 줄무늬 같은 무늬에서 폭주하면(|평균| 100/255 초과) 그 이미지는 건너뛴다.
    CPU fp32 — 무늬는 512²(256² 스크린톤은 v1.1 잔차가 21/255 로 작아 예전 규칙도 통과)."""

    @classmethod
    def setUpClass(cls):
        cls.models = []
        for path in dict.fromkeys((REAL_MODEL, REAL_MODEL_ALT)):
            if path.is_file():
                cls.models.append((vdm.ModelEntry(path.stem, str(path)), vdm.load_nafnet(path)))

    def _runtime(self, model):
        return vdr.DegridRuntime(loader=lambda path: model, forge_memory=lambda: None, logger=lambda m: None)

    def test_screentone_and_checker_are_degridded_not_refused(self):
        for entry, model in self.models:
            rt = self._runtime(model)
            for name, image in (("screentone 4px", _screentone_image()), ("checker 1px", _checker_image())):
                with self.subTest(model=entry.name, input=name):
                    x, _ = vd.pil_to_tensor(image)
                    with torch.inference_mode():
                        fn, _ = vdr.make_residual_fn(model, torch.device("cpu"), False)
                        check = vd.check_residual(x, vd.tiled_residual(x, fn, tile=512))
                    # CPU 실측(512²) v1.1: 스크린톤 31.3/255 r -0.92, 체커 56.7/255 r -0.47 · Anzhc: 46.8 r -0.93, 47.3 r -0.53
                    self.assertGreater(check.mean_abs, vd.IMAGE_LIKE_ABS_MEAN)     # 크기만 보던 예전 규칙은 거절
                    self.assertLess(check.correlation, 0.0)
                    self.assertFalse(check.looks_like_image)
                    self.assertFalse(check.follows_input_mean)                    # 투영 비 실측 0.00~0.17
                    self.assertFalse(check.blew_up)                               # 폭주 문턱 100/255 아래
                    out = rt.run(image, entry, mode="full", strength=1.0, tile=512, device="cpu")
                    self.assertFalse(out.skipped)
                    self.assertGreater(out.stats["changed_pixels"], 0.2)

    def test_anima_like_illustration_is_accepted(self):
        x, _ = vd.pil_to_tensor(_synthetic_illustration(256, seed=4))
        for entry, model in self.models:
            with self.subTest(model=entry.name), torch.inference_mode():
                check = vd.check_residual(x, model(x))
                self.assertFalse(check.looks_like_image)
                self.assertFalse(check.blew_up)

    def test_anima_crops_are_applied(self):
        # 실제 Anima 생성 이미지(저장소에 넣지 않음) — SAM3_DEGRID_ANIMA_IMAGES 에 PNG 파일·폴더(os.pathsep 구분)를 주면
        # 앞 8장의 512² 조각(가로 가운데·세로 1/3)으로 돈다. CPU 실측: 잔차 |평균| 0.3~0.6/255.
        crops = _anima_crops()
        if not crops:
            self.skipTest("SAM3_DEGRID_ANIMA_IMAGES 가 없음(실제 Anima PNG 파일·폴더)")
        for entry, model in self.models:
            rt = self._runtime(model)
            for name, crop in crops:
                with self.subTest(model=entry.name, image=name):
                    with torch.inference_mode():
                        out = rt.run(crop, entry, mode="full", strength=1.0, tile=512, device="cpu")
                    self.assertFalse(out.skipped)
                    self.assertLess(out.stats["abs_mean_255"], 2.0)

    def test_degenerate_full_frame_patterns_are_skipped_as_blow_ups(self):
        # 화면을 채운 1px 줄무늬·3px 세로줄·저대비 1px 줄무늬 — 실제 DeGrid 가 |평균| 183~2143/255 로 폭주(적용하면 PSNR 4~8 dB).
        # 1px 가로줄은 평균이 입력 밝기 쪽(+89/255, 투영 비 +0.70)이지만 양쪽으로 흔들려(부호 쏠림 0.25~0.33) 이미지로 보지 않고
        # 폭주로 건너뛴다. 저대비(118/138) 1px 체커는 v1.1 만 폭주(183.1/255), Anzhc 는 23.8/255 로 적용.
        blow_ups = [("vstripes 1px", _pattern_image(lambda yy, xx: (xx % 2) * 255)),
                    ("hstripes 1px", _pattern_image(lambda yy, xx: (yy % 2) * 255)),
                    ("vstripes 3px", _pattern_image(lambda yy, xx: ((xx // 3) % 2) * 255)),
                    ("vstripes 1px 118/138", _pattern_image(lambda yy, xx: 118 + (xx % 2) * 20))]
        low_contrast_checker = _pattern_image(lambda yy, xx: 118 + ((yy + xx) % 2) * 20)
        checker_blows_up = {REAL_MODEL_V11_STEM: True, REAL_MODEL_ANZHC_STEM: False}   # 다른 파일이면 체커는 보지 않음
        for entry, model in self.models:
            rt = self._runtime(model)
            cases = [(name, image, True) for name, image in blow_ups]
            if entry.name in checker_blows_up:
                cases.append(("checker 1px 118/138", low_contrast_checker, checker_blows_up[entry.name]))
            for name, image, blows_up in cases:
                with self.subTest(model=entry.name, input=name):
                    before = image.tobytes()
                    if blows_up:
                        with torch.inference_mode(), self.assertRaises(vdr.ResidualBlowUpError) as caught:
                            rt.run(image, entry, tile=512, device="cpu")
                        self.assertTrue(str(caught.exception).startswith("output blew up (mean |residual| "))
                        self.assertEqual(image.tobytes(), before)
                    else:
                        with torch.inference_mode():
                            out = rt.run(image, entry, tile=512, device="cpu")
                        self.assertLess(out.stats["abs_mean_255"], vd.RESIDUAL_BLOWUP_ABS_MEAN * 255)

    def test_grain_is_degridded_but_image_models_on_it_are_refused(self):
        # 같은 회색 입자 바탕(200±4): 실제 DeGrid 는 거의 그대로(잔차 |평균| 0.05~0.6/255) 적용, 흐림·median 이미지 모델은 거절
        image = _grain_image(256)
        for entry, model in self.models:
            with self.subTest(model=entry.name):
                with torch.inference_mode():
                    out = self._runtime(model).run(image, entry, tile=512, device="cpu")
                self.assertLess(out.stats["abs_mean_255"], 2.0)
        for name, stand_in in (("gaussian s2", _BlurImageModel(2.0)), ("median3", _Median3ImageModel())):
            with self.subTest(image_model=name):
                with torch.inference_mode(), self.assertRaises(vdr.NotResidualModelError):
                    self._runtime(stand_in).run(image, self.models[0][0], tile=512, device="cpu")

    def test_same_network_returning_the_image_is_refused(self):
        for entry, model in self.models:
            rt = self._runtime(_RestoredImage(model))
            for name, image in (("illustration", _synthetic_illustration(256, seed=5)),
                                ("screentone 4px", _screentone_image()), ("checker 1px", _checker_image())):
                with self.subTest(model=entry.name, input=name):
                    with self.assertRaises(vdr.NotResidualModelError) as caught:
                        rt.run(image, entry, tile=512, device="cpu")
                    self.assertIn("output does not look like a residual", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
