"""효율 감사(2026-09-23) [8] SAM3 마스크 후처리: hull 은 findContours 한 번, edge-aware 는 bbox ROI(Canny 는 run 안에서
한 번), dilation 은 bbox ROI. 전부 옛 구현과 픽셀 단위로 같아야 한다.

아래 `_ref_*` 는 바꾸기 전 sam3ext/core.py 의 구현을 그대로 옮긴 참조 구현이다(고치지 말 것).
무작위 마스크 수백 개(중첩 성분, 가장자리에 닿는 성분, 빈 마스크, 1픽셀, 이미지보다 큰 반경 포함)에서
np.array_equal 로 비교한다. GPU·모델 없이 cv2 만 쓴다.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

core = importlib.import_module("sam3ext.core")


# ---------------------------------------------------------------------------
# 참조 구현(바꾸기 전 core.py 그대로)
# ---------------------------------------------------------------------------


def _ref_dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    import cv2

    k = 2 * int(px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return dilated.astype(bool)


def _ref_edge_aware_dilate(mask: np.ndarray, image_rgb: np.ndarray, max_px: int, canny_low: int = 100, canny_high: int = 200) -> np.ndarray:
    if max_px <= 0 or image_rgb is None:
        return mask.astype(bool)
    import cv2

    if image_rgb.ndim == 3 and image_rgb.shape[2] >= 3:
        gray = cv2.cvtColor(image_rgb[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, kernel) > 0  # bool, thickened

    out = mask.astype(bool)
    for _ in range(int(max_px)):
        dilated = cv2.dilate(out.astype(np.uint8), kernel) > 0
        new_pixels = dilated & ~out & ~edges
        if not new_pixels.any():
            break
        out = out | new_pixels
    return out


def _ref_convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    import cv2

    mask_u8 = mask.astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(mask_u8)
    out = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull = cv2.convexHull(np.vstack(contours))
        cv2.fillPoly(out, [hull], 255)
    return (out > 0).astype(bool)


# ---------------------------------------------------------------------------
# 무작위 마스크·이미지
# ---------------------------------------------------------------------------


def _random_shape(rng: np.random.Generator) -> tuple[int, int]:
    kind = rng.integers(0, 10)
    if kind == 0:
        return 1, int(rng.integers(1, 40))
    if kind == 1:
        return int(rng.integers(1, 40)), 1
    if kind == 2:
        return int(rng.integers(1, 6)), int(rng.integers(1, 6))
    return int(rng.integers(8, 110)), int(rng.integers(8, 110))


def _random_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    kind = int(rng.integers(0, 9))
    m = np.zeros((h, w), dtype=np.uint8)
    if kind == 0:  # 빈 마스크
        return m.astype(bool)
    if kind == 1:  # 1픽셀(모서리·가장자리 포함)
        y = int(rng.choice([0, h - 1, rng.integers(0, h)]))
        x = int(rng.choice([0, w - 1, rng.integers(0, w)]))
        m[y, x] = 1
        return m.astype(bool)
    if kind == 2:  # 점 잡음 — 대각선으로만 닿는 성분이 많다(8-연결 확인)
        p = float(rng.uniform(0.02, 0.45))
        return rng.random((h, w)) < p
    if kind == 3:  # 중첩 성분: 고리 + 구멍 안의 성분(+ 그 안의 고리)
        cy, cx = h // 2, w // 2
        r = max(2, min(h, w) // 2)
        cv2.circle(m, (cx, cy), r, 1, -1)
        cv2.circle(m, (cx, cy), max(1, int(r * 0.7)), 0, -1)
        cv2.circle(m, (cx, cy), max(0, int(r * 0.4)), 1, -1)
        cv2.circle(m, (cx, cy), max(0, int(r * 0.25)), 0, -1)
        if rng.random() < 0.5:
            m[cy, cx] = 1
        if rng.random() < 0.5:  # 오목한 구멍(C 자) 안의 성분
            m[max(0, cy - 1):cy + 2, cx:] = 0
        return m.astype(bool)
    if kind == 4:  # 가장자리에 닿는 성분
        side = int(rng.integers(0, 4))
        t = int(rng.integers(1, max(2, min(h, w) // 3 + 1)))
        if side == 0:
            m[:t, :] = 1
        elif side == 1:
            m[-t:, :] = 1
        elif side == 2:
            m[:, :t] = 1
        else:
            m[:, -t:] = 1
        m[rng.random((h, w)) < 0.05] = 1
        # 가장자리 행/열로만 이어진 성분
        m[0, :: max(1, int(rng.integers(1, 4)))] = 1
        return m.astype(bool)
    if kind == 5:  # 꽉 찬 마스크
        return np.ones((h, w), dtype=bool)
    # 도형 여러 개(타원·사각형·선, 일부는 이미지 밖으로 걸침)
    for _ in range(int(rng.integers(1, 12))):
        s = int(rng.integers(0, 3))
        y, x = int(rng.integers(-5, h + 5)), int(rng.integers(-5, w + 5))
        if s == 0:
            ax = (int(rng.integers(0, max(1, w // 2))), int(rng.integers(0, max(1, h // 2))))
            cv2.ellipse(m, (x, y), ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        elif s == 1:
            y2, x2 = int(rng.integers(-5, h + 5)), int(rng.integers(-5, w + 5))
            cv2.rectangle(m, (x, y), (x2, y2), 1, -1)
        else:
            y2, x2 = int(rng.integers(-5, h + 5)), int(rng.integers(-5, w + 5))
            cv2.line(m, (x, y), (x2, y2), 1, int(rng.integers(1, 3)))
    if rng.random() < 0.4:  # 구멍 뚫기
        for _ in range(int(rng.integers(1, 5))):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            cv2.circle(m, (x, y), int(rng.integers(1, max(2, min(h, w) // 4))), 0, -1)
    return m.astype(bool)


def _random_image(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    kind = int(rng.integers(0, 6))
    if kind == 0:  # 잡음 — 가장자리가 아주 많음
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    if kind == 1:  # 가장자리 없음
        return np.full((h, w, 3), int(rng.integers(0, 256)), dtype=np.uint8)
    if kind == 2:  # 흑백 2D
        img = np.zeros((h, w), dtype=np.uint8)
        for _ in range(int(rng.integers(1, 6))):
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 40)), int(rng.integers(0, 256)), -1)
        return img
    img = np.full((h, w, 3), int(rng.integers(0, 256)), dtype=np.uint8)
    for _ in range(int(rng.integers(1, 8))):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 50)), color, -1)
        else:
            cv2.rectangle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), (int(rng.integers(0, w)), int(rng.integers(0, h))), color, -1)
    if kind == 3:  # RGBA
        alpha = rng.integers(0, 256, (h, w, 1), dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    if kind == 4:
        img = cv2.GaussianBlur(img, (0, 0), 1.5)
    return img


def _radius(rng: np.random.Generator) -> int:
    r = float(rng.random())
    if r < 0.1:
        return 0
    if r < 0.5:
        return int(rng.integers(1, 6))
    if r < 0.85:
        return int(rng.integers(6, 40))
    return int(rng.choice([64, 128, 256]))  # 이미지보다 큰 반경


class _Base(unittest.TestCase):
    def assertSameMask(self, got: np.ndarray, want: np.ndarray, msg: str = ""):
        self.assertEqual(got.dtype, np.bool_, msg)
        self.assertEqual(got.shape, want.shape, msg)
        self.assertTrue(np.array_equal(got, want), f"{msg}: {int((got != want).sum())} px differ")


class ConvexHullMaskTests(_Base):
    def test_random_masks_match_reference(self):
        rng = np.random.default_rng(1234)
        for i in range(500):
            h, w = _random_shape(rng)
            mask = _random_mask(rng, h, w)
            if i % 7 == 0:
                mask = mask.astype(np.uint8) * int(rng.choice([1, 255]))  # bool 이 아닌 입력
            want = _ref_convex_hull_mask(mask)
            got = core._convex_hull_mask(mask)
            self.assertSameMask(got, want, f"case {i} {h}x{w}")

    def test_nested_components_are_each_hulled(self):
        m = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(m, (40, 40), 35, 1, -1)
        cv2.circle(m, (40, 40), 28, 0, -1)
        m[:, 38:43][m[:, 38:43] > 0] = 0  # 고리를 끊어 C 자로 — 구멍이 오목해진다
        cv2.circle(m, (40, 40), 3, 1, -1)
        m[40, 60] = 1
        mask = m.astype(bool)
        self.assertSameMask(core._convex_hull_mask(mask), _ref_convex_hull_mask(mask), "nested")

    def test_does_not_alias_input(self):
        mask = np.zeros((10, 10), dtype=bool)
        out = core._convex_hull_mask(mask)
        out[0, 0] = True
        self.assertFalse(mask[0, 0])


class DilateMaskTests(_Base):
    def test_random_masks_match_reference(self):
        rng = np.random.default_rng(99)
        for i in range(400):
            h, w = _random_shape(rng)
            mask = _random_mask(rng, h, w)
            px = _radius(rng)
            if i % 9 == 0:
                mask = mask.astype(np.uint8)
            want = _ref_dilate_mask(mask, px)
            got = core._dilate_mask(mask, px)
            self.assertSameMask(got, want, f"case {i} {h}x{w} px={px}")

    def test_large_frame_large_radius(self):
        rng = np.random.default_rng(7)
        for px in (1, 17, 64, 256):
            mask = np.zeros((300, 260), dtype=bool)
            cv2.circle(mask.view(np.uint8), (200, 60), 20, 1, -1)
            mask[299, 0] = True
            mask |= rng.random(mask.shape) < 0.0005
            self.assertSameMask(core._dilate_mask(mask, px), _ref_dilate_mask(mask, px), f"px={px}")

    def test_frame_above_cv_parallel_threshold(self):
        # 프레임이 2^20 픽셀 이상이면 스레드 수에 따라 ROI 또는 전체 프레임(예전 경로)을 고른다 — 둘 다 옛 구현과 같아야 한다
        h, w = 1030, 1024
        self.assertGreaterEqual(h * w, core._CV_PARALLEL_MIN_PIXELS)
        cases = ((3, (5, 5)), (21, (500, 1010)), (60, (1020, 3)), (9, (515, 512)))
        for threads in (1, 10_000):  # 1: ROI, 10000: 전체 프레임
            with mock.patch.object(cv2, "getNumThreads", return_value=threads):
                for px, (cy, cx) in cases:
                    mask = np.zeros((h, w), dtype=bool)
                    cv2.circle(mask.view(np.uint8), (cx, cy), 12, 1, -1)
                    mask[min(h - 1, cy + 30), max(0, cx - 40)] = True
                    self.assertSameMask(
                        core._dilate_mask(mask, px), _ref_dilate_mask(mask, px), f"threads={threads} px={px} at {(cy, cx)}"
                    )

    def test_non_bool_dtypes_keep_old_uint8_cast(self):
        # 예전 구현은 astype(np.uint8) 로 바꿨다 — 0.5 는 0, 256 은 0 으로 떨어진다. ROI 경로도 같아야 한다.
        rng = np.random.default_rng(11)
        for i in range(60):
            h, w = int(rng.integers(5, 60)), int(rng.integers(5, 60))
            base = _random_mask(rng, h, w)
            if i % 3 == 0:
                mask = base.astype(np.float32) * rng.choice([0.5, 1.0, 2.0], size=(h, w)).astype(np.float32)
            elif i % 3 == 1:
                mask = base.astype(np.int32) * rng.choice([1, 256, 300], size=(h, w)).astype(np.int32)
            else:
                mask = base.astype(np.uint8) * 255
            px = int(rng.integers(1, 12))
            self.assertSameMask(core._dilate_mask(mask, px), _ref_dilate_mask(mask, px), f"case {i} {mask.dtype}")

    def test_roi_choice_rule(self):
        limit = core._CV_PARALLEL_MIN_PIXELS
        small_frame = (1000, 1000)  # 문턱 미만 — 둘 다 1스레드라 항상 ROI
        self.assertTrue(core._cv_roi_is_cheaper((0, 1000, 0, 1000), small_frame, 24))
        big = (1536, 1536)
        self.assertTrue(core._cv_roi_is_cheaper((0, 300, 0, 300), big, 24))  # 90k*24 <= 2.36M
        self.assertFalse(core._cv_roi_is_cheaper((0, 368, 0, 368), big, 24))  # 135k*24 > 2.36M
        self.assertTrue(core._cv_roi_is_cheaper((0, 368, 0, 368), big, 4))
        self.assertFalse(core._cv_roi_is_cheaper((0, 1024, 0, 1024), big, 1))  # 병렬 영역 ROI 는 예측 불가 → 전체
        self.assertFalse(core._cv_roi_is_cheaper((0, 1536, 0, 1536), big, 1))
        self.assertTrue(core._cv_roi_is_cheaper((0, 1, 0, 1), big, 0))
        self.assertLess(1023 * 1024, limit)

    def test_does_not_alias_input(self):
        for px in (0, 3):
            mask = np.zeros((10, 10), dtype=bool)
            mask[5, 5] = True
            out = core._dilate_mask(mask, px)
            out[0, 0] = True
            self.assertFalse(mask[0, 0])
            empty = np.zeros((10, 10), dtype=bool)
            out = core._dilate_mask(empty, px)
            out[0, 0] = True
            self.assertFalse(empty[0, 0])


class EdgeAwareDilateTests(_Base):
    def test_random_masks_match_reference(self):
        rng = np.random.default_rng(2026)
        for i in range(400):
            h, w = _random_shape(rng)
            mask = _random_mask(rng, h, w)
            img = _random_image(rng, h, w)
            px = _radius(rng)
            thresholds = [(100, 200), (30, 90), (250, 400)][i % 3]
            want = _ref_edge_aware_dilate(mask, img, px, *thresholds)
            got = core._edge_aware_dilate(mask, img, px, *thresholds)
            self.assertSameMask(got, want, f"case {i} {h}x{w} px={px}")
            # run_sam3 처럼 두꺼운 Canny 를 미리 한 번 계산해 넘겨도 같아야 한다
            edges = core._edge_aware_edges(img, *thresholds)
            got2 = core._edge_aware_dilate(mask, img, px, *thresholds, edges=edges)
            self.assertSameMask(got2, want, f"case {i} cached edges")

    def test_none_image_and_zero_px(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[3:6, 3:6] = 1
        self.assertSameMask(core._edge_aware_dilate(mask, None, 5), _ref_edge_aware_dilate(mask, None, 5))
        img = np.zeros((12, 12, 3), dtype=np.uint8)
        self.assertSameMask(core._edge_aware_dilate(mask, img, 0), _ref_edge_aware_dilate(mask, img, 0))

    def test_face_sized_mask_in_large_frame(self):
        rng = np.random.default_rng(5)
        img = np.full((400, 360, 3), 200, dtype=np.uint8)
        cv2.ellipse(img, (180, 200), (120, 160), 0, 0, 360, (120, 80, 60), -1)
        img = cv2.GaussianBlur(img, (0, 0), 2)
        img = np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
        for px in (4, 16, 64):
            face = np.zeros((400, 360), dtype=np.uint8)
            cv2.circle(face, (180, 120), 40, 1, -1)
            face = face.astype(bool)
            self.assertSameMask(core._edge_aware_dilate(face, img, px), _ref_edge_aware_dilate(face, img, px), f"px={px}")

    def test_does_not_alias_input(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = core._edge_aware_dilate(mask, img, 3)
        self.assertTrue(out[4, 4])
        self.assertFalse(mask[4, 4])


class RunSam3CannyOnceTests(unittest.TestCase):
    """run_sam3_on_pil 이 그룹마다 Canny 를 다시 돌리지 않고, 결과는 그룹마다 옛 구현과 같은지."""

    def test_edges_computed_once_per_run_and_groups_match_reference(self):
        h, w = 64, 80
        rgb = np.full((h, w, 3), 180, dtype=np.uint8)
        cv2.rectangle(rgb, (10, 10), (50, 40), (30, 60, 90), -1)
        cv2.circle(rgb, (65, 45), 10, (250, 250, 250), -1)
        pil = Image.fromarray(rgb)

        m1 = np.zeros((h, w), dtype=bool)
        m1[15:25, 15:25] = True
        m2 = np.zeros((h, w), dtype=bool)
        m2[42:48, 60:70] = True
        per_prompt = {"a": [m1], "b": [m2]}

        class _Proc:
            def set_confidence_threshold(self, t):
                pass

            def set_image(self, image):
                return {"backbone_out": {}}

            def set_text_prompt(self, prompt, state):
                state["masks"] = np.stack(per_prompt[prompt])[:, None].astype(np.float32)
                state["boxes"] = np.zeros((0, 4), dtype=np.float32)
                state["scores"] = np.zeros((0,), dtype=np.float32)
                return state

        calls = []
        real_edges = core._edge_aware_edges

        def _counting_edges(*a, **k):
            calls.append(1)
            return real_edges(*a, **k)

        with mock.patch.object(core, "resolve_checkpoint_path", return_value=None), \
             mock.patch.object(core, "_get_model_bundle", return_value=(None, _Proc())), \
             mock.patch.object(core, "_edge_aware_edges", side_effect=_counting_edges):
            result = core.run_sam3_on_pil(
                pil, "a / b", 0.5, "", "cpu",
                mask_dilation=2, mask_hull=True, mask_outline_px=6,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result.masks), 2)
        for got_img, src in zip(result.masks, (m1, m2)):
            want = _ref_dilate_mask(_ref_edge_aware_dilate(_ref_convex_hull_mask(src), rgb, 6), 2)
            got = np.asarray(got_img.convert("L")) > 127
            self.assertTrue(np.array_equal(got, want))


if __name__ == "__main__":
    unittest.main()
