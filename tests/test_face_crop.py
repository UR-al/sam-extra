"""SAM3 '얼굴 안에서만 찾기' — 순수 기하(sam3ext/face_crop.py)와 얼굴 검출기 연결(sam3ext/face_detect.py).

GPU·모델·다운로드 없이: crop 기하는 numpy 만, 검출기는 가짜 ultralytics 모듈로 본다.
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import face_crop as fc  # noqa: E402
from sam3ext import face_detect as fd  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# 상자 → crop
# ---------------------------------------------------------------------------------------------------------------------


class ExpandBoxTests(unittest.TestCase):
    def test_sides_and_bottom_20_percent_top_40_percent(self):
        self.assertEqual(fc.expand_box((100, 100, 200, 150)), (80.0, 80.0, 220.0, 160.0))
        self.assertEqual((fc.EXPAND_SIDE, fc.EXPAND_TOP, fc.EXPAND_BOTTOM), (0.20, 0.40, 0.20))

    def test_ratios_are_parameters(self):
        self.assertEqual(fc.expand_box((0, 0, 10, 10), side=0, top=0, bottom=0), (0.0, 0.0, 10.0, 10.0))
        self.assertEqual(fc.expand_box((10, 10, 20, 30), side=0.5, top=1.0, bottom=0.0), (5.0, -10.0, 25.0, 30.0))


class SquareCropTests(unittest.TestCase):
    def assertInside(self, crop, size):
        width, height = size
        self.assertTrue(0 <= crop.x0 < crop.x1 <= width and 0 <= crop.y0 < crop.y1 <= height, crop)

    def test_centred_square_in_the_middle(self):
        crop = fc.face_crop_box((100, 100, 200, 200), (512, 512))
        # 넓힌 상자 x 80..220, y 60..220 → 한 변 160, 중심 (150, 140)
        self.assertEqual(crop, fc.CropBox(70, 60, 230, 220))
        self.assertEqual((crop.width, crop.height), (160, 160))

    def test_crops_are_integer_pil_boxes(self):
        crop = fc.face_crop_box((10.4, 20.6, 50.2, 61.9), (300, 200))
        for value in crop.as_tuple():
            self.assertIsInstance(value, int)
        image = Image.new("RGB", (300, 200))
        self.assertEqual(image.crop(crop.as_tuple()).size, (crop.width, crop.height))

    def test_corner_faces_are_pushed_inside_and_stay_square(self):
        size = (400, 300)
        for box in ((0, 0, 60, 60), (340, 240, 400, 300), (0, 240, 60, 300), (340, 0, 400, 60), (-30, -30, 40, 40)):
            with self.subTest(box=box):
                crop = fc.face_crop_box(box, size)
                self.assertInside(crop, size)
                self.assertEqual(crop.width, crop.height, "이미지보다 작으면 정사각형을 지킨다(밀어 넣음, 자르지 않음)")
                clipped = fc.clip_box(box, size)
                self.assertTrue(crop.x0 <= clipped[0] and crop.y0 <= clipped[1])
                self.assertTrue(crop.x1 >= clipped[2] and crop.y1 >= clipped[3], "얼굴 상자 전체가 crop 안")

    def test_tiny_face_gets_the_minimum_side(self):
        crop = fc.face_crop_box((200, 150, 203, 152), (400, 300))
        self.assertEqual((crop.width, crop.height), (fc.MIN_CROP_SIDE, fc.MIN_CROP_SIDE))
        self.assertTrue(crop.contains(201.5, 151))

    def test_tiny_image_is_used_whole(self):
        crop = fc.face_crop_box((5, 5, 10, 10), (20, 12))
        self.assertEqual(crop, fc.CropBox(0, 0, 20, 12), "한 변이 이미지보다 길면 그 축은 이미지 전체")

    def test_huge_face_is_clamped_to_the_image(self):
        crop = fc.face_crop_box((10, 10, 500, 290), (512, 300))
        self.assertEqual(crop, fc.CropBox(0, 0, 512, 300))
        wide = fc.face_crop_box((100, 20, 300, 280), (1000, 300))
        self.assertEqual((wide.y0, wide.y1), (0, 300), "세로만 이미지 크기로")
        self.assertGreater(wide.width, wide.height)
        self.assertInside(wide, (1000, 300))

    def test_unusable_boxes_are_dropped(self):
        size = (100, 100)
        for box in ((10, 10, 10, 40), (200, 200, 300, 300), (-50, -50, -1, -1), (0, 0, float("nan"), 10),
                    (0, 0, float("inf"), 10), ("a", 0, 1, 1), (1, 2, 3)):
            with self.subTest(box=box):
                self.assertIsNone(fc.face_crop_box(box, size))
        self.assertIsNone(fc.square_crop((0, 0, 10, 10), (0, 10)))

    def test_swapped_corners_are_normalised(self):
        self.assertEqual(fc.face_crop_box((200, 200, 100, 100), (512, 512)), fc.face_crop_box((100, 100, 200, 200), (512, 512)))

    def test_random_boxes_always_give_an_inside_crop_containing_the_face(self):
        rng = np.random.default_rng(7)
        for _ in range(500):
            width, height = int(rng.integers(8, 900)), int(rng.integers(8, 900))
            x0, y0 = rng.uniform(-50, width), rng.uniform(-50, height)
            box = (x0, y0, x0 + rng.uniform(0.5, width), y0 + rng.uniform(0.5, height))
            crop = fc.face_crop_box(box, (width, height))
            clipped = fc.clip_box(box, (width, height))
            if clipped is None:
                self.assertIsNone(crop)
                continue
            self.assertInside(crop, (width, height))
            self.assertTrue(crop.x0 <= clipped[0] + 1e-9 and crop.y0 <= clipped[1] + 1e-9, (box, crop))
            self.assertTrue(crop.x1 >= clipped[2] - 1e-9 and crop.y1 >= clipped[3] - 1e-9, (box, crop))
            if crop.width < width and crop.height < height:
                self.assertEqual(crop.width, crop.height)


class TransformTests(unittest.TestCase):
    def test_box_round_trip_is_a_translation(self):
        crop = fc.CropBox(30, 40, 94, 104)
        self.assertEqual(fc.box_to_image((1, 2, 11, 12), crop), [31.0, 42.0, 41.0, 52.0])
        self.assertEqual(fc.box_to_crop(fc.box_to_image((1.5, 2, 11, 12.25), crop), crop), [1.5, 2.0, 11.0, 12.25])

    def test_paste_mask_is_bit_exact(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            height, width = int(rng.integers(4, 80)), int(rng.integers(4, 80))
            x0, y0 = int(rng.integers(0, width - 1)), int(rng.integers(0, height - 1))
            x1, y1 = int(rng.integers(x0 + 1, width + 1)), int(rng.integers(y0 + 1, height + 1))
            crop = fc.CropBox(x0, y0, x1, y1)
            piece = rng.random((crop.height, crop.width)) < 0.4
            full = fc.paste_mask(piece, crop, (height, width))
            self.assertEqual(full.dtype, np.bool_)
            self.assertEqual(full.shape, (height, width))
            np.testing.assert_array_equal(full[y0:y1, x0:x1], piece)
            outside = full.copy()
            outside[y0:y1, x0:x1] = False
            self.assertFalse(outside.any())

    def test_paste_resizes_a_different_size_mask_nearest(self):
        crop = fc.CropBox(2, 2, 6, 6)
        piece = np.zeros((2, 2), dtype=bool)
        piece[0, 0] = True
        full = fc.paste_mask(piece, crop, (8, 8))
        expected = np.zeros((8, 8), dtype=bool)
        expected[2:4, 2:4] = True
        np.testing.assert_array_equal(full, expected)

    def test_paste_refuses_a_crop_outside_the_image_or_a_bad_mask(self):
        with self.assertRaises(ValueError):
            fc.paste_mask(np.ones((4, 4), bool), fc.CropBox(6, 6, 10, 10), (8, 8))
        with self.assertRaises(ValueError):
            fc.paste_mask(np.ones((2, 4, 4), bool), fc.CropBox(0, 0, 4, 4), (8, 8))

    def test_centroid(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:4, 6:8] = True
        self.assertEqual(fc.mask_centroid(mask), (7.0, 3.0))
        self.assertIsNone(fc.mask_centroid(np.zeros((3, 3), bool)))


class OwnershipTests(unittest.TestCase):
    """겹친 crop 에서 찾은 조각은 가장 가까운 얼굴(얼굴 크기로 나눈 거리)의 것."""

    def setUp(self):
        self.faces = [[40, 70, 80, 110], [84, 70, 124, 110]]
        self.crops = [fc.face_crop_box(box, (256, 192)) for box in self.faces]

    def test_crops_overlap_in_this_setup(self):
        a, b = self.crops
        self.assertLess(b.x0, a.x1)

    def test_piece_of_the_neighbour_goes_to_the_neighbour(self):
        self.assertEqual(fc.assign_owner((90.0, 82.0), self.faces, self.crops, source=0), 1)
        self.assertEqual(fc.assign_owner((50.0, 82.0), self.faces, self.crops, source=0), 0)
        self.assertEqual(fc.assign_owner((110.0, 82.0), self.faces, self.crops, source=1), 1)

    def test_only_faces_whose_crop_holds_the_point_compete(self):
        faces = [[0, 0, 10, 10], [200, 0, 210, 10]]
        crops = [fc.CropBox(0, 0, 40, 40), fc.CropBox(180, 0, 220, 40)]
        # 오른쪽 얼굴 중심에 더 가깝지만 그 crop 밖 → 찾은 crop(왼쪽)의 것
        self.assertEqual(fc.assign_owner((39.0, 5.0), faces, crops, source=0), 0)

    def test_distance_is_scaled_by_face_size(self):
        faces = [[0, 0, 100, 100], [110, 40, 120, 50]]          # 큰 얼굴, 작은 얼굴
        crops = [fc.CropBox(0, 0, 140, 140), fc.CropBox(0, 0, 140, 140)]
        # 큰 얼굴 중심(50,50)까지 55, 작은 얼굴 중심(115,45)까지 ~10 → 크기로 나누면 0.55 vs 1.0 → 큰 얼굴
        self.assertEqual(fc.assign_owner((105.0, 50.0), faces, crops, source=1), 0)

    def test_ties_keep_the_source(self):
        faces = [[0, 0, 10, 10], [0, 0, 10, 10]]
        crops = [fc.CropBox(0, 0, 32, 32), fc.CropBox(0, 0, 32, 32)]
        self.assertEqual(fc.assign_owner((5.0, 5.0), faces, crops, source=1), 1)
        self.assertEqual(fc.assign_owner((5.0, 5.0), faces, crops, source=0), 0)

    def test_a_group_of_sources_are_all_candidates_and_win_ties(self):
        faces = [[0, 0, 10, 10], [200, 0, 210, 10], [0, 0, 10, 10]]
        crops = [fc.CropBox(0, 0, 40, 40), fc.CropBox(180, 0, 220, 40), fc.CropBox(0, 0, 40, 40)]
        # 점(150, 5)은 어느 crop 에도 없다 — 찾은 crop 들(0, 1)만 후보, 오른쪽 얼굴이 가깝다
        self.assertEqual(fc.assign_owner((150.0, 5.0), faces, crops, source=[0, 1]), 1)
        self.assertEqual(fc.assign_owner((5.0, 5.0), faces, crops, source={2}), 2, "같으면 찾은 crop")
        self.assertEqual(fc.assign_owner((5.0, 5.0), faces, crops, source=np.int64(0)), 0)


def _box_mask(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class CutPieceTests(unittest.TestCase):
    """crop 의 안쪽 가장자리에 잘린 조각은 옆 crop 에서 이어지는 조각과 한 물체로 묶어 주인을 정한다."""

    SHAPE = (300, 400)   # (높이, 폭)

    def setUp(self):
        # 큰 얼굴 A(100 px)와 그 오른쪽의 작은 얼굴 B(24 px) — crop 이 맞닿는다(A 는 x < 230, B 는 x ≥ 230)
        self.faces = [[100, 100, 200, 200], [238, 150, 262, 174]]
        self.crops = [fc.face_crop_box(box, (400, 300)) for box in self.faces]

    def test_crops_touch_in_this_setup(self):
        a, b = self.crops
        self.assertEqual((a.x1, b.x0), (230, 230))

    def test_edge_contact_is_the_line_just_outside_inner_edges(self):
        crop = fc.CropBox(10, 20, 30, 40)
        piece = _box_mask((50, 50), 10, 25, 12, 28)          # 왼쪽 변(x 10)에 닿음
        (y0, y1, x0, x1), where = fc.edge_contact(piece, crop)
        self.assertEqual((y0, y1, x0, x1), (20, 40, 9, 10))
        self.assertEqual(np.nonzero(where[:, 0])[0].tolist(), list(range(4, 9)), "닿은 줄 ±1(대각선 이웃 포함)")
        self.assertIsNone(fc.edge_contact(_box_mask((50, 50), 15, 25, 20, 30), crop), "변에 닿지 않음")
        image_edge = fc.CropBox(0, 0, 30, 40)
        self.assertIsNone(fc.edge_contact(_box_mask((50, 50), 0, 5, 3, 9), image_edge), "이미지 테두리는 자른 변이 아니다")
        corner = fc.edge_contact(_box_mask((50, 50), 27, 37, 30, 40), crop)   # 오른쪽 · 아래 두 변
        (y0, y1, x0, x1), where = corner
        self.assertEqual((y0, y1, x0, x1), (20, 41, 10, 31))
        self.assertTrue(where[40 - y0, 27 - x0] and where[38 - y0, 30 - x0])
        self.assertFalse(where[40 - y0, 20 - x0], "닿지 않은 곳은 비어 있다")

    def test_a_neighbours_part_cut_at_this_crops_edge_goes_to_the_neighbour(self):
        hair_a = np.zeros(self.SHAPE, dtype=bool)
        hair_a[70:100, 95:234] = True
        hair_a[70:230, 200:234] = True          # A 의 머리카락 — B 의 crop 으로 x 233 까지 들어간 한 덩어리
        a, b = self.crops
        seen_a = hair_a & _box_mask(self.SHAPE, *a.as_tuple())
        seen_b = hair_a & _box_mask(self.SHAPE, *b.as_tuple())   # B 의 crop 이 본 띠(x 230..233)
        own_b = _box_mask(self.SHAPE, 240, 146, 260, 150)
        self.assertFalse(a.contains(*fc.mask_centroid(seen_b)), "띠의 무게중심은 A 의 crop 밖 — 예전 규칙이면 B 의 것")
        owners = fc.assign_owners([seen_a, seen_b, own_b], [0, 1, 1], self.faces, self.crops)
        self.assertEqual(owners, [0, 0, 1])

    def test_pieces_that_do_not_meet_across_the_edge_stay_apart(self):
        a, b = self.crops
        near_a = _box_mask(self.SHAPE, 210, 150, 230, 160)    # A 의 crop 오른쪽 끝에 닿음
        b_low = _box_mask(self.SHAPE, 230, 165, 236, 170)     # B 의 crop 왼쪽 끝에 닿지만 A 쪽 조각과 줄이 다름
        owners = fc.assign_owners([near_a, b_low], [0, 1], self.faces, self.crops)
        self.assertEqual(owners, [0, 1])

    def test_single_pieces_match_assign_owner(self):
        rng = np.random.default_rng(3)
        checked = 0
        for _ in range(200):
            x0, y0 = int(rng.integers(60, 270)), int(rng.integers(50, 220))
            piece = _box_mask(self.SHAPE, x0, y0, x0 + 3, y0 + 3)
            sources = [i for i, crop in enumerate(self.crops) if crop.contains(x0, y0) and crop.contains(x0 + 2, y0 + 2)]
            if not sources:
                continue
            want = fc.assign_owner(fc.mask_centroid(piece), self.faces, self.crops, sources[0])
            self.assertEqual(fc.assign_owners([piece], sources[:1], self.faces, self.crops), [want])
            checked += 1
        self.assertGreater(checked, 50)


class GrowCropTests(unittest.TestCase):
    def test_zero_margin_is_the_crop(self):
        crop = fc.CropBox(10, 10, 74, 74)
        self.assertIs(fc.grow_crop(crop, 0, (256, 192)), crop)
        self.assertIs(fc.grow_crop(crop, -3, (256, 192)), crop)

    def test_grown_square_in_the_middle(self):
        self.assertEqual(fc.grow_crop(fc.CropBox(100, 50, 164, 114), 6, (256, 192)), fc.CropBox(94, 44, 170, 120))

    def test_random_crops_cover_the_margin_inside_the_image(self):
        rng = np.random.default_rng(11)
        for _ in range(500):
            width, height = int(rng.integers(40, 400)), int(rng.integers(40, 400))
            box = [float(rng.uniform(-20, width)), float(rng.uniform(-20, height))]
            box += [box[0] + float(rng.uniform(1, 120)), box[1] + float(rng.uniform(1, 120))]
            crop = fc.face_crop_box(box, (width, height))
            if crop is None:
                continue
            margin = int(rng.integers(1, 40))
            grown = fc.grow_crop(crop, margin, (width, height))
            self.assertTrue(0 <= grown.x0 < grown.x1 <= width and 0 <= grown.y0 < grown.y1 <= height)
            self.assertLessEqual(grown.x0, max(0, crop.x0 - margin))
            self.assertLessEqual(grown.y0, max(0, crop.y0 - margin))
            self.assertGreaterEqual(grown.x1, min(width, crop.x1 + margin))
            self.assertGreaterEqual(grown.y1, min(height, crop.y1 + margin))
            if grown.width < width and grown.height < height:
                self.assertEqual(grown.width, grown.height, "이미지에 들어가면 정사각형")


class TouchingSidesTests(unittest.TestCase):
    """crop 키우기의 방아쇠 — 마스크가 crop 의 안쪽 가장자리(이미지 테두리가 아닌 변)에 충분히 닿았나."""

    SHAPE = (300, 400)   # (높이, 폭)
    CROP = fc.CropBox(100, 50, 200, 150)   # 한 변 100 → 닿은 길이 기준 max(4, 2) = 4 px

    def test_constants(self):
        self.assertEqual((fc.GROW_FACTOR, fc.GROW_ROUNDS), (1.5, 2))
        self.assertEqual((fc.TOUCH_BAND, fc.TOUCH_MIN_PX, fc.TOUCH_MIN_FRACTION), (2, 4, 0.02))
        self.assertEqual(fc.SIDES, ("left", "top", "right", "bottom"))

    def test_inside_the_crop_does_not_touch(self):
        self.assertEqual(fc.touching_sides(_box_mask(self.SHAPE, 110, 60, 190, 140), self.CROP), ())
        self.assertEqual(fc.touching_sides(np.zeros(self.SHAPE, bool), self.CROP), ())

    def test_each_side(self):
        cases = {
            "left": (100, 80, 110, 90), "top": (130, 50, 140, 60),
            "right": (190, 80, 200, 90), "bottom": (130, 140, 140, 150),
        }
        for side, box in cases.items():
            with self.subTest(side=side):
                self.assertEqual(fc.touching_sides(_box_mask(self.SHAPE, *box), self.CROP), (side,))
        everything = _box_mask(self.SHAPE, *self.CROP.as_tuple())
        self.assertEqual(fc.touching_sides(everything, self.CROP), fc.SIDES, "변 순서는 SIDES")

    def test_band_is_two_pixels(self):
        near = _box_mask(self.SHAPE, 198, 80, 199, 90)      # 오른쪽 변 1 px 안쪽(띠 안)
        self.assertEqual(fc.touching_sides(near, self.CROP), ("right",))
        far = _box_mask(self.SHAPE, 196, 80, 198, 90)       # 2 px 안쪽(띠 밖)
        self.assertEqual(fc.touching_sides(far, self.CROP), ())
        self.assertEqual(fc.touching_sides(far, self.CROP, band=4), ("right",))

    def test_a_few_pixels_on_the_edge_do_not_count(self):
        self.assertEqual(fc.touching_sides(_box_mask(self.SHAPE, 195, 80, 200, 83), self.CROP), (), "3 px < 4 px")
        self.assertEqual(fc.touching_sides(_box_mask(self.SHAPE, 195, 80, 200, 84), self.CROP), ("right",), "4 px")
        big = fc.CropBox(0, 0, 300, 300)    # 한 변 300 → max(4, 6) = 6 px
        self.assertEqual(fc.touching_sides(_box_mask(self.SHAPE, 290, 80, 300, 85), big), ())
        self.assertEqual(fc.touching_sides(_box_mask(self.SHAPE, 290, 80, 300, 86), big), ("right",))
        scattered = np.zeros(self.SHAPE, bool)
        scattered[60:140:10, 199] = True     # 띠를 따라 흩어진 8 점 — 길이로 센다
        self.assertEqual(fc.touching_sides(scattered, self.CROP), ("right",))

    def test_image_edges_never_trigger(self):
        corner = fc.CropBox(0, 0, 100, 100)
        hair = _box_mask(self.SHAPE, 0, 0, 60, 30) | _box_mask(self.SHAPE, 0, 0, 30, 60)   # 위 · 왼쪽 변 = 이미지 테두리
        self.assertEqual(fc.touching_sides(hair, corner), ())
        self.assertEqual(fc.touching_sides(hair | _box_mask(self.SHAPE, 60, 90, 70, 100), corner), ("bottom",))
        whole = fc.CropBox(0, 0, 400, 300)
        self.assertEqual(fc.touching_sides(np.ones(self.SHAPE, bool), whole), (), "이미지 전체 crop 은 닿을 변이 없다")


class GrowTowardTests(unittest.TestCase):
    SIZE = (400, 300)   # (폭, 높이)

    def test_grows_toward_the_touched_side_only(self):
        crop = fc.CropBox(100, 50, 200, 150)
        self.assertEqual(fc.grow_toward(crop, ("bottom",), self.SIZE), fc.CropBox(75, 50, 225, 200), "아래로, 옆은 양쪽 반씩")
        self.assertEqual(fc.grow_toward(crop, ("right",), self.SIZE), fc.CropBox(100, 25, 250, 175))
        self.assertEqual(fc.grow_toward(crop, ("left", "top"), self.SIZE), fc.CropBox(50, 0, 200, 150))
        self.assertEqual(fc.grow_toward(crop, ("left", "right"), self.SIZE), fc.CropBox(75, 25, 225, 175),
                         "두 변 = 중심 유지")
        self.assertEqual(fc.grow_toward(crop, (), self.SIZE), fc.CropBox(75, 25, 225, 175))
        self.assertEqual(fc.grow_toward(crop, ("right",), self.SIZE, factor=2.0), fc.CropBox(100, 0, 300, 200))

    def test_pushed_inside_and_keeps_the_crop(self):
        crop = fc.CropBox(300, 200, 380, 280)
        grown = fc.grow_toward(crop, ("right",), self.SIZE)      # 120 → 오른쪽 끝(400)에 걸려 왼쪽으로 밀림
        self.assertEqual(grown, fc.CropBox(280, 180, 400, 300))
        self.assertEqual((grown.width, grown.height), (120, 120))

    def test_long_side_is_clamped_to_the_image_then_whole_image(self):
        crop = fc.CropBox(100, 0, 300, 200)
        grown = fc.grow_toward(crop, ("left", "bottom"), self.SIZE)   # 300 ≥ 높이 300 → 세로는 이미지 전체
        self.assertEqual(grown, fc.CropBox(0, 0, 300, 300))
        whole = fc.grow_toward(grown, ("right",), self.SIZE)            # 450 → 이미지 전체
        self.assertEqual(whole, fc.CropBox(0, 0, 400, 300))
        self.assertEqual(fc.grow_toward(whole, ("right",), self.SIZE), whole, "더 키울 수 없으면 그대로")

    def test_tiny_crop_grows_at_least_a_pixel(self):
        self.assertEqual(fc.grow_toward(fc.CropBox(10, 10, 11, 11), ("right",), self.SIZE), fc.CropBox(10, 10, 12, 12))

    def test_random_growth_is_square_inside_and_contains_the_crop(self):
        rng = np.random.default_rng(5)
        for _ in range(500):
            width, height = int(rng.integers(40, 500)), int(rng.integers(40, 500))
            crop = fc.face_crop_box([float(rng.uniform(0, width)), float(rng.uniform(0, height)),
                                     float(rng.uniform(0, width)), float(rng.uniform(0, height))], (width, height))
            if crop is None:
                continue
            sides = tuple(s for s in fc.SIDES if rng.random() < 0.5)
            grown = fc.grow_toward(crop, sides, (width, height))
            self.assertTrue(0 <= grown.x0 <= crop.x0 and crop.x1 <= grown.x1 <= width, (crop, grown))
            self.assertTrue(0 <= grown.y0 <= crop.y0 and crop.y1 <= grown.y1 <= height, (crop, grown))
            want = max(max(crop.width, crop.height) + 1, int(np.ceil(max(crop.width, crop.height) * fc.GROW_FACTOR)))
            self.assertEqual(grown.width, min(want, width))
            self.assertEqual(grown.height, min(want, height))

    def test_enclosing_crop(self):
        a, b = fc.CropBox(10, 10, 60, 60), fc.CropBox(40, 30, 120, 110)
        self.assertIs(fc.enclosing_crop([a], self.SIZE), a)
        both = fc.enclosing_crop([a, b], self.SIZE)
        self.assertEqual(both, fc.CropBox(10, 5, 120, 115), "둘을 감싼 상자(10, 10, 120, 110)의 정사각형 — 중심 유지")
        self.assertEqual(fc.enclosing_crop([fc.CropBox(0, 0, 400, 10), fc.CropBox(0, 290, 10, 300)], self.SIZE),
                         fc.CropBox(0, 0, 400, 300))


class GrowthRequestTests(unittest.TestCase):
    """어느 얼굴의 crop 을 키우나 — 그 얼굴이 주인이고 그 crop 에서 찾은 조각이 그 crop 가장자리에 닿았을 때만."""

    SHAPE = (300, 400)
    SIZE = (400, 300)

    def setUp(self):
        self.faces = [[100, 100, 200, 200], [238, 150, 262, 174]]
        self.crops = [fc.face_crop_box(box, self.SIZE) for box in self.faces]   # A x < 230, B x ≥ 230

    def test_no_touch_no_request(self):
        inside = _box_mask(self.SHAPE, 120, 120, 180, 180)
        self.assertEqual(fc.growth_requests([inside], [0], self.faces, self.crops, self.SIZE), {})
        self.assertEqual(fc.growth_requests([], [], self.faces, self.crops, self.SIZE), {})

    def test_own_cut_piece_grows_its_face(self):
        a = self.crops[0]
        hair = _box_mask(self.SHAPE, 120, 150, 200, a.y1)    # A 의 crop 아래 가장자리에 잘림
        requests = fc.growth_requests([hair], [0], self.faces, self.crops, self.SIZE)
        self.assertEqual(requests, {0: (("bottom",), fc.grow_toward(a, ("bottom",), self.SIZE))})

    def test_a_neighbours_piece_cut_at_this_edge_does_not_grow_this_face(self):
        a, b = self.crops
        hair_a = np.zeros(self.SHAPE, dtype=bool)
        hair_a[70:100, 95:234] = True
        hair_a[70:180, 200:234] = True                       # A 의 머리카락 — A 의 crop 오른쪽 끝을 넘어 B 의 crop 으로
        seen_a = hair_a & _box_mask(self.SHAPE, *a.as_tuple())
        seen_b = hair_a & _box_mask(self.SHAPE, *b.as_tuple())   # B 의 crop 왼쪽 끝에 잘린 띠 — 주인은 A
        requests = fc.growth_requests([seen_a, seen_b], [0, 1], self.faces, self.crops, self.SIZE)
        self.assertEqual(set(requests), {0}, "B 의 crop 은 키우지 않는다")
        self.assertEqual(requests[0][0], ("right",))

    def test_faces_filter_and_whole_image(self):
        a = self.crops[0]
        hair = _box_mask(self.SHAPE, 120, 150, 200, a.y1)
        self.assertEqual(fc.growth_requests([hair], [0], self.faces, self.crops, self.SIZE, faces={1}), {})
        whole = [fc.CropBox(0, 0, 400, 300), self.crops[1]]
        self.assertEqual(fc.growth_requests([np.ones(self.SHAPE, bool)], [0], self.faces, whole, self.SIZE), {})

    def test_status_and_meta_record_the_growth(self):
        info = fc.plan_face_crops([[100, 100, 200, 200]], [0.8], self.SIZE, model="m.pt", confidence=0.3)
        before_status, before_meta = info.status(), info.as_meta()
        self.assertNotIn("grown", before_status)
        self.assertNotIn("grown", before_meta["faces"][0])
        grown = fc.grow_toward(info.crops[0], ("bottom",), self.SIZE)
        info.growth.append(fc.CropGrowth(face=0, prompt="hair", crop=grown, rounds=1, sides=("bottom",)))
        self.assertEqual(
            info.status(),
            before_status + f"; grown where the mask reached the crop edge: face 1 'hair' → "
            f"{grown.width}x{grown.height}@{grown.x0},{grown.y0} (1 round)",
        )
        self.assertEqual(info.as_meta()["faces"][0]["grown"],
                         [{"prompt": "hair", "crop": list(grown.as_tuple()), "rounds": 1, "sides": ["bottom"]}])
        self.assertEqual(info.as_meta()["faces"][0]["crop"], before_meta["faces"][0]["crop"], "원래 crop 은 그대로")


class DuplicateBoxTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(fc.box_iou((40, 70, 80, 110), (48, 66, 92, 112)), 0.546, places=3)
        self.assertEqual(fc.box_iou((0, 0, 10, 10), (20, 0, 30, 10)), 0.0)
        self.assertEqual(fc.box_iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

    def test_same_face(self):
        self.assertTrue(fc.same_face((40, 70, 80, 110), (48, 66, 92, 112)), "IoU 0.55 — NMS(0.7)가 둘 다 남긴 상자")
        self.assertTrue(fc.same_face((0, 0, 100, 100), (40, 40, 60, 60)), "작은 상자의 중심이 큰 상자 안")
        self.assertFalse(fc.same_face((40, 70, 80, 110), (84, 70, 124, 110)), "나란한 두 얼굴")
        self.assertFalse(fc.same_face((0, 0, 40, 40), (30, 30, 70, 70)), "모서리만 겹침(IoU 0.03)")

    def test_dedupe_keeps_the_higher_score(self):
        boxes = [[40, 70, 80, 110], [48, 66, 92, 112], [170, 70, 210, 110]]
        self.assertEqual(fc.dedupe_faces(boxes, [0.6, 0.8, 0.5]), [1, 2])
        self.assertEqual(fc.dedupe_faces(boxes, [0.8, 0.6, 0.5]), [0, 2])
        self.assertEqual(fc.dedupe_faces(boxes, [0.7, 0.7, 0.5]), [0, 2], "같으면 앞 번호")
        self.assertEqual(fc.dedupe_faces(boxes, [float("nan"), 0.6]), [1, 2], "점수 없는 상자는 나중")
        self.assertEqual(fc.dedupe_faces([], []), [])


class PlanTests(unittest.TestCase):
    def test_two_boxes_on_one_face_are_one_face(self):
        info = fc.plan_face_crops(
            [[170, 70, 210, 110], [40, 70, 80, 110], [48, 66, 92, 112]], [0.9, 0.6, 0.8], (256, 192),
            model="face_yolov8n.pt", confidence=0.3,
        )
        self.assertEqual(info.boxes, [[48.0, 66.0, 92.0, 112.0], [170.0, 70.0, 210.0, 110.0]])
        self.assertEqual(info.scores, [0.8, 0.9])
        self.assertEqual(info.count, 2)

    def test_faces_left_to_right_with_their_scores_and_bad_boxes_dropped(self):
        info = fc.plan_face_crops(
            [[300, 50, 340, 90], [10, 10, 10, 50], [20, 60, 60, 100], [900, 900, 950, 950]],
            [0.9, 0.8, 0.7, 0.6],
            (400, 200),
            model="face_yolov8n.pt",
            confidence=0.3,
        )
        self.assertEqual(info.count, 2)
        self.assertEqual(info.scores, [0.7, 0.9], "왼쪽 얼굴 먼저, 점수는 같이 움직인다")
        self.assertEqual(info.boxes, [[20.0, 60.0, 60.0, 100.0], [300.0, 50.0, 340.0, 90.0]])
        self.assertEqual(info.crops, [fc.face_crop_box(b, (400, 200)) for b in info.boxes])

    def test_status_and_meta(self):
        none = fc.plan_face_crops([], [], (64, 64), model="face_yolov8s.pt", confidence=0.45)
        self.assertEqual(none.status(), "face crop — no faces found by face_yolov8s.pt (confidence 0.45); nothing to mask.")
        two = fc.plan_face_crops([[0, 0, 20, 20], [40, 40, 60, 60]], [0.5, 0.6], (64, 64), model="m.pt", confidence=0.3)
        self.assertIn("2 face(s) by m.pt (confidence 0.30)", two.status())
        meta = two.as_meta()
        self.assertEqual(meta["model"], "m.pt")
        self.assertEqual([face["crop"] for face in meta["faces"]], [list(c.as_tuple()) for c in two.crops])
        self.assertEqual([face["score"] for face in meta["faces"]], [0.5, 0.6])

    def test_order_faces(self):
        self.assertEqual(fc.order_faces([[50, 0, 60, 10], [0, 30, 10, 40], [0, 0, 10, 10]]), [2, 1, 0])


class FaceCropKwargsTests(unittest.TestCase):
    def test_off_gives_the_old_call(self):
        for args in ({}, {"sam3_face_crop": False}, {"sam3_face_crop": "False"}, {"sam3_face_crop": None}):
            with self.subTest(args=args):
                self.assertEqual(fc.face_crop_kwargs(args), {})

    def test_on_is_normalised(self):
        self.assertEqual(
            fc.face_crop_kwargs({"sam3_face_crop": True, "sam3_face_crop_model": " face_yolov8s.pt ",
                                 "sam3_face_crop_confidence": 0.45}),
            {"face_crop": True, "face_crop_model": "face_yolov8s.pt", "face_crop_confidence": 0.45},
        )
        self.assertEqual(
            fc.face_crop_kwargs({"sam3_face_crop": "True", "sam3_face_crop_model": "", "sam3_face_crop_confidence": "x"}),
            {"face_crop": True, "face_crop_model": fd.DEFAULT_MODEL, "face_crop_confidence": fd.DEFAULT_CONFIDENCE},
        )
        self.assertEqual(fc.face_crop_kwargs({"sam3_face_crop": True, "sam3_face_crop_confidence": 7})["face_crop_confidence"], 1.0)
        self.assertEqual(
            fc.face_crop_kwargs({"sam3_face_crop": True, "sam3_face_crop_confidence": float("nan")})["face_crop_confidence"],
            fd.DEFAULT_CONFIDENCE,
        )


# ---------------------------------------------------------------------------------------------------------------------
# 얼굴 검출기 — models/adetailer, 가짜 ultralytics
# ---------------------------------------------------------------------------------------------------------------------


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)


def _fake_ultralytics(calls: list, boxes=((10, 20, 30, 40),), conf=(0.8,)):
    module = types.ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            calls.append(("load", path))

        def __call__(self, image, **kwargs):
            calls.append(("predict", image.size, kwargs))
            return [types.SimpleNamespace(boxes=_Boxes(boxes, conf))]

    module.YOLO = YOLO
    utils = types.ModuleType("ultralytics.utils")
    checks = types.ModuleType("ultralytics.utils.checks")
    checks.check_yolov5u_filename = lambda file, verbose=True: file
    utils.checks = checks
    module.utils = utils
    return {"ultralytics": module, "ultralytics.utils": utils, "ultralytics.utils.checks": checks}


class FaceModelFolderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.models = Path(tmp.name) / "models"
        self.folder = self.models / "adetailer"
        self.folder.mkdir(parents=True)
        for name in ("hand_yolov8n.pt", "face_yolov8s.pt", "face_yolov8n.pt", "yolov8x-worldv2.pt", "notes.txt",
                     "person_yolov8n-seg.pt"):
            (self.folder / name).write_bytes(b"x")
        (self.folder / "sub").mkdir()
        (self.folder / "sub" / "face_custom.pt").write_bytes(b"x")
        patcher = mock.patch.object(fd, "_models_root", lambda: self.models)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_listing_puts_the_default_and_face_models_first_and_skips_world(self):
        self.assertEqual(
            fd.find_face_models(),
            ["face_yolov8n.pt", "face_custom.pt", "face_yolov8s.pt", "hand_yolov8n.pt", "person_yolov8n-seg.pt"],
        )

    def test_empty_folder_lists_just_the_default(self):
        with mock.patch.object(fd, "_models_root", lambda: Path(tempfile.gettempdir()) / "no-such-forge-models"):
            self.assertEqual(fd.find_face_models(), [fd.DEFAULT_MODEL])

    def test_resolve(self):
        self.assertEqual(fd.resolve_face_model("face_yolov8s.pt"), self.folder / "face_yolov8s.pt")
        self.assertEqual(fd.resolve_face_model("FACE_CUSTOM.pt"), self.folder / "sub" / "face_custom.pt")
        self.assertEqual(fd.resolve_face_model(""), self.folder / "face_yolov8n.pt", "빈 값은 기본 모델")
        self.assertEqual(fd.resolve_face_model(str(self.folder / "hand_yolov8n.pt")), self.folder / "hand_yolov8n.pt")
        self.assertIsNone(fd.resolve_face_model("face_yolov9c.pt"))
        self.assertIsNone(fd.resolve_face_model("yolov8x-worldv2.pt"))
        self.assertIsNone(fd.resolve_face_model(str(self.folder / "notes.txt")))

    def test_adetailer_extra_model_dirs_are_scanned(self):
        extra = self.models.parent / "extra"
        extra.mkdir()
        (extra / "face_extra.pt").write_bytes(b"x")
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(data={fd.OPT_EXTRA_MODEL_DIRS: f" {extra} | "}))
        package = types.ModuleType("modules")
        package.__path__ = []
        package.shared = shared
        with mock.patch.dict(sys.modules, {"modules": package, "modules.shared": shared}):
            self.assertIn("face_extra.pt", fd.find_face_models())
            self.assertEqual(fd.resolve_face_model("face_extra.pt"), extra / "face_extra.pt")


class DetectFacesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.models = Path(tmp.name) / "models"
        (self.models / "adetailer").mkdir(parents=True)
        self.model = self.models / "adetailer" / "face_yolov8n.pt"
        self.model.write_bytes(b"x")
        patcher = mock.patch.object(fd, "_models_root", lambda: self.models)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.image = Image.new("RGB", (64, 48))

    def test_boxes_scores_and_the_ultralytics_call(self):
        calls: list = []
        with mock.patch.dict(sys.modules, _fake_ultralytics(calls, boxes=((1, 2, 3, 4), (5, 6, 7, 8)), conf=(0.9, 0.4))):
            found = fd.detect_faces(self.image, "face_yolov8n.pt", 0.35, "cuda")
        self.assertEqual(found.boxes, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        self.assertEqual([round(s, 4) for s in found.scores], [0.9, 0.4])
        self.assertEqual(found.model_path, str(self.model))
        self.assertEqual(calls[0], ("load", str(self.model)), "있는 파일의 절대 경로로만 불러온다(이름이면 ultralytics 가 받는다)")
        _, size, kwargs = calls[1]
        self.assertEqual(size, (64, 48))
        self.assertEqual(kwargs, {"conf": 0.35, "device": "", "verbose": False})

    def test_device_mapping(self):
        self.assertEqual(fd.ultralytics_device("cpu"), "cpu")
        for device in ("cuda", "cuda:1", "auto", ""):
            self.assertEqual(fd.ultralytics_device(device), "", "장치 번호를 주면 ultralytics 가 set_device 한다")

    def test_no_faces(self):
        calls: list = []
        with mock.patch.dict(sys.modules, _fake_ultralytics(calls, boxes=(), conf=())):
            found = fd.detect_faces(self.image, "face_yolov8n.pt", 0.3, "cpu")
        self.assertEqual((found.boxes, found.scores), ([], []))
        self.assertEqual(calls[1][2]["device"], "cpu")

    def test_missing_model_is_a_clear_error_and_nothing_is_loaded(self):
        calls: list = []
        with mock.patch.dict(sys.modules, _fake_ultralytics(calls)):
            with self.assertRaises(fd.FaceDetectorUnavailable) as ctx:
                fd.detect_faces(self.image, "face_yolov8s.pt", 0.3, "cuda")
        self.assertEqual(calls, [])
        message = str(ctx.exception)
        self.assertIn("face_yolov8s.pt", message)
        self.assertIn(str(self.models / "adetailer"), message)
        self.assertIn("never downloaded", message)

    def test_missing_folder_still_names_where_to_put_the_model(self):
        empty = self.models.parent / "other" / "models"
        empty.mkdir(parents=True)
        with mock.patch.object(fd, "_models_root", lambda: empty):
            self.assertEqual(fd.model_dirs(), [])
            message = fd.missing_model_message("face_yolov8n.pt")
        self.assertIn(str(empty / "adetailer"), message)
        with mock.patch.object(fd, "_models_root", lambda: None):
            self.assertIn("<webui>/models/adetailer", fd.missing_model_message("face_yolov8n.pt"))

    def test_missing_ultralytics_is_a_clear_error(self):
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            with self.assertRaises(fd.FaceDetectorUnavailable) as ctx:
                fd.detect_faces(self.image, "face_yolov8n.pt", 0.3, "cuda")
        self.assertIn("ultralytics", str(ctx.exception))
        self.assertIsInstance(ctx.exception, RuntimeError)

    def test_a_name_ultralytics_would_download_is_refused(self):
        calls: list = []
        stubs = _fake_ultralytics(calls)
        stubs["ultralytics.utils.checks"].check_yolov5u_filename = lambda file, verbose=True: file.replace(".pt", "u.pt")
        with mock.patch.dict(sys.modules, stubs):
            with self.assertRaises(fd.FaceDetectorUnavailable) as ctx:
                fd.detect_faces(self.image, "face_yolov8n.pt", 0.3, "cuda")
        self.assertIn("download", str(ctx.exception))
        self.assertEqual(calls, [])

    def test_module_import_does_not_pull_in_ultralytics_or_torch(self):
        import subprocess

        code = (
            "import sys; sys.path.insert(0, r'%s'); import sam3ext.face_detect, sam3ext.face_crop; "
            "print('ultralytics' in sys.modules, 'torch' in sys.modules)" % ROOT
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertEqual(out.stdout.split(), ["False", "False"])


if __name__ == "__main__":
    unittest.main()
