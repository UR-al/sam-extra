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


class PlanTests(unittest.TestCase):
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
