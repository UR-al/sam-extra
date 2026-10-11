"""SAM3 '얼굴 안에서만 찾기' 종단 — run_sam3_on_pil · In-flight 스크립트 · Refine · 인자/infotext/XYZ.

가짜 SAM3 처리기(받은 이미지에서 프롬프트 색의 연결 성분을 찾는다 — crop 을 받으면 crop 좌표로)와 가짜 얼굴 검출기로
GPU·체크포인트·ultralytics 없이 돈다. 같은 그림을 전체 이미지로 돌린 결과와 비교해 붙여 넣기가 비트 단위로 맞는지 본다.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

core = importlib.import_module("sam3ext.core")
from sam3ext import face_crop as fc  # noqa: E402
from sam3ext import face_detect as fd  # noqa: E402
from sam3ext.args import ALL_ARGS, Sam3Args  # noqa: E402

COLORS = {
    "eye": (255, 0, 0),
    "mouth": (0, 255, 0),
    "skin": (250, 220, 200),
    "hair": (0, 0, 255),
}
BACKGROUND = (128, 128, 128)
SIZE = (256, 192)   # (폭, 높이)


class ColorProcessor:
    """가짜 Sam3Processor: 프롬프트 색 픽셀의 8-연결 성분 하나 = 검출 하나(마스크·상자·점수 0.9)."""

    def __init__(self):
        self.images: list[tuple[int, int]] = []
        self.prompts: list[str] = []
        self.threshold = None

    def set_confidence_threshold(self, value):
        self.threshold = value

    def set_image(self, image):
        self.images.append(image.size)
        return {"backbone_out": {"rgb": np.asarray(image.convert("RGB")).copy()}}

    def set_text_prompt(self, prompt, state):
        self.prompts.append(prompt)
        rgb = state["backbone_out"]["rgb"]
        h, w = rgb.shape[:2]
        color = COLORS.get(prompt)
        hit = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1) if color else np.zeros((h, w), bool)
        count, labels = cv2.connectedComponents(hit.astype(np.uint8), connectivity=8)
        masks, boxes = [], []
        for label in range(1, count):
            mask = labels == label
            ys, xs = np.nonzero(mask)
            masks.append(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        state["masks"] = np.stack(masks)[:, None] if masks else np.zeros((0, 1, h, w), dtype=bool)
        state["boxes"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        state["scores"] = np.full((len(masks),), 0.9, dtype=np.float32)
        return state


def _draw_face(rgb, box, *, eyes=True, mouth=True):
    x0, y0, x1, y1 = box
    rgb[y0:y1, x0:x1] = COLORS["skin"]
    if eyes:
        rgb[y0 + 10:y0 + 14, x0 + 8:x0 + 12] = COLORS["eye"]
        rgb[y0 + 10:y0 + 14, x1 - 12:x1 - 8] = COLORS["eye"]
    if mouth:
        rgb[y1 - 12:y1 - 9, x0 + 16:x1 - 16] = COLORS["mouth"]


def _color(rgb, name):
    return np.all(rgb == np.array(COLORS[name], dtype=np.uint8), axis=-1)


def _crops_mask(crops, shape):
    out = np.zeros(shape, dtype=bool)
    for crop in crops:
        out[crop.y0:crop.y1, crop.x0:crop.x1] = True
    return out


def _as_bool(img):
    return np.asarray(img.convert("L")) > 127


FACE_A = (40, 70, 80, 110)
FACE_B = (170, 70, 210, 110)


def _two_faces_image():
    rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
    _draw_face(rgb, FACE_A)
    _draw_face(rgb, FACE_B)
    rgb[5:12, 120:127] = COLORS["eye"]      # 얼굴 밖의 같은 색 — 얼굴 crop 이면 찾지 않아야 한다
    rgb[150:160, 10:20] = COLORS["mouth"]
    return rgb


class _Run:
    """run_sam3_on_pil 을 가짜 처리기·검출기로 돌린다."""

    def __init__(self, test, faces, scores=None, *, bundle_allowed=True):
        self.test = test
        self.faces = [list(map(float, f)) for f in faces]
        self.scores = scores if scores is not None else [0.8] * len(faces)
        self.processor = ColorProcessor()
        self.detect_calls: list = []
        self.bundle_allowed = bundle_allowed
        self.stderr = ""

    def _detect(self, image, model, confidence, device):
        self.detect_calls.append((image.size, model, confidence, device))
        return fd.FaceDetection(boxes=self.faces, scores=self.scores, model_path=f"C:/models/adetailer/{model}")

    def __call__(self, rgb, prompt, **kwargs):
        def bundle(*a, **k):
            if not self.bundle_allowed:
                raise AssertionError("SAM3 bundle must not load")
            return (None, self.processor)

        stderr = io.StringIO()
        with mock.patch.object(core, "resolve_checkpoint_path", return_value=None), \
                mock.patch.object(core, "_get_model_bundle", side_effect=bundle), \
                mock.patch.object(core, "detect_faces", side_effect=self._detect), \
                contextlib.redirect_stderr(stderr):
            result = core.run_sam3_on_pil(Image.fromarray(rgb), prompt, 0.4, "", "cpu", **kwargs)
        self.stderr = stderr.getvalue()
        return result


class RunSam3FaceCropTests(unittest.TestCase):
    def setUp(self):
        self.rgb = _two_faces_image()
        self.crop_a = fc.face_crop_box(FACE_A, SIZE)
        self.crop_b = fc.face_crop_box(FACE_B, SIZE)

    def test_face_crop_off_never_calls_the_detector(self):
        run = _Run(self, [FACE_A])
        with mock.patch.object(core, "detect_faces", side_effect=AssertionError("detector")):
            stderr = io.StringIO()
            with mock.patch.object(core, "resolve_checkpoint_path", return_value=None), \
                    mock.patch.object(core, "_get_model_bundle", return_value=(None, run.processor)), \
                    contextlib.redirect_stderr(stderr):
                result = core.run_sam3_on_pil(Image.fromarray(self.rgb), "eye", 0.4, "", "cpu")
        self.assertIsNone(result.face_crop)
        np.testing.assert_array_equal(_as_bool(result.mask), _color(self.rgb, "eye"), "꺼져 있으면 전체 이미지(얼굴 밖 포함)")
        self.assertEqual(run.processor.images, [SIZE])

    def test_paste_back_matches_full_image_detection_inside_the_crops(self):
        run = _Run(self, [FACE_B, FACE_A], scores=[0.7, 0.9])
        result = run(self.rgb, "eye, mouth", face_crop=True, face_crop_model="face_yolov8s.pt", face_crop_confidence=0.45)
        crops = _crops_mask([self.crop_a, self.crop_b], self.rgb.shape[:2])
        want = (_color(self.rgb, "eye") | _color(self.rgb, "mouth")) & crops
        np.testing.assert_array_equal(_as_bool(result.mask), want)
        self.assertFalse(_as_bool(result.mask)[5:12, 120:127].any(), "얼굴 밖의 같은 색은 찾지 않는다")
        self.assertEqual(run.detect_calls, [(SIZE, "face_yolov8s.pt", 0.45, "cpu")])
        self.assertEqual(run.processor.images, [(self.crop_a.width, self.crop_a.height)] * 2, "SAM3 는 crop 만 본다")
        self.assertEqual(run.processor.threshold, 0.4)
        info = result.face_crop
        self.assertEqual(info.count, 2)
        self.assertEqual(info.crops, [self.crop_a, self.crop_b], "왼쪽 → 오른쪽")
        self.assertEqual(info.scores, [0.9, 0.7])
        self.assertEqual(info.model, "face_yolov8s.pt")
        self.assertIn("2 face(s) by face_yolov8s.pt (confidence 0.45)", run.stderr)

    def test_boxes_come_back_in_image_coordinates(self):
        result = _Run(self, [FACE_A])(self.rgb, "mouth", face_crop=True)
        x0, y0, x1, y1 = FACE_A
        self.assertEqual(result.boxes, [[float(x0 + 16), float(y1 - 12), float(x1 - 16), float(y1 - 9)]])
        self.assertEqual(result.scores, [0.8999999761581421])

    def test_individual_is_one_mask_per_group_per_face_left_to_right(self):
        result = _Run(self, [FACE_B, FACE_A])(self.rgb, "eye / mouth", face_crop=True)
        eye, mouth = _color(self.rgb, "eye"), _color(self.rgb, "mouth")
        a = _crops_mask([self.crop_a], eye.shape)
        b = _crops_mask([self.crop_b], eye.shape)
        expected = [eye & a, eye & b, mouth & a, mouth & b]
        self.assertEqual(len(result.masks), 4)
        for got, want in zip(result.masks, expected):
            np.testing.assert_array_equal(_as_bool(got), want)
        np.testing.assert_array_equal(_as_bool(result.mask), np.any(np.stack(expected), axis=0), "Combined = 합집합")

    def test_a_face_without_the_part_gives_no_empty_pass(self):
        rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        _draw_face(rgb, FACE_A)
        _draw_face(rgb, FACE_B, mouth=False)
        result = _Run(self, [FACE_A, FACE_B])(rgb, "mouth", face_crop=True)
        self.assertEqual(len(result.masks), 1)
        np.testing.assert_array_equal(_as_bool(result.masks[0]), _color(rgb, "mouth"))

    def test_overlapping_faces_keep_each_part_with_its_own_face(self):
        face_a, face_b = (40, 70, 80, 110), (84, 70, 124, 110)
        rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        _draw_face(rgb, face_a, mouth=False)
        _draw_face(rgb, face_b, mouth=False)
        crop_a, crop_b = fc.face_crop_box(face_a, SIZE), fc.face_crop_box(face_b, SIZE)
        self.assertLess(crop_b.x0, crop_a.x1, "두 crop 이 겹친다")
        # 오른쪽 얼굴의 왼쪽 눈(원래 x 92..96)을 x 88..92 로 옮긴다 — 두 crop(왼쪽은 x < 92) 모두에 온전히 들어간다
        rgb[80:84, 92:96] = COLORS["skin"]
        rgb[80:84, 88:92] = COLORS["eye"]
        eye = _color(rgb, "eye")
        self.assertTrue(eye[80:84, 88:92].all() and crop_a.contains(88, 80) and crop_b.contains(88, 80))
        result = _Run(self, [face_b, face_a])(rgb, "eye", face_crop=True)
        own_a = eye.copy()
        own_a[:, 86:] = False      # 왼쪽 얼굴의 두 눈(x 48..52, 68..72)
        own_b = eye & ~own_a       # 오른쪽 얼굴의 두 눈 — 하나는 왼쪽 crop 에서도 보였다
        self.assertEqual(len(result.masks), 2)
        np.testing.assert_array_equal(_as_bool(result.masks[0]), own_a, "왼쪽 얼굴 패스에 옆 얼굴의 눈이 섞이지 않는다")
        np.testing.assert_array_equal(_as_bool(result.masks[1]), own_b)
        np.testing.assert_array_equal(_as_bool(result.mask), eye & _crops_mask([crop_a, crop_b], eye.shape))

    def test_a_part_cut_by_the_neighbours_crop_edge_is_whole_again(self):
        face_a, face_b = (40, 70, 80, 110), (84, 70, 124, 110)
        rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        _draw_face(rgb, face_a, eyes=False, mouth=False)
        _draw_face(rgb, face_b, eyes=False, mouth=False)
        crop_a = fc.face_crop_box(face_a, SIZE)
        rgb[80:84, crop_a.x1 - 2:crop_a.x1 + 2] = COLORS["eye"]   # 왼쪽 crop 의 오른쪽 끝에 걸친 오른쪽 얼굴의 눈
        result = _Run(self, [face_a, face_b])(rgb, "eye", face_crop=True)
        self.assertEqual(len(result.masks), 1, "잘린 조각과 온전한 조각이 같은 얼굴(오른쪽)로 모인다")
        np.testing.assert_array_equal(_as_bool(result.masks[0]), _color(rgb, "eye"))

    def test_exclude_runs_in_the_crops_after_hull(self):
        run = _Run(self, [FACE_A, FACE_B])
        result = run(self.rgb, "skin", face_crop=True, mask_hull=True, exclude_prompt="eye")
        skin, eye = _color(self.rgb, "skin"), _color(self.rgb, "eye")
        crops = _crops_mask([self.crop_a, self.crop_b], skin.shape)
        expected = []
        for crop in (self.crop_a, self.crop_b):
            hulled = core._convex_hull_mask(skin & _crops_mask([crop], skin.shape))
            expected.append(hulled & ~(eye & crops))
        self.assertEqual(len(result.masks), 2)
        for got, want in zip(result.masks, expected):
            np.testing.assert_array_equal(_as_bool(got), want)
        self.assertFalse(_as_bool(result.mask)[80:84, 48:52].any(), "hull 이 메운 눈 자리를 Exclude 가 다시 뺀다")
        self.assertTrue(_as_bool(result.mask)[98:101, 56:64].all(), "입 자리는 hull 로 채워진 채 남는다")
        self.assertEqual(run.processor.images, [(64, 64), (64, 64)], "검출·제외 토큰이 crop 마다 백본 한 번")
        self.assertEqual(run.processor.prompts, ["skin", "eye", "skin", "eye"])

    def test_a_neighbours_part_cut_at_this_crops_edge_stays_with_the_neighbour(self):
        """큰 얼굴 A 의 머리카락(한 덩어리)이 작은 얼굴 B 의 crop 왼쪽 끝으로 들어온다 — B 의 crop 이 본 띠는 잘린 조각이고
        그 무게중심은 A 의 crop 밖이다. 띠는 A 의 crop 에서 이어지는 조각과 묶여 A 의 패스로 간다."""
        size = (400, 300)
        face_a, face_b = (100, 100, 200, 200), (238, 150, 262, 174)
        rgb = np.full((size[1], size[0], 3), BACKGROUND, dtype=np.uint8)
        rgb[face_a[1]:face_a[3], face_a[0]:face_a[2]] = COLORS["skin"]
        rgb[face_b[1]:face_b[3], face_b[0]:face_b[2]] = COLORS["skin"]
        rgb[70:100, 95:234] = COLORS["hair"]
        rgb[70:230, 200:234] = COLORS["hair"]       # A 의 머리카락 — x 233 까지(B 의 crop 은 x 230 부터)
        rgb[146:150, 240:260] = COLORS["hair"]      # B 의 머리카락(A 의 것과 닿지 않음)
        crop_a, crop_b = fc.face_crop_box(face_a, size), fc.face_crop_box(face_b, size)
        self.assertEqual((crop_a.x1, crop_b.x0), (230, 230))
        hair = _color(rgb, "hair")
        own_b = np.zeros_like(hair)
        own_b[146:150, 240:260] = True
        own_a = hair & ~own_b

        result = _Run(self, [face_a, face_b])(rgb, "hair", face_crop=True)
        self.assertEqual(len(result.masks), 2)
        crops = _crops_mask([crop_a, crop_b], hair.shape)
        np.testing.assert_array_equal(_as_bool(result.masks[0]), own_a & crops, "A 의 머리카락은 B 의 crop 이 본 띠까지 A 의 패스")
        np.testing.assert_array_equal(_as_bool(result.masks[1]), own_b, "B 의 패스엔 B 의 머리카락만")
        np.testing.assert_array_equal(_as_bool(result.mask), hair & crops, "Combined 는 그대로")

    def test_two_detector_boxes_on_one_face_make_one_pass(self):
        face, duplicate = (40, 70, 80, 110), (48, 66, 92, 112)   # IoU 0.55 — ultralytics NMS(0.7)가 둘 다 남긴다
        rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        _draw_face(rgb, face)
        run = _Run(self, [face, duplicate], scores=[0.6, 0.8])
        result = run(rgb, "eye", face_crop=True)
        self.assertEqual(result.face_crop.count, 1)
        self.assertEqual(result.face_crop.boxes, [list(map(float, duplicate))], "점수가 높은 상자")
        self.assertEqual(len(result.masks), 1, "한 얼굴의 두 눈이 한 패스")
        np.testing.assert_array_equal(_as_bool(result.masks[0]), _color(rgb, "eye"))
        self.assertEqual(len(run.processor.images), 1, "crop 도 하나")

    def test_exclude_is_searched_as_far_as_the_mask_grows(self):
        """Dilation · Outline 은 crop 밖으로 번진다 — Exclude 도 그만큼 넓힌 영역에서 찾아, crop 바로 밖의 보호 대상이 먹히지 않는다."""
        face = (0, 0, 40, 40)
        crop = fc.face_crop_box(face, SIZE)
        rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        rgb[10:14, crop.x1 - 4:crop.x1] = COLORS["eye"]               # 대상 — crop 오른쪽 끝에 닿음
        rgb[6:18, crop.x1 + 1:crop.x1 + 6] = COLORS["mouth"]           # 보호할 것 — crop 밖 1 px
        protect = _color(rgb, "mouth")
        for kwargs in ({"mask_dilation": 4}, {"mask_outline_px": 2, "mask_dilation": 2}):
            with self.subTest(**kwargs):
                run = _Run(self, [face])
                on = run(rgb, "eye", face_crop=True, exclude_prompt="mouth", **kwargs)
                self.assertFalse((_as_bool(on.mask) & protect).any(), "넓힌 마스크가 보호 대상을 먹지 않는다")
                grown = fc.grow_crop(crop, 4, SIZE)
                self.assertEqual(run.processor.images, [(crop.width, crop.height), (grown.width, grown.height)])
                self.assertEqual(run.processor.prompts, ["eye", "mouth"])
                off = _Run(self, [face])
                with mock.patch.object(core, "resolve_checkpoint_path", return_value=None), \
                        mock.patch.object(core, "_get_model_bundle", return_value=(None, off.processor)), \
                        contextlib.redirect_stderr(io.StringIO()):
                    full = core.run_sam3_on_pil(Image.fromarray(rgb), "eye", 0.4, "", "cpu",
                                                exclude_prompt="mouth", **kwargs)
                np.testing.assert_array_equal(_as_bool(on.mask), _as_bool(full.mask), "얼굴 crop 을 끈 결과와 같다")

    def test_exclude_outside_the_faces_is_not_searched(self):
        rgb = self.rgb.copy()
        rgb[150:160, 10:20] = COLORS["skin"]     # 얼굴 밖 skin 색
        result = _Run(self, [FACE_A])(rgb, "skin", face_crop=True, exclude_prompt="mouth")
        self.assertFalse(_as_bool(result.mask)[150:160, 10:20].any())
        np.testing.assert_array_equal(_as_bool(result.mask), _color(rgb, "skin") & _crops_mask([self.crop_a], rgb.shape[:2]))

    def test_dilation_and_outline_happen_in_image_coordinates(self):
        result = _Run(self, [FACE_A])(self.rgb, "mouth", face_crop=True, mask_dilation=3, mask_outline_px=2)
        mouth = _color(self.rgb, "mouth") & _crops_mask([self.crop_a], self.rgb.shape[:2])
        want = core._dilate_mask(core._edge_aware_dilate(mouth, self.rgb, 2), 3)
        np.testing.assert_array_equal(_as_bool(result.mask), want)

    def test_mask_dilation_can_grow_past_the_crop(self):
        face = (0, 0, 40, 40)
        rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        crop = fc.face_crop_box(face, SIZE)
        rgb[10:14, crop.x1 - 4:crop.x1] = COLORS["eye"]
        result = _Run(self, [face])(rgb, "eye", face_crop=True, mask_dilation=4)
        self.assertTrue(_as_bool(result.mask)[10:14, crop.x1:crop.x1 + 3].all(), "Dilation 은 crop 밖까지 번진다(예전과 같은 px 의미)")

    def test_user_mask_narrowing_still_applies(self):
        user = np.zeros(self.rgb.shape[:2], dtype=bool)
        user[:, :128] = True     # 왼쪽 얼굴만
        result = _Run(self, [FACE_A, FACE_B])(self.rgb, "eye", face_crop=True, user_mask=user)
        np.testing.assert_array_equal(_as_bool(result.mask), _color(self.rgb, "eye") & user & _crops_mask([self.crop_a], user.shape))
        self.assertEqual(len(result.masks), 1)

    def test_user_mask_is_kept_as_is_when_no_faces(self):
        """얼굴이 없으면 '검출 없음' 과 같다 — Manual Mask 는 SAM3 가 못 찾았을 때처럼 칠한 그대로 쓴다(SAM3 는 올리지 않음)."""
        user = np.zeros(self.rgb.shape[:2], dtype=bool)
        user[20:60, 20:60] = True
        for prompt, exclude in (("hair", ""), ("", "mouth"), ("eye", "mouth")):
            with self.subTest(prompt=prompt, exclude=exclude):
                run = _Run(self, [], bundle_allowed=False)
                result = run(self.rgb, prompt, user_mask=user, exclude_prompt=exclude, face_crop=True)
                np.testing.assert_array_equal(_as_bool(result.mask), user)
                self.assertEqual(len(result.masks), 1)
                self.assertEqual(result.face_crop.count, 0)
                self.assertIn("kept as-is — face crop found no faces", run.stderr)
        small = np.zeros((48, 64), dtype=bool)       # 크기가 다른 Manual Mask 도 예전처럼 늘려 쓴다
        small[:, :32] = True
        result = _Run(self, [], bundle_allowed=False)(self.rgb, "hair", user_mask=small, face_crop=True)
        np.testing.assert_array_equal(_as_bool(result.mask)[:, :128], np.ones((SIZE[1], 128), bool))
        self.assertFalse(_as_bool(result.mask)[:, 128:].any())

    def test_no_faces_is_an_empty_result_without_loading_sam3(self):
        run = _Run(self, [], bundle_allowed=False)
        result = run(self.rgb, "eye", face_crop=True)
        self.assertFalse(_as_bool(result.mask).any())
        self.assertEqual(result.masks, [])
        self.assertEqual((result.boxes, result.scores), ([], []))
        self.assertEqual(result.face_crop.count, 0)
        self.assertEqual(result.checkpoint, "facebook/sam3::sam3.pt")
        self.assertEqual(result.overlay.size, SIZE)
        np.testing.assert_array_equal(np.asarray(result.overlay), self.rgb)
        self.assertIn("no faces found by face_yolov8n.pt (confidence 0.30)", run.stderr)
        self.assertEqual(core.describe_face_crop(result), result.face_crop.status())
        self.assertIsNone(core.describe_face_crop(types.SimpleNamespace(mask=None)))

    def test_detector_errors_propagate_before_sam3_loads(self):
        with mock.patch.object(core, "resolve_checkpoint_path", return_value=None), \
                mock.patch.object(core, "_get_model_bundle", side_effect=AssertionError("bundle")), \
                mock.patch.object(core, "detect_faces", side_effect=fd.FaceDetectorUnavailable("no ultralytics")):
            with self.assertRaises(fd.FaceDetectorUnavailable):
                core.run_sam3_on_pil(Image.fromarray(self.rgb), "eye", 0.4, "", "cpu", face_crop=True)

    def test_overlay_draws_the_face_crops(self):
        result = _Run(self, [FACE_A])(self.rgb, "mouth", face_crop=True)
        overlay = np.asarray(result.overlay)
        crop = self.crop_a
        np.testing.assert_array_equal(overlay[crop.y0 + 30, crop.x0], core._FACE_CROP_OVERLAY_COLOR)

    def test_artifact_meta_carries_the_faces(self):
        import json
        import tempfile

        result = _Run(self, [FACE_A])(self.rgb, "mouth", face_crop=True)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(core, "DEFAULT_OUTPUT_DIR", Path(tmp)):
            paths = core.write_artifacts(result, 7, label="mouth")
            meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
        self.assertEqual(meta["face_crop"]["model"], "face_yolov8n.pt")
        self.assertEqual(meta["face_crop"]["faces"][0]["crop"], list(self.crop_a.as_tuple()))
        self.assertEqual(meta["mask_count"], 1)


# ---------------------------------------------------------------------------------------------------------------------
# 인자 · infotext · XYZ · UI
# ---------------------------------------------------------------------------------------------------------------------


FACE_ATTRS = ("sam3_face_crop", "sam3_face_crop_model", "sam3_face_crop_confidence")
FACE_NAMES = ("SAM3 Face Crop", "SAM3 Face Crop Model", "SAM3 Face Crop Confidence")


class FaceCropArgsTests(unittest.TestCase):
    def test_defaults_are_off_and_match_the_detector_module(self):
        args = Sam3Args()
        self.assertIs(args.sam3_face_crop, False)
        self.assertEqual(args.sam3_face_crop_model, fd.DEFAULT_MODEL)
        self.assertEqual(args.sam3_face_crop_confidence, fd.DEFAULT_CONFIDENCE)
        self.assertEqual((fd.DEFAULT_MODEL, fd.DEFAULT_CONFIDENCE), ("face_yolov8n.pt", 0.3))

    def test_new_arguments_are_appended_last(self):
        self.assertEqual(tuple(ALL_ARGS[-3:]), tuple(zip(FACE_ATTRS, FACE_NAMES)))
        self.assertEqual(ALL_ARGS.attrs[:-3][-1], "sam3_cn_threshold_b", "앞의 순서는 예전 그대로")
        self.assertEqual(len(ALL_ARGS), 52)

    def test_coercion(self):
        self.assertEqual(Sam3Args(sam3_face_crop_confidence=1.7).sam3_face_crop_confidence, 1.0)
        self.assertEqual(Sam3Args(sam3_face_crop_confidence="x").sam3_face_crop_confidence, 0.3)
        self.assertEqual(Sam3Args(sam3_face_crop_confidence="0.55").sam3_face_crop_confidence, 0.55)
        self.assertEqual(Sam3Args(sam3_face_crop_model="  ").sam3_face_crop_model, "face_yolov8n.pt")
        self.assertEqual(Sam3Args(sam3_face_crop_model=None).sam3_face_crop_model, "face_yolov8n.pt")
        self.assertEqual(Sam3Args(sam3_face_crop_model=" face_yolov8s.pt ").sam3_face_crop_model, "face_yolov8s.pt")
        self.assertIs(Sam3Args(sam3_face_crop="True").sam3_face_crop, True)

    def test_infotext_has_the_keys_only_when_on(self):
        off = Sam3Args().extra_params()
        for name in FACE_NAMES:
            self.assertNotIn(name, off)
        on = Sam3Args(sam3_face_crop=True, sam3_face_crop_model="face_yolov8s.pt", sam3_face_crop_confidence=0.45)
        params = on.extra_params()
        self.assertEqual([params[name] for name in FACE_NAMES], [True, "face_yolov8s.pt", 0.45])
        self.assertEqual(list(params)[-3:], list(FACE_NAMES), "infotext 에서도 맨 뒤")

    def test_infotext_round_trip_through_forges_paste_conversion(self):
        """infotext 값은 문자열로 돌아온다 — Forge 붙여 넣기(infotext_utils.connect_paste)는 위젯 값의 형으로 바꾼다
        (bool 은 "False" 만 False). 그 값으로 다시 만든 Sam3Args 가 같아야 한다."""
        original = Sam3Args(sam3_face_crop=True, sam3_face_crop_model="face_yolov8s.pt", sam3_face_crop_confidence=0.45)
        params = original.extra_params()
        text = ", ".join(f"{name}: {value}" for name, value in params.items() if name in FACE_NAMES)
        parsed = dict(part.split(": ", 1) for part in text.split(", "))
        defaults = Sam3Args()
        restored = {}
        for attr, name in zip(FACE_ATTRS, FACE_NAMES):
            default = getattr(defaults, attr)
            value = parsed[name]
            if isinstance(default, bool):
                restored[attr] = value != "False"
            else:
                restored[attr] = type(default)(value)
        self.assertEqual({a: getattr(Sam3Args(**restored), a) for a in FACE_ATTRS},
                         {a: getattr(original, a) for a in FACE_ATTRS})

    def test_sam3_ui_widgets_and_paste_fields(self):
        import gradio as gr

        from sam3ext import ui

        with mock.patch.object(ui, "find_face_models", return_value=["face_yolov8n.pt", "face_yolov8s.pt"]), gr.Blocks():
            components, infotext_fields = ui.sam3_ui(False, ui.WebuiButtons(None, None))
        by_name = {name: widget for widget, name in infotext_fields}
        for name in FACE_NAMES:
            self.assertIn(name, by_name)
        self.assertEqual([name for _w, name in infotext_fields][-3:], list(FACE_NAMES))
        checkbox = by_name["SAM3 Face Crop"]
        self.assertIs(checkbox.value, False)
        self.assertTrue(checkbox.label.startswith("Search inside faces only"))
        labels = [str(getattr(w, "label", "") or "") for w in by_name.values()]
        hangul = [label for label in labels if any("가" <= ch <= "힣" for ch in label)]
        self.assertEqual(hangul, [], "SAM3 Mask 아코디언의 라벨은 이웃처럼 영어만")
        self.assertEqual(by_name["SAM3 Face Crop Model"].value, "face_yolov8n.pt")
        self.assertEqual(by_name["SAM3 Face Crop Confidence"].value, 0.3)
        self.assertEqual(by_name["SAM3 Face Crop Model"].elem_id, "sam3_txt2img_face_crop_model")
        state = components[1].value
        self.assertEqual([state[a] for a in FACE_ATTRS], [False, "face_yolov8n.pt", 0.3])

    def test_old_ui_value_lists_leave_face_crop_off(self):
        from sam3ext import ui

        # 예전 UI(위젯이 세 개 적던 화면)의 값 목록은 짧다 — zip 이 앞에서부터 맞추므로 앞의 값은 제자리, 새 키는 없음.
        values = [getattr(Sam3Args(), attr) for attr in ALL_ARGS.attrs[:-3]]
        values[ALL_ARGS.attrs.index("sam3_prompt")] = "eyes"
        state = ui.on_generate_click({}, True, *values)
        for attr in FACE_ATTRS:
            self.assertNotIn(attr, state)
        self.assertEqual(state["sam3_prompt"], "eyes")
        self.assertEqual(state["sam3_cn_threshold_b"], -1.0)
        self.assertIs(Sam3Args(**{k: v for k, v in state.items() if k in ALL_ARGS.attrs}).sam3_face_crop, False)
        # 하나만 짧아도(신뢰도 없음) 나머지 둘은 제자리
        values = [getattr(Sam3Args(), attr) for attr in ALL_ARGS.attrs[:-1]]
        values[-2] = True
        state = ui.on_generate_click({}, True, *values)
        self.assertIs(state["sam3_face_crop"], True)
        self.assertNotIn("sam3_face_crop_confidence", state)
        self.assertEqual(Sam3Args(**{k: v for k, v in state.items() if k in ALL_ARGS.attrs}).sam3_face_crop_confidence, 0.3)

    def test_default_face_model_choice(self):
        from sam3ext import ui

        self.assertEqual(ui.default_face_model(["hand_yolov8n.pt", "face_yolov8n.pt"]), "face_yolov8n.pt")
        self.assertEqual(ui.default_face_model(["face_yolov8s.pt"]), "face_yolov8s.pt")
        self.assertEqual(ui.default_face_model([]), "face_yolov8n.pt")


# ---------------------------------------------------------------------------------------------------------------------
# In-flight 스크립트(scripts/!sam3.py)
# ---------------------------------------------------------------------------------------------------------------------


from tests.test_sam3_script import _enabled_args, _processing, _sam3_script  # noqa: E402


def _result(size=(8, 8), *, masks=1, faces=1):
    mask_images = []
    for index in range(masks):
        mask = Image.new("L", size, 0)
        mask.paste(255, (index, 0, index + 1, 1))
        mask_images.append(mask)
    combined = Image.new("L", size, 255 if masks else 0)
    info = fc.FaceCropInfo(model="face_yolov8n.pt", confidence=0.3,
                           boxes=[[0, 0, 4, 4]] * faces, scores=[0.9] * faces,
                           crops=[fc.CropBox(0, 0, 8, 8)] * faces)
    return types.SimpleNamespace(mask=combined, masks=mask_images, overlay=Image.new("RGB", size), face_crop=info)


class ScriptFaceCropTests(unittest.TestCase):
    def setUp(self):
        self.module = _sam3_script()

    def _postprocess(self, args, result, params=None):
        p = _processing(extra_generation_params=dict(params or {"SAM3 Enable": True}), _sam3_args=args)
        pp = types.SimpleNamespace(image=Image.new("RGB", (8, 8), "white"))
        seen: dict = {}

        def fake_detect(**kwargs):
            seen["detect"] = kwargs
            return result

        def fake_inpaint(p_, image, masks, prompt, negative_prompt, args_, cn_args=None):
            seen["masks"] = masks
            return Image.new("RGB", image.size, "blue")

        with mock.patch.object(self.module, "run_sam3_on_pil", fake_detect), \
                mock.patch.object(self.module, "run_inpaint_passes", fake_inpaint), \
                mock.patch.object(self.module, "unload_sam3", lambda: True), \
                contextlib.redirect_stderr(io.StringIO()):
            self.module.Sam3MaskScript().postprocess_image(p, pp)
        return p, pp, seen

    def test_process_reads_ui_state_and_xyz(self):
        script = self.module.Sam3MaskScript()
        p = _processing()
        script.process(p, True, {"sam3_prompt": "eyes", "sam3_face_crop": True,
                                 "sam3_face_crop_model": "face_yolov8s.pt", "sam3_face_crop_confidence": 0.5})
        self.assertEqual([p._sam3_args[a] for a in FACE_ATTRS], [True, "face_yolov8s.pt", 0.5])
        self.assertEqual(p.extra_generation_params["SAM3 Face Crop Model"], "face_yolov8s.pt")
        p = _processing(_sam3_xyz={"sam3_face_crop": "True", "sam3_face_crop_confidence": "0.6"})
        script.process(p, True, {"sam3_prompt": "eyes"})
        self.assertIs(p._sam3_args["sam3_face_crop"], True)
        self.assertEqual(p._sam3_args["sam3_face_crop_confidence"], 0.6)
        p = _processing(_sam3_xyz={"sam3_face_crop": "False"})
        script.process(p, True, {"sam3_prompt": "eyes", "sam3_face_crop": True})
        self.assertIs(p._sam3_args["sam3_face_crop"], False, "XYZ 축이 UI 값을 이긴다")
        self.assertNotIn("SAM3 Face Crop", p.extra_generation_params)

    def test_old_api_state_without_the_keys_keeps_it_off(self):
        p = _processing()
        self.module.Sam3MaskScript().process(p, True, {"sam3_prompt": "face"})
        self.assertIs(p._sam3_args["sam3_face_crop"], False)
        self.assertNotIn("SAM3 Face Crop", p.extra_generation_params)
        self.assertNotIn("SAM3 Error", p.extra_generation_params)

    def test_a_stale_args_module_after_reload_ui_keeps_sam3_on(self):
        """Reload UI 는 스크립트만 새로 읽는다 — 옛 Sam3Args(extra=forbid, 얼굴 키 없음)면 얼굴 키를 빼고 예전처럼 돈다.

        실제와 같게 스크립트를 옛 Sam3Args 가 있는 채로 새로 실행한다 — 모듈 수준의 기본값 표(_SAM3_DEFAULTS)도 그
        클래스로 만들어진다(예전 테스트는 이미 만들어진 표 뒤에 클래스만 바꿔 KeyError 를 놓쳤다).
        """
        import sam3ext

        from tests.test_sam3_core_cache import _load_sam3_script

        seen: dict = {}

        class OldSam3Args:
            __fields__ = {name: None for name in Sam3Args.__fields__ if name not in FACE_ATTRS}

            def __init__(self, **kwargs):
                unknown = sorted(set(kwargs) - set(self.__fields__))
                if unknown:
                    raise ValueError(f"{unknown}: extra fields not permitted")
                seen.update(kwargs)
                self._args = Sam3Args(**kwargs)

            def dict(self):
                return {k: v for k, v in self._args.dict().items() if k not in FACE_ATTRS}

            def extra_params(self):
                return {k: v for k, v in self._args.extra_params().items() if k not in FACE_NAMES}

            def __getattr__(self, name):
                return getattr(self._args, name)

        with mock.patch.object(sam3ext, "Sam3Args", OldSam3Args), contextlib.redirect_stderr(io.StringIO()):
            module = _load_sam3_script([])
        self.assertIs(module.Sam3Args, OldSam3Args, "스크립트가 옛 클래스를 잡았다(Reload UI 와 같은 상태)")
        for state in ({"sam3_prompt": "face"}, {"sam3_prompt": "face", "sam3_face_crop": True}):
            with self.subTest(state=state):
                seen.clear()
                p = _processing()
                with contextlib.redirect_stderr(io.StringIO()):
                    module.Sam3MaskScript().process(p, True, state)
                self.assertIs(p._sam3_args["enabled"], True)
                self.assertNotIn("SAM3 Error", p.extra_generation_params)
                self.assertFalse(set(FACE_ATTRS) & set(seen), "옛 검증기에 얼굴 키를 넘기지 않는다")
                self.assertEqual(seen["sam3_prompt"], "face")
        p = _processing(_sam3_xyz={"sam3_face_crop": "True"})
        with contextlib.redirect_stderr(io.StringIO()):
            module.Sam3MaskScript().process(p, True, {"sam3_prompt": "face"})
        self.assertIs(p._sam3_args["enabled"], True, "XYZ 축에 얼굴 키가 있어도")

    def test_off_calls_detection_exactly_as_before(self):
        _p, _pp, seen = self._postprocess(_enabled_args(), _result())
        for key in ("face_crop", "face_crop_model", "face_crop_confidence"):
            self.assertNotIn(key, seen["detect"])

    def test_on_passes_the_detector_settings_and_records_the_face_count(self):
        args = _enabled_args(sam3_face_crop=True, sam3_face_crop_model="face_yolov8s.pt",
                             sam3_face_crop_confidence=0.45, sam3_mask_mode="Individual")
        result = _result(masks=3, faces=2)
        p, pp, seen = self._postprocess(args, result)
        self.assertEqual({k: seen["detect"][k] for k in ("face_crop", "face_crop_model", "face_crop_confidence")},
                         {"face_crop": True, "face_crop_model": "face_yolov8s.pt", "face_crop_confidence": 0.45})
        self.assertEqual(p.extra_generation_params[self.module.INFOTEXT_FACE_CROP_FACES], 2)
        self.assertIs(seen["masks"], result.masks, "Individual: 얼굴 × 그룹 마스크마다 인페인트 패스")

    def test_combined_inpaints_the_union_once(self):
        result = _result(masks=3, faces=2)
        _p, _pp, seen = self._postprocess(_enabled_args(sam3_face_crop=True, sam3_mask_mode="Combined"), result)
        self.assertEqual(seen["masks"], [result.mask])

    def test_no_faces_leaves_the_image_like_no_detection(self):
        result = _result(masks=0, faces=0)
        p, pp, seen = self._postprocess(_enabled_args(sam3_face_crop=True), result)
        self.assertNotIn("masks", seen, "인페인트 없음")
        self.assertEqual(pp.image.getpixel((0, 0)), (255, 255, 255))
        self.assertIs(p._sam3_mask_found, False)
        self.assertEqual(p.extra_generation_params[self.module.INFOTEXT_FACE_CROP_FACES], 0)
        self.assertIs(p.extra_generation_params["SAM3 Enable"], True)

    def test_the_face_crop_summary_is_left_for_the_quick_button(self):
        result = _result(masks=0, faces=0)
        p, _pp, _seen = self._postprocess(_enabled_args(sam3_face_crop=True), result)
        self.assertIs(p._sam3_face_crop_info, result.face_crop)
        p, _pp, _seen = self._postprocess(_enabled_args(), _result(masks=0, faces=0))
        self.assertIsNone(p._sam3_face_crop_info, "얼굴 crop 을 끄면 없음(결과에 요약이 실려 와도)")

    def test_quick_button_names_the_face_detector_when_no_faces(self):
        """🎯 은 얼굴을 못 찾은 경우를 '프롬프트·임계값 확인' 이 아니라 얼굴 검출기 쪽으로 안내한다."""
        from sam3ext import quick_button as qb
        from tests.test_quick_button import _FakeSam3Script, _Forge, _gallery

        class NoFaces(_FakeSam3Script):
            def __init__(self, info):
                super().__init__(mask=False)
                self.info = info

            def postprocess_image(self, p, pp, *args):
                super().postprocess_image(p, pp, *args)
                p._sam3_face_crop_info = self.info

        geninfo = {"infotexts": ["girl\nSeed: 111"]}
        none = fc.FaceCropInfo(model="face_<x>.pt", confidence=0.45, boxes=[], scores=[], crops=[])
        _gallery_out, _info, message, _ = _Forge(sam3=NoFaces(none)).run(_gallery("a"), 0, geninfo)
        self.assertIn("얼굴을 찾지 못했습니다 (face_&lt;x&gt;.pt, 신뢰도 0.45)", message)
        self.assertIn("Search inside faces only", message)
        self.assertNotIn("감지 프롬프트", message)
        found = fc.FaceCropInfo(model="face_yolov8n.pt", confidence=0.3, boxes=[[0, 0, 4, 4]], scores=[0.9],
                                crops=[fc.CropBox(0, 0, 8, 8)])
        for info in (found, None):
            with self.subTest(info=info):
                _g, _i, message, _ = _Forge(sam3=NoFaces(info)).run(_gallery("a"), 0, geninfo)
                self.assertEqual(message, qb.NO_MASK_MESSAGE, "얼굴은 찾았거나 얼굴 crop 이 꺼져 있으면 예전 안내")

    def test_the_face_count_belongs_to_each_image_of_a_batch(self):
        params = {"SAM3 Enable": True, self.module.INFOTEXT_FACE_CROP_FACES: 5}
        p, _pp, _seen = self._postprocess(_enabled_args(), _result(), params=params)
        self.assertNotIn(self.module.INFOTEXT_FACE_CROP_FACES, p.extra_generation_params,
                         "얼굴 crop 없이 돈 장에 앞 장의 얼굴 수가 남지 않는다")

    def test_xyz_axes(self):
        class AxisOption:
            def __init__(self, label, type, apply, **kwargs):
                self.label, self.type, self.apply, self.kwargs = label, type, apply, kwargs

        xyz = types.SimpleNamespace(AxisOption=AxisOption, axis_options=[], format_value=lambda *a: "")
        script_class = type("XYZ", (), {"__module__": "xyz_grid.py"})
        with mock.patch.object(self.module.scripts, "scripts_data",
                               [types.SimpleNamespace(script_class=script_class, module=xyz)]):
            self.module.make_axis_on_xyz_grid()
        by_label = {axis.label: axis for axis in xyz.axis_options}
        self.assertEqual(by_label["[SAM3] Face Crop"].kwargs["choices"](), ["True", "False"])
        self.assertIs(by_label["[SAM3] Face Crop Model"].kwargs["choices"], self.module.find_face_models)
        self.assertIs(by_label["[SAM3] Face Crop Confidence"].type, float)
        p = types.SimpleNamespace()
        by_label["[SAM3] Face Crop"].apply(p, "True", None)
        by_label["[SAM3] Face Crop Confidence"].apply(p, 0.5, None)
        self.assertEqual(p._sam3_xyz, {"sam3_face_crop": "True", "sam3_face_crop_confidence": 0.5})
        for label in ("[SAM3] Face Crop", "[SAM3] Face Crop Model", "[SAM3] Face Crop Confidence"):
            self.assertNotIn("cost", by_label[label].kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------------------------------------------------


from sam3ext import ui_refine  # noqa: E402
from tests.test_sam3_core_cache import _inpaint_core  # noqa: E402


class RefineFaceCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inpaint_core = _inpaint_core()

    def test_keys_sit_before_the_optional_canvas(self):
        keys = ui_refine.REFINE_ARG_KEYS
        self.assertEqual(keys[-5:], ("face_crop", "face_crop_model", "face_crop_confidence",
                                     "canvas_background", "canvas_foreground"))
        self.assertEqual(keys[keys.index("insert_mode") + 1], "face_crop")

    def test_panel_widgets_line_up_with_the_keys(self):
        import gradio as gr

        with mock.patch.object(ui_refine, "find_face_models", return_value=["face_yolov8n.pt"]), gr.Blocks():
            panel = ui_refine.build_refine_panel(["Euler a"], ["Simple"], ["sam3.pt"])
        widgets = panel.all_widgets()
        keyed = dict(zip(ui_refine.REFINE_ARG_KEYS, widgets))
        self.assertIs(keyed["face_crop"], panel.face_crop)
        self.assertIs(keyed["face_crop_model"], panel.face_crop_model)
        self.assertIs(keyed["face_crop_confidence"], panel.face_crop_confidence)
        self.assertIn("얼굴 안에서만 찾기", panel.face_crop.label)
        self.assertEqual(panel.face_crop_model.elem_id, "sam3_refine_face_crop_model")
        args = ui_refine.map_widget_values_to_sam3_args(tuple(w.value for w in widgets))
        self.assertEqual([args[a] for a in FACE_ATTRS], [False, "face_yolov8n.pt", 0.3])

    def test_mapping(self):
        base = {key: None for key in ui_refine.REFINE_ARG_KEYS}
        base.update(face_crop=True, face_crop_model="face_yolov8s.pt", face_crop_confidence=0.5)
        args = ui_refine.map_widget_values_to_sam3_args(tuple(base[k] for k in ui_refine.REFINE_ARG_KEYS))
        self.assertEqual([args[a] for a in FACE_ATTRS], [True, "face_yolov8s.pt", 0.5])
        old = tuple(base[k] for k in ui_refine.REFINE_ARG_KEYS[: ui_refine.REFINE_ARG_KEYS.index("face_crop")])
        args = ui_refine.map_widget_values_to_sam3_args(old)
        self.assertEqual([args[a] for a in FACE_ATTRS], [False, "face_yolov8n.pt", 0.3], "값이 없으면 꺼짐")

    def _passes(self, args, result):
        seen: dict = {}

        def fake_run(**kwargs):
            seen.update(kwargs)
            return result

        with mock.patch.object(core, "run_sam3_on_pil", fake_run), contextlib.redirect_stderr(io.StringIO()):
            out = self.inpaint_core._run_sam3_refine_passes(
                Image.new("RGB", (8, 8)), args, user_mask=None, scripts_runner=None, script_args_template=[],
                allow_huggingface=False, sd_model=None, outpath_samples="", outpath_grids="", override_settings=None,
            )
        return out, seen

    def test_no_faces_reports_its_own_reason(self):
        args = {"sam3_prompt": "eyes", "sam3_threshold": 0.4, "sam3_checkpoint": "sam3.pt", "sam3_device": "cpu",
                "sam3_face_crop": True, "sam3_face_crop_model": "face_yolov8s.pt", "sam3_face_crop_confidence": 0.45}
        result = _result(masks=0, faces=0)
        result.face_crop = fc.FaceCropInfo(model="face_yolov8s.pt", confidence=0.45, boxes=[], scores=[], crops=[])
        out, seen = self._passes(args, result)
        self.assertEqual(seen["face_crop"], True)
        self.assertEqual(seen["face_crop_model"], "face_yolov8s.pt")
        self.assertEqual(out.reason, "no_faces")
        self.assertFalse(out)
        message = ui_refine._refine_empty_message(out)
        self.assertIn("얼굴을 찾지 못했습니다 (face_yolov8s.pt, 신뢰도 0.45)", message)

    def test_manual_mask_with_no_faces_is_inpainted_like_no_detection(self):
        """Refine 의 Manual Mask: 얼굴이 없어도 '검출 없음' 처럼 칠한 마스크로 인페인트한다(no_faces 로 끝내지 않음)."""
        rgb = _two_faces_image()
        user = np.zeros(rgb.shape[:2], dtype=bool)
        user[20:60, 20:60] = True
        args = {"sam3_prompt": "hair", "sam3_threshold": 0.4, "sam3_checkpoint": "sam3.pt", "sam3_device": "cpu",
                "sam3_exclude_prompt": "mouth", "sam3_face_crop": True}
        reached: dict = {}

        class Reached(Exception):
            pass

        def inject(p2, args_):
            reached["mask"] = p2.image_mask
            reached["params"] = dict(p2.extra_generation_params)
            raise Reached

        detector = _Run(self, [])
        with mock.patch.object(core, "resolve_checkpoint_path", return_value=None), \
                mock.patch.object(core, "_get_model_bundle", side_effect=AssertionError("SAM3 bundle must not load")), \
                mock.patch.object(core, "detect_faces", side_effect=detector._detect), \
                mock.patch.object(self.inpaint_core, "build_standalone_i2i",
                                  lambda *a, **k: types.SimpleNamespace(extra_generation_params={})), \
                mock.patch.object(self.inpaint_core, "inject_controlnet_unit", inject), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(Reached):
                self.inpaint_core._run_sam3_refine_passes(
                    Image.fromarray(rgb), args, user_mask=user, scripts_runner=None, script_args_template=[],
                    allow_huggingface=False, sd_model=None, outpath_samples="", outpath_grids="", override_settings=None,
                )
        np.testing.assert_array_equal(_as_bool(reached["mask"]), user)
        self.assertEqual(reached["params"]["SAM3 Refine Face Crop Faces"], 0)
        self.assertIs(reached["params"]["SAM3 Refine Manual Mask"], True)

    def test_off_calls_detection_exactly_as_before(self):
        args = {"sam3_prompt": "eyes", "sam3_threshold": 0.4, "sam3_checkpoint": "sam3.pt", "sam3_device": "cpu"}
        empty = types.SimpleNamespace(mask=Image.new("L", (8, 8)), masks=[], face_crop=None)
        out, seen = self._passes(args, empty)
        self.assertNotIn("face_crop", seen)
        self.assertEqual(out.reason, "empty_mask")

    def test_png_metadata(self):
        mask = Image.new("L", (4, 4), 255)
        result = types.SimpleNamespace(checkpoint="x", device="cpu", face_crop=fc.FaceCropInfo(
            model="face_yolov8n.pt", confidence=0.3, boxes=[[0, 0, 2, 2]], scores=[0.9], crops=[fc.CropBox(0, 0, 4, 4)]))
        on = self.inpaint_core._refine_generation_params(
            {"sam3_face_crop": True, "sam3_face_crop_model": "face_yolov8n.pt", "sam3_face_crop_confidence": 0.3},
            result, mask, None, 1)
        self.assertEqual((on["SAM3 Refine Face Crop"], on["SAM3 Refine Face Crop Model"],
                          on["SAM3 Refine Face Crop Confidence"], on["SAM3 Refine Face Crop Faces"]),
                         (True, "face_yolov8n.pt", 0.3, 1))
        off = self.inpaint_core._refine_generation_params({}, types.SimpleNamespace(checkpoint="x", device="cpu"), mask, None, 1)
        self.assertFalse([key for key in off if "Face Crop" in key])

    def test_unavailable_detector_is_shown_in_the_status_line(self):
        from tests.test_ui_refine import RefineHandlerTests, _refine_values

        def boom(*a, **k):
            raise fd.FaceDetectorUnavailable("face crop: face detector model 'face_yolov8s.pt' not found — <dir>")

        with mock.patch.object(ui_refine, "find_face_models", return_value=["face_yolov8n.pt"]):
            values = _refine_values(detect_prompt="eyes", face_crop=True)
        outputs, calls = RefineHandlerTests._click(self, values, refine=boom)
        self.assertEqual(calls, [("sam3_refine", True)])
        self.assertIn("얼굴 검출기를 쓸 수 없습니다", outputs[1])
        self.assertIn("face_yolov8s.pt", outputs[1])
        self.assertIn("&lt;dir&gt;", outputs[1], "메시지는 HTML 이스케이프")


if __name__ == "__main__":
    unittest.main()
