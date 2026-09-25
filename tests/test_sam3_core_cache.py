"""효율 감사(2026-09-23) SAM3 항목: 'Unload after' CPU 캐시([1]), set_image 한 번(M2), bool 마스크 직접 변환,
overlay PNG 저속 압축(supp-06), VRAM 회수 한 번(L78), '[SAM3] Checkpoint' XYZ cost.

GPU 없이 돈다 — '다른 장치로 옮김' 은 float32 ↔ float64 로 흉내 낸다(값이 정확히 되돌아오는지 보기 위해).
실제 SAM3 구조(무작위 초기화)로 옛/새 run_sam3_on_pil 을 비교한 검증은 보고서에 적었다(시간이 걸려 여기엔 없음).
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

core = importlib.import_module("sam3ext.core")


# ---------------------------------------------------------------------------
# 가짜 번들
# ---------------------------------------------------------------------------


class _FakeChild(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        # sam3 의 PositionEmbeddingSine.cache / TransformerDecoder.compilable_cord_cache 같은 state_dict 밖 텐서
        self.cache = {(2, 2): torch.full((2, 2), 0.5)}
        self.coords = (torch.linspace(0, 1, 3), torch.linspace(0, 1, 4))
        self.nested = [torch.zeros(1), {"k": torch.ones(1)}]
        self.device_hint = torch.device("cpu")


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)
        self.register_buffer("buf", torch.arange(4, dtype=torch.float32))
        self.register_buffer("ids", torch.arange(3))
        self.child = _FakeChild()
        self.plain = torch.ones(2)


@dataclasses.dataclass
class _FindStage:
    img_ids: torch.Tensor
    text_ids: torch.Tensor
    input_boxes: object = None


class _FakeProcessor:
    def __init__(self, model):
        self.model = model
        self.transform = torch.nn.Identity()
        self.find_stage = _FindStage(torch.tensor([0]), torch.tensor([0]))
        self.scale = torch.tensor([2.0])


def _float_slots(model, processor):
    """테스트가 확인할 float 텐서 자리들(이름 → 텐서)."""
    return {
        "lin.weight": model.lin.weight,
        "lin.bias": model.lin.bias,
        "child.proj.weight": model.child.proj.weight,
        "child.proj.bias": model.child.proj.bias,
        "buf": model.buf,
        "plain": model.plain,
        "child.cache": model.child.cache[(2, 2)],
        "child.coords0": model.child.coords[0],
        "child.coords1": model.child.coords[1],
        "child.nested0": model.child.nested[0],
        "child.nested1.k": model.child.nested[1]["k"],
        "processor.scale": processor.scale,
    }


def _park_as_float64(model, processor):
    return core._park_bundle_tensors(
        model,
        processor,
        needs_park=lambda t: t.dtype == torch.float32,
        park=lambda t: t.to(torch.float64),
        origin=lambda t: t.dtype,
    )


class ParkRestoreTests(unittest.TestCase):
    """unload 가 번들을 CPU 에 남길 때 파라미터·버퍼뿐 아니라 state_dict 밖 텐서까지 옮기고 정확히 되돌리는지."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = _FakeModel()
        self.processor = _FakeProcessor(self.model)

    def test_park_reaches_parameters_buffers_and_plain_attribute_caches(self):
        params_before = list(self.model.parameters())
        cache_dict = self.model.child.cache
        nested_list = self.model.child.nested
        saved = _park_as_float64(self.model, self.processor)

        for name, tensor in _float_slots(self.model, self.processor).items():
            self.assertEqual(tensor.dtype, torch.float64, f"{name} 가 옮겨지지 않았다(GPU 에 남는 자리)")
        self.assertEqual(self.model.ids.dtype, torch.int64, "대상이 아닌 텐서는 그대로")
        self.assertEqual(self.processor.find_stage.img_ids.dtype, torch.int64)
        self.assertEqual(len(saved), len(_float_slots(self.model, self.processor)))
        self.assertTrue(all(a is b for a, b in zip(params_before, self.model.parameters())), "Parameter 객체는 유지")
        self.assertIs(self.model.child.cache, cache_dict, "dict 캐시는 제자리에서 바뀐다")
        self.assertIs(self.model.child.nested, nested_list)
        self.assertIsInstance(self.model.child.coords, tuple)

    def test_restore_returns_exact_values_and_dtypes(self):
        before = {k: v.detach().clone() for k, v in _float_slots(self.model, self.processor).items()}
        saved = _park_as_float64(self.model, self.processor)
        core._restore_bundle(self.model, self.processor, saved)
        after = _float_slots(self.model, self.processor)
        for name, tensor in before.items():
            self.assertEqual(after[name].dtype, torch.float32, name)
            self.assertTrue(torch.equal(after[name], tensor), name)

    def test_find_stage_tensors_are_walked(self):
        saved = core._park_bundle_tensors(
            self.model,
            self.processor,
            needs_park=lambda t: t.dtype == torch.int64,
            park=lambda t: t.to(torch.int32),
            origin=lambda t: t.dtype,
        )
        self.assertEqual(self.processor.find_stage.img_ids.dtype, torch.int32)
        self.assertEqual(self.model.ids.dtype, torch.int32)
        core._restore_bundle(self.model, self.processor, saved)
        self.assertEqual(self.processor.find_stage.text_ids.dtype, torch.int64)
        self.assertTrue(torch.equal(self.model.ids, torch.arange(3)))

    def test_discard_mechanism_frees_storage_via_meta(self):
        # _discard_bundle 은 GPU 텐서만 meta 로 바꾼다 — 같은 경로로 CPU 텐서를 바꿔 모든 자리에 닿는지 본다.
        core._park_bundle_tensors(
            self.model, self.processor,
            needs_park=lambda t: t.dtype == torch.float32,
            park=lambda t: t.to("meta"),
            origin=lambda t: t.device,
        )
        for name, tensor in _float_slots(self.model, self.processor).items():
            self.assertTrue(tensor.is_meta, name)
        core._discard_bundle(self.model, self.processor)   # meta·CPU 만 남았으니 아무 일 없음

    def test_offload_on_cpu_moves_nothing(self):
        plain = self.model.plain
        weight_data = self.model.lin.weight.data
        self.assertEqual(core._offload_bundle(self.model, self.processor), {})
        self.assertIs(self.model.plain, plain)
        self.assertEqual(self.model.lin.weight.data.data_ptr(), weight_data.data_ptr())

    def test_real_model_output_unchanged_after_round_trip(self):
        x = torch.randn(4, 3)
        expected = self.model.lin(x)
        core._restore_bundle(self.model, self.processor, _park_as_float64(self.model, self.processor))
        self.assertTrue(torch.equal(self.model.lin(x), expected))


# ---------------------------------------------------------------------------
# 번들 캐시(키·무효화)
# ---------------------------------------------------------------------------


class BundleCacheTests(unittest.TestCase):
    def setUp(self):
        self.builds: list[tuple[str, str]] = []
        self.offloads: list = []
        self.restores: list = []
        self.discards: list = []

        def build(key, device):
            self.builds.append((key, device))
            model = _FakeModel()
            return model, _FakeProcessor(model)

        def offload(model, processor):
            self.offloads.append(model)
            return {id(model): "cuda"}

        def restore(model, processor, saved):
            self.restores.append((model, dict(saved)))

        self.patches = [
            mock.patch.object(core, "_load_model_bundle", build),
            mock.patch.object(core, "_offload_bundle", offload),
            mock.patch.object(core, "_restore_bundle", restore),
            mock.patch.object(core, "_discard_bundle", lambda model, processor: self.discards.append(model)),
            mock.patch.object(core, "_empty_cuda_cache", lambda: None),
        ]
        for patch in self.patches:
            patch.start()
        core.release_sam3()
        self.offloads.clear()
        self.discards.clear()

    def tearDown(self):
        core.release_sam3()
        for patch in reversed(self.patches):
            patch.stop()

    def test_same_key_reuses_the_bundle(self):
        first = core._get_model_bundle("__hf__", "cuda")
        self.assertIs(core._get_model_bundle("__hf__", "cuda"), first)
        self.assertEqual(len(self.builds), 1)
        self.assertEqual(self.restores, [])

    def test_unload_keeps_the_bundle_and_next_detection_only_moves_it_back(self):
        bundle = core._get_model_bundle("__hf__", "cuda")
        core.unload_sam3()
        self.assertIs(core._BUNDLE, bundle, "번들을 버리면 다음 이미지가 처음부터 다시 빌드한다(감사 [1])")
        self.assertEqual(core._BUNDLE_OFFLOADED, {id(bundle[0]): "cuda"})
        core.unload_sam3()
        self.assertEqual(len(self.offloads), 1, "이미 내려간 번들은 다시 옮기지 않는다")

        self.assertIs(core._get_model_bundle("__hf__", "cuda"), bundle)
        self.assertEqual(len(self.builds), 1)
        self.assertEqual(self.restores, [(bundle[0], {id(bundle[0]): "cuda"})])
        self.assertIsNone(core._BUNDLE_OFFLOADED)

    def test_checkpoint_or_device_change_releases_and_rebuilds(self):
        first = core._get_model_bundle("C:/models/sam3/a.pt", "cuda")
        second = core._get_model_bundle("C:/models/sam3/b.pt", "cuda")
        self.assertIsNot(second, first)
        self.assertEqual(self.discards, [first[0]], "옛 번들은 버리기 전에 VRAM 을 푼다")
        self.assertEqual(self.offloads, [], "버릴 번들을 CPU 로 복사하지 않는다")
        third = core._get_model_bundle("C:/models/sam3/b.pt", "cpu")
        self.assertIsNot(third, second)
        self.assertEqual(len(self.builds), 3)

    def test_changed_checkpoint_file_rebuilds_even_with_the_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sam3.pt"
            path.write_bytes(b"v1")
            first = core._get_model_bundle(str(path), "cuda")
            self.assertIs(core._get_model_bundle(str(path), "cuda"), first)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
            self.assertIsNot(core._get_model_bundle(str(path), "cuda"), first, "같은 경로의 새 파일은 새로 빌드")
            path.write_bytes(b"version 2")
            core._get_model_bundle(str(path), "cuda")
            self.assertEqual(len(self.builds), 3)
            core.release_sam3()

    def test_release_drops_the_bundle(self):
        core._get_model_bundle("__hf__", "cuda")
        core.unload_sam3()
        core.release_sam3()
        self.assertIsNone(core._BUNDLE)
        self.assertIsNone(core._BUNDLE_OFFLOADED)
        core._get_model_bundle("__hf__", "cuda")
        self.assertEqual(len(self.builds), 2)

    def test_failed_restore_releases_and_raises(self):
        core._get_model_bundle("__hf__", "cuda")
        core.unload_sam3()
        with mock.patch.object(core, "_restore_bundle", side_effect=RuntimeError("CUDA out of memory")):
            with self.assertRaises(RuntimeError):
                core._get_model_bundle("__hf__", "cuda")
        self.assertIsNone(core._BUNDLE, "반쯤 올라간 번들을 캐시에 두면 안 된다")
        core._get_model_bundle("__hf__", "cuda")
        self.assertEqual(len(self.builds), 2)

    def test_failed_offload_falls_back_to_release(self):
        core._get_model_bundle("__hf__", "cuda")
        with mock.patch.object(core, "_offload_bundle", side_effect=RuntimeError("boom")):
            self.assertIs(core.unload_sam3(), False, "버렸으면 RAM 에 보관했다고 말하면 안 된다")
        self.assertIsNone(core._BUNDLE)

    # --- 후속: 'Unload after' 뒤 RAM 보관을 끄는 Forge 설정 ---------------------------------

    def _forge_opts(self, **values):
        package = types.ModuleType("modules")
        package.__path__ = []
        shared = types.ModuleType("modules.shared")
        shared.opts = types.SimpleNamespace(**values)
        package.shared = shared
        return mock.patch.dict(sys.modules, {"modules": package, "modules.shared": shared})

    def test_unload_reports_that_the_bundle_is_kept_in_ram(self):
        bundle = core._get_model_bundle("__hf__", "cuda")
        self.assertIs(core.unload_sam3(), True)
        self.assertIs(core._BUNDLE, bundle)
        self.assertIs(core.unload_sam3(), True, "이미 내려가 있어도 RAM 에 있다")

    def test_unload_without_a_bundle_reports_nothing_kept(self):
        self.assertIs(core.unload_sam3(), False)

    def test_keep_in_ram_off_releases_like_before(self):
        bundle = core._get_model_bundle("__hf__", "cuda")
        self.assertIs(core.unload_sam3(keep_in_ram=False), False)
        self.assertIsNone(core._BUNDLE)
        self.assertEqual(self.offloads, [], "버릴 번들을 CPU 로 복사하지 않는다")
        self.assertEqual(self.discards, [bundle[0]])
        core._get_model_bundle("__hf__", "cuda")
        self.assertEqual(len(self.builds), 2, "예전처럼 다음 검출은 새로 빌드")

    def test_setting_is_read_from_forge_opts(self):
        core._get_model_bundle("__hf__", "cuda")
        with self._forge_opts(**{core.OPT_UNLOAD_KEEP_IN_RAM: False}):
            self.assertIs(core.unload_sam3(), False)
        self.assertIsNone(core._BUNDLE)

    def test_setting_defaults_to_keeping_the_bundle(self):
        bundle = core._get_model_bundle("__hf__", "cuda")
        with self._forge_opts():   # 아직 저장된 적 없는 설정 → 지금 동작(보관)
            self.assertIs(core.unload_sam3(), True)
        self.assertIs(core._BUNDLE, bundle)

    def test_drop_offloaded_releases_only_a_bundle_parked_in_ram(self):
        bundle = core._get_model_bundle("__hf__", "cuda")
        self.assertIs(core.drop_offloaded_sam3(), False, "장치에 올라가 쓰이는 번들은 건드리지 않는다")
        self.assertIs(core._BUNDLE, bundle)
        core.unload_sam3()
        self.assertIs(core.drop_offloaded_sam3(), True)
        self.assertIsNone(core._BUNDLE)
        self.assertIs(core.drop_offloaded_sam3(), False)

    def test_describe_unload(self):
        self.assertEqual(core.describe_unload(True), "model moved from VRAM to CPU RAM (moves back on next detection).")
        self.assertEqual(core.describe_unload(False), "model released from VRAM and RAM (reloads on next detection).")
        self.assertEqual(
            core.describe_unload(True, after_failure=True),
            "model moved from VRAM to CPU RAM after the failure (moves back on next detection).",
        )
        self.assertEqual(
            core.describe_unload(False, after_failure=True),
            "model released from VRAM and RAM after the failure (reloads on next detection).",
        )

    def test_run_sam3_on_pil_does_not_rebuild_after_unload(self):
        processor = _GroundingProcessor()
        with mock.patch.object(core, "_load_model_bundle", lambda key, device: (self.builds.append(key), (None, processor))[1]):
            with tempfile.TemporaryDirectory() as tmp:
                ckpt = Path(tmp) / "sam3.pt"
                ckpt.write_bytes(b"x")
                for _ in range(3):
                    core.run_sam3_on_pil(_image(), "face", 0.4, str(ckpt), "cpu", allow_huggingface=False)
                    core.unload_sam3()
                core.release_sam3()
        self.assertEqual(len(self.builds), 1)


# ---------------------------------------------------------------------------
# M2: 이미지 백본은 이미지당 한 번
# ---------------------------------------------------------------------------


def _image(width: int = 40, height: int = 32) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack([xx * 6 % 256, yy * 8 % 256, (xx + yy) * 3 % 256], -1).astype(np.uint8)
    return Image.fromarray(rgb)


class _GroundingProcessor:
    """Sam3Processor 흉내 — set_text_prompt 는 진짜처럼 backbone_out.update(텍스트) 와 state[...] = 결과."""

    IMAGE_KEYS = {"original_height", "original_width", "backbone_out"}
    BACKBONE_KEYS = {"backbone_fpn", "vision_pos_enc"}

    def __init__(self):
        self.set_image_calls = 0
        self.seen: list[tuple[set, set]] = []

    def set_confidence_threshold(self, threshold, state=None):
        self.threshold = threshold

    def set_image(self, image, state=None):
        self.set_image_calls += 1
        state = {} if state is None else state
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32))
        state["original_height"], state["original_width"] = pixels.shape[:2]
        state["backbone_out"] = {"backbone_fpn": [pixels.mean(-1)], "vision_pos_enc": [torch.zeros(1)]}
        return state

    def set_text_prompt(self, prompt, state):
        self.seen.append((set(state), set(state["backbone_out"])))
        state["backbone_out"].update({"language_features": torch.tensor([float(len(prompt))])})
        state.setdefault("geometric_prompt", object())
        h, w = state["original_height"], state["original_width"]
        seed = sum(map(ord, prompt))
        y0, x0 = seed % (h // 2), (seed * 7) % (w // 2)
        feat = state["backbone_out"]["backbone_fpn"][0]
        masks = torch.zeros(1, 1, h, w, dtype=torch.bool)
        masks[0, 0, y0:y0 + h // 3, x0:x0 + w // 3] = feat[y0:y0 + h // 3, x0:x0 + w // 3] >= 0
        state["masks"] = masks
        state["boxes"] = torch.tensor([[x0, y0, x0 + w // 3, y0 + h // 3]], dtype=torch.float32)
        state["scores"] = torch.tensor([0.5 + (seed % 10) / 100])
        return state


class _PerTokenProcessor(_GroundingProcessor):
    """옛 동작(토큰마다 set_image 새로)의 기준 — 받은 state 를 버리고 이미지부터 다시 한다."""

    def __init__(self, image):
        super().__init__()
        self.image = image

    def set_text_prompt(self, prompt, state):
        return super().set_text_prompt(prompt, self.set_image(self.image))


class SetImageOnceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ckpt = Path(self.tmp.name) / "sam3.pt"
        self.ckpt.write_bytes(b"x")
        core.release_sam3()

    def tearDown(self):
        core.release_sam3()
        self.tmp.cleanup()

    def _run(self, processor, **kwargs):
        with mock.patch.object(core, "_load_model_bundle", lambda key, device: (None, processor)):
            result = core.run_sam3_on_pil(
                _image(), kwargs.pop("prompt", "face, hair / hand"), 0.4, str(self.ckpt), "cpu",
                allow_huggingface=False, **kwargs,
            )
        core.release_sam3()
        return result

    def test_backbone_runs_once_per_image_including_exclude_tokens(self):
        processor = _GroundingProcessor()
        self._run(processor, exclude_prompt="eyes, hand")
        self.assertEqual(processor.set_image_calls, 1)
        self.assertEqual(len(processor.seen), 5, "검출 3 + 제외 2 토큰")

    def test_every_token_sees_a_fresh_post_set_image_state(self):
        processor = _GroundingProcessor()
        self._run(processor, exclude_prompt="eyes")
        for state_keys, backbone_keys in processor.seen:
            self.assertEqual(state_keys, _GroundingProcessor.IMAGE_KEYS, "이전 토큰의 결과·geometric_prompt 가 새면 안 된다")
            self.assertEqual(backbone_keys, _GroundingProcessor.BACKBONE_KEYS, "이전 토큰의 텍스트 특징이 새면 안 된다")

    def test_result_matches_the_per_token_set_image_reference(self):
        for kwargs in ({}, {"exclude_prompt": "eyes"}, {"mask_dilation": 2, "mask_hull": True}):
            new = self._run(_GroundingProcessor(), **dict(kwargs))
            old = self._run(_PerTokenProcessor(_image().convert("RGB")), **dict(kwargs))
            self.assertTrue(np.array_equal(np.asarray(new.mask), np.asarray(old.mask)), kwargs)
            self.assertEqual(len(new.masks), len(old.masks))
            for a, b in zip(new.masks, old.masks):
                self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)), kwargs)
            self.assertTrue(np.array_equal(np.asarray(new.overlay), np.asarray(old.overlay)), kwargs)
            self.assertEqual(new.boxes, old.boxes)
            self.assertEqual(new.scores, old.scores)


# ---------------------------------------------------------------------------
# TF32: sam3 import 가 Forge 전역 설정을 바꾸지 않게 (2026-09-23 GPU 렌더에서 발견)
# ---------------------------------------------------------------------------


class _TF32State:
    """테스트마다 전역 TF32 설정을 저장·복원한다."""

    def setUp(self):
        self._saved = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False

    def tearDown(self):
        torch.set_float32_matmul_precision(self._saved[0])
        torch.backends.cudnn.allow_tf32 = self._saved[1]


def _flags():
    return torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32


class Tf32IsolationTests(_TF32State, unittest.TestCase):
    def test_import_side_effect_is_undone_after_building_the_bundle(self):
        # sam3/model_builder.py 의 _setup_tf32() 가 import 때 하는 일을 흉내 낸다.
        def build(key, device):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return None, object()

        with mock.patch.object(core, "_build_model_bundle", build):
            core._load_model_bundle("x", "cpu")
        self.assertEqual(_flags(), (False, False))
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_flags_are_restored_even_when_the_build_fails(self):
        def build(key, device):
            torch.backends.cuda.matmul.allow_tf32 = True
            raise RuntimeError("boom")

        with mock.patch.object(core, "_build_model_bundle", build), self.assertRaises(RuntimeError):
            core._load_model_bundle("x", "cpu")
        self.assertEqual(_flags(), (False, False))

    def test_user_chosen_tf32_is_kept(self):
        torch.backends.cuda.matmul.allow_tf32 = True  # Forge/사용자가 이미 켜 둔 경우는 그대로
        with mock.patch.object(core, "_build_model_bundle", lambda key, device: (None, object())):
            core._load_model_bundle("x", "cpu")
        self.assertEqual(_flags(), (True, False))

    def test_detection_runs_with_tf32_on_ampere_and_restores_it(self):
        seen = []

        class Recording(_GroundingProcessor):
            def set_image(self, image, state=None):
                seen.append(("image", _flags()))
                return super().set_image(image, state)

            def set_text_prompt(self, prompt, state):
                seen.append(("text", _flags()))
                return super().set_text_prompt(prompt, state)

        props = types.SimpleNamespace(major=12)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "sam3.pt"
            ckpt.write_bytes(b"x")
            with mock.patch.object(core, "_load_model_bundle", lambda key, device: (None, Recording())),                     mock.patch.object(torch.cuda, "is_available", lambda: True),                     mock.patch.object(torch.cuda, "get_device_properties", lambda index: props):
                core.run_sam3_on_pil(_image(), "face / hand", 0.4, str(ckpt), "cpu", allow_huggingface=False)
            core.release_sam3()
        self.assertTrue(seen)
        self.assertTrue(all(flags == (True, True) for _, flags in seen), seen)  # SAM3 는 예전처럼 TF32
        self.assertEqual(_flags(), (False, False))  # 검출이 끝나면 Forge 설정으로


# ---------------------------------------------------------------------------
# bool 마스크 직접 변환, overlay PNG
# ---------------------------------------------------------------------------


class BoolMaskTests(unittest.TestCase):
    def test_bool_tensor_converts_directly_and_matches_the_float_round_trip(self):
        torch.manual_seed(1)
        masks = torch.rand(3, 1, 9, 11) > 0.5
        direct = core._to_numpy(masks)
        self.assertEqual(direct.dtype, np.bool_)
        old = masks.detach().float().cpu().numpy()
        self.assertTrue(np.array_equal(direct, old.astype(bool)))
        new_split = core._split_masks(direct, 9, 11)
        old_split = core._split_masks(old, 9, 11)
        self.assertEqual(len(new_split), len(old_split))
        for a, b in zip(new_split, old_split):
            self.assertTrue(np.array_equal(a, b))

    def test_float_tensors_still_become_float32(self):
        out = core._to_numpy(torch.tensor([[1.5, 2.0]], dtype=torch.bfloat16))
        self.assertEqual(out.dtype, np.float32)


def _zlib_level(png_path: Path) -> int:
    """첫 IDAT 의 zlib 헤더 FLEVEL — 0: 레벨 0~1, 1: 레벨 2~5, 2: 레벨 6(기본), 3: 레벨 7~9."""
    data = png_path.read_bytes()
    index = data.index(b"IDAT")
    return data[index + 5] >> 6


class ArtifactPngTests(unittest.TestCase):
    def test_overlay_is_saved_fast_with_identical_pixels(self):
        rng = np.random.default_rng(0)
        overlay = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        mask = np.zeros((48, 64), np.uint8)
        mask[10:30, 20:40] = 255
        result = core.Sam3Result(
            mask=Image.fromarray(mask, "L"),
            masks=[Image.fromarray(mask, "L")],
            overlay=Image.fromarray(overlay),
            boxes=[[20.0, 10.0, 40.0, 30.0]],
            scores=[0.9],
            device="cpu",
            checkpoint="sam3.pt",
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(core, "DEFAULT_OUTPUT_DIR", Path(tmp)):
            paths = core.write_artifacts(result, 7, label="face")
            with Image.open(paths["overlay"]) as saved:
                self.assertTrue(np.array_equal(np.asarray(saved), overlay), "PNG 는 무손실 — 픽셀이 같아야 한다")
            with Image.open(paths["mask"]) as saved:
                self.assertTrue(np.array_equal(np.asarray(saved), mask))
            self.assertEqual(_zlib_level(Path(paths["overlay"])), 1, "overlay 는 빠른 레벨(compress_level=2)")
            self.assertEqual(_zlib_level(Path(paths["mask"])), 2, "마스크는 기본 압축 그대로")


# ---------------------------------------------------------------------------
# inpaint_core._reclaim_vram (기존 L78)
# ---------------------------------------------------------------------------


def _inpaint_core():
    if "sam3ext.inpaint_core" in sys.modules:
        return sys.modules["sam3ext.inpaint_core"]
    package = types.ModuleType("modules")
    package.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace()
    shared.cmd_opts = types.SimpleNamespace()
    shared.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False, textinfo=None)
    processing = types.ModuleType("modules.processing")
    processing.StableDiffusionProcessingImg2Img = object
    processing.process_images = lambda p: None
    package.shared = shared
    package.processing = processing
    stubs = {"modules": package, "modules.shared": shared, "modules.processing": processing}
    saved = {key: sys.modules.get(key) for key in stubs}
    sys.modules.update(stubs)
    try:
        return importlib.import_module("sam3ext.inpaint_core")
    finally:
        for key, original in saved.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


class ReclaimVramTests(unittest.TestCase):
    def setUp(self):
        self.inpaint_core = _inpaint_core()

    def test_forge_torch_gc_runs_once_and_nothing_else(self):
        calls: list[str] = []
        devices = types.SimpleNamespace(torch_gc=lambda: calls.append("torch_gc"))
        with mock.patch.object(self.inpaint_core, "_devices", devices), \
                mock.patch.object(torch.cuda, "is_available", lambda: True), \
                mock.patch.object(torch.cuda, "empty_cache", lambda: calls.append("empty_cache")), \
                mock.patch.object(torch.cuda, "ipc_collect", lambda: calls.append("ipc_collect")):
            self.inpaint_core._reclaim_vram()
        self.assertEqual(calls, ["torch_gc"], "soft_empty_cache 가 이미 empty_cache·ipc_collect 를 한다")

    def test_falls_back_to_torch_without_forge(self):
        for devices in (None, types.SimpleNamespace(), types.SimpleNamespace(torch_gc=mock.Mock(side_effect=RuntimeError))):
            calls: list[str] = []
            with mock.patch.object(self.inpaint_core, "_devices", devices), \
                    mock.patch.object(torch.cuda, "is_available", lambda: True), \
                    mock.patch.object(torch.cuda, "empty_cache", lambda: calls.append("empty_cache")), \
                    mock.patch.object(torch.cuda, "ipc_collect", lambda: calls.append("ipc_collect")):
                self.inpaint_core._reclaim_vram()
            self.assertEqual(calls, ["empty_cache", "ipc_collect"], devices)


# ---------------------------------------------------------------------------
# scripts/!sam3.py — XYZ cost, 스크립트 언로드 때 진짜 해제, Reload UI 의 옛 core
# ---------------------------------------------------------------------------


def _load_sam3_script(unloaded: list):
    package = types.ModuleType("modules")
    package.__path__ = []

    class Script:
        is_img2img = False

    scripts = types.ModuleType("modules.scripts")
    scripts.Script = Script
    scripts.AlwaysVisible = object()
    scripts.scripts_data = []
    script_callbacks = types.ModuleType("modules.script_callbacks")
    script_callbacks.callback_map = {}
    for name in ("on_before_ui", "on_app_started", "on_after_component", "on_ui_settings"):
        setattr(script_callbacks, name, lambda fn, **kw: None)
    script_callbacks.on_script_unloaded = lambda fn, **kw: unloaded.append(fn)
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace()
    shared.cmd_opts = types.SimpleNamespace()
    shared.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False, textinfo=None)
    processing = types.ModuleType("modules.processing")
    processing.StableDiffusionProcessingImg2Img = type("StableDiffusionProcessingImg2Img", (), {})
    processing.process_images = lambda p: None
    sd_samplers = types.ModuleType("modules.sd_samplers")
    sd_samplers.all_samplers = []
    sd_schedulers = types.ModuleType("modules.sd_schedulers")
    sd_schedulers.schedulers = []
    stubs = {"modules": package}
    for name, module in (
        ("scripts", scripts),
        ("script_callbacks", script_callbacks),
        ("shared", shared),
        ("processing", processing),
        ("sd_samplers", sd_samplers),
        ("sd_schedulers", sd_schedulers),
    ):
        setattr(package, name, module)
        stubs[f"modules.{name}"] = module
    saved = {key: sys.modules.get(key) for key in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_t_sam3_script_cache", ROOT / "scripts" / "!sam3.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for key, original in saved.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original
    return module


class _AxisOption:
    def __init__(self, label, type, apply, **kwargs):
        self.label = label
        self.kwargs = kwargs


class Sam3ScriptTests(unittest.TestCase):
    def test_script_unload_really_releases_the_bundle(self):
        unloaded: list = []
        module = _load_sam3_script(unloaded)
        self.assertIn(module.release_sam3, unloaded)
        self.assertIs(module.release_sam3, sys.modules["sam3ext.core"].release_sam3)

    def test_checkpoint_axis_gets_a_cost_and_other_axes_do_not(self):
        module = _load_sam3_script([])
        xyz = types.SimpleNamespace(AxisOption=_AxisOption, axis_options=[], format_value=lambda *a: "")
        script_class = type("XYZ", (), {"__module__": "xyz_grid.py"})
        module.scripts.scripts_data = [types.SimpleNamespace(script_class=script_class, module=xyz)]
        module.make_axis_on_xyz_grid()
        by_label = {axis.label: axis for axis in xyz.axis_options}
        self.assertEqual(by_label["[SAM3] Checkpoint"].kwargs.get("cost"), module.SAM3_CHECKPOINT_AXIS_COST)
        self.assertTrue(0.8 < module.SAM3_CHECKPOINT_AXIS_COST < 1.0, "Checkpoint(1.0) 과 [DoRA](0.8) 사이")
        # 장치가 바뀌어도 번들을 버리고 다시 빌드한다(_get_model_bundle 의 키) — 같은 cost.
        self.assertEqual(by_label["[SAM3] Device"].kwargs.get("cost"), module.SAM3_CHECKPOINT_AXIS_COST)
        rebuild_axes = {"[SAM3] Checkpoint", "[SAM3] Device"}
        others = [axis for label, axis in by_label.items() if label not in rebuild_axes]
        self.assertTrue(others)
        self.assertTrue(all("cost" not in axis.kwargs for axis in others))

    def test_reload_ui_with_a_stale_core_reloads_it_after_emptying_the_old_cache(self):
        # 진짜 sam3ext.core 는 건드리지 않는다 — 같은 파일로 따로 만든 '옛' 모듈을 잠시 끼운다.
        package = sys.modules["sam3ext"]
        real = sys.modules["sam3ext.core"]
        spec = importlib.util.spec_from_file_location("sam3ext.core", ROOT / "sam3ext" / "core.py")
        stale = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stale)
        del stale.release_sam3
        emptied: list = []
        stale.unload_sam3 = lambda: emptied.append(1)
        sys.modules["sam3ext.core"] = stale
        package.core = stale
        try:
            module = _load_sam3_script([])
            self.assertEqual(emptied, [1], "옛 모듈의 캐시 번들을 먼저 비운다")
            self.assertTrue(hasattr(stale, "release_sam3"), "옛 모듈을 새 코드로 다시 불러온다")
            self.assertIs(module.release_sam3, stale.release_sam3)
            self.assertIs(module.run_sam3_on_pil.__globals__["release_sam3"], stale.release_sam3,
                          "run_sam3_on_pil 도 새 캐시를 쓰는 함수여야 한다")
        finally:
            sys.modules["sam3ext.core"] = real
            package.core = real


# ---------------------------------------------------------------------------
# M3: 로컬에 없는 기본값 'sam3.pt' 는 HF(facebook/sam3) 자동 다운로드로(allow_huggingface 일 때만)
# ---------------------------------------------------------------------------


class HuggingFaceFallbackTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.models = self.root / "webui" / "models"
        (self.models / "sam3").mkdir(parents=True)
        self.ext = self.root / "ext"
        (self.ext / "models").mkdir(parents=True)
        paths = types.SimpleNamespace(models_path=str(self.models))
        for patch in (
            mock.patch.object(core, "_safe_import_webui_modules", lambda: paths),
            mock.patch.object(core, "EXTENSION_ROOT", self.ext),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_fresh_install_offers_only_the_hf_default(self):
        self.assertEqual(core.find_checkpoint_options(), [core.HF_CHECKPOINT_NAME])

    def test_missing_default_resolves_to_huggingface(self):
        self.assertIsNone(core.resolve_checkpoint_path("sam3.pt", allow_huggingface=True))
        self.assertIsNone(core.resolve_checkpoint_path(" SAM3.PT ", allow_huggingface=True))

    def test_local_default_still_wins(self):
        local = self.models / "sam3" / "sam3.pt"
        local.write_bytes(b"x")
        self.assertEqual(core.resolve_checkpoint_path("sam3.pt", allow_huggingface=True), local)

    def test_no_huggingface_keeps_the_missing_path(self):
        self.assertEqual(core.resolve_checkpoint_path("sam3.pt", allow_huggingface=False), Path("sam3.pt"))

    def test_other_missing_names_are_not_downloaded(self):
        for value in ("sam3.safetensors", "sam3.1_multiplex_fp16.safetensors", "sub/sam3.pt"):
            self.assertIsNotNone(core.resolve_checkpoint_path(value, allow_huggingface=True), value)

    def test_empty_and_auto_are_unchanged(self):
        for value in ("", "auto", "huggingface"):
            self.assertIsNone(core.resolve_checkpoint_path(value, allow_huggingface=True))
            self.assertEqual(core.resolve_checkpoint_path(value, allow_huggingface=False), Path("sam3.pt"))

    def _run(self, value, allow):
        keys = []
        processor = _GroundingProcessor()

        def get_bundle(key, device):
            keys.append(key)
            return None, processor

        with mock.patch.object(core, "_get_model_bundle", get_bundle):
            result = core.run_sam3_on_pil(_image(), "face", 0.4, value, "cpu", allow_huggingface=allow)
        return keys, result

    def test_detection_with_the_missing_default_loads_from_huggingface(self):
        keys, result = self._run("sam3.pt", True)
        self.assertEqual(keys, ["__hf__"])
        self.assertEqual(result.checkpoint, "facebook/sam3::sam3.pt")

    def test_no_huggingface_raises_a_clear_error(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self._run("sam3.pt", False)
        message = str(caught.exception)
        self.assertIn("SAM3 checkpoint not found", message, "기존 로그 문구(사용자 검색어)는 유지")
        self.assertIn("--sam3-no-huggingface", message)
        self.assertIn("models/sam3", message)

    def test_missing_custom_name_error_does_not_blame_the_flag(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self._run("sam3.safetensors", True)
        self.assertNotIn("--sam3-no-huggingface", str(caught.exception))

    def test_local_checkpoint_result_is_unchanged(self):
        local = self.models / "sam3" / "sam3.pt"
        local.write_bytes(b"x")
        keys, result = self._run("sam3.pt", True)
        self.assertEqual(keys, [str(local.resolve())])
        self.assertEqual(result.checkpoint, str(local))

    def test_hf_build_announces_the_download_and_local_build_does_not(self):
        import contextlib
        import io

        calls = []
        builder = types.ModuleType("sam3.model_builder")
        builder.build_sam3_image_model = lambda **kw: calls.append(kw) or object()
        proc_mod = types.ModuleType("sam3.model.sam3_image_processor")
        proc_mod.Sam3Processor = lambda model, device: ("processor", model)
        stubs = {"sam3": types.ModuleType("sam3"), "sam3.model": types.ModuleType("sam3.model"),
                 "sam3.model_builder": builder, "sam3.model.sam3_image_processor": proc_mod}
        for key, want_notice in (("__hf__", True), (str(self.models / "sam3" / "sam3.pt"), False)):
            out = io.StringIO()
            with mock.patch.dict(sys.modules, stubs), mock.patch.object(core, "_ensure_bpe_vocab", lambda: None), \
                    contextlib.redirect_stdout(out):
                core._build_model_bundle(key, "cpu")
            if want_notice:
                self.assertIn("facebook/sam3", out.getvalue())
                self.assertIn("--sam3-no-huggingface", out.getvalue())
            else:
                self.assertEqual(out.getvalue(), "", "로컬 체크포인트는 안내 없음")
        self.assertEqual((calls[0]["load_from_HF"], calls[0]["checkpoint_path"]), (True, None))
        self.assertEqual((calls[1]["load_from_HF"], calls[1]["checkpoint_path"]),
                         (False, str(self.models / "sam3" / "sam3.pt")))


if __name__ == "__main__":
    unittest.main()
