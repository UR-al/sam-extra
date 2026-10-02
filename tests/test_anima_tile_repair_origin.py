"""Anima Tile & Repair — 원본과 같게(UR_IV 원본 동등성 계획 §7.3 F, 패키지 TR-F).

원본: civitai 2708551 (Anima Tile & Repair ControlNet-LLLite v1.0/v2.0, 3채널 fp32) 를 돌리는
kohya-ss/sd-scripts@690ea7f9 의 anima_minimal_inference(_control_net_lllite).py 와
kohya-ss/ComfyUI-Anima-LLLite@b7495bd8 노드, 그리고 확장이 따라 한 ComfyUI 워크플로
(ResizeImagesByShorterEdge → AnimaLLLiteApply → KSampler).

- LLLite 목록: safetensors 헤더(JSON)만 읽어 cond_in_channels == 3 인 파일만. 4채널 인페인트 LLLite 는
  마스크 없이 실패한다(sd-scripts :400-406). 기본값은 가장 새 tilerepair 파일(v20).
- 크기: 원본 비율 유지, 짧은 변 = 슬라이더(기본 1024), 32 배수로 내림, 최소 256.
- 디코드: sd-scripts save_images 식으로 고정(범위 추측 없음).
- multiplier −10..10 step .01(기본 1.0), negative 기본 "".

원본 식은 아래에 그대로 옮기고 출처를 단다(sd-scripts·kohya 노드는 Apache-2.0, ComfyUI 는 GPL-3.0 — 확장은 GPL-3.0).
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import anima_core, ui_anima  # noqa: E402
from sam3ext.anima_core import (  # noqa: E402
    AnimaTileRepairArgs,
    _tensor_to_pil,
    decode_pixels_to_uint8,
    default_lllite_choice,
    list_lllite_choices,
    lllite_cond_in_channels,
    read_safetensors_header,
    tile_repair_size,
)
from sam3ext.ui_anima import ANIMA_ARG_KEYS, _map_widget_values, build_anima_panel  # noqa: E402


# ---------------------------------------------------------------------------
# 원본 식 (복사)
# ---------------------------------------------------------------------------

# origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:59-62
def _origin_read_lllite_metadata(weights_path: str):
    with safe_open(weights_path, framework="pt") as f:
        meta = f.metadata()
    return meta or {}


# origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:325-329
# (args.lllite_cond_in_channels 는 확장이 None 으로 넘긴다 → 메타데이터, 없으면 3)
def _origin_cond_in_channels(meta) -> int:
    return int(meta.get("lllite.cond_in_channels", 3))


# origin: kohya-ss/sd-scripts@690ea7f9:networks/control_net_lllite_anima.py:628-634
# (load_lllite_weights — 파일을 읽는 부분만. 확장자가 .safetensors 가 아니면 torch.load)
def _origin_load_lllite_state_dict(file: str):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")
    return weights_sd


# origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference.py:658-660 (save_images)
def _origin_save_images_array(sample: torch.Tensor) -> np.ndarray:
    x = torch.clamp(sample, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C
    return x


# origin: comfyanonymous/ComfyUI@387f98aa:comfy_extras/nodes_dataset.py:959-968
# (ResizeImagesByShorterEdgeNode._process — 크기 계산 부분만)
def _origin_shorter_edge_size(w: int, h: int, shorter_edge: int) -> tuple[int, int]:
    if w < h:
        new_w = shorter_edge
        new_h = int(h * (shorter_edge / w))
    else:
        new_h = shorter_edge
        new_w = int(w * (shorter_edge / h))
    return new_w, new_h


# origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference.py:226 (check_inputs: height/width % 32)
_ORIGIN_SIZE_MULTIPLE = 32


def _snap(v: int) -> int:
    return max(256, (v // _ORIGIN_SIZE_MULTIPLE) * _ORIGIN_SIZE_MULTIPLE)


# ---------------------------------------------------------------------------
# 가짜 safetensors (실제 형식 — safetensors.safe_open 으로도 열린다)
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {"F32": 4, "BF16": 2}


def _write_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> Path:
    header: dict = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = _DTYPE_BYTES[dtype] * math.prod(shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * offset)
    return path


# 이 PC 의 models/ControlNet 세 파일(2026-09-25) 헤더에서 옮긴 모양 — 텐서는 conditioning 입력 conv 하나만.
_TILE_META = {
    "lllite.version": "2", "lllite.cond_dim": "64", "lllite.mlp_dim": "64", "lllite.cond_emb_dim": "32",
    "lllite.cond_resblocks": "3", "lllite.aspp_dilations": "1,2,4,8", "lllite.target_layers": "self_attn_qkv",
    "lllite.target_atomics": "self_attn_q_pre,self_attn_kv_pre", "lllite.use_aspp": "true",
}
_INPAINT_META = {
    "lllite.version": "2", "lllite.target_atomics": "self_attn_q_pre,self_attn_kv_pre,mlp_fc1_pre",
    "lllite.cond_resblocks": "4", "lllite.use_aspp": "false", "lllite.cond_in_channels": "4",
    "lllite.mlp_dim": "64", "lllite.target_layers": "self_attn_q_pre,self_attn_kv_pre,mlp_fc1_pre",
    "lllite.cond_emb_dim": "64", "lllite.inpaint_masked_input": "true", "lllite.cond_dim": "128",
}


def _tile_file(path: Path) -> Path:
    return _write_safetensors(path, {
        "lllite_conditioning1.conv1.weight": ("F32", (32, 3, 4, 4)),
        "lllite_dit_blocks_0_self_attn_q_proj.down.weight": ("F32", (64, 8)),
    }, _TILE_META)


def _inpaint_file(path: Path) -> Path:
    return _write_safetensors(path, {
        "lllite_conditioning1.conv1.weight": ("BF16", (64, 4, 4, 4)),
        "lllite_dit_blocks_0_self_attn_q_proj.down.weight": ("BF16", (64, 8)),
    }, _INPAINT_META)


def _sdxl_controlnet_file(path: Path) -> Path:
    return _write_safetensors(path, {"control_model.input_blocks.0.0.weight": ("F32", (8, 4, 3, 3))})


def _sdxl_lllite_file(path: Path) -> Path:
    # kohya SDXL ControlNet-LLLite — conditioning1 이 모듈마다 붙고 공유 lllite_conditioning1. 은 없다.
    return _write_safetensors(path, {
        "lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.conditioning1.0.weight": ("F32", (8, 3, 4, 4)),
    })


class _TempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# 헤더 파서
# ---------------------------------------------------------------------------


class HeaderParserTests(_TempDir):
    def test_channels_match_the_sd_scripts_reader(self):
        for maker, expected in ((_tile_file, 3), (_inpaint_file, 4)):
            with self.subTest(maker=maker.__name__):
                path = maker(self.dir / f"{maker.__name__}.safetensors")
                origin_meta = _origin_read_lllite_metadata(str(path))
                self.assertEqual(read_safetensors_header(path)["__metadata__"], origin_meta)
                self.assertEqual(lllite_cond_in_channels(path), _origin_cond_in_channels(origin_meta))
                self.assertEqual(lllite_cond_in_channels(path), expected)

    def test_tile_file_without_the_key_defaults_to_three_like_sd_scripts(self):
        path = _tile_file(self.dir / "t.safetensors")
        self.assertNotIn("lllite.cond_in_channels", _origin_read_lllite_metadata(str(path)))
        self.assertEqual(lllite_cond_in_channels(path), 3)

    def test_no_metadata_block_is_still_three(self):
        path = _write_safetensors(self.dir / "bare.safetensors",
                                  {"lllite_conditioning1.conv1.weight": ("F32", (4, 3, 4, 4))})
        self.assertEqual(_origin_read_lllite_metadata(str(path)), {})
        self.assertEqual(lllite_cond_in_channels(path), 3)

    def test_other_models_are_not_anima_lllite(self):
        self.assertIsNone(lllite_cond_in_channels(_sdxl_controlnet_file(self.dir / "cn.safetensors")))
        self.assertIsNone(lllite_cond_in_channels(_sdxl_lllite_file(self.dir / "xl_lllite.safetensors")))

    def test_unreadable_files_are_skipped(self):
        empty = self.dir / "empty.safetensors"
        empty.write_bytes(b"")
        truncated = self.dir / "truncated.safetensors"
        truncated.write_bytes(struct.pack("<Q", 1000) + b'{"a":')
        not_json = self.dir / "not_json.safetensors"
        not_json.write_bytes(struct.pack("<Q", 4) + b"\xff\xfe\x00\x01")
        huge = self.dir / "huge.safetensors"
        huge.write_bytes(struct.pack("<Q", 2**40) + b"{}")
        json_list = self.dir / "list.safetensors"
        json_list.write_bytes(struct.pack("<Q", 2) + b"[]")
        for path in (empty, truncated, not_json, huge, json_list, self.dir / "missing.safetensors"):
            with self.subTest(path=path.name):
                self.assertIsNone(read_safetensors_header(path))
                self.assertIsNone(lllite_cond_in_channels(path))

    def test_bad_channel_metadata_is_skipped(self):
        path = _write_safetensors(self.dir / "bad.safetensors",
                                  {"lllite_conditioning1.conv1.weight": ("F32", (4, 3, 4, 4))},
                                  {"lllite.cond_in_channels": "rgb"})
        with self.assertRaises(ValueError):  # sd-scripts 는 로드 중 여기서 죽는다
            _origin_cond_in_channels(_origin_read_lllite_metadata(str(path)))
        self.assertIsNone(lllite_cond_in_channels(path))

    def test_real_forge_files_when_present(self):
        cn = Path(r"C:\sd-webui-forge-classic\models\ControlNet")
        expected = {
            "animaTileRepair_v10.safetensors": 3,
            "animaTileRepair_v20.safetensors": 3,
            "anima-lllite-inpainting-v2.safetensors": 4,
        }
        present = {name: n for name, n in expected.items() if (cn / name).is_file()}
        if not present:
            self.skipTest("models/ControlNet 에 Anima LLLite 파일 없음")
        for name, n in present.items():
            with self.subTest(name=name):
                origin = _origin_cond_in_channels(_origin_read_lllite_metadata(str(cn / name)))
                self.assertEqual(origin, n)
                self.assertEqual(lllite_cond_in_channels(cn / name), origin)


# ---------------------------------------------------------------------------
# 목록과 기본값
# ---------------------------------------------------------------------------


class ListChoicesTests(_TempDir):
    def test_only_three_channel_anima_lllite_files_are_listed(self):
        cn = self.dir / "ControlNet"
        cn.mkdir()
        _tile_file(cn / "animaTileRepair_v10.safetensors")
        _tile_file(cn / "animaTileRepair_v20.safetensors")
        _tile_file(cn / "my_tile_control.safetensors")          # 이름에 anima/lllite 가 없어도 헤더로 안다
        _inpaint_file(cn / "anima-lllite-inpainting-v2.safetensors")
        _sdxl_controlnet_file(cn / "anima_named_sdxl_cn.safetensors")
        _sdxl_lllite_file(cn / "kohya_controllllite_xl_canny.safetensors")
        with mock.patch.object(anima_core, "_models_path", return_value=self.dir):
            choices = list_lllite_choices()
        self.assertEqual(choices[0], "None")
        self.assertEqual(sorted(choices[1:]), sorted([
            "animaTileRepair_v10.safetensors",
            "animaTileRepair_v20.safetensors",
            "my_tile_control.safetensors",
        ]))
        self.assertEqual(default_lllite_choice(choices), "animaTileRepair_v20.safetensors")

    def test_no_models_dir_or_no_files(self):
        with mock.patch.object(anima_core, "_models_path", return_value=None):
            self.assertEqual(list_lllite_choices(), ["None"])
        with mock.patch.object(anima_core, "_models_path", return_value=self.dir):
            self.assertEqual(list_lllite_choices(), ["None"])
        cn = self.dir / "ControlNet"
        cn.mkdir()
        _inpaint_file(cn / "anima-lllite-inpainting-v2.safetensors")
        with mock.patch.object(anima_core, "_models_path", return_value=self.dir):
            self.assertEqual(list_lllite_choices(), ["None"], "4채널만 있으면 고를 것이 없다")

    def test_typo_extension_is_not_listed_because_sd_scripts_cannot_load_it(self):
        cn = self.dir / "ControlNet"
        cn.mkdir()
        good = _tile_file(cn / "animaTileRepair_v20.safetensors")
        typo = _tile_file(cn / "animaTileRepair_v30.saftensors")
        # 헤더만 보면 둘 다 3채널 Anima LLLite 다.
        self.assertEqual(lllite_cond_in_channels(typo), 3)
        # 원본 로더: .safetensors 는 읽고, 오타 확장자는 torch.load 로 가서 실패한다.
        self.assertIn("lllite_conditioning1.conv1.weight", _origin_load_lllite_state_dict(str(good)))
        with self.assertRaises(Exception):
            _origin_load_lllite_state_dict(str(typo))
        with mock.patch.object(anima_core, "_models_path", return_value=self.dir):
            choices = list_lllite_choices()
        self.assertEqual(choices, ["None", "animaTileRepair_v20.safetensors"])
        self.assertEqual(default_lllite_choice(choices), "animaTileRepair_v20.safetensors")

    def test_default_is_the_newest_tile_repair(self):
        v10, v20 = "animaTileRepair_v10.safetensors", "animaTileRepair_v20.safetensors"
        self.assertEqual(default_lllite_choice(["None", v10, v20]), v20)
        self.assertEqual(default_lllite_choice(["None", v20, v10]), v20)
        self.assertEqual(default_lllite_choice(["None", "canny_lllite.safetensors", v10]), v10)
        self.assertEqual(
            default_lllite_choice(["None", v20, "Anima-Tile-Repair_v30.safetensors"]),
            "Anima-Tile-Repair_v30.safetensors",
        )
        self.assertEqual(
            default_lllite_choice(["None", "anima_tilerepair_v2.safetensors", "anima_tilerepair_v2.1.safetensors"]),
            "anima_tilerepair_v2.1.safetensors",
        )

    def test_default_without_a_tile_repair_file(self):
        self.assertEqual(default_lllite_choice(["None", "canny_lllite.safetensors"]), "canny_lllite.safetensors")
        self.assertEqual(default_lllite_choice(["None"]), "None")
        self.assertEqual(default_lllite_choice([]), "None")


# ---------------------------------------------------------------------------
# 디코드
# ---------------------------------------------------------------------------


class DecodeTests(unittest.TestCase):
    def _sample(self) -> torch.Tensor:
        g = torch.Generator().manual_seed(0)
        sample = torch.randn(3, 17, 23, generator=g) * 1.5    # [-1, 1] 밖 값도 섞는다
        flat = sample.view(-1)
        # 경계·반올림≠버림 값: 버림(to uint8)이 원본이다
        flat[:8] = torch.tensor([-1.0, 1.0, 0.0, 0.999, -0.999, 0.0039, 0.5, -0.5])
        return sample

    def test_matches_sd_scripts_save_images(self):
        sample = self._sample()
        origin = _origin_save_images_array(sample)
        np.testing.assert_array_equal(decode_pixels_to_uint8(sample), origin)
        np.testing.assert_array_equal(np.asarray(_tensor_to_pil(sample)), origin)
        self.assertEqual(_tensor_to_pil(sample).mode, "RGB")

    def test_bright_image_is_not_rescaled(self):
        # 예전 코드는 min >= -0.01 이면 [0,1] 로 여겨 0.5 → 128 로 어둡게 만들었다.
        sample = torch.full((3, 4, 4), 0.5)
        origin = _origin_save_images_array(sample)
        self.assertEqual(int(origin[0, 0, 0]), 191)
        np.testing.assert_array_equal(np.asarray(_tensor_to_pil(sample)), origin)

    def test_batched_layouts_collapse_to_the_first_image(self):
        sample = self._sample()
        origin = _origin_save_images_array(sample)
        np.testing.assert_array_equal(np.asarray(_tensor_to_pil(sample.unsqueeze(0))), origin)
        np.testing.assert_array_equal(np.asarray(_tensor_to_pil(sample.unsqueeze(0).unsqueeze(2))), origin)
        with self.assertRaises(ValueError):
            _tensor_to_pil(sample[0, 0])


# ---------------------------------------------------------------------------
# 출력 크기
# ---------------------------------------------------------------------------


class SizeTests(unittest.TestCase):
    CASES = (
        (1024, 1024, 1024),
        (832, 1216, 1024),
        (1216, 832, 1024),
        (1920, 1080, 1024),
        (1080, 1920, 1024),
        (64, 64, 1024),
        (1000, 1500, 1000),
        (1000, 2000, 200),
        (3000, 100, 1024),
        (777, 1333, 1536),
        (1333, 777, 768),
        (512, 513, 1024),
    )

    def test_matches_shorter_edge_resize_then_multiple_of_32(self):
        for w, h, edge in self.CASES:
            with self.subTest(src=(w, h), edge=edge):
                ow, oh = _origin_shorter_edge_size(w, h, edge)
                expected = (_snap(ow), _snap(oh))
                got = tile_repair_size(w, h, edge)
                self.assertEqual(got, expected)
                self.assertTrue(all(v % 32 == 0 and v >= 256 for v in got))

    def test_known_values(self):
        self.assertEqual(tile_repair_size(1024, 1024, 1024), (1024, 1024))
        self.assertEqual(tile_repair_size(832, 1216, 1024), (1024, 1472))   # int(1496.6) → 1472
        self.assertEqual(tile_repair_size(1216, 832, 1024), (1472, 1024))
        self.assertEqual(tile_repair_size(1920, 1080, 1024), (1792, 1024))  # int(1820.4) → 1792
        self.assertEqual(tile_repair_size(1000, 1500, 1000), (992, 1472))   # 짧은 변도 32 배수로 내림
        self.assertEqual(tile_repair_size(1000, 2000, 200), (256, 384))     # 최소 256

    def test_rejects_empty_source(self):
        with self.assertRaises(ValueError):
            tile_repair_size(0, 512, 1024)


# ---------------------------------------------------------------------------
# 패널 기본값
# ---------------------------------------------------------------------------


class PanelDefaultsTests(unittest.TestCase):
    CHOICES = ["None", "animaTileRepair_v10.safetensors", "animaTileRepair_v20.safetensors"]

    def _panel(self):
        with mock.patch.object(ui_anima, "list_lllite_choices", return_value=list(self.CHOICES)):
            with gr.Blocks():
                return build_anima_panel()

    def test_lllite_default_is_v20(self):
        panel = self._panel()
        self.assertEqual(panel.lllite_model.value, "animaTileRepair_v20.safetensors")

    def test_sampler_and_prompt_defaults_match_sd_scripts(self):
        panel = self._panel()
        # origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:127-134
        self.assertEqual(panel.negative.value, "")          # --negative_prompt default ""
        self.assertEqual(panel.steps.value, 50)             # --infer_steps 50
        self.assertEqual(panel.cfg.value, 3.5)              # --guidance_scale 3.5
        self.assertEqual(panel.flow_shift.value, 5.0)       # --flow_shift 5.0

    def test_multiplier_range_matches_the_kohya_node(self):
        panel = self._panel()
        s = panel.lllite_multiplier
        # origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:130
        #   "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
        # origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:167-170 (default 1.0)
        self.assertEqual((s.minimum, s.maximum, s.step, s.value), (-10.0, 10.0, 0.01, 1.0))

    def test_short_side_replaces_width_and_height(self):
        panel = self._panel()
        self.assertIn("short_side", ANIMA_ARG_KEYS)
        self.assertNotIn("width", ANIMA_ARG_KEYS)
        self.assertNotIn("height", ANIMA_ARG_KEYS)
        self.assertFalse(hasattr(panel, "width") or hasattr(panel, "height"))
        s = panel.short_side
        self.assertEqual((s.minimum, s.maximum, s.step, s.value), (256, 4096, 32, 1024))
        widgets = panel.all_widgets()
        self.assertEqual(len(widgets), len(ANIMA_ARG_KEYS))
        repair = _map_widget_values(tuple(w.value for w in widgets))
        self.assertEqual(repair.short_side, 1024)
        self.assertEqual(repair.negative, "")
        self.assertEqual(repair.lllite_model, "animaTileRepair_v20.safetensors")


# ---------------------------------------------------------------------------
# run_tile_repair 배선 — 크기와 control 이미지
# ---------------------------------------------------------------------------


class _StopBeforeVendor(Exception):
    pass


class RunTileRepairSizeTests(unittest.TestCase):
    def _run(self, source: Image.Image, repair: AnimaTileRepairArgs) -> dict:
        seen: dict = {}

        def fake_args(rep, control_image_path):
            seen["size"] = (rep.width, rep.height)
            with Image.open(control_image_path) as img:
                seen["control_size"] = img.size
            raise _StopBeforeVendor

        mods = {
            "anima_minimal_inference_control_net_lllite": types.ModuleType("lllite"),
            "anima_minimal_inference": types.ModuleType("anima_minimal_inference"),
        }
        with (
            mock.patch.dict(sys.modules, mods),
            mock.patch.object(anima_core, "_ensure_vendor_importable", return_value=True),
            mock.patch.object(anima_core, "ANIMA_LLLITE_SENTINEL", types.SimpleNamespace(exists=lambda: True)),
            mock.patch.object(anima_core, "_vendor_sys_modules", contextlib.nullcontext),
            mock.patch.object(anima_core, "_build_anima_args", side_effect=fake_args),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(_StopBeforeVendor):
                anima_core.run_tile_repair(source, repair)
        return seen

    def test_output_keeps_the_source_aspect_ratio(self):
        repair = AnimaTileRepairArgs(lllite_model="animaTileRepair_v20.safetensors", short_side=1024)
        seen = self._run(Image.new("RGB", (832, 1216), "white"), repair)
        self.assertEqual(seen["size"], (1024, 1472))
        self.assertEqual((repair.width, repair.height), (1024, 1472), "infotext 의 Size 도 이 값")
        # control 이미지는 원래 크기로 넘기고, sd-scripts _load_control_image(:65-74)가 BICUBIC 으로 맞춘다
        self.assertEqual(seen["control_size"], (832, 1216))

    def test_landscape_source(self):
        repair = AnimaTileRepairArgs(lllite_model="animaTileRepair_v20.safetensors", short_side=768)
        seen = self._run(Image.new("RGB", (1920, 1080), "white"), repair)
        self.assertEqual(seen["size"], tile_repair_size(1920, 1080, 768))
        self.assertEqual(seen["size"], (1344, 768))


if __name__ == "__main__":
    unittest.main()
