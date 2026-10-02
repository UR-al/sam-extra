"""SAM3 ControlNet unit — Anima LLLite 판별(헤더·이름)과 전처리기 가드 (plan §7.2 #7, TR-SAM3).

torch·Forge 없이 돈다. safetensors 헤더는 임시 파일에 JSON 만 써서 만든다(가중치 없음).
실제 모델(Forge models/ControlNet)이 있으면 그 헤더도 읽는다 — 헤더 JSON 만 읽고 GPU 는 쓰지 않는다.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import sam3_cn_lllite as g  # noqa: E402


# origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:168
#   cond_in_channels = int(meta.get("lllite.cond_in_channels", 3))
# origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:325-329 (같은 식)
def _origin_cond_in_channels(meta: dict) -> int:
    return int(meta.get("lllite.cond_in_channels", 3))


# origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:171-183 — 4ch 는 MASK 필수, 3ch 는 MASK 를 버린다
#   if cond_in_channels != 4 and mask is not None: logger.warning(...ignored...); mask = None
def _origin_uses_mask(cond_in_channels: int) -> bool:
    return cond_in_channels == 4


# origin: kohya-ss/sd-scripts@690ea7f9:docs/anima_train_control_net_lllite.md:355
#   | `lllite.cond_in_channels` | conditioning1 input channel count (`"3"` standard, `"4"` inpaint) |
# origin: kohya-ss/sd-scripts@690ea7f9:docs/anima_train_control_net_lllite.md:77
#   conditioning images (e.g. lineart, canny, depth)
# origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:23,32
#   --control_image canny.png / --cn images/canny_a.png
# → 3채널은 Tile & Repair 전용이 아니다. lineart·canny·depth LLLite 도 3채널이고 원본은 사용자가 준 맵을 쓴다.
ORIGIN_STANDARD_CHANNELS = 3

# Tile & Repair (civitai 2708551) 가중치의 __metadata__ modelspec.title (설치본 헤더에서 읽은 값)
#   animaTileRepair_v10.safetensors → 'anima_tiled_lllite_v1', animaTileRepair_v20 → 'anima_tile_multitask_v1'
TILE_REPAIR_TITLES = {"animaTileRepair_v10": "anima_tiled_lllite_v1",
                      "animaTileRepair_v20": "anima_tile_multitask_v1"}


def _tile_header() -> dict:
    """animaTileRepair_v20.safetensors 헤더의 모양 (civitai 2708551 v2.0, 메타데이터에 cond_in_channels 없음)."""
    return {
        "__metadata__": {"lllite.version": "2", "lllite.cond_dim": "64", "lllite.use_aspp": "true",
                         "modelspec.title": "anima_tile_multitask_v1"},
        "lllite_conditioning1.conv1.weight": {"dtype": "F32", "shape": [32, 3, 4, 4], "data_offsets": [0, 0]},
        "lllite_dit_blocks_0_self_attn_q_proj.down.weight": {"dtype": "F32", "shape": [64, 2048],
                                                             "data_offsets": [0, 0]},
    }


def _inpaint_header() -> dict:
    """anima-lllite-inpainting-v2.safetensors 헤더의 모양 (메타데이터 cond_in_channels=4)."""
    return {
        "__metadata__": {"lllite.cond_in_channels": "4", "lllite.inpaint_masked_input": "true"},
        "lllite_conditioning1.conv1.weight": {"dtype": "BF16", "shape": [64, 4, 4, 4], "data_offsets": [0, 0]},
        "lllite_dit_blocks_0_self_attn_q_proj.down.weight": {"dtype": "BF16", "shape": [64, 2048],
                                                             "data_offsets": [0, 0]},
    }


def _lineart_header(title="anima_lineart_lllite_v1") -> dict:
    """lineart·canny·depth 같은 표준 3채널 Anima LLLite (sd-scripts 로 학습 — 메타데이터 cond_in_channels=3)."""
    meta = {"lllite.version": "2", "lllite.cond_in_channels": "3"}
    if title is not None:
        meta["modelspec.title"] = title
    return {
        "__metadata__": meta,
        "lllite_conditioning1.conv1.weight": {"dtype": "BF16", "shape": [64, 3, 4, 4], "data_offsets": [0, 0]},
        "lllite_dit_blocks_0_self_attn_q_proj.down.weight": {"dtype": "BF16", "shape": [64, 2048],
                                                             "data_offsets": [0, 0]},
    }


def _sdxl_lllite_header() -> dict:
    """SDXL ControlNet-LLLite (kohya controllllite) — Forge 는 Anima 패처가 아니라 SDXL 패처로 보낸다."""
    return {"lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.down.weight":
            {"dtype": "F16", "shape": [32, 640], "data_offsets": [0, 0]}}


def _write(directory: Path, name: str, header: dict) -> Path:
    data = json.dumps(header).encode("utf-8")
    path = directory / name
    path.write_bytes(struct.pack("<Q", len(data)) + data)
    return path


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_tile_repair_header_is_three_channels(self):
        path = _write(self.dir, "renamed_model.safetensors", _tile_header())
        self.assertEqual(g.lllite_channels_from_header(g.read_safetensors_header(path)), 3)
        # 헤더가 정답 — 이름이 LLLite 처럼 보이지 않아도 3채널이다
        self.assertEqual(g.lllite_channels("renamed_model", path), 3)

    def test_inpaint_header_is_four_channels(self):
        path = _write(self.dir, "x.safetensors", _inpaint_header())
        self.assertEqual(g.lllite_channels("x", path), 4)

    def test_non_anima_lllite_header_wins_over_a_misleading_name(self):
        path = _write(self.dir, "anima-lllite-inpainting-sdxl.safetensors", _sdxl_lllite_header())
        self.assertIsNone(g.lllite_channels("anima-lllite-inpainting-sdxl", path))

    def test_header_matches_the_origin_channel_rule(self):
        cases = [
            {"lllite.cond_in_channels": "4"},
            {"lllite.cond_in_channels": "3"},
            {},                                   # 메타데이터 없음 → 원본 기본 3
            {"lllite.version": "2"},
        ]
        for meta in cases:
            with self.subTest(meta=meta):
                header = {"__metadata__": meta,
                          "lllite_dit_blocks_0_self_attn_q_proj.down.weight": {"shape": [64, 2048]}}
                self.assertEqual(g.lllite_channels_from_header(header), _origin_cond_in_channels(meta))

    def test_tile_repair_is_told_apart_by_the_header_title(self):
        tile = _write(self.dir, "renamed_model.safetensors", _tile_header())
        self.assertEqual(g.anima_lllite("renamed_model", tile), g.AnimaLLLite(3, True))
        lineart = _write(self.dir, "my_lineart_cn.safetensors", _lineart_header())
        self.assertEqual(g.anima_lllite("my_lineart_cn", lineart), g.AnimaLLLite(ORIGIN_STANDARD_CHANNELS, False))
        # 제목이 있으면 제목이 정답 — 이름에 'tile' 이 있어도 lineart 모델이다
        misleading = _write(self.dir, "anima_tile_lineart.safetensors", _lineart_header())
        self.assertEqual(g.anima_lllite("anima_tile_lineart", misleading), g.AnimaLLLite(3, False))
        # 제목이 없으면 이름으로
        untitled = _write(self.dir, "animaTileRepair_v30.safetensors", _lineart_header(title=None))
        self.assertEqual(g.anima_lllite("animaTileRepair_v30", untitled), g.AnimaLLLite(3, True))
        untitled = _write(self.dir, "anima_depth.safetensors", _lineart_header(title=None))
        self.assertEqual(g.anima_lllite("anima_depth", untitled), g.AnimaLLLite(3, False))
        # 4채널은 Tile & Repair 가 아니다
        inpaint = _write(self.dir, "anima_tile_inpaint.safetensors", _inpaint_header())
        self.assertEqual(g.anima_lllite("anima_tile_inpaint", inpaint), g.AnimaLLLite(4, False))
        sdxl = _write(self.dir, "kohya_controllllite_xl_inpaint.safetensors", _sdxl_lllite_header())
        self.assertIsNone(g.anima_lllite("kohya_controllllite_xl_inpaint", sdxl))

    def test_conv_shape_is_used_when_metadata_is_missing(self):
        """Forge 내장 infer_anima_config 는 conv1 입력 채널로 정한다 — 메타데이터가 없을 때 같은 값을 쓴다."""
        header = {"lllite_conditioning1.conv1.weight": {"shape": [64, 4, 4, 4]},
                  "lllite_dit_blocks_0_self_attn_q_proj.down.weight": {"shape": [64, 2048]}}
        self.assertEqual(g.lllite_channels_from_header(header), 4)

    def test_unreadable_files_fall_back_to_the_name(self):
        bad = self.dir / "animaTileRepair_v20.safetensors"
        bad.write_bytes(b"\x00\x01")                              # 8바이트도 안 된다
        self.assertIsNone(g.read_safetensors_header(bad))
        self.assertEqual(g.lllite_channels("animaTileRepair_v20", bad), 3)
        huge = self.dir / "huge.safetensors"
        huge.write_bytes(struct.pack("<Q", 2 ** 40) + b"{}")      # 헤더 길이가 말이 안 된다
        self.assertIsNone(g.read_safetensors_header(huge))
        garbage = self.dir / "garbage.safetensors"
        garbage.write_bytes(struct.pack("<Q", 3) + b"\xff\xfe\xfd")
        self.assertIsNone(g.read_safetensors_header(garbage))
        self.assertIsNone(g.read_safetensors_header(self.dir / "missing.safetensors"))
        self.assertIsNone(g.read_safetensors_header(None))
        pt = self.dir / "anima-lllite-inpainting-v2.pth"
        pt.write_bytes(b"not a safetensors file")
        self.assertIsNone(g.read_safetensors_header(pt))
        self.assertEqual(g.lllite_channels("anima-lllite-inpainting-v2", pt), 4)

    def test_real_models_when_installed(self):
        forge_models = ROOT.parents[1] / "models"
        found = 0
        for folder in ("ControlNet", "sam3"):
            for name, want, tile in (("animaTileRepair_v10", 3, True), ("animaTileRepair_v20", 3, True),
                                     ("anima-lllite-inpainting-v2", 4, False)):
                path = forge_models / folder / f"{name}.safetensors"
                if not path.is_file():
                    continue
                found += 1
                with self.subTest(path=str(path)):
                    header = g.read_safetensors_header(path)
                    self.assertIsNotNone(header)
                    self.assertEqual(g.lllite_channels_from_header(header), want)
                    self.assertEqual(g.lllite_channels_from_header(header),
                                     _origin_cond_in_channels(header.get("__metadata__") or {}))
                    self.assertEqual(g.lllite_tile_repair_from_header(header, "renamed"), tile)
                    if name in TILE_REPAIR_TITLES:
                        self.assertEqual(header["__metadata__"]["modelspec.title"], TILE_REPAIR_TITLES[name])
        if not found:
            self.skipTest("Forge models/ControlNet 에 Anima LLLite 파일 없음")


class NameTests(unittest.TestCase):
    # (이름, 채널, Tile & Repair) — 앱 core/sam3_cn_names.py · frontend/src/utils/sam3ControlNet.ts 의 표와 같아야 한다.
    CASES = (
        ("animaTileRepair_v20", 3, True),
        ("animaTileRepair_v10", 3, True),
        ("anima_tile-repair_v3", 3, True),
        ("anima_tiled_lllite_v1", 3, True),
        ("anima_lllite_lineart_v1", 3, False),
        ("anima-lllite-canny", 3, False),
        ("Anima_LLLite_Depth", 3, False),
        ("anima-lllite-inpainting-v2", 4, False),
        ("Anima-LLLite-Inpainting-V2", 4, False),
        ("new-lllite-inpaint", None, False),
        ("kohya_controllllite_xl_inpaint", None, False),
        ("kohya_controllllite_xl_canny_anime", None, False),
        ("controlnet_tile_sdxl", None, False),
        ("TileRepair_sdxl", None, False),
        ("my_lineart_cn", None, False),
        ("None", None, False),
        ("", None, False),
        (None, None, False),
    )

    def test_table(self):
        for name, want, tile in self.CASES:
            with self.subTest(name=name):
                self.assertEqual(g.lllite_channels_from_name(name), want)
                self.assertIs(g.lllite_tile_repair_from_name(name), tile)
                self.assertEqual(g.anima_lllite(name), None if want is None else g.AnimaLLLite(want, tile))


class ModuleGuardTests(unittest.TestCase):
    TILE = g.AnimaLLLite(3, True)
    LINEART = g.AnimaLLLite(ORIGIN_STANDARD_CHANNELS, False)
    INPAINT = g.AnimaLLLite(4, False)

    def test_tile_repair_always_runs_without_preprocessor(self):
        for module in ("inpaint_only", "inpaint_global_harmonious", "tile_resample", "canny", "none"):
            with self.subTest(module=module):
                forced, reason = g.forced_cn_module(module, self.TILE, "animaTileRepair_v20")
                self.assertEqual(forced, "None")
                self.assertIn("animaTileRepair_v20", reason)
                self.assertIn("Tile & Repair", reason)
                self.assertIn(module, reason)
        self.assertEqual(g.forced_cn_module("None", self.TILE), ("None", None))
        self.assertFalse(_origin_uses_mask(3))

    def test_standard_three_channel_lllite_keeps_its_preprocessor(self):
        """lineart·canny·depth LLLite — 원본은 사용자가 준 맵을 쓴다. SAM3 유닛에서는 전처리기가 그 맵을 만든다."""
        for module in ("lineart_anime", "canny", "depth_anything_v2", "tile_resample"):
            with self.subTest(module=module):
                self.assertEqual(g.forced_cn_module(module, self.LINEART, "my_lineart_cn"), (module, None))
        self.assertEqual(g.forced_cn_module("None", self.LINEART), ("None", None))

    def test_standard_three_channel_lllite_drops_inpaint_preprocessors(self):
        """원본 3채널은 마스크를 버린다 — inpaint_* 가 마스크 영역을 제어 이미지에서 비우면 원본과 다르다."""
        self.assertFalse(_origin_uses_mask(ORIGIN_STANDARD_CHANNELS))
        for module in ("inpaint_only", "inpaint_global_harmonious", "inpaint_only+lama", "inpaint_noobai"):
            with self.subTest(module=module):
                forced, reason = g.forced_cn_module(module, self.LINEART, "my_lineart_cn")
                self.assertEqual(forced, "None")
                self.assertIn("3-channel", reason)
                self.assertIn("ignores the mask", reason)
                self.assertIn("my_lineart_cn", reason)

    def test_four_channel_lllite_only_drops_inpaint_preprocessors(self):
        self.assertTrue(_origin_uses_mask(4))
        forced, reason = g.forced_cn_module("inpaint_only", self.INPAINT, "anima-lllite-inpainting-v2")
        self.assertEqual(forced, "None")
        self.assertIn("strips the mask", reason)
        self.assertEqual(g.forced_cn_module("inpaint_noobai", self.INPAINT)[0], "None")
        self.assertEqual(g.forced_cn_module("None", self.INPAINT), ("None", None))
        self.assertEqual(g.forced_cn_module("depth_leres", self.INPAINT), ("depth_leres", None))

    def test_other_models_are_untouched(self):
        for module in ("inpaint_only", "tile_resample", "lineart_anime", "None"):
            self.assertEqual(g.forced_cn_module(module, None), (module, None))


# ---------------------------------------------------------------------------
# inpaint_core.inject_controlnet_unit — Forge·ControlNet 스텁으로 끝까지
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


class _Unit:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class InjectControlNetUnitTests(unittest.TestCase):
    def setUp(self):
        self.inpaint_core = _inpaint_core()
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.files = {
            "animaTileRepair_v20": str(_write(tmp, "animaTileRepair_v20.safetensors", _tile_header())),
            "my_renamed_tile": str(_write(tmp, "my_renamed_tile.safetensors", _tile_header())),
            "anima-lllite-inpainting-v2": str(_write(tmp, "anima-lllite-inpainting-v2.safetensors",
                                                     _inpaint_header())),
            "my_lineart_cn": str(_write(tmp, "my_lineart_cn.safetensors", _lineart_header())),
            "kohya_controllllite_xl_inpaint": str(_write(tmp, "kohya_controllllite_xl_inpaint.safetensors",
                                                         _sdxl_lllite_header())),
        }

    def tearDown(self):
        self._tmp.cleanup()

    def _inject(self, module, model):
        lib = types.ModuleType("lib_controlnet")
        lib.__path__ = []
        external = types.ModuleType("lib_controlnet.external_code")
        external.ControlNetUnit = _Unit
        state = types.ModuleType("lib_controlnet.global_state")
        state.controlnet_filename_dict = {"None": None, **self.files}
        lib.external_code, lib.global_state = external, state
        ui = types.ModuleType("sam3ext.ui")
        ui._scan_sam3_dir_for_cn_models = lambda: {}
        script = types.SimpleNamespace(filename="extensions-builtin/sd_forge_controlnet/scripts/controlnet.py",
                                       args_from=1, args_to=3)
        p2 = types.SimpleNamespace(scripts=types.SimpleNamespace(alwayson_scripts=[script]),
                                   script_args=("before", _Unit(enabled=False), _Unit(enabled=False), "after"))
        err = io.StringIO()
        modules = {"lib_controlnet": lib, "lib_controlnet.external_code": external,
                   "lib_controlnet.global_state": state, "sam3ext.ui": ui}
        with mock.patch.dict(sys.modules, modules), contextlib.redirect_stderr(err):
            self.inpaint_core.inject_controlnet_unit(p2, {
                "sam3_cn_enable": True, "sam3_cn_module": module, "sam3_cn_model": model})
        return p2.script_args[1], err.getvalue()

    def test_tile_repair_lllite_forces_none_and_logs(self):
        unit, log = self._inject("inpaint_only", "animaTileRepair_v20")
        self.assertEqual(unit.module, "None")
        self.assertEqual(unit.model, "animaTileRepair_v20")
        self.assertIn("Anima Tile & Repair ControlNet-LLLite", log)
        unit, log = self._inject("tile_resample", "animaTileRepair_v20")
        self.assertEqual(unit.module, "None")

    def test_standard_three_channel_lllite_runs_the_chosen_preprocessor(self):
        """3채널 lineart Anima LLLite (이름에 lllite 없음, 헤더로 판별) — lineart_anime 가 그대로 간다."""
        for module in ("lineart_anime", "canny"):
            with self.subTest(module=module):
                unit, log = self._inject(module, "my_lineart_cn")
                self.assertEqual((unit.module, log), (module, ""))
        unit, log = self._inject("inpaint_only", "my_lineart_cn")
        self.assertEqual(unit.module, "None")
        self.assertIn("ignores the mask", log)

    def test_non_anima_lllite_inpaint_name_keeps_the_module(self):
        """SDXL controllllite — 이름에 lllite·inpaint 가 있어도 헤더가 Anima 가 아니면 그대로."""
        unit, log = self._inject("inpaint_only", "kohya_controllllite_xl_inpaint")
        self.assertEqual((unit.module, log), ("inpaint_only", ""))

    def test_renamed_tile_repair_is_found_by_header(self):
        unit, log = self._inject("tile_resample", "my_renamed_tile")
        self.assertEqual(unit.module, "None")
        self.assertIn("my_renamed_tile", log)

    def test_inpaint_lllite_keeps_the_previous_rule(self):
        unit, log = self._inject("inpaint_only", "anima-lllite-inpainting-v2")
        self.assertEqual(unit.module, "None")
        self.assertIn("strips the mask", log)
        unit, log = self._inject("None", "anima-lllite-inpainting-v2")
        self.assertEqual((unit.module, log), ("None", ""))

    def test_regular_controlnet_is_untouched(self):
        unit, log = self._inject("inpaint_only", "None")
        self.assertEqual((unit.module, log), ("inpaint_only", ""))


class TipTextTests(unittest.TestCase):
    def test_tip_mentions_the_tile_repair_override(self):
        source = (ROOT / "sam3ext" / "ui.py").read_text(encoding="utf-8")
        self.assertIn("Tile & Repair models (e.g. `animaTileRepair_*`) always run with", source)
        self.assertIn("Other Anima LLLite", source)
        self.assertIn("keep your preprocessor", source)
        self.assertIn("anima-lllite-inpainting-*", source)


if __name__ == "__main__":
    unittest.main()
