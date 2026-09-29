"""Anima VAE DeGrid — 잔차 산술(모드·부호·강도·마지막 clamp), 타일(ComfyUI tiled_scale 과 같음), OOM 재시도, PIL 변환,
NAFNet 파일 찾기, spandrel 로더, 런타임(장치·Forge 메모리 관리). GPU 없이(CPU) 돈다.

대조 오라클
- 잔차 적용: ComfyUI-NAFNet-Residual(DraconicDragon, Apache-2.0 — LICENSE 에 저작권자 이름 없음, NOTICE 없음)
  commit e15460d3724c70d428e333518b58eb7ba8903d76 ``patch.py`` 48-62줄 ``_apply_residual_mode`` 를 아래에 그대로 옮겼다.
  노드(``nafnet_node.py`` 132-149줄)는 같은 식 뒤에 ``torch.clamp(result, 0, 1)``.
- 타일: Forge ``backend/patcher/vae.py`` 의 ``tiled_scale_multidim``(ComfyUI v0.3.64 ``comfy/utils.py`` 를 옮긴 것 — 노드 팩이
  부르는 ``comfy.utils.tiled_scale`` 과 같은 함수)을 Forge 가 확장 옆에 있을 때만 AST 로 꺼내 쓴다(CI 에서는 건너뜀).
- 실제 가중치(qwenVAEDegridNafnet_v11)는 SAM3_RUN_FORGE_INTEGRATION_TESTS=1 이고 파일이 있을 때만(CPU, 1~2 초).
"""
from __future__ import annotations

import ast
import itertools
import math
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

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
        # v1.0 은 modelspec.version 을 적지 않았고 v1.1 파인튜닝은 "1.1" — 이름순이면 v1.0 이 앞이다
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


class _ShiftModel(torch.nn.Module):
    """잔차 = 0.1·(x − 0.5) (국소·결정적). 호출 크기를 기록한다."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape))
        return (x - 0.5) * 0.1 + self.p


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
        expected = vd.tensor_to_pil(torch.clamp(x + (x - 0.5) * 0.1, 0, 1))
        self.assertEqual(out.image.tobytes(), expected.tobytes())
        self.assertEqual((out.device, out.precision, out.tile_used), ("cpu", "fp32", 0))
        self.assertGreater(out.stats["changed_pixels"], 0.5)
        self.assertIn("Full", out.summary())

    def test_modes_and_strength_through_the_runtime(self):
        image = self._image()
        x, _ = vd.pil_to_tensor(image)
        delta = (x - 0.5) * 0.1
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

    @unittest.skipUnless(FORGE_VAE_PY.is_file(), "Forge 의 backend/patcher/vae.py 가 확장 옆에 없음")
    def test_real_model_tiles_like_comfyui(self):
        x, _ = vd.pil_to_tensor(_synthetic_illustration(384, seed=2))
        tiled_scale = _forge_tiled_scale()
        with torch.inference_mode():
            ours = vd.tiled_residual(x, lambda t: self.model(t.float()), tile=256)
            theirs = tiled_scale(x, lambda t: self.model(t.float()), tile_x=256, tile_y=256, overlap=32,
                                 upscale_amount=1, out_channels=3, output_device="cpu")
        self.assertTrue(torch.equal(ours, theirs))


if __name__ == "__main__":
    unittest.main()
