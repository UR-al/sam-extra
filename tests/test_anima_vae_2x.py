"""Anima VAE 2x (scripts/anima_vae_2x.py) — 감지·pixel-shuffle 산술·is_wan 5D 계약·
폴백·장치 배치 테스트. GPU 없이(CUDA_VISIBLE_DEVICES=-1) 돈다.

배경(감사 H4/M22/M23): Anima 엔진은 ``VAE(is_wan=True)`` 라 순정 ``VAE.decode`` 가
5D ``[B,T,H,W,3]`` 를 돌려주고, ``decode_first_stage`` 가 ``movedim(-1, 2)``,
processing.py 가 5D 를 ``reshape(-1, 3, H, W)`` 로 편다. 래퍼가 4D 를 돌려주면
그 뒤 단계(래퍼 밖)에서 죽어 폴백도 타지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def _load_vae2x_module():
    """WebUI 를 띄우지 않고 스크립트만 로드한다 (modules 스텁, gradio 는 실제)."""
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    modules_stub.scripts = types.SimpleNamespace(
        Script=Script,
        AlwaysVisible=object(),
        scripts_data=[],
    )

    old_modules = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_anima_vae_2x", ROOT / "scripts" / "anima_vae_2x.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_modules is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = old_modules


vae2x = _load_vae2x_module()


# ---------------------------------------------------------------------------
# 가짜 부품
# ---------------------------------------------------------------------------


class FakeDecoder(torch.nn.Module):
    """WanVAE(conv_out_channels=12).decode 흉내: [B,16,T,h,w] → [B,12,T,8h,8w].
    프레임마다 다른 상수를 넣어 T 축이 보존되는지 확인할 수 있게 한다."""

    def __init__(self, fail: bool = False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.fail = fail
        self.calls: list[tuple] = []

    def decode(self, z):
        self.calls.append(tuple(z.shape))
        if self.fail:
            raise RuntimeError("boom")
        b, _, t, h, w = z.shape
        out = torch.empty(b, 12, t, h * 8, w * 8, dtype=z.dtype, device=z.device)
        for i in range(t):
            out[:, :, i] = -1.0 + 2.0 * (i + 1) / (t + 1)   # 프레임별 상수 (-1,1)
        return out


class FakeStockVAE:
    """Forge ``VAE`` 의 is_wan 인스턴스 흉내 (decode 호출 기록)."""

    def __init__(self, is_wan: bool = True):
        self.is_wan = is_wan
        self.device = torch.device("cpu")
        self.output_device = torch.device("cpu")
        self.vae_dtype = torch.float32
        self.first_stage_model = torch.nn.Linear(1, 1)
        self.decode_calls: list = []
        if is_wan:
            self.memory_used_decode = lambda shape, dtype: 2200 * shape[3] * shape[4] * 64 * 4
        else:
            self.memory_used_decode = lambda shape, dtype: 2178 * shape[2] * shape[3] * 64 * 4

    def decode(self, samples_in, *args, **kwargs):
        self.decode_calls.append((tuple(samples_in.shape), args, kwargs))
        if samples_in.ndim == 5:
            b, _, t, h, w = samples_in.shape
            return torch.full((b, t, h * 8, w * 8, 3), 0.25)
        b, _, h, w = samples_in.shape
        return torch.full((b, h * 8, w * 8, 3), 0.25)

    def clone(self):
        return self


def _write_fake_safetensors(path: Path, out_channels: int | None) -> str:
    """헤더만 있는 가짜 safetensors (텐서 데이터 없음, 감지는 헤더만 읽는다)."""
    header = {"__metadata__": {"format": "pt"}}
    if out_channels is not None:
        header["decoder.head.2.weight"] = {
            "dtype": "BF16", "shape": [out_channels, 96, 3, 3, 3], "data_offsets": [0, 0],
        }
    header["decoder.conv1.weight"] = {"dtype": "BF16", "shape": [96, 16, 3, 3, 3], "data_offsets": [0, 0]}
    body = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
    return str(path)


def _forge_is_wan_pipeline(decoded):
    """base.decode_first_stage(is_wan) + processing.py 가 하는 일을 그대로 흉내내
    최종 PIL 이미지까지 만든다. 래퍼 밖 단계라 여기서 죽으면 폴백이 없다."""
    from PIL import Image

    sample = decoded.movedim(-1, 2).mul_(2.0).sub_(1.0)          # → [B,T,3,H,W]
    x = torch.stack(list(sample)).float()
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    if len(x.shape) == 5:
        x = x.reshape(-1, *x.shape[-3:])                            # → [N,3,H,W]
    images = []
    for x_sample in x:
        arr = 255.0 * np.moveaxis(x_sample.cpu().numpy(), 0, 2)     # → (H,W,3)
        images.append(Image.fromarray(arr.astype(np.uint8)))
    return images


# ---------------------------------------------------------------------------
# 감지 (safetensors 헤더)
# ---------------------------------------------------------------------------


class DetectOutputChannelsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_12ch_head_is_spacepxl(self):
        path = _write_fake_safetensors(self.dir / "spacepxl_2x.safetensors", 12)
        self.assertEqual(vae2x.detect_output_channels(path), 12)
        self.assertTrue(vae2x.is_spacepxl_2x(path))

    def test_stock_3ch_head_is_not_spacepxl(self):
        path = _write_fake_safetensors(self.dir / "wan_vae.safetensors", 3)
        self.assertEqual(vae2x.detect_output_channels(path), 3)
        self.assertFalse(vae2x.is_spacepxl_2x(path))

    def test_prefixed_key_still_detected(self):
        path = self.dir / "prefixed.safetensors"
        header = {"vae.decoder.head.2.weight": {"dtype": "BF16", "shape": [12, 96, 3, 3, 3], "data_offsets": [0, 0]}}
        body = json.dumps(header).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(body)) + body)
        self.assertEqual(vae2x.detect_output_channels(str(path)), 12)

    def test_no_head_key_or_missing_file(self):
        path = _write_fake_safetensors(self.dir / "other.safetensors", None)
        self.assertIsNone(vae2x.detect_output_channels(path))
        self.assertIsNone(vae2x.detect_output_channels(str(self.dir / "nope.safetensors")))
        self.assertIsNone(vae2x.detect_output_channels(""))
        self.assertFalse(vae2x.is_spacepxl_2x(""))

    def test_garbage_header_returns_none(self):
        path = self.dir / "garbage.safetensors"
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", 5) + b"notjs")
        with mock.patch.object(vae2x, "_log"):
            self.assertIsNone(vae2x.detect_output_channels(str(path)))


# ---------------------------------------------------------------------------
# pixel-shuffle 변환 산술
# ---------------------------------------------------------------------------


class TransformTests(unittest.TestCase):
    def test_5d_keeps_t_axis_2x(self):
        dec = torch.rand(2, 12, 3, 8, 8) * 2 - 1
        out = vae2x._transform(dec, refine_1x=False, blur_sigma=0.0)
        self.assertEqual(tuple(out.shape), (2, 3, 16, 16, 3))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_5d_keeps_t_axis_1x(self):
        dec = torch.rand(2, 12, 3, 8, 8) * 2 - 1
        out = vae2x._transform(dec, refine_1x=True, blur_sigma=0.5)
        self.assertEqual(tuple(out.shape), (2, 3, 8, 8, 3))
        self.assertTrue(out.is_contiguous())

    def test_4d_stays_4d(self):
        dec = torch.rand(2, 12, 8, 8) * 2 - 1
        self.assertEqual(tuple(vae2x._transform(dec, False, 0.0).shape), (2, 16, 16, 3))
        self.assertEqual(tuple(vae2x._transform(dec, True, 0.0).shape), (2, 8, 8, 3))

    def test_2x_is_exact_pixel_shuffle_inverse_per_frame(self):
        # 프레임·배치마다 다른 이미지 → pixel_unshuffle 로 12ch 를 만들고 되돌려 본다.
        b, t, h, w = 2, 3, 6, 10
        img = torch.rand(b, t, 3, 2 * h, 2 * w) * 2 - 1                 # [-1,1]
        packed = torch.stack([F.pixel_unshuffle(img[:, i], 2) for i in range(t)], dim=2)
        self.assertEqual(tuple(packed.shape), (b, 12, t, h, w))
        out = vae2x._transform(packed, refine_1x=False, blur_sigma=0.0)  # [B,T,2h,2w,3]
        expect = img.add(1.0).div(2.0).movedim(2, -1)
        torch.testing.assert_close(out, expect)
        # 프레임이 뒤섞이지 않았는지(순서) 한 번 더 — 각 프레임이 서로 다르다.
        self.assertFalse(torch.allclose(out[:, 0], out[:, 1]))

    def test_1x_refine_no_blur_is_2x2_mean(self):
        dec = torch.rand(1, 12, 1, 4, 4) * 2 - 1
        full = vae2x._transform(dec, refine_1x=False, blur_sigma=0.0)     # [1,1,8,8,3]
        half = vae2x._transform(dec, refine_1x=True, blur_sigma=0.0)      # [1,1,4,4,3]
        expect = F.avg_pool2d(full[0].movedim(-1, 1), 2).movedim(1, -1).unsqueeze(0)
        torch.testing.assert_close(half, expect)

    def test_gaussian_blur_preserves_shape_and_constant(self):
        x = torch.full((1, 3, 8, 8), 0.3)
        y = vae2x._gaussian_blur(x, 0.5)
        self.assertEqual(tuple(y.shape), (1, 3, 8, 8))
        # 커널 합이 1 이라 내부 픽셀은 상수가 유지된다(경계는 zero-pad 로 어두워짐).
        torch.testing.assert_close(y[:, :, 2:-2, 2:-2], x[:, :, 2:-2, 2:-2])
        self.assertIs(vae2x._gaussian_blur(x, 0.0), x)


# ---------------------------------------------------------------------------
# 래퍼: is_wan 5D 계약 + 폴백
# ---------------------------------------------------------------------------


class WrapperDecodeTests(unittest.TestCase):
    def _wrapper(self, refine_1x, decoder=None, stock=None, renorm=False):
        stock = stock or FakeStockVAE()
        decoder = decoder or FakeDecoder()
        return vae2x._VAE2xWrapper(stock, decoder, refine_1x, 0.5, renorm), stock, decoder

    def test_5d_latent_returns_5d_like_stock_is_wan(self):
        for refine_1x, hw in ((True, 64), (False, 128)):
            with self.subTest(mode="1x" if refine_1x else "2x"):
                wrapper, stock, decoder = self._wrapper(refine_1x)
                latent = torch.randn(1, 16, 1, 8, 8)              # Anima(is_wan) 5D latent
                out = wrapper.decode(latent)
                self.assertEqual(tuple(out.shape), (1, 1, hw, hw, 3))
                self.assertEqual(out.dtype, torch.float32)
                self.assertEqual(decoder.calls, [(1, 16, 1, 8, 8)])
                self.assertEqual(stock.decode_calls, [], "순정 decode 로 폴백하면 안 된다")

    def test_5d_output_survives_forge_is_wan_pipeline(self):
        # H4 재현: 예전 4D 출력은 movedim(-1,2) → [B,H,3,W] 가 되어 Image.fromarray 에서 죽었다.
        for refine_1x in (True, False):
            with self.subTest(mode="1x" if refine_1x else "2x"):
                wrapper, _, _ = self._wrapper(refine_1x)
                out = wrapper.decode(torch.randn(2, 16, 1, 8, 8))
                images = _forge_is_wan_pipeline(out)
                self.assertEqual(len(images), 2)
                side = 64 if refine_1x else 128
                self.assertEqual(images[0].size, (side, side))
                self.assertEqual(images[0].mode, "RGB")

    def test_old_4d_output_would_break_pipeline(self):
        # 회귀 방지 근거: 4D [B,H,W,3] 를 is_wan 파이프라인에 넣으면 실패한다.
        bad = torch.rand(1, 64, 64, 3)
        with self.assertRaises(Exception):
            _forge_is_wan_pipeline(bad)

    def test_multi_frame_latent_keeps_frame_order(self):
        wrapper, _, _ = self._wrapper(False)
        out = wrapper.decode(torch.randn(1, 16, 3, 4, 4))          # T=3
        self.assertEqual(tuple(out.shape), (1, 3, 64, 64, 3))
        # FakeDecoder 는 프레임 i 에 -1+2(i+1)/4 → [0,1] 로 바꾸면 (i+1)/4
        for i in range(3):
            torch.testing.assert_close(out[0, i], torch.full((64, 64, 3), (i + 1) / 4))

    def test_4d_latent_returns_4d(self):
        wrapper, _, decoder = self._wrapper(False, stock=FakeStockVAE(is_wan=False))
        out = wrapper.decode(torch.randn(2, 16, 8, 8))
        self.assertEqual(tuple(out.shape), (2, 128, 128, 3))
        self.assertEqual(decoder.calls, [(2, 16, 1, 8, 8)], "디코더에는 T=1 로 넣는다")

    def test_decoder_error_falls_back_to_stock_with_args(self):
        wrapper, stock, _ = self._wrapper(True, decoder=FakeDecoder(fail=True))
        latent = torch.randn(1, 16, 1, 8, 8)
        with mock.patch.object(vae2x, "_log") as log:
            out = wrapper.decode(latent, "extra", key="v")
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64, 3))
        self.assertEqual(stock.decode_calls, [((1, 16, 1, 8, 8), ("extra",), {"key": "v"})])
        self.assertIn("stock decode", log.call_args[0][0])

    def test_decode_outcome_is_recorded_per_pass(self):
        # 2026-10-02 review (generation/vae2x_decode_status): pending at attach, then the real result
        params = {}
        report = vae2x._VAE2xDecodeReport(params, "spacepxl_2x.safetensors, 2x, blur=0.5, renorm=False", "main")
        self.assertEqual(params["Anima VAE 2x"], "spacepxl_2x.safetensors, 2x, blur=0.5, renorm=False, decode=pending")
        self.assertEqual(params["Anima VAE 2x main outcome"], "pending decode")
        stock = FakeStockVAE()
        ok = vae2x._VAE2xWrapper(stock, FakeDecoder(), False, 0.5, False, report=report)
        ok.decode(torch.randn(1, 16, 1, 8, 8))
        self.assertTrue(params["Anima VAE 2x"].endswith("decode=applied"))
        self.assertEqual(params["Anima VAE 2x main outcome"], "applied_calls=1; stock_fallback_calls=0; last=applied")
        failing = vae2x._VAE2xWrapper(stock, FakeDecoder(fail=True), False, 0.5, False, report=report)
        with mock.patch.object(vae2x, "_log"):
            failing.clone().decode(torch.randn(1, 16, 1, 8, 8))   # clones share the pass report
        self.assertTrue(params["Anima VAE 2x"].endswith("decode=stock fallback"))
        self.assertEqual(params["Anima VAE 2x main outcome"],
                         "applied_calls=1; stock_fallback_calls=1; last=stock fallback")
        self.assertIn("Anima VAE 2x main outcome last error", params)
        self.assertLessEqual(len(params["Anima VAE 2x main outcome last error"]), 240)

    def test_hires_pass_gets_its_own_outcome_key(self):
        params = {}
        vae2x._VAE2xDecodeReport(params, "s", "main")
        hires = vae2x._VAE2xDecodeReport(params, "s", "hires")
        hires.record(True)
        self.assertEqual(params["Anima VAE 2x main outcome"], "pending decode")
        self.assertEqual(params["Anima VAE 2x hires outcome"], "applied_calls=1; stock_fallback_calls=0; last=applied")

    def test_report_errors_never_turn_a_decode_into_a_fallback(self):
        class Broken:
            def record(self, applied, error=None):
                raise RuntimeError("metadata failure")

        stock = FakeStockVAE()
        wrapper = vae2x._VAE2xWrapper(stock, FakeDecoder(), False, 0.5, False, report=Broken())
        out = wrapper.decode(torch.randn(1, 16, 1, 8, 8))
        self.assertEqual(tuple(out.shape), (1, 1, 128, 128, 3))
        self.assertEqual(stock.decode_calls, [])

    def test_wrong_rank_from_transform_falls_back(self):
        # 래퍼 밖에서 터질 shape(4D) 가 나오면 래퍼 안에서 잡아 순정으로 넘긴다.
        wrapper, stock, _ = self._wrapper(False)
        with mock.patch.object(vae2x, "_transform", return_value=torch.rand(1, 64, 64, 3)), \
                mock.patch.object(vae2x, "_log"):
            out = wrapper.decode(torch.randn(1, 16, 1, 8, 8))
        self.assertEqual(len(stock.decode_calls), 1)
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64, 3))

    def test_wrong_channel_axis_falls_back(self):
        wrapper, stock, _ = self._wrapper(False)
        with mock.patch.object(vae2x, "_transform", return_value=torch.rand(1, 1, 3, 64, 64)), \
                mock.patch.object(vae2x, "_log"):
            wrapper.decode(torch.randn(1, 16, 1, 8, 8))
        self.assertEqual(len(stock.decode_calls), 1)

    def test_renorm_normalizes_latent_before_decode(self):
        seen = {}

        class Spy(FakeDecoder):
            def decode(self, z):
                seen["z"] = z.clone()
                return super().decode(z)

        wrapper, _, _ = self._wrapper(False, decoder=Spy(), renorm=True)
        wrapper.decode(torch.randn(1, 16, 1, 8, 8) * 5 + 3)
        self.assertAlmostEqual(float(seen["z"].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(seen["z"].std()), 1.0, places=3)

    def test_delegates_other_attributes_and_clone(self):
        wrapper, stock, decoder = self._wrapper(True)
        self.assertIs(wrapper.first_stage_model, stock.first_stage_model)
        self.assertTrue(wrapper.is_wan)
        c = wrapper.clone()
        self.assertIsInstance(c, vae2x._VAE2xWrapper)
        self.assertIs(c._decoder, decoder)
        self.assertTrue(c._refine_1x)

    def test_memory_estimate_uses_stock_formula_on_5d(self):
        # 4D latent 라도 is_wan 공식(shape[3], shape[4]) 이 깨지지 않게 5D 로 만든 뒤 추정한다.
        wrapper, stock, _ = self._wrapper(False)
        with mock.patch.object(vae2x, "_load_decoder", wraps=vae2x._load_decoder) as ld:
            wrapper.decode(torch.randn(1, 16, 8, 8))
        self.assertEqual(ld.call_args[0][1], float(stock.memory_used_decode((1, 16, 1, 8, 8), torch.float32)))


# ---------------------------------------------------------------------------
# 장치 배치 (M22): load_device 로 짓고 Forge 메모리 관리(ModelPatcher) 에 맡긴다
# ---------------------------------------------------------------------------


class FakePatcher:
    def __init__(self, model, load_device, offload_device):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device


class DevicePlacementTests(unittest.TestCase):
    def test_load_device_comes_from_vae_device_not_parameters(self):
        vae = FakeStockVAE()
        vae.device = torch.device("cuda:1")          # Forge VAE.device = load_device (GPU)
        # first_stage_model 파라미터는 CPU 에 오프로드돼 있다 — 이걸 따라가면 안 된다.
        self.assertEqual(next(vae.first_stage_model.parameters()).device.type, "cpu")
        self.assertEqual(vae2x._decoder_load_device(vae), torch.device("cuda:1"))

    def test_load_device_fallback_to_memory_management(self):
        mm = types.SimpleNamespace(get_torch_device=lambda: torch.device("cuda:0"))
        backend = types.ModuleType("backend")
        backend.memory_management = mm
        with mock.patch.dict(sys.modules, {"backend": backend, "backend.memory_management": mm}):
            self.assertEqual(vae2x._decoder_load_device(object()), torch.device("cuda:0"))

    def test_load_decoder_bare_module_uses_its_device(self):
        dec = FakeDecoder()
        model, dev = vae2x._load_decoder(dec, 123.0)
        self.assertIs(model, dec)
        self.assertEqual(dev, torch.device("cpu"))

    def test_load_decoder_patcher_goes_through_load_models_gpu(self):
        calls = []
        mm = types.SimpleNamespace(load_models_gpu=lambda models, **kw: calls.append((models, kw)))
        backend = types.ModuleType("backend")
        backend.memory_management = mm
        dec = FakeDecoder()
        patcher = FakePatcher(dec, torch.device("cuda:0"), torch.device("cpu"))
        with mock.patch.dict(sys.modules, {"backend": backend, "backend.memory_management": mm}):
            model, dev = vae2x._load_decoder(patcher, 4096.0)
        self.assertIs(model, dec)
        self.assertEqual(dev, torch.device("cuda:0"))
        self.assertEqual(calls, [([patcher], {"memory_required": 4096.0, "force_full_load": True})])

    def test_wrapper_decode_loads_patcher_before_decoding(self):
        order = []
        dec = FakeDecoder()
        patcher = FakePatcher(dec, torch.device("cpu"), torch.device("cpu"))
        mm = types.SimpleNamespace(load_models_gpu=lambda models, **kw: order.append("load"))
        backend = types.ModuleType("backend")
        backend.memory_management = mm
        orig_decode = dec.decode
        dec.decode = lambda z: (order.append("decode"), orig_decode(z))[1]
        wrapper = vae2x._VAE2xWrapper(FakeStockVAE(), patcher, False, 0.5, False)
        with mock.patch.dict(sys.modules, {"backend": backend, "backend.memory_management": mm}):
            out = wrapper.decode(torch.randn(1, 16, 1, 8, 8))
        self.assertEqual(order, ["load", "decode"])
        self.assertEqual(tuple(out.shape), (1, 1, 128, 128, 3))

    def test_build_decoder_builds_on_offload_device_and_wraps_patcher(self):
        built = {}

        class FakeWanVAE(torch.nn.Module):
            def __init__(self, conv_out_channels=3, **cfg):
                super().__init__()
                built["conv_out_channels"] = conv_out_channels
                built["cfg"] = cfg
                self.head = torch.nn.Conv2d(4, conv_out_channels, 1)

        wan_vae = types.ModuleType("backend.nn.wan_vae")
        wan_vae.WanVAE = FakeWanVAE
        nn_pkg = types.ModuleType("backend.nn")
        nn_pkg.wan_vae = wan_vae
        patcher_mod = types.ModuleType("backend.patcher.base")
        patcher_mod.ModelPatcher = FakePatcher
        patcher_pkg = types.ModuleType("backend.patcher")
        patcher_pkg.base = patcher_mod
        mm = types.SimpleNamespace(vae_offload_device=lambda: torch.device("cpu"))
        backend = types.ModuleType("backend")
        backend.nn = nn_pkg
        backend.patcher = patcher_pkg
        backend.memory_management = mm
        st_torch = types.ModuleType("safetensors.torch")
        st_torch.load_file = lambda path: {"vae.head.weight": torch.zeros(12, 4, 1, 1), "vae.head.bias": torch.zeros(12)}
        st_pkg = types.ModuleType("safetensors")
        st_pkg.torch = st_torch

        with tempfile.TemporaryDirectory() as d:
            path = _write_fake_safetensors(Path(d) / "spacepxl.safetensors", 12)
            with mock.patch.dict(sys.modules, {
                "backend": backend, "backend.nn": nn_pkg, "backend.nn.wan_vae": wan_vae,
                "backend.patcher": patcher_pkg, "backend.patcher.base": patcher_mod,
                "backend.memory_management": mm,
                "safetensors": st_pkg, "safetensors.torch": st_torch,
            }), mock.patch.dict(vae2x._DECODER_CACHE, {}, clear=True), mock.patch.object(vae2x, "_log"):
                load_dev = torch.device("cuda:0")
                out = vae2x._build_decoder(path, load_dev, torch.float32)
                self.assertIsInstance(out, FakePatcher)
                self.assertEqual(out.load_device, load_dev)
                self.assertEqual(out.offload_device, torch.device("cpu"))
                # 가중치는 오프로드 장치(CPU)에 그대로 — 지을 때 GPU 로 올리지 않는다.
                self.assertEqual(next(out.model.parameters()).device.type, "cpu")
                self.assertFalse(out.model.training)
                self.assertEqual(built["conv_out_channels"], 12)
                # 접두사 'vae.' 가 벗겨져 strict=False 로도 head 가 실제로 로드된다.
                self.assertEqual(tuple(out.model.head.weight.shape), (12, 4, 1, 1))
                # 캐시: 같은 (path, device, dtype) 는 재빌드하지 않는다.
                again = vae2x._build_decoder(path, load_dev, torch.float32)
                self.assertIs(again, out)

    def test_build_decoder_rejects_3ch_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_fake_safetensors(Path(d) / "stock.safetensors", 3)
            with mock.patch.dict(vae2x._DECODER_CACHE, {}, clear=True), mock.patch.object(vae2x, "_log"):
                self.assertIsNone(vae2x._build_decoder(path, torch.device("cpu"), torch.float32))

    def test_strip_prefix(self):
        sd = {"first_stage_model.a": 1, "first_stage_model.b": 2}
        self.assertEqual(vae2x._strip_prefix(sd), {"a": 1, "b": 2})
        self.assertEqual(vae2x._strip_prefix({"a": 1}), {"a": 1})


# ---------------------------------------------------------------------------
# --novram 폴백 로그: load_models_gpu 가 force_full_load 를 무시하고(NO_VRAM →
# lowvram_model_memory=0.1) 수동 캐스트가 없는 모듈(nn.Conv2d 등)을 CPU 에 남기면
# 디코드는 항상 장치 불일치로 실패해 순정 decode 로 간다. 결과 경로는 그대로 두고,
# 이유를 담은 로그를 한 번만 남긴다. GPU 가 없으니 load_device 는 "meta" 로 흉내 낸다.
# ---------------------------------------------------------------------------


class _ManualCastConv(torch.nn.Module):
    """Forge ops 모듈 흉내: parameters_manual_cast=True 면 가중치를 그때그때 캐스트한다."""

    parameters_manual_cast = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class CpuStrandedFallbackLogTests(unittest.TestCase):
    def setUp(self):
        self.mm = types.SimpleNamespace(
            load_models_gpu=lambda models, **kw: None,       # NO_VRAM: 아무것도 안 올림
            vram_state=types.SimpleNamespace(name="NO_VRAM"),
        )
        backend = types.ModuleType("backend")
        backend.memory_management = self.mm
        patcher_ctx = mock.patch.dict(
            sys.modules, {"backend": backend, "backend.memory_management": self.mm}
        )
        patcher_ctx.start()
        self.addCleanup(patcher_ctx.stop)
        flag_ctx = mock.patch.object(vae2x, "_CPU_STRANDED_LOGGED", False)
        flag_ctx.start()
        self.addCleanup(flag_ctx.stop)

    def _wrapper(self, decoder):
        # 가중치는 CPU, load_device 는 "GPU"(meta) — --novram 부분 로드 상태.
        patcher = FakePatcher(decoder, torch.device("meta"), torch.device("cpu"))
        stock = FakeStockVAE()
        return vae2x._VAE2xWrapper(stock, patcher, False, 0.5, False), stock

    def test_stranded_params_lists_plain_modules_off_device(self):
        dec = FakeDecoder()
        self.assertEqual(vae2x._stranded_params(dec, torch.device("meta")), ["p"])
        self.assertEqual(vae2x._stranded_params(dec, torch.device("cpu")), [])
        self.assertEqual(vae2x._stranded_params(None, torch.device("meta")), [])
        self.assertEqual(vae2x._stranded_params(dec, None), [])

    def test_stranded_params_ignores_manual_cast_modules(self):
        # lowvram 로 CPU 에 남아도 ops 모듈은 캐스트돼 동작하므로 원인이 아니다.
        dec = FakeDecoder()
        dec.cast = _ManualCastConv()
        self.assertEqual(vae2x._stranded_params(dec, torch.device("meta")), ["p"])
        cast_only = torch.nn.Module()
        cast_only.cast = _ManualCastConv()
        self.assertEqual(vae2x._stranded_params(cast_only, torch.device("meta")), [])

    def test_stranded_failure_logs_reason_once_and_still_falls_back(self):
        wrapper, stock = self._wrapper(FakeDecoder(fail=True))
        with mock.patch.object(vae2x, "_log") as log:
            out1 = wrapper.decode(torch.randn(1, 16, 1, 8, 8))
            out2 = wrapper.decode(torch.randn(1, 16, 1, 8, 8))
        # 결과 경로 그대로: 매번 시도 후 순정 decode.
        self.assertEqual(len(stock.decode_calls), 2)
        self.assertEqual(tuple(out1.shape), (1, 1, 64, 64, 3))
        self.assertEqual(tuple(out2.shape), (1, 1, 64, 64, 3))
        # 로그는 한 번만, 이유(CPU 잔류·VRAM 상태·--novram)와 함께.
        self.assertEqual(log.call_count, 1)
        msg = log.call_args[0][0]
        self.assertIn("CPU", msg)
        self.assertIn("NO_VRAM", msg)
        self.assertIn("--novram", msg)
        self.assertIn("순정 decode", msg)
        self.assertIn("RuntimeError", msg)

    def test_non_stranded_failure_keeps_generic_log_every_time(self):
        # 가중치가 load_device 에 다 올라간 경우의 실패는 예전처럼 매번 알린다.
        patcher = FakePatcher(FakeDecoder(fail=True), torch.device("cpu"), torch.device("cpu"))
        stock = FakeStockVAE()
        wrapper = vae2x._VAE2xWrapper(stock, patcher, False, 0.5, False)
        with mock.patch.object(vae2x, "_log") as log:
            wrapper.decode(torch.randn(1, 16, 1, 8, 8))
            wrapper.decode(torch.randn(1, 16, 1, 8, 8))
        self.assertEqual(log.call_count, 2)
        for call in log.call_args_list:
            self.assertIn("2x decode failed → stock decode: RuntimeError: boom", call[0][0])
        self.assertEqual(len(stock.decode_calls), 2)

    def test_stranded_log_without_vram_state_still_explains(self):
        del self.mm.vram_state
        wrapper, stock = self._wrapper(FakeDecoder(fail=True))
        with mock.patch.object(vae2x, "_log") as log:
            wrapper.decode(torch.randn(1, 16, 1, 8, 8))
        self.assertEqual(log.call_count, 1)
        self.assertIn("CPU", log.call_args[0][0])
        self.assertEqual(len(stock.decode_calls), 1)

    def test_stranded_but_successful_decode_is_silent(self):
        # 진단은 실패했을 때만 — 성공하면 결과도 로그도 바뀌지 않는다.
        dec = FakeDecoder()
        patcher = FakePatcher(dec, torch.device("cpu"), torch.device("cpu"))
        wrapper = vae2x._VAE2xWrapper(FakeStockVAE(), patcher, False, 0.5, False)
        with mock.patch.object(vae2x, "_log") as log:
            out = wrapper.decode(torch.randn(1, 16, 1, 8, 8))
        self.assertEqual(tuple(out.shape), (1, 1, 128, 128, 3))
        log.assert_not_called()


# ---------------------------------------------------------------------------
# 스크립트 진입: args 파싱, 재래핑 가드, load_device 전달
# ---------------------------------------------------------------------------


def _processing(vae):
    forge_objects = types.SimpleNamespace(vae=vae)
    sd_model = types.SimpleNamespace(forge_objects=forge_objects)
    return types.SimpleNamespace(sd_model=sd_model, extra_generation_params={})


class ScriptEntryTests(unittest.TestCase):
    def setUp(self):
        self.script = vae2x.AnimaVAE2x()
        self.built = []

        def fake_build(path, device, dtype):
            self.built.append((path, device, dtype))
            return FakePatcher(FakeDecoder(), device, torch.device("cpu"))

        self._patches = [
            mock.patch.object(vae2x, "_resolve_vae_path", lambda name: None if name == "None" else f"/models/VAE/{name}"),
            mock.patch.object(vae2x, "is_spacepxl_2x", lambda path: "2x" in path),
            mock.patch.object(vae2x, "_build_decoder", fake_build),
            mock.patch.object(vae2x, "_log"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_disabled_leaves_vae_untouched(self):
        stock = FakeStockVAE()
        p = _processing(stock)
        self.script.process_before_every_sampling(p, False, "spacepxl_2x.safetensors", "2x upscaled", 0.5, False)
        self.assertIs(p.sd_model.forge_objects.vae, stock)
        self.assertEqual(self.built, [])
        self.assertEqual(p.extra_generation_params, {})

    def test_no_vae_or_non_2x_file_skips(self):
        stock = FakeStockVAE()
        for name in ("None", "stock_wan.safetensors"):
            with self.subTest(name=name):
                p = _processing(stock)
                self.script.process_before_every_sampling(p, True, name, "2x upscaled", 0.5, False)
                self.assertIs(p.sd_model.forge_objects.vae, stock)
        self.assertEqual(self.built, [])

    def test_attach_uses_load_device_and_records_infotext(self):
        stock = FakeStockVAE()
        stock.device = torch.device("cuda:0")
        p = _processing(stock)
        self.script.process_before_every_sampling(p, True, "spacepxl_2x.safetensors", "1x refined (downsample)", 0.7, True)
        vae = p.sd_model.forge_objects.vae
        self.assertIsInstance(vae, vae2x._VAE2xWrapper)
        self.assertIs(vae._orig, stock)
        self.assertTrue(vae._refine_1x)
        self.assertEqual(vae._blur_sigma, 0.7)
        self.assertTrue(vae._renorm)
        self.assertEqual(self.built, [("/models/VAE/spacepxl_2x.safetensors", torch.device("cuda:0"), torch.float32)])
        # Attach records the settings with decode=pending; the real decode rewrites it
        # (applied / stock fallback) together with the per-pass outcome key.
        self.assertEqual(
            p.extra_generation_params["Anima VAE 2x"],
            "spacepxl_2x.safetensors, 1x-refined, blur=0.7, renorm=True, decode=pending",
        )
        self.assertEqual(p.extra_generation_params["Anima VAE 2x main outcome"], "pending decode")

    def test_rewrap_guard_wraps_stock_not_wrapper(self):
        stock = FakeStockVAE()
        p = _processing(stock)
        self.script.process_before_every_sampling(p, True, "spacepxl_2x.safetensors", "2x upscaled", 0.5, False)
        first = p.sd_model.forge_objects.vae
        self.script.process_before_every_sampling(p, True, "spacepxl_2x.safetensors", "2x upscaled", 0.5, False)
        second = p.sd_model.forge_objects.vae
        self.assertIsNot(second, first)
        self.assertIs(second._orig, stock, "래퍼를 다시 감싸면 pixel-shuffle 이 두 번 적용된다")

    def test_decoder_unavailable_leaves_stock(self):
        stock = FakeStockVAE()
        p = _processing(stock)
        with mock.patch.object(vae2x, "_build_decoder", lambda *a: None):
            self.script.process_before_every_sampling(p, True, "spacepxl_2x.safetensors", "2x upscaled", 0.5, False)
        self.assertIs(p.sd_model.forge_objects.vae, stock)

    def test_missing_forge_objects_is_safe(self):
        p = types.SimpleNamespace(sd_model=None, extra_generation_params={})
        self.script.process_before_every_sampling(p, True, "spacepxl_2x.safetensors", "2x upscaled", 0.5, False)
        self.assertEqual(self.built, [])

    def test_ui_returns_five_components(self):
        comps = self.script.ui(False)
        self.assertEqual(len(comps), 5)
        self.assertEqual(comps[0].elem_id, "anima_vae2x_enable")
        self.assertFalse(comps[0].value)


if __name__ == "__main__":
    unittest.main()
