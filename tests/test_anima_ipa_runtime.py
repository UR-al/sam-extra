"""IPA 런타임 — 받기 잠금, 로드 캐시, 한 번의 생성용 세션."""
from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.anima_ipa.checkpoint import AdapterSpec  # noqa: E402
from sam3ext.anima_ipa.options import IpaOptions  # noqa: E402
from sam3ext.anima_ipa.runtime import IpaRuntime, LoadedAdapter  # noqa: E402


class OptionTests(unittest.TestCase):
    def test_defaults_match_the_spec(self):
        options = IpaOptions()
        self.assertEqual(options.strength, 1.0)
        self.assertEqual(options.ref_size, 512)
        self.assertFalse(options.separate_cfg)
        self.assertEqual(options.cfg_scale, 4.0)
        self.assertEqual(options.siglip_layer, -1)
        self.assertFalse(options.gray_null)
        self.assertTrue(options.use_lora)
        options.validate()

    def test_out_of_range_values_are_refused(self):
        for bad in (
            IpaOptions(strength=-0.1),
            IpaOptions(strength=2.5),
            IpaOptions(ref_size=100),
            IpaOptions(ref_size=2048),
            IpaOptions(cfg_scale=0.5),
            IpaOptions(cfg_scale=11.0),
            IpaOptions(siglip_layer=-2),
            IpaOptions(siglip_layer=25),
        ):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_the_reference_size_must_be_a_multiple_of_the_patch_size(self):
        with self.assertRaises(ValueError):
            IpaOptions(ref_size=500).validate()
        IpaOptions(ref_size=512).validate()


def _spec(**overrides):
    values = dict(
        num_blocks=2, embed_dim=8, inner_dim=8, shared_projection=False, compressor=None,
        has_self_attn=False, has_siglip_norm=False, has_null_tokens=False,
        lora_blocks=(), lora_rank=0, norm_keys=False, unsupported=(),
    )
    values.update(overrides)
    return AdapterSpec(**values)


class _FakeEncoder:
    def __init__(self, tokens=None):
        self.tokens = tokens if tokens is not None else torch.ones(1, 4, 8)
        self.calls = []

    def encode(self, image, *, size, layer=-1, device="cpu"):
        self.calls.append({"size": size, "layer": layer, "image": image, "device": device})
        return self.tokens


class DownloadTests(unittest.TestCase):
    def test_download_reports_true_and_false_without_blocking(self):
        runtime = IpaRuntime(models_dir=Path("nowhere"))
        with mock.patch("sam3ext.anima_ipa.paths.download") as download:
            self.assertTrue(runtime.download(downloader=object()))
            download.assert_called_once()
            runtime._download_lock.acquire()
            try:
                self.assertFalse(runtime.download(downloader=object()))
            finally:
                runtime._download_lock.release()

    def test_the_lock_is_released_even_when_the_download_raises(self):
        runtime = IpaRuntime(models_dir=Path("nowhere"))
        with mock.patch("sam3ext.anima_ipa.paths.download", side_effect=OSError("net")):
            with self.assertRaises(OSError):
                runtime.download(downloader=object())
        self.assertTrue(runtime._download_lock.acquire(blocking=False))
        runtime._download_lock.release()


class LoadTests(unittest.TestCase):
    def test_load_happens_once_and_is_cached(self):
        loaded = LoadedAdapter(spec=_spec(), weights={}, encoder=_FakeEncoder())
        calls = []

        def loader(directory):
            calls.append(directory)
            return loaded

        runtime = IpaRuntime(models_dir=Path("nowhere"), loader=loader)
        self.assertIs(runtime.load(), loaded)
        self.assertIs(runtime.load(), loaded)
        self.assertEqual(len(calls), 1)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.encoder = _FakeEncoder()
        self.loaded = LoadedAdapter(spec=_spec(), weights={}, encoder=self.encoder)
        self.runtime = IpaRuntime(
            models_dir=Path("nowhere"), loader=lambda directory: self.loaded
        )
        self.seen = {}

        @contextmanager
        def fake_patched_unet(sd_model, spec, weights, injection, *, use_lora):
            self.seen.update(
                sd_model=sd_model, spec=spec, weights=weights,
                injection=injection, use_lora=use_lora,
            )
            yield

        self.fake_patched_unet = fake_patched_unet

    def _session(self, options):
        with mock.patch(
            "sam3ext.anima_ipa.runtime.patched_unet", self.fake_patched_unet
        ):
            with self.runtime.session(object(), Image.new("RGB", (64, 64)), options):
                pass

    def test_the_session_encodes_and_hands_the_injection_to_the_patch(self):
        self._session(
            IpaOptions(strength=0.5, ref_size=256, siglip_layer=2, use_lora=False)
        )
        self.assertEqual(self.encoder.calls[0]["size"], 256)
        self.assertEqual(self.encoder.calls[0]["layer"], 2)
        self.assertEqual(self.seen["injection"].gate_scale, 0.5)
        self.assertFalse(self.seen["use_lora"])

    def test_invalid_options_never_reach_the_encoder(self):
        with self.assertRaises(ValueError):
            self._session(IpaOptions(ref_size=500))
        self.assertEqual(self.encoder.calls, [])

    def test_without_learned_null_tokens_the_fallback_is_a_grey_encode(self):
        """0 으로 두면 uncond 게이트가 꺼져 IP 기여가 텍스트 CFG 로 증폭된다."""
        self._session(IpaOptions())
        self.assertEqual(len(self.encoder.calls), 2)   # 참조 + 회색
        grey = self.encoder.calls[1]["image"]
        self.assertEqual(grey.getpixel((0, 0)), (128, 128, 128))
        injection = self.seen["injection"]
        self.assertEqual(tuple(injection.null_tokens.shape), tuple(injection.tokens.shape))

    def test_learned_null_tokens_are_used_when_the_checkpoint_has_them(self):
        stored = torch.full((1, 4, 8), 0.25)
        self.loaded.spec = _spec(has_null_tokens=True)
        self.loaded.weights = {"null_tokens": stored}
        self._session(IpaOptions())
        self.assertEqual(len(self.encoder.calls), 1)   # 회색을 굽지 않는다
        self.assertTrue(torch.allclose(self.seen["injection"].null_tokens, stored))

    def test_gray_null_overrides_the_learned_tokens(self):
        self.loaded.spec = _spec(has_null_tokens=True)
        self.loaded.weights = {"null_tokens": torch.full((1, 4, 8), 0.25)}
        self._session(IpaOptions(gray_null=True))
        self.assertEqual(len(self.encoder.calls), 2)

    def test_cfg_settings_travel_into_the_injection(self):
        self._session(IpaOptions(separate_cfg=True, cfg_scale=5.5))
        self.assertTrue(self.seen["injection"].separate_cfg)
        self.assertEqual(self.seen["injection"].cfg_scale, 5.5)


if __name__ == "__main__":
    unittest.main()
