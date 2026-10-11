"""XYZ ``VAE`` axis — the pure logic of sam3ext/xyz_vae.py (no Forge, no torch).

Families from headers, the strict-load comparison (the keys ``_load_vae_dict`` drops, dtypes not compared), the VAE
Forge builds from a module list or a checkpoint, the cell's list with the VAE replaced, the header cache, and the
development PC's five VAE files (as tiny files of their family). Forge's own code runs in tests/test_xyz_vae_origin.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import xyz_text_encoder as xte  # noqa: E402
from sam3ext import xyz_vae as xvae  # noqa: E402
from tests import _xyz_text_encoder_support as tfx  # noqa: E402
from tests import _xyz_vae_support as fx  # noqa: E402


def shapes_of(spec: dict) -> dict:
    return {key: tuple(shape) for key, shape in spec.items()}


class FamilyTests(unittest.TestCase):
    def test_families(self):
        self.assertEqual(xvae.vae_family(tfx.header(fx.tiny_3d())), "3d")
        self.assertEqual(xvae.vae_family(tfx.header(fx.tiny_2d())), "2d")
        self.assertEqual(xvae.vae_family(tfx.header(fx.tiny_kl())), "kl")
        self.assertEqual(xvae.vae_family(tfx.header(fx.tiny_kl(diffusers=True))), "kl")
        self.assertEqual(xvae.vae_family(shapes_of(fx.tiny_2d())), "2d", "a shapes dict works too")
        # Forge's rule (backend/loader.py): post_quant_conv.weight makes a Wan-family file 2D
        self.assertEqual(xvae.vae_family({**fx.tiny_3d(), "post_quant_conv.weight": [4, 4, 1, 1]}), "2d")
        self.assertEqual(xvae.vae_family(tfx.CASES["vae_wan"][0]), "3d")
        self.assertEqual(xvae.vae_family(tfx.CASES["vae_sd"][0]), "kl")

    def test_not_a_vae(self):
        for name in ("qwen3_06b", "t5xxl", "clip_l_hf", "empty"):
            with self.subTest(case=name):
                self.assertIsNone(xvae.vae_family(tfx.header(tfx.CASES[name][0])))
        self.assertIsNone(xvae.vae_family(None))
        self.assertIsNone(xvae.vae_family({"__metadata__": {"x": "y"}}))

    def test_describe(self):
        self.assertEqual(xvae.describe(tfx.header(fx.tiny_3d(z=16))), "3D Wan / Qwen-Image VAE (16 ch)")
        self.assertEqual(xvae.describe(shapes_of(fx.tiny_2d())), "2D Qwen2D VAE (4 ch)")
        self.assertEqual(xvae.describe(tfx.header(fx.tiny_kl(z=4))), "AutoencoderKL VAE (4 ch)")
        self.assertEqual(xvae.describe({"decoder.middle.0.residual.0.gamma": {"shape": [8, 1, 1, 1]}}),
                         "3D Wan / Qwen-Image VAE")
        self.assertIn("does not know", xvae.describe({}))


class StrictLoadTests(unittest.TestCase):
    def test_dropped_keys(self):
        """modules/sd_vae.py _load_vae_dict drops keys starting with "loss" and vae_ignore_keys before the load."""
        header = tfx.header({**fx.tiny_3d(), "loss.logvar": [1], "lossy": [2], "model_ema.decay": [1],
                             "model_ema.num_updates": [1]}, metadata={"format": "pt"})
        self.assertEqual(xvae.strict_shapes(header), shapes_of(fx.tiny_3d()))

    def test_not_a_header(self):
        self.assertIsNone(xvae.strict_shapes(None))
        self.assertIsNone(xvae.strict_shapes({"a": {"dtype": "F32"}}))
        self.assertIsNone(xvae.strict_shapes({"a": {"shape": ["x"]}}))
        self.assertEqual(xvae.strict_shapes({}), {})

    def test_swappable(self):
        three, two = shapes_of(fx.tiny_3d()), shapes_of(fx.tiny_2d())
        self.assertTrue(xvae.swappable(three, dict(three)))
        self.assertTrue(xvae.swappable(two, dict(two)))
        self.assertFalse(xvae.swappable(three, two))
        self.assertFalse(xvae.swappable(two, three))
        self.assertFalse(xvae.swappable(three, shapes_of(fx.tiny_3d(z=16))), "same family, other shapes")
        self.assertFalse(xvae.swappable(three, {**three, "extra.weight": (1,)}))
        self.assertFalse(xvae.swappable(three, {k: v for k, v in three.items() if k != "conv1.bias"}))
        self.assertFalse(xvae.swappable(None, three))
        self.assertFalse(xvae.swappable(three, None))

    def test_dtypes_are_not_compared(self):
        bf16 = xvae.strict_shapes({k: {"dtype": "BF16", "shape": v} for k, v in fx.tiny_3d().items()})
        f32 = xvae.strict_shapes({k: {"dtype": "F32", "shape": v} for k, v in fx.tiny_3d().items()})
        self.assertTrue(xvae.swappable(bf16, f32))

    def test_module_shapes(self):
        class T:
            def __init__(self, shape):
                self.shape = shape
        self.assertEqual(xvae.module_shapes({"a": T((1, 2)), "b": T([3])}), {"a": (1, 2), "b": (3,)})

    def test_mismatch(self):
        three, two = shapes_of(fx.tiny_3d()), shapes_of(fx.tiny_2d())
        text = xvae.mismatch(three, two)
        self.assertIn(f"{len(three)} of the model's VAE tensors are not in the file", text)
        self.assertIn(f"{len(two)} tensors of the file are not in the model's VAE", text)
        self.assertIn("7 tensors have other shapes (e.g. conv1.bias)", xvae.mismatch(three, shapes_of(fx.tiny_3d(z=16))))


class Kind:
    def __init__(self, kind):
        self.kind = kind


def classify_by_name(path):
    name = os.path.basename(str(path))
    if name.startswith("vae"):
        return Kind("vae")
    if name.endswith(".pt"):
        return Kind("opaque")
    if name.startswith("broken"):
        return Kind("unreadable")
    return Kind("te")


class ModuleListTests(unittest.TestCase):
    TE = "C:/m/text_encoder/qwen.safetensors"
    VAE3D = "C:/m/VAE/vae3d.safetensors"
    VAE2D = "C:/m/VAE/vae2d.safetensors"

    def test_replace_the_vae(self):
        self.assertEqual(xvae.replace_vae([self.VAE3D, self.TE], self.VAE2D, classify_by_name),
                         sorted([self.VAE2D, self.TE]))

    def test_added_when_the_list_has_none(self):
        """No VAE file: the checkpoint's own VAE - the pick replaces it, as when chosen in the dropdown."""
        self.assertEqual(xvae.replace_vae([self.TE], self.VAE2D, classify_by_name), sorted([self.TE, self.VAE2D]))
        self.assertEqual(xvae.replace_vae([], self.VAE2D, classify_by_name), [self.VAE2D])

    def test_every_vae_entry_goes(self):
        other = "C:/m/VAE/vae_other.safetensors"
        self.assertEqual(xvae.replace_vae([self.VAE3D, other, self.TE], self.VAE2D, classify_by_name),
                         sorted([self.VAE2D, self.TE]))

    def test_pickle_files(self):
        """A pickle (or unreadable) file is the VAE only when it is in Forge's VAE list (sd_vae.vae_dict)."""
        vae_pt = "C:/m/VAE/old.pt"
        te_pt = "C:/m/text_encoder/enc.pt"
        broken = "C:/m/VAE/broken.safetensors"
        entries = [vae_pt, te_pt, broken, self.TE]
        names = {"old.pt", "broken.safetensors"}
        self.assertEqual(xvae.vae_entries(entries, classify_by_name, names), [vae_pt, broken])
        self.assertEqual(xvae.vae_entries(entries, classify_by_name), [])
        self.assertEqual(xvae.replace_vae(entries, self.VAE2D, classify_by_name, names),
                         sorted([te_pt, self.TE, self.VAE2D]))

    def test_no_duplicates(self):
        self.assertEqual(xvae.replace_vae([self.TE, self.TE], self.VAE2D, classify_by_name),
                         sorted([self.TE, self.VAE2D]))

    def test_same_file(self):
        self.assertTrue(xvae.same_file("C:/m/VAE/a.safetensors", os.path.join("C:/m/VAE", "a.safetensors")))
        self.assertFalse(xvae.same_file("C:/m/VAE/a.safetensors", "C:/m/VAE/b.safetensors"))


class PredictionTests(unittest.TestCase):
    def read(self, table):
        return lambda path: table.get(os.path.basename(path))

    def test_last_vae_file_wins(self):
        """forge_loader applies the module list in order; each VAE file replaces the VAE (replace_state_dict)."""
        table = {"vae_a.safetensors": shapes_of(fx.tiny_3d()), "vae_b.safetensors": shapes_of(fx.tiny_2d())}
        entries = ["C:/m/VAE/vae_a.safetensors", "C:/m/t/te.safetensors", "C:/m/VAE/vae_b.safetensors"]
        self.assertEqual(xvae.predicted_shapes(entries, None, classify=classify_by_name, read=self.read(table)),
                         shapes_of(fx.tiny_2d()))

    def test_checkpoint_vae_without_a_vae_file(self):
        checkpoint = shapes_of(fx.tiny_kl())
        self.assertEqual(xvae.predicted_shapes(["C:/t/te.safetensors"], checkpoint, classify=classify_by_name,
                                               read=self.read({})), checkpoint)
        self.assertIsNone(xvae.predicted_shapes([], None, classify=classify_by_name, read=self.read({})))

    def test_unknown(self):
        """An unreadable VAE file, or one replace_state_dict converts first (its names are not the model's)."""
        entries = ["C:/m/VAE/vae_a.safetensors"]
        self.assertIsNone(xvae.predicted_shapes(entries, None, classify=classify_by_name, read=self.read({})))
        for spec in (fx.tiny_kl(diffusers=True), {"decoder.post_quant_conv.weight": [4, 4, 1, 1]},
                     {"first_stage_model.decoder.conv_in.weight": [8, 4, 3, 3]}):
            with self.subTest(spec=sorted(spec)[0]):
                table = {"vae_a.safetensors": shapes_of(spec)}
                self.assertIsNone(xvae.predicted_shapes(entries, None, classify=classify_by_name,
                                                        read=self.read(table)))

    def test_checkpoint_vae_shapes(self):
        unet = {"model.diffusion_model.input_blocks.0.0.weight": [320, 4, 3, 3]}
        sdxl = {**unet, **{f"first_stage_model.{k}": v for k, v in fx.tiny_kl().items()}, "loss.logvar": [1]}
        self.assertEqual(xvae.checkpoint_vae_shapes(tfx.header(sdxl)), shapes_of(fx.tiny_kl()))
        flux = {"model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": [128],
                **{f"vae.{k}": v for k, v in fx.tiny_kl(z=16).items()}}
        self.assertEqual(xvae.checkpoint_vae_shapes(tfx.header(flux)), shapes_of(fx.tiny_kl(z=16)))
        self.assertIsNone(xvae.checkpoint_vae_shapes(tfx.header(unet)), "no VAE built in")
        # a UNet-only file: preprocess_state_dict prefixes every key, nothing is a VAE
        self.assertIsNone(xvae.checkpoint_vae_shapes(tfx.header({f"vae.{k}": v for k, v in fx.tiny_kl().items()})))
        self.assertIsNone(xvae.checkpoint_vae_shapes(None))


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        xte.clear_cache()

    def tearDown(self):
        xte.clear_cache()
        self.tmp.cleanup()

    def test_file_shapes_from_the_header(self):
        path = fx.write_vae(self.folder / "a.safetensors", {**fx.tiny_2d(), "loss.w": [1]}, 2.0, "BF16")
        self.assertEqual(xvae.file_shapes(path), shapes_of(fx.tiny_2d()))
        self.assertEqual(xte.classify_file(str(path)).kind, "vae", "the Text Encoder axis calls it a VAE too")

    def test_cache_follows_the_file(self):
        path = fx.write_vae(self.folder / "a.safetensors", fx.tiny_3d(), 1.0)
        self.assertEqual(xvae.vae_family(xvae.file_shapes(path)), "3d")
        stat = os.stat(path)
        fx.write_vae(path, fx.tiny_2d(), 1.0)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        self.assertEqual(xvae.vae_family(xvae.file_shapes(path)), "2d")

    def test_not_read(self):
        pickle = self.folder / "a.pt"
        pickle.write_bytes(b"\x80\x02}q\x00.")
        self.assertIsNone(xvae.file_shapes(pickle))
        broken = self.folder / "b.safetensors"
        broken.write_bytes(b"\xff" * 16)
        self.assertIsNone(xvae.file_shapes(broken))
        self.assertIsNone(xvae.file_shapes(self.folder / "missing.safetensors"))
        self.assertIsNone(xvae.file_shapes(None))

    def test_checkpoint_file_shapes(self):
        unet = {"model.diffusion_model.input_blocks.0.0.weight": [320, 4, 3, 3]}
        path = tfx.write_safetensors(self.folder / "sdxl.safetensors",
                                     {**unet, **{f"first_stage_model.{k}": v for k, v in fx.tiny_kl().items()}},
                                     fake_shapes=True)
        self.assertEqual(xvae.checkpoint_file_shapes(path), shapes_of(fx.tiny_kl()))
        self.assertIsNone(xvae.checkpoint_file_shapes(self.folder / "x.ckpt"))
        self.assertIsNone(xvae.checkpoint_file_shapes(None))

    def test_the_users_five_vaes(self):
        """Two 3D Wan files (qwen_image_vae BF16, its HDR fp32) and three 2D Qwen2D files: the strict load works
        within a family, never across."""
        paths = fx.write_user_vaes(self.folder)
        families = {name: xvae.vae_family(xvae.file_shapes(path)) for name, path in paths.items()}
        self.assertEqual(families, {name: spec[0] for name, spec in fx.USER_VAES.items()})
        for base in paths:
            for pick in paths:
                with self.subTest(base=base, pick=pick):
                    same = fx.USER_VAES[base][0] == fx.USER_VAES[pick][0]
                    self.assertEqual(xvae.swappable(xvae.file_shapes(paths[base]), xvae.file_shapes(paths[pick])),
                                     same)

    def test_files_are_written_as_valid_safetensors(self):
        path = fx.write_vae(self.folder / "v.safetensors", {"a": [2, 3], "b": [1]}, 5.0, "BF16")
        header = xte.read_header(str(path))
        self.assertEqual(header["a"]["shape"], [2, 3])
        self.assertEqual(header["b"]["data_offsets"], [12, 14])
        raw = path.read_bytes()
        start = 8 + int.from_bytes(raw[:8], "little")
        self.assertEqual(raw[start:], b"\xa0\x40" * 7, "seven BF16 5.0 values")


if __name__ == "__main__":
    unittest.main()
