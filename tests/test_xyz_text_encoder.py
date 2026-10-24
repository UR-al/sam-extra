"""XYZ ``[Modules] Text Encoder`` — the pure logic (sam3ext/xyz_text_encoder.py, sam3ext/gguf_header.py), without Forge.

* the slot classifier on headers of every replace_state_dict branch (Forge's own function on the same headers:
  tests/test_xyz_text_encoder_origin.py) and the special files (Qwen3.5, the Anima 3.8B adapter, VAEs …);
* real small ``.safetensors`` and ``.gguf`` files: the header readers and their (path, size, mtime) cache, GGUF tensor
  names through Forge's renames;
* what a checkpoint has built in (model type, encoder slots);
* values (name, relative path, stem, ``a + b``, ``None``), the cell's module list (``compose``, also with the
  checkpoint's built-in encoders and on top of another Text Encoder axis), the dropdown choices, the cell labels and the
  cell's ``Module N`` list.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import gguf_header  # noqa: E402
from sam3ext import xyz_text_encoder as xte  # noqa: E402
from tests import _xyz_text_encoder_support as fx  # noqa: E402
from sam3ext.xyz_text_encoder import Checkpoint, Kind  # noqa: E402


class TeSlotsTests(unittest.TestCase):
    def test_every_case(self):
        for name, (shapes, metadata, expected) in fx.CASES.items():
            with self.subTest(case=name):
                h = fx.header(shapes, metadata)
                if expected == fx.RAISES:
                    with self.assertRaises(xte.ForgeLoadError):
                        xte.te_slots(h)
                else:
                    self.assertEqual(xte.te_slots(h), frozenset(expected))

    def test_qwen35_9b(self):
        """Forge 2.31's Qwen3.5 9B branch (MOSS-Image) comes before the Gemma-2 / Qwen chain and asserts the width."""
        h = fx.header(fx.CASES["qwen35_9b_with_qwen3_keys"][0])
        self.assertEqual(xte.classify_header("qwen3.5_9b_moss.safetensors", h), Kind("te", ("qwen35_9b",)))
        h = fx.header(fx.CASES["qwen35_other_width"][0])
        kind = xte.classify_header("qwen3.5_4b_hf.safetensors", h)
        self.assertEqual(kind.kind, "invalid")
        self.assertIn("not 4096 wide", kind.detail)
        # Anima 3.8B's Qwen3.5 file (no "model." key) stays what it was
        self.assertEqual(self.kind_of("qwen35_4b", "qwen35_4b.safetensors").kind, "qwen35")
        self.assertEqual(xte.slot_name("qwen35_9b"), "Qwen3.5 9B")
        # an all-in-one MOSS checkpoint: its encoder is built in
        moss = xte.checkpoint_profile("moss.safetensors", fx.header(fx.CHECKPOINTS["moss_all_in_one"][0]))
        self.assertEqual(moss.slots, {"qwen35_9b"})

    @staticmethod
    def kind_of(case, name):
        shapes, metadata, _ = fx.CASES[case]
        return xte.classify_header(name, fx.header(shapes, metadata))

    def test_order_qwen3vl_before_qwen3(self):
        h = fx.header(fx.CASES["qwen3vl_heretic_with_qwen3_keys"][0])
        self.assertEqual(xte.classify_header("qwen3-vl-4b-heretic.safetensors", h), Kind("te", ("qwen3vl_4b",)))

    def test_qwen3_widths(self):
        for width, slot in ((1024, "qwen3_06b"), (2560, "qwen3_4b"), (4096, "qwen3_8b"), (5120, "qwen3_8b")):
            with self.subTest(width=width):
                h = fx.header({fx.QWEN3_LAYER: [width], fx.Q_NORM: [128]})
                self.assertEqual(xte.te_slots(h), {slot})

    def test_metadata_is_not_a_key(self):
        h = fx.header({}, {"architecture": "x", fx.QWEN3_LAYER: "1024"})
        self.assertEqual(xte.te_slots(h), frozenset())


class ClassifyTests(unittest.TestCase):
    def kind(self, case, name="file.safetensors"):
        shapes, metadata, _ = fx.CASES[case]
        return xte.classify_header(name, fx.header(shapes, metadata))

    def test_kinds(self):
        self.assertEqual(self.kind("qwen3_06b", "qwen_3_06b_base.safetensors"), Kind("te", ("qwen3_06b",)))
        self.assertEqual(self.kind("qwen35_4b", "qwen35_4b.safetensors").kind, "qwen35")
        self.assertEqual(self.kind("anima38_adapter", "Anima-3.8B-expanded_adapter.safetensors").kind, "adapter")
        self.assertEqual(self.kind("vae_sd").kind, "vae")
        self.assertEqual(self.kind("vae_wan").kind, "vae")
        self.assertEqual(self.kind("tap_w_mean_std", "h3_qwen3vl_8b_tap24.safetensors").kind, "unknown")
        self.assertEqual(self.kind("qwen32b_tail_layers_50_63").kind, "unknown")
        self.assertEqual(self.kind("t5gemma", "t5gemma_b_b_ul2.safetensors").kind, "unknown")
        self.assertEqual(self.kind("qwen32b_two_slots"), Kind("multi", ("qwen25_7b", "qwen3_8b")))
        self.assertEqual(self.kind("gemma2_with_q_norm").kind, "invalid")
        self.assertEqual(xte.classify_header("x.safetensors", None).kind, "unreadable")

    def test_qwen35_marker_needs_no_model_keys(self):
        # a Qwen3.5 file in a format Forge reads (model.* keys) is classified like any other file
        h = fx.header({fx.QWEN3_LAYER: [2560], fx.Q_NORM: [128]})
        self.assertEqual(xte.classify_header("qwen35_4b_hf.safetensors", h), Kind("te", ("qwen3_4b",)))
        # the marker is matched ignoring case, as anima38/files.py does
        self.assertEqual(self.kind("qwen35_4b", "Qwen3.5-4B.safetensors").kind, "qwen35")
        self.assertEqual(self.kind("qwen35_4b", "other.safetensors").kind, "unknown")

    def test_qwen35_filter_only_takes_safetensors(self):
        # sam-extra's loader filter (anima38/files.py is_unused_qwen35_module) drops only .safetensors files
        self.assertEqual(self.kind("qwen35_4b", "qwen35_4b.sft").kind, "unknown")
        self.assertEqual(self.kind("qwen35_4b", "qwen35_4b.gguf").kind, "unknown")

    def test_v2_connector_metadata_is_an_adapter(self):
        h = fx.header({"x": [1]}, {"architecture": "anima_qwen35_quality_anchored_semantic_connector_v2"})
        self.assertEqual(xte.classify_header("c.safetensors", h).kind, "adapter")

    def test_is_te_entry(self):
        self.assertTrue(Kind("te", ("qwen3_06b",)).is_te_entry)
        self.assertTrue(Kind("multi", ("clip_g", "clip_l")).is_te_entry)
        self.assertTrue(Kind("opaque", detail=".pt", in_te_dir=True).is_te_entry)
        for kind in (Kind("opaque", detail=".pt"), Kind("vae"), Kind("qwen35"), Kind("adapter"), Kind("unknown")):
            self.assertFalse(kind.is_te_entry, kind)


class GgufRemapTests(unittest.TestCase):
    def test_every_case(self):
        for name, (shapes, expected) in fx.GGUF_CASES.items():
            with self.subTest(case=name):
                header = xte.gguf_header(shapes)
                self.assertEqual(xte.te_slots(header), frozenset(expected))

    def test_t5_names(self):
        header = xte.gguf_header(fx.GGUF_CASES["t5xxl_q8"][0])
        self.assertEqual(set(header), {"encoder.block.0.layer.0.SelfAttention.k.weight", "shared.weight",
                                       "encoder.block.0.layer.0.layer_norm.weight", "encoder.final_layer_norm.weight"})
        self.assertEqual(header["shared.weight"], {"shape": [32128, 4096]})

    def test_llm_names(self):
        header = xte.gguf_header(fx.GGUF_CASES["qwen3_06b_q8"][0])
        self.assertEqual(set(header), {"model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_norm.weight",
                                       "model.layers.0.post_attention_layernorm.weight",
                                       "model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight"})

    def test_vision_names(self):
        shapes = dict(fx.GGUF_CASES["qwen25_vision_mmproj"][0], **{"v.patch_embd.weight.1": [1280, 3, 14, 14]})
        header = xte.gguf_header(shapes)
        self.assertIn("visual.blocks.0.attn.proj.weight", header)
        self.assertIn("visual.merger.mlp.0.weight", header)
        self.assertNotIn("v.patch_embd.weight.1", header)

    def test_none(self):
        self.assertIsNone(xte.gguf_header(None))


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        xte.clear_cache()

    def tearDown(self):
        xte.clear_cache()
        self.tmp.cleanup()

    def test_reads_real_files(self):
        te = fx.write_safetensors(self.dir / "text_encoder" / "qwen_3_06b_base.safetensors", fx.CASES["qwen3_06b"][0])
        vae = fx.write_safetensors(self.dir / "VAE" / "qwen_image_vae.safetensors", fx.CASES["vae_sd"][0])
        q35 = fx.write_safetensors(self.dir / "text_encoder" / "qwen35_4b.safetensors", fx.CASES["qwen35_4b"][0],
                                   fake_shapes=True)
        adapter = fx.write_safetensors(self.dir / "text_encoder" / "adapter.safetensors",
                                       *fx.CASES["anima38_adapter"][:2])
        sft = fx.write_safetensors(self.dir / "text_encoder" / "t5.sft", fx.CASES["t5xxl"][0], fake_shapes=True)
        self.assertEqual(xte.classify_file(te), Kind("te", ("qwen3_06b",)))
        self.assertEqual(xte.classify_file(vae).kind, "vae")
        self.assertEqual(xte.classify_file(q35).kind, "qwen35")
        self.assertEqual(xte.classify_file(adapter).kind, "adapter")
        self.assertEqual(xte.classify_file(sft), Kind("te", ("t5xxl",)))

    def test_gguf_files_are_read_by_their_tensor_names(self):
        for name, (shapes, expected) in fx.GGUF_CASES.items():
            with self.subTest(case=name):
                path = fx.write_gguf(self.dir / "text_encoder" / f"{name}.gguf", shapes)
                kind = xte.classify_file(path)
                if expected:
                    self.assertEqual(kind, Kind("te", tuple(expected)))
                else:
                    self.assertEqual(kind.kind, "unknown")

    def test_gguf_vae(self):
        path = fx.write_gguf(self.dir / "VAE" / "vae.gguf", {"decoder.conv_in.weight": [512, 16, 3, 3]})
        self.assertEqual(xte.classify_file(path).kind, "vae")

    def test_unreadable_and_missing(self):
        bad = self.dir / "broken.safetensors"
        bad.write_bytes(b"\x05\x00")
        self.assertEqual(xte.classify_file(bad).kind, "unreadable")
        self.assertEqual(xte.classify_file(self.dir / "missing.safetensors").kind, "unreadable")
        gguf = self.dir / "broken.gguf"
        gguf.write_bytes(b"GGUF")
        self.assertEqual(xte.classify_file(gguf).kind, "unreadable")
        self.assertEqual(xte.classify_file(self.dir / "missing.gguf").kind, "unreadable")

    def test_pickle_files_are_placed_by_folder(self):
        roots = [self.dir / "text_encoder"]
        te = self.dir / "text_encoder" / "sub" / "t5.pt"
        te.parent.mkdir(parents=True)
        te.write_bytes(b"PK")
        other = self.dir / "VAE" / "vae.pt"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"PK")
        self.assertEqual(xte.classify_file(te, te_roots=roots), Kind("opaque", detail=".pt", in_te_dir=True))
        self.assertEqual(xte.classify_file(other, te_roots=roots), Kind("opaque", detail=".pt", in_te_dir=False))
        self.assertEqual(xte.classify_file(self.dir / "x.bin", te_roots=roots).in_te_dir, False)

    def test_cache_follows_size_and_mtime(self):
        path = self.dir / "te.safetensors"
        fx.write_safetensors(path, fx.CASES["qwen3_06b"][0])
        self.assertEqual(xte.classify_file(path).slot, "qwen3_06b")
        # replaced by another file (other size)
        fx.write_safetensors(path, fx.CASES["vae_sd"][0])
        self.assertEqual(xte.classify_file(path).kind, "vae")
        # same size, other content: only the modification time tells
        fx.write_safetensors(path, {fx.QWEN3_LAYER: [2560], fx.Q_NORM: [128]}, fake_shapes=True)
        stat = os.stat(path)
        self.assertEqual(xte.classify_file(path).slot, "qwen3_4b")
        fx.write_safetensors(path, {fx.QWEN3_LAYER: [1024], fx.Q_NORM: [128]}, fake_shapes=True)
        self.assertEqual(os.stat(path).st_size, stat.st_size)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(xte.classify_file(path).slot, "qwen3_06b")

    def test_cache_hit_does_not_read_the_file(self):
        path = fx.write_safetensors(self.dir / "te.safetensors", fx.CASES["qwen3_06b"][0])
        gguf = fx.write_gguf(self.dir / "te.gguf", fx.GGUF_CASES["t5xxl_q8"][0])
        xte.classify_file(path)
        xte.classify_file(gguf)
        calls = []
        original_st, original_gguf = xte.read_safetensors_header, xte.read_gguf_shapes
        xte.read_safetensors_header = lambda p: calls.append(p) or original_st(p)
        xte.read_gguf_shapes = lambda p: calls.append(p) or original_gguf(p)
        try:
            xte.classify_file(path)
            xte.classify_file(gguf)
        finally:
            xte.read_safetensors_header, xte.read_gguf_shapes = original_st, original_gguf
        self.assertEqual(calls, [])


class GgufReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, shapes, **kwargs):
        return gguf_header.read_gguf_shapes(fx.write_gguf(self.dir / "f.gguf", shapes, **kwargs))

    def test_shapes_are_the_dimensions_reversed(self):
        self.assertEqual(self.read({"a": [32128, 4096], "b": [1024], "c": [2, 3, 4]}),
                         {"a": (32128, 4096), "b": (1024,), "c": (2, 3, 4)})

    def test_orig_shape_wins(self):
        got = self.read({"a": [4, 8], "b": [2]}, orig_shapes={"a": [32128, 1]})
        self.assertEqual(got, {"a": (32128, 1), "b": (2,)})

    def test_other_keys_are_skipped(self):
        kv = [("general.name", fx.GGUF_STRING, "t5"), ("general.alignment", fx.GGUF_UINT32, 32),
              ("tokenizer.tokens", fx.GGUF_ARRAY, (fx.GGUF_STRING, ["<pad>", "</s>", "▁the"] * 50)),
              ("scores", fx.GGUF_ARRAY, (fx.GGUF_FLOAT32, [0.5] * 40)),
              ("nested", fx.GGUF_ARRAY, (fx.GGUF_ARRAY, [(fx.GGUF_UINT32, [1, 2]), (fx.GGUF_UINT32, [3])])),
              # not INT32: not a shape (Forge's get_orig_shape wants ARRAY of INT32)
              ("comfy.gguf.orig_shape.b", fx.GGUF_ARRAY, (fx.GGUF_UINT64, [7, 7])),
              ("comfy.gguf.orig_shape.c", fx.GGUF_ARRAY, (fx.GGUF_INT32, []))]
        self.assertEqual(self.read({"a": [3], "b": [5], "c": [6]}, kv=kv), {"a": (3,), "b": (5,), "c": (6,)})

    def test_a_duplicated_key_keeps_its_first_value(self):
        kv = [("comfy.gguf.orig_shape.a", fx.GGUF_ARRAY, (fx.GGUF_INT32, [10]))]
        got = self.read({"a": [3]}, kv=kv, orig_shapes={"a": [20]})
        self.assertEqual(got, {"a": (10,)})

    def test_version_two_and_data(self):
        self.assertEqual(self.read({"a": [2, 2]}, version=2, with_data=True), {"a": (2, 2)})

    def test_what_forge_cannot_open(self):
        cases = {
            "bad magic": dict(magic=b"GGUG"),
            "version 1": dict(version=1),
            "version 4": dict(version=4),
            "big-endian": dict(order=">"),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(self.read({"a": [2]}, **kwargs))
        with self.subTest(case="not utf-8"):
            self.assertIsNone(self.read({"a": [2]}, raw_names={"a": b"\xff\xfe"}))

    def test_duplicated_tensor(self):
        path = fx.write_gguf(self.dir / "f.gguf", {"a": [2], "b": [3]}, raw_names={"b": b"a"})
        self.assertIsNone(gguf_header.read_gguf_shapes(path))

    def test_truncated(self):
        path = fx.write_gguf(self.dir / "f.gguf", {"a": [2], "b": [3]})
        data = path.read_bytes()
        for cut in (3, 10, 30, len(data) - 1):
            with self.subTest(cut=cut):
                path.write_bytes(data[:cut])
                self.assertIsNone(gguf_header.read_gguf_shapes(path))

    def test_absurd_counts(self):
        path = self.dir / "f.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1 << 60, 0))
        self.assertIsNone(gguf_header.read_gguf_shapes(path))
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1 << 60))
        self.assertIsNone(gguf_header.read_gguf_shapes(path))

    def test_missing_file(self):
        self.assertIsNone(gguf_header.read_gguf_shapes(self.dir / "nope.gguf"))


# ---------------------------------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------------------------------

class CheckpointTests(unittest.TestCase):
    def test_every_case(self):
        for name, (shapes, model_type, slots) in fx.CHECKPOINTS.items():
            with self.subTest(case=name):
                self.assertEqual(xte.checkpoint_profile(name, fx.header(shapes)), Checkpoint(name, model_type,
                                                                                            frozenset(slots)))

    def test_loads(self):
        expected = {"-": (False, False), "sd1": (True, False), "xlrf": (False, True), "sdxl": (True, True),
                    "flux": (True, False)}
        for model_type, (clip_l, clip_g) in expected.items():
            checkpoint = Checkpoint("c", model_type)
            self.assertEqual((checkpoint.loads("clip_l"), checkpoint.loads("clip_g")), (clip_l, clip_g), model_type)
            self.assertTrue(checkpoint.loads("t5xxl"))
            self.assertTrue(checkpoint.loads("qwen3_06b"))

    def test_metadata_and_unreadable(self):
        self.assertIsNone(xte.checkpoint_profile("x", None))
        self.assertEqual(xte.checkpoint_profile("x", fx.header({}, {"a": "b"})), Checkpoint("x"))

    def test_files(self):
        with tempfile.TemporaryDirectory() as folder:
            xte.clear_cache()
            path = fx.write_checkpoint(folder, "sdxl")
            self.assertEqual(xte.checkpoint_file(path), Checkpoint("sdxl.safetensors", "sdxl",
                                                                  frozenset({"clip_l", "clip_g"})))
            self.assertEqual(xte.checkpoint_file(path, "sdxl_base_1.0").name, "sdxl_base_1.0")
            gguf = fx.write_gguf(Path(folder) / "flux1-dev-Q8_0.gguf", fx.GGUF_CASES["flux_unet_q8"][0])
            self.assertEqual(xte.checkpoint_file(gguf), Checkpoint("flux1-dev-Q8_0.gguf", "flux"))
            ckpt = Path(folder) / "v1-5.ckpt"
            ckpt.write_bytes(b"PK")
            self.assertIsNone(xte.checkpoint_file(ckpt))
            self.assertIsNone(xte.checkpoint_file(Path(folder) / "missing.safetensors"))
            self.assertIsNone(xte.checkpoint_file(None))
            broken = Path(folder) / "broken.safetensors"
            broken.write_bytes(b"\x01")
            self.assertIsNone(xte.checkpoint_file(broken))
            xte.clear_cache()


# ---------------------------------------------------------------------------------------------------------------------
# A module list like Forge's: file name -> full path, and the kinds of the files
# ---------------------------------------------------------------------------------------------------------------------

TE = r"C:\forge\models\text_encoder"
VAE = r"C:\forge\models\VAE"
FILES = {
    "qwen_image_vae.safetensors": (VAE, Kind("vae")),
    "qwen_image_vae_HDR_fp32.safetensors": (VAE, Kind("vae")),
    "qwen_3_06b_base.safetensors": (TE, Kind("te", ("qwen3_06b",))),
    "qwen_3_06b_base_fp16_uncensored.safetensors": (TE, Kind("te", ("qwen3_06b",))),
    "qwen_3_06b_base_fp32_uncensored.safetensors": (TE, Kind("te", ("qwen3_06b",))),
    "qwen_3_06b_base_full_fp32.safetensors": (TE, Kind("te", ("qwen3_06b",))),
    "krea2UncensoredLLMCLIP_v10.safetensors": (TE, Kind("te", ("qwen3vl_4b",))),
    "qwen3-vl-4b-heretic.safetensors": (TE, Kind("te", ("qwen3vl_4b",))),
    "qwen3vl_4b_bf16.safetensors": (TE, Kind("te", ("qwen3vl_4b",))),
    "qwen3-vl-8b-heretic-1.3.0-int8convrot.safetensors": (TE, Kind("te", ("qwen3vl_8b",))),
    "qwen35_4b.safetensors": (TE, Kind("qwen35")),
    "Anima-3.8B-expanded_adapter.safetensors": (TE, Kind("adapter")),
    "h3_qwen3vl_8b_tap24.safetensors": (TE, Kind("unknown")),
    "qwen3vl_32b_two_slots.safetensors": (TE, Kind("multi", ("qwen25_7b", "qwen3_8b"))),
    # Flux / SDXL-style files (not on the development PC)
    "ae.safetensors": (VAE, Kind("vae")),
    "clip_l.safetensors": (TE + r"\flux", Kind("te", ("clip_l",))),
    "clip_l_other.safetensors": (TE, Kind("te", ("clip_l",))),
    "clip_g.safetensors": (TE, Kind("te", ("clip_g",))),
    "t5xxl_fp16.safetensors": (TE + r"\flux", Kind("te", ("t5xxl",))),
    "t5xxl_fp8.safetensors": (TE, Kind("te", ("t5xxl",))),
    "flan-t5-xxl-fp16.safetensors": (TE, Kind("te", ("t5xxl",))),
    "clip_lg_combined.safetensors": (TE, Kind("multi", ("clip_g", "clip_l"))),
    # GGUF encoders are read by their tensor names (FileTests): a slot like any safetensors encoder
    "t5-v1_1-xxl-encoder-Q8_0.gguf": (TE, Kind("te", ("t5xxl",))),
    "t5-v1_1-xxl-encoder-Q4_K_M.gguf": (TE, Kind("te", ("t5xxl",))),
    "Qwen3-0.6B-Q8_0.gguf": (TE, Kind("te", ("qwen3_06b",))),
    # pickle files are not read
    "t5-old.pt": (TE, Kind("opaque", detail=".pt", in_te_dir=True)),
    "t5-older.bin": (TE, Kind("opaque", detail=".bin", in_te_dir=True)),
    "vae-old.pt": (VAE, Kind("opaque", detail=".pt")),
    "a+b.safetensors": (TE, Kind("te", ("qwen3_06b",))),
}
MODULE_LIST = {name: f"{folder}\\{name}" for name, (folder, _) in FILES.items()}
MODULE_MAP = {name: ("flux\\" + name if folder.endswith("flux") else name) for name, (folder, _) in FILES.items()}
KINDS = {MODULE_LIST[name]: kind for name, (_, kind) in FILES.items()}


def classify(path):
    return KINDS[path]


def paths(*names):
    return sorted(MODULE_LIST[name] for name in names)


ANIMA = paths("qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors")
KREA = paths("qwen_image_vae.safetensors", "krea2UncensoredLLMCLIP_v10.safetensors")
FLUX = paths("ae.safetensors", "clip_l.safetensors", "t5xxl_fp16.safetensors")
FLUX_GGUF = paths("ae.safetensors", "clip_l.safetensors", "t5-v1_1-xxl-encoder-Q8_0.gguf")

ANIMA_CKPT = Checkpoint("anima.safetensors", "-")
FLUX_UNET = Checkpoint("flux1-dev-fp8.safetensors", "flux")
FLUX_AIO = Checkpoint("flux1-dev.safetensors", "flux", frozenset({"clip_l", "t5xxl"}))
SDXL = Checkpoint("sdxl.safetensors", "sdxl", frozenset({"clip_l", "clip_g"}))
SD1 = Checkpoint("v1-5.safetensors", "sd1", frozenset({"clip_l"}))


def plan(value, base, checkpoint=None):
    return xte.plan(value, base=base, module_list=MODULE_LIST, classify=classify, checkpoint=checkpoint)


class ValueTests(unittest.TestCase):
    def test_current(self):
        for value in ("Use Forge current", "use forge current", " current ", "Current", "None", "none", " NONE "):
            with self.subTest(value=value):
                self.assertEqual(xte.parse_value(value, MODULE_LIST), ())
                self.assertIsNone(plan(value, ANIMA))

    def test_words_for_the_selection_point_to_current(self):
        for value in ("Default", "Automatic", "Built-in", "Use same choices"):
            with self.subTest(value=value):
                with self.assertRaises(xte.AxisValueError) as caught:
                    xte.parse_value(value, MODULE_LIST)
                self.assertIn('use "Use Forge current"', str(caught.exception))
                self.assertNotIn("put it in models/text_encoder", str(caught.exception))

    def test_resolve_forms(self):
        name = "qwen_3_06b_base_full_fp32.safetensors"
        for value in (name, f"  {name} ", f'"{name}"', name.upper(), "qwen_3_06b_base_full_fp32",
                      "QWEN_3_06B_BASE_FULL_FP32", MODULE_LIST[name], f"sub/{name}"):
            with self.subTest(value=value):
                self.assertEqual(xte.resolve(value, MODULE_LIST), name)
        self.assertEqual(xte.resolve(r"flux\clip_l.safetensors", MODULE_LIST), "clip_l.safetensors")
        self.assertEqual(xte.resolve("flux/clip_l.safetensors", MODULE_LIST), "clip_l.safetensors")
        # a file really named like one of the words is still that file
        self.assertEqual(xte.resolve("default", {"default.safetensors": "x"}), "default.safetensors")

    def test_resolve_errors(self):
        with self.assertRaisesRegex(xte.AxisValueError, 'unknown file "nope.safetensors".*"Use Forge current" keeps'):
            xte.resolve("nope.safetensors", MODULE_LIST)
        with self.assertRaisesRegex(xte.AxisValueError, "empty value"):
            xte.resolve("  ", MODULE_LIST)
        listing = {"te.safetensors": "a", "te.gguf": "b"}
        with self.assertRaisesRegex(xte.AxisValueError, "matches several files"):
            xte.resolve("te", listing)
        self.assertTrue(issubclass(xte.AxisValueError, ValueError))

    def test_plus(self):
        self.assertEqual(xte.parse_value("clip_l_other + t5xxl_fp8", MODULE_LIST),
                         ("clip_l_other.safetensors", "t5xxl_fp8.safetensors"))
        self.assertEqual(xte.parse_value("clip_l_other.safetensors+t5xxl_fp8.safetensors", MODULE_LIST),
                         ("clip_l_other.safetensors", "t5xxl_fp8.safetensors"))
        # a file name with "+" is a file name first
        self.assertEqual(xte.parse_value("a+b.safetensors", MODULE_LIST), ("a+b.safetensors",))
        with self.assertRaisesRegex(xte.AxisValueError, "a file on each side"):
            xte.parse_value("clip_l_other + ", MODULE_LIST)
        with self.assertRaisesRegex(xte.AxisValueError, "same file twice"):
            xte.parse_value("clip_l_other + clip_l_other.safetensors", MODULE_LIST)
        with self.assertRaisesRegex(xte.AxisValueError, 'unknown file "nope"'):
            xte.parse_value("clip_l_other + nope", MODULE_LIST)


class ComposeTests(unittest.TestCase):
    def test_anima_replace_keeps_the_vae(self):
        self.assertEqual(plan("qwen_3_06b_base_full_fp32", ANIMA),
                         paths("qwen_image_vae.safetensors", "qwen_3_06b_base_full_fp32.safetensors"))
        self.assertEqual(plan("qwen_3_06b_base_full_fp32", ANIMA, ANIMA_CKPT),
                         paths("qwen_image_vae.safetensors", "qwen_3_06b_base_full_fp32.safetensors"))

    def test_same_value_is_the_selection(self):
        self.assertEqual(plan("qwen_3_06b_base", ANIMA), ANIMA)

    def test_krea(self):
        self.assertEqual(plan("qwen3vl_4b_bf16", KREA),
                         paths("qwen_image_vae.safetensors", "qwen3vl_4b_bf16.safetensors"))

    def test_flux_t5_only(self):
        self.assertEqual(plan("t5xxl_fp8", FLUX), paths("ae.safetensors", "clip_l.safetensors", "t5xxl_fp8.safetensors"))

    def test_flux_clip_and_t5(self):
        self.assertEqual(plan("clip_l_other + t5xxl_fp8", FLUX),
                         paths("ae.safetensors", "clip_l_other.safetensors", "t5xxl_fp8.safetensors"))

    def test_duplicate_slot(self):
        with self.assertRaisesRegex(xte.AxisValueError, 'both fill the T5-XXL slot'):
            plan("t5xxl_fp16 + t5xxl_fp8", FLUX)

    def test_incompatible(self):
        with self.assertRaises(xte.AxisValueError) as caught:
            plan("qwen3vl_4b_bf16", ANIMA)
        message = str(caught.exception)
        self.assertIn('"qwen3vl_4b_bf16.safetensors" is a Qwen3-VL 4B encoder but the current VAE / Text Encoder '
                      'selection has none (it has: Qwen3 0.6B) and the axis cannot read the checkpoint', message)
        self.assertNotIn("not use it", message)
        with self.assertRaises(xte.AxisValueError) as caught:
            plan("qwen3vl_4b_bf16", ANIMA, ANIMA_CKPT)
        self.assertIn('and the checkpoint "anima.safetensors" has none built in (it has: no encoder) - the axis only '
                      'swaps an encoder the model already has', str(caught.exception))
        # Flux UNet-only with a T5 but no CLIP-L: whether the model reads a CLIP-L is not guessed
        with self.assertRaisesRegex(xte.AxisValueError, r"CLIP-L encoder but .* \(it has: T5-XXL\)"):
            plan("clip_l_other", paths("ae.safetensors", "t5xxl_fp16.safetensors"), FLUX_UNET)

    def test_checkpoint_built_in_encoder_is_replaced(self):
        # Flux all-in-one with only a T5 in the list: a CLIP-L pick replaces the checkpoint's own CLIP-L
        self.assertEqual(plan("clip_l_other", paths("ae.safetensors", "t5xxl_fp16.safetensors"), FLUX_AIO),
                         paths("ae.safetensors", "clip_l_other.safetensors", "t5xxl_fp16.safetensors"))
        # SDXL with only a CLIP-L override: a CLIP-G pick replaces the built-in CLIP-G
        self.assertEqual(plan("clip_g", paths("clip_l_other.safetensors"), SDXL),
                         paths("clip_g.safetensors", "clip_l_other.safetensors"))
        # SDXL with its built-in encoders only
        self.assertEqual(plan("clip_l_other", [], SDXL), paths("clip_l_other.safetensors"))
        self.assertEqual(plan("clip_l_other + clip_g", [], SDXL), paths("clip_g.safetensors", "clip_l_other.safetensors"))

    def test_slots_the_checkpoint_does_not_have(self):
        # SDXL / SD1 with the encoders built in and nothing in the list: T5 / Qwen would be loaded and never read
        for value in ("t5xxl_fp8", "qwen_3_06b_base", "krea2UncensoredLLMCLIP_v10"):
            with self.subTest(value=value):
                with self.assertRaises(xte.AxisValueError) as caught:
                    plan(value, [], SDXL)
                self.assertIn('the checkpoint "sdxl.safetensors" has none built in (it has: CLIP-G, CLIP-L)',
                              str(caught.exception))
        with self.assertRaises(xte.AxisValueError) as caught:
            plan("t5xxl_fp8", [], SD1)
        self.assertIn('selection has no text encoder and the checkpoint "v1-5.safetensors" has none built in (it has: '
                      'CLIP-L)', str(caught.exception))

    def test_clip_for_a_family_without_clip(self):
        # replace_state_dict drops CLIP-G for SD1 and CLIP-L / CLIP-G for model type "-" (Anima, Krea2 …)
        with self.assertRaisesRegex(xte.AxisValueError, 'is a CLIP-G encoder, which Forge does not load for the '
                                                        'checkpoint "v1-5.safetensors"'):
            plan("clip_g", [], SD1)
        with self.assertRaisesRegex(xte.AxisValueError, "is a CLIP-L encoder, which Forge does not load"):
            plan("clip_l_other", ANIMA + paths("clip_l.safetensors"), ANIMA_CKPT)

    def test_append_when_the_model_has_no_encoder_the_axis_can_see(self):
        self.assertEqual(plan("qwen_3_06b_base", paths("qwen_image_vae.safetensors")), ANIMA)
        self.assertEqual(plan("qwen_3_06b_base", paths("qwen_image_vae.safetensors"), ANIMA_CKPT), ANIMA)
        self.assertEqual(plan("clip_l + t5xxl_fp8", paths("ae.safetensors"), FLUX_UNET),
                         paths("ae.safetensors", "clip_l.safetensors", "t5xxl_fp8.safetensors"))
        self.assertEqual(plan("qwen_3_06b_base", []), paths("qwen_3_06b_base.safetensors"))

    def test_every_entry_of_the_slot_is_replaced(self):
        base = paths("qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors", "qwen_3_06b_base_full_fp32.safetensors")
        self.assertEqual(plan("qwen_3_06b_base_fp16_uncensored", base),
                         paths("qwen_image_vae.safetensors", "qwen_3_06b_base_fp16_uncensored.safetensors"))

    def test_other_slots_and_other_files_stay(self):
        # both families' encoders in the list (Checkpoint x Text Encoder grid): only the picked slot changes
        base = paths("qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors", "qwen3vl_4b_bf16.safetensors",
                     "qwen35_4b.safetensors")
        self.assertEqual(plan("krea2UncensoredLLMCLIP_v10", base),
                         paths("qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors",
                               "krea2UncensoredLLMCLIP_v10.safetensors", "qwen35_4b.safetensors"))

    def test_base_is_normalised_like_modules_change(self):
        base = ["qwen_3_06b_base.safetensors", r"D:\elsewhere\qwen_image_vae.safetensors", r"C:\gone\deleted.safetensors",
                MODULE_LIST["qwen_3_06b_base.safetensors"]]
        self.assertEqual(xte.normalise(base, MODULE_LIST), ANIMA)
        self.assertEqual(plan("qwen_3_06b_base_full_fp32", base),
                         paths("qwen_image_vae.safetensors", "qwen_3_06b_base_full_fp32.safetensors"))

    def test_not_a_text_encoder(self):
        expected = {
            "qwen_image_vae_HDR_fp32": "is a VAE, not a text encoder - use the VAE axis",
            "qwen35_4b": "is the Qwen3.5 encoder of Anima 3.8B",
            "Anima-3.8B-expanded_adapter": "is an Anima 3.8B adapter, not a text encoder - choose it in the Anima 3.8B",
            "h3_qwen3vl_8b_tap24": "no text-encoder key Forge recognises",
            "qwen3vl_32b_two_slots": r"fills several slots: Qwen2.5-VL 7B, Qwen3 8B",
            "vae-old.pt": "a pickle file outside a text-encoder folder",
        }
        for value, message in expected.items():
            with self.subTest(value=value):
                with self.assertRaises(xte.AxisValueError) as caught:
                    plan(value, ANIMA)
                self.assertIn(message, str(caught.exception))

    def test_partial_combined_file(self):
        with self.assertRaisesRegex(xte.AxisValueError, "only part of"):
            plan("clip_l_other", paths("clip_lg_combined.safetensors"))

    def test_gguf_encoders_have_a_slot(self):
        # the common low-VRAM Flux setup: CLIP-L safetensors + T5 GGUF
        self.assertEqual(plan("t5xxl_fp8", FLUX_GGUF),
                         paths("ae.safetensors", "clip_l.safetensors", "t5xxl_fp8.safetensors"))
        self.assertEqual(plan("t5-v1_1-xxl-encoder-Q4_K_M", FLUX_GGUF),
                         paths("ae.safetensors", "clip_l.safetensors", "t5-v1_1-xxl-encoder-Q4_K_M.gguf"))
        self.assertEqual(plan("t5-v1_1-xxl-encoder-Q4_K_M", FLUX),
                         paths("ae.safetensors", "clip_l.safetensors", "t5-v1_1-xxl-encoder-Q4_K_M.gguf"))
        self.assertEqual(plan("clip_l_other + t5-v1_1-xxl-encoder-Q4_K_M.gguf", FLUX),
                         paths("ae.safetensors", "clip_l_other.safetensors", "t5-v1_1-xxl-encoder-Q4_K_M.gguf"))
        self.assertEqual(plan("Qwen3-0.6B-Q8_0", ANIMA), paths("qwen_image_vae.safetensors", "Qwen3-0.6B-Q8_0.gguf"))

    def test_a_gguf_of_the_slot_never_survives_next_to_the_pick(self):
        # a GGUF T5 and a safetensors T5 both in the list: the pick replaces both (Forge would load the GGUF last,
        # sorted, and its T5 branch would wipe the pick)
        base = paths("ae.safetensors", "clip_l.safetensors", "t5-v1_1-xxl-encoder-Q8_0.gguf", "t5xxl_fp16.safetensors")
        self.assertEqual(plan("flan-t5-xxl-fp16", base),
                         paths("ae.safetensors", "clip_l.safetensors", "flan-t5-xxl-fp16.safetensors"))

    def test_pickle(self):
        # a pickle file replaces the selection's one encoder ...
        self.assertEqual(plan("t5-old.pt", paths("ae.safetensors", "t5xxl_fp16.safetensors")),
                         paths("ae.safetensors", "t5-old.pt"))
        self.assertEqual(plan("t5-old.pt", paths("ae.safetensors", "t5-older.bin")), paths("ae.safetensors", "t5-old.pt"))
        # ... a safetensors encoder replaces a pickle file that is the selection's one encoder
        self.assertEqual(plan("t5xxl_fp8", paths("ae.safetensors", "t5-old.pt")),
                         paths("ae.safetensors", "t5xxl_fp8.safetensors"))
        # ... but not with two encoders, none, or in a pair
        with self.assertRaisesRegex(xte.AxisValueError, "which has 2"):
            plan("t5-old.pt", FLUX)
        with self.assertRaisesRegex(xte.AxisValueError, "which has 0"):
            plan("t5-old.pt", paths("ae.safetensors"))
        with self.assertRaisesRegex(xte.AxisValueError, "only be used alone"):
            plan("clip_l_other + t5-old.pt", FLUX)
        # a pickle encoder next to a CLIP-L: the axis cannot tell whether it is the T5 - it says so
        with self.assertRaises(xte.AxisValueError) as caught:
            plan("t5xxl_fp8", paths("ae.safetensors", "clip_l.safetensors", "t5-old.pt"))
        message = str(caught.exception)
        self.assertIn('cannot tell whether "t5-old.pt" (a pickle file, not read without loading it) fills that slot',
                      message)
        self.assertNotIn("not use it", message)
        # a pickle VAE in the selection is not an encoder
        self.assertEqual(plan("qwen_3_06b_base", paths("vae-old.pt")), paths("vae-old.pt", "qwen_3_06b_base.safetensors"))

    def test_effective_base(self):
        self.assertEqual(xte.effective_base({}, ANIMA), ANIMA)
        self.assertEqual(xte.effective_base(None, ANIMA), ANIMA)
        self.assertEqual(xte.effective_base({"forge_additional_modules": ["a"]}, ANIMA), ["a"])
        self.assertEqual(xte.effective_base({"forge_additional_modules": "a"}, ANIMA), ANIMA)


class TwoAxesTests(unittest.TestCase):
    def slots(self, value):
        return xte.pick_slots(value, MODULE_LIST, classify)

    def test_pick_slots(self):
        self.assertEqual(self.slots("Use Forge current"), frozenset())
        self.assertEqual(self.slots("None"), frozenset())
        self.assertEqual(self.slots("clip_l_other"), {"clip_l"})
        self.assertEqual(self.slots("clip_l_other + t5xxl_fp8"), {"clip_l", "t5xxl"})
        self.assertEqual(self.slots("t5-old.pt"), {xte.UNKNOWN_SLOT})
        with self.assertRaises(xte.AxisValueError):
            self.slots("nope")

    def test_overlap(self):
        self.assertEqual(xte.overlap(frozenset({"clip_l"}), frozenset({"t5xxl"})), frozenset())
        self.assertEqual(xte.overlap(frozenset({"clip_l", "t5xxl"}), frozenset({"t5xxl"})), {"t5xxl"})
        self.assertTrue(xte.overlap(frozenset({xte.UNKNOWN_SLOT}), frozenset({"t5xxl"})))
        self.assertEqual(xte.overlap(frozenset({xte.UNKNOWN_SLOT}), frozenset()), frozenset())

    def test_second_axis_builds_on_the_first(self):
        first = plan("clip_l_other", FLUX)
        self.assertEqual(plan("t5xxl_fp8", first),
                         paths("ae.safetensors", "clip_l_other.safetensors", "t5xxl_fp8.safetensors"))

    def test_plan_values_is_the_axes_one_after_the_other(self):
        """``plan_values``: the values of a cell's Text Encoder axes, each on the list of the one before, as ``apply``
        and the hook build the cell's list ("Use Forge current" keeps the list)."""
        first = plan("clip_l_other", FLUX)
        added = [entry for entry in first if entry not in FLUX]
        second = xte.plan("t5xxl_fp8", base=first, module_list=MODULE_LIST, classify=classify, added=added)
        self.assertEqual(xte.plan_values(("clip_l_other", "t5xxl_fp8"), base=FLUX, module_list=MODULE_LIST,
                                         classify=classify), second)
        self.assertEqual(xte.plan_values(("clip_l_other", "Use Forge current"), base=FLUX, module_list=MODULE_LIST,
                                         classify=classify), first)
        self.assertIsNone(xte.plan_values(("Use Forge current",), base=FLUX, module_list=MODULE_LIST,
                                          classify=classify))
        self.assertIsNone(xte.plan_values((), base=FLUX, module_list=MODULE_LIST, classify=classify))
        # an empty list: each axis adds its encoder (what the first added is not the selection's)
        for checkpoint in (None, FLUX_UNET):
            self.assertEqual(xte.plan_values(("clip_l_other", "t5xxl_fp8"), base=paths("ae.safetensors"),
                                             module_list=MODULE_LIST, classify=classify, checkpoint=checkpoint),
                             paths("ae.safetensors", "clip_l_other.safetensors", "t5xxl_fp8.safetensors"))
        # the base is the selection the cell has (a VAE list another axis wrote, say): its VAE stays
        hdr = paths("qwen_image_vae_HDR_fp32.safetensors", "qwen_3_06b_base.safetensors")
        self.assertEqual(xte.plan_values(("qwen_3_06b_base_full_fp32",), base=hdr, module_list=MODULE_LIST,
                                         classify=classify),
                         paths("qwen_image_vae_HDR_fp32.safetensors", "qwen_3_06b_base_full_fp32.safetensors"))
        with self.assertRaisesRegex(xte.AxisValueError, "Forge does not load"):
            xte.plan_values(("clip_l_other",), base=FLUX, module_list=MODULE_LIST, classify=classify,
                            checkpoint=ANIMA_CKPT)

    def test_what_the_first_axis_added_is_not_the_selections(self):
        # nothing in the list (a checkpoint the axis cannot read, or Flux UNet-only): each axis adds its encoder, as its
        # confirm accepted against the selection
        for checkpoint in (None, FLUX_UNET):
            with self.subTest(checkpoint=checkpoint):
                first = plan("clip_l_other", paths("ae.safetensors"), checkpoint)
                added = [entry for entry in first if entry not in paths("ae.safetensors")]
                self.assertEqual(xte.plan("t5xxl_fp8", base=first, module_list=MODULE_LIST, classify=classify,
                                          checkpoint=checkpoint, added=added),
                                 paths("ae.safetensors", "clip_l_other.safetensors", "t5xxl_fp8.safetensors"))
                with self.assertRaisesRegex(xte.AxisValueError, "T5-XXL encoder but"):
                    plan("t5xxl_fp8", first, checkpoint)


class ChoicesTests(unittest.TestCase):
    def choices(self, current, checkpoint=None, **kwargs):
        return xte.choices(MODULE_LIST, MODULE_MAP, current, classify, checkpoint=checkpoint, **kwargs)

    def test_anima_selection_gives_the_qwen3_06b_files(self):
        expected = ["Use Forge current", "a+b.safetensors", "Qwen3-0.6B-Q8_0.gguf", "qwen_3_06b_base.safetensors",
                    "qwen_3_06b_base_fp16_uncensored.safetensors", "qwen_3_06b_base_fp32_uncensored.safetensors",
                    "qwen_3_06b_base_full_fp32.safetensors"]
        self.assertEqual(self.choices(ANIMA), expected)
        self.assertEqual(self.choices(ANIMA, ANIMA_CKPT), expected)

    def test_krea_selection(self):
        self.assertEqual(self.choices(KREA), [
            "Use Forge current", "krea2UncensoredLLMCLIP_v10.safetensors", "qwen3-vl-4b-heretic.safetensors",
            "qwen3vl_4b_bf16.safetensors"])

    def test_no_encoder_lists_every_single_slot_encoder(self):
        got = self.choices(paths("qwen_image_vae.safetensors"))
        expected = sorted((name for name, (_, kind) in FILES.items() if kind.kind == "te"), key=str.casefold)
        self.assertEqual(got, ["Use Forge current", *expected])
        for excluded in ("qwen35_4b.safetensors", "Anima-3.8B-expanded_adapter.safetensors", "t5-old.pt",
                         "qwen3vl_32b_two_slots.safetensors", "qwen_image_vae.safetensors"):
            self.assertNotIn(excluded, got)

    def test_no_encoder_and_a_family_without_clip(self):
        got = self.choices(paths("qwen_image_vae.safetensors"), ANIMA_CKPT)
        for name in ("clip_l.safetensors", "clip_l_other.safetensors", "clip_g.safetensors"):
            self.assertNotIn(name, got)
        self.assertIn("qwen_3_06b_base.safetensors", got)

    def test_built_in_encoders_count_as_the_models(self):
        # SDXL with its own CLIPs and nothing in the list: only CLIP-L / CLIP-G files
        self.assertEqual(self.choices([], SDXL), ["Use Forge current", "clip_g.safetensors", "clip_l.safetensors",
                                                  "clip_l_other.safetensors"])
        # SD1: CLIP-L only (Forge drops CLIP-G)
        self.assertEqual(self.choices([], SD1), ["Use Forge current", "clip_l.safetensors", "clip_l_other.safetensors"])
        # Flux all-in-one with a T5 override: CLIP-L files too
        got = self.choices(paths("ae.safetensors", "t5xxl_fp16.safetensors"), FLUX_AIO)
        self.assertIn("clip_l_other.safetensors", got)
        self.assertIn("t5xxl_fp8.safetensors", got)
        self.assertNotIn("clip_g.safetensors", got)
        self.assertNotIn("qwen_3_06b_base.safetensors", got)

    def test_relative_paths_when_the_dropdown_shows_them(self):
        got = self.choices(FLUX, use_short=False)
        self.assertEqual(got, ["Use Forge current", "clip_l_other.safetensors", "flan-t5-xxl-fp16.safetensors",
                               r"flux\clip_l.safetensors", r"flux\t5xxl_fp16.safetensors",
                               "t5-v1_1-xxl-encoder-Q4_K_M.gguf", "t5-v1_1-xxl-encoder-Q8_0.gguf",
                               "t5xxl_fp8.safetensors"])

    def test_gguf_is_listed_like_safetensors(self):
        got = self.choices(FLUX_GGUF)
        for name in ("t5-v1_1-xxl-encoder-Q8_0.gguf", "t5-v1_1-xxl-encoder-Q4_K_M.gguf", "t5xxl_fp16.safetensors",
                     "clip_l_other.safetensors"):
            self.assertIn(name, got)

    def test_pickle_is_never_listed(self):
        for current in (paths("ae.safetensors", "t5xxl_fp16.safetensors"), paths("ae.safetensors", "t5-old.pt"), FLUX,
                        paths("ae.safetensors")):
            got = self.choices(current)
            for name in ("t5-old.pt", "t5-older.bin", "vae-old.pt"):
                self.assertNotIn(name, got)
        # a selection whose one encoder is a pickle file has an encoder (of no known slot): nothing else is offered
        self.assertEqual(self.choices(paths("ae.safetensors", "t5-old.pt")), ["Use Forge current"])


class LabelTests(unittest.TestCase):
    def label(self, value, current=ANIMA):
        return xte.format_label(value, module_list=MODULE_LIST, current=current, classify=classify)

    def test_labels(self):
        self.assertEqual(self.label("qwen_3_06b_base_full_fp32.safetensors"), "Text Encoder: qwen_3_06b_base_full_fp32")
        self.assertEqual(self.label("clip_l_other + t5xxl_fp8", FLUX), "Text Encoder: clip_l_other + t5xxl_fp8")
        self.assertEqual(self.label("Use Forge current"), "Text Encoder: current (qwen_3_06b_base)")
        self.assertEqual(self.label("None"), "Text Encoder: current (qwen_3_06b_base)")
        self.assertEqual(self.label("Use Forge current", FLUX), "Text Encoder: current (clip_l + t5xxl_fp16)")
        self.assertEqual(self.label("Use Forge current", paths("ae.safetensors")), "Text Encoder: current")
        self.assertEqual(self.label("nope"), "Text Encoder: nope")


class SwapEntriesTests(unittest.TestCase):
    def test_cell_encoders_go_into_forges_list(self):
        base = ["vae", "te"]
        self.assertEqual(xte.swap_entries(["vae", "te"], base, ["vae", "te2"]), ["vae", "te2"])
        # the stock VAE axis rebuilt the list from the selection with another VAE last
        self.assertEqual(xte.swap_entries(["te", "vae_hdr"], base, ["vae", "te2"]), ["vae_hdr", "te2"])
        # an encoder added to a selection without one
        self.assertEqual(xte.swap_entries(["vae"], ["vae"], ["te", "vae"]), ["vae", "te"])
        self.assertEqual(xte.swap_entries([], [], ["te"]), ["te"])


class ConstantsTests(unittest.TestCase):
    def test_axis(self):
        self.assertEqual(xte.AXIS_LABEL, "[Modules] Text Encoder")
        self.assertEqual(xte.CURRENT, "Use Forge current")
        self.assertTrue(0.7 < xte.COST < 1.0, "between the stock VAE (0.7) and Checkpoint (1.0; 2.0 from Forge 2.31) "
                                              "axes")

    def test_slot_names_cover_every_slot(self):
        slots = set()
        for _shapes, _metadata, expected in fx.CASES.values():
            if expected != fx.RAISES:
                slots |= set(expected)
        for _shapes, expected in fx.GGUF_CASES.values():
            slots |= set(expected)
        self.assertLessEqual(slots, set(xte.SLOT_NAMES))


if __name__ == "__main__":
    unittest.main()
