"""XYZ ``[Modules] Text Encoder`` — the pure logic (sam3ext/xyz_text_encoder.py), without Forge.

* the slot classifier on headers of every replace_state_dict branch (Forge's own function on the same headers:
  tests/test_xyz_text_encoder_origin.py) and the special files (Qwen3.5, the Anima 3.8B adapter, VAEs …);
* real small ``.safetensors`` files: the header reader and its (path, size, mtime) cache;
* values (name, relative path, stem, ``a + b``), the cell's module list (``compose``), the dropdown choices, the
  cell labels and the cell's ``override_settings``.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import xyz_text_encoder as xte  # noqa: E402
from tests import _xyz_text_encoder_support as fx  # noqa: E402
from sam3ext.xyz_text_encoder import Kind  # noqa: E402


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

    def test_v2_connector_metadata_is_an_adapter(self):
        h = fx.header({"x": [1]}, {"architecture": "anima_qwen35_quality_anchored_semantic_connector_v2"})
        self.assertEqual(xte.classify_header("c.safetensors", h).kind, "adapter")

    def test_is_te_entry(self):
        self.assertTrue(Kind("te", ("qwen3_06b",)).is_te_entry)
        self.assertTrue(Kind("multi", ("clip_g", "clip_l")).is_te_entry)
        self.assertTrue(Kind("opaque", detail=".gguf", in_te_dir=True).is_te_entry)
        for kind in (Kind("opaque", detail=".gguf"), Kind("vae"), Kind("qwen35"), Kind("adapter"), Kind("unknown")):
            self.assertFalse(kind.is_te_entry, kind)


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

    def test_unreadable_and_missing(self):
        bad = self.dir / "broken.safetensors"
        bad.write_bytes(b"\x05\x00")
        self.assertEqual(xte.classify_file(bad).kind, "unreadable")
        self.assertEqual(xte.classify_file(self.dir / "missing.safetensors").kind, "unreadable")

    def test_opaque_files_are_placed_by_folder(self):
        roots = [self.dir / "text_encoder"]
        te = self.dir / "text_encoder" / "sub" / "t5-q8.gguf"
        te.parent.mkdir(parents=True)
        te.write_bytes(b"GGUF")
        other = self.dir / "VAE" / "vae.gguf"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"GGUF")
        self.assertEqual(xte.classify_file(te, te_roots=roots), Kind("opaque", detail=".gguf", in_te_dir=True))
        self.assertEqual(xte.classify_file(other, te_roots=roots), Kind("opaque", detail=".gguf", in_te_dir=False))
        self.assertEqual(xte.classify_file(self.dir / "x.pt", te_roots=roots).in_te_dir, False)

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
        xte.classify_file(path)
        calls = []
        original = xte.read_safetensors_header
        xte.read_safetensors_header = lambda p: calls.append(p) or original(p)
        try:
            xte.classify_file(path)
        finally:
            xte.read_safetensors_header = original
        self.assertEqual(calls, [])


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
    # Flux-style files (not on the development PC)
    "ae.safetensors": (VAE, Kind("vae")),
    "clip_l.safetensors": (TE + r"\flux", Kind("te", ("clip_l",))),
    "clip_l_other.safetensors": (TE, Kind("te", ("clip_l",))),
    "t5xxl_fp16.safetensors": (TE + r"\flux", Kind("te", ("t5xxl",))),
    "t5xxl_fp8.safetensors": (TE, Kind("te", ("t5xxl",))),
    "clip_lg_combined.safetensors": (TE, Kind("multi", ("clip_g", "clip_l"))),
    "t5-q8.gguf": (TE, Kind("opaque", detail=".gguf", in_te_dir=True)),
    "t5-q4.gguf": (TE, Kind("opaque", detail=".gguf", in_te_dir=True)),
    "vae-q8.gguf": (VAE, Kind("opaque", detail=".gguf")),
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


def plan(value, base):
    return xte.plan(value, base=base, module_list=MODULE_LIST, classify=classify)


class ValueTests(unittest.TestCase):
    def test_current(self):
        for value in ("Use Forge current", "use forge current", " current ", "Current"):
            self.assertEqual(xte.parse_value(value, MODULE_LIST), ())
            self.assertIsNone(plan(value, ANIMA))

    def test_resolve_forms(self):
        name = "qwen_3_06b_base_full_fp32.safetensors"
        for value in (name, f"  {name} ", f'"{name}"', name.upper(), "qwen_3_06b_base_full_fp32",
                      "QWEN_3_06B_BASE_FULL_FP32", MODULE_LIST[name], f"sub/{name}"):
            with self.subTest(value=value):
                self.assertEqual(xte.resolve(value, MODULE_LIST), name)
        self.assertEqual(xte.resolve(r"flux\clip_l.safetensors", MODULE_LIST), "clip_l.safetensors")
        self.assertEqual(xte.resolve("flux/clip_l.safetensors", MODULE_LIST), "clip_l.safetensors")

    def test_resolve_errors(self):
        with self.assertRaisesRegex(xte.AxisValueError, 'unknown file "nope.safetensors"'):
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

    def test_same_value_is_the_selection(self):
        self.assertEqual(plan("qwen_3_06b_base", ANIMA), ANIMA)
        self.assertIsNone(xte.cell_overrides({}, plan("qwen_3_06b_base", ANIMA), ANIMA))

    def test_krea(self):
        self.assertEqual(plan("qwen3vl_4b_bf16", KREA), paths("qwen_image_vae.safetensors", "qwen3vl_4b_bf16.safetensors"))

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
        self.assertIn('"qwen3vl_4b_bf16.safetensors" is a Qwen3-VL 4B encoder but the current VAE / Text Encoder '
                      'selection has none (it has: Qwen3 0.6B)', str(caught.exception))
        # Flux: a CLIP-L next to a T5 that the checkpoint's own CLIP-L may or may not go with — not guessed
        with self.assertRaisesRegex(xte.AxisValueError, r"CLIP-L encoder .* \(it has: T5-XXL\)"):
            plan("clip_l_other", paths("ae.safetensors", "t5xxl_fp16.safetensors"))

    def test_append_when_the_selection_has_no_encoder(self):
        self.assertEqual(plan("qwen_3_06b_base", paths("qwen_image_vae.safetensors")), ANIMA)
        self.assertEqual(plan("clip_l + t5xxl_fp8", paths("ae.safetensors")),
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
            "vae-q8.gguf": "outside a text-encoder folder",
        }
        for value, message in expected.items():
            with self.subTest(value=value):
                with self.assertRaises(xte.AxisValueError) as caught:
                    plan(value, ANIMA)
                self.assertIn(message, str(caught.exception))

    def test_partial_combined_file(self):
        with self.assertRaisesRegex(xte.AxisValueError, "only part of"):
            plan("clip_l_other", paths("clip_lg_combined.safetensors"))

    def test_gguf(self):
        gguf_base = paths("ae.safetensors", "clip_l.safetensors", "t5-q8.gguf")
        # GGUF replaces the selection's one encoder ...
        self.assertEqual(plan("t5-q4.gguf", paths("ae.safetensors", "t5xxl_fp16.safetensors")),
                         paths("ae.safetensors", "t5-q4.gguf"))
        self.assertEqual(plan("t5-q4.gguf", paths("ae.safetensors", "t5-q8.gguf")), paths("ae.safetensors", "t5-q4.gguf"))
        # ... a safetensors encoder replaces a GGUF that is the selection's one encoder
        self.assertEqual(plan("t5xxl_fp8", paths("ae.safetensors", "t5-q8.gguf")),
                         paths("ae.safetensors", "t5xxl_fp8.safetensors"))
        # ... but not with two encoders, none, or in a pair
        with self.assertRaisesRegex(xte.AxisValueError, "which has 2"):
            plan("t5-q4.gguf", FLUX)
        with self.assertRaisesRegex(xte.AxisValueError, "which has 0"):
            plan("t5-q4.gguf", paths("ae.safetensors"))
        with self.assertRaisesRegex(xte.AxisValueError, "only be used alone"):
            plan("clip_l_other + t5-q4.gguf", FLUX)
        with self.assertRaisesRegex(xte.AxisValueError, r"it has: CLIP-L, a GGUF encoder"):
            plan("t5xxl_fp8", gguf_base)
        # a GGUF VAE in the selection is not an encoder
        self.assertEqual(plan("qwen_3_06b_base", paths("vae-q8.gguf")), paths("vae-q8.gguf", "qwen_3_06b_base.safetensors"))

    def test_effective_base(self):
        self.assertEqual(xte.effective_base({}, ANIMA), ANIMA)
        self.assertEqual(xte.effective_base(None, ANIMA), ANIMA)
        self.assertEqual(xte.effective_base({"forge_additional_modules": ["a"]}, ANIMA), ["a"])
        self.assertEqual(xte.effective_base({"forge_additional_modules": "a"}, ANIMA), ANIMA)


class ChoicesTests(unittest.TestCase):
    def test_anima_selection_gives_the_four_qwen3_06b_files(self):
        self.assertEqual(xte.choices(MODULE_LIST, MODULE_MAP, ANIMA, classify), [
            "Use Forge current", "a+b.safetensors", "qwen_3_06b_base.safetensors",
            "qwen_3_06b_base_fp16_uncensored.safetensors", "qwen_3_06b_base_fp32_uncensored.safetensors",
            "qwen_3_06b_base_full_fp32.safetensors"])

    def test_krea_selection(self):
        self.assertEqual(xte.choices(MODULE_LIST, MODULE_MAP, KREA, classify), [
            "Use Forge current", "krea2UncensoredLLMCLIP_v10.safetensors", "qwen3-vl-4b-heretic.safetensors",
            "qwen3vl_4b_bf16.safetensors"])

    def test_no_encoder_lists_every_single_slot_encoder(self):
        got = xte.choices(MODULE_LIST, MODULE_MAP, paths("qwen_image_vae.safetensors"), classify)
        expected = sorted((name for name, (_, kind) in FILES.items() if kind.kind == "te"), key=str.casefold)
        self.assertEqual(got, ["Use Forge current", *expected])
        for excluded in ("qwen35_4b.safetensors", "Anima-3.8B-expanded_adapter.safetensors", "t5-q8.gguf",
                         "qwen3vl_32b_two_slots.safetensors", "qwen_image_vae.safetensors"):
            self.assertNotIn(excluded, got)

    def test_relative_paths_when_the_dropdown_shows_them(self):
        got = xte.choices(MODULE_LIST, MODULE_MAP, FLUX, classify, use_short=False)
        self.assertEqual(got, ["Use Forge current", "clip_l_other.safetensors", r"flux\clip_l.safetensors",
                               r"flux\t5xxl_fp16.safetensors", "t5xxl_fp8.safetensors"])

    def test_gguf_is_never_listed(self):
        # its slot is not read; it can still be typed in text mode (ComposeTests.test_gguf)
        for current in (paths("ae.safetensors", "t5xxl_fp16.safetensors"), paths("ae.safetensors", "t5-q8.gguf"), FLUX,
                        paths("ae.safetensors")):
            got = xte.choices(MODULE_LIST, MODULE_MAP, current, classify)
            for name in ("t5-q8.gguf", "t5-q4.gguf", "vae-q8.gguf"):
                self.assertNotIn(name, got)
        # a selection whose one encoder is a GGUF has an encoder (of no known slot): nothing else is offered
        self.assertEqual(xte.choices(MODULE_LIST, MODULE_MAP, paths("ae.safetensors", "t5-q8.gguf"), classify),
                         ["Use Forge current"])


class LabelTests(unittest.TestCase):
    def label(self, value, current=ANIMA):
        return xte.format_label(value, module_list=MODULE_LIST, current=current, classify=classify)

    def test_labels(self):
        self.assertEqual(self.label("qwen_3_06b_base_full_fp32.safetensors"), "Text Encoder: qwen_3_06b_base_full_fp32")
        self.assertEqual(self.label("clip_l_other + t5xxl_fp8", FLUX), "Text Encoder: clip_l_other + t5xxl_fp8")
        self.assertEqual(self.label("Use Forge current"), "Text Encoder: current (qwen_3_06b_base)")
        self.assertEqual(self.label("Use Forge current", FLUX), "Text Encoder: current (clip_l + t5xxl_fp16)")
        self.assertEqual(self.label("Use Forge current", paths("ae.safetensors")), "Text Encoder: current")
        self.assertEqual(self.label("nope"), "Text Encoder: nope")


class CellOverridesTests(unittest.TestCase):
    NEW = paths("qwen_image_vae.safetensors", "qwen_3_06b_base_full_fp32.safetensors")

    def test_new_dict_parent_untouched(self):
        parent = {"CLIP_stop_at_last_layers": 2}
        new = xte.cell_overrides(parent, self.NEW, ANIMA)
        self.assertIsNot(new, parent)
        self.assertEqual(parent, {"CLIP_stop_at_last_layers": 2})
        self.assertEqual(new, {"CLIP_stop_at_last_layers": 2, "forge_additional_modules": self.NEW})
        new["forge_additional_modules"].append("x")
        self.assertNotIn("x", self.NEW)

    def test_no_op_when_unchanged(self):
        self.assertIsNone(xte.cell_overrides({}, ANIMA, ANIMA, carry_checkpoint="other.safetensors"))

    def test_checkpoint(self):
        self.assertEqual(xte.cell_overrides({}, self.NEW, ANIMA, "b.safetensors")["sd_model_checkpoint"], "b.safetensors")
        self.assertNotIn("sd_model_checkpoint", xte.cell_overrides({}, self.NEW, ANIMA, None))
        kept = xte.cell_overrides({"sd_model_checkpoint": "api.safetensors"}, self.NEW, ANIMA, "b.safetensors")
        self.assertEqual(kept["sd_model_checkpoint"], "api.safetensors")

    def test_checkpoint_to_carry(self):
        a = types.SimpleNamespace(filename=r"C:\ckpt\a.safetensors", name="a.safetensors", title="a.safetensors [1]")
        a2 = types.SimpleNamespace(filename=r"C:\ckpt\a.safetensors", name="a.safetensors", title="a.safetensors [1]")
        b = types.SimpleNamespace(filename=r"C:\ckpt\b.safetensors", name="b.safetensors", title="b.safetensors [2]")
        self.assertIsNone(xte.checkpoint_to_carry({"checkpoint_info": a}, a2))   # same file, other object
        self.assertEqual(xte.checkpoint_to_carry({"checkpoint_info": b}, a), "b.safetensors")
        self.assertIsNone(xte.checkpoint_to_carry({}, a))
        self.assertIsNone(xte.checkpoint_to_carry(None, a))
        self.assertIsNone(xte.checkpoint_to_carry({"checkpoint_info": b}, None))


class ConstantsTests(unittest.TestCase):
    def test_axis(self):
        self.assertEqual(xte.AXIS_LABEL, "[Modules] Text Encoder")
        self.assertEqual(xte.CURRENT, "Use Forge current")
        self.assertTrue(0.7 < xte.COST < 1.0, "between the stock VAE (0.7) and Checkpoint (1.0) axes")

    def test_slot_names_cover_every_slot(self):
        slots = set()
        for _shapes, _metadata, expected in fx.CASES.values():
            if expected != fx.RAISES:
                slots |= set(expected)
        self.assertLessEqual(slots, set(xte.SLOT_NAMES))


if __name__ == "__main__":
    unittest.main()
