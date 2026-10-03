"""Colorcraft basis vectors shipped in sam3ext/colorcraft/data — bytes, keys, shapes and invariants.

colorcraft-krea2 / colorcraft-zimage are muerrilla/ComfyUI-Colorcraft@d28ac6a's ``vectors/`` files
(git blobs 1c56e073… / f2bd1c27…, identical in the fork); colorcraft-flux2 is the fork's
(aoleg/ComfyUI-Colorcraft@f00066c, blob 8afcf2e7…). Both repositories are MIT.
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.colorcraft import basis, hook  # noqa: E402

DATA = ROOT / "sam3ext" / "colorcraft" / "data"

# (size in bytes, SHA-256, git blob id at the pinned commit, channels)
FILES = {
    "krea2": (1424, "0612ace7de9857927fc843f008a64633a097ab260b2c9ec3dc76b64c86b46cf2",
              "1c56e07329898e4f334025dcdfc08a77b12f4917", 16),
    "zimage": (1424, "824313efd6046492927087b3745cdc7e83ef8cd5dd5858fe42ec86c9b693c3f4",
               "f2bd1c27f73ea1e4bee18d4cc542de3d718f2609", 16),
    "flux2": (6384, "303e8a77d8fcba89162cbcc7b90a2e1094af5ff923b76be81b72a27c411bfacc",
              "8afcf2e778e620391e23022e9c5c1d0eca203a55", 128),
}
KEYS = {"exposure", "temperature", "tint", "temp+tint", "temp-tint", "lab-a", "lab-b", "lab-a+b", "lab-a-b",
        "clarity", "sharpness"}
DIAGONALS = {"temp+tint": ("temperature", "tint", 1.0), "temp-tint": ("temperature", "tint", -1.0),
             "lab-a+b": ("lab-a", "lab-b", 1.0), "lab-a-b": ("lab-a", "lab-b", -1.0)}


class VectorFileTests(unittest.TestCase):
    def test_exactly_the_three_families_ship(self):
        self.assertEqual(sorted(p.name for p in DATA.iterdir()),
                         sorted(f"colorcraft-{family}.safetensors" for family in FILES))
        self.assertEqual(basis.BASIS_FAMILIES, ["krea2", "zimage", "flux2"])
        self.assertEqual(Path(basis.VECTORS_DIR), DATA)

    def test_bytes_are_the_pinned_files(self):
        for family, (size, digest, blob, _) in FILES.items():
            data = (DATA / f"colorcraft-{family}.safetensors").read_bytes()
            with self.subTest(family=family):
                self.assertEqual(len(data), size)
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
                self.assertEqual(hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), blob)

    def test_keys_shapes_dtypes(self):
        for family, (_, _, _, channels) in FILES.items():
            path = DATA / f"colorcraft-{family}.safetensors"
            with safe_open(str(path), "pt") as handle:
                self.assertIsNone(handle.metadata())
                keys = set(handle.keys())
            loaded = basis.load_basis(family, basis.VECTORS_DIR)
            with self.subTest(family=family):
                self.assertEqual(keys, KEYS)
                self.assertEqual(set(loaded), KEYS)
                for name, vector in loaded.items():
                    self.assertEqual(tuple(vector.shape), (channels,), name)
                    self.assertEqual(vector.dtype, torch.float32, name)
                    self.assertTrue(bool(torch.isfinite(vector).all()), name)

    def test_norms(self):
        """krea2 is unit-norm; zimage carries its axes' strength in the norm; flux2 was matched by effect."""
        krea2 = basis.load_basis("krea2", basis.VECTORS_DIR)
        for vector in krea2.values():
            self.assertAlmostEqual(float(vector.norm()), 1.0, places=5)
        zimage = basis.load_basis("zimage", basis.VECTORS_DIR)
        expected = {"exposure": 2.5, "clarity": 1.25, "sharpness": 1.25}
        for name, vector in zimage.items():
            self.assertAlmostEqual(float(vector.norm()), expected.get(name, 2.0), places=5, msg=name)
        flux2 = basis.load_basis("flux2", basis.VECTORS_DIR)
        for name, vector in flux2.items():
            self.assertTrue(1.5 < float(vector.norm()) < 2.2, name)

    def test_diagonal_axes_are_their_parents_sum_and_difference(self):
        """The four diagonals point along the normalised sum/difference of their parents (cosine 1)."""
        for family in FILES:
            loaded = basis.load_basis(family, basis.VECTORS_DIR)
            for name, (a, b, sign) in DIAGONALS.items():
                direction = loaded[a] / loaded[a].norm() + sign * loaded[b] / loaded[b].norm()
                cosine = torch.nn.functional.cosine_similarity(direction, loaded[name], dim=0)
                with self.subTest(family=family, axis=name):
                    self.assertGreater(float(cosine), 0.9999)

    def test_flux2_directions_are_replicated_over_the_packed_subpixels(self):
        """Flux2's latent is a 2x2 packing (channel i = channel i // 4, slot i % 4); the fork projects every
        direction onto the replicated subspace so it cannot stamp a 2x2 tile into the image."""
        for name, vector in basis.load_basis("flux2", basis.VECTORS_DIR).items():
            groups = vector.view(-1, 4)
            with self.subTest(axis=name):
                self.assertTrue(bool((groups == groups[:, :1]).all()))

    def test_hook_cache_holds_copies_not_the_file_mapping(self):
        """The hook keeps cloned tensors so a running WebUI does not pin the files (Windows: git pull)."""
        hook._BASIS_CACHE.clear()
        cached = hook.load_family_basis("krea2")
        fresh = basis.load_basis("krea2", basis.VECTORS_DIR)
        for name in KEYS:
            self.assertTrue(torch.equal(cached[name], fresh[name]))
            self.assertNotEqual(cached[name].data_ptr(), fresh[name].data_ptr())
        self.assertIsNone(hook.load_family_basis("no-such-family"))

    def test_missing_file_reads_as_none(self):
        self.assertIsNone(basis.load_basis("krea2", str(ROOT / "tests")))

    def test_gitignore_keeps_the_vectors(self):
        """``*.safetensors`` is ignored repository-wide; the negation re-includes these three files."""
        lines = [line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
        self.assertIn("*.safetensors", lines)
        self.assertIn("!sam3ext/colorcraft/data/*.safetensors", lines)
        self.assertGreater(lines.index("!sam3ext/colorcraft/data/*.safetensors"), lines.index("*.safetensors"))


if __name__ == "__main__":
    unittest.main()
