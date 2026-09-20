"""IP-Adapter 가중치 경로·받기 — sam3ext/tipo/runtime.py 와 같은 규칙."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.anima_ipa import paths  # noqa: E402


class PathTests(unittest.TestCase):
    def test_everything_lives_under_models_anima_ipa(self):
        models = Path("C:/forge/models")
        self.assertEqual(paths.ipa_dir(models), models / "anima_ipa")
        self.assertEqual(
            paths.adapter_path(models),
            models / "anima_ipa" / "ip_adapter-Character_Reference-10.safetensors",
        )
        self.assertEqual(
            paths.encoder_dir(models),
            models / "anima_ipa" / "siglip2-base-patch16-512",
        )

    def test_revisions_are_pinned(self):
        self.assertEqual(
            paths.ADAPTER_REVISION, "99e9c351c9f00bdd188b5added0066a80d3b1de6"
        )
        self.assertEqual(
            paths.ENCODER_REVISION, "a89f5c5093f902bf39d3cd4d81d2c09867f0724b"
        )


class MissingFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = Path(self.tmp.name)

    def _write(self, path: Path, size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * size)

    def test_nothing_downloaded_means_everything_is_missing(self):
        missing = paths.missing_files(self.models)
        self.assertIn(paths.ADAPTER_FILE, missing)
        for name in paths.ENCODER_FILES:
            self.assertIn(name, missing)

    def test_a_truncated_file_counts_as_missing(self):
        """크기가 다르면 받다 만 파일이다 — 있다고 세면 로드가 이상하게 실패한다."""
        self._write(paths.adapter_path(self.models), paths.ADAPTER_BYTES - 1)
        self.assertIn(paths.ADAPTER_FILE, paths.missing_files(self.models))

    def test_complete_small_files_are_not_missing(self):
        self._write(
            paths.encoder_dir(self.models) / "config.json",
            paths.ENCODER_FILES["config.json"],
        )
        self.assertNotIn("config.json", paths.missing_files(self.models))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = Path(self.tmp.name)
        self.calls = []

    def _downloader(self, **kwargs):
        self.calls.append(kwargs)
        return "unused"

    def test_every_missing_file_is_requested_at_the_pinned_revision(self):
        paths.download(self.models, downloader=self._downloader)
        repos = {call["repo_id"] for call in self.calls}
        self.assertEqual(repos, {paths.ADAPTER_REPO, paths.ENCODER_REPO})
        for call in self.calls:
            expected = (
                paths.ADAPTER_REVISION
                if call["repo_id"] == paths.ADAPTER_REPO
                else paths.ENCODER_REVISION
            )
            self.assertEqual(call["revision"], expected)
        names = {call["filename"] for call in self.calls}
        self.assertEqual(names, {paths.ADAPTER_FILE, *paths.ENCODER_FILES})

    def test_the_encoder_goes_into_its_own_subfolder(self):
        paths.download(self.models, downloader=self._downloader)
        for call in self.calls:
            expected = (
                str(paths.ipa_dir(self.models))
                if call["repo_id"] == paths.ADAPTER_REPO
                else str(paths.encoder_dir(self.models))
            )
            self.assertEqual(call["local_dir"], expected)

    def test_files_that_are_already_complete_are_not_requested_again(self):
        target = paths.adapter_path(self.models)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\0" * paths.ADAPTER_BYTES)
        paths.download(self.models, downloader=self._downloader)
        self.assertNotIn(
            paths.ADAPTER_FILE, {call["filename"] for call in self.calls}
        )


if __name__ == "__main__":
    unittest.main()
