"""sam-extra MCP server: finding the Forge installation from the extension's own location.

Upstream needed FORGE_PATH_MAP even when Forge ran on the same PC, so the output folder, LoRA
sidecars and module files were invisible on the default setup. These tests cover
``sam_extra_mcp.forge_paths`` and the same-machine path mapping in ``forgeneo.config``.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from _mcp_support import make_forge_tree

from sam_extra_mcp import forge_paths
from sam_extra_mcp.forgeneo.config import Config, is_loopback_url
from sam_extra_mcp.forgeneo.generate import resolve_output_dir


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class DiscoveryTests(_Tmp):
    def test_everything_follows_from_the_extension_location(self):
        tree = make_forge_tree(self.base, settings={})
        paths = forge_paths.discover({}, module_file=tree["module_file"])
        forge = str(tree["forge"])
        self.assertEqual(paths.extension_root, str(tree["extension"]))
        self.assertEqual(paths.data_dir, forge)
        self.assertEqual(paths.forge_root, forge)
        self.assertEqual(paths.settings_file, os.path.join(forge, "config.json"))
        self.assertFalse(paths.settings_explicit)
        self.assertEqual(paths.models_dir, os.path.join(forge, "models"))
        self.assertEqual(paths.fallback_dir, os.path.join(forge, "sam-extra", "mcp", "outputs"))
        self.assertEqual(paths.cache_dir, os.path.join(forge, "sam-extra", "mcp"))
        self.assertEqual(paths.notes, ())

    def test_data_dir_without_program_files_is_reported(self):
        # Forge started with --data-dir: the extensions live in the data folder, webui.py elsewhere.
        tree = make_forge_tree(self.base, program=False)
        paths = forge_paths.discover({}, module_file=tree["module_file"])
        self.assertEqual(paths.data_dir, str(tree["forge"]))
        self.assertIsNone(paths.forge_root)
        self.assertTrue(any("--data-dir" in note for note in paths.notes))
        self.assertEqual(paths.settings_file, os.path.join(str(tree["forge"]), "config.json"))

    def test_program_folder_override(self):
        tree = make_forge_tree(self.base, program=False)
        program = self.base / "program"
        program.mkdir()
        paths = forge_paths.discover({forge_paths.ENV_FORGE_ROOT: str(program)}, module_file=tree["module_file"])
        self.assertEqual(paths.forge_root, str(program))

    def test_environment_overrides(self):
        tree = make_forge_tree(self.base)
        custom = self.base / "custom-settings.json"
        env = {
            forge_paths.ENV_SETTINGS_FILE: str(custom),
            forge_paths.ENV_MODELS_DIR: str(self.base / "models-elsewhere"),
            forge_paths.ENV_FALLBACK_DIR: str(self.base / "fallback"),
            forge_paths.ENV_CACHE_DIR: str(self.base / "cache"),
        }
        paths = forge_paths.discover(env, module_file=tree["module_file"])
        self.assertEqual(paths.settings_file, str(custom))
        self.assertTrue(paths.settings_explicit)
        self.assertIn(forge_paths.ENV_SETTINGS_FILE, paths.settings_source)
        self.assertEqual(paths.models_dir, str(self.base / "models-elsewhere"))
        self.assertEqual(paths.fallback_dir, str(self.base / "fallback"))
        self.assertEqual(paths.cache_dir, str(self.base / "cache"))

    def test_upstream_cache_variable_is_honoured(self):
        tree = make_forge_tree(self.base)
        paths = forge_paths.discover({"FORGENEO_CACHE_DIR": str(self.base / "upstream-cache")}, module_file=tree["module_file"])
        self.assertEqual(paths.cache_dir, str(self.base / "upstream-cache"))

    def test_outside_an_extensions_folder_nothing_is_guessed(self):
        project = self.base / "checkout" / "mcp_server"
        (project / "sam_extra_mcp").mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        paths = forge_paths.discover({}, module_file=str(project / "sam_extra_mcp" / "forge_paths.py"))
        self.assertEqual(paths.extension_root, str(self.base / "checkout"))
        self.assertIsNone(paths.data_dir)
        self.assertIsNone(paths.settings_file)
        self.assertEqual(paths.settings_source, "unknown")
        self.assertTrue(any(forge_paths.ENV_SETTINGS_FILE in note for note in paths.notes))
        # Images are still never lost: the fallback folder lives in the user's home then.
        self.assertTrue(paths.fallback_dir.endswith(os.path.join(".sam-extra-mcp", "outputs")))

    def test_not_a_server_layout(self):
        paths = forge_paths.discover({}, module_file=str(self.base / "somewhere" / "module.py"))
        self.assertIsNone(paths.extension_root)
        self.assertIsNone(paths.settings_file)
        self.assertTrue(paths.notes)

    def test_real_module_location_is_the_extension(self):
        # This checkout: <extension>/mcp_server/sam_extra_mcp/forge_paths.py.
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(forge_paths.extension_root_for(forge_paths.__file__), str(root))

    def test_as_dict_is_plain_data(self):
        tree = make_forge_tree(self.base)
        data = forge_paths.discover({}, module_file=tree["module_file"]).as_dict()
        self.assertEqual(set(data), {"extension_root", "data_dir", "forge_root", "settings_file", "settings_source",
                                     "models_dir", "fallback_dir", "cache_dir", "notes"})


class OutputFolderTests(_Tmp):
    def setUp(self):
        super().setUp()
        self.tree = make_forge_tree(self.base)
        self.paths = forge_paths.discover({}, module_file=self.tree["module_file"])

    def test_relative_outdir_resolves_against_the_program_folder(self):
        self.tree["output"].mkdir(parents=True)
        self.assertEqual(self.paths.resolve_output_dir("output\\txt2img-images"), str(self.tree["output"]))

    def test_folder_forge_has_not_created_yet_is_accepted(self):
        (self.tree["forge"] / "output").mkdir()
        expected = str(self.tree["forge"] / "output" / "img2img-images")
        self.assertEqual(self.paths.resolve_output_dir("output/img2img-images"), expected)

    def test_fresh_install_folder_is_accepted(self):
        # Even "output" appears only with Forge's first image (os.makedirs in images.save_image).
        self.assertFalse((self.tree["forge"] / "output").exists())
        self.assertEqual(self.paths.resolve_output_dir("output/txt2img-images"), str(self.tree["output"]))

    def test_unreachable_folder_is_none(self):
        self.assertIsNone(self.paths.resolve_output_dir(""))
        self.assertIsNone(self.paths.resolve_output_dir(str(self.base / "gone" / "deeper" / "still")))
        unknown = forge_paths.discover({}, module_file=str(self.base / "elsewhere" / "module.py"))
        self.assertIsNone(unknown.resolve_output_dir("output/txt2img-images"))

    def test_absolute_outdir_is_taken_as_is(self):
        target = self.base / "elsewhere"
        target.mkdir()
        self.assertEqual(self.paths.resolve_output_dir(str(target)), str(target))

    def test_generate_finds_the_folder_without_a_path_map(self):
        """The upstream defect: no FORGE_PATH_MAP -> no output folder -> ok:false after rendering."""
        self.tree["output"].mkdir(parents=True)
        config = Config.from_env({"FORGE_URL": "http://127.0.0.1:7860"})

        class _Client:
            pass

        client = _Client()
        client.config = config
        options = {"outdir_samples": "", "outdir_txt2img_samples": "output\\txt2img-images"}
        self.assertEqual(resolve_output_dir(client, options, paths=self.paths), str(self.tree["output"]))
        # Upstream's behaviour without the paths: nothing found.
        upstream = _Client()
        upstream.config = Config()
        self.assertIsNone(resolve_output_dir(upstream, options))

    def test_api_results_ignore_the_output_directory_override(self):
        """Forge's API saves to outdir_txt2img_samples even when outdir_samples is set
        (modules/api/api.py); the web UI honours the override (modules/txt2img.py)."""
        override = self.base / "ui-override"
        override.mkdir()
        self.tree["output"].mkdir(parents=True)

        class _Client:
            config = Config.from_env({"FORGE_URL": "http://127.0.0.1:7860"})

        options = {"outdir_samples": str(override), "outdir_txt2img_samples": "output/txt2img-images"}
        self.assertEqual(resolve_output_dir(_Client(), options, paths=self.paths, saved_by="ui"), str(override).replace("\\", "/"))
        self.assertEqual(resolve_output_dir(_Client(), options, paths=self.paths, saved_by="api"), str(self.tree["output"]))


class SameMachineMappingTests(unittest.TestCase):
    def test_loopback_detection(self):
        for url in ("http://127.0.0.1:7860", "http://localhost:7861", "http://[::1]:7860", "http://127.1.2.3"):
            self.assertTrue(is_loopback_url(url), url)
        for url in ("http://gpu-box:7860", "http://192.168.0.5:7860", "not a url"):
            self.assertFalse(is_loopback_url(url), url)

    def test_loopback_forge_needs_no_path_map(self):
        config = Config.from_env({"FORGE_URL": "http://127.0.0.1:7860"})
        self.assertTrue(config.same_machine)
        self.assertEqual(config.localise(r"C:\forge\models\VAE\a.safetensors"), "C:/forge/models/VAE/a.safetensors")
        self.assertEqual(config.localise("/opt/forge/models/a.safetensors"), "/opt/forge/models/a.safetensors")

    def test_relative_paths_are_never_guessed(self):
        config = Config.from_env({"FORGE_URL": "http://127.0.0.1:7860"})
        self.assertIsNone(config.localise(r"output\txt2img-images"))

    def test_path_map_still_wins(self):
        config = Config.from_env({"FORGE_URL": "http://127.0.0.1:7860", "FORGE_PATH_MAP": r"C:\forge=D:/mirror"})
        self.assertEqual(config.localise(r"C:\forge\models\a.safetensors"), "D:/mirror/models/a.safetensors")

    def test_remote_forge_keeps_upstream_behaviour(self):
        config = Config.from_env({"FORGE_URL": "http://gpu-box:7860"})
        self.assertFalse(config.same_machine)
        self.assertIsNone(config.localise(r"C:\forge\models\a.safetensors"))

    def test_explicit_override(self):
        self.assertTrue(Config.from_env({"FORGE_URL": "http://gpu-box:7860", "SAM_EXTRA_MCP_SAME_MACHINE": "1"}).same_machine)
        self.assertFalse(Config.from_env({"FORGE_URL": "http://127.0.0.1:7860", "SAM_EXTRA_MCP_SAME_MACHINE": "0"}).same_machine)

    def test_credentials_never_show_in_repr_or_description(self):
        config = Config.from_env({"FORGE_URL": "http://127.0.0.1:7860", "FORGE_AUTH": "alice:s3cret-pass"})
        self.assertEqual(config.auth, ("alice", "s3cret-pass"))
        self.assertNotIn("s3cret", repr(config))
        self.assertNotIn("s3cret", str(config.describe()))
        self.assertEqual(config.describe()["auth"], "configured")

    def test_loopback_means_localhost_or_a_loopback_literal(self):
        for url in ("http://localhost.:7860", "http://[::ffff:127.0.0.1]:7860", "http://127.255.255.254"):
            self.assertTrue(is_loopback_url(url), url)
        # Names that merely look local can resolve anywhere.
        for url in ("http://127.example.com:7860", "http://127.0.0.1.example.com:7860",
                    "http://localhost.example.com:7860", "http://0.0.0.0:7860"):
            self.assertFalse(is_loopback_url(url), url)
        self.assertFalse(Config.from_env({"FORGE_URL": "http://127.example.com:7860"}).same_machine)

    def test_credentials_in_the_url_are_moved_out_of_it(self):
        # httpx would send them as Basic auth; the URL itself is shown to the agent.
        config = Config.from_env({"FORGE_URL": "http://al%40ice:s3cret%3Apass@127.0.0.1:7860/"})
        self.assertEqual(config.url, "http://127.0.0.1:7860")
        self.assertEqual(config.auth, ("al@ice", "s3cret:pass"))
        self.assertTrue(config.same_machine)
        self.assertNotIn("s3cret", repr(config))
        self.assertNotIn("s3cret", str(config.describe()))
        explicit = Config.from_env({"FORGE_URL": "http://bob:other@127.0.0.1:7860", "FORGE_AUTH": "alice:first"})
        self.assertEqual(explicit.auth, ("alice", "first"), "FORGE_AUTH wins, as it does in httpx")
        self.assertEqual(explicit.url, "http://127.0.0.1:7860")
        # A Config built by hand still never reports what its URL carries.
        self.assertNotIn("s3cret", str(Config(url="http://u:s3cret@127.0.0.1:7860").describe()))

    def test_history_settings_from_environment(self):
        config = Config.from_env({"FORGE_HISTORY_MAX_AGE": "42", "FORGE_HISTORY_LIMIT": "50"})
        self.assertEqual(config.history_max_age, 42.0)
        self.assertEqual(config.history_limit, 50)
        self.assertEqual(Config.from_env({"FORGE_HISTORY_MAX_AGE": "nope"}).history_max_age, 300.0)


if __name__ == "__main__":
    unittest.main()
