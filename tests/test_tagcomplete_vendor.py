"""Built-in Tag Autocomplete (sam3ext/tagcomplete — sd-webui-tagcomplete-neo 8e0d81f, MIT): what was copied is
upstream's, what was changed is listed, and the MIT terms travel with it.

* The 15 JS files, LICENSE and the five shipped tag lists are byte-identical to upstream 8e0d81f (SHA-256 pinned —
  ``.gitattributes`` keeps them out of line-ending conversion).
* In the four Python files, every upstream function sam-extra did not change is character-identical to upstream
  (``ast.get_source_segment`` SHA-256, the same check as tests/test_negpip_vendor.py). Changed: ``api_tac`` (only its
  ``get_wildcard_contents`` route; its other 19 inner definitions are pinned too); new: ``_is_listed_wildcard_file``,
  ``_api_tac_once``; the module-level imports (``scripts.*`` -> ``sam3ext.tagcomplete.*``) and ``TAGS_PATH``.
* Each vendored Python file carries the upstream, copyright, MIT and modification notice; the own files say they are
  sam-extra's; THIRD_PARTY_NOTICES.md records the vendoring with both authors and the fork commit.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import _tagcomplete_support as tac  # noqa: E402

PACKAGE = ROOT / "sam3ext" / "tagcomplete"

JS_SHA256 = {
    "__globals.js": "3ad1881baee19d916dfc93d1b1ebd4712e70d29bf8041a74f30adb41c6f66746",
    "_baseParser.js": "68cf54bac5bfcf31402b0bad181aced76182302f824c306d352bf195a8f3e2fd",
    "_caretPosition.js": "5f544d1f5c2933ec0e500ed8568391f4ee44e86d40e8e02e8f1c307b436d6e14",
    "_result.js": "572eca5cac3128cccebcba016fbeca990d2d4f88ce15d557739ba471bad10c87",
    "_textAreas.js": "0fbf27fd495871dcb70f63fbdb1f10fc16929d4a84e8aba183f90e85bddfdbc7",
    "_utils.js": "34ff779b018610cf8ef16fdb572c771f4994b6c6ea84987fad96c8bdb70211d6",
    "ext_chants.js": "b546eaf8b795c42ef6b2b43ba8d4080dadfcc97bc4105af539bc05e33713e387",
    "ext_embeddings.js": "bce7b228bec47f9f97542e7916f0b0effeba92a6cf271da86a953c7c582daa33",
    "ext_loras.js": "451319bd5f732943864bbc639e0189679c7b7d3ea188224c47d6360a81c5e0cf",
    "ext_lycos.js": "90ffb8b485a3afb9155d58cfbc42dd9cd14411b3577924c2bcefa94b014e7b40",
    "ext_modelKeyword.js": "ae37d2113c2bb1832006946690b3f7436cc2c616c69f2ddf02ca7fd21e6ce7bf",
    "ext_styles.js": "af18f31f7cdf35639e0fc2e14cbb62e1290361bfce05a6434adb2b8145c5af23",
    "ext_umi.js": "ea8c8ccf424c72d9ad5b467f4d3d23f3ea234d0e4f79f56e9bcfbad49a0a3e33",
    "ext_wildcards.js": "e0b31d7c11584e039a4529d01d064d6869648a7c5d08643fc7978dc3b021406d",
    "tagAutocomplete.js": "0f4943c581395e90e71d9eb59f26258c38acbb028b6e641a22c84b3936ad7a40",
}
LICENSE_SHA256 = "c1532c29734187eddd3948321379fa95ba79191bc3cb41d2787d7dff6e5913b5"
# tags/ shipped in git (README: the other upstream lists are linked, a user's own lists stay local)
TAG_SHA256 = {
    "danbooru.csv": "f936684fa0b041e9a55d35f2052588e28d95eef8672be6829241f8b1a7214732",
    "danbooru_2025.csv": "ca7dec6dc1c80e931657857b4c1f7a7aa44afcc720e2e34fa0d647e62a1f7176",
    "danbooru_e621_merged.csv": "aa9eab8435562f86d7ac0d81f2f7bedcba9f6b93a806010708702406bba85e18",
    "extra-quality-tags.csv": "dd1e3246f9a0548615bb47530369ace101788a0dd665e54eaa20736a91197e61",
    "demo-chants.json": "dd5e09cdbd0be73efa51a78307eb2881cc58027eddded919b5e458af152fd7c0",
}

# Upstream 8e0d81f definitions sam-extra kept as they are (SHA-256[:16] of ast.get_source_segment, LF).
UPSTREAM_FUNCTION_HASHES = {
    "shared_paths.py": {"find_ext_wildcard_paths": "1dbee98e627db4b2"},
    "model_keyword_support.py": {
        "load_hash_cache": "fafa4a79698d5811", "update_hash_cache": "a94963301b006b2e",
        "get_lora_simple_hash": "7057ce110ce5f101", "write_model_keyword_path": "6f090c434f4da40d",
    },
    "tag_frequency_db.py": {
        "transaction": "def1f3be7e6266ad", "TagFrequencyDb": "1ecf6d75ce0d1e43",
        "TagFrequencyDb.__init__": "e1277fde8e8aeee1", "TagFrequencyDb.__check": "9bb662485071a763",
        "TagFrequencyDb.__create_db": "ff7c407e5974e67b", "TagFrequencyDb.__update_db_data": "4d9653972d539a9f",
        "TagFrequencyDb.__get_version": "9d74758d84349842", "TagFrequencyDb.get_all_tags": "9b201846287203b4",
        "TagFrequencyDb.get_tag_count": "339e3c2cff65791a", "TagFrequencyDb.get_tag_counts": "7079a5a5388b9b07",
        "TagFrequencyDb.increase_tag_count": "d6cadf590aa9a6b6", "TagFrequencyDb.reset_tag_count": "4d3dad5ad08a0817",
    },
    "tag_autocomplete_helper.py": {
        "get_embed_db": "8ae5187b026e7a79", "sort_models": "fc0bd6480a8e5e80", "get_wildcards": "9e8eee387b54d558",
        "get_ext_wildcards": "3f8bb1e773431e20", "is_umi_format": "002f1af09d2f403b",
        "parse_umi_format": "0761f31aed81e591", "parse_dynamic_prompt_format": "0656266629f8f02b",
        "get_yaml_wildcards": "ddfb03f42ce25ef9", "get_embeddings": "f146dba9428ad619",
        "_read_safetensors_alias": "0003c3b62c633b81", "_build_alias_map": "aa6353f83a25fed4",
        "_get_lora": "caf812d35c50bb98", "_get_lyco": "1f8e352d926d3fed", "is_visible": "ecfa56b1fb3dba1d",
        "get_lora": "87069e2699a57f2d", "get_lyco": "7e74de68ecc0c84a", "get_style_names": "060f4aff6098355f",
        "write_tag_base_path": "96b9b5769b912fd9", "write_to_temp_file": "312c84d21083dc76",
        "update_tag_files": "c4c2a0d57d717cb4", "update_json_files": "ed8e9c29bac7aa0a",
        "refresh_embeddings": "c1a6a4a6a259231b", "refresh_temp_files": "8589567000444dd8",
        "write_style_names": "3f43ca3b0e756b39", "write_temp_files": "bb681b26fb7083ec",
        "on_ui_settings": "80dd6a2665f7cce7", "get_style_mtime": "17aaf33724797d83",
    },
}
# api_tac's inner definitions (decorators included) other than get_wildcard_contents
UPSTREAM_ROUTE_HASHES = {
    "get_json_info": "b83b8a8e89e1cf76", "get_preview_thumbnail": "0470daf49a9a46c2",
    "api_refresh_temp_files": "d3b0289f5be83b57", "api_refresh_embeddings": "8d485b7a640ecf18",
    "get_lora_info": "0c9373c8ecc76bcb", "get_lyco_info": "ec8c605938776e82",
    "get_civitai_trigger_words": "a93b2863b6b1422f", "get_lora_cached_hash": "6774a5ceb7b6b361",
    "get_path_for_type": "237fd602644da67f", "get_thumb_preview": "a4e30f33b5d5411d",
    "get_thumb_preview_blob": "716dc680e232a622", "refresh_styles_if_changed": "2216e02a4127afbc",
    "db_request": "7e57a3de7433cc6f", "increase_use_count": "ff05b9fb0e5862f6", "get_use_count": "65688bb7da608a32",
    "UseCountListRequest": "a882ad50f9c59c59", "get_use_count_list": "e78f5ac7af0c1970",
    "reset_use_count": "5871bbef16c0842c", "get_all_tag_counts": "c2d199ecc87b164f",
}
VENDORED_PY = tuple(UPSTREAM_FUNCTION_HASHES)
OWN_PY = ("__init__.py", "paths.py", "coexist.py", "boot.py")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _function_hashes(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = _digest(ast.get_source_segment(source, node))
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = _digest(ast.get_source_segment(source, sub))
    return out


def _route_hashes(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    api = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "api_tac")
    out = {}
    for node in api.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators = "".join("@" + ast.get_source_segment(source, d) + "\n" for d in node.decorator_list)
            out[node.name] = _digest(decorators + ast.get_source_segment(source, node))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UpstreamFilesTests(unittest.TestCase):
    def test_the_15_js_files_are_upstream_bytes(self):
        folder = PACKAGE / "javascript"
        self.assertEqual(sorted(p.name for p in folder.iterdir()), sorted(JS_SHA256))
        self.assertEqual(tuple(sorted(JS_SHA256)), tac.JS_ORDER, "Forge's order is the sorted file names")
        for name, digest in JS_SHA256.items():
            with self.subTest(file=name):
                self.assertEqual(_sha256(folder / name), digest)
        self.assertTrue((folder / "tagAutocomplete.js").read_bytes().startswith(b"\xef\xbb\xbf"), "upstream BOM kept")

    def test_licence_is_upstream_bytes(self):
        data = (PACKAGE / "LICENSE").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), LICENSE_SHA256)
        self.assertIn(b"Copyright (c) 2022 Dominik Reh", data)

    def test_shipped_tag_lists_are_upstream_bytes(self):
        for name, digest in TAG_SHA256.items():
            with self.subTest(file=name):
                self.assertEqual(_sha256(ROOT / "tags" / name), digest)
        self.assertIn(b"\r\n", (ROOT / "tags" / "danbooru_e621_merged.csv").read_bytes()[:4096], "CRLF as upstream")

    def test_git_keeps_these_files_byte_for_byte(self):
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").split()
        for pattern in ("/tags/*.csv", "/tags/*.json", "/sam3ext/tagcomplete/javascript/*.js", "/sam3ext/tagcomplete/LICENSE"):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, attributes)
                self.assertEqual(attributes[attributes.index(pattern) + 1], "-text")

    def test_only_the_five_lists_are_tracked_in_tags(self):
        lines = [line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
        self.assertIn("/tags/*", lines)
        allowed = sorted(line[len("!/tags/"):] for line in lines if line.startswith("!/tags/"))
        self.assertEqual(allowed, sorted(TAG_SHA256))


class UpstreamPythonTests(unittest.TestCase):
    def test_unchanged_definitions_match_upstream(self):
        for name, expected in UPSTREAM_FUNCTION_HASHES.items():
            actual = _function_hashes(PACKAGE / name)
            for function, digest in expected.items():
                with self.subTest(file=name, function=function):
                    self.assertEqual(actual.get(function), digest)

    def test_every_definition_is_upstream_or_a_listed_change(self):
        changed = {"tag_autocomplete_helper.py": {"api_tac", "_is_listed_wildcard_file", "_api_tac_once"}}
        for name, expected in UPSTREAM_FUNCTION_HASHES.items():
            actual = set(_function_hashes(PACKAGE / name))
            with self.subTest(file=name):
                self.assertEqual(actual - set(expected), changed.get(name, set()))

    def test_api_tac_changes_only_the_wildcard_route(self):
        routes = _route_hashes(PACKAGE / "tag_autocomplete_helper.py")
        self.assertEqual(set(routes) - set(UPSTREAM_ROUTE_HASHES), {"get_wildcard_contents"})
        for name, digest in UPSTREAM_ROUTE_HASHES.items():
            with self.subTest(route=name):
                self.assertEqual(routes.get(name), digest)

    def test_upstream_definitions_match_the_installed_copy(self):
        """The pins above are the installed copy's (8e0d81f) — checked again where this PC still has it."""
        upstream = tac.upstream_helper()
        if upstream is None:
            self.skipTest("no installed sd-webui-tagcomplete-neo on this PC")
        for name, expected in UPSTREAM_FUNCTION_HASHES.items():
            actual = _function_hashes(upstream.parent / name)
            for function, digest in expected.items():
                with self.subTest(file=name, function=function):
                    self.assertEqual(actual.get(function), digest)
        self.assertLessEqual(UPSTREAM_ROUTE_HASHES.items(), _route_hashes(upstream).items())
        javascript = upstream.parents[1] / "javascript"
        for name, digest in JS_SHA256.items():
            with self.subTest(js=name):
                self.assertEqual(_sha256(javascript / name), digest)

    def test_no_import_of_the_scripts_namespace_is_left(self):
        for name in VENDORED_PY:
            tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
            modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
            with self.subTest(file=name):
                self.assertFalse({m for m in modules if m == "scripts" or m.startswith("scripts.")})
                self.assertNotIn("tag_frequency_db", {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                                                      for a in n.names})

    def test_helper_is_not_a_forge_script(self):
        # Forge runs every .py in scripts/ as a script; the vendored files must not sit there
        names = {p.name for p in (ROOT / "scripts").glob("*.py")}
        self.assertFalse(names & {*VENDORED_PY})
        self.assertIn("builtin_tagcomplete.py", names)
        self.assertLess("builtin_tagcomplete.py", "negpip.py", "negpip.py stays the extension's last script")


class LicenseNoticeTests(unittest.TestCase):
    def test_every_vendored_file_keeps_the_mit_notice_and_lists_its_changes(self):
        for name in VENDORED_PY:
            head = (PACKAGE / name).read_text(encoding="utf-8")[:2500]
            with self.subTest(file=name):
                self.assertIn("https://github.com/eduardoabreu81/sd-webui-tagcomplete-neo @ 8e0d81f", head)
                self.assertIn("Copyright (c) 2022 Dominik Reh", head)
                self.assertIn("Eduardo Abreu", head)
                self.assertIn("MIT License", head)
                self.assertIn("sam3ext/tagcomplete/LICENSE", head)
                self.assertIn("MODIFIED by sam-extra, 2026-10-11", head)

    def test_own_files_say_so(self):
        for path in [*(PACKAGE / name for name in OWN_PY), ROOT / "scripts" / "builtin_tagcomplete.py"]:
            with self.subTest(file=path.name):
                self.assertTrue(path.read_text(encoding="utf-8").startswith("# sam-extra 자체 코드(GPL-3.0-only)"))
        loader = (ROOT / "javascript" / "tagcomplete_loader.js").read_text(encoding="utf-8")
        self.assertTrue(loader.startswith("// sam-extra 자체 코드(GPL-3.0-only)"))

    def test_every_file_in_the_package_is_accounted_for(self):
        names = {p.name for p in PACKAGE.iterdir() if p.is_file()}
        self.assertEqual(names, {*VENDORED_PY, *OWN_PY, "LICENSE"})

    def test_upstream_commit_is_recorded(self):
        from sam3ext import tagcomplete

        self.assertEqual(tagcomplete.UPSTREAM_COMMIT, tac.UPSTREAM_COMMIT)
        self.assertTrue(tac.UPSTREAM_COMMIT.startswith("8e0d81f"))

    def test_third_party_notices_record_the_vendoring(self):
        text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        start = text.index("## sd-webui-tagcomplete-neo")
        section = text[start:text.index("\n## ", start + 1)]
        for needle in ("eduardoabreu81/sd-webui-tagcomplete-neo", "8e0d81f", "Dominik Reh", "Eduardo Abreu",
                       "DominikDoom/a1111-sd-webui-tagcomplete", "sam3ext/tagcomplete/LICENSE", "MIT",
                       "tag_autocomplete_helper.py", "shared_paths.py", "model_keyword_support.py", "tag_frequency_db.py",
                       "wildcard-contents", "_api_tac_once", "install.py", "tags/"):
            with self.subTest(needle=needle):
                self.assertIn(needle, section)
        mit = text[text.index("## MIT License 전문"):]
        self.assertIn("sam3ext/tagcomplete/", mit[:mit.index("```")])
        self.assertIn("Dominik Reh", mit[:mit.index("```")])


if __name__ == "__main__":
    unittest.main()
