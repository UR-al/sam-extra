"""내장 NegPiP(scripts/negpip.py)의 로드 순서 — 예전 sd-forge-negpip 자리(sd-dynamic-thresholding 뒤)를 지키는가.

Dynamic Thresholding 의 process_batch 는 p.sampler_name 을 '<이름>_dynthres<N>' 로 바꾸고, NegPiP 의 process_batch 는
p.sampler_name 으로 cond/uncond 절반을 고른다(rev = 이름이 DDIM·PLMS·UniPC 가 아님). 따로 설치된 sd-forge-negpip 는 폴더
이름순으로 sd-dynamic-thresholding 뒤라 바뀐 이름을 봤다 — 내장은 이 확장 폴더(forge_sam3_extension) 자리라 앞에 오므로
metadata.ini 의 [scripts/negpip.py] After 로 예전 순서를 되살린다.

- metadata.ini 를 Forge 의 ExtensionMetadata 와 같은 방식으로 읽는다.
- Forge 의 실제 list_scripts·ExtensionMetadata·topological_sort 소스(AST 로 꺼냄)를 가짜 확장 목록에 돌려 순서를 본다
  (Forge 소스가 없으면 건너뜀): 설치됨 → NegPiP 가 Dynamic Thresholding 뒤·우리 anima_* 뒤, 없음·꺼짐 → 오류 없이 예전 순서.
"""
from __future__ import annotations

import ast
import configparser
import dataclasses
import os
import re
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]
OURS = "forge_sam3_extension"
DYNTHRES = "sd-dynamic-thresholding"


class MetadataIniTests(unittest.TestCase):
    def _config(self):
        config = configparser.ConfigParser()
        self.assertTrue(config.read(ROOT / "metadata.ini", encoding="utf-8"))
        return config

    def test_negpip_loads_after_dynamic_thresholding(self):
        # Forge: get_script_requirements("After", "scripts/negpip.py", "scripts") — 소문자로, 쉼표·공백으로 나눈다
        value = self._config().get("scripts/negpip.py", "After", fallback="")
        self.assertIn(DYNTHRES, re.split(r"[,\s]+", value.lower().strip()))

    def test_no_folder_wide_scripts_section(self):
        # [scripts] 는 이 확장의 모든 스크립트에 붙는다 — NegPiP 한 파일만 옮긴다
        self.assertFalse(self._config().has_section("scripts"))

    def test_section_names_the_real_script(self):
        self.assertTrue((ROOT / "scripts" / "negpip.py").is_file())

    def test_file_is_ascii(self):
        # Forge 는 metadata.ini 를 encoding 없이 읽는다(configparser.read) — cp949 등 로캘에서도 읽히게 ASCII 만
        (ROOT / "metadata.ini").read_bytes().decode("ascii")


# ---- Forge 의 실제 코드로 순서 계산 ----

def _source(rel: str) -> tuple[str, ast.Module]:
    path = FORGE / rel
    if not path.is_file():
        raise unittest.SkipTest(f"Forge 소스 없음: {path}")
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text)


def _segment(rel: str, name: str) -> str:
    text, tree = _source(rel)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    raise unittest.SkipTest(f"Forge {rel} 에 {name} 없음")


class _Errors:
    def __init__(self):
        self.reports = []

    def report(self, message, exc_info=False):
        self.reports.append(message)


class ForgeListScriptsTests(unittest.TestCase):
    """modules/scripts.py list_scripts 를 그대로 돌린다 — 확장·경로만 가짜."""

    @classmethod
    def setUpClass(cls):
        cls.list_scripts_src = _segment("modules/scripts.py", "list_scripts")
        cls.metadata_src = _segment("modules/extensions.py", "ExtensionMetadata")
        cls.topological_sort_src = _segment("modules/util.py", "topological_sort")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _fake_extension(self, name, scripts, *, builtin=False):
        path = self.tmp / ("builtin" if builtin else "ext") / name
        (path / "scripts").mkdir(parents=True, exist_ok=True)
        for script in scripts:
            (path / "scripts" / script).write_text("# fake\n", encoding="utf-8")
        return name, path, builtin

    def _order(self, installed, *, disabled=()):
        """installed: (이름, 경로, builtin) 목록 — Forge 처럼 builtin 먼저, 각각 이름순. 반환: '확장/파일' 순서와 오류 보고."""
        errors = _Errors()
        ns = {"os": os, "re": re, "configparser": configparser, "dataclasses": dataclasses,
              "namedtuple": namedtuple, "dataclass": dataclasses.dataclass, "errors": errors}
        exec(self.topological_sort_src, ns)
        ns["loaded_extensions"] = {}
        exec(self.metadata_src, ns)
        ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])

        @dataclasses.dataclass
        class ScriptWithDependencies:
            script_canonical_name: str
            file: ScriptFile
            requires: list
            load_before: list
            load_after: list

        extensions = []
        ordered = sorted(installed, key=lambda e: (not e[2], e[0]))
        for name, path, builtin in ordered:
            metadata = ns["ExtensionMetadata"](str(path), name)

            def list_files(subdir, extension, _path=path):
                dirpath = os.path.join(_path, subdir)
                if not os.path.isdir(dirpath):
                    return []
                res = [ScriptFile(str(_path), f, os.path.join(dirpath, f)) for f in sorted(os.listdir(dirpath))]
                return [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

            ext = types.SimpleNamespace(name=name, canonical_name=metadata.canonical_name, is_builtin=builtin,
                                        metadata=metadata, list_files=list_files, enabled=name not in disabled)
            extensions.append(ext)
            ns["loaded_extensions"][metadata.canonical_name] = ext   # Forge: 꺼진 확장도 loaded_extensions 에 있다
        ns.update(
            ScriptFile=ScriptFile, ScriptWithDependencies=ScriptWithDependencies,
            extensions=types.SimpleNamespace(active=lambda: [e for e in extensions if e.enabled]),
            paths=types.SimpleNamespace(script_path=str(self.tmp / "webui")),
            util=types.SimpleNamespace(topological_sort=ns["topological_sort"]),
        )
        exec(self.list_scripts_src, ns)
        files = ns["list_scripts"]("scripts", ".py")
        by_path = {str(e_path): e_name for e_name, e_path, _ in installed}
        return [f"{by_path[f.basedir]}/{f.filename}" for f in files], errors.reports

    def _ours(self):
        return OURS, ROOT, False

    def _world(self, *, dynthres=True, standalone=False):
        world = [
            self._fake_extension("sd_forge_lora", ["lora_script.py"], builtin=True),
            self._fake_extension("aadetailer-neoforge", ["!adetailer.py"]),
            self._ours(),
            self._fake_extension("img2img-hires-fix", ["img2img_hires_fix.py"]),
            self._fake_extension("sd-dynamic-prompts", ["dynamic_prompting.py"]),
        ]
        if dynthres:
            world.append(self._fake_extension(DYNTHRES, ["dynamic_thresholding.py"]))
        if standalone:
            world.append(self._fake_extension("sd-forge-negpip", ["negpip.py"]))
        return world

    def test_negpip_runs_after_dynamic_thresholding_and_after_our_anima_scripts(self):
        order, reports = self._order(self._world())
        self.assertEqual(reports, [])
        negpip = order.index(f"{OURS}/negpip.py")
        self.assertGreater(negpip, order.index(f"{DYNTHRES}/dynamic_thresholding.py"))
        ours = [i for i, name in enumerate(order) if name.startswith(f"{OURS}/") and name != f"{OURS}/negpip.py"]
        self.assertTrue(any(order[i] == f"{OURS}/anima_3_8b.py" for i in ours))
        self.assertGreater(negpip, max(ours), "3.8B 런타임이 검증된 순서(NegPiP 가 우리 스크립트 뒤)")

    def test_standalone_still_sorts_after_the_builtin(self):
        # 둘 다 로드된 설치(내장이 쉬어야 함) — 공존 가드의 '스크립트 객체' 판별은 순서와 무관하지만 예전과 같게
        order, _ = self._order(self._world(standalone=True))
        self.assertLess(order.index(f"{OURS}/negpip.py"), order.index("sd-forge-negpip/negpip.py"))

    def test_missing_or_disabled_dynamic_thresholding_is_harmless(self):
        for label, kwargs in (("not installed", {"world": self._world(dynthres=False)}),
                              ("disabled", {"world": self._world(), "disabled": (DYNTHRES,)})):
            with self.subTest(label):
                order, reports = self._order(kwargs["world"], disabled=kwargs.get("disabled", ()))
                self.assertEqual(reports, [])
                self.assertNotIn(f"{DYNTHRES}/dynamic_thresholding.py", order)
                ours = [name for name in order if name.startswith(f"{OURS}/")]
                self.assertEqual(ours, sorted(ours), "우리 스크립트는 파일 이름순 그대로")
                self.assertEqual(ours[-1], f"{OURS}/negpip.py")
                self.assertLess(order.index(f"{OURS}/negpip.py"), order.index("img2img-hires-fix/img2img_hires_fix.py"))

    def test_without_the_section_negpip_would_run_first(self):
        # 이 테스트가 뜻 있는 비교임을 보인다 — 섹션이 없으면 폴더 이름순으로 Dynamic Thresholding 앞이다
        original = self.metadata_src
        try:
            self.metadata_src = original.replace('self.config.read(filepath)', 'self.config.read(filepath)\n'
                                                 '            self.config.remove_section("scripts/negpip.py")')
            self.assertNotEqual(self.metadata_src, original)
            order, _ = self._order(self._world())
        finally:
            self.metadata_src = original
        self.assertLess(order.index(f"{OURS}/negpip.py"), order.index(f"{DYNTHRES}/dynamic_thresholding.py"))


if __name__ == "__main__":
    unittest.main()
