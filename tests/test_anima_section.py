"""Anima 스크립트 5개가 Forge 의 사용자 섹션(sam3_anima)으로 가는지, 끄기 설정이 먹는지.

Forge 는 사용자 섹션을 설정값 칼럼(#txt2img_settings) 안에 만든다(modules/ui.py:318-319). 그래서 섹션 하나로
Anima 튜닝 묶음이 1열에 모인다. 설정을 끄면 예전처럼 스크립트 컨테이너로 돌아가야 한다.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import layout_lanes as ll  # noqa: E402


def _load_script(name: str):
    """Forge 를 띄우지 않고 스크립트 하나만 불러온다 (tests/test_anima_detail_daemon.py:16-45 와 같은 방식).

    ``mock.patch.dict(sys.modules, …)`` 를 쓰면 스크립트를 이어서 불러올 때 CPython 이 죽는다 — try/finally 로
    ``sys.modules["modules"]`` 를 갈아 끼운다.
    """
    modules_stub = types.ModuleType("modules")

    class Script:
        is_img2img = False

    modules_stub.scripts = types.SimpleNamespace(
        Script=Script, AlwaysVisible=object(), scripts_data=[]
    )
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_cfg_denoiser=lambda fn: None,
        on_model_loaded=lambda fn: None,
        on_script_unloaded=lambda fn: None,
        on_before_ui=lambda fn: None,
        on_ui_settings=lambda fn: None,
    )
    modules_stub.shared = types.SimpleNamespace(opts=types.SimpleNamespace())
    old_modules = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        path = ROOT / "scripts" / name
        spec = importlib.util.spec_from_file_location(f"_t_{name.replace('.', '_')}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_modules is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = old_modules


class SectionValueTests(unittest.TestCase):
    def test_section_is_txt2img_only_and_respects_the_option(self):
        with mock.patch.object(ll, "sections_enabled", return_value=True):
            self.assertEqual(ll.anima_section(False), "sam3_anima")
            self.assertIsNone(ll.anima_section(True), "img2img 는 그대로 둔다")
        with mock.patch.object(ll, "sections_enabled", return_value=False):
            self.assertIsNone(ll.anima_section(False))

    def test_anima38_priority_leads_the_lane_only_when_the_section_is_on(self):
        with mock.patch.object(ll, "sections_enabled", return_value=True):
            self.assertEqual(ll.anima_priority(False, 260209301), -35)
            self.assertEqual(ll.anima_priority(True, 260209301), 260209301)
        with mock.patch.object(ll, "sections_enabled", return_value=False):
            self.assertEqual(ll.anima_priority(False, 260209301), 260209301)

    def test_option_missing_defaults_to_on(self):
        original = sys.modules.get("modules")
        sys.modules.pop("modules", None)
        try:
            self.assertTrue(ll.sections_enabled(), "Forge 밖(테스트)에서는 켜진 것으로 본다")
        finally:
            if original is not None:
                sys.modules["modules"] = original


class ScriptSectionTests(unittest.TestCase):
    SCRIPTS = {
        "anima_3_8b.py": "Anima38Script",
        "anima_detail_daemon.py": "AnimaDetailDaemon",
        "anima_skimmed_cfg.py": "AnimaSkimmedCFG",
        "anima_safe_pag.py": "AnimaSafePAG",
        "anima_vae_2x.py": "AnimaVAE2x",
    }

    def test_each_anima_script_joins_the_section_on_txt2img(self):
        for filename, class_name in self.SCRIPTS.items():
            with self.subTest(script=filename):
                module = _load_script(filename)
                script = getattr(module, class_name)()
                script.is_img2img = False
                with mock.patch.object(ll, "sections_enabled", return_value=True):
                    self.assertEqual(script.section, "sam3_anima")
                    script.is_img2img = True
                    self.assertIsNone(script.section)
                script.is_img2img = False
                with mock.patch.object(ll, "sections_enabled", return_value=False):
                    self.assertIsNone(script.section, "설정을 끄면 Forge 기본 순서로 돌아간다")

    def test_anima38_priority_follows_the_section(self):
        module = _load_script("anima_3_8b.py")
        script = module.Anima38Script()
        script.is_img2img = False
        with mock.patch.object(ll, "sections_enabled", return_value=True):
            self.assertEqual(script.sorting_priority, -35)
        with mock.patch.object(ll, "sections_enabled", return_value=False):
            self.assertEqual(script.sorting_priority, module.LEGACY_SORTING_PRIORITY)

    def test_reference_poc_stays_out_of_the_section(self):
        module = _load_script("anima_ref_poc.py")
        script = module.AnimaRefPoC()
        script.is_img2img = False
        self.assertIsNone(getattr(script, "section", None))


class OnClassTests(unittest.TestCase):
    FEATURES = {"pag", "slg", "apg", "adg", "dcw", "rdc", "cwm", "smc", "dave", "cns", "mod"}

    def test_simple_enables_carry_sam3_on(self):
        sources = {
            "scripts/anima_detail_daemon.py": "anima_dd_enable",
            "scripts/anima_skimmed_cfg.py": "anima_skim_enable",
            "scripts/anima_vae_2x.py": "anima_vae2x_enable",
            "scripts/anima_ref_poc.py": "anima_ref_poc_enable",
            "sam3ext/ui.py": 'eid("enable")',
        }
        for path, marker in sources.items():
            with self.subTest(path=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                index = text.index(marker)
                window = text[max(0, index - 400):index + 200]
                self.assertIn('"sam3-on"', window, f"{path}: 켜기 체크박스에 sam3-on 이 없다")

    def test_guidance_marks_every_feature_once(self):
        import re

        text = (ROOT / "scripts" / "anima_safe_pag.py").read_text(encoding="utf-8")
        found = set(re.findall(r'sam3-on--([a-z]+)"', text))
        self.assertEqual(found, self.FEATURES)
        self.assertIn('"sam3-on-radio"', text, "레거시 cfg_mode 라디오도 세어야 한다")


if __name__ == "__main__":
    unittest.main()
