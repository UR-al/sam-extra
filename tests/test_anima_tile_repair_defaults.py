"""Tile-Repair 패널 기본값 — 감사 M13(Text Encoder 자동 기본값)·M14(전달되지 않던 LLLite Strength/Start/End).

M13: ``default_te_choice`` 가 정렬된 파일명 순으로 'qwen3' 부분일치를 하다 보니 이 PC 의 models/text_encoder 에서는
Qwen3-VL 8B 가 실제 Anima TE(qwen_3_06b_base) 보다 먼저 잡혔다. 벤더 로더는 size mismatch 로 죽고 확장은 이를
'DiT/VAE 가 Anima 체크포인트가 아닐 수 있습니다' 로 번역해 오진을 유도했다.
M14: Strength/Start %/End % 슬라이더는 벤더(``lllite_multiplier`` 만 받음)에 전달되지 않는데 infotext 에는 적용된
것처럼 기록됐다.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.anima_core import AnimaTileRepairArgs, _build_infotext, default_te_choice  # noqa: E402
from sam3ext.ui_anima import ANIMA_ARG_KEYS, _map_widget_values, build_anima_panel  # noqa: E402

# 이 PC 의 models/text_encoder 목록(2026-09-23) — 감사 재현에 쓰인 실제 파일명.
REAL_TE_FILES = [
    "Anima-3.8B-expanded_adapter.safetensors",
    "h3_qwen3vl_8b_tap24.safetensors",
    "krea2UncensoredLLMCLIP_v10.safetensors",
    "qwen3-vl-4b-heretic.safetensors",
    "qwen3-vl-8b-heretic-1.3.0-int8convrot.safetensors",
    "qwen35_4b.safetensors",
    "qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
    "qwen3vl_4b_bf16.safetensors",
    "qwen_3_06b_base.safetensors",
]


class DefaultTextEncoderTests(unittest.TestCase):
    def test_picks_the_anima_qwen3_06b_over_vl_and_qwen35(self):
        choices = ["Use Forge current"] + sorted(REAL_TE_FILES)
        self.assertEqual(default_te_choice(choices), "qwen_3_06b_base.safetensors")

    def test_other_spellings_of_the_06b_encoder(self):
        for name in (
            "qwen3_06b.safetensors",
            "Qwen3-0.6B-fp16.safetensors",
            "qwen_3_0.6b_base.safetensors",
            "qwen3-0_6b.safetensors",
            "anima_baseV10_txt.safetensors",
        ):
            with self.subTest(name=name):
                choices = ["Use Forge current", "h3_qwen3vl_8b_tap24.safetensors", name, "qwen35_4b.safetensors"]
                self.assertEqual(default_te_choice(choices), name)

    def test_vl_qwen35_adapter_and_llmclip_are_never_picked(self):
        wrong = [f for f in REAL_TE_FILES if f != "qwen_3_06b_base.safetensors"]
        self.assertEqual(default_te_choice(["Use Forge current"] + wrong), "Use Forge current",
                         "확신이 없으면 'Use Forge current' — 실행 시 TE 를 고르라는 안내가 뜬다")
        self.assertEqual(default_te_choice(["Use Forge current", "Anima-3.8B-expanded_adapter.safetensors"]),
                         "Use Forge current")

    def test_vl_and_clip_only_count_as_whole_tokens(self):
        # 후속(M13): 'vl'/'clip' 부분 문자열이 정상 0.6B TE 를 떨어뜨리던 과잉 제외.
        for name in (
            "qwen_3_06b_base_eclipse.safetensors",
            "qwen3_06b_clipped_fp8.safetensors",
            "anima_qwen3_06b_devlab.safetensors",
            "Qwen3-0.6B-vlad-merge.safetensors",
            "qwen_3_06b_devlin.safetensors",
        ):
            with self.subTest(name=name):
                choices = ["Use Forge current", "h3_qwen3vl_8b_tap24.safetensors", name]
                self.assertEqual(default_te_choice(choices), name)

    def test_vl_and_clip_tokens_are_still_rejected(self):
        # 숫자·구분자·camelCase 경계의 vl/vlm/clip 은 계속 제외(qwen3vl, Qwen3-VL, AnimaQwenVL, clip_l …).
        for name in (
            "qwen3vl_4b_bf16.safetensors",
            "qwen3-vl-4b-heretic.safetensors",
            "Qwen3VL-0.6B.safetensors",
            "anima_qwen3_06b_vl.safetensors",
            "AnimaQwenVL.safetensors",
            "qwen3_06b_vlm.safetensors",
            "anima_clip_l.safetensors",
            "qwen3_06b-clip.safetensors",
            "krea2UncensoredLLMCLIP_v10.safetensors",
        ):
            with self.subTest(name=name):
                self.assertEqual(default_te_choice(["Use Forge current", name]), "Use Forge current")

    def test_empty_or_forge_only_lists(self):
        self.assertEqual(default_te_choice(["Use Forge current"]), "Use Forge current")
        self.assertEqual(default_te_choice([]), "Use Forge current")


class LLLiteUnimplementedSlidersTests(unittest.TestCase):
    def test_the_three_sliders_are_gone_from_the_panel(self):
        for key in ("lllite_strength", "lllite_start", "lllite_end"):
            self.assertNotIn(key, ANIMA_ARG_KEYS)
        self.assertIn("lllite_multiplier", ANIMA_ARG_KEYS, "벤더가 실제로 받는 multiplier 는 남는다")
        with gr.Blocks():
            panel = build_anima_panel()
        widgets = panel.all_widgets()
        self.assertEqual(len(widgets), len(ANIMA_ARG_KEYS), "위젯 수와 키 수가 어긋나면 값이 한 칸씩 밀린다")
        labels = [getattr(w, "label", "") or "" for w in widgets]
        for label in labels:
            self.assertNotIn("LLLite Strength", label)
            self.assertNotIn("LLLite Start", label)
            self.assertNotIn("LLLite End", label)
        for attr in ("lllite_strength", "lllite_start", "lllite_end"):
            self.assertFalse(hasattr(panel, attr), f"AnimaPanel.{attr} 가 남아 있다")

    def test_widget_values_map_without_the_removed_keys(self):
        with gr.Blocks():
            panel = build_anima_panel()
        values = tuple(w.value for w in panel.all_widgets())
        repair = _map_widget_values(values)
        self.assertEqual(repair.lllite_multiplier, 1.0)
        for attr in ("lllite_strength", "lllite_start", "lllite_end"):
            self.assertFalse(hasattr(repair, attr), f"AnimaTileRepairArgs.{attr} 가 남아 있다")

    def test_infotext_records_only_the_multiplier(self):
        repair = AnimaTileRepairArgs(lllite_model="animaTileRepair_v10.safetensors", lllite_multiplier=0.8)
        text = _build_infotext(repair, seed_used=42)
        self.assertIn("LLLite: animaTileRepair_v10.safetensors (mult 0.8)", text)
        self.assertNotIn("strength", text)
        self.assertNotIn("sched", text)
        self.assertIn("Seed: 42", text)


if __name__ == "__main__":
    unittest.main()
