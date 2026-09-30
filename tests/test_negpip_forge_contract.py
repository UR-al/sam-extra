"""내장 NegPiP 훅(sam3ext/negpip/anima.py·sd.py)과 스크립트가 기대는 Forge 이름·서명이 설치된 Forge 에 있는지 (AST).

sd.py·anima.py 의 훅에서 바꾸지 않은 함수는 상류 0585496 그대로다(test_negpip_vendor 가 글자까지 고정). 여기서는 반대쪽 —
그 훅이 패치하고 기대는 Forge 소스(SD 훅이 읽는 transformer_options["cond_or_uncond"] 포함) — 가 아직 같은 모양인지 본다. Forge 는 텍스트 엔진을 두 번 바꿨다(ad88b6b4 까지 classic_engine /
AnimaTextProcessingEngine, 21886f41 부터 sd_engine.ClipEngine / Qwen06Engine). 둘 중 어느 쪽이 깔려 있어도 통과해야 하고,
Forge 가 없는 CI 에서는 건너뛴다. 깨지면 Forge 가 훅 대상을 바꾼 것 — 훅을 새 Forge 에 맞춘다.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]


def _tree(rel: str) -> ast.Module:
    path = FORGE / rel
    if not path.is_file():
        raise unittest.SkipTest(f"Forge 소스 없음: {path}")
    return ast.parse(path.read_text(encoding="utf-8"))


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def _params(func: ast.FunctionDef) -> list[str]:
    return [a.arg for a in func.args.args]


def _self_attrs(cls: ast.ClassDef) -> set[str]:
    return {
        target.attr
        for node in ast.walk(cls) if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
    }


class AnimaHookTargetTests(unittest.TestCase):
    def test_cross_attention_forward_and_parts(self):
        cls = _classes(_tree("backend/nn/anima.py"))["SelfCrossAttention"]
        methods = _methods(cls)
        # 훅은 (self, x, context, rope_emb, transformer_options) 로 받아 compute_attention 으로 넘긴다
        self.assertEqual(_params(methods["forward"]), ["self", "x", "context", "rope_emb", "transformer_options"])
        self.assertIn("compute_attention", methods)
        self.assertEqual(_params(methods["compute_attention"])[:4], ["self", "q", "k", "v"])
        self.assertLessEqual(
            {"is_SelfAttn", "n_heads", "head_dim", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "v_norm"},
            _self_attrs(cls),
        )

    def test_dit_forward_signature(self):
        forward = _methods(_classes(_tree("backend/nn/anima.py"))["Anima"])["forward"]
        self.assertEqual(_params(forward), ["self", "x", "timesteps", "context", "padding_mask"])
        self.assertIsNotNone(forward.args.kwarg, "c_negpip_mask·transformer_options 는 **kwargs 로 들어온다")

    def test_compile_conditions_is_imported_by_name_into_sampling_function(self):
        # 훅은 condition.compile_conditions 와 sampling_function.compile_conditions 를 둘 다 바꾼다
        condition = _tree("backend/sampling/condition.py")
        names = {n.name for n in condition.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertLessEqual({"compile_conditions", "Condition", "ConditionCrossAttn"}, names)
        imports = {
            alias.name
            for node in _tree("backend/sampling/sampling_function.py").body
            if isinstance(node, ast.ImportFrom) and node.module == "backend.sampling.condition"
            for alias in node.names
        }
        self.assertIn("compile_conditions", imports)

    def test_anima_engine_attribute_and_conditioning(self):
        cls = _classes(_tree("backend/diffusion_engine/anima.py"))["Anima"]
        self.assertIn("get_learned_conditioning", _methods(cls))
        self.assertIn("text_processing_engine_anima", _self_attrs(cls))

    def test_anima_text_engine_is_one_the_mask_builder_knows(self):
        classes = _classes(_tree("backend/text_processing/anima_engine.py"))
        if "Qwen06Engine" in classes:
            cls = classes["Qwen06Engine"]
            methods = _methods(cls)
            self.assertNotIn("tokenize_line", methods, "tokenize_line 이 있으면 옛 엔진 경로로 간다")
            self.assertIn("t5_tokenizer", _self_attrs(cls))
            source = ast.unparse(methods["__call__"])
            # 마스크 규칙이 따라 하는 것: None → 가중치 파싱 끔, Ignore → T5 가중치 1.0, T5 첫 조각
            self.assertIn("disable_weights=n", source)
            self.assertIn("1.0 if i else x[1]", source)
            self.assertIn("t5_chunk[0]", source)
        else:
            legacy = [c for c in classes.values() if "tokenize_line" in _methods(c)]
            self.assertTrue(legacy, f"Anima 텍스트 엔진을 모른다: {sorted(classes)}")
            self.assertIn("t5_multipliers", ast.unparse(legacy[0]))


class SdHookTargetTests(unittest.TestCase):
    def test_cross_attention_forward_and_parts(self):
        cls = _classes(_tree("backend/nn/unet.py"))["CrossAttention"]
        self.assertEqual(_params(_methods(cls)["forward"])[:5], ["self", "x", "context", "value", "mask"])
        self.assertLessEqual({"heads", "to_q", "to_k", "to_v", "to_out"}, _self_attrs(cls))
        self.assertIn("attn2", ast.unparse(_tree("backend/nn/unet.py")))

    def test_cond_or_uncond_reaches_attn2(self):
        # 훅은 cond/uncond 조각을 Forge 가 넘기는 transformer_options["cond_or_uncond"] 로 가린다(sd.py _chunk_labels) —
        # calc_cond_uncond_batch 가 그것을 적어 apply_model 에 넘기고, BasicTransformerBlock 이 attn2 에 transformer_options 로 넘긴다
        sampling = _tree("backend/sampling/sampling_function.py")
        calc = next(n for n in sampling.body if isinstance(n, ast.FunctionDef) and n.name == "calc_cond_uncond_batch")
        source = ast.unparse(calc)
        self.assertIn("transformer_options['cond_or_uncond'] = cond_or_uncond[:]", source)
        self.assertIn("c['transformer_options'] = transformer_options", source)
        self.assertIn("COND = 0", source)
        self.assertIn("UNCOND = 1", source)
        block = _classes(_tree("backend/nn/unet.py"))["BasicTransformerBlock"]
        calls = [
            n for n in ast.walk(block) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.attn2"
        ]
        self.assertTrue(calls)
        for call in calls:
            self.assertIn("transformer_options", {k.arg for k in call.keywords})

    def test_attention_function_exists(self):
        source = ast.unparse(_tree("backend/attention.py"))
        self.assertIn("attention_function =", source)

    def test_clip_engine_tokenize_line_returns_chunks_and_count(self):
        # 새 Forge: sd_engine.ClipEngine, 옛 Forge: classic_engine.ClassicTextProcessingEngine — 스크립트는 둘 다
        # tokenize_line(line) → (chunks, token_count) 로 쓴다(청크 길이 75, 조건 77 행).
        for rel, name in (("backend/text_processing/sd_engine.py", "ClipEngine"),
                          ("backend/text_processing/classic_engine.py", "ClassicTextProcessingEngine")):
            if (FORGE / rel).is_file():
                cls = _classes(_tree(rel))[name]
                tokenize = _methods(cls)["tokenize_line"]
                returns = [n.value for n in ast.walk(tokenize) if isinstance(n, ast.Return) and n.value is not None]
                self.assertTrue(any(isinstance(r, ast.Tuple) and len(r.elts) == 2 for r in returns))
                self.assertIn("chunk_length=75", ast.unparse(_methods(cls)["__init__"]))
                return
        raise unittest.SkipTest("Forge SD 텍스트 엔진 없음")

    def test_sd_models_expose_the_engines_the_script_reads(self):
        self.assertIn("text_processing_engine", _self_attrs(_classes(_tree("backend/diffusion_engine/sd15.py"))["StableDiffusion"]))
        self.assertIn(
            "text_processing_engine_l",
            _self_attrs(_classes(_tree("backend/diffusion_engine/sdxl.py"))["StableDiffusionXL"]),
        )
        base = _methods(_classes(_tree("backend/diffusion_engine/base.py"))["ForgeDiffusionEngine"])
        self.assertIn("is_webui_legacy_model", base)


class ScriptApiTests(unittest.TestCase):
    def test_neo_detection_and_prompt_cache_width(self):
        # IS_NEO 는 backend.shared import 로, reset_prompt_cache 는 Neo 에서 캐시 칸 3개
        if not (FORGE / "backend" / "shared.py").is_file():
            raise unittest.SkipTest("Forge Neo 아님")
        source = ast.unparse(_tree("modules/processing.py"))
        self.assertIn("cached_c = [None, None, None]", source)

    def test_prompt_parser_and_callbacks(self):
        parser = {n.name for n in _tree("modules/prompt_parser.py").body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertLessEqual({"SdConditioning", "get_learned_conditioning", "get_learned_conditioning_prompt_schedules"}, parser)
        callbacks = {n.name for n in _tree("modules/script_callbacks.py").body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertLessEqual({"on_cfg_denoiser", "CFGDenoiserParams"}, callbacks)

    def test_emphasis_helpers(self):
        names = {n.name for n in _tree("backend/text_processing/emphasis.py").body if isinstance(n, ast.FunctionDef)}
        self.assertLessEqual({"get_current_option", "uses_emphasis"}, names)


if __name__ == "__main__":
    unittest.main()
