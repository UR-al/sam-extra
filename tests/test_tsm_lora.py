"""TSM LoRA — 파일 형식·검증, σ→t, 1·2단계 Δ, 샘플별 라우터 풀링, 블록 대응, DiT 훅, Forge 로더 가로채기.

기준 식은 학습 쪽 두 원본을 루프 그대로 옮겨 적었다(이 테스트 환경에는 둘 다 없다):

- LyCORIS 3.2.0.dev55 ``lycoris/modules/tsm.py`` — ``_get_expert_index`` (:310-315), ``_forward_stage1`` (:416-429),
  ``_compute_gates`` (:362-389), ``_forward_stage2``.
- sd-scripts 포크 ``library/lycoris_algo_fixes.py`` — ``_tsm_compute_gates`` (:708-727, 모든 앞 축 평균),
  ``set_timestep`` (:1383-1385, ``round(v·T/1000)`` 을 [0, T] 로).

선택 테스트(환경 변수가 있을 때만, CI 에서는 건너뜀):

- ``SAM3_TSM_REF_DIR`` — 패치한 LyCORIS 모듈로 만든 기준 출력(``stage1_subset.safetensors`` · ``stage1_cases.pt`` ·
  ``stage2_synth.safetensors`` · ``stage2_cases.pt``). 1단계는 실제 학습 파일의 모듈 4개, 2단계는 합성 모듈 3개.
- ``SAM3_TSM_REAL_FILE`` — 실제 TSM 파일. 헤더만 읽어 모든 모듈 이름이 Anima DiT 경로로 풀리는지 본다.
- ``SAM3_TSM_SD_SCRIPTS`` — sd-scripts 포크 폴더(``lycoris`` 가 import 될 때). 살아 있는 ``TSMModule`` 과 대조한다.

Forge 모듈(backend.*)은 import 하지 않는다. Forge 체크아웃이 있으면 ``backend/nn/anima.py`` 의 ``Block`` 클래스
정의만 AST 로 꺼내 meta 장치에서 만든다.
"""
from __future__ import annotations

import ast
import json
import math
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import anima_lora_blocks as alb  # noqa: E402
from sam3ext.tsm_lora import blocks as tb  # noqa: E402
from sam3ext.tsm_lora import delta as td  # noqa: E402
from sam3ext.tsm_lora import fileformat as tf  # noqa: E402
from sam3ext.tsm_lora import listing as tl  # noqa: E402
from sam3ext.tsm_lora import runtime as tr  # noqa: E402

torch.set_num_threads(4)

# 실제 1단계 파일(2.9B, preset anima-exp)의 블록 안 모듈 16개 — 헤더에서 옮겨 적었다.
REAL_FILE_SUFFIXES = (
    "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_output_proj",
    "cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj", "cross_attn_output_proj",
    "mlp_layer1", "mlp_layer2",
    "adaln_modulation_self_attn_1", "adaln_modulation_self_attn_2",
    "adaln_modulation_cross_attn_1", "adaln_modulation_cross_attn_2",
    "adaln_modulation_mlp_1", "adaln_modulation_mlp_2",
)
ATTN_SUFFIXES = tuple(s for s in REAL_FILE_SUFFIXES if s.startswith(("self_attn", "cross_attn")))


# ── 기준 식(LyCORIS 루프 그대로) ──

def oracle_expert_index(t, n, T=1000):
    i = math.ceil(t / T * n)
    return max(1, min(i, n)) - 1


def oracle_t(sigma):
    v = float(torch.tensor(float(sigma), dtype=torch.float32) * 1000.0)
    return max(0, min(1000, int(round(v * 1000 / 1000.0))))


def _module_tensors(sd, name):
    prefix = name + "."
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def oracle_delta_one(sd, name, x, t, stage, mult=1.0, router_input="input"):
    """샘플 하나(또는 같이 평균할 묶음) — tsm.py 의 스케일별 루프, 모든 계산 float64."""
    m = _module_tensors(sd, name)
    x = x.double()
    scales = sorted({int(k.split(".")[1]) for k in m if k.startswith("experts.")})
    n_scales = [len({int(k.split(".")[2]) for k in m if k.startswith(f"experts.{s}.") and k.endswith("down.weight")}) for s in scales]
    rank = m["experts.0.0.down.weight"].shape[0]
    alpha = float(m["alpha"]) if "alpha" in m else float(rank)
    scale = (alpha or rank) / rank
    T = m["timestep_embed.weight"].shape[0] if "timestep_embed.weight" in m else 1000
    t = T if t is None else int(t)

    def expert(s, i, inp):
        return inp @ m[f"experts.{s}.{i}.down.weight"].double().T @ m[f"experts.{s}.{i}.up.weight"].double().T

    if stage == 1:
        out = 0
        for s, n in enumerate(n_scales):
            out = out + expert(s, oracle_expert_index(t, n, T), x)
        return out * scale * mult
    core = oracle_expert_index(t, n_scales[0], T)
    bott = x @ m[f"experts.0.{core}.down.weight"].double().T
    out = bott @ m[f"experts.0.{core}.up.weight"].double().T
    src = bott if router_input == "bottleneck" else x
    z = src.reshape(-1, src.shape[-1]).mean(0)
    g = m["router_fc.weight"].double() @ z + m["router_fc.bias"].double() + m["timestep_embed.weight"].double()[min(max(t - 1, 0), T - 1)]
    for j, n in enumerate(n_scales[1:]):
        out = out + g[j] * expert(j + 1, oracle_expert_index(t, n, T), x)
    return out * scale * mult


def oracle_delta(sd, name, x, ts, stage, mult=1.0, router_input="input", per_sample=True):
    if per_sample or stage == 1:
        return torch.cat([oracle_delta_one(sd, name, x[i:i + 1], ts[i], stage, mult, router_input) for i in range(x.shape[0])])
    out = torch.empty(x.shape[:-1] + (sd[f"{name}.experts.0.0.up.weight"].shape[0],), dtype=torch.float64)
    for t in dict.fromkeys(ts):
        rows = [i for i in range(x.shape[0]) if ts[i] == t]
        out[rows] = oracle_delta_one(sd, name, x[rows], t, stage, mult, router_input)
    return out


# ── 합성 파일 ──

def make_module_sd(name, fan_in, fan_out, rank=2, n_scales=(8, 1), T=1000, router="input", zero_router=False,
                   alpha=None, dtype=torch.float32, seed=0, with_router=True):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for s, n in enumerate(n_scales):
        for i in range(n):
            sd[f"{name}.experts.{s}.{i}.down.weight"] = (torch.randn(rank, fan_in, generator=g) * 0.3).to(dtype)
            sd[f"{name}.experts.{s}.{i}.up.weight"] = (torch.randn(fan_out, rank, generator=g) * 0.3).to(dtype)
    n_ctx = len(n_scales) - 1
    width = fan_in if router == "input" else rank
    if with_router:
        if zero_router:
            sd[f"{name}.router_fc.weight"] = torch.zeros(n_ctx, width, dtype=dtype)
            sd[f"{name}.router_fc.bias"] = torch.zeros(n_ctx, dtype=dtype)
            sd[f"{name}.timestep_embed.weight"] = torch.zeros(T, n_ctx, dtype=dtype)
        else:
            sd[f"{name}.router_fc.weight"] = (torch.randn(n_ctx, width, generator=g) * 0.5).to(dtype)
            sd[f"{name}.router_fc.bias"] = (torch.randn(n_ctx, generator=g) * 0.5).to(dtype)
            sd[f"{name}.timestep_embed.weight"] = (torch.randn(T, n_ctx, generator=g) * 0.5).to(dtype)
    sd[f"{name}.alpha"] = torch.tensor(float(alpha if alpha is not None else rank), dtype=dtype)
    return sd


def stage_meta(stage, **extra):
    args = {"algo": "tsm", "tsm_stage": str(stage), "preset": "anima-exp"}
    args.update(extra)
    return {"ss_saved_format": "tsm-native", "ss_lycoris_algo": "tsm", "ss_network_args": json.dumps(args)}


# ── Anima DiT 스텁(Forge backend/nn/anima.py 와 같은 이름) ──

class _Attn(nn.Module):
    def __init__(self, dim, context_dim, device=None):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False, device=device)
        self.k_proj = nn.Linear(context_dim, dim, bias=False, device=device)
        self.v_proj = nn.Linear(context_dim, dim, bias=False, device=device)
        self.output_proj = nn.Linear(dim, dim, bias=False, device=device)


class _MLP(nn.Module):
    def __init__(self, dim, hidden, device=None):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(dim, hidden, bias=False, device=device)
        self.layer2 = nn.Linear(hidden, dim, bias=False, device=device)


def _adaln(dim, lora_dim, device=None):
    return nn.Sequential(nn.SiLU(), nn.Linear(dim, lora_dim, bias=False, device=device),
                         nn.Linear(lora_dim, 3 * dim, bias=False, device=device))


class StubBlock(nn.Module):
    def __init__(self, dim=16, context_dim=8, hidden=32, lora_dim=4, device=None):
        super().__init__()
        self.self_attn = _Attn(dim, dim, device)
        self.cross_attn = _Attn(dim, context_dim, device)
        self.mlp = _MLP(dim, hidden, device)
        self.adaln_modulation_self_attn = _adaln(dim, lora_dim, device)
        self.adaln_modulation_cross_attn = _adaln(dim, lora_dim, device)
        self.adaln_modulation_mlp = _adaln(dim, lora_dim, device)

    def forward(self, x, emb, context):
        b, d = x.shape[0], x.shape[-1]
        tokens = x.reshape(b, -1, d)
        mod = sum(m(emb)[..., :d] for m in (self.adaln_modulation_self_attn, self.adaln_modulation_cross_attn, self.adaln_modulation_mlp))
        a = self.self_attn.output_proj(self.self_attn.q_proj(tokens) + self.self_attn.k_proj(tokens) + self.self_attn.v_proj(tokens))
        c = self.cross_attn.output_proj(
            self.cross_attn.q_proj(tokens)
            + self.cross_attn.k_proj(context).mean(1, keepdim=True)
            + self.cross_attn.v_proj(context).mean(1, keepdim=True)
        )
        f = self.mlp.layer2(self.mlp.layer1(x))  # 5-D, Forge 처럼
        return x + 0.1 * (a + c + mod).reshape(x.shape) + 0.1 * f


class StubDiT(nn.Module):
    def __init__(self, blocks=2, **kw):
        super().__init__()
        self.blocks = nn.ModuleList([StubBlock(**kw) for _ in range(blocks)])
        self.final = nn.Linear(kw.get("dim", 16), 4, bias=False, device=kw.get("device"))
        self.calls = 0

    def forward(self, x, timesteps, context=None, transformer_options=None, **kwargs):
        self.calls += 1
        emb = x.reshape(x.shape[0], -1, x.shape[-1]).mean(1, keepdim=True) + timesteps.reshape(-1, 1, 1)
        for block in self.blocks:
            x = block(x, emb, context)
        return x


def stub_tsm_sd(dit, n_scales=(8, 1), rank=2, router="input", zero_router=False, seed=0, blocks=None):
    sd = {}
    table = tb.dit_module_table(dit)
    for i, name in enumerate(sorted(table)):
        split = tb.split_block_name(name)
        module = table[name]
        if split is None or not isinstance(module, nn.Linear):
            continue
        if blocks is not None and split[0] not in blocks:
            continue
        sd.update(make_module_sd(name, module.in_features, module.out_features, rank, n_scales, router=router,
                                 zero_router=zero_router, seed=seed + i))
    return sd


class _SysModules:
    """테스트 동안만 sys.modules 항목을 바꾼다(mock.patch.dict 는 torch 가 처음 import 될 때 CPython 을 죽인다)."""

    def __init__(self, **modules):
        self.modules = {name.replace("__", "."): value for name, value in modules.items()}
        self.saved: dict = {}

    def __enter__(self):
        for name, value in self.modules.items():
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = value
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _save_safetensors(path, sd, metadata):
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in sd.items()}, str(path), metadata=metadata)


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 시간
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TimestepTests(unittest.TestCase):
    def test_sigma_to_t_uses_a_float32_multiply_and_half_to_even(self):
        cases = {0.0: 0, 0.0004: 0, 0.0005: 0, 0.0015: 2, 0.0025: 2, 0.001: 1, 0.1245: 124, 0.1255: 125,
                 0.125: 125, 0.12549: 125, 0.5: 500, 0.8755: 876, 0.876: 876, 0.9995: 1000, 1.0: 1000}
        for sigma, expected in cases.items():
            with self.subTest(sigma=sigma):
                self.assertEqual(td.sigma_to_t(sigma), expected)
                self.assertEqual(td.sigma_to_t(sigma), oracle_t(sigma))
        # 0.1255 를 double 로 곱하면 126 — float32 곱이라 125.
        self.assertEqual(int(round(0.1255 * 1000)), 126)

    def test_tensor_rows_and_multiplier(self):
        sig = torch.tensor([0.5, 0.1255, 0.0], dtype=torch.float32)
        values = td.timestep_values(sig)
        self.assertEqual([td.t_from_value(v) for v in values], [500, 125, 0])
        self.assertEqual([td.t_from_value(v) for v in td.timestep_values(sig.reshape(3, 1))], [500, 125, 0])
        # timestep = σ·multiplier 를 받으면 multiplier 로 나눠 σ 를 되찾는다
        self.assertEqual([td.t_from_value(v) for v in td.timestep_values(sig * 1000, multiplier=1000.0)], [500, 125, 0])

    def test_t_is_clamped_and_scaled_to_T(self):
        self.assertEqual(td.t_from_value(-3.0), 0)
        self.assertEqual(td.t_from_value(1200.0), 1000)
        self.assertEqual(td.t_from_value(500.0, num_timesteps=100), 50)

    def test_expert_index_boundaries(self):
        for t, idx in ((0, 0), (1, 0), (124, 0), (125, 0), (126, 1), (250, 1), (251, 2), (500, 3), (501, 4),
                       (875, 6), (876, 7), (999, 7), (1000, 7)):
            with self.subTest(t=t):
                self.assertEqual(td.expert_index(t, 8), idx)
        self.assertEqual(td.expert_index(700, 1), 0)
        # float ceil 그대로: 280/1000*25 = 7.000000000000001 → 8 → 7 (정수 ceil 이면 6)
        self.assertEqual(td.expert_index(280, 25), 7)
        self.assertEqual(td.expert_index(560, 25), oracle_expert_index(560, 25))
        for n in (2, 4, 8, 25, 50, 100):
            for t in range(0, 1001):
                self.assertEqual(td.expert_index(t, n), oracle_expert_index(t, n))

    def test_bf16_training_rounds_sigma_times_1000_like_the_trainer(self):
        # 학습기: timesteps = (fp32 σ · 1000).to(weight_dtype) → set_timestep 이 round. bf16 이면 512 이상은 4 단위.
        def trainer_t(sigma, dtype):
            v = (torch.tensor([sigma], dtype=torch.float32) * 1000).to(dtype).float()
            return max(0, min(1000, int(round(float(v[0])))))

        g = torch.Generator().manual_seed(0)
        sigmas = [0.0, 1.0, 0.8735, 0.874, 0.8755, 0.6255, 0.3755, 0.1245] + torch.rand(4000, generator=g).tolist()
        for name, dtype in ((tf.TIMESTEP_BF16, torch.bfloat16), (tf.TIMESTEP_FP16, torch.float16)):
            for sigma in sigmas:
                self.assertEqual(td.sigma_to_t(sigma, timestep_dtype=name), trainer_t(sigma, dtype), (name, sigma))
        self.assertEqual(td.sigma_to_t(0.8735, timestep_dtype=tf.TIMESTEP_BF16), 872)   # fp32 이면 874
        self.assertEqual(td.sigma_to_t(0.8735), 874)
        # 같은 σ 에서 fp32 식은 다른 전문가를 고를 수 있다(경계 ±2)
        self.assertNotEqual(td.expert_index(td.sigma_to_t(0.8752), 8),
                            td.expert_index(td.sigma_to_t(0.8752, timestep_dtype=tf.TIMESTEP_BF16), 8))

    def test_timestep_dtype_comes_from_ss_mixed_precision(self):
        self.assertEqual(tf.timestep_dtype_from_metadata({"ss_mixed_precision": "bf16"}), tf.TIMESTEP_BF16)
        self.assertEqual(tf.timestep_dtype_from_metadata({"ss_mixed_precision": "fp16"}), tf.TIMESTEP_FP16)
        self.assertEqual(tf.timestep_dtype_from_metadata({"ss_mixed_precision": "no"}), tf.TIMESTEP_FP32)
        self.assertEqual(tf.timestep_dtype_from_metadata({}), tf.TIMESTEP_FP32)
        sd = make_module_sd("lora_unet_blocks_0_mlp_layer1", 24, 20, rank=3)
        meta = dict(stage_meta(2), ss_mixed_precision="bf16")
        tsm = tf.build_tsm_file(sd, meta)
        self.assertEqual(tsm.timestep_dtype, tf.TIMESTEP_BF16)
        self.assertEqual({m.timestep_dtype for m in tsm.modules.values()}, {tf.TIMESTEP_BF16})
        self.assertEqual(tf.build_tsm_file(sd, stage_meta(2)).timestep_dtype, tf.TIMESTEP_FP32)

    def test_embed_row(self):
        self.assertEqual([td.embed_row(t) for t in (0, 1, 2, 500, 1000)], [0, 0, 1, 499, 999])


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 파일 형식
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

class FileFormatTests(unittest.TestCase):
    NAME = "lora_unet_blocks_0_self_attn_q_proj"

    def test_parses_scales_rank_router_and_T(self):
        sd = make_module_sd(self.NAME, 16, 12, rank=3, n_scales=(4, 2, 1), T=100, alpha=6.0)
        tsm = tf.build_tsm_file(sd, stage_meta(2), path="x.safetensors")
        module = tsm.modules[self.NAME]
        self.assertEqual(module.n_scales, (4, 2, 1))
        self.assertEqual((module.rank, module.in_features, module.out_features), (3, 16, 12))
        self.assertEqual(module.scale, 2.0)
        self.assertEqual(module.num_timesteps, 100)
        self.assertEqual(module.router_input, tf.ROUTER_INPUT)
        self.assertEqual((tsm.stage, tsm.stage_source), (2, tf.STAGE_FROM_METADATA))
        self.assertEqual(tsm.summary(), "s2 4+2+1 r3")
        self.assertEqual(tsm.name, "x")

    def test_bottleneck_router_is_read_from_the_width(self):
        sd = make_module_sd(self.NAME, 16, 12, rank=3, router="bottleneck")
        self.assertEqual(tf.build_tsm_file(sd, stage_meta(2)).modules[self.NAME].router_input, tf.ROUTER_BOTTLENECK)

    def test_alpha_zero_or_missing_means_rank(self):
        sd = make_module_sd(self.NAME, 8, 8, rank=4, alpha=0.0)
        self.assertEqual(tf.build_tsm_file(sd, stage_meta(1)).modules[self.NAME].scale, 1.0)
        sd.pop(self.NAME + ".alpha")
        self.assertEqual(tf.build_tsm_file(sd, stage_meta(1)).modules[self.NAME].scale, 1.0)

    def test_stage_from_metadata_then_router(self):
        nonzero = make_module_sd(self.NAME, 8, 8)
        zero = make_module_sd(self.NAME, 8, 8, zero_router=True)
        self.assertEqual(tf.build_tsm_file(zero, stage_meta(2)).stage, 2)
        self.assertTrue(any("0" in w for w in tf.build_tsm_file(zero, stage_meta(2)).warnings))
        self.assertEqual(tf.build_tsm_file(nonzero, stage_meta(1)).stage, 1)
        self.assertEqual((tf.build_tsm_file(nonzero, {}).stage, tf.build_tsm_file(nonzero, {}).stage_source),
                         (2, tf.STAGE_FROM_ROUTER))
        self.assertEqual((tf.build_tsm_file(zero, {}).stage, tf.build_tsm_file(zero, {}).stage_source),
                         (1, tf.STAGE_FROM_ZERO_ROUTER))

    def test_validation_errors_are_bilingual(self):
        good = make_module_sd(self.NAME, 8, 8)
        cases = []
        cases.append(({k: v for k, v in good.items() if not k.endswith("experts.0.3.up.weight")}, stage_meta(1)))
        cases.append((good, {"ss_lycoris_algo": "lokr"}))
        cases.append((good, stage_meta(3)))
        cases.append((good, stage_meta(1, tsm_n_scales="[4, 1]")))
        bad_router = dict(good)
        bad_router[self.NAME + ".router_fc.weight"] = torch.zeros(1, 5)
        cases.append((bad_router, stage_meta(1)))
        no_router = {k: v for k, v in good.items() if "router" not in k}
        cases.append((no_router, stage_meta(2)))  # 라우터 없는 2단계 파일 — 하지만 전문가 키만 있으면 TSM 도 아니다
        conv = dict(good)
        conv[self.NAME + ".experts.0.0.down.weight"] = torch.zeros(2, 8, 1, 1)
        cases.append((conv, stage_meta(1)))
        mixed = dict(good)
        mixed.update(make_module_sd("lora_unet_blocks_1_mlp_layer1", 8, 8, n_scales=(4, 1)))
        cases.append((mixed, stage_meta(1, tsm_n_scales="8,1")))  # 학습기가 적은 구성과 다른 모듈
        for sd, meta in cases:
            with self.subTest(meta=meta, keys=len(sd)):
                with self.assertRaises(tf.TsmFormatError) as ctx:
                    tf.build_tsm_file(sd, meta)
                self.assertIn(" / ", str(ctx.exception))
                self.assertTrue(ctx.exception.korean and ctx.exception.english)

    def test_scale_layouts_may_differ_between_modules_without_a_declared_value(self):
        mixed = make_module_sd(self.NAME, 8, 8)
        mixed.update(make_module_sd("lora_unet_blocks_1_mlp_layer1", 8, 8, n_scales=(4, 2, 1)))
        mixed.update(make_module_sd("lora_unet_blocks_2_mlp_layer1", 8, 8, n_scales=(4, 2, 1)))
        tsm = tf.build_tsm_file(mixed, stage_meta(2))
        self.assertTrue(tsm.mixed_scales)
        self.assertEqual(tsm.n_scales, (4, 2, 1))
        self.assertEqual(tsm.summary(), "s2 4+2+1* r2")
        self.assertEqual(tsm.modules[self.NAME].n_scales, (8, 1))
        self.assertTrue(any("x2" in w and "x1" in w for w in tsm.warnings))

    def test_declared_n_scales_that_match_are_fine(self):
        good = make_module_sd(self.NAME, 8, 8)
        self.assertEqual(tf.build_tsm_file(good, stage_meta(1, tsm_n_scales="8,1")).n_scales, (8, 1))
        self.assertEqual(tf.build_tsm_file(good, stage_meta(1, tsm_n_scales=[8, 1])).n_scales, (8, 1))

    def test_detection_and_split(self):
        tsm = make_module_sd(self.NAME, 8, 8)
        plain = {"lora_unet_blocks_1_mlp_layer1.lora_down.weight": torch.zeros(2, 8),
                 "lora_unet_blocks_1_mlp_layer1.lora_up.weight": torch.zeros(8, 2),
                 "lora_unet_blocks_1_mlp_layer1.alpha": torch.tensor(2.0)}
        self.assertTrue(tf.is_tsm_state_dict(tsm))
        self.assertFalse(tf.is_tsm_state_dict(plain))
        # 전문가 키만(라우터·시간 임베딩 없음)은 TSM 으로 보지 않는다
        self.assertFalse(tf.is_tsm_state_dict({k: v for k, v in tsm.items() if "router" not in k and "embed" not in k}))
        both = dict(tsm)
        both.update(plain)
        part, rest = tf.split_state_dict(both)
        self.assertEqual(set(part), set(tsm))
        self.assertEqual(set(rest), set(plain))
        self.assertEqual(tf.split_state_dict(plain), ({}, plain))

    def test_metadata_helpers(self):
        self.assertTrue(tf.metadata_says_tsm({"ss_saved_format": "tsm-native"}))
        self.assertTrue(tf.metadata_says_tsm({"ss_network_args": '{"algo": "tsm"}'}))
        self.assertFalse(tf.metadata_says_tsm({"ss_lycoris_algo": "lokr"}))
        self.assertTrue(tf.metadata_may_hide_tsm({}))
        self.assertTrue(tf.metadata_may_hide_tsm({"ss_network_module": "lycoris.kohya"}))
        self.assertFalse(tf.metadata_may_hide_tsm({"ss_network_module": "networks.lora"}))
        self.assertFalse(tf.metadata_may_hide_tsm({"ss_network_module": "lycoris.kohya", "ss_lycoris_algo": "locon"}))
        self.assertEqual(tf.network_args({"ss_network_args": "not json"}), {})

    def test_header_load_cache_and_listing(self):
        sd = make_module_sd(self.NAME, 8, 8, dtype=torch.float16)
        plain = {"lora_unet_x.lora_down.weight": torch.zeros(2, 8)}
        with tempfile.TemporaryDirectory() as tmp:
            tsm_path = Path(tmp, "sub", "my_tsm.safetensors")
            tsm_path.parent.mkdir()
            nometa_path = Path(tmp, "nometa.safetensors")
            plain_path = Path(tmp, "plain.safetensors")
            _save_safetensors(tsm_path, sd, stage_meta(1))
            _save_safetensors(nometa_path, sd, {})
            _save_safetensors(plain_path, plain, {"ss_network_module": "networks.lora"})
            meta, header = tf.read_header(tsm_path)
            self.assertEqual(meta["ss_saved_format"], "tsm-native")
            self.assertEqual(header[self.NAME + ".alpha"]["shape"], [])
            tf.release_cache()
            first = tf.load_tsm_file(str(tsm_path))
            self.assertIs(tf.load_tsm_file(str(tsm_path)), first)
            self.assertEqual(first.modules[self.NAME].downs[0][0].dtype, torch.float16)
            self.assertTrue(tf.release_cache())
            with self.assertRaises(tf.TsmFormatError):
                tf.load_tsm_file(str(plain_path))
            rows = tl.walk_safetensors([tmp])
            self.assertEqual({r[0] for r in rows}, {"sub/my_tsm", "nometa", "plain"})
            found = tl.find_tsm_files(rows)
            self.assertEqual(list(found), ["nometa", "sub/my_tsm"])
            # Forge 가 넘기는 메타데이터: TSM 이라고 적혔으면 헤더를 보지 않고, kohya LoRA 라고 적혔으면 건너뛴다
            found = tl.find_tsm_files([("a", str(tsm_path), {"ss_saved_format": "tsm-native"}),
                                       ("b", str(nometa_path), {"ss_network_module": "networks.lora"}),
                                       ("c", str(nometa_path), {})])
            self.assertEqual(list(found), ["a", "c"])
            bad = Path(tmp, "bad.safetensors")
            bad.write_bytes(struct.pack("<Q", 10) + b"{not json}")
            with self.assertRaises(tf.TsmFormatError):
                tf.read_header(bad)
            self.assertEqual(tl.find_tsm_files([("bad", str(bad), None)]), {})
        tf.release_cache()


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Δ — 기준 식 대조
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

class DeltaTests(unittest.TestCase):
    NAME = "lora_unet_blocks_0_mlp_layer1"
    TS = (0, 1, 124, 125, 126, 250, 251, 500, 501, 875, 876, 999, 1000)

    def _check(self, sd, stage, router, shape, n_scales, mult=1.0):
        tsm = tf.build_tsm_file(sd, stage_meta(stage))
        module = tsm.modules[self.NAME]
        g = torch.Generator().manual_seed(1)
        for t in self.TS:
            x = torch.randn(*shape, generator=g)
            with self.subTest(stage=stage, shape=shape, t=t, n_scales=n_scales):
                expected = oracle_delta(sd, self.NAME, x, [t] * shape[0], stage, mult, router)
                got64 = td.module_delta(module, x.double(), t, stage, mult)
                got32 = td.module_delta(module, x, t, stage, mult)
                self.assertEqual(got32.dtype, torch.float32)
                self.assertEqual(got32.shape, expected.shape)
                torch.testing.assert_close(got64, expected, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(got32.double(), expected, rtol=1e-4, atol=1e-5)

    def test_stage1_and_stage2_match_the_lycoris_loop(self):
        for n_scales in ((8, 1), (4, 2, 1), (8,)):
            for router in ("input", "bottleneck"):
                sd = make_module_sd(self.NAME, 24, 20, rank=3, n_scales=n_scales, router=router, alpha=1.5)
                for stage in (1, 2):
                    for shape in ((1, 7, 24), (1, 1, 3, 5, 24), (1, 1, 24)):
                        self._check(sd, stage, router, shape, n_scales)
        sd = make_module_sd(self.NAME, 24, 20, rank=3)
        self._check(sd, 2, "input", (1, 6, 24), (8, 1), mult=0.7)
        self._check(sd, 1, "input", (1, 6, 24), (8, 1), mult=-0.5)

    def test_t_none_falls_back_to_T_like_lycoris(self):
        sd = make_module_sd(self.NAME, 24, 20, rank=3)
        module = tf.build_tsm_file(sd, stage_meta(1)).modules[self.NAME]
        x = torch.randn(1, 5, 24)
        torch.testing.assert_close(td.module_delta(module, x, None, 1), td.module_delta(module, x, 1000, 1))

    def test_cfg_batch_pools_each_sample_like_batch_one(self):
        sd = make_module_sd(self.NAME, 24, 20, rank=3, n_scales=(4, 2, 1))
        module = tf.build_tsm_file(sd, stage_meta(2)).modules[self.NAME]
        x = torch.randn(2, 1, 3, 4, 24, dtype=torch.float64) + torch.tensor([[0.0], [2.0]], dtype=torch.float64).reshape(2, 1, 1, 1, 1)
        for ts in ([600, 600], [30, 980], [0, 1]):
            with self.subTest(ts=ts):
                alone = torch.cat([td.module_delta(module, x[i:i + 1], ts[i], 2) for i in range(2)])
                batched = td.module_delta(module, x, ts, 2)
                torch.testing.assert_close(batched, alone, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(batched, oracle_delta(sd, self.NAME, x, ts, 2), rtol=1e-10, atol=1e-10)
                joint = td.module_delta(module, x, ts, 2, pooling=td.POOL_BATCH)
                torch.testing.assert_close(joint, oracle_delta(sd, self.NAME, x, ts, 2, per_sample=False), rtol=1e-10, atol=1e-10)
        # 같은 t 의 CFG 배치에서 LyCORIS 식(함께 평균)은 샘플별과 다르다
        joint = td.module_delta(module, x, [600, 600], 2, pooling=td.POOL_BATCH)
        self.assertGreater((joint - td.module_delta(module, x, [600, 600], 2)).abs().max().item(), 1e-3)
        # 1단계는 풀링이 없다
        torch.testing.assert_close(td.module_delta(module, x, [600, 600], 1, pooling=td.POOL_BATCH),
                                   td.module_delta(module, x, [600, 600], 1))

    def test_rows_with_different_t_are_grouped(self):
        sd = make_module_sd(self.NAME, 24, 20, rank=3)
        module = tf.build_tsm_file(sd, stage_meta(1)).modules[self.NAME]
        x = torch.randn(3, 4, 24, dtype=torch.float64)
        got = td.module_delta(module, x, [30, 980, 30], 1)
        torch.testing.assert_close(got, oracle_delta(sd, self.NAME, x, [30, 980, 30], 1), rtol=1e-10, atol=1e-10)
        with self.assertRaises(ValueError):
            td.module_delta(module, x, [1, 2], 1)

    def test_stage1_file_through_stage2_formula_drops_the_context_expert(self):
        sd = make_module_sd(self.NAME, 24, 20, rank=3, zero_router=True)
        module = tf.build_tsm_file(sd, stage_meta(1)).modules[self.NAME]
        x = torch.randn(1, 5, 24, dtype=torch.float64)
        core_only = oracle_delta({k: v for k, v in sd.items()}, self.NAME, x, [500], 2)
        torch.testing.assert_close(td.module_delta(module, x, 500, 2), core_only)
        self.assertGreater((td.module_delta(module, x, 500, 1) - core_only).abs().max().item(), 1e-3)

    def test_bfloat16_stays_close_and_keeps_the_dtype(self):
        sd = make_module_sd(self.NAME, 24, 20, rank=3, dtype=torch.float16)
        module = tf.build_tsm_file(sd, stage_meta(2)).modules[self.NAME]
        x = torch.randn(2, 6, 24)
        got = td.module_delta(module, x.bfloat16(), [400, 400], 2)
        self.assertEqual(got.dtype, torch.bfloat16)
        ref = oracle_delta(sd, self.NAME, x, [400, 400], 2)
        self.assertLess(((got.double() - ref).norm() / ref.norm()).item(), 2e-2)

    def test_fused_cache_rebuilds_only_when_the_interval_changes(self):
        sd = make_module_sd(self.NAME, 24, 20, rank=3)
        module = tf.build_tsm_file(sd, stage_meta(1)).modules[self.NAME]
        cache = td.FusedCache()
        x = torch.randn(1, 4, 24)
        for t in (1000, 990, 900, 880, 876):
            td.module_delta(module, x, t, 1, cache=cache)
        self.assertEqual(cache.builds, 1)
        td.module_delta(module, x, 875, 1, cache=cache)
        self.assertEqual(cache.builds, 2)
        td.module_delta(module, x.double(), 875, 1, cache=cache)  # 다른 dtype 은 따로
        self.assertEqual(cache.builds, 3)
        self.assertEqual(cache.nbytes(), 3 * 2 * (24 + 20) * 4 + 3 * 2 * (24 + 20) * 8)
        self.assertTrue(cache.clear())
        self.assertEqual(cache.nbytes(), 0)

    def test_stage2_router_is_copied_to_the_device_once_and_matches(self):
        for router in ("input", "bottleneck"):
            sd = make_module_sd(self.NAME, 24, 20, rank=3, n_scales=(4, 2, 1), router=router)
            module = tf.build_tsm_file(sd, stage_meta(2)).modules[self.NAME]
            cache = td.FusedCache()
            x = torch.randn(2, 1, 3, 4, 24, dtype=torch.float64)
            for t in (1000, 640, 640, 300, 1):
                for pooling in td.POOLINGS:
                    with self.subTest(router=router, t=t, pooling=pooling):
                        torch.testing.assert_close(td.module_delta(module, x, [t, t], 2, pooling=pooling, cache=cache),
                                                   td.module_delta(module, x, [t, t], 2, pooling=pooling),
                                                   rtol=0, atol=0)
            self.assertEqual(len(cache._routers), 1)   # fp64 입력 → fp64 라우터 사본 하나
            self.assertGreater(cache.nbytes(), 0)
            self.assertTrue(cache.clear())
            self.assertEqual((cache.nbytes(), len(cache._routers)), (0, 0))

    def test_total_delta_sums_files_and_skips_zero_strength(self):
        sd_a = make_module_sd(self.NAME, 24, 20, rank=3, seed=1)
        sd_b = make_module_sd(self.NAME, 24, 20, rank=2, seed=2, n_scales=(4, 1))
        a = tf.build_tsm_file(sd_a, stage_meta(1)).modules[self.NAME]
        b = tf.build_tsm_file(sd_b, stage_meta(2)).modules[self.NAME]
        x = torch.randn(2, 3, 24, dtype=torch.float64)
        got = td.total_delta([(a, 0.5, 1), (b, 1.0, 2), (a, 0.0, 1)], x, [300, 300])
        expected = oracle_delta(sd_a, self.NAME, x, [300, 300], 1, 0.5) + oracle_delta(sd_b, self.NAME, x, [300, 300], 2)
        torch.testing.assert_close(got, expected)
        self.assertIsNone(td.total_delta([(a, 0.0, 1)], x, 300))


class _OpSpy:
    """TorchDispatchMode 로 디스패치된 ATen op 의 (이름, 출력 모양, dtype) 를 모은다 — 큰 fp32 사본을 만드는지 본다."""

    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        spy = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for o in out if isinstance(out, (tuple, list)) else (out,):
                    if isinstance(o, torch.Tensor):
                        spy.ops.append((str(func), tuple(o.shape), o.dtype, o.numel()))
                return out

        self.ops = []
        self._mode = _Mode()

    def __enter__(self):
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self._mode.__exit__(*exc)


class FastPathTests(unittest.TestCase):
    """Forge 훅이 쓰는 :class:`td.Slot` — 기준 식(:func:`td.module_delta`)과 같은 값, 제자리 더하기, 라우터 평균 재사용."""

    NAME = "lora_unet_blocks_0_mlp_layer1"

    def _module(self, stage=2, router="input", n_scales=(4, 2, 1), dtype=torch.float32, rank=3, seed=0):
        sd = make_module_sd(self.NAME, 24, 20, rank=rank, n_scales=n_scales, router=router, alpha=1.5, dtype=dtype, seed=seed)
        return sd, tf.build_tsm_file(sd, stage_meta(stage)).modules[self.NAME]

    def test_slot_add_equals_the_reference_delta(self):
        """scale·강도를 addmm 의 alpha 로 준 제자리 더하기 = 기준 식 out + Δ (fp64 는 1e-12, fp32 는 반올림 몇 번)."""
        g = torch.Generator().manual_seed(4)
        for n_scales in ((8, 1), (4, 2, 1), (8,)):
            for router in ("input", "bottleneck"):
                _sd, module = self._module(router=router, n_scales=n_scales)
                for stage in (1, 2):
                    for pooling in td.POOLINGS:
                        for shape in ((2, 7, 24), (2, 1, 3, 5, 24), (2, 24), (1, 4, 24)):
                            for t in (1, 125, 126, 500, 999, 1000):
                                for mult in (1.0, 0.7, -0.4):
                                    x = torch.randn(*shape, generator=g, dtype=torch.float64)
                                    base = torch.randn(*shape[:-1], 20, generator=g, dtype=torch.float64)
                                    ref = base + td.module_delta(module, x, t, stage, mult, pooling)
                                    slot = td.Slot(module, mult, stage)
                                    out = base.clone()
                                    got = slot.add(out, x, t, pooling, td.FusedCache(), td.PoolCache())
                                    self.assertIs(got, out)   # 제자리
                                    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
                                    x32, base32 = x.float(), base.float()
                                    ref32 = base32 + td.module_delta(module, x32, t, stage, mult, pooling)
                                    got32 = slot.add(base32.clone(), x32, t, pooling, td.FusedCache(), td.PoolCache())
                                    torch.testing.assert_close(got32, ref32, rtol=1e-5, atol=1e-5)

    def test_out_of_place_paths_and_dtype_mismatch(self):
        _sd, module = self._module()
        x = torch.randn(2, 5, 24, dtype=torch.float64)
        base = torch.randn(2, 5, 20, dtype=torch.float64)
        ref = base + td.module_delta(module, x, 600, 2, 0.8)
        slot = td.Slot(module, 0.8, 2)
        kept = base.clone()
        got = slot.add(kept, x, 600, in_place=False)
        self.assertIsNot(got, kept)
        torch.testing.assert_close(kept, base, rtol=0, atol=0)   # 원래 출력은 그대로
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
        # 출력 dtype 이 입력과 다르면(제자리 불가) Δ 를 따로 만들어 출력 dtype 으로 더한다
        out32 = base.float()
        self.assertFalse(td.can_add_in_place(out32, x))
        got = slot.add(out32, x, 600)
        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(got.double(), ref, rtol=1e-6, atol=1e-6)
        # 기울기를 쓰는 출력·연속이 아닌 출력은 제자리로 고치지 않는다
        self.assertFalse(td.can_add_in_place(base.clone().requires_grad_(True), x))
        self.assertFalse(td.can_add_in_place(torch.randn(2, 20, 5, dtype=torch.float64).transpose(1, 2), x))
        with torch.inference_mode():
            inf_out = torch.randn(2, 5, 20, dtype=torch.float64)
            self.assertTrue(td.can_add_in_place(inf_out, x))
        self.assertFalse(td.can_add_in_place(inf_out, x))        # 추론 텐서는 inference_mode 밖에서 못 고친다

    def test_bfloat16_in_place_stays_within_one_rounding_of_the_old_hook(self):
        """bf16: 옛 훅 ``out + (F.linear(mid, up) * scale).to(bf16)`` (반올림 3번) 과 새 ``addmm_`` (fp32 로 모아 1번) 의 차이는
        bf16 반올림 몇 번 안이고, fp64 기준에는 새 쪽이 더 가깝거나 같다."""
        for stage in (1, 2):
            _sd, module = self._module(stage=stage, dtype=torch.float16)
            g = torch.Generator().manual_seed(7)
            x = torch.randn(2, 64, 24, generator=g)
            base = torch.randn(2, 64, 20, generator=g)
            exact = base.double() + td.module_delta(module, x.double(), 400, stage, 0.9)
            old = base.bfloat16() + td.module_delta(module, x.bfloat16(), 400, stage, 0.9)
            new = td.Slot(module, 0.9, stage).add(base.bfloat16(), x.bfloat16(), 400)
            self.assertEqual(new.dtype, torch.bfloat16)
            scale = exact.abs().max().item()
            self.assertLess((new.double() - old.double()).abs().max().item(), 4 * 2 ** -8 * scale)
            err_new = ((new.double() - exact).norm() / exact.norm()).item()
            err_old = ((old.double() - exact).norm() / exact.norm()).item()
            self.assertLess(err_new, 5e-3)
            self.assertLessEqual(err_new, err_old * 1.05)

    def test_fused_upload_equals_per_expert_conversion(self):
        """이어 붙인 뒤 한 번에 dtype 을 바꿔 옮겨도 전문가마다 바꿔 이어 붙이던 값과 비트 단위로 같다."""
        _sd, module = self._module(dtype=torch.float16, n_scales=(4, 2, 1))
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            for t in (1, 500, 1000):
                indices = module.expert_indices(t)
                down, up = td.fuse_experts(module, indices, dtype, "cpu")
                old_down = torch.cat([module.downs[s][i].to(dtype) for s, i in enumerate(indices)], 0)
                old_up = torch.cat([module.ups[s][i].to(dtype) for s, i in enumerate(indices)], 1)
                self.assertTrue(down.is_contiguous() and up.is_contiguous())
                torch.testing.assert_close(down, old_down, rtol=0, atol=0)
                torch.testing.assert_close(up, old_up, rtol=0, atol=0)

    def test_weak_copies_share_one_device_copy(self):
        """같은 모듈을 다른 강도로 거는 약한 복사 블록은 장치 사본 하나를 같이 쓴다(alpha 를 up 에 접지 않으므로)."""
        _sd, module = self._module(stage=1)
        cache = td.FusedCache()
        x = torch.randn(2, 3, 24)
        a, b = td.Slot(module, 1.0, 1), td.Slot(module, 0.12, 1)
        for t in (900, 880):
            ya = a.add(torch.zeros(2, 3, 20), x, t, cache=cache)
            yb = b.add(torch.zeros(2, 3, 20), x, t, cache=cache)
            torch.testing.assert_close(yb, ya * 0.12, rtol=1e-6, atol=1e-7)
        self.assertEqual(cache.builds, 1)
        self.assertIs(a._down, b._down)
        a.add(torch.zeros(2, 3, 20), x, 600, cache=cache)
        b.add(torch.zeros(2, 3, 20), x, 600, cache=cache)
        self.assertEqual(cache.builds, 2)
        self.assertTrue(a.holds_device_tensors())
        a.forget()
        self.assertFalse(a.holds_device_tensors())

    def test_pooled_mean_is_the_old_value_without_a_full_fp32_copy(self):
        _sd, module = self._module(stage=2)
        x = torch.randn(2, 1, 16, 16, 24).bfloat16()
        for pooling in td.POOLINGS:
            rows = 1 if pooling == td.POOL_BATCH else 2
            old = x.reshape(rows, -1, 24).to(torch.float32).mean(1)   # 옛 식(전체 fp32 사본)
            with _OpSpy() as spy:
                z = td.pooled_mean(x, 24, pooling, torch.float32)
                g = td.gates(module, x, 500, pooling)
            torch.testing.assert_close(z, old, rtol=0, atol=0)          # CPU 는 비트 단위로 같다
            self.assertEqual(g.shape, (rows, 2))
            big = [op for op in spy.ops if op[2] == torch.float32 and op[3] >= x.numel()]
            self.assertEqual(big, [], f"full-size fp32 tensor made: {big}")
        # 옛 식이었다면 스파이가 잡는다(스파이가 살아 있는지)
        with _OpSpy() as spy:
            x.reshape(2, -1, 24).to(torch.float32).mean(1)
        self.assertTrue(any(op[2] == torch.float32 and op[3] == x.numel() for op in spy.ops))

    def test_pool_cache_reuses_only_the_same_live_tensor(self):
        cache = td.PoolCache()
        x = torch.randn(2, 6, 24)
        first = cache.mean(x, 24, td.POOL_SAMPLE, torch.float32)
        self.assertIs(cache.mean(x, 24, td.POOL_SAMPLE, torch.float32), first)   # k_proj · v_proj
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        # 같은 값의 다른 텐서 · 다른 묶음 방식은 다시 낸다
        cache.mean(x.clone(), 24, td.POOL_SAMPLE, torch.float32)
        self.assertEqual(cache.misses, 2)
        y = torch.randn(2, 6, 24)
        a = cache.mean(y, 24, td.POOL_SAMPLE, torch.float32)
        b = cache.mean(y, 24, td.POOL_BATCH, torch.float32)
        self.assertEqual((a.shape, b.shape, cache.misses), ((2, 24), (1, 24), 4))
        # 제자리로 바뀐 텐서(버전이 오른 것)는 맞추지 않는다
        z0 = cache.mean(y, 24, td.POOL_SAMPLE, torch.float32)
        y.add_(1.0)
        z1 = cache.mean(y, 24, td.POOL_SAMPLE, torch.float32)
        torch.testing.assert_close(z1, z0 + 1.0)
        # 약한 참조 — 풀린 텐서의 자리를 새 텐서가 받아도 맞추지 않고, 텐서의 수명을 늘리지 않는다
        import gc
        import weakref

        w = torch.randn(2, 6, 24)
        cache.mean(w, 24, td.POOL_SAMPLE, torch.float32)
        ref = weakref.ref(w)
        del w
        gc.collect()
        self.assertIsNone(ref())
        misses = cache.misses
        cache.mean(torch.randn(2, 6, 24), 24, td.POOL_SAMPLE, torch.float32)
        self.assertEqual(cache.misses, misses + 1)
        # Forge 처럼 inference_mode 의 추론 텐서(버전 없음)도 같은 객체면 맞춘다
        with torch.inference_mode():
            v = torch.randn(2, 6, 24)
            first = cache.mean(v, 24, td.POOL_SAMPLE, torch.float32)
            self.assertIs(cache.mean(v, 24, td.POOL_SAMPLE, torch.float32), first)
        cache.clear()
        self.assertIsNone(cache._ref)

    def test_shared_pool_gives_the_same_gates_as_separate_pools(self):
        """q·k·v 처럼 같은 입력을 받는 서로 다른 모듈 셋 — 평균을 같이 써도 각자 낸 것과 비트 단위로 같다."""
        modules = [self._module(stage=2, seed=s)[1] for s in (1, 2, 3)]
        x = torch.randn(2, 9, 24)
        shared = td.PoolCache()
        outs_shared = [td.Slot(m, 0.9, 2).add(torch.zeros(2, 9, 20), x, 300, pool=shared) for m in modules]
        outs_alone = [td.Slot(m, 0.9, 2).add(torch.zeros(2, 9, 20), x, 300, pool=None) for m in modules]
        self.assertEqual((shared.hits, shared.misses), (2, 1))
        for a, b in zip(outs_shared, outs_alone):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 블록 대응
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

def _forge_block_class():
    """Forge backend/nn/anima.py 의 Block·SelfCrossAttention·GPT2FeedForward 정의만 꺼내 실행한다(없으면 None)."""
    try:
        from tests._forge_checkout import require_forge_file

        path = require_forge_file("backend/nn/anima.py")
    except unittest.SkipTest:
        return None
    except Exception:
        return None
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"GPT2FeedForward", "SelfCrossAttention", "Block"}
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in wanted]
    if {node.name for node in body} != wanted:
        return None
    namespace = {"torch": torch, "nn": nn, "Optional": __import__("typing").Optional,
                 "Callable": __import__("typing").Callable, "rearrange": None}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Block"]


class _MetaDiT(nn.Module):
    """2.9B 크기의 DiT 를 meta 장치에(메모리 없이). Forge 체크아웃이 있으면 Forge 의 Block 클래스로."""

    def __init__(self, blocks=40):
        super().__init__()
        forge_block = _forge_block_class()
        with torch.device("meta"):
            if forge_block is not None:
                made = [forge_block(x_dim=2048, context_dim=1024, num_heads=16, mlp_ratio=4.0, adaln_lora_dim=256) for _ in range(blocks)]
            else:
                made = [StubBlock(dim=2048, context_dim=1024, hidden=8192, lora_dim=256) for _ in range(blocks)]
        self.uses_forge = forge_block is not None
        self.blocks = nn.ModuleList(made)


class BlockMappingTests(unittest.TestCase):
    def test_suffixes_of_the_real_file_resolve_on_the_dit(self):
        dit = _MetaDiT(blocks=40)
        table = tb.dit_module_table(dit)
        self.assertTrue(tb.is_anima_dit(dit))
        shapes = {
            "self_attn_q_proj": (2048, 2048), "cross_attn_k_proj": (1024, 2048), "cross_attn_v_proj": (1024, 2048),
            "mlp_layer1": (2048, 8192), "mlp_layer2": (8192, 2048),
            "adaln_modulation_mlp_1": (2048, 256), "adaln_modulation_mlp_2": (256, 6144),
        }
        names = [f"lora_unet_blocks_{i}_{s}" for i in range(40) for s in REAL_FILE_SUFFIXES]
        self.assertEqual(len(names), 640)
        for name in names:
            module = table.get(name)
            self.assertIsInstance(module, nn.Linear, name)
            suffix = tb.split_block_name(name)[1]
            if suffix in shapes:
                self.assertEqual((module.in_features, module.out_features), shapes[suffix], name)

    def test_plan_and_bind_on_the_same_layout(self):
        dit = StubDiT(blocks=2)
        sd = stub_tsm_sd(dit)
        tsm = tf.build_tsm_file(sd, stage_meta(1))
        plan = tb.plan_file(tsm, 2)
        self.assertFalse(plan.remapped)
        binding = tb.bind_plan(plan, tb.dit_module_table(dit), 0.8, 1)
        self.assertEqual(len(binding), 32)
        self.assertEqual((binding.missing, binding.shape_mismatch), ([], []))
        items = next(iter(binding.modules.values()))[1]
        self.assertEqual(items[0][1:], (0.8, 1))

    def _full(self, blocks):
        names = {f"lora_unet_blocks_{i}_{s}": None for i in range(blocks) for s in REAL_FILE_SUFFIXES}
        modules = {name: types.SimpleNamespace(name=name) for name in names}
        return types.SimpleNamespace(modules=modules)

    def test_40_to_52_follows_the_inserted_block_policy(self):
        tsm = self._full(40)
        keep = tb.plan_file(tsm, 52, alb.DUPLICATE_KEEP)
        self.assertTrue(keep.remapped)
        self.assertEqual(len(keep.targets), 52 * 16)
        self.assertEqual(keep.duplicated_blocks, 12)
        self.assertEqual(len(tb.plan_file(tsm, 52, alb.DUPLICATE_ADDITIVE).targets), 52 * 16)
        skip = tb.plan_file(tsm, 52, alb.DUPLICATE_SKIP)
        self.assertEqual(len(skip.targets), 40 * 16)
        self.assertEqual(skip.skipped_inserted, 12 * 16)
        weak = tb.plan_file(tsm, 52, alb.DUPLICATE_WEAK, (0.12, alb.WEAK_SCOPE_ATTN))
        self.assertEqual(len(weak.targets), 40 * 16 + 12 * len(ATTN_SUFFIXES))
        factors = sorted({f for items in weak.targets.values() for _m, f in items})
        self.assertEqual(factors, [0.12, 1.0])
        weak_all = tb.plan_file(tsm, 52, alb.DUPLICATE_WEAK, (0.5, alb.WEAK_SCOPE_ALL))
        self.assertEqual(len(weak_all.targets), 52 * 16)
        # 끼워 넣은 블록은 매핑의 앞 원본 블록과 같은 TsmModule 을 쓴다
        mapping = alb.BLOCK_MAPPINGS[(40, 52)]
        for target, source in enumerate(mapping):
            got = keep.targets[f"lora_unet_blocks_{target}_mlp_layer1"][0][0]
            self.assertIs(got, tsm.modules[f"lora_unet_blocks_{source}_mlp_layer1"])
        self.assertIn("40->52", keep.describe())

    def test_contraction_and_base(self):
        tsm = self._full(40)
        down = tb.plan_file(tsm, 28)
        self.assertEqual(len(down.targets), 28 * 16)
        for target, source in enumerate(alb.BLOCK_MAPPINGS[(40, 28)]):
            self.assertIs(down.targets[f"lora_unet_blocks_{target}_self_attn_q_proj"][0][0],
                          tsm.modules[f"lora_unet_blocks_{source}_self_attn_q_proj"])
        self.assertEqual(len(tb.plan_file(self._full(28), 52).targets), 52 * 16)

    def test_sparse_files(self):
        sparse_same = types.SimpleNamespace(modules={f"lora_unet_blocks_{i}_mlp_layer1": object() for i in (0, 5, 39)})
        plan = tb.plan_file(sparse_same, 40)
        self.assertEqual(len(plan.targets), 3)
        self.assertFalse(plan.remapped)
        with self.assertRaises(tf.TsmFormatError):
            tb.plan_file(sparse_same, 52)
        text_side = types.SimpleNamespace(modules={"lora_unet_llm_adapter_blocks_0_mlp_0": object(),
                                                   "lora_unet_final_layer_linear": object()})
        plan = tb.plan_file(text_side, 40)
        self.assertEqual(plan.text_side, ["lora_unet_llm_adapter_blocks_0_mlp_0"])
        self.assertEqual(list(plan.targets), ["lora_unet_final_layer_linear"])

    def test_bind_reports_missing_and_shape_mismatch(self):
        dit = StubDiT(blocks=1)
        sd = stub_tsm_sd(dit)
        sd.update(make_module_sd("lora_unet_blocks_0_not_there", 16, 16))
        wrong = make_module_sd("lora_unet_blocks_0_mlp_layer1", 15, 32)
        for key in list(sd):
            if key.startswith("lora_unet_blocks_0_mlp_layer1."):
                del sd[key]
        sd.update(wrong)
        plan = tb.plan_file(tf.build_tsm_file(sd, stage_meta(1)), 1)
        binding = tb.bind_plan(plan, tb.dit_module_table(dit), 1.0, 1)
        self.assertEqual(binding.missing, ["lora_unet_blocks_0_not_there"])
        self.assertEqual(binding.shape_mismatch, ["lora_unet_blocks_0_mlp_layer1"])
        self.assertEqual(len(binding), 15)

    def test_non_anima_dit_is_not_recognised(self):
        self.assertFalse(tb.is_anima_dit(nn.Linear(2, 2)))
        self.assertFalse(tb.is_anima_dit(types.SimpleNamespace(blocks=[])))
        wrapped = types.SimpleNamespace(_orig_mod=StubDiT(blocks=1))
        self.assertTrue(tb.is_anima_dit(wrapped))


@unittest.skipUnless(os.environ.get("SAM3_TSM_REAL_FILE") and Path(os.environ.get("SAM3_TSM_REAL_FILE", "")).is_file(),
                     "set SAM3_TSM_REAL_FILE to a real TSM .safetensors to check its module names")
class RealFileHeaderTests(unittest.TestCase):
    def test_every_module_of_the_real_file_resolves_to_an_anima_path(self):
        path = os.environ["SAM3_TSM_REAL_FILE"]
        meta, header = tf.read_header(path)
        self.assertTrue(tf.metadata_says_tsm(meta))
        names = sorted(tf.tsm_module_names(header.keys()))
        self.assertTrue(tl.is_tsm_candidate(path, meta))
        blocks = {tb.split_block_name(n)[0] for n in names}
        dit = _MetaDiT(blocks=max(blocks) + 1)
        table = tb.dit_module_table(dit)
        unresolved = [n for n in names if not isinstance(table.get(n), nn.Linear)]
        self.assertEqual(unresolved, [])
        for name in names:
            linear = table[name]
            down = header[f"{name}.experts.0.0.down.weight"]["shape"]
            up = header[f"{name}.experts.0.0.up.weight"]["shape"]
            self.assertEqual((down[1], up[0]), (linear.in_features, linear.out_features), name)
        print(f"\n  real file: {len(names)} modules over {len(blocks)} blocks resolve "
              f"({'Forge Block class' if dit.uses_forge else 'stub Block'})")
        # 3.8B 로 올려도 모든 이름이 52블록 DiT 에 풀린다
        modules = {n: types.SimpleNamespace(name=n) for n in names}
        plan = tb.plan_file(types.SimpleNamespace(modules=modules), 52, alb.DUPLICATE_KEEP)
        table52 = tb.dit_module_table(_MetaDiT(blocks=52))
        self.assertEqual([n for n in plan.targets if n not in table52], [])
        self.assertEqual(len(plan.targets), len(names) // 40 * 52)


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 세션·훅
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

class HookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dit = StubDiT(blocks=2).double()
        for p in self.dit.parameters():
            p.requires_grad_(False)
        self.sd = stub_tsm_sd(self.dit, n_scales=(4, 2, 1), rank=2, seed=3)
        self.tsm = tf.build_tsm_file(self.sd, stage_meta(2), path="")
        self.x = torch.randn(2, 1, 2, 3, 16, dtype=torch.float64)
        self.ctx = torch.randn(2, 5, 8, dtype=torch.float64)
        self.sigma = torch.tensor([0.4321, 0.4321], dtype=torch.float32)

    def tearDown(self):
        tr.finish_generation()

    def _session(self, stage=2, mult=0.9, pooling=td.POOL_SAMPLE):
        entry = tr.TsmEntry(self.tsm, mult, tr.SOURCE_PICKER, stage=stage, label="t")
        session = tr.TsmSession([entry], pooling)
        report = session.bind(self.dit, alb.DUPLICATE_KEEP, (0.12, "attn"))
        self.assertEqual(report.refused, [])
        return session

    def _capture(self):
        outputs = {}
        handles = []
        for name, module in self.dit.named_modules():
            if isinstance(module, nn.Linear):
                def hook(mod, args, out, name=name):
                    outputs[name] = (args[0].detach().clone(), out.detach().clone())
                handles.append(module.register_forward_hook(hook))
        return outputs, handles

    def test_every_bound_linear_gets_exactly_the_oracle_delta(self):
        session = self._session()
        self.assertEqual(len(session.bound), 32)
        tr.HOOKS.attach(self.dit, session)
        outputs, handles = self._capture()
        try:
            self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session})
        finally:
            for h in handles:
                h.remove()
        t = td.sigma_to_t(0.4321)
        self.assertEqual(session.seen_t, {t})
        self.assertEqual(session.calls, 1)
        checked = 0
        for path, (inp, out) in outputs.items():
            name = tb.module_key(path)
            if name not in self.tsm.modules:
                continue
            linear = self.dit.get_submodule(path)
            expected = torch.nn.functional.linear(inp, linear.weight) + oracle_delta(self.sd, name, inp, [t, t], 2, 0.9)
            torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-10, msg=name)
            checked += 1
        self.assertEqual(checked, 32)

    def test_no_session_means_no_change_and_hooks_come_off(self):
        base = self.dit(self.x, self.sigma, context=self.ctx, transformer_options={})
        session = self._session()
        tr.HOOKS.attach(self.dit, session)
        tr.remember_session(session)
        self.assertEqual(tr.HOOKS.counts(), (1, 32))
        same = self.dit(self.x, self.sigma, context=self.ctx, transformer_options={})
        torch.testing.assert_close(same, base, rtol=0, atol=0)
        on = self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session})
        self.assertGreater((on - base).abs().max().item(), 1e-4)
        # Tile-Repair 의 벤더 루프처럼 transformer_options 없이 부르면 앞 호출의 상태가 새지 않는다
        torch.testing.assert_close(self.dit(self.x, self.sigma, context=self.ctx), base, rtol=0, atol=0)
        self.assertGreater(session.cache.nbytes(), 0)
        sessions, hooked = tr.finish_generation()
        self.assertEqual((sessions, hooked), ([session], True))
        self.assertEqual(session.cache.nbytes(), 0)
        self.assertEqual(tr.HOOKS.counts(), (0, 0))
        off = self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session})
        torch.testing.assert_close(off, base, rtol=0, atol=0)

    def test_attach_is_idempotent_and_an_exception_does_not_leak_state(self):
        session = self._session()
        self.assertEqual(tr.HOOKS.attach(self.dit, session), 32)
        self.assertEqual(tr.HOOKS.attach(self.dit, session), 0)
        self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session})
        self.assertGreater(session.cache.nbytes(), 0)
        bad_ctx = torch.randn(2, 5, 7, dtype=torch.float64)  # cross k_proj 가 실패한다
        with self.assertRaises(RuntimeError):
            self.dit(self.x, self.sigma, context=bad_ctx, transformer_options={tr.OPTION_KEY: session})
        # post-hook 은 always_call — 던져도 활성 세션을 비우고(OOM 뒤 postprocess 가 안 불려도) 장치 캐시를 놓는다
        self.assertIsNone(tr.active_session())
        self.assertEqual(session.cache.nbytes(), 0)
        layer = self.dit.blocks[0].mlp.layer1
        x = torch.randn(1, 16, dtype=torch.float64)
        torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, layer.weight), rtol=0, atol=0)  # DiT 밖 호출
        # 다시 돌리면 캐시를 새로 만들어 같은 값을 낸다
        on = self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session})
        self.assertGreater(session.cache.nbytes(), 0)
        self.assertIsNone(tr.active_session())
        torch.testing.assert_close(
            on, self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session}), rtol=0, atol=0
        )

    def test_hooks_hold_modules_weakly_and_rehook_a_reused_id(self):
        import gc
        import weakref

        session = self._session()
        tr.HOOKS.attach(self.dit, session)
        session.targets.clear()  # apply_tsm 처럼 — 이제 아무도 DiT 를 강하게 쥐지 않는다
        layer = self.dit.blocks[0].mlp.layer1
        # id 가 다른(풀린 뒤 재사용된) 모듈을 가리키는 낡은 기록이면 다시 건다 — 앞 모듈이 풀린 상황을 흉내 낸다
        tr.HOOKS._linear_handles[id(layer)][1].remove()
        other = nn.Linear(16, 32, bias=False).double()
        stale = other.register_forward_hook(lambda *a: None)
        tr.HOOKS._linear_handles[id(layer)] = (weakref.ref(other), stale)
        session2 = self._session()
        self.assertEqual(tr.HOOKS.attach(self.dit, session2), 1)
        self.assertEqual(len(other._forward_hooks), 0)   # 낡은 훅은 뗐다
        self.assertEqual(len(layer._forward_hooks), 1)   # 같은 모듈에 두 번 걸지 않았다
        session2.targets.clear()
        # 세션·훅 관리자는 DiT 를 붙잡지 않는다
        ref = weakref.ref(self.dit)
        tr.remember_session(session2)
        self.dit = None
        layer = None
        gc.collect()
        self.assertIsNone(ref())

    def test_a_new_pass_releases_the_previous_pass_caches(self):
        first = self._session()
        tr.HOOKS.attach(self.dit, first)
        tr.remember_session(first)
        self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: first})
        self.assertGreater(first.cache.nbytes(), 0)
        second = self._session()
        tr.HOOKS.attach(self.dit, second)
        tr.remember_session(second)
        self.assertEqual(first.cache.nbytes(), 0)        # batch count·하이레스마다 쌓이지 않는다
        self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: second})
        self.assertGreater(second.cache.nbytes(), 0)
        self.assertTrue(tr.release_device_caches())       # post_sample
        self.assertEqual(second.cache.nbytes(), 0)
        sessions, _hooked = tr.finish_generation()
        self.assertEqual(sessions, [first, second])

    def test_batch_mismatch_with_equal_t_broadcasts_otherwise_skips(self):
        session = self._session(stage=1)
        module = self.dit.blocks[0].mlp.layer1
        x = torch.randn(4, 3, 16, dtype=torch.float64)
        same = session.delta_for(id(module), x, [500.0, 500.0])
        self.assertEqual(same.shape, (4, 3, 32))
        self.assertIsNone(session.delta_for(id(module), x, [500.0, 30.0]))
        self.assertEqual(session.skipped_calls, 1)
        self.assertIsNone(session.delta_for(id(module), torch.randn(4, 3, 15, dtype=torch.float64), [500.0]))

    def test_missing_timestep_skips_instead_of_using_T(self):
        session = self._session()
        tr.HOOKS.attach(self.dit, session)
        base = self.dit(self.x, self.sigma, context=self.ctx, transformer_options={})
        logs = []
        tr.set_logger(logs.append)
        try:
            hook_args = (self.x,)
            tr._dit_pre_hook(self.dit, hook_args, {"transformer_options": {tr.OPTION_KEY: session}})
            self.assertIsNone(tr.active_session())
        finally:
            tr.set_logger(None)
        self.assertTrue(logs)
        torch.testing.assert_close(self.dit(self.x, self.sigma, context=self.ctx, transformer_options={}), base)

    def test_non_anima_dit_refuses_every_file(self):
        entry = tr.TsmEntry(self.tsm, 1.0, tr.SOURCE_PROMPT, stage=1, label="t")
        session = tr.TsmSession([entry])
        report = session.bind(nn.Sequential(nn.Linear(2, 2)))
        self.assertEqual(len(report.refused), 1)
        self.assertEqual(session.bound, {})

    def test_mostly_unmatched_file_is_refused_like_forge(self):
        other = StubDiT(blocks=2, dim=12, context_dim=8, hidden=32, lora_dim=4).double()
        entry = tr.TsmEntry(self.tsm, 1.0, tr.SOURCE_PROMPT, stage=1, label="t")
        session = tr.TsmSession([entry])
        report = session.bind(other)
        self.assertEqual(len(report.refused), 1)
        self.assertIn("do not match", report.refused[0][1])

    def test_session_uses_each_files_training_timestep_dtype(self):
        meta = dict(stage_meta(2), ss_mixed_precision="bf16")
        tsm_bf16 = tf.build_tsm_file(self.sd, meta)
        session = tr.TsmSession([tr.TsmEntry(tsm_bf16, 1.0, tr.SOURCE_PROMPT, stage=2, label="b")])
        session.bind(self.dit, alb.DUPLICATE_KEEP)
        module = self.dit.blocks[0].cross_attn.k_proj
        name = "lora_unet_blocks_0_cross_attn_k_proj"
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        values = td.timestep_values(torch.tensor([0.8735, 0.8735]))
        got = session.delta_for(id(module), x, values)
        torch.testing.assert_close(got, oracle_delta(self.sd, name, x, [872, 872], 2))
        self.assertEqual(session.seen_t, {872})

    def test_two_files_on_one_module_add_up(self):
        sd2 = stub_tsm_sd(self.dit, n_scales=(8, 1), rank=3, seed=11)
        tsm2 = tf.build_tsm_file(sd2, stage_meta(1))
        session = tr.TsmSession([tr.TsmEntry(self.tsm, 0.5, tr.SOURCE_PROMPT, stage=2, label="a"),
                                 tr.TsmEntry(tsm2, 1.2, tr.SOURCE_PICKER, stage=1, label="b")])
        session.bind(self.dit, alb.DUPLICATE_KEEP)
        module = self.dit.blocks[1].cross_attn.k_proj
        name = "lora_unet_blocks_1_cross_attn_k_proj"
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        got = session.delta_for(id(module), x, [700.0, 700.0])
        expected = oracle_delta(self.sd, name, x, [700, 700], 2, 0.5) + oracle_delta(sd2, name, x, [700, 700], 1, 1.2)
        torch.testing.assert_close(got, expected)
        self.assertGreater(session.fused_bytes(), 0)

    def test_qkv_share_one_router_mean_per_dit_call(self):
        """2단계(input 라우터): self_attn q·k·v 가 같은 x, cross_attn k·v 가 같은 context — 블록마다 평균 3번을 다시 쓴다."""
        session = self._session()
        tr.HOOKS.attach(self.dit, session)
        with torch.inference_mode():   # Forge 처럼 — 추론 텐서
            self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session})
        self.assertEqual(session.pool.hits, 3 * len(self.dit.blocks))
        self.assertIsNone(session.pool._ref)   # DiT 호출이 끝나면 비운다

    def test_hook_adds_in_place_and_mixed_t_rows_use_the_reference(self):
        session = self._session()
        module = self.dit.blocks[0].mlp.layer1
        name = "lora_unet_blocks_0_mlp_layer1"
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        out = torch.nn.functional.linear(x, module.weight)
        base = out.clone()
        call = session.begin_call([500.0, 500.0])
        got = session.apply(id(module), x, out, call)
        self.assertIs(got, out)
        torch.testing.assert_close(got, base + oracle_delta(self.sd, name, x, [500, 500], 2, 0.9), rtol=1e-10, atol=1e-10)
        # 행마다 t 가 다르면(드묾) 기준 식으로 새 텐서를 만든다
        call = session.begin_call([500.0, 30.0])
        self.assertIsNone(call.uniform[0])
        got = session.apply(id(module), x, base.clone(), call)
        torch.testing.assert_close(got, base + oracle_delta(self.sd, name, x, [500, 30], 2, 0.9), rtol=1e-10, atol=1e-10)
        # 배치와 t 개수가 다르면 건너뛴다(같은 t 면 넓혀 쓴다)
        x4 = torch.randn(4, 3, 16, dtype=torch.float64)
        out4 = torch.nn.functional.linear(x4, module.weight)
        self.assertIsNone(session.apply(id(module), x4, out4.clone(), session.begin_call([500.0, 30.0])))
        got = session.apply(id(module), x4, out4.clone(), session.begin_call([500.0, 500.0]))
        torch.testing.assert_close(got, out4 + oracle_delta(self.sd, name, x4, [500] * 4, 2, 0.9), rtol=1e-10, atol=1e-10)
        # 묶이지 않은 모듈 · 1-D 입력은 그대로
        self.assertIsNone(session.apply(id(self.dit), x, out, call))
        self.assertIsNone(session.apply(id(module), torch.randn(16, dtype=torch.float64), out, call))

    def test_release_drops_every_device_reference(self):
        import gc
        import weakref

        session = self._session()
        tr.HOOKS.attach(self.dit, session)
        tr.remember_session(session)
        self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session})
        slot = session.slots[id(self.dit.blocks[0].mlp.layer1)][0]
        self.assertTrue(slot.holds_device_tensors())
        down = weakref.ref(slot._down)
        router = weakref.ref(slot._router[0])
        self.assertTrue(tr.release_device_caches())
        gc.collect()
        self.assertIsNone(down())         # 캐시 · Slot 어디에도 남지 않아 실제로 풀린다
        self.assertIsNone(router())
        self.assertFalse(any(s.holds_device_tensors() for slots in session.slots.values() for s in slots))
        self.assertFalse(tr.release_device_caches())
        # 다음 패스가 다시 쓰면 다시 만든다(같은 값)
        first = self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session})
        self.assertTrue(slot.holds_device_tensors())
        tr.release_device_caches()
        torch.testing.assert_close(
            self.dit(self.x, self.sigma, context=self.ctx, transformer_options={tr.OPTION_KEY: session}), first, rtol=0, atol=0
        )

    def test_transient_bytes_is_the_largest_single_call(self):
        session = self._session()
        # 스텁: rank 2 · 스케일 3 → 중간값 너비 6, 가장 큰 출력은 adaLN 의 3·16 = 48
        self.assertEqual(session.transient_bytes(100, 2), 100 * (6 + 48) * 2)

    def test_stage_and_pooling_normalisation(self):
        self.assertEqual(tr.normalize_stage("자동 (파일 메타데이터)"), tr.STAGE_AUTO)
        self.assertEqual(tr.normalize_stage("1단계 식 (전문가 합)"), tr.STAGE_1)
        self.assertEqual(tr.normalize_stage("2단계 식 (라우터 게이트)"), tr.STAGE_2)
        self.assertEqual(tr.normalize_stage("끔 (TSM 빼고 · 비교용)"), tr.STAGE_OFF)
        self.assertEqual(tr.normalize_stage(2), tr.STAGE_2)
        self.assertIsNone(tr.normalize_stage("what"))
        self.assertEqual(tr.normalize_pooling("배치 전체 (LyCORIS 원본 forward)"), td.POOL_BATCH)
        self.assertEqual(tr.normalize_pooling("샘플별 (학습과 같음 · 기본)"), td.POOL_SAMPLE)
        self.assertEqual(tr.resolve_stage(self.tsm, "auto"), 2)
        self.assertEqual(tr.resolve_stage(self.tsm, "1"), 1)
        self.assertIsNone(tr.resolve_stage(self.tsm, "off"))


class _Interrupted(Exception):
    pass


class FakeForgeSampler:
    """Forge ``sd_samplers_common.Sampler`` 의 스텝 루프 부분만 — ``sample``/``sample_img2img`` 가 ``launch_sampling`` 으로
    스텝을 돌리고 돌아온 뒤에야 processing 이 VAE 디코드를 한다(``decode`` 가 그 자리)."""

    def __init__(self, step, log):
        self.step = step
        self.log = log

    def launch_sampling(self, steps, func):   # modules/sd_samplers_common.py:436 와 같은 모양
        try:
            return func()
        except _Interrupted:
            return "interrupted-latent"

    def sample(self, p, steps=3, interrupt_at=None, fail_at=None):
        def run():
            for i in range(steps):
                if i == interrupt_at:
                    raise _Interrupted()
                if i == fail_at:
                    raise RuntimeError("boom")
                self.step(i)
            return "latent"

        samples = self.launch_sampling(steps, run)
        self.log.append(("sampled", samples))
        return samples

    sample_img2img = sample


class SamplerReleaseTests(unittest.TestCase):
    """패스의 샘플링이 끝나면(``launch_sampling`` 이 돌아오면) VAE 디코드 전에 장치 사본이 풀린다."""

    def setUp(self):
        torch.manual_seed(0)
        self.dit = StubDiT(blocks=2).double()
        for p in self.dit.parameters():
            p.requires_grad_(False)
        self.sd = stub_tsm_sd(self.dit, n_scales=(8, 1), rank=2, seed=3)
        self.tsm = tf.build_tsm_file(self.sd, stage_meta(1), path="")
        self.x = torch.randn(2, 1, 2, 3, 16, dtype=torch.float64)
        self.ctx = torch.randn(2, 5, 8, dtype=torch.float64)
        self.log = []

    def tearDown(self):
        tr.finish_generation()

    def _pass(self):
        """apply_tsm 처럼: 세션을 만들고 훅을 걸고 기억한다."""
        session = tr.TsmSession([tr.TsmEntry(self.tsm, 1.0, tr.SOURCE_PICKER, stage=1, label="t")])
        session.bind(self.dit, alb.DUPLICATE_KEEP)
        tr.HOOKS.attach(self.dit, session)
        session.targets.clear()
        tr.remember_session(session)
        return session

    def _step(self, session, sigmas=(0.9, 0.5, 0.1)):
        def step(i):
            self.dit(self.x, torch.tensor([sigmas[i % len(sigmas)]] * 2), context=self.ctx,
                     transformer_options={tr.OPTION_KEY: session})
            self.log.append(("step", i, session.cache.nbytes() > 0))
        return step

    def _decode(self, sessions):
        # processing 의 decode_latent_batch 자리 — 이 시점에 장치 사본이 하나도 남지 않아야 한다
        held = sum(s.cache.nbytes() for s in sessions) + sum(
            slot.holds_device_tensors() for s in sessions for slots in s.slots.values() for slot in slots)
        self.log.append(("decode", held))
        return held

    def test_txt2img_hires_releases_before_each_decode_and_rebuilds_for_hires(self):
        first = self._pass()
        sampler = FakeForgeSampler(None, self.log)
        sampler.step = self._step(first)
        self.assertEqual(tr.wrap_sampler_release(sampler), "installed")
        sampler.sample(None)
        self.assertEqual(self._decode([first]), 0)      # 1차 뒤 latent 가 아닌 업스케일러의 디코드(processing.py ~1395)
        hires = self._pass()                            # sample_hr_pass: 새 샘플러 · 새 세션
        hr_sampler = FakeForgeSampler(self._step(hires), self.log)
        self.assertEqual(tr.wrap_sampler_release(hr_sampler), "installed")
        hr_sampler.sample_img2img(None)
        self.assertEqual(self._decode([first, hires]), 0)   # 하이레스 패스 안의 디코드(~1557) — post_sample 보다 먼저
        steps = [entry for entry in self.log if entry[0] == "step"]
        self.assertTrue(all(held for _s, _i, held in steps))   # 스텝 동안은 사본이 있다(하이레스는 다시 만들었다)
        self.assertEqual(len(steps), 6)

    def test_img2img_batch_count_wraps_the_reused_sampler_once(self):
        sampler = FakeForgeSampler(None, self.log)
        for n in range(2):
            session = self._pass()
            sampler.step = self._step(session)
            self.assertEqual(tr.wrap_sampler_release(sampler), "installed" if n == 0 else "already")
            sampler.sample_img2img(None)
            self.assertEqual(self._decode([session]), 0)
        self.assertTrue(getattr(sampler.launch_sampling, "_sam3_tsm_release", False))
        self.assertFalse(getattr(sampler.launch_sampling.__wrapped__, "_sam3_tsm_release", False))  # 겹쳐 감싸지 않았다

    def test_interrupt_and_exception_still_release(self):
        session = self._pass()
        sampler = FakeForgeSampler(self._step(session), self.log)
        tr.wrap_sampler_release(sampler)
        self.assertEqual(sampler.sample(None, interrupt_at=2), "interrupted-latent")
        self.assertEqual(self._decode([session]), 0)
        with self.assertRaises(RuntimeError):
            sampler.sample(None, fail_at=1)
        self.assertEqual(self._decode([session]), 0)

    def test_unwrappable_sampler_is_reported(self):
        self.assertEqual(tr.wrap_sampler_release(None), "unavailable")
        self.assertEqual(tr.wrap_sampler_release(object()), "unavailable")

        class Frozen:
            __slots__ = ()

            def launch_sampling(self, steps, func):
                return func()

        self.assertEqual(tr.wrap_sampler_release(Frozen()), "unavailable")


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Forge 로더 가로채기
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

class LoaderHookTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def load_lora_for_models(model, clip, lora, strength_model, strength_clip, filename="default", online_mode=False):
            self.calls.append((dict(lora), strength_model, filename))
            return "patched-model", clip

        self.module = types.ModuleType("networks")
        self.module.__file__ = "/forge/extensions-builtin/sd_forge_lora/networks.py"
        self.module.load_lora_for_models = load_lora_for_models
        self.module.dynamic_args = types.SimpleNamespace(anima=True)
        self.original = load_lora_for_models
        self.sd_model = types.SimpleNamespace(current_lora_hash="h1")
        self.shared = types.SimpleNamespace(sd_model=self.sd_model)
        self.modules_pkg = types.ModuleType("modules")
        self.modules_pkg.shared = self.shared
        self.logs = []
        tr.set_logger(self.logs.append)
        tf.release_cache()

    def tearDown(self):
        tr.uninstall_loader_hook(self.module)
        tr.set_logger(None)
        tf.release_cache()

    def _load(self, lora, strength, filename):
        with _SysModules(modules=self.modules_pkg, modules__shared=self.shared):
            return self.module.load_lora_for_models("model", "clip", lora, strength, 1.0, filename, False)

    def test_install_is_idempotent_and_uninstall_restores(self):
        self.assertTrue(tr.install_loader_hook(self.module))
        self.assertFalse(tr.install_loader_hook(self.module))
        self.assertTrue(tr.loader_hook_installed(self.module))
        self.assertIs(self.module.load_lora_for_models._sam3_tsm_original, self.original)
        self.assertTrue(tr.uninstall_loader_hook(self.module))
        self.assertIs(self.module.load_lora_for_models, self.original)
        self.assertFalse(tr.uninstall_loader_hook(self.module))

    def test_tsm_files_are_recorded_by_lora_hash_and_plain_files_pass_through(self):
        tr.install_loader_hook(self.module)
        tsm_sd = make_module_sd("lora_unet_blocks_0_self_attn_q_proj", 8, 8)
        plain = {"lora_unet_blocks_0_mlp_layer1.lora_down.weight": torch.zeros(2, 8)}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "a_tsm.safetensors"))
            _save_safetensors(path, tsm_sd, stage_meta(1))
            self.assertEqual(self._load(dict(tsm_sd), 0.7, path), ("model", "clip"))
            self.assertEqual(self.calls, [])  # 순정 로더는 TSM 만 든 파일을 보지 않는다
            records = tr.records_for(self.sd_model)
            self.assertEqual(len(records), 1)
            self.assertEqual((records[0]["strength"], records[0]["filename"], records[0]["error"]), (0.7, path, ""))
            self.assertEqual(records[0]["tsm"].stage, 1)
            self.assertEqual(records[0]["tsm"].metadata["ss_saved_format"], "tsm-native")  # 헤더에서 읽었다
            # 섞인 파일: TSM 키만 떼고 나머지는 순정 로더로
            mixed = dict(tsm_sd)
            mixed.update(plain)
            self.assertEqual(self._load(mixed, 1.0, path), ("patched-model", "clip"))
            self.assertEqual(set(self.calls[-1][0]), set(plain))
            # 같은 해시에서 같은 파일을 다시 읽으면 덮어쓴다
            self.assertEqual(len(tr.records_for(self.sd_model)), 1)
            self.assertEqual(tr.records_for(self.sd_model)[0]["strength"], 1.0)
            # 일반 LoRA 는 그대로
            self.assertEqual(self._load(dict(plain), 1.0, "plain.safetensors"), ("patched-model", "clip"))
            self.assertEqual(set(self.calls[-1][0]), set(plain))
            # 해시가 바뀌면(다른 LoRA 묶음) 옛 기록은 보이지 않는다
            self.sd_model.current_lora_hash = "h2"
            self.assertEqual(tr.records_for(self.sd_model), [])
            self._load(dict(tsm_sd), 0.3, path)
            self.assertEqual([r["strength"] for r in tr.records_for(self.sd_model)], [0.3])
        tr.forget_records(self.sd_model)
        self.assertEqual(tr.records_for(self.sd_model), [])

    def test_broken_tsm_file_and_non_anima_are_recorded_as_errors(self):
        tr.install_loader_hook(self.module)
        notified = []
        tr.uninstall_loader_hook(self.module)
        tr.install_loader_hook(self.module, notify=notified.append)
        broken = make_module_sd("lora_unet_blocks_0_self_attn_q_proj", 8, 8)
        broken["lora_unet_blocks_0_self_attn_q_proj.router_fc.weight"] = torch.zeros(1, 5)
        self.assertEqual(self._load(broken, 1.0, "broken.safetensors"), ("model", "clip"))
        record = tr.records_for(self.sd_model)[0]
        self.assertIsNone(record["tsm"])
        self.assertIn("router_fc", record["error"])
        self.module.dynamic_args.anima = False
        self._load(make_module_sd("lora_unet_blocks_0_self_attn_q_proj", 8, 8), 1.0, "flux.safetensors")
        self.assertEqual(len(notified), 2)
        self.assertEqual(self.calls, [])


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 선택: 패치한 LyCORIS 로 만든 기준 출력
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

_REF_DIR = os.environ.get("SAM3_TSM_REF_DIR", "")


@unittest.skipUnless(_REF_DIR and Path(_REF_DIR, "stage1_cases.pt").is_file(),
                     "set SAM3_TSM_REF_DIR to the TSM reference outputs (stage1_cases.pt ...)")
class LycorisReferenceTests(unittest.TestCase):
    def _run(self, file_name, cases_name):
        """기준: delta64 = fp64 모듈(배치 1 의미 — CFG 배치 경우는 샘플별), delta32 · delta_bf16 = 그 dtype 모듈,
        delta64_joint_pool = 패치한 LyCORIS 가 CFG 배치를 함께 평균한 값, gates64 = 그 경우의 게이트."""
        tf.release_cache()
        tsm = tf.load_tsm_file(str(Path(_REF_DIR, file_name)))
        cases = torch.load(str(Path(_REF_DIR, cases_name)), weights_only=False)
        worst = {"delta32": 0.0, "delta64": 0.0, "delta_bf16": 0.0, "joint": 0.0, "gates": 0.0,
                 "hook64": 0.0, "hook32": 0.0, "hook_bf16": 0.0, "hook_joint": 0.0}
        counts = {key: 0 for key in worst}
        for case in cases:
            module = tsm.modules[case["module"]]
            if case.get("router_input"):
                self.assertEqual(module.router_input, case["router_input"], case["module"])
            ts = [None if t is None else int(t) for t in case["ts"]]
            x = case["x"]
            stage, mult = case["stage"], case["multiplier"]
            label = (case["module"], stage, tuple(case["ts"]), mult)
            with self.subTest(case=label):
                self.assertEqual(tsm.stage, stage)
                if "experts" in case and all(t is not None for t in ts):
                    self.assertEqual([list(module.expert_indices(t)) for t in ts], [list(e) for e in case["experts"]])
                ref64 = case["delta64"]
                scale = ref64.abs().max().item() or 1.0
                got64 = td.module_delta(module, x.double(), ts, stage, mult)
                err64 = (got64 - ref64).abs().max().item() / scale
                self.assertLess(err64, 1e-6)
                worst["delta64"], counts["delta64"] = max(worst["delta64"], err64), counts["delta64"] + 1
                if "delta32" in case:
                    got32 = td.module_delta(module, x, ts, stage, mult)
                    self.assertEqual(got32.dtype, torch.float32)
                    err32 = (got32.double() - case["delta32"].double()).abs().max().item() / scale
                    self.assertLess(err32, 2e-6)
                    worst["delta32"], counts["delta32"] = max(worst["delta32"], err32), counts["delta32"] + 1
                if "delta_bf16" in case:
                    gotbf = td.module_delta(module, x.bfloat16(), ts, stage, mult)
                    self.assertEqual(gotbf.dtype, torch.bfloat16)
                    errbf = (gotbf.double() - ref64).norm().item() / (ref64.norm().item() or 1.0)
                    refbf = (case["delta_bf16"].double() - ref64).norm().item() / (ref64.norm().item() or 1.0)
                    self.assertLess(errbf, max(2e-2, 3 * refbf))  # LyCORIS 자신의 bf16 오차와 같은 크기
                    worst["delta_bf16"], counts["delta_bf16"] = max(worst["delta_bf16"], errbf), counts["delta_bf16"] + 1
                if "delta64_joint_pool" in case:
                    joint = td.module_delta(module, x.double(), ts, stage, mult, pooling=td.POOL_BATCH)
                    errj = (joint - case["delta64_joint_pool"]).abs().max().item() / scale
                    self.assertLess(errj, 1e-6)
                    worst["joint"], counts["joint"] = max(worst["joint"], errj), counts["joint"] + 1
                if len(set(ts)) == 1 and ts[0] is not None:
                    # Forge 훅의 길(Slot.add — 0 출력에 제자리로 더함)도 같은 기준에 댄다
                    def hook(xx, pooling=td.POOL_SAMPLE):
                        out = torch.zeros(tuple(xx.shape[:-1]) + (module.out_features,), dtype=xx.dtype)
                        return td.Slot(module, mult, stage).add(out, xx, ts[0], pooling, td.FusedCache(), td.PoolCache())
                    errh = (hook(x.double()) - ref64).abs().max().item() / scale
                    self.assertLess(errh, 1e-6)
                    worst["hook64"], counts["hook64"] = max(worst["hook64"], errh), counts["hook64"] + 1
                    if "delta32" in case:
                        errh = (hook(x).double() - case["delta32"].double()).abs().max().item() / scale
                        self.assertLess(errh, 2e-6)
                        worst["hook32"], counts["hook32"] = max(worst["hook32"], errh), counts["hook32"] + 1
                    if "delta_bf16" in case:
                        errh = (hook(x.bfloat16()).double() - ref64).norm().item() / (ref64.norm().item() or 1.0)
                        self.assertLess(errh, max(2e-2, 3 * refbf))
                        worst["hook_bf16"], counts["hook_bf16"] = max(worst["hook_bf16"], errh), counts["hook_bf16"] + 1
                    if "delta64_joint_pool" in case:
                        errh = (hook(x.double(), td.POOL_BATCH) - case["delta64_joint_pool"]).abs().max().item() / scale
                        self.assertLess(errh, 1e-6)
                        worst["hook_joint"], counts["hook_joint"] = max(worst["hook_joint"], errh), counts["hook_joint"] + 1
                if "gates64" in case and ts[0] is not None and module.n_ctx:
                    src = x.double()
                    if module.router_input == tf.ROUTER_BOTTLENECK:
                        idx = module.expert_indices(ts[0])[0]
                        src = src @ module.downs[0][idx].double().T
                    gates = td.gates(module, src, ts[0]).reshape(-1)
                    errg = (gates - case["gates64"].reshape(-1)).abs().max().item()
                    self.assertLess(errg, 1e-6)
                    worst["gates"], counts["gates"] = max(worst["gates"], errg), counts["gates"] + 1
        tf.release_cache()
        return len(cases), worst, counts

    def test_stage1_real_modules(self):
        n, worst, counts = self._run("stage1_subset.safetensors", "stage1_cases.pt")
        self.assertEqual((counts["delta64"], counts["delta32"], counts["delta_bf16"]), (n, n, n))
        print(f"\n  stage1 real modules: {n} cases {counts}, worst rel err {worst}")

    def test_stage2_synthetic_modules(self):
        n, worst, counts = self._run("stage2_synth.safetensors", "stage2_cases.pt")
        self.assertEqual(counts["delta64"], n)
        self.assertEqual(counts["joint"], 9)
        print(f"\n  stage2 synthetic: {n} cases {counts}, worst rel err {worst}")


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 선택: 살아 있는 LyCORIS TSMModule(+ 포크 패치)
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

def _live_lycoris():
    path = os.environ.get("SAM3_TSM_SD_SCRIPTS", "")
    if not path or not Path(path, "library", "lycoris_algo_fixes.py").is_file():
        return None
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from library import lycoris_algo_fixes as fx
        from lycoris.modules import tsm as lyco_tsm
    except Exception:
        return None
    fx.apply_module_patches()
    return lyco_tsm


@unittest.skipUnless(_live_lycoris() is not None, "set SAM3_TSM_SD_SCRIPTS to the sd-scripts fork with lycoris installed")
class LiveLycorisTests(unittest.TestCase):
    def test_trainer_set_timestep_with_bf16_timesteps_gives_the_same_t(self):
        _live_lycoris()
        from library import lycoris_algo_fixes as fx
        from lycoris.modules import tsm as lyco_tsm

        lin = nn.Linear(8, 8, bias=False)
        m = lyco_tsm.TSMModule("lora_unet_blocks_0_mlp_layer1", lin, lora_dim=2, alpha=1.0)
        net = types.SimpleNamespace(unet_loras=[m], text_encoder_loras=[])
        g = torch.Generator().manual_seed(3)
        sigmas = torch.rand(512, generator=g)
        for name, dtype in ((tf.TIMESTEP_BF16, torch.bfloat16), (tf.TIMESTEP_FP16, torch.float16), (tf.TIMESTEP_FP32, torch.float32)):
            # anima_train_network → flux_train_utils: timesteps = sigmas * 1000 (fp32) → .to(weight_dtype)
            fx.set_timestep(net, (sigmas * 1000).to(dtype))
            try:
                trained = list(fx._TSM_SAMPLE_T[0])
            finally:
                fx.clear_timestep(net)
            mine = [td.sigma_to_t(float(s), timestep_dtype=name) for s in sigmas]
            with self.subTest(dtype=name):
                self.assertEqual(mine, trained)

    def test_tsm_module_round_trip_through_the_state_dict(self):
        lyco = _live_lycoris()
        torch.manual_seed(0)
        for n_scales, router in (([8, 1], "input"), ([4, 2, 1], "input"), ([8, 1], "bottleneck")):
            lin = nn.Linear(40, 24, bias=False)
            for stage in (1, 2):
                kwargs = {"lora_dim": 4, "alpha": 2.0, "tsm_n_scales": n_scales, "tsm_stage": stage}
                if router != "input":
                    kwargs["tsm_router_input"] = router
                m = lyco.TSMModule("lora_unet_blocks_0_mlp_layer1", lin, **kwargs)
                with torch.no_grad():
                    for scale in m.experts:
                        for e in scale:
                            e["up"].weight.normal_()
                    m.router_fc.weight.normal_()
                    m.router_fc.bias.normal_()
                    m.timestep_embed.weight.normal_()
                sd = {f"lora_unet_blocks_0_mlp_layer1.{k}": v.detach().clone() for k, v in m.state_dict().items()
                      if not k.startswith("org_")}
                if "lora_unet_blocks_0_mlp_layer1.alpha" not in sd:
                    sd["lora_unet_blocks_0_mlp_layer1.alpha"] = torch.tensor(2.0)
                tsm = tf.build_tsm_file(sd, stage_meta(stage, tsm_router_input=router))
                module = tsm.modules["lora_unet_blocks_0_mlp_layer1"]
                self.assertEqual(module.router_input, router)
                m.apply_to()
                m.eval()
                try:
                    for shape in ((1, 7, 40), (1, 1, 3, 5, 40)):
                        for t in (0, 1, 125, 126, 500, 999, 1000):
                            x = torch.randn(*shape)
                            lyco.set_tsm_timestep(t)
                            with torch.no_grad():
                                ref = m(x) - torch.nn.functional.linear(x, lin.weight)
                            lyco.clear_tsm_timestep()
                            got = td.module_delta(module, x, t, stage)
                            with self.subTest(n_scales=n_scales, router=router, stage=stage, shape=shape, t=t):
                                torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
                finally:
                    m.restore()


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TSM-LoKr (algo=tsm-lokr) — 기준 식은 설계(scratchpad/tsm_lokr/DESIGN.md §3.2)를 torch.kron 과 샘플별 dense V_b 로
# 그대로 옮겨 적었다(구현의 kron4 · 세 벡터 노름 · 가중치 함수와 독립). 포크의 학습 모듈(library/tsm_lokr.py)이 생기면
# LiveTsmLokrTests 가 그 rebuild 와도 대조한다. LiveLycorisLokrTests 는 venv 의 LyCORIS LokrModule 과 대조한다.
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════

from sam3ext.tsm_lora import lokr_delta as tld  # noqa: E402
from sam3ext.tsm_lora import lokr_format as tlf  # noqa: E402

LOKR_EPS = float(torch.finfo(torch.float32).eps)


def lokr_meta(stage=1, n_scales=(8, 1), factors=(4, 4), dora="out", timestep_dtype="fp32", full_matrix=True,
              decompose_both=False, **overrides):
    config = {
        "version": 1, "n_scales": list(n_scales), "num_timesteps": 1000, "stage": stage, "router_input": "input",
        "factors": list(factors), "full_matrix": full_matrix, "decompose_both": decompose_both, "dora": dora,
        "timestep_dtype": timestep_dtype, "timestep_scale": 1000, "pooling": "sample", "multiplier": "forge-blend",
        "gate_init": 0.0,
    }
    config.update(overrides)
    args = {"algo": "tsm-lokr", "tsm_stage": str(stage), "preset": "anima-exp", "factor": "4", "full_matrix": "True",
            "dora_wd": str(dora == "out"), "wd_on_output": "True"}
    return {
        "ss_lycoris_algo": "tsm-lokr", "ss_saved_format": "tsm-lokr-native", "ss_tsm_expert": "lokr",
        "ss_tsm_lokr": json.dumps(config), "ss_network_args": json.dumps(args), "ss_mixed_precision": "bf16",
    }


def make_lokr_module_sd(name, fan_in, fan_out, n_scales=(8, 1), factors=(4, 4), dora=True, zero_router=False,
                        low_rank=None, alpha=None, dtype=torch.float32, seed=0, T=1000):
    """low_rank=r 이면 w2 를 [b, r] @ [r, d] 로(full_matrix 아님)."""
    g = torch.Generator().manual_seed(seed)
    per_scale = list(factors) if len(factors) == len(n_scales) else [factors[0]] + [factors[1]] * (len(n_scales) - 1)
    sd = {}
    for s, n in enumerate(n_scales):
        a, b = tlf.factorization(fan_out, per_scale[s])
        c, d = tlf.factorization(fan_in, per_scale[s])
        for i in range(n):
            sd[f"{name}.experts.{s}.{i}.lokr_w1"] = (torch.randn(a, c, generator=g) * 0.5).to(dtype)
            if low_rank:
                sd[f"{name}.experts.{s}.{i}.lokr_w2_a"] = (torch.randn(b, low_rank, generator=g) * 0.3).to(dtype)
                sd[f"{name}.experts.{s}.{i}.lokr_w2_b"] = (torch.randn(low_rank, d, generator=g) * 0.3).to(dtype)
            else:
                sd[f"{name}.experts.{s}.{i}.lokr_w2"] = (torch.randn(b, d, generator=g) * 0.15).to(dtype)
    n_ctx = len(n_scales) - 1
    if zero_router:
        sd[f"{name}.router_fc.weight"] = torch.zeros(n_ctx, fan_in, dtype=dtype)
        sd[f"{name}.router_fc.bias"] = torch.zeros(n_ctx, dtype=dtype)
        sd[f"{name}.timestep_embed.weight"] = torch.zeros(T, n_ctx, dtype=dtype)
    else:
        sd[f"{name}.router_fc.weight"] = (torch.randn(n_ctx, fan_in, generator=g) * 0.5).to(dtype)
        sd[f"{name}.router_fc.bias"] = (torch.randn(n_ctx, generator=g) * 0.5 + 0.8).to(dtype)
        sd[f"{name}.timestep_embed.weight"] = (torch.randn(T, n_ctx, generator=g) * 0.5).to(dtype)
    if dora:
        sd[f"{name}.dora_scale"] = (torch.rand(fan_out, 1, generator=g) + 0.5).to(torch.float32 if dtype != torch.float64 else dtype)
    sd[f"{name}.alpha"] = torch.tensor(float(alpha if alpha is not None else (low_rank or 10000)))
    return sd


def stub_lokr_sd(dit, n_scales=(8, 1), factors=(4, 4), dora=True, zero_router=False, seed=0, blocks=None, dtype=torch.float64):
    sd = {}
    table = tb.dit_module_table(dit)
    for i, name in enumerate(sorted(table)):
        split = tb.split_block_name(name)
        module = table[name]
        if split is None or not isinstance(module, nn.Linear):
            continue
        if blocks is not None and split[0] not in blocks:
            continue
        sd.update(make_lokr_module_sd(name, module.in_features, module.out_features, n_scales, factors, dora,
                                      zero_router=zero_router, seed=seed + i, dtype=dtype))
    return sd


def _oracle_expert(sd, name, s, i):
    """(K = scale·torch.kron(w1, w2)) — fp64, 저랭크는 a@b, 스케일은 LyCORIS 규칙."""
    prefix = f"{name}.experts.{s}.{i}."
    w1 = sd[prefix + "lokr_w1"].double() if prefix + "lokr_w1" in sd else sd[prefix + "lokr_w1_a"].double() @ sd[prefix + "lokr_w1_b"].double()
    if prefix + "lokr_w2" in sd:
        w2, scale = sd[prefix + "lokr_w2"].double(), 1.0
    else:
        w2 = sd[prefix + "lokr_w2_a"].double() @ sd[prefix + "lokr_w2_b"].double()
        r = sd[prefix + "lokr_w2_a"].shape[1]
        alpha = float(sd[f"{name}.alpha"])
        scale = (alpha or r) / r
    return scale * torch.kron(w1, w2)


def oracle_lokr_rows(sd, name, weight, x, ts, stage, mu=1.0, n_scales=(8, 1), pooling="sample", T=1000):
    """설계 §3.2 를 그대로: 샘플(또는 같은 t 묶음)마다 dense V 와 DoRA, Forge 블렌드, y = x W'ᵀ. 모두 fp64."""
    w0 = weight.double()
    x = x.double()
    m = sd.get(f"{name}.dora_scale")
    out = torch.empty(tuple(x.shape[:-1]) + (w0.shape[0],), dtype=torch.float64)
    rows = list(range(x.shape[0]))
    if pooling == "sample" or stage == 1:
        groups = [[r] for r in rows]
    else:
        groups = {}
        for r in rows:
            groups.setdefault(ts[r], []).append(r)
        groups = list(groups.values())
    for members in groups:
        t = int(ts[members[0]])
        idx = [oracle_expert_index(t, n, T) for n in n_scales]
        experts = [_oracle_expert(sd, name, s, i) for s, i in enumerate(idx)]
        if stage == 1 or len(n_scales) == 1:
            v = w0 + sum(experts)
        else:
            xs = x[members]
            z = xs.reshape(-1, xs.shape[-1]).mean(0)
            g = sd[f"{name}.router_fc.weight"].double() @ z + sd[f"{name}.router_fc.bias"].double() \
                + sd[f"{name}.timestep_embed.weight"].double()[min(max(t - 1, 0), T - 1)]
            v = w0 + experts[0]
            for j in range(len(n_scales) - 1):
                v = v + g[j] * experts[1 + j]
        if m is not None:
            d = v * (m.double() / (v.norm(dim=1, keepdim=True) + LOKR_EPS))
        else:
            d = v
        w = w0 + mu * (d - w0)
        out[members] = x[members] @ w.T
    return out


class ForgeLikeLinear(nn.Linear):
    """Forge ``ForgeOperations.Linear`` 의 가중치 경로만: 클래스 목록 ``weight_function`` 과 ``get_weight_and_bias``
    (backend/operations.py:86-103 — 가중치를 복제한 뒤 함수들을 차례로)."""

    weight_function = []
    bias_function = []

    def forward(self, x):
        weight, bias = self.weight, self.bias
        if weight is not None and self.weight_function:
            weight = weight.clone()
            for f in self.weight_function:
                weight = f(weight)
        return torch.nn.functional.linear(x, weight, bias)


def forge_like(dit):
    """스텁 DiT 의 nn.Linear 를 모두 :class:`ForgeLikeLinear` 로(같은 가중치)."""
    for parent in list(dit.modules()):
        for child_name, child in list(parent.named_children()):
            if type(child) is nn.Linear:
                new = ForgeLikeLinear(child.in_features, child.out_features, bias=child.bias is not None,
                                      dtype=child.weight.dtype)
                new.load_state_dict(child.state_dict())
                new.requires_grad_(False)
                setattr(parent, child_name, new)
    return dit


def _lokr_dit(forge=True, seed=0):
    torch.manual_seed(seed)
    dit = StubDiT(blocks=2).double()
    for p in dit.parameters():
        p.requires_grad_(False)
    return forge_like(dit) if forge else dit


class LokrFormatTests(unittest.TestCase):
    name = "lora_unet_blocks_0_mlp_layer1"

    def _sd(self, **kw):
        return make_lokr_module_sd(self.name, 16, 32, **kw)

    def test_parses_scales_factors_dora_and_metadata(self):
        tsm = tf.build_any_tsm_file(self._sd(), lokr_meta(1, timestep_dtype="bf16"), path="x/my_lokr.safetensors")
        self.assertIsInstance(tsm, tlf.TsmLokrFile)
        self.assertEqual((tsm.expert_kind, tsm.stage, tsm.n_scales, tsm.factors, tsm.dora), ("lokr", 1, (8, 1), (4, 4), "out"))
        self.assertEqual(tsm.timestep_dtype, tf.TIMESTEP_BF16)   # ss_tsm_lokr 에서 — ss_mixed_precision 이 아니라
        self.assertEqual(tsm.summary(), "s1 8+1 lokr f4 dora")
        self.assertEqual(tsm.name, "my_lokr")
        m = tsm.modules[self.name]
        self.assertEqual((m.kind, m.in_features, m.out_features, m.n_ctx, m.num_timesteps), ("lokr", 16, 32, 1, 1000))
        self.assertEqual(m.experts[0][3].shape, (4, 4, 8, 4))   # 32 = 4·8, 16 = 4·4
        self.assertEqual(m.experts[0][3].scale, 1.0)            # w1·w2 둘 다 꽉 찬 행렬 → 스케일 1
        self.assertEqual(m.timestep_dtype, tf.TIMESTEP_BF16)
        self.assertEqual(m.expert_indices(500), (3, 0))
        # 코어·문맥 factor 가 다르고 DoRA 가 없으면
        sd = self._sd(factors=(4, 8), dora=False)
        other = tf.build_any_tsm_file(sd, lokr_meta(2, factors=(4, 8), dora="none"))
        self.assertEqual(other.summary(), "s2 8+1 lokr f4/8 nodora")
        self.assertEqual(other.modules[self.name].experts[1][0].shape, (4, 2, 8, 8))   # factorization(16, 8) = (2, 8)
        # 저랭크 w2(full_matrix 아님): 스케일 alpha / r
        low = tf.build_any_tsm_file(self._sd(low_rank=2, alpha=1.0), lokr_meta(1, full_matrix=False))
        self.assertEqual(low.modules[self.name].experts[0][0].scale, 0.5)

    def test_lora_tsm_files_still_take_the_old_path(self):
        sd = make_module_sd("lora_unet_blocks_0_self_attn_q_proj", 8, 8)
        tsm = tf.build_any_tsm_file(sd, stage_meta(1))
        self.assertIs(type(tsm), tf.TsmFile)
        self.assertEqual((tsm.expert_kind, tsm.kind_label(), tsm.summary()), ("lora", "r2", "s1 8+1 r2"))

    def test_detection_split_and_metadata(self):
        sd = self._sd()
        plain = {"lora_unet_blocks_1_mlp_layer1.lora_down.weight": torch.zeros(2, 8), "lora_unet_blocks_1_mlp_layer1.alpha": torch.tensor(2.0)}
        mixed = dict(sd)
        mixed.update(plain)
        self.assertTrue(tf.is_tsm_state_dict(mixed))
        self.assertTrue(tlf.is_tsm_lokr_state_dict(sd))
        tsm_part, rest = tf.split_state_dict(mixed)
        self.assertEqual(set(tsm_part), set(sd))               # alpha · dora_scale · 라우터까지 TSM-LoKr 쪽
        self.assertEqual(set(rest), set(plain))
        self.assertIn(f"{self.name}.dora_scale", tsm_part)
        # 순정 LoKr 파일(전문가 없음)은 TSM 이 아니다
        lokr = {f"{self.name}.lokr_w1": torch.zeros(4, 4), f"{self.name}.lokr_w2": torch.zeros(8, 4), f"{self.name}.alpha": torch.tensor(1.0)}
        self.assertFalse(tf.is_tsm_state_dict(lokr))
        self.assertEqual(tf.split_state_dict(lokr), ({}, lokr))
        self.assertTrue(tf.metadata_says_tsm({"ss_lycoris_algo": "tsm-lokr"}))
        self.assertTrue(tf.metadata_says_tsm({"ss_saved_format": "tsm-lokr-native"}))
        self.assertTrue(tf.metadata_says_tsm({"ss_network_args": json.dumps({"algo": "tsm-lokr"})}))
        self.assertFalse(tf.metadata_says_tsm({"ss_lycoris_algo": "lokr"}))
        self.assertFalse(tf.metadata_may_hide_tsm({"ss_lycoris_algo": "lokr"}))

    def test_validation_errors_are_bilingual(self):
        sd = self._sd()
        cases = {
            "no json": ({k: v for k, v in lokr_meta(1).items() if k != "ss_tsm_lokr"}, sd, "ss_tsm_lokr"),
            "version": (lokr_meta(1, version=2), sd, "version"),
            "bottleneck": (lokr_meta(1, router_input="bottleneck"), sd, "router_input"),
            "input-axis dora": (lokr_meta(1, dora="in"), sd, "dora"),
            "timestep scale": (lokr_meta(1, timestep_scale=1), sd, "timestep_scale"),
            "multiplier": (lokr_meta(1, multiplier="lycoris"), sd, "multiplier"),
            "factor": (lokr_meta(1, factors=(8, 8)), sd, "factor"),
            "n_scales": (lokr_meta(1, n_scales=(4, 1)), sd, "n_scales"),
            "missing dora": (lokr_meta(1), {k: v for k, v in sd.items() if not k.endswith("dora_scale")}, "dora_scale"),
            "dora none": (lokr_meta(1, dora="none"), sd, "dora_scale"),
            "full_matrix": (lokr_meta(1), self._sd(low_rank=2), "full_matrix"),
            "stage": (dict(lokr_meta(1), ss_network_args=json.dumps({"algo": "tsm-lokr", "tsm_stage": "2"})), sd, "tsm_stage"),
            "algo": (dict(lokr_meta(1), ss_lycoris_algo="tsm"), sd, "algo"),
            "router": (lokr_meta(1), dict(sd, **{f"{self.name}.router_fc.weight": torch.zeros(1, 5)}), "router"),
            "kind": (dict(lokr_meta(1), ss_tsm_expert="lora"), sd, "ss_tsm_expert"),
            "format": (dict(lokr_meta(1), ss_saved_format="tsm-native"), sd, "ss_saved_format"),
        }
        for label, (meta, state, needle) in cases.items():
            with self.subTest(label), self.assertRaises(tf.TsmFormatError) as ctx:
                tf.build_any_tsm_file(state, meta)
            self.assertIn(needle, str(ctx.exception))
            self.assertIn(" / ", str(ctx.exception))
            self.assertRegex(ctx.exception.korean, "[가-힣]")
        both = dict(sd)
        both.update(make_module_sd("lora_unet_blocks_0_self_attn_q_proj", 8, 8))
        with self.assertRaises(tf.TsmFormatError) as ctx:
            tf.build_any_tsm_file(both, lokr_meta(1))
        self.assertIn("mixes", str(ctx.exception))
        self.assertEqual(tlf.parse_config(dict(lokr_meta(1, extra="x")))["unknown_keys"], ["extra"])
        # 포크는 모듈 종류별 형식을 + 로 잇는다; factors 는 [코어, 문맥] 두 개(문맥 스케일이 둘이어도)
        mixed_format = dict(lokr_meta(1), ss_saved_format="native+tsm-lokr-native")
        self.assertTrue(tf.metadata_says_tsm({"ss_saved_format": "native+tsm-lokr-native"}))
        self.assertEqual(tf.build_any_tsm_file(sd, mixed_format).stage, 1)
        three = make_lokr_module_sd(self.name, 16, 32, n_scales=(4, 2, 1), factors=(4, 8))
        self.assertEqual(tf.build_any_tsm_file(three, lokr_meta(1, n_scales=(4, 2, 1), factors=(4, 8))).factors, (4, 8, 8))

    def test_factorization_matches_lycoris_examples(self):
        # LyCORIS factorization 의 docstring 표
        table = {(127, 4): (1, 127), (128, 4): (4, 32), (250, 4): (2, 125), (360, 8): (8, 45), (1024, -1): (32, 32),
                 (512, 16): (16, 32), (2048, 4): (4, 512), (6144, 4): (4, 1536), (256, 4): (4, 64), (8192, 4): (4, 2048)}
        for (dim, factor), want in table.items():
            self.assertEqual(tlf.factorization(dim, factor), want, (dim, factor))

    def test_file_load_cache_and_listing(self):
        sd = {k: (v.to(torch.float16) if k.endswith(("lokr_w1", "lokr_w2")) else v) for k, v in self._sd().items()}
        plain = {"lora_unet_blocks_1_mlp_layer1.lora_down.weight": torch.zeros(2, 8)}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "lokr_tsm.safetensors"))
            _save_safetensors(path, dict(sd, **plain), lokr_meta(2))
            tf.release_cache()
            tsm = tf.load_tsm_file(path)
            self.assertIsInstance(tsm, tlf.TsmLokrFile)
            self.assertIs(tf.load_tsm_file(path), tsm)
            self.assertEqual(tsm.modules[self.name].experts[0][0].w1.dtype, torch.float16)
            # 메타데이터 없이도 헤더 키로 목록에 든다; 메타데이터가 tsm-lokr 라고 하면 키를 보지 않는다
            bare = str(Path(tmp, "bare.safetensors"))
            _save_safetensors(bare, sd, {})
            found = tl.find_tsm_files([("lokr_tsm", path, lokr_meta(2)), ("bare", bare, None), ("other", bare, {"ss_lycoris_algo": "lokr"})])
            self.assertEqual(found, {"bare": bare, "lokr_tsm": path})
            # 메타데이터 없는 TSM-LoKr 는 읽지 않는다(단계·시간 dtype 을 추측하지 않음)
            with self.assertRaises(tf.TsmFormatError):
                tf.load_tsm_file(bare)
        tf.release_cache()


class LokrMathTests(unittest.TestCase):
    def test_kron4_and_kron_mv(self):
        g = torch.Generator().manual_seed(0)
        w1, w2 = torch.randn(4, 3, generator=g), torch.randn(5, 7, generator=g)
        self.assertTrue(torch.equal(tld.kron4(w1, w2), torch.kron(w1, w2)))
        x = torch.randn(6, 21, dtype=torch.float64, generator=g)
        dense = x @ torch.kron(w1.double(), w2.double()).T
        torch.testing.assert_close(tld.kron_mv(x, w1.double(), w2.double()), dense, rtol=1e-12, atol=1e-12)

    def test_three_vector_norms_are_exact(self):
        g = torch.Generator().manual_seed(1)
        for n_ctx, dtype, tol in ((1, torch.float64, 1e-13), (2, torch.float64, 1e-13), (1, torch.float32, 2e-6)):
            w = torch.randn(48, 32, dtype=dtype, generator=g)
            core = (torch.randn(4, 4, generator=g).to(dtype), torch.randn(12, 8, generator=g).to(dtype) * 0.2, 1.0)
            ctx = [(torch.randn(4, 4, generator=g).to(dtype), torch.randn(12, 8, generator=g).to(dtype) * 0.2, 0.5)
                   for _ in range(n_ctx)]
            m = torch.rand(48, 1, dtype=dtype, generator=g) + 0.5
            a, b, c = tld.stage2_terms(w, core, ctx)
            gates = torch.tensor([[-3.0] * n_ctx, [0.0] * n_ctx, [2.5] * n_ctx, [0.7, -1.1][:n_ctx]], dtype=dtype)
            s = tld.stage2_row_scale(gates, a, b, c, m)
            for row in range(gates.shape[0]):
                v = w + core[2] * torch.kron(core[0], core[1])
                for j in range(n_ctx):
                    v = v + gates[row, j] * ctx[j][2] * torch.kron(ctx[j][0], ctx[j][1])
                want = m[:, 0] / (v.norm(dim=1) + LOKR_EPS)
                with self.subTest(n_ctx=n_ctx, dtype=dtype, row=row):
                    torch.testing.assert_close(s[row], want, rtol=tol, atol=0)

    def test_merged_weight_matches_the_design_formula(self):
        name = "m"
        sd = make_lokr_module_sd(name, 16, 32, n_scales=(4, 2, 1), factors=(4, 8), dtype=torch.float64, seed=4)
        meta = lokr_meta(1, n_scales=(4, 2, 1), factors=(4, 8))
        module = tf.build_any_tsm_file(sd, meta).modules[name]
        w = torch.randn(32, 16, dtype=torch.float64)
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        for mu in (1.0, 0.8, 0.0, -0.5):
            for t in (0, 1, 333, 600, 1000):
                got = tld.reference_output(module, w, x, [t] * 3, 1, mu)
                want = oracle_lokr_rows(sd, name, w, x, [t] * 3, 1, mu, n_scales=(4, 2, 1))
                with self.subTest(mu=mu, t=t):
                    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
        # μ = 0 은 정확히 W
        exact = tld.merged_weight(w, [(e.w1, e.w2, e.scale) for e in (module.experts[0][0], module.experts[1][0], module.experts[2][0])],
                                  module.dora_scale, 0.0)
        self.assertTrue(torch.equal(exact, w))

    def test_stage2_reference_matches_the_dense_per_sample_formula(self):
        name = "m"
        for n_scales, factors, pooling in (((8, 1), (4, 4), "sample"), ((4, 2, 1), (4, 4), "sample"), ((8, 1), (4, 4), "batch")):
            sd = make_lokr_module_sd(name, 16, 32, n_scales=n_scales, factors=factors, dtype=torch.float64, seed=7)
            module = tf.build_any_tsm_file(sd, lokr_meta(2, n_scales=n_scales, factors=factors)).modules[name]
            w = torch.randn(32, 16, dtype=torch.float64)
            x = torch.randn(4, 3, 16, dtype=torch.float64)
            for ts in ([420] * 4, [420, 420, 77, 900]):
                for mu in (1.0, 0.6):
                    got = tld.reference_output(module, w, x, ts, 2, mu, pooling)
                    want = oracle_lokr_rows(sd, name, w, x, ts, 2, mu, n_scales=n_scales, pooling=pooling)
                    with self.subTest(n_scales=n_scales, pooling=pooling, ts=ts, mu=mu):
                        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


class LokrRuntimeTests(unittest.TestCase):
    """스텁 DiT 위에서 세션·훅·가중치 함수 — 대상 Linear 출력마다 기준 식(oracle_lokr_rows)과 대조한다."""

    def setUp(self):
        tf.release_cache()
        self.x = torch.randn(2, 1, 2, 3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
        self.ctx = torch.randn(2, 5, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(12))

    def tearDown(self):
        tr.finish_generation()

    def _setup(self, forge=True, stage=1, n_scales=(8, 1), dora=True, seed=3):
        dit = _lokr_dit(forge)
        sd = stub_lokr_sd(dit, n_scales=n_scales, dora=dora, seed=seed)
        tsm = tf.build_any_tsm_file(sd, lokr_meta(stage, n_scales=n_scales, dora="out" if dora else "none"))
        return dit, sd, tsm

    def _session(self, dit, tsm, mult=0.8, stage=None, pooling=td.POOL_SAMPLE, merge_cache=False, extra=()):
        entry = tr.TsmEntry(tsm, mult, tr.SOURCE_PICKER, stage=stage or tsm.stage, label="lokr")
        session = tr.TsmSession([entry, *extra], pooling, lokr_merge_cache=merge_cache)
        report = session.bind(dit, alb.DUPLICATE_KEEP)
        self.assertEqual(report.refused, [])
        tr.HOOKS.attach(dit, session)
        tr.remember_session(session)
        return session

    def _run(self, dit, session, sigmas=(0.4321, 0.4321)):
        outputs = {}
        handles = []
        for name, module in dit.named_modules():
            if isinstance(module, nn.Linear):
                def hook(mod, args, out, name=name):
                    outputs[name] = (args[0].detach().clone(), out.detach().clone())
                handles.append(module.register_forward_hook(hook))
        try:
            final = dit(self.x, torch.tensor(list(sigmas), dtype=torch.float32), context=self.ctx,
                        transformer_options={tr.OPTION_KEY: session})
        finally:
            for h in handles:
                h.remove()
        return outputs, final

    def _check_all(self, dit, sd, tsm, outputs, ts, stage, mu, n_scales=(8, 1), pooling="sample", tol=1e-10):
        checked = 0
        for path, (inp, out) in outputs.items():
            name = tb.module_key(path)
            if name not in tsm.modules:
                continue
            weight = dit.get_submodule(path).weight
            want = oracle_lokr_rows(sd, name, weight, inp, ts, stage, mu, n_scales=n_scales, pooling=pooling)
            torch.testing.assert_close(out, want, rtol=tol, atol=tol, msg=name)
            checked += 1
        self.assertEqual(checked, 32)

    def test_every_linear_matches_the_reference_in_both_paths(self):
        for forge in (True, False):
            for stage, n_scales in ((1, (8, 1)), (2, (8, 1)), (2, (4, 2, 1)), (1, (4, 2, 1))):
                for mu in (1.0, 0.8):
                    with self.subTest(forge=forge, stage=stage, n_scales=n_scales, mu=mu):
                        dit, sd, tsm = self._setup(forge, stage, n_scales)
                        session = self._session(dit, tsm, mu)
                        mode = tld.MODE_WEIGHT_FN if forge else tld.MODE_HOOK
                        self.assertEqual(session.report.lokr_modes, {mode: 32})
                        self.assertEqual(tr.HOOKS.pre_hook_count(), 32 if forge else 0)
                        outputs, _ = self._run(dit, session)
                        t = td.sigma_to_t(0.4321)
                        self._check_all(dit, sd, tsm, outputs, [t, t], stage, mu, n_scales)
                        self.assertEqual(session.skipped_calls, 0)
                        tr.finish_generation()

    def test_mixed_t_rows_and_batch_pooling(self):
        for forge in (True, False):
            for stage in (1, 2):
                with self.subTest(forge=forge, stage=stage):
                    dit, sd, tsm = self._setup(forge, stage)
                    session = self._session(dit, tsm, 0.9)
                    outputs, _ = self._run(dit, session, sigmas=(0.9, 0.2))
                    ts = [td.sigma_to_t(0.9), td.sigma_to_t(0.2)]
                    self._check_all(dit, sd, tsm, outputs, ts, stage, 0.9)
                    tr.finish_generation()
                    if stage == 2:
                        session = self._session(dit, tsm, 1.0, pooling=td.POOL_BATCH)
                        outputs, _ = self._run(dit, session)
                        t = td.sigma_to_t(0.4321)
                        self._check_all(dit, sd, tsm, outputs, [t, t], 2, 1.0, pooling="batch")
                        tr.finish_generation()

    def test_stage1_file_through_the_stage2_formula_and_no_dora(self):
        dit, sd, tsm = self._setup(True, 1, dora=False)
        session = self._session(dit, tsm, 0.7, stage=2)
        outputs, _ = self._run(dit, session)
        t = td.sigma_to_t(0.4321)
        self._check_all(dit, sd, tsm, outputs, [t, t], 2, 0.7)

    def test_bf16_training_timesteps_and_bf16_compute(self):
        dit = _lokr_dit(True)
        sd = stub_lokr_sd(dit, seed=5)
        tsm = tf.build_any_tsm_file(sd, lokr_meta(1, timestep_dtype="bf16"))
        session = self._session(dit, tsm, 1.0)
        sigma = 0.6137                                     # σ·1000 = 613.7 → fp32 t 614, bf16 612
        outputs, _ = self._run(dit, session, sigmas=(sigma, sigma))
        t = td.sigma_to_t(sigma, timestep_dtype=tf.TIMESTEP_BF16)
        self.assertNotEqual(t, td.sigma_to_t(sigma))       # 512 이상: bf16 반올림으로 다른 t
        self.assertEqual(session.seen_t, {t})
        self._check_all(dit, sd, tsm, outputs, [t, t], 1, 1.0)
        tr.finish_generation()
        # bf16 계산: 가중치 함수 길과 출력 훅 길이 기준(fp64)에서 bf16 반올림 몇 번 안쪽
        for forge in (True, False):
            dit = _lokr_dit(forge).to(torch.bfloat16)
            session = self._session(dit, tsm, 0.8)
            linear = dit.blocks[0].mlp.layer1
            x = torch.randn(2, 64, 16, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
            call = session.begin_call([400.0, 400.0])
            tr._ACTIVE.session, tr._ACTIVE.call = session, call
            try:
                got = linear(x)
            finally:
                tr._ACTIVE.session, tr._ACTIVE.call = None, None
            want = oracle_lokr_rows(sd, "lora_unet_blocks_0_mlp_layer1", linear.weight, x, [400, 400], 1, 0.8)
            self.assertEqual(got.dtype, torch.bfloat16)
            err = ((got.double() - want).abs().max() / want.abs().max()).item()
            with self.subTest(forge=forge):
                self.assertLess(err, 2e-2)
            tr.finish_generation()

    def test_weight_function_install_and_remove_leave_forge_lists_alone(self):
        dit, sd, tsm = self._setup(True, 1)
        linear = dit.blocks[0].mlp.layer1
        other = dit.blocks[0].mlp.layer2
        forge_fn = lambda w: w * 1.0  # noqa: E731 — Forge 의 LowVramPatch 자리
        other.weight_function = [forge_fn]
        session = self._session(dit, tsm)
        self.assertNotIn("weight_function", linear.__dict__)       # 호출 전에는 아무것도 붙이지 않는다
        self._run(dit, session)
        self.assertEqual(ForgeLikeLinear.weight_function, [])        # 클래스 목록은 그대로
        self.assertEqual(len(linear.__dict__["weight_function"]), 1)
        self.assertEqual(other.weight_function[0], forge_fn)
        self.assertTrue(getattr(other.weight_function[-1], "_sam3_tsm_lokr", False))
        ours = linear.weight_function[0]
        w = torch.ones(2, 2)
        self.assertIs(ours(w), w)                                     # 세션 없이 부르면 받은 것 그대로
        # Forge 가 패스 사이에 목록을 갈아 끼워도(wipe_lowvram_weight) 다음 호출이 다시 붙이고 값도 같다
        first, _ = self._run(dit, session)
        linear.weight_function = []
        again, _ = self._run(dit, session)
        path = "blocks.0.mlp.layer1"
        torch.testing.assert_close(again[path][1], first[path][1], rtol=0, atol=0)
        # 놓으면 우리 함수만 빠진다
        self.assertTrue(tr.release_device_caches())
        self.assertEqual(linear.weight_function, [])                 # 우리 것만 빠졌다(남은 것 없음 → 클래스 목록)
        self.assertEqual(other.weight_function, [forge_fn])
        self.assertNotIn("weight_function", dit.blocks[1].mlp.layer1.__dict__)   # 원래 인스턴스 목록이 없던 것은 지운다
        tr.finish_generation()
        self.assertEqual(tr.HOOKS.pre_hook_count(), 0)
        self.assertEqual(len(tr.WEIGHT_FNS), 0)
        # 훅을 뗀 뒤에는 순정 출력
        base = dit(self.x, torch.tensor([0.4, 0.4]), context=self.ctx, transformer_options={})
        torch.testing.assert_close(dit(self.x, torch.tensor([0.4, 0.4]), context=self.ctx,
                                       transformer_options={tr.OPTION_KEY: session}), base, rtol=0, atol=0)

    def test_merge_cache_gives_the_same_outputs_and_holds_one_interval(self):
        dit, sd, tsm = self._setup(True, 1)
        plain_session = self._session(dit, tsm)
        plain, _ = self._run(dit, plain_session)
        tr.finish_generation()
        session = self._session(dit, tsm, merge_cache=True)
        cached, _ = self._run(dit, session)
        for path in plain:
            torch.testing.assert_close(cached[path][1], plain[path][1], rtol=0, atol=0)
        slots = [slot for slots in session.lokr_slots.values() for slot in slots]
        weight_bytes = sum(s.module.merged_bytes(8) for s in slots)
        self.assertEqual(sum(s._merged[1].numel() * 8 for s in slots), weight_bytes)
        keys = {s._merged[0][0] for s in slots}
        self.assertEqual(len(keys), 1)
        self._run(dit, session)                                    # 같은 구간 — 캐시를 그대로
        self._run(dit, session, sigmas=(0.05, 0.05))               # 다른 구간 — 갈아 끼우고 하나만 쥔다
        self.assertEqual({s._merged[0][0] for s in slots} - keys, {s._merged[0][0] for s in slots})
        self.assertEqual(sum(s.nbytes() for s in slots), weight_bytes)
        self.assertGreaterEqual(session.fused_bytes(8), weight_bytes)
        # 다른 가중치 함수가 앞에 있으면 그 모듈은 캐시를 쓰지 않는다(Forge 가 넘기는 W 가 이 모듈 가중치가 아님)
        dit.blocks[0].mlp.layer1.weight_function = [lambda w: w]
        self._run(dit, session)
        slot = session.lokr_slots[id(dit.blocks[0].mlp.layer1)][0]
        self.assertFalse(slot._cache_ok)

    def test_device_cache_holds_only_the_current_interval(self):
        for stage in (1, 2):
            dit, sd, tsm = self._setup(True, stage)
            session = self._session(dit, tsm)
            self._run(dit, session)
            modules = list(tsm.modules.values())
            experts = sum(m.interval_bytes() for m in modules)
            dora = sum(m.out_features * 8 for m in modules)
            # fp64 스텁: 라우터는 입력 dtype(fp64)으로, DoRA 크기는 파일 dtype(fp64)으로 올라간다(실제 bf16 생성은 fp32)
            routers = sum((m.router_w.numel() + m.router_b.numel() + m.embed.numel()) * 8 for m in modules) if stage == 2 else 0
            terms = sum(m.out_features * 8 * 3 for m in modules) if stage == 2 else 0   # fp64 스텁 → a·b·c 한 구간
            with self.subTest(stage=stage):
                self.assertEqual(session.lokr_cache.nbytes(), experts + dora + routers)
                self.assertEqual(session.device_bytes(), experts + dora + routers + terms)
                self._run(dit, session, sigmas=(0.05, 0.05))         # 구간이 바뀌어도 코어 전문가는 하나
                self.assertEqual(session.lokr_cache.nbytes(), experts + dora + routers)
            tr.finish_generation()

    def test_release_before_decode_drops_every_device_reference(self):
        import gc
        import weakref

        dit, sd, tsm = self._setup(True, 2)
        session = self._session(dit, tsm)
        session.targets.clear()     # apply_tsm 처럼
        held_during = []
        refs = []

        def step(i):
            sigma = 0.9 * (1 - i / 3)                      # 0.9 · 0.6 · 0.3 — 스텝마다 다른 코어 구간
            dit(self.x, torch.tensor([sigma, sigma]), context=self.ctx, transformer_options={tr.OPTION_KEY: session})
            held_during.append(session.device_bytes())
            if not refs:
                # CPU 에서 같은 dtype 의 "업로드"는 파일 텐서 그 자체라 전문가 사본은 약한 참조로 볼 수 없다 — 새로 셈한 노름 항으로
                slot = next(iter(session.lokr_slots.values()))[0]
                refs.extend(weakref.ref(t) for t in next(iter(slot._terms.values())))

        sampler = FakeForgeSampler(step, [])
        self.assertEqual(tr.wrap_sampler_release(sampler), "installed")
        sampler.sample(None)
        self.assertEqual(len(held_during), 3)
        self.assertTrue(all(b > 0 for b in held_during))   # 스텝 동안은 장치 사본이 있다
        gc.collect()
        self.assertTrue(refs and all(r() is None for r in refs))      # 디코드 자리에서 장치 사본이 하나도 없다
        self.assertEqual(session.device_bytes(), 0)
        self.assertEqual((session.lokr_cache._experts, session.lokr_cache._dora, session.lokr_cache._routers), ({}, {}, {}))
        self.assertFalse(any(slot.holds_device_tensors() for slots in session.lokr_slots.values() for slot in slots))
        self.assertFalse(any(tr.WEIGHT_FNS.installed(m) for m in dit.modules() if isinstance(m, ForgeLikeLinear)))
        # 다음 패스(하이레스)는 다시 만들고 값이 같다
        first, _ = self._run(dit, session)
        tr.release_device_caches()
        again, _ = self._run(dit, session)
        for path in first:
            torch.testing.assert_close(again[path][1], first[path][1], rtol=0, atol=0)

    def test_lora_tsm_and_tsm_lokr_on_the_same_linear_add_up(self):
        dit, sd, tsm = self._setup(True, 1)
        lora_sd = stub_tsm_sd(dit, n_scales=(8, 1), rank=2, seed=21)
        lora = tf.build_tsm_file(lora_sd, stage_meta(1))
        session = self._session(dit, tsm, 0.8, extra=[tr.TsmEntry(lora, 0.5, tr.SOURCE_PROMPT, stage=1, label="lora")])
        outputs, _ = self._run(dit, session)
        t = td.sigma_to_t(0.4321)
        for path, (inp, out) in outputs.items():
            name = tb.module_key(path)
            if name not in tsm.modules:
                continue
            weight = dit.get_submodule(path).weight
            want = oracle_lokr_rows(sd, name, weight, inp, [t, t], 1, 0.8) + oracle_delta(lora_sd, name, inp, [t, t], 1, 0.5)
            torch.testing.assert_close(out, want, rtol=1e-10, atol=1e-10, msg=name)
        # 같은 모듈에 TSM-LoKr 둘은 거부(한 Linear 에 하나)
        second = tf.build_any_tsm_file(stub_lokr_sd(dit, seed=40), lokr_meta(1))
        both = tr.TsmSession([tr.TsmEntry(tsm, 1.0, tr.SOURCE_PICKER, 1, "a"), tr.TsmEntry(second, 1.0, tr.SOURCE_PROMPT, 1, "b")])
        report = both.bind(dit)
        self.assertEqual([name for name, _ in report.refused], [second.name])
        self.assertEqual(len(both.lokr_slots), 32)
        self.assertIsNone(both.delta_for(id(dit.blocks[0].mlp.layer1), inp, [400.0, 400.0]))

    def test_unsupported_linear_and_transient_bytes(self):
        dit, sd, tsm = self._setup(True, 1)

        class Opaque(nn.Module):
            """GGUF 처럼 실수 가중치도 weight_function 도 없는 Linear."""
            in_features, out_features = 16, 32

            def forward(self, x):
                return x.new_zeros(*x.shape[:-1], 32)

        dit.blocks[0].mlp.layer1 = Opaque()
        session = tr.TsmSession([tr.TsmEntry(tsm, 1.0, tr.SOURCE_PICKER, 1, "lokr")])
        report = session.bind(dit)
        self.assertEqual(report.unsupported, 1)
        self.assertEqual(len(session.lokr_slots), 31)
        self.assertTrue(any("no float weight" in note for note in report.notes))
        # 가중치 쪽(fp32 세 개 + 결과 — 가장 큰 가중치는 mlp 32x16)과 출력 쪽(fp32 누산기 + 항 + 새 출력, 가장 큰 출력은 adaLN 48)
        self.assertEqual(session.transient_bytes(10, 2), max(32 * 16 * (12 + 2), 10 * 48 * (4 + 2 * 2)))
        self.assertEqual(session.transient_bytes(1000, 2), 1000 * 48 * (4 + 2 * 2))

    def test_40_to_52_dora_policy_for_inserted_blocks(self):
        names = [f"lora_unet_blocks_{i}_mlp_layer1" for i in range(40)]
        sd = {}
        for i, name in enumerate(names):
            sd.update(make_lokr_module_sd(name, 16, 32, seed=i))
        tsm = tf.build_any_tsm_file(sd, lokr_meta(1))
        mapping = alb.BLOCK_MAPPINGS[(40, 52)]
        inserted = {target for target, source in enumerate(mapping) if source in mapping[:target]}
        self.assertEqual(len(inserted), 12)
        keep = tb.plan_file(tsm, 52, alb.DUPLICATE_KEEP)
        additive = tb.plan_file(tsm, 52, alb.DUPLICATE_ADDITIVE)
        weak = tb.plan_file(tsm, 52, alb.DUPLICATE_WEAK, (0.3, alb.WEAK_SCOPE_ALL))
        for target, source in enumerate(mapping):
            key = f"lora_unet_blocks_{target}_mlp_layer1"
            original = tsm.modules[f"lora_unet_blocks_{source}_mlp_layer1"]
            self.assertIs(keep.targets[key][0][0], original)
            module, factor = additive.targets[key][0]
            if target in inserted:
                self.assertIsNone(module.dora_scale)                   # additive: 크기 없이 W + μ·ΔW
                self.assertIs(module.root, original)                    # 텐서·장치 사본은 원래 모듈의 것
                self.assertIs(module.experts, original.experts)
                self.assertEqual(weak.targets[key][0][1], 0.3)
                self.assertIsNone(weak.targets[key][0][0].dora_scale)
            else:
                self.assertIs(module, original)

    def test_loader_hook_records_tsm_lokr_and_passes_the_rest(self):
        calls = []

        def load_lora_for_models(model, clip, lora, strength_model, strength_clip, filename="default", online_mode=False):
            calls.append(dict(lora))
            return "patched-model", clip

        module = types.ModuleType("networks")
        module.load_lora_for_models = load_lora_for_models
        module.dynamic_args = types.SimpleNamespace(anima=True)
        sd_model = types.SimpleNamespace(current_lora_hash="h1")
        shared = types.SimpleNamespace(sd_model=sd_model)
        modules_pkg = types.ModuleType("modules")
        modules_pkg.shared = shared
        logs = []
        tr.set_logger(logs.append)
        try:
            tr.install_loader_hook(module)
            sd = make_lokr_module_sd("lora_unet_blocks_0_mlp_layer1", 16, 32)
            plain = {"lora_unet_blocks_1_mlp_layer1.lora_down.weight": torch.zeros(2, 16)}
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp, "a_lokr.safetensors"))
                _save_safetensors(path, dict(sd, **plain), lokr_meta(1))
                with _SysModules(modules=modules_pkg, modules__shared=shared):
                    result = module.load_lora_for_models("model", "clip", dict(sd, **plain), 0.6, 1.0, path, False)
            self.assertEqual(result, ("patched-model", "clip"))
            self.assertEqual(set(calls[-1]), set(plain))                 # alpha·dora_scale·라우터는 Forge 가 보지 않는다
            record = tr.records_for(sd_model)[0]
            self.assertEqual((record["error"], record["strength"]), ("", 0.6))
            self.assertIsInstance(record["tsm"], tlf.TsmLokrFile)
            self.assertTrue(any("lokr f4 dora" in line for line in logs))
        finally:
            tr.uninstall_loader_hook(module)
            tr.set_logger(None)
            tf.release_cache()


# ── 선택: venv 의 LyCORIS LokrModule · 포크의 TSM-LoKr rebuild 와 대조 ──

def _lycoris_lokr():
    try:
        from lycoris.modules.lokr import LokrModule
    except Exception:
        return None
    return LokrModule


@unittest.skipUnless(_lycoris_lokr() is not None, "LyCORIS is not installed in this venv (use the sd-scripts fork venv)")
class LiveLycorisLokrTests(unittest.TestCase):
    def test_one_expert_with_a_zero_context_equals_stock_lokr_dora(self):
        """설계 T3 의 추론 쪽: 문맥 w2 = 0 이면 TSM-LoKr 병합 가중치 = 순정 LokrModule.get_merged_weight(μ=1)."""
        LokrModule = _lycoris_lokr()
        torch.manual_seed(0)
        for fan_in, fan_out in ((64, 128), (2048, 256)):
            lin = nn.Linear(fan_in, fan_out, bias=False)
            stock = LokrModule("lora_unet_blocks_0_mlp_layer1", lin, 1.0, 10000, 10000, factor=4, full_matrix=True,
                               weight_decompose=True, wd_on_output=True)
            with torch.no_grad():
                stock.lokr_w1.normal_()
                stock.lokr_w2.normal_(std=0.05)
                stock.dora_scale.mul_(1.3)
            ref = stock.get_merged_weight(multiplier=1)[0]
            zero = torch.zeros_like(stock.lokr_w2)
            got = tld.merged_weight(lin.weight.detach(), [(stock.lokr_w1.detach(), stock.lokr_w2.detach(), stock.scale),
                                                          (stock.lokr_w1.detach(), zero, 1.0)], stock.dora_scale.detach())
            ulp = torch.finfo(torch.float32).eps
            with self.subTest(shape=(fan_out, fan_in)):
                self.assertEqual(stock.scale, 1.0)
                torch.testing.assert_close(got, ref.detach(), rtol=2 * ulp, atol=0)


def _fork_tsm_lokr():
    path = os.environ.get("SAM3_TSM_SD_SCRIPTS", "")
    if not path or not Path(path, "library", "tsm_lokr.py").is_file():
        return None
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from library import tsm_lokr
        from lycoris.modules import tsm as _lyco_tsm  # noqa: F401
    except Exception:
        return None
    return tsm_lokr if hasattr(tsm_lokr, "get_module_class") and hasattr(tsm_lokr, "metadata_json") else None


@unittest.skipUnless(_fork_tsm_lokr() is not None, "set SAM3_TSM_SD_SCRIPTS to a fork with library/tsm_lokr.py")
class LiveTsmLokrTests(unittest.TestCase):
    """포크의 학습 모듈(library/tsm_lokr.py TSMLokrModule)이 만든 state dict·메타데이터를 그대로 읽어, 같은 x·t 에서
    학습 forward 와 런타임(가중치 함수 길·출력 훅 길·기준 구현)을 대조한다."""

    name = "lora_unet_blocks_0_mlp_layer1"

    def _module(self, stage, n_scales="8,1", factor="4", dora=True, seed=0, fan_in=64, fan_out=128):
        fork = _fork_tsm_lokr()
        cfg = fork.parse_config({"tsm_n_scales": n_scales, "factor": factor, "full_matrix": "True",
                                 "dora_wd": str(dora), "tsm_stage": str(stage)})
        fork.set_active_config(cfg)
        try:
            torch.manual_seed(seed)
            lin = nn.Linear(fan_in, fan_out, bias=False)
            module = fork.get_module_class()(self.name, lin, 1.0, 10000, 10000)
        finally:
            fork.set_active_config(None)
        with torch.no_grad():
            for scale in module.experts:
                for expert in scale:
                    expert.lokr_w1.normal_()
                    expert.lokr_w2.normal_(std=0.05)
            if module.dora_scale is not None:
                module.dora_scale.mul_(1.2)
            module.router_fc.weight.normal_(std=0.3)
            module.router_fc.bias.normal_(mean=0.8, std=0.3)
            module.timestep_embed.weight.normal_(std=0.3)
        sd = {f"{self.name}.{k}": v.detach().clone() for k, v in module.state_dict().items()}
        meta = {"ss_lycoris_algo": "tsm-lokr", "ss_saved_format": "tsm-lokr-native", "ss_tsm_expert": "lokr",
                "ss_lycoris_algo_fixes": "1", "ss_tsm_lokr": fork.metadata_json(module, "fp32")}
        return fork, lin, module, tf.build_any_tsm_file(sd, meta).modules[self.name]

    def _train_forward(self, module, x, t, mu):
        from lycoris.modules import tsm as lyco_tsm

        module.multiplier = mu
        module.apply_to()
        module.eval()
        lyco_tsm.set_tsm_timestep(t)
        try:
            with torch.no_grad():
                return module(x)
        finally:
            lyco_tsm.clear_tsm_timestep()
            module.restore()

    def test_runtime_paths_equal_the_training_forward(self):
        for stage, n_scales, dora in ((1, "8,1", True), (2, "8,1", True), (2, "4,2,1", True), (1, "8,1", False), (2, "8,1", False)):
            fork, lin, module, mine = self._module(stage, n_scales, dora=dora)
            self.assertEqual(mine.n_scales, tuple(int(n) for n in n_scales.split(",")))
            x = torch.randn(2, 5, 64, generator=torch.Generator().manual_seed(4))
            for t in (1, 377, 1000):
                for mu in (1.0, 0.8):
                    ref = self._train_forward(module, x, t, mu)
                    w = lin.weight.detach()
                    cache = tld.LokrCache()
                    hook = tld.LokrSlot(mine, mu, stage, mode=tld.MODE_HOOK)
                    call = object()
                    via_hook = hook.finish(torch.nn.functional.linear(x, w), x, call, t, None, td.POOL_SAMPLE, cache, None, False, lin)
                    fn = tld.LokrSlot(mine, mu, stage, mode=tld.MODE_WEIGHT_FN)
                    merged = fn.weight_for(w.clone(), mine.expert_indices(t), cache)
                    fn.note_weight(call, None)
                    via_fn = fn.finish(torch.nn.functional.linear(x, merged), x, call, t, None, td.POOL_SAMPLE, cache, None, False, lin)
                    via_ref = tld.reference_output(mine, w, x, [t, t], stage, mu)
                    with self.subTest(stage=stage, n_scales=n_scales, dora=dora, t=t, mu=mu):
                        if stage == 1:
                            self.assertTrue(torch.equal(via_fn, ref))   # 같은 rebuild 식 → 비트 단위로 같다
                        for got in (via_fn, via_hook, via_ref):
                            torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-6)

    def test_merged_weight_equals_the_training_rebuild_bitwise(self):
        fork = _fork_tsm_lokr()
        g = torch.Generator().manual_seed(2)
        w0 = (torch.randn(256, 128, generator=g) * 0.05).to(torch.bfloat16)
        w1c, w1x = torch.randn(4, 4, generator=g), torch.randn(4, 4, generator=g)
        w2c, w2x = torch.randn(64, 32, generator=g) * 0.01, torch.randn(64, 32, generator=g) * 0.01
        m = torch.rand(256, 1, generator=g) + 0.5
        eps = torch.tensor(LOKR_EPS)
        for mu in (1.0, 0.7):
            ref = fork._rebuild_tsm_lokr(w0, m, eps, mu, torch.bfloat16, (1.0, 1.0), w1c, w2c, w1x, w2x)
            got = tld.merged_weight(w0, [(w1c, w2c, 1.0), (w1x, w2x, 1.0)], m, mu)
            with self.subTest(mu=mu):
                self.assertTrue(torch.equal(got, ref))


@unittest.skipUnless(_fork_tsm_lokr() is not None, "set SAM3_TSM_SD_SCRIPTS to a fork with library/tsm_lokr.py")
class LiveTsmLokrFileTests(unittest.TestCase):
    """처음부터 끝까지: 포크의 실제 저장 길(network.save_weights · fp16 · 포크가 쓰는 메타데이터) → load_tsm_file → 스텁 DiT
    위의 런타임(가중치 함수 길). 대상 Linear 마다 출력이 같은 입력·같은 t 의 포크 TSMLokrModule forward 와 같아야 한다
    (bf16 학습 timestep: 메타데이터 timestep_dtype 이 bf16 이고 σ→t 가 학습기와 같은 정수를 낸다)."""

    PRESET = {"enable_conv": False, "unet_target_module": ["StubBlock"], "unet_target_name": [],
              "text_encoder_target_module": [], "text_encoder_target_name": [], "exclude_name": [],
              "module_algo_map": {}, "name_algo_map": {}}

    def tearDown(self):
        tr.finish_generation()

    def _net(self, dit, stage):
        import argparse
        import logging

        from library import lycoris_algo_fixes as laf
        from lycoris import kohya

        laf.apply_module_patches()
        raw = {"algo": "tsm-lokr", "factor": "4", "full_matrix": "True", "dora_wd": "True", "wd_on_output": "True",
               "tsm_n_scales": "8,1", "tsm_stage": str(stage)}
        log = logging.getLogger("live-tsm-lokr")
        kw = laf.prepare_network_kwargs("tsm-lokr", dict(raw), log)
        net = kohya.create_network(1.0, 10000, 10000, None, [], dit, preset=dict(self.PRESET), **kw)
        net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
        args = argparse.Namespace(network_args=[f"{k}={v}" for k, v in raw.items()], network_dim=10000,
                                  network_alpha=10000, lycoris_algo_fixes=True, mixed_precision="bf16")
        laf.after_network_created(net, args, log)
        net.prepare_grad_etc(None, dit)
        return laf, net, raw

    def _check(self, stage, mu):
        import copy

        torch.manual_seed(0)
        base = StubDiT(blocks=2)
        for p in base.parameters():
            p.requires_grad_(False)
        laf, net, raw = self._net(copy.deepcopy(base), stage)
        g = torch.Generator().manual_seed(7 + stage)
        with torch.no_grad():
            for n, p in net.named_parameters():
                if "lokr_w1" in n:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.7)
                elif "lokr_w2" in n:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.05)
                elif "dora_scale" in n:
                    p.mul_(1 + 0.2 * torch.randn(p.shape, generator=g))
                elif "router_fc.weight" in n:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.3)
                elif "router_fc.bias" in n:
                    p.copy_(0.8 + 0.3 * torch.randn(p.shape, generator=g))
                elif "timestep_embed" in n:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        laf.set_timestep(net, torch.tensor([512.0]).bfloat16())  # a training call: the trainer's t dtype
        laf.clear_timestep(net)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "live.safetensors")
            net.save_weights(path, torch.float16, {"ss_network_args": json.dumps(raw), "ss_mixed_precision": "bf16"})
            tsm = tf.load_tsm_file(path)
            _, ref_net, _ = self._net(copy.deepcopy(base), stage)
            info = ref_net.load_weights(path)  # the training module from the same fp16 file
        self.assertIsInstance(tsm, tlf.TsmLokrFile)
        self.assertEqual((tsm.stage, tsm.timestep_dtype, tsm.n_scales), (stage, tf.TIMESTEP_BF16, (8, 1)))
        self.assertFalse(info.get("missing keys") or info.get("unexpected keys"), info)
        mods = {m.lora_name: m for m in laf._all_loras(ref_net)}
        self.assertEqual(set(tsm.modules), set(mods))
        dit = forge_like(copy.deepcopy(base))
        session = tr.TsmSession([tr.TsmEntry(tsm, mu, tr.SOURCE_PICKER, stage=stage, label="live")], td.POOL_SAMPLE)
        self.assertEqual(session.bind(dit, alb.DUPLICATE_KEEP).refused, [])
        tr.HOOKS.attach(dit, session)
        tr.remember_session(session)
        for sigmas in ((0.4321, 0.4321), (0.88, 0.13)):
            captured, handles = {}, []
            for path_, module in dit.named_modules():
                if isinstance(module, nn.Linear):
                    def hook(mod, args, out, path_=path_):
                        captured[path_] = (args[0].detach().clone(), out.detach().clone())
                    handles.append(module.register_forward_hook(hook))
            x = torch.randn(2, 1, 2, 3, 16, generator=g)
            ctx = torch.randn(2, 5, 8, generator=g)
            try:
                with torch.no_grad():
                    dit(x, torch.tensor(list(sigmas), dtype=torch.float32), context=ctx,
                        transformer_options={tr.OPTION_KEY: session})
            finally:
                for h in handles:
                    h.remove()
            self.assertEqual(session.skipped_calls, 0)
            train_ts = (torch.tensor(list(sigmas), dtype=torch.float32) * 1000.0).to(torch.bfloat16)
            self.assertEqual([round(float(v)) for v in train_ts.float()], [td.sigma_to_t(s, 1000, 1.0, "bf16") for s in sigmas])
            checked = 0
            for path_, (inp, out) in captured.items():
                name = tb.module_key(path_)
                if name not in mods:
                    continue
                module = mods[name]
                module.eval()
                module.multiplier = mu
                laf.set_timestep(ref_net, train_ts)
                try:
                    with torch.no_grad():
                        want = module(inp)
                finally:
                    laf.clear_timestep(ref_net)
                    module.multiplier = 1.0
                with self.subTest(stage=stage, mu=mu, sigmas=sigmas, module=name):
                    if mu == 1.0 and sigmas[0] == sigmas[1]:
                        self.assertTrue(torch.equal(out, want))  # the same rebuild / three-vector formula
                    else:
                        torch.testing.assert_close(out, want, rtol=2e-6, atol=2e-7)
                checked += 1
            self.assertEqual(checked, len(mods))

    def test_fork_saved_file_runs_like_the_training_module(self):
        for stage in (1, 2):
            for mu in (1.0, 0.7):
                self._check(stage, mu)
                tr.finish_generation()


if __name__ == "__main__":
    unittest.main()
