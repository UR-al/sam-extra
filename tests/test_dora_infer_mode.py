"""DoRA 추론 방식 — LyCORIS 공식·Forge 공식 fp32 계산, 어댑터 훅, 캐시 무효화, 스크립트 모드 전환.

기준 공식은 두 원본을 옮겨 적었다(둘 다 이 테스트 환경에는 없다).
- Forge/Comfy: modules_forge/packages/comfy/weight_adapter/base.py:40-58 weight_decompose
- LyCORIS 3.2: lycoris/modules/lokr.py get_merged_weight + apply_weight_decompose (_dora_norm_dims, _dora_eps)

Forge 모듈(backend.*, modules_forge.*)은 import 하지 않는다 — backend.memory_management 는 import 만으로 CUDA 를
조회한다. 필요한 것은 sys.modules 스텁으로 넣는다.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import anima_lora_blocks as alb  # noqa: E402
from sam3ext import dora_infer_mode as dim  # noqa: E402


def forge_weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function):
    """Forge base.py:40-58 그대로(cast_to_device → .to)."""
    dora_scale = dora_scale.to(device=weight.device, dtype=intermediate_dtype)
    lora_diff *= alpha
    weight_calc = weight + function(lora_diff).type(weight.dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = weight.reshape(weight.shape[0], -1).norm(dim=1, keepdim=True).reshape(weight.shape[0], *[1] * (weight.dim() - 1))
    else:
        weight_norm = weight_calc.transpose(0, 1).reshape(weight_calc.shape[1], -1).norm(dim=1, keepdim=True).reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1)).transpose(0, 1)
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight


def lycoris_merged_weight(w0, diff, dora_scale, wd_on_output):
    """LyCORIS lokr.py: multiplier 1 의 get_merged_weight(W₀+ΔW 를 apply_weight_decompose)."""
    weight = (w0 + diff).to(dora_scale.dtype)
    ndim = weight.dim()
    dims = tuple(range(1, ndim)) if wd_on_output else (0,) + tuple(range(2, ndim))
    norm = torch.linalg.vector_norm(weight, dim=dims, keepdim=True) + torch.tensor(torch.finfo(torch.float32).eps)
    scale = dora_scale / norm
    scale = 1.0 * (scale - 1) + 1
    return weight * scale


def _case(out=24, inp=16, seed=0, wd_on_output=True, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    w0 = torch.randn(out, inp, generator=g) * 0.05
    diff = torch.randn(out, inp, generator=g) * 0.02  # ΔW 가 W₀ 의 약 40% — 두 공식 차이가 드러나게
    dims = tuple(range(1, 2)) if wd_on_output else (0,)
    dora_scale = torch.linalg.vector_norm(w0, dim=dims, keepdim=True) * (1.0 + 0.3 * torch.rand(
        (out, 1) if wd_on_output else (1, inp), generator=g))
    return w0.to(dtype), diff, dora_scale


def _identity(a):
    return a


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


class DecomposeTests(unittest.TestCase):
    def test_lycoris_mode_matches_lycoris_training_formula_on_both_axes(self):
        for wd_on_output in (True, False):
            with self.subTest(wd_on_output=wd_on_output):
                w0, diff, scale = _case(wd_on_output=wd_on_output)
                expected = lycoris_merged_weight(w0, diff, scale, wd_on_output)
                weight = w0.clone()
                out = dim.decompose(scale, weight, diff.clone(), 1.0, 1.0, _identity, merged_norm=True)
                self.assertIs(out, weight)
                torch.testing.assert_close(weight, expected, rtol=1e-6, atol=1e-7)

    def test_forge_fp32_mode_matches_forge_formula_computed_in_fp32(self):
        for wd_on_output in (True, False):
            for strength in (1.0, 0.6):
                with self.subTest(wd_on_output=wd_on_output, strength=strength):
                    w0, diff, scale = _case(wd_on_output=wd_on_output, seed=3)
                    expected = forge_weight_decompose(scale, w0.clone(), diff.clone(), 0.5, strength, torch.float32, _identity)
                    weight = w0.clone()
                    dim.decompose(scale, weight, diff.clone(), 0.5, strength, _identity, merged_norm=False)
                    torch.testing.assert_close(weight, expected, rtol=1e-6, atol=1e-7)

    def test_output_axis_formulas_differ_input_axis_formulas_agree(self):
        w0, diff, scale = _case(wd_on_output=True, seed=5)
        forge, lyco = w0.clone(), w0.clone()
        dim.decompose(scale, forge, diff.clone(), 1.0, 1.0, _identity, merged_norm=False)
        dim.decompose(scale, lyco, diff.clone(), 1.0, 1.0, _identity, merged_norm=True)
        self.assertGreater((forge - lyco).abs().max().item(), 1e-4)
        # LyCORIS 는 행 노름을 dora_scale 로 맞추고, Forge 는 ‖W₀+ΔW‖/‖W₀‖ 배만큼 어긋난다.
        torch.testing.assert_close(lyco.norm(dim=1, keepdim=True), scale, rtol=1e-5, atol=1e-6)

        w0, diff, scale = _case(wd_on_output=False, seed=6)
        forge, lyco = w0.clone(), w0.clone()
        dim.decompose(scale, forge, diff.clone(), 1.0, 1.0, _identity, merged_norm=False)
        dim.decompose(scale, lyco, diff.clone(), 1.0, 1.0, _identity, merged_norm=True)
        torch.testing.assert_close(forge, lyco, rtol=0, atol=0)

    def test_strength_blends_linearly_and_zero_leaves_the_weight(self):
        w0, diff, scale = _case(seed=7)
        full = w0.clone()
        dim.decompose(scale, full, diff.clone(), 1.0, 1.0, _identity, merged_norm=True)
        zero = w0.clone()
        dim.decompose(scale, zero, diff.clone(), 1.0, 0.0, _identity, merged_norm=True)
        torch.testing.assert_close(zero, w0, rtol=0, atol=1e-7)
        half = w0.clone()
        dim.decompose(scale, half, diff.clone(), 1.0, 0.5, _identity, merged_norm=True)
        torch.testing.assert_close(half, w0 + 0.5 * (full - w0), rtol=1e-6, atol=1e-7)

    def test_writes_in_place_into_a_narrowed_half_precision_view(self):
        # 오프셋 패치는 narrow 뷰를 넘기고 반환값을 버린다(backend/patcher/lora.py:47-65).
        w0, diff, scale = _case(out=8, inp=6, seed=9)
        parent = torch.zeros(20, 6, dtype=torch.float16)
        parent[4:12] = w0.to(torch.float16)
        view = parent.narrow(0, 4, 8)
        dim.decompose(scale, view, diff.clone(), 1.0, 1.0, _identity, merged_norm=True)
        self.assertEqual(parent.dtype, torch.float16)
        expected = lycoris_merged_weight(w0.to(torch.float16).float(), diff, scale, True).to(torch.float16)
        torch.testing.assert_close(parent[4:12], expected, rtol=0, atol=0)
        self.assertTrue(bool((parent[:4] == 0).all()) and bool((parent[12:] == 0).all()))

    def test_not_inplace_returns_fp32_and_leaves_weight(self):
        w0, diff, scale = _case(seed=10)
        weight = w0.to(torch.float16)
        before = weight.clone()
        out = dim.decompose(scale, weight, diff.clone(), 1.0, 1.0, _identity, merged_norm=True, inplace=False)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(weight, before, rtol=0, atol=0)
        torch.testing.assert_close(out, lycoris_merged_weight(before.float(), diff, scale, True), rtol=1e-6, atol=1e-7)

    def test_fp32_beats_stock_fp16_against_the_training_formula(self):
        # 순정 fp16 경로(eps 9.8e-4, fp16 노름)와 fp32 경로를 LyCORIS 기준에 대 본다 — 입력 축은 공식이 같아 정밀도만 남는다.
        w0, diff, scale = _case(out=64, inp=48, seed=11, wd_on_output=False)
        expected = lycoris_merged_weight(w0.to(torch.float16).float(), diff, scale, False)
        stock = forge_weight_decompose(scale, w0.to(torch.float16), diff.clone(), 1.0, 1.0, torch.float32, _identity)
        ours = w0.to(torch.float16)
        dim.decompose(scale, ours, diff.clone(), 1.0, 1.0, _identity, merged_norm=True)
        err_stock = (stock.float() - expected).abs().mean().item()
        err_ours = (ours.float() - expected).abs().mean().item()
        self.assertLess(err_ours, err_stock)

    def test_does_not_modify_lora_diff_or_weight_on_failure(self):
        w0, diff, _ = _case(seed=13)
        bad_scale = torch.ones(3, 1)  # 어느 축과도 안 맞는다
        weight, lora_diff = w0.clone(), diff.clone()
        with self.assertRaises(RuntimeError):
            dim.decompose(bad_scale, weight, lora_diff, 2.0, 1.0, _identity, merged_norm=True)
        torch.testing.assert_close(weight, w0, rtol=0, atol=0)
        torch.testing.assert_close(lora_diff, diff, rtol=0, atol=0)


class ModeTests(unittest.TestCase):
    def tearDown(self):
        dim.set_mode(dim.MODE_FORGE)
        dim.take_counters()

    def test_normalize_accepts_keys_labels_xyz_and_infotext_values(self):
        cases = {
            "lycoris": dim.MODE_LYCORIS, "LyCORIS": dim.MODE_LYCORIS, "LyCORIS (학습과 동일 · fp32)": dim.MODE_LYCORIS,
            "LyCORIS fp32": dim.MODE_LYCORIS,
            "forge_fp32": dim.MODE_FORGE_FP32, "Forge fp32": dim.MODE_FORGE_FP32, "Forge/Comfy 공식 · fp32": dim.MODE_FORGE_FP32,
            "forge": dim.MODE_FORGE, "Forge/Comfy (순정)": dim.MODE_FORGE, "Forge (stock)": dim.MODE_FORGE, "off": dim.MODE_FORGE,
            "no_magnitude": dim.MODE_NO_MAGNITUDE, "No magnitude": dim.MODE_NO_MAGNITUDE,
            "DoRA off (no magnitude)": dim.MODE_NO_MAGNITUDE, "DoRA 끔 (크기 보정 없이 ΔW만 · 일반 LoKr처럼)": dim.MODE_NO_MAGNITUDE,
        }
        for value, mode in cases.items():
            with self.subTest(value=value):
                self.assertEqual(dim.normalize_mode(value), mode)
        for value in (None, "", "dora", 3):
            with self.subTest(value=value):
                self.assertIsNone(dim.normalize_mode(value))

    def test_set_mode_reports_changes_and_rejects_unknown(self):
        self.assertFalse(dim.set_mode("forge"))
        self.assertTrue(dim.set_mode("lycoris"))
        self.assertFalse(dim.set_mode("LyCORIS"))
        with self.assertRaises(ValueError):
            dim.set_mode("nope")
        self.assertEqual(dim.current_mode(), dim.MODE_LYCORIS)

    def test_every_non_stock_mode_has_an_infotext_value(self):
        self.assertEqual(set(dim.INFOTEXT_VALUES), set(dim.MODES) - {dim.MODE_FORGE})
        for value in dim.INFOTEXT_VALUES.values():
            self.assertRegex(value, r"^[\w\s\-/]+$")
        self.assertRegex(dim.INFOTEXT_KEY, r"^[\w\s\-/]+$")

    def test_stock_state_matches_the_block_policy_default(self):
        self.assertEqual(dim.STOCK_STATE, (dim.MODE_FORGE, alb.DUPLICATE_KEEP))


def _fake_adapters():
    calls = []

    def make(name):
        module = types.ModuleType(f"fake_weight_adapter.{name}")

        def original(*args):
            calls.append((name, args))
            return "stock"

        module.weight_decompose = original
        return module, original

    mods, originals = {}, {}
    for name in ("lora", "lokr", "loha", "glora", "oft"):
        mods[name], originals[name] = make(name)
    return mods, originals, calls


class HookTests(unittest.TestCase):
    def tearDown(self):
        dim.uninstall()
        dim.take_counters()

    def test_install_wraps_only_same_argument_adapters_and_is_idempotent(self):
        mods, originals, _ = _fake_adapters()
        self.assertEqual(dim.install(mods), 4)
        self.assertEqual(dim.install(mods), 0)
        self.assertEqual(dim.installed_adapters(), ("lora", "lokr", "loha", "glora"))
        for name in ("lora", "lokr", "loha", "glora"):
            self.assertIsNot(mods[name].weight_decompose, originals[name])
        self.assertIs(mods["oft"].weight_decompose, originals["oft"])
        self.assertEqual(dim.uninstall(), 4)
        for name in ("lora", "lokr", "loha", "glora", "oft"):
            self.assertIs(mods[name].weight_decompose, originals[name])

    def test_stock_mode_passes_through_untouched(self):
        mods, _, calls = _fake_adapters()
        dim.install(mods)
        args = (object(), object(), object(), 0.5, 0.7, torch.float32, _identity)
        self.assertEqual(mods["lokr"].weight_decompose(*args), "stock")
        self.assertEqual(calls, [("lokr", args)])
        self.assertEqual(dim.take_counters(), dim.Counters(0, 0, 0, None))

    def test_normal_patch_path_returns_fp32_without_touching_the_buffer(self):
        # patch_weight_to_device: fp16 임시 버퍼, intermediate fp32 → fp32 결과를 받아 한 번만 bf16 으로 반올림한다.
        mods, _, calls = _fake_adapters()
        dim.install(mods)
        dim.set_mode(dim.MODE_LYCORIS)
        w0, diff, scale = _case(seed=21)
        weight = w0.to(torch.float16)
        before = weight.clone()
        out = mods["lora"].weight_decompose(scale, weight, diff.clone(), 1.0, 1.0, torch.float32, _identity)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(weight, before, rtol=0, atol=0)
        torch.testing.assert_close(out, lycoris_merged_weight(before.float(), diff, scale, True), rtol=1e-6, atol=1e-7)
        self.assertEqual(calls, [])

    def test_lowvram_and_view_paths_write_in_place_in_the_storage_dtype(self):
        mods, _, _ = _fake_adapters()
        dim.install(mods)
        dim.set_mode(dim.MODE_LYCORIS)
        w0, diff, scale = _case(seed=22)
        weight = w0.to(torch.bfloat16)  # LowVramPatch/OnlineLoRAPatch: computation_dtype = weight.dtype
        out = mods["lokr"].weight_decompose(scale, weight, diff.clone(), 1.0, 1.0, torch.bfloat16, _identity)
        self.assertIs(out, weight)
        self.assertEqual(out.dtype, torch.bfloat16)
        parent = torch.zeros(30, 16, dtype=torch.float16)
        parent[2:26] = w0.to(torch.float16)
        view = parent.narrow(0, 2, 24)
        out = mods["lokr"].weight_decompose(scale, view, diff.clone(), 1.0, 1.0, torch.float32, _identity)
        self.assertIs(out, view)
        self.assertFalse(bool((parent[2:26] == w0.to(torch.float16)).all()))

    def test_no_magnitude_mode_is_bit_identical_to_forge_without_dora_scale(self):
        # dora_scale 이 없는 파일을 Forge 가 합치는 줄(lora.py·lokr.py·loha.py·glora.py)과 같아야 한다.
        mods, _, calls = _fake_adapters()
        dim.install(mods)
        dim.set_mode(dim.MODE_NO_MAGNITUDE)
        w0, diff, scale = _case(seed=26)
        for dtype, strength, alpha in ((torch.float16, 1.0, 1.0), (torch.float16, 0.7, 0.5), (torch.bfloat16, 1.3, 1.0)):
            with self.subTest(dtype=dtype, strength=strength):
                weight = w0.to(dtype)
                expected = weight.clone()
                expected += _identity(((strength * alpha) * diff.clone()).type(dtype))
                out = mods["lokr"].weight_decompose(scale, weight, diff.clone(), alpha, strength, torch.float32, _identity)
                self.assertIs(out, weight)
                self.assertEqual(out.dtype, dtype)
                torch.testing.assert_close(weight, expected, rtol=0, atol=0)
        self.assertEqual(calls, [])
        self.assertEqual(dim.take_counters(), dim.Counters(layers=1, calls=3, fallbacks=0, fallback_reason=None))

    def test_no_magnitude_mode_writes_in_place_into_views(self):
        mods, _, _ = _fake_adapters()
        dim.install(mods)
        dim.set_mode(dim.MODE_NO_MAGNITUDE)
        w0, diff, scale = _case(out=8, inp=6, seed=27)
        parent = torch.zeros(20, 6, dtype=torch.float16)
        parent[4:12] = w0.to(torch.float16)
        view = parent.narrow(0, 4, 8)
        mods["lora"].weight_decompose(scale, view, diff.clone(), 1.0, 1.0, torch.float32, _identity)
        torch.testing.assert_close(parent[4:12], w0.to(torch.float16) + diff.to(torch.float16), rtol=0, atol=0)
        self.assertTrue(bool((parent[:4] == 0).all()) and bool((parent[12:] == 0).all()))

    def test_counts_distinct_layers_and_calls(self):
        mods, _, _ = _fake_adapters()
        dim.install(mods)
        dim.set_mode(dim.MODE_FORGE_FP32)
        w0, diff, scale = _case(seed=23)
        for _ in range(3):  # Low VRAM 은 같은 레이어를 매 forward 합친다
            mods["lokr"].weight_decompose(scale, w0.clone(), diff.clone(), 1.0, 1.0, torch.float32, _identity)
        other = scale.clone()
        mods["lokr"].weight_decompose(other, w0.clone(), diff.clone(), 1.0, 1.0, torch.float32, _identity)
        self.assertEqual(dim.take_counters(), dim.Counters(layers=2, calls=4, fallbacks=0, fallback_reason=None))

    def test_failure_and_oom_fall_back_to_stock(self):
        mods, _, calls = _fake_adapters()
        dim.install(mods)
        dim.set_mode(dim.MODE_FORGE_FP32)
        w0, diff, _ = _case(seed=24)
        bad = (torch.ones(3, 1), w0.clone(), diff.clone(), 1.0, 1.0, torch.float32, _identity)
        self.assertEqual(mods["loha"].weight_decompose(*bad), "stock")
        self.assertEqual(len(calls), 1)

        emptied = []
        stub = types.SimpleNamespace(
            is_oom=lambda e: "out of memory" in str(e),
            soft_empty_cache=lambda: emptied.append(True),
        )

        def boom(_):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        w0, diff, scale = _case(seed=25)
        with _SysModules(backend__memory_management=stub):
            self.assertEqual(
                mods["loha"].weight_decompose(scale, w0.clone(), diff.clone(), 1.0, 1.0, torch.float32, boom),
                "stock",
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(emptied, [True])
        counters = dim.take_counters()
        self.assertEqual((counters.layers, counters.fallbacks), (0, 2))
        self.assertIn("RuntimeError", counters.fallback_reason)

    def test_stale_wrapper_from_a_previous_load_is_replaced_not_nested(self):
        mods, originals, _ = _fake_adapters()
        dim.install(mods)
        stale = mods["lokr"].weight_decompose
        dim._INSTALLED.clear()  # "Reload scripts" — 모듈 상태는 사라지고 래퍼만 Forge 모듈에 남는다
        self.assertEqual(dim.install(mods), 4)
        fresh = mods["lokr"].weight_decompose
        self.assertIsNot(fresh, stale)
        self.assertIs(getattr(fresh, dim._ORIGINAL_ATTR), originals["lokr"])

    def test_uninstall_leaves_a_foreign_wrapper_on_top_and_turns_ours_stock(self):
        mods, _, calls = _fake_adapters()
        dim.install(mods)
        dim.set_mode(dim.MODE_LYCORIS)
        ours = mods["lokr"].weight_decompose

        def foreign(*args):
            return ours(*args)

        mods["lokr"].weight_decompose = foreign
        self.assertEqual(dim.uninstall(), 3)
        self.assertIs(mods["lokr"].weight_decompose, foreign)
        self.assertEqual(dim.current_mode(), dim.MODE_FORGE)
        args = (object(), object(), object(), 1.0, 1.0, torch.float32, _identity)
        self.assertEqual(mods["lokr"].weight_decompose(*args), "stock")
        self.assertEqual(len(calls), 1)


def _proc_classes():
    class Proc:
        cached_c = [("old",), "cond", None]
        cached_uc = [("old",), "uncond", None]

    class Txt2Img(Proc):
        cached_hr_c = [("old",), "hr-cond"]  # processing.py:1567 이 2개짜리로 바꿔 두기도 한다
        cached_hr_uc = [("old",), "hr-uncond", None]

        def __init__(self):
            # __post_init__ 처럼 클래스 목록을 가리킨다(processing.py:266-268)
            self.cached_c = Proc.cached_c
            self.cached_uc = Proc.cached_uc
            self.cached_hr_c = Txt2Img.cached_hr_c
            self.cached_hr_uc = Txt2Img.cached_hr_uc

    return Proc, Txt2Img


class InvalidateTests(unittest.TestCase):
    def test_lora_hash_never_equals_the_no_lora_hash(self):
        # str([]) 는 LoRA 없는 프롬프트의 해시와 같아 load_networks 가 이전 LoRA 를 그대로 두고 돌아간다(networks.py:187).
        model = types.SimpleNamespace(current_lora_hash="[['x.safetensors', 1.0, 1.0, False]]")
        dim.invalidate_lora_cache(None, model)
        self.assertIsNone(model.current_lora_hash)
        compiled_empty = str([])
        self.assertFalse(model.current_lora_hash == compiled_empty)

    def test_cond_caches_are_cleared_in_place_for_every_alias(self):
        Proc, Txt2Img = _proc_classes()
        p = Txt2Img()
        cell = types.SimpleNamespace(**vars(p))  # XYZ 칸 = copy(p): 같은 목록을 쥔다(xyz_grid.py:769)
        original_c = p.cached_c
        dim.invalidate_lora_cache(p, None)
        self.assertIs(p.cached_c, original_c)  # 바꿔치기가 아니라 비운다
        self.assertEqual(cell.cached_c, [None, None, None])
        self.assertEqual(Proc.cached_uc, [None, None, None])
        self.assertEqual(Txt2Img.cached_hr_c, [None, None])  # 길이 유지
        self.assertEqual(cell.cached_hr_uc, [None, None, None])

    def test_sync_merged_state_invalidates_on_change_or_unknown(self):
        model = types.SimpleNamespace(current_lora_hash="h")
        # 표시 없는 모델은 어떤 상태로 합쳐졌는지 모른다(process() 를 거치지 않고 올라온 모델) — 순정 요청이어도 다시 만든다.
        self.assertTrue(dim.sync_merged_state(None, model, dim.STOCK_STATE))
        self.assertIsNone(model.current_lora_hash)
        model.current_lora_hash = "h"
        self.assertFalse(dim.sync_merged_state(None, model, dim.STOCK_STATE))
        self.assertEqual(model.current_lora_hash, "h")
        self.assertTrue(dim.sync_merged_state(None, model, (dim.MODE_LYCORIS, "keep")))
        self.assertIsNone(model.current_lora_hash)
        model.current_lora_hash = "h2"
        self.assertFalse(dim.sync_merged_state(None, model, (dim.MODE_LYCORIS, "keep")))
        self.assertEqual(model.current_lora_hash, "h2")
        self.assertTrue(dim.sync_merged_state(None, model, (dim.MODE_LYCORIS, "additive")))

    def test_uninstall_invalidates_a_model_merged_in_a_non_stock_state(self):
        # Reload UI: 전역 모드만 순정으로 돌고 모델·해시·조건 캐시는 남는다(webui.py:180).
        Proc, Txt2Img = _proc_classes()
        model = types.SimpleNamespace(current_lora_hash="h")
        setattr(model, dim.MERGED_STATE_ATTR, (dim.MODE_LYCORIS, "additive"))
        shared = types.SimpleNamespace(sd_model=model)
        processing = types.SimpleNamespace(StableDiffusionProcessing=Proc, StableDiffusionProcessingTxt2Img=Txt2Img)
        with _SysModules(modules__shared=shared, modules__processing=processing):
            dim.uninstall()
        self.assertIsNone(model.current_lora_hash)
        self.assertEqual(getattr(model, dim.MERGED_STATE_ATTR), dim.STOCK_STATE)
        self.assertEqual(Proc.cached_c, [None, None, None])
        self.assertEqual(Txt2Img.cached_hr_c, [None, None])

        clean = types.SimpleNamespace(current_lora_hash="h")
        setattr(clean, dim.MERGED_STATE_ATTR, dim.STOCK_STATE)
        with _SysModules(modules__shared=types.SimpleNamespace(sd_model=clean)):
            dim.uninstall()
        self.assertEqual(clean.current_lora_hash, "h")

        untagged = types.SimpleNamespace(current_lora_hash="h")  # 표시 없이 합쳐진 모델 — 상태를 모른다
        with _SysModules(modules__shared=types.SimpleNamespace(sd_model=untagged)):
            dim.uninstall()
        self.assertIsNone(untagged.current_lora_hash)
        self.assertEqual(getattr(untagged, dim.MERGED_STATE_ATTR), dim.STOCK_STATE)

    def test_img2img_invalidation_also_clears_txt2img_hires_class_caches(self):
        Proc, Txt2Img = _proc_classes()

        class Img2Img(Proc):
            pass

        p = Img2Img()  # MRO 에 Txt2Img 가 없다
        processing = types.SimpleNamespace(StableDiffusionProcessing=Proc, StableDiffusionProcessingTxt2Img=Txt2Img)
        with _SysModules(modules__processing=processing):
            dim.invalidate_lora_cache(p, None)
        self.assertEqual(Proc.cached_c, [None, None, None])
        self.assertEqual(Txt2Img.cached_hr_c, [None, None])
        self.assertEqual(Txt2Img.cached_hr_uc, [None, None, None])


def _load_script():
    modules_stub = types.ModuleType("modules")

    class Script:
        def elem_id(self, item_id):
            return f"script_{item_id}"

    callbacks = {"before_ui": [], "unloaded": [], "model_loaded": []}
    modules_stub.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_before_ui=callbacks["before_ui"].append,
        on_script_unloaded=callbacks["unloaded"].append,
        on_model_loaded=callbacks["model_loaded"].append,
    )
    modules_stub.shared = types.SimpleNamespace(sd_model=None)
    modules_stub.ui_components = types.SimpleNamespace(InputAccordion=None)
    original_loader = dim._load_adapter_module
    dim._load_adapter_module = lambda name: None  # 스크립트 import 시 Forge 패키지를 찾지 않게
    try:
        with _SysModules(modules=modules_stub, modules__ui_components=modules_stub.ui_components):
            spec = importlib.util.spec_from_file_location("_test_dora_infer_mode_script", ROOT / "scripts" / "dora_infer_mode.py")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
    finally:
        dim._load_adapter_module = original_loader
    return module, modules_stub, callbacks


class ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod, cls.stub, cls.callbacks = _load_script()

    def setUp(self):
        self.adapters, _, _ = _fake_adapters()
        self._loader = dim._load_adapter_module
        self._hook_installed = alb.forge_hook_installed
        dim._load_adapter_module = lambda name: None  # process() 의 _install_hook 이 실제 Forge 를 찾지 않게
        alb.forge_hook_installed = lambda: True
        dim.uninstall()
        dim.install(self.adapters)
        dim.take_counters()

    def tearDown(self):
        dim.uninstall()
        dim.take_counters()
        alb.set_duplicate_policy(alb.DUPLICATE_KEEP)
        alb.set_weak_copy(alb.DEFAULT_WEAK_STRENGTH, alb.WEAK_SCOPE_ATTN)
        dim._load_adapter_module = self._loader
        alb.forge_hook_installed = self._hook_installed

    def _p(self, model=None):
        _, Txt2Img = _proc_classes()
        p = Txt2Img()
        p.sd_model = model or types.SimpleNamespace(current_lora_hash="[['a.safetensors', 1.0, 1.0, False]]")
        p.extra_generation_params = {}
        return p

    def _process(self, p, *args):
        self.stub.shared.sd_model = p.sd_model
        with _SysModules(modules=self.stub):
            self.mod.DoraInferenceMode().process(p, *args)

    def test_registers_unload_model_loaded_and_before_ui_callbacks(self):
        self.assertIn(self.mod._on_unloaded, self.callbacks["unloaded"])
        self.assertIn(self.mod._on_model_loaded, self.callbacks["model_loaded"])
        self.assertEqual(len(self.callbacks["before_ui"]), 2)

    def test_new_model_is_tagged_with_the_state_its_loras_merge_under(self):
        # ✨ 루프 안 재로드·hires 체크포인트·🎯 — process() 뒤에 올라온 모델은 지금 전역 상태로 합쳐진다.
        dim.set_mode(dim.MODE_LYCORIS)
        alb.set_duplicate_policy(alb.DUPLICATE_ADDITIVE)
        model = types.SimpleNamespace(current_lora_hash="[]")
        self.mod._on_model_loaded(model)
        self.assertEqual(getattr(model, dim.MERGED_STATE_ATTR), (dim.MODE_LYCORIS, alb.DUPLICATE_ADDITIVE))
        model.current_lora_hash = "h"
        self._process(self._p(model), False)  # 다음 생성이 순정 — 낡은 LyCORIS·덧셈형 가중치를 버린다
        self.assertIsNone(model.current_lora_hash)

    def test_untagged_model_is_rebuilt_once_then_reused(self):
        model = types.SimpleNamespace(current_lora_hash="h")
        self._process(self._p(model), False)
        self.assertIsNone(model.current_lora_hash)
        self.assertEqual(getattr(model, dim.MERGED_STATE_ATTR), dim.STOCK_STATE)
        model.current_lora_hash = "h2"
        self._process(self._p(model), False)
        self.assertEqual(model.current_lora_hash, "h2")

    def test_standalone_inner_pass_records_the_inherited_state(self):
        m = self.mod
        self._process(self._p(), True, m.LABEL_FORGE_FP32, m.LABEL_DUP_SKIP)
        p2 = self._p()  # 단독 Refine: 빈 infotext 로 저장된다(inpaint_core.py:473)
        p2._sam3_inner = True
        self._process(p2, False)
        self.assertEqual(p2.extra_generation_params, {"DoRA mode": "Forge fp32", "DoRA inserted": "skip"})
        self.assertEqual((dim.current_mode(), alb.current_duplicate_policy()), (dim.MODE_FORGE_FP32, alb.DUPLICATE_SKIP))

    def test_unload_resets_the_block_policy(self):
        alb.set_duplicate_policy(alb.DUPLICATE_SKIP)
        with _SysModules(modules__shared=types.SimpleNamespace(sd_model=None)):
            self.mod._on_unloaded()
        self.assertEqual(alb.current_duplicate_policy(), alb.DUPLICATE_KEEP)

    def test_coerce_args_positional_dict_and_defaults(self):
        m = self.mod
        # 끼워 넣은 블록 기본값은 순정(그대로 복제) — 2개 인자만 보내는 예전 API 요청도 순정으로 돈다.
        self.assertEqual(m.coerce_args([]), (False, dim.MODE_LYCORIS, alb.DUPLICATE_KEEP))
        self.assertEqual(m.coerce_args([True, m.LABEL_FORGE_FP32]), (True, dim.MODE_FORGE_FP32, alb.DUPLICATE_KEEP))
        self.assertEqual(m.coerce_args([True, m.LABEL_FORGE, m.LABEL_DUP_SKIP]), (True, dim.MODE_FORGE, alb.DUPLICATE_SKIP))
        self.assertEqual(
            m.coerce_args([{"enabled": "true", "mode": "lycoris", "inserted": "keep"}]),
            (True, dim.MODE_LYCORIS, alb.DUPLICATE_KEEP),
        )
        self.assertEqual(m.coerce_args([True, "???", "???"]), (True, dim.MODE_LYCORIS, alb.DUPLICATE_KEEP))
        self.assertEqual(m.coerce_args([True, m.LABEL_NO_MAGNITUDE]), (True, dim.MODE_NO_MAGNITUDE, alb.DUPLICATE_KEEP))
        self.assertEqual(m.DUP_CHOICES[0], m.LABEL_DUP_KEEP)  # 라디오 기본값·맨 앞

    def test_state_change_invalidates_once_and_records_infotext(self):
        m = self.mod
        model = types.SimpleNamespace(current_lora_hash="h1")
        p = self._p(model)
        self._process(p, True, m.LABEL_LYCORIS, m.LABEL_DUP_ADDITIVE)
        self.assertEqual(dim.current_mode(), dim.MODE_LYCORIS)
        self.assertEqual(alb.current_duplicate_policy(), alb.DUPLICATE_ADDITIVE)
        self.assertIsNone(model.current_lora_hash)
        self.assertEqual(p.extra_generation_params, {"DoRA mode": "LyCORIS", "DoRA inserted": "additive"})

        model.current_lora_hash = "h2"
        self._process(self._p(model), True, m.LABEL_LYCORIS, m.LABEL_DUP_ADDITIVE)
        self.assertEqual(model.current_lora_hash, "h2")  # 같은 상태 — 합쳐 둔 가중치를 그대로 쓴다

        self._process(self._p(model), True, m.LABEL_LYCORIS, m.LABEL_DUP_KEEP)
        self.assertIsNone(model.current_lora_hash)  # 블록 정책만 바뀌어도 다시 합친다

        model.current_lora_hash = "h3"
        p3 = self._p(model)
        self._process(p3, False, m.LABEL_LYCORIS, m.LABEL_DUP_ADDITIVE)
        self.assertEqual((dim.current_mode(), alb.current_duplicate_policy()), (dim.MODE_FORGE, alb.DUPLICATE_KEEP))
        self.assertIsNone(model.current_lora_hash)
        self.assertEqual(p3.extra_generation_params, {})

    def test_reload_ui_then_stock_generation_rebuilds(self):
        # 전역 모드가 순정으로 돌아가도 모델 표시가 LyCORIS 면 다시 합친다.
        m = self.mod
        model = types.SimpleNamespace(current_lora_hash="h")
        self._process(self._p(model), True, m.LABEL_LYCORIS, m.LABEL_DUP_KEEP)
        dim._STATE["mode"] = dim.MODE_FORGE  # 모듈 전역만 초기화된 상태
        model.current_lora_hash = "h"
        self._process(self._p(model), False)
        self.assertIsNone(model.current_lora_hash)

    def test_xyz_overrides_each_item_and_stock_cells_drop_stale_keys(self):
        m = self.mod
        p = self._p()
        shared_params = p.extra_generation_params  # XYZ 칸들은 이 dict 를 같이 쓴다
        m._xyz_set_mode(p, "LyCORIS fp32", None)
        m._xyz_set_dup(p, "skip", None)
        self._process(p, False)
        self.assertEqual(shared_params, {"DoRA mode": "LyCORIS", "DoRA inserted": "skip"})
        cell = self._p()
        cell.extra_generation_params = shared_params
        m._xyz_set_mode(cell, "Forge (stock)", None)
        self._process(cell, True, m.LABEL_LYCORIS, m.LABEL_DUP_KEEP)
        self.assertEqual(dim.current_mode(), dim.MODE_FORGE)
        self.assertEqual(shared_params, {})

    def test_inner_sam3_pass_keeps_the_outer_state(self):
        m = self.mod
        model = types.SimpleNamespace(current_lora_hash="h")
        outer = self._p(model)
        m._xyz_set_mode(outer, "LyCORIS fp32", None)
        self._process(outer, False)
        model.current_lora_hash = "h2"
        inner = self._p(model)
        inner._sam3_inner = True
        inner._sam3_outer = outer
        self._process(inner, False)  # 아코디언은 꺼져 있다 — 그래도 바깥 칸의 LyCORIS 를 유지
        self.assertEqual(dim.current_mode(), dim.MODE_LYCORIS)
        self.assertEqual(model.current_lora_hash, "h2")
        self.assertEqual(inner.extra_generation_params, {"DoRA mode": "LyCORIS"})

    def test_without_hooks_it_stays_stock_and_writes_nothing(self):
        dim.uninstall()
        alb.forge_hook_installed = lambda: False
        p = self._p()
        self._process(p, True, self.mod.LABEL_LYCORIS, self.mod.LABEL_DUP_ADDITIVE)
        self.assertEqual((dim.current_mode(), alb.current_duplicate_policy()), (dim.MODE_FORGE, alb.DUPLICATE_KEEP))
        self.assertEqual(p.extra_generation_params, {})

    def test_no_magnitude_generation_records_and_pastes(self):
        m = self.mod
        model = types.SimpleNamespace(current_lora_hash="h")
        p = self._p(model)
        self._process(p, True, m.LABEL_NO_MAGNITUDE, m.LABEL_DUP_KEEP)
        self.assertEqual(dim.current_mode(), dim.MODE_NO_MAGNITUDE)
        self.assertIsNone(model.current_lora_hash)
        self.assertEqual(p.extra_generation_params, {"DoRA mode": "No magnitude"})
        self.assertTrue(m._paste_enabled(p.extra_generation_params))
        self.assertEqual(m._paste_mode(p.extra_generation_params), m.LABEL_NO_MAGNITUDE)
        self.assertEqual(m._paste_dup(p.extra_generation_params), m.LABEL_DUP_KEEP)

    def test_weak_copy_args_defaults_and_xyz(self):
        m = self.mod
        self.assertEqual(m.ARG_NAMES[:3], ("enabled", "mode", "inserted"))  # 앞 인자 위치는 그대로
        self.assertEqual(m.coerce_weak([]), (alb.DEFAULT_WEAK_STRENGTH, alb.WEAK_SCOPE_ATTN))
        self.assertEqual(m.coerce_weak([True, m.LABEL_LYCORIS, m.LABEL_DUP_WEAK, 0.08, m.LABEL_SCOPE_ATTN_MLP]), (0.08, "attn_mlp"))
        self.assertEqual(m.coerce_weak([{"weak_strength": "0.5", "weak_scope": "all"}]), (0.5, "all"))
        self.assertEqual(m.coerce_weak([True, "x", "x", "bad", "bad"]), (alb.DEFAULT_WEAK_STRENGTH, "attn"))
        self.assertEqual(m.coerce_args([True, m.LABEL_LYCORIS, m.LABEL_DUP_WEAK, 0.1, "attn"]), (True, dim.MODE_LYCORIS, alb.DUPLICATE_WEAK))
        p = self._p()
        m._xyz_set_weak(p, 0.18, None)
        # 강도 축만 걸면 약한 복사로 돈다 — 아코디언이 꺼져 있어도.
        self.assertEqual(m.resolve_state(p, [False]), (dim.MODE_FORGE, alb.DUPLICATE_WEAK))
        self.assertEqual(m.resolve_weak(p, [True, m.LABEL_LYCORIS, m.LABEL_DUP_WEAK, 0.08, "all"]), (0.18, "all"))
        m._xyz_set_dup(p, "keep", None)
        self.assertEqual(m.resolve_state(p, [False]), (dim.MODE_FORGE, alb.DUPLICATE_KEEP))  # 블록 축이 먼저다
        q = self._p()
        m._xyz_set_scope(q, "attn_mlp", None)
        self.assertEqual(m.resolve_state(q, [True, m.LABEL_LYCORIS]), (dim.MODE_LYCORIS, alb.DUPLICATE_WEAK))
        self.assertEqual(m.resolve_weak(q, [True, m.LABEL_LYCORIS]), (alb.DEFAULT_WEAK_STRENGTH, "attn_mlp"))

    def test_weak_copy_strength_change_rebuilds_records_and_pastes(self):
        m = self.mod
        model = types.SimpleNamespace(current_lora_hash="h")
        p = self._p(model)
        self._process(p, True, m.LABEL_LYCORIS, m.LABEL_DUP_WEAK, 0.12, m.LABEL_SCOPE_ATTN)
        self.assertEqual(alb.current_duplicate_policy(), alb.DUPLICATE_WEAK)
        self.assertEqual(getattr(model, dim.MERGED_STATE_ATTR), (dim.MODE_LYCORIS, "weak 0.12 attn"))
        self.assertEqual(p.extra_generation_params, {"DoRA mode": "LyCORIS", "DoRA inserted": "weak 0.12 attn"})
        self.assertEqual(m._paste_dup(p.extra_generation_params), m.LABEL_DUP_WEAK)
        self.assertEqual(m._paste_weak_strength(p.extra_generation_params), 0.12)
        self.assertEqual(m._paste_weak_scope(p.extra_generation_params), m.LABEL_SCOPE_ATTN)

        model.current_lora_hash = "h2"
        self._process(self._p(model), True, m.LABEL_LYCORIS, m.LABEL_DUP_WEAK, 0.12, m.LABEL_SCOPE_ATTN)
        self.assertEqual(model.current_lora_hash, "h2")  # 같은 강도·범위 — 그대로 쓴다
        self._process(self._p(model), True, m.LABEL_LYCORIS, m.LABEL_DUP_WEAK, 0.18, m.LABEL_SCOPE_ATTN)
        self.assertIsNone(model.current_lora_hash)  # 강도만 바꿔도 다시 합친다
        model.current_lora_hash = "h3"
        self._process(self._p(model), True, m.LABEL_LYCORIS, m.LABEL_DUP_WEAK, 0.18, m.LABEL_SCOPE_ALL)
        self.assertIsNone(model.current_lora_hash)  # 범위만 바꿔도
        model.current_lora_hash = "h4"
        self._process(self._p(model), True, m.LABEL_LYCORIS, m.LABEL_DUP_KEEP, 0.5, m.LABEL_SCOPE_ATTN)
        self.assertIsNone(model.current_lora_hash)
        model.current_lora_hash = "h5"
        self._process(self._p(model), True, m.LABEL_LYCORIS, m.LABEL_DUP_KEEP, 0.9, m.LABEL_SCOPE_ALL)
        self.assertEqual(model.current_lora_hash, "h5")  # 약한 복사가 아니면 강도·범위는 상태에 없다
        for params in ({"DoRA inserted": "additive"}, {"Steps": "20"}):
            self.assertIsNone(m._paste_weak_strength(params))
            self.assertIsNone(m._paste_weak_scope(params))

    def test_unload_resets_weak_copy(self):
        alb.set_weak_copy(0.5, "all")
        with _SysModules(modules__shared=types.SimpleNamespace(sd_model=None)):
            self.mod._on_unloaded()
        self.assertEqual(alb.current_weak_copy(), (alb.DEFAULT_WEAK_STRENGTH, alb.WEAK_SCOPE_ATTN))

    def test_ui_returns_all_arguments_in_order(self):
        import gradio as gr

        m = self.mod
        script = m.DoraInferenceMode()
        with gr.Blocks():
            components = script.ui(False)
        self.assertEqual(len(components), len(m.ARG_NAMES))
        self.assertEqual([c for c, _ in script.infotext_fields], components)
        self.assertEqual(components[3].value, alb.DEFAULT_WEAK_STRENGTH)
        self.assertEqual(components[4].value, m.LABEL_SCOPE_ATTN)

    def test_xyz_axes_are_registered_with_a_cost(self):
        # cost 0 이면 xyz_grid 가 이 축을 가장 안쪽에 둬 칸마다 LoRA 를 다시 합친다(scripts/xyz_grid.py:740-760).
        m = self.mod
        created = []

        class AxisOption:
            def __init__(self, label, type, apply, format_value=None, confirm=None, cost=0.0, choices=None, prepare=None):
                self.label, self.cost = label, cost
                created.append(self)

        fake = types.SimpleNamespace(axis_options=[], AxisOption=AxisOption)
        entry = types.SimpleNamespace(script_class=types.SimpleNamespace(__module__="xyz_grid.py"), module=fake)
        original = self.stub.scripts.scripts_data
        self.stub.scripts.scripts_data = [entry]
        try:
            m._make_xyz_axis()
            m._make_xyz_axis()  # 두 번 불려도 중복 등록하지 않는다
        finally:
            self.stub.scripts.scripts_data = original
        labels = [a.label for a in fake.axis_options]
        self.assertEqual(sorted(labels), sorted([m.XYZ_MODE_LABEL, m.XYZ_DUP_LABEL, m.XYZ_SCOPE_LABEL, m.XYZ_WEAK_LABEL]))
        for axis in fake.axis_options:
            with self.subTest(axis=axis.label):
                self.assertGreater(axis.cost, 0.7)  # VAE(0.7) 보다 비싸고
                self.assertLess(axis.cost, 1.0)     # Checkpoint(1.0) 보다 싸다

    def test_paste_round_trip(self):
        m = self.mod
        full = {"DoRA mode": "LyCORIS", "DoRA inserted": "additive"}
        self.assertTrue(m._paste_enabled(full))
        self.assertEqual(m._paste_mode(full), m.LABEL_LYCORIS)
        self.assertEqual(m._paste_dup(full), m.LABEL_DUP_ADDITIVE)
        only_mode = {"DoRA mode": "Forge fp32"}
        self.assertEqual(m._paste_mode(only_mode), m.LABEL_FORGE_FP32)
        self.assertEqual(m._paste_dup(only_mode), m.LABEL_DUP_KEEP)
        only_dup = {"DoRA inserted": "skip"}
        self.assertTrue(m._paste_enabled(only_dup))
        self.assertEqual(m._paste_mode(only_dup), m.LABEL_FORGE)
        self.assertEqual(m._paste_dup(only_dup), m.LABEL_DUP_SKIP)
        none = {"Steps": "20"}
        self.assertFalse(m._paste_enabled(none))
        self.assertIsNone(m._paste_mode(none))
        self.assertIsNone(m._paste_dup(none))

    def test_choices_parse_and_xyz_choices_are_ascii(self):
        m = self.mod
        for label in m.XYZ_MODE_CHOICES + m.MODE_CHOICES:
            with self.subTest(label=label):
                self.assertIsNotNone(dim.normalize_mode(label))
        for label in m.XYZ_DUP_CHOICES + m.DUP_CHOICES:
            with self.subTest(label=label):
                self.assertIsNotNone(alb.normalize_duplicate_policy(label))
        for label in m.SCOPE_CHOICES + m.XYZ_SCOPE_CHOICES:
            with self.subTest(label=label):
                self.assertIsNotNone(alb.normalize_weak_scope(label))
        for label in (
            m.XYZ_MODE_CHOICES + m.XYZ_DUP_CHOICES + m.XYZ_SCOPE_CHOICES
            + [m.XYZ_MODE_LABEL, m.XYZ_DUP_LABEL, m.XYZ_WEAK_LABEL, m.XYZ_SCOPE_LABEL]
        ):
            with self.subTest(label=label):
                self.assertTrue(label.isascii())  # 격자 범례 글꼴(Roboto)에 한글이 없다
        self.assertRegex(m.INFOTEXT_DUP_KEY, r"^[\w\s\-/]+$")
        self.assertEqual(m.DORA_INFER_NAME, "DoRA Inference Mode")


if __name__ == "__main__":
    unittest.main()
