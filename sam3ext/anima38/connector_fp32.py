"""Semantic Connector v2 를 샘플링 동안 fp32 로 상주시키는 Forge 패처.

커넥터(와 전용 llm_adapter 사본)는 bf16 로 저장돼 있고 fp32 로 계산한다(입력 dtype = llm_adapter embed = fp32).
Forge manual cast 는 호출마다 모듈 295개의 bf16 가중치를 fp32 임시 사본으로 만들고(weights_manual_cast→cast_to),
``--cuda-stream`` 이면 모듈마다 스트림 대기까지 한다 — 50스텝 × run 2개면 한 장에 수만 번이다.

이 패처는 Forge 가 가중치를 다 올린 뒤(LoRA 패치까지 끝난 bf16 값) 그 값을 fp32 로 **한 번** 바꿔 두고
``parameters_manual_cast`` 를 끈다. bf16→fp32 는 값이 그대로인 올림이라, 호출마다 캐스트한 fp32 와 같은 값으로
같은 fp32 계산을 한다 — 결과가 비트 단위로 같다. Forge 가 내리거나(unpatch)·일부만 내리거나(partially_unload)·
다시 패치할 때는 먼저 원래 dtype 으로 되돌린다(fp32→bf16 도 값이 그대로) — Forge 의 백업·LoRA 합치기·
stochastic rounding 은 늘 원래 dtype 가중치를 본다. model_size 는 바꾼 뒤에만 fp32 크기다 — 바꾸기 전(또는 자리가
없어 못 바꾼 채)에는 원래 크기라, Forge 의 적재(full_load)·퇴출 판단이 기본 ModelPatcher 와 같다. 바꾸는 것은
Forge 가 적재하고 남긴 여유(extra_memory - 적재량) 안에서만이고, 실제 빈 VRAM 도 넘지 않는다.

설정(OPT_CONNECTOR_FP32)을 끄면 바꾸지 않는다 — Forge 기본 ModelPatcher 와 같은 경로다.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Forge 설정 — scripts/anima_3_8b.py 가 같은 이름으로 등록한다
OPT_CONNECTOR_FP32 = "sam3_anima38_connector_fp32"
# 값이 그대로 보존되는 올림 캐스트 (저장 dtype, 계산 dtype) — runtime._EXACT_UPCASTS 와 같다
EXACT_UPCASTS = frozenset({
    (torch.bfloat16, torch.float32),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float64),
    (torch.float16, torch.float64),
    (torch.float32, torch.float64),
})
PLAN_ATTR = "_sam3_fp32_plan"   # 모델(BundledV2Models)에 붙인다 — 같은 모듈을 쥔 패처 복제본이 함께 본다


def connector_fp32() -> bool:
    """커넥터를 샘플링 동안 fp32 로 상주시킬까 (Forge 설정 OPT_CONNECTOR_FP32). 설정이 없거나 Forge 밖이면 켜짐."""
    try:
        from modules import shared

        return bool(getattr(shared.opts, OPT_CONNECTOR_FP32, True))
    except Exception:
        return True


def _same_device(left, right) -> bool:
    left, right = torch.device(left), torch.device(right)
    if left.type != right.type:
        return False
    return left.index is None or right.index is None or left.index == right.index


class Fp32Plan:
    """모델 하나의 fp32 상주 상태. 계획(무엇을 얼마나)은 만들 때 한 번, 바꾼 목록은 지금 상태."""

    def __init__(self, model: torch.nn.Module, compute_dtype, skip_prefixes=()) -> None:
        from backend import memory_management

        self.compute_dtype = compute_dtype
        self.skip_prefixes = tuple(skip_prefixes)
        # 원래 dtype 기준 크기 — 기본 ModelPatcher.model_size 와 같은 셈(state_dict)
        self.storage_size = int(memory_management.module_size(model))
        self.growth = 0
        if compute_dtype is not None:
            for name, module in self._modules(model):
                for parameter in module.parameters(recurse=False):
                    if (parameter.dtype, compute_dtype) in EXACT_UPCASTS:
                        self.growth += parameter.numel() * (_itemsize(compute_dtype) - parameter.element_size())
        self.converted: list[tuple[torch.nn.Module, str, torch.dtype]] = []
        self.manual_cast_off: list[torch.nn.Module] = []
        self.converted_growth = 0

    def _modules(self, model):
        for name, module in model.named_modules():
            if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in self.skip_prefixes):
                continue
            yield name, module

    @property
    def active(self) -> bool:
        return bool(self.converted or self.manual_cast_off)

    def _candidates(self, model, device):
        """fp32 로 바꿀 (모듈, 바꿀 파라미터 이름들, 그 증가 바이트). 한 모듈의 파라미터가 모두 device 에 있고
        Forge 부분 로드(weight_function·prev_parameters_manual_cast)가 아니며 정확히 올릴 수 있을 때만."""
        compute = self.compute_dtype
        for _, module in self._modules(model):
            parameters = dict(module.named_parameters(recurse=False))
            forge = hasattr(module, "parameters_manual_cast")
            if not parameters and not forge:
                continue
            # 파라미터 없는 Forge 모듈(affine 없는 LayerNorm)도 manual cast 를 끈다 — 캐스트할 것은 없어도 스트림 대기를 한다
            if forge and (
                getattr(module, "weight_function", None)
                or getattr(module, "bias_function", None)
                or hasattr(module, "prev_parameters_manual_cast")
            ):
                continue
            if not all(_same_device(parameter.device, device) for parameter in parameters.values()):
                continue
            names, growth = [], 0
            exact = True
            for name, parameter in parameters.items():
                if parameter.dtype == compute:
                    continue
                if (parameter.dtype, compute) not in EXACT_UPCASTS:
                    exact = False
                    break
                names.append(name)
                growth += parameter.numel() * (_itemsize(compute) - parameter.element_size())
            if not exact:
                continue   # 정확히 올릴 수 없는 dtype(양자화 등) — 그 모듈은 manual cast 그대로
            yield module, names, growth, forge

    def pending_growth(self, model, device) -> int:
        return sum(growth for _, _, growth, _ in self._candidates(model, device))

    def convert(self, model, device) -> int:
        """device 에 다 올라온 모듈을 fp32 로 바꾸고 manual cast 를 끈다. 늘어난 바이트를 돌려준다."""
        if self.compute_dtype is None or self.active:
            return 0
        grown = 0
        # 인퍼런스 모드 안(샘플링 준비)에서 불려도 일반 텐서로 — 밖에서 옮기거나 되돌릴 때 버전 카운터 오류가 없게
        with torch.inference_mode(False), torch.no_grad():
            for module, names, growth, forge in list(self._candidates(model, device)):
                for name in names:
                    parameter = getattr(module, name)
                    setattr(module, name, torch.nn.Parameter(parameter.detach().to(dtype=self.compute_dtype), requires_grad=False))
                    self.converted.append((module, name, parameter.dtype))
                grown += growth
                if forge and module.parameters_manual_cast:
                    module.parameters_manual_cast = False
                    self.manual_cast_off.append(module)
        self.converted_growth = grown
        return grown

    def revert(self) -> int:
        """원래 dtype·manual cast 로 되돌린다(값 그대로). 줄어든 바이트를 돌려준다."""
        if not self.active:
            return 0
        with torch.inference_mode(False), torch.no_grad():
            for module, name, dtype in self.converted:
                parameter = getattr(module, name)
                if parameter.dtype != dtype:
                    setattr(module, name, torch.nn.Parameter(parameter.detach().to(dtype=dtype), requires_grad=False))
            for module in self.manual_cast_off:
                module.parameters_manual_cast = True
        shrunk = self.converted_growth
        self.converted, self.manual_cast_off, self.converted_growth = [], [], 0
        return shrunk


def _itemsize(dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def plan_of(model) -> Fp32Plan | None:
    return getattr(model, PLAN_ATTR, None)


_PATCHER_CLASS = None


def _patcher_class():
    global _PATCHER_CLASS
    if _PATCHER_CLASS is not None:
        return _PATCHER_CLASS
    from backend import memory_management
    from backend.patcher.base import ModelPatcher

    class Fp32ConnectorPatcher(ModelPatcher):
        """커넥터 번들 전용 ModelPatcher — 다 올린 뒤 fp32 로 한 번 바꾸고, Forge 가 가중치를 만지기 전에 되돌린다."""

        def _plan(self) -> Fp32Plan:
            return plan_of(self.model)

        def model_size(self) -> int:
            plan = self._plan()
            if plan is None:
                return super().model_size()
            # 바꾼 몫만 더한다 — 아직 못 바꿨는데 fp32 크기를 보고하면 Forge 가 여유가 bf16 이상·fp32 미만일 때
            # full_load 대신 lowvram 부분 적재를 하고, 다 올라간 bf16 을 "덜 올라감(offloaded)"으로 보아 다른 모델이나
            # 커넥터 자신을 퇴출한다. 기본 ModelPatcher(옛 경로)와 같은 판단이 되도록 바꾸기 전에는 원래 크기다.
            return plan.storage_size + (plan.converted_growth if plan.active else 0)

        def sync_fp32_setting(self) -> int:
            """설정을 껐으면 바꿔 둔 fp32 를 되돌린다(적재 메모리도 줄인다). 줄어든 바이트."""
            plan = self._plan()
            if plan is None or not plan.active or connector_fp32():
                return 0
            return self._revert()

        def _revert(self) -> int:
            plan = self._plan()
            if plan is None:
                return 0
            shrunk = plan.revert()
            if shrunk:
                self.model.model_loaded_weight_memory = max(0, self.model.model_loaded_weight_memory - shrunk)
            return shrunk

        def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
            self._revert()   # Forge 는 원래 dtype 가중치에 패치·백업한다
            result = super().load(device_to, lowvram_model_memory=lowvram_model_memory,
                                  force_patch_weights=force_patch_weights, full_load=full_load)
            if not self.model.model_lowvram and self._plan() is not None:
                # 다 올렸다 — full_load 면 Forge 는 model_size() 를 적어 두므로 실제 크기로 맞춘다(되돌린 뒤라 원래 크기)
                self.model.model_loaded_weight_memory = memory_management.module_size(self.model)
            return result

        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            self.sync_fp32_setting()
            gained = super().partially_load(device_to, extra_memory, force_patch_weights=force_patch_weights)
            return gained + self._maybe_convert(device_to, extra_memory - gained)

        def _maybe_convert(self, device_to, allowance) -> int:
            plan = self._plan()
            if (
                plan is None
                or plan.active
                or not connector_fp32()
                or device_to is None
                or self.model.model_lowvram
                or self.model.model_loaded_weight_memory <= 0
            ):
                return 0
            pending = plan.pending_growth(self.model, device_to)
            if pending <= 0:
                return 0
            if not memory_management.is_device_cpu(device_to):
                # Forge 가 한도 없이(HIGH_VRAM·1e32) 주더라도 실제 빈 VRAM 에서 샘플링 최소 몫을 남긴 만큼까지만.
                # model_size 가 fp32 몫을 미리 보고하지 않으므로 Forge 가 이 몫을 비워 두지 않는다.
                free = memory_management.get_free_memory(device_to) - memory_management.minimum_inference_memory()
                allowance = min(allowance, free)
            if pending > allowance:
                print(f"[Anima38] connector fp32 resident skipped: needs {pending / 2**20:.0f} MB more VRAM "
                      f"(allowed {max(0.0, allowance) / 2**20:.0f} MB) — using Forge manual cast")
                return 0
            grown = plan.convert(self.model, device_to)
            self.model.model_loaded_weight_memory += grown
            print(f"[Anima38] connector fp32 resident: +{grown / 2**20:.0f} MB VRAM, "
                  f"{len(plan.manual_cast_off)} modules without per-step cast")
            return grown

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            freed = self._revert()
            if freed >= memory_to_free:
                return freed   # fp32→bf16 로 되돌린 것만으로 충분 — 가중치는 VRAM 에 그대로
            return freed + super().partially_unload(device_to, memory_to_free - freed,
                                                    force_patch_weights=force_patch_weights)

        def unpatch_model(self, device_to=None, unpatch_weights=True):
            if unpatch_weights:
                self._revert()
            return super().unpatch_model(device_to, unpatch_weights=unpatch_weights)

    _PATCHER_CLASS = Fp32ConnectorPatcher
    return _PATCHER_CLASS


def make_connector_patcher(unet, models, *, skip_prefixes=()):
    """커넥터 번들 패처를 만들어 UNet 의 샘플링 추가 패처로 단다. Forge 가 없거나(테스트 스텁) UNet 이 Forge
    UnetPatcher 가 아니면 None — 부르는 쪽이 예전처럼 add_extra_torch_module_during_sampling 을 쓴다."""
    add = getattr(unet, "add_extra_model_patcher_during_sampling", None)
    load_device = getattr(unet, "load_device", None)
    offload_device = getattr(unet, "offload_device", None)
    if add is None or load_device is None or offload_device is None:
        return None
    try:
        patcher_class = _patcher_class()
    except Exception as exc:  # pragma: no cover - Forge 버전 차이
        logger.debug("[Anima38] fp32 connector patcher unavailable (%s)", exc)
        return None
    if plan_of(models) is None:
        compute = None
        embed = getattr(getattr(getattr(models, "native_adapter", None), "embed", None), "weight", None)
        if isinstance(embed, torch.Tensor):
            compute = embed.dtype
        object.__setattr__(models, PLAN_ATTR, Fp32Plan(models, compute, skip_prefixes))
    patcher = patcher_class(model=models, load_device=load_device, offload_device=offload_device)
    add(patcher)
    return patcher
