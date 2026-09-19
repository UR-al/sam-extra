from __future__ import annotations

import functools
import logging
import uuid
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from backend import memory_management
from backend.operations import ForgeOperations, using_forge_operations
from backend.patcher.clip import CLIP

from .adapter import ProgressiveCrossAdapter
from .marker import as_rows, read_run_ids, stamp_run_id
from .files import (
    ARCHITECTURE,
    CONNECTOR_PREFIX,
    V2_ARCHITECTURE,
    adapters,
    bundle_metadata,
    qwen35_models,
    tokenizer_dir,
)
from .qwen35 import Qwen35HybridModel
from .semantic_v2 import BundledV2Models, QualityAnchoredSemanticConnectorV2
from .tokenizer import Qwen35Tokenizer

logger = logging.getLogger(__name__)
LAYER_INDICES = (7, 15, 23, 31)
NATIVE_ADAPTER_PREFIX = "net.llm_adapter."   # 번들 안의 원본 llm_adapter (Forge 는 TE 로 옮겨 쓴다)
TE_ADAPTER_PATCH_PREFIX = "qwen3_06b.llm_adapter."   # TE 패처(JointTextEncoder) 기준 LoRA 키
SEMANTIC_CACHE_LINES = 32   # 줄당 수 MB (CPU) — batch count·Feature 6 후보·ADetailer 가 같은 줄을 되풀이한다


def _outside_inference_mode(load):
    """캐시할 가중치는 호출 문맥과 상관없이 일반 텐서로 만든다.

    Forge 는 process_batch·setup_conds 를 torch.inference_mode() 안에서 부르지만 추가 패처 언로드
    (postprocess 의 model.to(offload)) 는 그 밖에서 한다. inference 텐서로 만든 파라미터가 밖에서
    옮겨지면 버전 카운터 없는 파라미터가 되고, 부분 로드로 GPU 이동(= 복구)을 건너뛴 다음 샘플링의
    인덱싱이 "Inference tensors do not track version counter." 로 죽는다.
    """

    @functools.wraps(load)
    def wrapper(*args, **kwargs):
        with torch.inference_mode(False), torch.no_grad():
            return load(*args, **kwargs)

    return wrapper


def _anima_reference_state():
    """Forge 의 Anima 레퍼런스 상태 — (dynamic_args, opts). 테스트에서 바꿔 끼운다."""
    from backend.args import dynamic_args
    from modules.shared import opts

    return dynamic_args, opts


@dataclass
class _V2Run:
    source: torch.Tensor
    target_ids: torch.Tensor
    target_weights: torch.Tensor
    semantic: list[torch.Tensor]
    semantic_mask: torch.Tensor


class Anima3BRuntime:
    def __init__(self) -> None:
        self._qwen_path: str | None = None
        self._qwen: Qwen35HybridModel | None = None
        self._qwen_clip: CLIP | None = None
        self._tokenizer: Qwen35Tokenizer | None = None
        self._adapter_key: tuple[str, int] | None = None
        self._adapter: ProgressiveCrossAdapter | None = None
        self._active_bundle_path: str | None = None
        self._active_bundle_metadata: dict[str, str] | None = None
        self._v2_key: tuple[str, int] | None = None
        self._v2_models: BundledV2Models | None = None
        self._v2_source = None   # 커넥터를 만든 TE llm_adapter 의 weakref (붙잡지 않는다)
        self._v2_sampling_patcher = None
        self._v2_sampling_unets: list = []   # 패처를 단 UNet 의 weakref (설치 동안만 — restore 에서 비운다)
        self._installed_processing = None
        self._installed_model = None   # 설치한 sd_model 의 weakref — 반복 사이 모델 재로드 판별
        self._v2_runs: dict[int, _V2Run] = {}
        # (Qwen 파일, 줄, dtype) → Qwen3.5 의미 특징. 같은 줄이면 4.45 GiB 모델을 GPU 로 올리지 않는다.
        self._semantic_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._v2_run_counter = 0
        # 다른 확장(NegPiP 등)이 같은 속성을 감싸고 비중첩으로 되돌릴 수 있으므로, 우리 패치는
        # "속성 교체" 가 아니라 이 플래그로 켜고 끈다. 꺼진 채 남아 있어도 순정으로 위임한다.
        self._cond_active = False
        self._cond_params: tuple[str, float, float | None] | None = None
        self._v2_active = False
        # 모델마다 다른 것(우리 래퍼, 그 아래 원래 함수, DiT)은 여기 두지 않는다 — 공용 런타임은 프로세스
        # 내내 살아서, 여기 붙잡으면 체크포인트를 바꿔도 이전 모델(3.8B 면 약 8 GiB)이 RAM 에 남는다.
        # 래퍼는 약한 참조 집합으로 알아보고, 아래 함수는 래퍼 자신(_anima3b_below)이 들고 있다.
        self._wrappers: weakref.WeakSet = weakref.WeakSet()

    def _patched(self, candidate) -> bool:
        """우리 래퍼인가 — 표시 속성이 아니라 정체성으로 본다. NegPiP 는 functools.wraps 로 우리 래퍼를
        감싸 __dict__(_anima3b_patch 포함)를 복사하므로, 속성만 보면 위에 앉은 NegPiP 래퍼를 우리 것으로
        오인해 마스킹을 대신 적용하고 dict 를 돌려준다(NegPiP 의 assert 에서 생성이 죽는다)."""
        try:
            return candidate in self._wrappers
        except TypeError:   # None 처럼 약한 참조가 안 되는 값
            return False

    def _our_cond_wrapper(self, sd_model):
        """조건 체인에 살아 있는 우리 클로저 — 맨 위, 또는 NegPiP 가 위에 있으면 orig_forward 자리."""
        for candidate in (
            getattr(sd_model, "get_learned_conditioning", None),
            getattr(sd_model, "orig_forward", None),
        ):
            if self._patched(candidate):
                return candidate
        return None

    def _cond_installed(self, sd_model) -> bool:
        return self._our_cond_wrapper(sd_model) is not None

    @staticmethod
    def _negpip_patched(sd_model) -> bool:
        return hasattr(sd_model, "orig_forward")

    @staticmethod
    def _native_conditioning(sd_model):
        """클래스에 정의된 순정 get_learned_conditioning — 다른 확장의 (낡은) 래퍼를 건너뛴다."""
        return type(sd_model).get_learned_conditioning.__get__(sd_model, type(sd_model))

    @staticmethod
    def _native_forward(diffusion_model):
        return type(diffusion_model).forward.__get__(diffusion_model, type(diffusion_model))

    def _conditioning_entry(self, sd_model, prompt):
        if not self._cond_active or self._cond_params is None:
            # 우리 패치는 영구적이지만 꺼져 있을 땐 투명해야 한다 — 아래에 NegPiP 래퍼가
            # 살아 있으면(orig_forward 존재) 그쪽으로 넘겨 NegPiP 가 계속 동작하게 한다.
            wrapper = self._our_cond_wrapper(sd_model)
            below = getattr(wrapper, "_anima3b_below", None)
            if below is not None and below is not wrapper and self._negpip_patched(sd_model):
                return below(prompt)
            return self._native_conditioning(sd_model)(prompt)
        adapter_name, strength, negative_strength = self._cond_params
        conds = self.encode(sd_model, prompt, adapter_name, strength, negative_strength)
        # NegPiP 가 켜져 있는데(orig_forward 는 NegPiP 가 패치 중일 때만 있다) 그 래퍼가 우리 *아래*
        # 에 있으면 — 스크립트 순서가 NegPiP 먼저인 설치 — 우리 출력은 NegPiP 를 거치지 않는다.
        # 그 경우 NegPiP 가 했을 마스킹을 그대로 대신 적용해 어느 순서에서도 NegPiP 가 살아 있게 한다.
        if (
            isinstance(conds, list)
            and self._negpip_patched(sd_model)
            and self._patched(sd_model.get_learned_conditioning)
        ):
            return self._apply_negpip(sd_model, prompt, conds)
        return conds

    def _apply_negpip(self, sd_model, prompt, conds):
        """NegPiP 의 negpip_learned_conditioning 과 같은 변환: 줄별 ±1 마스크를 곱하고 마스크를 함께 돌려준다."""
        try:
            from lib_negpip.anima import _build_negpip_mask
        except Exception as exc:  # pragma: no cover - NegPiP 가 없거나 내부가 바뀜
            if not getattr(self, "_warned_negpip", False):
                self._warned_negpip = True
                logger.warning("[Anima38] NegPiP mask helper unavailable (%s); NegPiP is inactive for v2", exc)
            return conds
        engine = sd_model.text_processing_engine_anima
        crossattn, masks, count = [], [], 0
        for line, cond in zip(prompt, conds):
            data = cond.reshape(-1, cond.shape[-1])
            mask = _build_negpip_mask(engine, str(line), data.shape[0], data.device, data.dtype)
            count += int((mask < 0).sum())
            crossattn.append(data * mask.unsqueeze(-1).to(data))
            masks.append(mask.unsqueeze(-1).to(data))
        if count > 0:
            key = "Negative" if getattr(prompt, "is_negative_prompt", False) else "Positive"
            print(f"[Anima38] NegPiP Enable ({key}: {count})")
        return {"crossattn": torch.stack(crossattn, dim=0), "c_negpip_mask": torch.stack(masks, dim=0)}

    @staticmethod
    def _require_anima(sd_model):
        engine = getattr(sd_model, "text_processing_engine_anima", None)
        clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
        if engine is None or clip is None:
            raise RuntimeError("Anima 3.8B requires a loaded Anima checkpoint.")
        return engine, clip

    @staticmethod
    def _hand_off_native_reference(sd_model) -> None:
        """순정 Anima.get_learned_conditioning 이 긍정 프롬프트에서 하는 레퍼런스 인계를 대신 한다.

        v1/v2 경로는 순정 함수를 부르지 않는데, 샘플링 때 DiT 가 읽는 dynamic_args.ref_latents 를
        채우는 곳은 그 함수(backend/diffusion_engine/anima.py) 뿐이다. 건너뛰면 img2img 캔버스
        (ini_latent)·ImageStitch 레퍼런스가 조용히 빠진다. 순정과 같은 순서로 옮긴다.
        """
        dynamic_args, opts = _anima_reference_state()
        if not getattr(opts, "anima_do_reference", False):
            dynamic_args.ref_latents.clear()
            return
        references = [*getattr(sd_model, "ref_latents", [])]
        if getattr(sd_model, "ini_latent", None) is not None:
            references.insert(0, sd_model.ini_latent)
            sd_model.ini_latent = None
        dynamic_args.ref_latents = references.copy()

    @staticmethod
    def _checkpoint_path(sd_model) -> str:
        info = getattr(sd_model, "sd_checkpoint_info", None)
        path = getattr(info, "filename", None) or getattr(sd_model, "filename", None)
        if not path:
            raise RuntimeError("Could not determine the selected Anima checkpoint.")
        return str(path)

    def is_v2_bundle(self, sd_model) -> bool:
        try:
            return bundle_metadata(self._checkpoint_path(sd_model)) is not None
        except RuntimeError:
            return False

    @staticmethod
    def _same_patcher(left, right) -> bool:
        if left is right:
            return True
        if left is None or right is None:
            return False
        try:
            return left.is_clone(right) or right.is_clone(left)
        except Exception:
            return False

    @classmethod
    def _unload_patchers(cls, *patchers) -> None:
        targets = [patcher for patcher in patchers if patcher is not None]
        if not targets:
            return
        removed = False
        loaded_models = memory_management.current_loaded_models
        for index in range(len(loaded_models) - 1, -1, -1):
            loaded = loaded_models[index]
            patcher = loaded.model
            if not any(cls._same_patcher(patcher, target) for target in targets):
                continue
            loaded.model_unload()
            loaded_models.pop(index)
            removed = True
        if removed:
            memory_management.soft_empty_cache()

    @_outside_inference_mode
    def _load_qwen(self) -> tuple[Qwen35HybridModel, Qwen35Tokenizer, CLIP]:
        path = self._qwen35_path()
        if self._qwen_path == path and self._qwen is not None:
            return self._qwen, self._tokenizer, self._qwen_clip

        logger.info("Loading Qwen3.5 4B from %s", path)
        state = load_file(path, device="cpu")
        dtype = next(
            (
                state[key].dtype
                for key in (
                    "norm.1.weight",
                    "layers.0.input_layernorm.weight",
                    "model.norm.weight",
                )
                if key in state
            ),
            torch.bfloat16,
        )
        with torch.device("meta"):
            with using_forge_operations(
                device=torch.device("meta"),
                dtype=dtype,
                manual_cast_enabled=True,
            ):
                model = Qwen35HybridModel(
                    dtype=dtype,
                    device=None,
                    operations=ForgeOperations,
                )
        incompatible = model.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Qwen3.5 checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del state
        model.eval()
        tokenizer = Qwen35Tokenizer()
        clip = CLIP(model_dict={"qwen35_4b": model}, tokenizer_dict={})
        self._qwen_path = path
        self._qwen = model
        self._tokenizer = tokenizer
        self._qwen_clip = clip
        self._adapter_key = None
        self._adapter = None
        return model, tokenizer, clip

    @_outside_inference_mode
    def _load_adapter(self, name: str, native_adapter) -> ProgressiveCrossAdapter:
        choices = adapters()
        path = choices.get(name)
        if path is None:
            raise FileNotFoundError(
                f"Adapter '{name}' is unavailable. Refresh Forge and select it again."
            )
        key = (path, id(native_adapter))
        if (
            key == self._adapter_key
            and getattr(self._adapter, "native_adapter", None) is native_adapter
        ):
            return self._adapter

        adapter = ProgressiveCrossAdapter(native_adapter)
        state = load_file(path, device="cpu")
        state["parameter_bank.layer_mix_logits"] = state.pop("layer_mix_logits")
        incompatible = adapter.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Adapter checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del state
        adapter.eval().requires_grad_(False)
        self._adapter_key = key
        self._adapter = adapter
        return adapter

    @staticmethod
    def _semantic_layers(model, tokenizer, line: str, device: torch.device):
        ids = tokenizer([line])["input_ids"]
        token_ids = torch.tensor(ids, device=device, dtype=torch.long)
        attention_mask = torch.ones_like(token_ids)
        output, intermediate = model(
            token_ids,
            attention_mask=attention_mask,
            intermediate_output=list(LAYER_INDICES),
            dtype=torch.float32,
        )
        del output
        if not isinstance(intermediate, dict):
            raise RuntimeError("Qwen3.5 did not return its semantic layers.")
        return [intermediate[index] for index in LAYER_INDICES], attention_mask

    @staticmethod
    def _native_inputs(native_engine, line: str, device, dtype):
        chunks = native_engine.tokenize_line(line)
        if len(chunks) != 1:
            raise RuntimeError("Anima 3.8B expects one prompt chunk.")
        chunk = chunks[0]
        source = native_engine.process_tokens(
            [chunk.qwen_tokens],
            [chunk.qwen_multipliers],
        )[0].unsqueeze(0)
        target_ids = torch.tensor(
            chunk.t5_tokens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        target_weights = torch.tensor(
            chunk.t5_multipliers,
            device=device,
            dtype=dtype,
        ).reshape(1, -1, 1)
        return source.to(device=device, dtype=dtype), target_ids, target_weights

    def _extract_prompt_features(self, native_engine, native_clip, prompt):
        qwen, tokenizer, qwen_clip = self._load_qwen()
        native_adapter = native_clip.cond_stage_model.qwen3_06b.llm_adapter
        dtype = native_adapter.embed.weight.dtype
        offload_device = native_clip.patcher.offload_device

        native_rows = []
        memory_management.load_model_gpu(native_clip.patcher)
        try:
            device = native_clip.patcher.load_device
            for line in prompt:
                source, target_ids, target_weights = self._native_inputs(
                    native_engine,
                    str(line),
                    device,
                    dtype,
                )
                native_rows.append(
                    (
                        source.to(offload_device),
                        target_ids.to(offload_device),
                        target_weights.to(offload_device),
                    )
                )
        finally:
            self._unload_patchers(native_clip.patcher)

        lines = [str(line) for line in prompt]
        rows: dict[str, tuple] = {}
        for line in dict.fromkeys(lines):
            key = (self._qwen_path, line, dtype)
            if key in self._semantic_cache:
                self._semantic_cache.move_to_end(key)
                rows[line] = self._semantic_cache[key]
        missing = [line for line in dict.fromkeys(lines) if line not in rows]
        if missing:
            memory_management.load_model_gpu(qwen_clip.patcher)
            try:
                device = qwen_clip.patcher.load_device
                for line in missing:
                    semantic, semantic_mask = self._semantic_layers(
                        qwen,
                        tokenizer,
                        line,
                        device,
                    )
                    rows[line] = (
                        [state.to(offload_device, dtype=dtype) for state in semantic],
                        semantic_mask.to(offload_device),
                    )
                    self._semantic_cache[(self._qwen_path, line, dtype)] = rows[line]
                    while len(self._semantic_cache) > SEMANTIC_CACHE_LINES:
                        self._semantic_cache.popitem(last=False)
            finally:
                self._unload_patchers(qwen_clip.patcher)
        return native_adapter, native_rows, [rows[line] for line in lines]

    @staticmethod
    def _adapter_metadata(metadata: dict[str, str]) -> dict[str, str]:
        prefix = "anima_v2_adapter_"
        return {
            name[len(prefix) :]: value
            for name, value in metadata.items()
            if name.startswith(prefix)
        }

    @_outside_inference_mode
    def _load_v2_models(
        self,
        sd_model,
        bundle_path: str,
        metadata: dict[str, str],
    ) -> BundledV2Models:
        _, native_clip = self._require_anima(sd_model)
        native_adapter = native_clip.cond_stage_model.qwen3_06b.llm_adapter
        key = (bundle_path, id(native_adapter))
        if (
            key == self._v2_key
            and self._v2_models is not None
            and self._v2_source is not None
            and self._v2_source() is native_adapter
        ):
            return self._v2_models

        adapter_metadata = self._adapter_metadata(metadata)
        if adapter_metadata.get("architecture") != V2_ARCHITECTURE:
            raise RuntimeError("The bundled connector is not Semantic Connector v2.")
        config = {
            "num_queries": int(adapter_metadata.get("semantic_query_tokens", "64")),
            "resampler_blocks": int(
                adapter_metadata.get("semantic_resampler_blocks", "6")
            ),
            "resampler_dim": int(
                adapter_metadata.get("semantic_resampler_dim", "2048")
            ),
            "resampler_heads": int(
                adapter_metadata.get("semantic_resampler_heads", "16")
            ),
            "mlp_hidden_dim": int(
                adapter_metadata.get(
                    "semantic_resampler_mlp_hidden_dim",
                    "5632",
                )
            ),
        }
        prefix = metadata.get("anima_v2_connector_prefix", CONNECTOR_PREFIX)
        connector_state = {}
        with safe_open(bundle_path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.startswith(prefix):
                    connector_state[name[len(prefix) :]] = checkpoint.get_tensor(name)
        if not connector_state:
            raise RuntimeError("The selected v2 bundle contains no connector tensors.")
        connector_state[
            "quality_anchor.parameter_bank.layer_mix_logits"
        ] = connector_state.pop("quality_anchor.layer_mix_logits")
        connector_state[
            "semantic_resampler.parameter_bank.query_tokens"
        ] = connector_state.pop("semantic_resampler.query_tokens")
        connector_state[
            "semantic_resampler.parameter_bank.layer_embeddings"
        ] = connector_state.pop("semantic_resampler.layer_embeddings")

        dtype = native_adapter.embed.weight.dtype
        # 커넥터는 샘플링 때 llm_adapter 를 다시 돌린다. TE 모듈을 같이 쓰면 LoRA 의 llm_adapter 몫을 옮겨
        # 걸 수 없다(두 패처가 같은 가중치를 제자리 패치해 이중 적용) — 번들 원본으로 만든 사본을 쓴다.
        connector_adapter = self._private_native_adapter(native_adapter, bundle_path, dtype)
        if connector_adapter is None:
            connector_adapter = native_adapter   # 번들에 원본이 없다 — LoRA 의 llm_adapter 몫은 빠진다
        with torch.device("meta"):
            with using_forge_operations(
                device=torch.device("meta"),
                dtype=dtype,
                manual_cast_enabled=True,
            ):
                connector = QualityAnchoredSemanticConnectorV2(
                    native_adapter=connector_adapter,
                    **config,
                )
        incompatible = connector.load_state_dict(
            connector_state,
            strict=True,
            assign=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Bundled v2 connector mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del connector_state
        connector.eval().requires_grad_(False)
        models = BundledV2Models(connector_adapter, connector)
        models.eval().requires_grad_(False)
        self._v2_key = key
        self._v2_models = models
        self._v2_source = weakref.ref(native_adapter)
        return models

    @staticmethod
    def _private_native_adapter(native_adapter, bundle_path: str, dtype):
        """번들의 원본 가중치로 만든 커넥터 전용 llm_adapter. 만들 수 없으면 None (TE 모듈을 같이 쓴다).

        CPU 에서 만든다 — RotaryEmbedding 의 inv_freq 는 state dict 에 없는(persistent=False) 버퍼라
        meta 에서 만들면 복구되지 않는다.
        """
        try:
            # TE 모듈과 파라미터마다 같은 dtype — Forge 는 임베딩만 fp32 로 두고 나머지는 저장 dtype 이다
            # (통째로 embed dtype 으로 만들면 RAM·VRAM 이 약 200 MB 더 든다)
            dtypes = {name: parameter.dtype for name, parameter in native_adapter.named_parameters()}
            state = {}
            with safe_open(bundle_path, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if name.startswith(NATIVE_ADAPTER_PREFIX):
                        key = name[len(NATIVE_ADAPTER_PREFIX) :]
                        state[key] = checkpoint.get_tensor(name).to(dtypes.get(key, dtype))
            if not state:
                return None
            with using_forge_operations(
                device=torch.device("cpu"),
                dtype=dtype,
                manual_cast_enabled=True,
            ):
                private = type(native_adapter)()
            incompatible = private.load_state_dict(state, strict=False, assign=True)
            parameters = {name for name, _ in private.named_parameters()}
            if incompatible.unexpected_keys or not parameters <= set(state):
                logger.warning("[Anima38] bundle llm_adapter does not match; sharing the TE module")
                return None
            return private.eval().requires_grad_(False)
        except Exception as exc:  # pragma: no cover - Forge 버전 차이
            logger.warning("[Anima38] private llm_adapter unavailable (%s); sharing the TE module", exc)
            return None

    def _sync_adapter_lora(self, native_clip) -> None:
        """TE 에 걸린 LoRA 의 llm_adapter 몫을 커넥터 사본의 패처로 옮긴다 (LoRA 활성화 뒤, 조건 인코딩 때).

        패치가 바뀔 때만 patches_uuid 를 바꿔 Forge 가 다음 로드 때 사본을 다시 패치하게 한다.
        커넥터가 TE 모듈을 같이 쓰는 경우엔 옮기지 않는다 — 두 패처의 이중 적용이 된다.
        """
        patcher = self._v2_sampling_patcher
        models = self._v2_models
        if patcher is None or models is None or self._v2_source is None:
            return
        if models.native_adapter is self._v2_source():
            return
        wanted = {
            "native_adapter." + key[len(TE_ADAPTER_PATCH_PREFIX) :]: list(value)
            for key, value in getattr(native_clip.patcher, "patches", {}).items()
            if key.startswith(TE_ADAPTER_PATCH_PREFIX)
        }
        current = {key: value for key, value in patcher.patches.items() if key.startswith("native_adapter.")}

        def identities(patches):   # 패치 튜플 안에 텐서가 있어 == 로 비교할 수 없다
            return {key: tuple(map(id, value)) for key, value in patches.items()}

        if identities(wanted) == identities(current):
            return
        for key in current:
            del patcher.patches[key]
        patcher.patches.update(wanted)
        patcher.patches_uuid = uuid.uuid4()

    def _register_v2_run(self, run: _V2Run) -> int:
        self._v2_run_counter += 1
        self._v2_runs[self._v2_run_counter] = run
        return self._v2_run_counter

    def _encode_v2(self, native_engine, native_clip, prompt):
        self._sync_adapter_lora(native_clip)
        _, native_rows, semantic_rows = self._extract_prompt_features(
            native_engine,
            native_clip,
            prompt,
        )
        placeholders = []
        run_ids = []
        for native, semantic in zip(native_rows, semantic_rows):
            source, target_ids, target_weights = native
            semantic_states, semantic_mask = semantic
            run_id = self._register_v2_run(
                _V2Run(
                    source=source,
                    target_ids=target_ids[:, :512],
                    target_weights=target_weights[:, :512],
                    semantic=semantic_states,
                    semantic_mask=semantic_mask,
                )
            )
            placeholder = source[:, :512]
            if placeholder.shape[1] < 512:
                placeholder = F.pad(
                    placeholder,
                    (0, 0, 0, 512 - placeholder.shape[1]),
                )
            # 줄마다 텐서 하나 — Forge 네이티브 계약. run id 는 마지막 토큰 행의 마커로 실어
            # 보내므로 NegPiP 처럼 이 함수를 감싸 list 를 기대하는 확장과도 함께 돈다.
            placeholders.append(stamp_run_id(placeholder, run_id))
            run_ids.append(run_id)
        return placeholders

    def _expand_v2_context(
        self,
        context: torch.Tensor,
        timesteps: torch.Tensor,
        run_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._v2_models is None:
            raise RuntimeError("The anima.3-8B-v2 connector is not loaded.")
        connector = self._v2_models.connector
        flat_ids = run_ids.reshape(-1).to(dtype=torch.long)
        if flat_ids.numel() != context.shape[0]:
            repeats = (context.shape[0] + flat_ids.numel() - 1) // flat_ids.numel()
            flat_ids = flat_ids.repeat(repeats)[: context.shape[0]]
        timestep_rows = timesteps.reshape(-1)
        if timestep_rows.numel() != context.shape[0]:
            repeats = (
                context.shape[0] + timestep_rows.numel() - 1
            ) // timestep_rows.numel()
            timestep_rows = timestep_rows.repeat(repeats)[: context.shape[0]]

        outputs: list[torch.Tensor | None] = [None] * context.shape[0]
        dtype = self._v2_models.native_adapter.embed.weight.dtype
        for run_id in flat_ids.unique().tolist():
            if int(run_id) < 0:
                # 마커 없는 행(네거티브 프롬프트 등 순정 경로)은 그대로 둔다
                for row_index in (flat_ids == run_id).nonzero(as_tuple=False).reshape(-1).tolist():
                    outputs[row_index] = context[row_index : row_index + 1]
                continue
            run = self._v2_runs.get(int(run_id))
            if run is None:
                raise RuntimeError(
                    f"Anima v2 conditioning run {run_id} is unavailable; "
                    "re-encode the prompt."
                )
            indices = (flat_ids == run_id).nonzero(as_tuple=False).reshape(-1)
            count = indices.numel()
            source = run.source.to(context.device, dtype=dtype).expand(count, -1, -1)
            target_ids = run.target_ids.to(context.device).expand(count, -1)
            semantic = [
                state.to(context.device, dtype=dtype).expand(count, -1, -1)
                for state in run.semantic
            ]
            semantic_mask = run.semantic_mask.to(context.device).expand(count, -1)
            expanded = connector(
                source,
                target_ids,
                semantic,
                semantic_source_mask=semantic_mask,
                timesteps=timestep_rows[indices].to(context.device),
            )
            weights = run.target_weights.to(
                context.device,
                dtype=expanded.dtype,
            ).expand(count, -1, -1)
            expanded = expanded * weights[:, : expanded.shape[1]]
            if expanded.shape[1] < 512:
                expanded = F.pad(
                    expanded,
                    (0, 0, 0, 512 - expanded.shape[1]),
                )
            for output_index, row_index in enumerate(indices.tolist()):
                outputs[row_index] = expanded[output_index : output_index + 1]
        return torch.cat(outputs).to(dtype=context.dtype)

    @staticmethod
    def _offload_conditioning(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [Anima3BRuntime._offload_conditioning(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(
                Anima3BRuntime._offload_conditioning(item, device) for item in value
            )
        if isinstance(value, dict):
            return {
                key: Anima3BRuntime._offload_conditioning(item, device)
                for key, item in value.items()
            }
        return value

    @torch.inference_mode()
    def encode(
        self,
        sd_model,
        prompt,
        adapter_name: str,
        strength: float,
        negative_strength: float | None,
    ):
        native_engine, native_clip = self._require_anima(sd_model)
        original = self._native_conditioning(sd_model)
        is_negative = bool(getattr(prompt, "is_negative_prompt", False))
        if is_negative:
            strength = negative_strength

        if strength is None or strength == 0.0:
            try:
                result = original(prompt)
                return self._offload_conditioning(
                    result,
                    native_clip.patcher.offload_device,
                )
            finally:
                self._unload_patchers(native_clip.patcher)

        if not is_negative:
            self._hand_off_native_reference(sd_model)
        if self._active_bundle_metadata is not None:
            return self._encode_v2(native_engine, native_clip, prompt)

        native_adapter, native_rows, semantic_rows = self._extract_prompt_features(
            native_engine,
            native_clip,
            prompt,
        )
        memory_management.load_model_gpu(native_clip.patcher)
        try:
            device = native_clip.patcher.load_device
            dtype = native_adapter.embed.weight.dtype
            adapter = self._load_adapter(adapter_name, native_adapter)
            adapter.to(device=device, dtype=dtype)
            outputs = []
            for native, semantic in zip(native_rows, semantic_rows):
                source, target_ids, target_weights = native
                semantic_states, semantic_mask = semantic
                source = source.to(device=device, dtype=dtype)
                target_ids = target_ids.to(device=device)
                target_weights = target_weights.to(device=device, dtype=dtype)
                semantic_states = [
                    state.to(device=device, dtype=dtype) for state in semantic_states
                ]
                semantic_mask = semantic_mask.to(device=device)
                expanded = adapter(
                    source,
                    target_ids[:, :512],
                    semantic_states,
                    semantic_mask=semantic_mask,
                )
                if strength != 1.0:
                    native_context = native_adapter(source, target_ids[:, :512])
                    expanded = native_context + float(strength) * (
                        expanded - native_context
                    )
                expanded = expanded * target_weights[:, : expanded.shape[1]]
                if expanded.shape[1] < 512:
                    expanded = F.pad(
                        expanded,
                        (0, 0, 0, 512 - expanded.shape[1]),
                    )
                outputs.append(expanded.to(native_clip.patcher.offload_device))
            return outputs
        finally:
            self._unload_patchers(native_clip.patcher)

    def _install_v2(
        self,
        processing,
        path: str,
        metadata: dict[str, str],
    ) -> None:
        models = self._load_v2_models(processing.sd_model, path, metadata)
        unet = processing.sd_model.forge_objects.unet
        self._v2_sampling_patcher = unet.add_extra_torch_module_during_sampling(
            models,
            cast_to_unet_dtype=False,
        )
        self._v2_sampling_unets = []
        self._attach_sampling_patcher(processing.sd_model)
        self._v2_active = True
        self._wrap_forward(unet.model.diffusion_model)

    def _attach_sampling_patcher(self, sd_model) -> None:
        """LoRA 세트가 바뀌면 Forge 는 forge_objects_original.unet 을 복제해 새 UNet 으로 샘플링한다 (설치가
        process_images 보다 앞서는 Feature 6, batch count 사이 LoRA 가 바뀌는 경우). 복제는 extra 패처
        목록을 복사하므로 원본·LoRA 적용본에도 같은 패처를 달아 두어야 커넥터가 메모리 관리 안에 든다.
        processing.sd_model 은 shared.sd_model 을 따라가 나중에 다른 체크포인트를 가리킬 수 있어 기록해 둔다
        (weakref — 샘플링이 죽어 restore 가 안 불려도 이전 모델을 붙잡지 않게)."""
        for owner in self._unet_slots(sd_model):
            if not any(item is self._v2_sampling_patcher for item in owner.extra_model_patchers_during_sampling):
                owner.extra_model_patchers_during_sampling.append(self._v2_sampling_patcher)
            if not any(ref() is owner for ref in self._v2_sampling_unets):
                self._v2_sampling_unets.append(weakref.ref(owner))

    def _wrap_forward(self, diffusion_model) -> None:
        if self._patched(diffusion_model.forward) or self._patched(
            getattr(diffusion_model, "orig_forward", None)
        ):
            return   # 우리 패치가 이미 체인에 있다 (NegPiP 가 위에 있을 수도) — 다시 감싸지 않는다
        native_forward = self._native_forward(diffusion_model)

        def patched_forward(x, timesteps, context, *args, **kwargs):
            # *args: NegPiP 의 dit.forward 래퍼는 padding_mask 를 위치 인자로 넘긴다.
            # 항상 클래스의 순정 forward 를 부른다 — 설치 시점에 잡아 둔 "이전 forward" 는
            # 다른 확장이 비중첩으로 되돌린 뒤 낡은 래퍼일 수 있다.
            run_ids = kwargs.pop("y", None)   # 옛 dict/vector 경로 호환
            negpip_mask_in = kwargs.get("c_negpip_mask")
            if negpip_mask_in is not None:
                # NegPiP 의 dit.forward 훅이 우리 *아래* 에 있으면(NegPiP 먼저 설치) 클래스 forward 로
                # 바로 가면서 건너뛰게 된다 — 그 훅이 하는 일(transformer_options 에 마스크 전달)을 대신 한다.
                # NegPiP 가 위에 있으면 이미 넣어 두었으므로 건드리지 않는다.
                options = kwargs.get("transformer_options") or {}
                if options.get("negpip_mask") is None:
                    options["negpip_mask"] = negpip_mask_in
                    kwargs["transformer_options"] = options
            if self._v2_active and self._v2_models is not None:
                # reconstruct_cond_batch 는 줄 텐서 [1,512,C] 를 stack 해 [B,1,512,C] 로 넘긴다 —
                # [B,512,C] 뷰에서 마커를 읽고 확장한 뒤 원래 모양으로 되돌린다.
                rows = as_rows(context)
                if run_ids is None:
                    run_ids = read_run_ids(rows)
                if bool((run_ids >= 0).any()):
                    expanded = self._expand_v2_context(rows, timesteps, run_ids)
                    negpip_mask = kwargs.get("c_negpip_mask")
                    if negpip_mask is not None:
                        # NegPiP 는 조건 텐서에 ±1 을 곱해 두고(K 부호) 어텐션에서 V 에 한 번 더 곱한다.
                        # 확장이 자리표시 행을 통째로 갈아끼우므로 확장된 행에도 같은 마스크를 곱해 준다.
                        mask = negpip_mask.reshape(negpip_mask.shape[0], -1, 1).to(expanded)
                        if mask.shape[0] != expanded.shape[0]:
                            mask = mask.repeat((expanded.shape[0] + mask.shape[0] - 1) // mask.shape[0], 1, 1)[: expanded.shape[0]]
                        marked = (run_ids.reshape(-1) >= 0).to(expanded.device)
                        if marked.numel() != expanded.shape[0]:
                            marked = marked.repeat((expanded.shape[0] + marked.numel() - 1) // marked.numel())[: expanded.shape[0]]
                        expanded = torch.where(marked.reshape(-1, 1, 1), expanded * mask[:, : expanded.shape[1]], expanded)
                    context = expanded.reshape(context.shape)
            return native_forward(x, timesteps, context, *args, **kwargs)

        patched_forward._anima3b_patch = self
        self._wrappers.add(patched_forward)
        diffusion_model.forward = patched_forward

    def install(
        self,
        processing,
        adapter_name: str,
        strength: float,
        negative_strength: float | None,
    ) -> None:
        if self._installed_processing is not None:
            self.restore(self._installed_processing)
        self._require_anima(processing.sd_model)
        checkpoint_path = self._checkpoint_path(processing.sd_model)
        metadata = bundle_metadata(checkpoint_path)
        self._require_encoder_files(adapter_name if metadata is None else None)
        self._installed_processing = processing
        self._installed_model = self._weak(processing.sd_model)
        self._active_bundle_path = checkpoint_path if metadata is not None else None
        self._active_bundle_metadata = metadata
        self._v2_runs.clear()

        sd_model = processing.sd_model
        if metadata is not None:
            self._install_v2(processing, checkpoint_path, metadata)
            strength = 1.0
            if negative_strength is not None:
                negative_strength = 1.0
        self._cond_params = (adapter_name, strength, negative_strength)
        self._cond_active = True
        self._wrap_conditioning(sd_model)
        # 순정 경로로 가는 부정 조건엔 마커가 없다 — Forge 의 공용 캐시(persistent_cond_cache)를 그대로 쓴다
        self._reset_cond_caches(processing, include_negative=negative_strength is not None)
        if metadata is not None:
            adapter_file = metadata.get(
                "anima_v2_adapter_filename",
                Path(checkpoint_path).name,
            )
            # infotext 키는 Forge 파서(re_param_code: [\w\s\-/])가 읽을 수 있어야 붙여 넣기가 된다 — '.' 금지
            processing.extra_generation_params.update(
                {
                    "Anima38 adapter": adapter_file,
                    "Anima38 strength": 1.0,
                    "Anima38 architecture": V2_ARCHITECTURE,
                    "Anima38 bundle": Path(checkpoint_path).name,
                    # 공식 v1.1 워크플로는 부정 프롬프트도 커넥터로 인코딩한다 — 어느 쪽이었는지 남긴다
                    "Anima38 negative": "connector" if negative_strength is not None else "native",
                }
            )
        else:
            processing.extra_generation_params.update(
                {
                    "Anima38 adapter": Path(adapter_name).name,
                    "Anima38 strength": float(strength),
                    "Anima38 architecture": ARCHITECTURE,
                }
            )
        processing.extra_generation_params["Anima38 encoder"] = Path(self._qwen35_path()).name
        if negative_strength is not None:
            processing.extra_generation_params[
                "Anima38 negative strength"
            ] = float(negative_strength)

    @staticmethod
    def _qwen35_path() -> str:
        """쓸 Qwen3.5 파일 — models/text_encoder 에서 자동으로 찾는다 (VAE/Text Encoder 목록과 무관)."""
        choices = qwen35_models()
        if not choices:
            raise FileNotFoundError(
                "qwen35_4b.safetensors was not found in models/text_encoder."
            )
        return choices.get("qwen35_4b.safetensors") or next(iter(choices.values()))

    def _wrap_conditioning(self, sd_model) -> None:
        if self._cond_installed(sd_model):
            return
        below = sd_model.get_learned_conditioning

        def patched(prompt):
            return self._conditioning_entry(sd_model, prompt)

        patched._anima3b_patch = self
        patched._anima3b_below = below   # 설치 당시 우리 아래에 있던 것 (NegPiP 래퍼일 수 있다)
        self._wrappers.add(patched)
        sd_model.get_learned_conditioning = patched

    @staticmethod
    def _weak(obj):
        try:
            return weakref.ref(obj)
        except TypeError:
            return None

    def install_is_current(self, processing) -> bool:
        """이 processing 을 위한 설치가 지금 모델 위에 그대로 있는가. Forge 는 Hires 체크포인트·Refiner 가
        있으면 batch count 반복마다 1차 모델을 새로 불러온다 — 그러면 다시 설치해야 한다."""
        if self._installed_processing is not processing or self._installed_model is None:
            return False
        return self._installed_model() is processing.sd_model

    def ensure_attached(self, processing) -> None:
        """설치를 내리지 않고(run·조건 캐시 유지) 체인에서 빠진 패치만 다시 건다. NegPiP 가 앞 순서면
        반복·안쪽 패스마다 자기 패치를 원복하면서 우리 래퍼까지 떨군다."""
        sd_model = processing.sd_model
        if self._cond_active:
            self._wrap_conditioning(sd_model)
        if self._v2_active and self._v2_sampling_patcher is not None:
            self._attach_sampling_patcher(sd_model)
            self._wrap_forward(sd_model.forge_objects.unet.model.diffusion_model)

    @classmethod
    def _require_encoder_files(cls, v1_adapter_name: str | None) -> None:
        """Qwen3.5 는 setup_conds 에서 게으르게 로드된다 — 파일이 없으면 무엇이든 패치하기 전에 여기서
        실패해야 스크립트가 경고 한 번 후 순정 Anima 로 진행한다(아니면 생성이 샘플링 직전에 죽는다)."""
        cls._qwen35_path()
        tokenizer_dir()   # 동봉 토크나이저가 없으면 FileNotFoundError
        if v1_adapter_name is not None and v1_adapter_name not in adapters():
            raise FileNotFoundError(
                f"Adapter '{v1_adapter_name}' is unavailable. Refresh Forge and select it again."
            )

    @staticmethod
    def _unet_slots(sd_model) -> list:
        """지금 샘플링용·LoRA 적용본·원본 UNet (겹치면 하나로)."""
        unets = []
        for slot in ("forge_objects", "forge_objects_after_applying_lora", "forge_objects_original"):
            unet = getattr(getattr(sd_model, slot, None), "unet", None)
            if unet is not None and not any(unet is seen for seen in unets):
                unets.append(unet)
        return unets

    def release_stale_caches(self, sd_model) -> None:
        """다른 모델이 로드되면 이전 모델의 llm_adapter 에 묶인 커넥터·v1 어댑터를 놓는다 (on_model_loaded).
        Qwen3.5 는 체크포인트와 무관해 그대로 캐시한다 — XYZ 로 3.8B 를 오갈 때 4.8 GB 를 다시 읽지 않게."""
        try:
            native_adapter = sd_model.forge_objects.clip.cond_stage_model.qwen3_06b.llm_adapter
        except AttributeError:
            native_adapter = None
        source = self._v2_source() if self._v2_source is not None else None
        if source is None or source is not native_adapter:
            self._v2_models = None
            self._v2_key = None
            self._v2_source = None
        if getattr(self._adapter, "native_adapter", None) is not native_adapter:
            self._adapter = None
            self._adapter_key = None

    def restore(self, processing) -> None:
        model = processing.sd_model
        installed_processing = self._installed_processing
        # 패치는 그대로 두고 플래그만 끈다. 우리 클로저는 꺼진 채 불려도 순정(또는 아래 래퍼)으로
        # 위임하므로 안전하고, 저장해 둔 '원본'(= 다른 확장의 래퍼일 수 있다)을 되돌려 놓다가
        # 이미 해제된 래퍼를 되살리는 사고를 원천적으로 막는다.
        self._cond_active = False
        self._cond_params = None
        self._v2_active = False
        if self._v2_sampling_patcher is not None:
            owners = [owner for owner in (ref() for ref in self._v2_sampling_unets) if owner is not None]
            # 설치 뒤 생긴 LoRA 복제본도 같은 패처를 복사해 갔다 — 지금 모델의 UNet 자리들도 비운다
            for unet in self._unet_slots(model):
                if not any(unet is owner for owner in owners):
                    owners.append(unet)
            for owner in owners:
                owner.extra_model_patchers_during_sampling = [
                    patcher
                    for patcher in owner.extra_model_patchers_during_sampling
                    if patcher is not self._v2_sampling_patcher
                ]
            self._unload_patchers(self._v2_sampling_patcher)
        self._v2_sampling_patcher = None
        self._v2_sampling_unets = []
        self._installed_processing = None
        self._installed_model = None
        self._active_bundle_path = None
        self._active_bundle_metadata = None
        self._v2_runs.clear()
        self._v2_run_counter = 0   # 마커 20비트 한도 — 캐시를 비웠으니 0 부터 다시
        self._reset_cond_caches(processing)
        if installed_processing is not None and installed_processing is not processing:
            self._reset_cond_caches(installed_processing)

    @staticmethod
    def _reset_cond_caches(processing, include_negative: bool = True) -> None:
        """마커가 든 조건이 다음 생성(또는 hires 2차 패스)에 재사용되지 않게 이 생성 전용 캐시로 바꾼다."""
        names = ("cached_c", "cached_hr_c")
        if include_negative:
            names += ("cached_uc", "cached_hr_uc")
        for name in names:
            if hasattr(processing, name):
                setattr(processing, name, [None, None, None])


_SHARED_RUNTIME: Anima3BRuntime | None = None


def shared_runtime() -> Anima3BRuntime:
    """Return the process-wide runtime.

    The runtime owns the loaded Qwen3.5 encoder, so txt2img's Anima38 script
    and Feature 6 must share one instance instead of loading it twice.
    """

    global _SHARED_RUNTIME
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = Anima3BRuntime()
    return _SHARED_RUNTIME
