from __future__ import annotations

import functools
import logging
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
)
from .qwen35 import Qwen35HybridModel
from .semantic_v2 import BundledV2Models, QualityAnchoredSemanticConnectorV2
from .tokenizer import Qwen35Tokenizer

logger = logging.getLogger(__name__)
LAYER_INDICES = (7, 15, 23, 31)


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
        self._v2_sampling_patcher = None
        self._v2_sampling_unet = None
        self._installed_processing = None
        self._v2_diffusion_model = None
        self._v2_original_forward = None
        self._v2_runs: dict[int, _V2Run] = {}
        self._v2_run_counter = 0
        # 다른 확장(NegPiP 등)이 같은 속성을 감싸고 비중첩으로 되돌릴 수 있으므로, 우리 패치는
        # "속성 교체" 가 아니라 이 플래그로 켜고 끈다. 꺼진 채 남아 있어도 순정으로 위임한다.
        self._cond_active = False
        self._cond_params: tuple[str, float, float | None] | None = None
        self._v2_active = False
        self._cond_wrapper = None    # 우리가 설치한 클로저 (체인에 남아 있는지 확인용)
        self._v2_forward_wrapper = None
        self._cond_below = None      # 설치 당시 우리 아래에 있던 것 (NegPiP 래퍼일 수 있다)

    def _cond_installed(self, sd_model) -> bool:
        """우리 클로저가 조건 체인에 (맨 위든 한 단계 아래든) 살아 있는가."""
        if self._cond_wrapper is None:
            return False
        current = getattr(sd_model, "get_learned_conditioning", None)
        if current is self._cond_wrapper:
            return True
        # NegPiP 가 우리 위에 있으면 우리를 orig_forward 에 담아 둔다
        return getattr(sd_model, "orig_forward", None) is self._cond_wrapper

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
            below = self._cond_below
            if below is not None and below is not self._cond_wrapper and self._negpip_patched(sd_model):
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
            and sd_model.get_learned_conditioning is self._cond_wrapper
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
        choices = qwen35_models()
        if not choices:
            raise FileNotFoundError(
                "qwen35_4b.safetensors was not found in models/text_encoder."
            )
        path = choices.get("qwen35_4b.safetensors") or next(iter(choices.values()))
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
        if key == self._adapter_key and self._adapter is not None:
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

        semantic_rows = []
        memory_management.load_model_gpu(qwen_clip.patcher)
        try:
            device = qwen_clip.patcher.load_device
            for line in prompt:
                semantic, semantic_mask = self._semantic_layers(
                    qwen,
                    tokenizer,
                    str(line),
                    device,
                )
                semantic_rows.append(
                    (
                        [state.to(offload_device, dtype=dtype) for state in semantic],
                        semantic_mask.to(offload_device),
                    )
                )
        finally:
            self._unload_patchers(qwen_clip.patcher)
        return native_adapter, native_rows, semantic_rows

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
        if key == self._v2_key and self._v2_models is not None:
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
        with torch.device("meta"):
            with using_forge_operations(
                device=torch.device("meta"),
                dtype=dtype,
                manual_cast_enabled=True,
            ):
                connector = QualityAnchoredSemanticConnectorV2(
                    native_adapter=native_adapter,
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
        models = BundledV2Models(native_adapter, connector)
        models.eval().requires_grad_(False)
        self._v2_key = key
        self._v2_models = models
        return models

    def _register_v2_run(self, run: _V2Run) -> int:
        self._v2_run_counter += 1
        self._v2_runs[self._v2_run_counter] = run
        return self._v2_run_counter

    def _encode_v2(self, native_engine, native_clip, prompt):
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
        if getattr(prompt, "is_negative_prompt", False):
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
        # processing.sd_model follows shared.sd_model and can already point at
        # another checkpoint when a failed generation is cleaned up later.
        self._v2_sampling_unet = unet
        self._v2_sampling_patcher = unet.add_extra_torch_module_during_sampling(
            models,
            cast_to_unet_dtype=False,
        )
        diffusion_model = unet.model.diffusion_model
        self._v2_diffusion_model = diffusion_model
        self._v2_active = True
        current = diffusion_model.forward
        wrapper = self._v2_forward_wrapper
        if wrapper is not None and (
            current is wrapper or getattr(diffusion_model, "orig_forward", None) is wrapper
        ):
            return   # 우리 패치가 이미 체인에 있다 (NegPiP 가 위에 있을 수도) — 다시 감싸지 않는다
        self._v2_original_forward = current
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
        self._v2_forward_wrapper = patched_forward
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
        self._installed_processing = processing
        checkpoint_path = self._checkpoint_path(processing.sd_model)
        metadata = bundle_metadata(checkpoint_path)
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
        if not self._cond_installed(sd_model):
            self._cond_below = sd_model.get_learned_conditioning

            def patched(prompt):
                return self._conditioning_entry(sd_model, prompt)

            patched._anima3b_patch = self
            self._cond_wrapper = patched
            sd_model.get_learned_conditioning = patched
        self._reset_cond_caches(processing)
        if metadata is not None:
            adapter_file = metadata.get(
                "anima_v2_adapter_filename",
                Path(checkpoint_path).name,
            )
            processing.extra_generation_params.update(
                {
                    "Anima 3.8B adapter": adapter_file,
                    "Anima 3.8B strength": 1.0,
                    "Anima 3.8B architecture": V2_ARCHITECTURE,
                    "Anima 3.8B bundle": Path(checkpoint_path).name,
                }
            )
        else:
            processing.extra_generation_params.update(
                {
                    "Anima 3.8B adapter": Path(adapter_name).name,
                    "Anima 3.8B strength": float(strength),
                    "Anima 3.8B architecture": ARCHITECTURE,
                }
            )
        if negative_strength is not None:
            processing.extra_generation_params[
                "Anima 3.8B negative strength"
            ] = float(negative_strength)

    def restore(self, processing) -> None:
        model = processing.sd_model
        installed_processing = self._installed_processing
        # 패치는 그대로 두고 플래그만 끈다. 우리 클로저는 꺼진 채 불려도 순정(또는 아래 래퍼)으로
        # 위임하므로 안전하고, 저장해 둔 '원본'(= 다른 확장의 래퍼일 수 있다)을 되돌려 놓다가
        # 이미 해제된 래퍼를 되살리는 사고를 원천적으로 막는다.
        self._cond_active = False
        self._cond_params = None
        self._v2_active = False
        unet = getattr(getattr(model, "forge_objects", None), "unet", None)
        if self._v2_sampling_patcher is not None:
            owners = [self._v2_sampling_unet]
            if unet is not self._v2_sampling_unet:
                # A sampling clone may have copied the same extra patcher.
                owners.append(unet)
            for owner in owners:
                if owner is not None:
                    owner.extra_model_patchers_during_sampling = [
                        patcher
                        for patcher in owner.extra_model_patchers_during_sampling
                        if patcher is not self._v2_sampling_patcher
                    ]
            self._unload_patchers(self._v2_sampling_patcher)
        self._v2_sampling_patcher = None
        self._v2_sampling_unet = None
        self._installed_processing = None
        self._active_bundle_path = None
        self._active_bundle_metadata = None
        self._v2_runs.clear()
        self._v2_run_counter = 0   # 마커 20비트 한도 — 캐시를 비웠으니 0 부터 다시
        self._reset_cond_caches(processing)
        if installed_processing is not None and installed_processing is not processing:
            self._reset_cond_caches(installed_processing)

    @staticmethod
    def _reset_cond_caches(processing) -> None:
        """마커가 든 조건이 다음 생성(또는 hires 2차 패스)에 재사용되지 않게 네 캐시를 모두 비운다."""
        for name in ("cached_c", "cached_uc", "cached_hr_c", "cached_hr_uc"):
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
