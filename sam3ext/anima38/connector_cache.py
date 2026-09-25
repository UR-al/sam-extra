"""Semantic Connector v2 의 run 별 캐시 — timestep 과 무관한 계산을 run(프롬프트 줄)마다 한 번만 한다.

커넥터는 스텝마다 같은 run 을 받아 timestep 만 바뀐 채 다시 돈다(semantic_v2.QualityAnchoredSemanticConnectorV2.forward).
그 안에서 timestep 과 무관한 것(코드로 확인):

- resampler 블록마다 cross-attention 의 K/V — ``source_norm(qwen_features)`` 는 의미 특징 + layer embedding 뿐이다.
  q 쪽(cross_norm·time_modulation)과 self-attention·MLP 는 timestep 에 따라 바뀐다.
- native llm_adapter 블록마다 cross-attention K/V(+RoPE) — context 는 0.6B TE 의 source.
- quality anchor 블록마다 K/V(+RoPE) — context 는 의미 특징 4층을 섞은 것(layer_mix_logits)의 source_norm.
- 첫 native 블록 전체와 첫 anchor attention — 입력이 T5 id·source·의미 특징뿐이다(v2 attention 은 그 뒤).
  두 번째 블록부터는 x 가 첫 블록의 v2 attention(= timestep 의존 semantic bank)을 거쳐 달라진다.
- RoPE cos/sin, 마스크 변환, anchor mix softmax, query token 캐스트.

여기 함수는 upstream forward 와 **같은 연산을 같은 순서·같은 모양**으로 한다 — 캐시한 텐서를 다시 쓰는 것만 다르다.
그래서 결과가 비트 단위로 같다(tests/test_anima38.py 가 실제 Forge 모듈로 upstream forward 와 torch.equal 비교).
Forge 모듈의 forward 가 바뀌었거나(버전 차이) 훅이 걸려 있으면 supports() 가 False — 부르는 쪽이 원래 forward 를 쓴다.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

# Forge 설정 — scripts/anima_3_8b.py 가 같은 이름으로 등록한다
OPT_CONNECTOR_RUN_CACHE = "sam3_anima38_connector_run_cache"

_HOOK_REGISTRIES = (
    "_global_forward_hooks",
    "_global_forward_pre_hooks",
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_forward_hooks_always_called",
)


def run_cache_enabled() -> bool:
    """run 별 캐시를 쓸까 (Forge 설정 OPT_CONNECTOR_RUN_CACHE). 설정이 없거나 Forge 밖이면 켜짐."""
    try:
        from modules import shared

        return bool(getattr(shared.opts, OPT_CONNECTOR_RUN_CACHE, True))
    except Exception:
        return True


def global_hooks_clear() -> bool:
    """전역 모듈 훅이 없는가 — 있으면 캐시 경로가 건너뛰는 모듈에서 훅이 안 불리므로 쓰지 않는다."""
    registry = torch.nn.modules.module
    return not any(getattr(registry, name, None) for name in _HOOK_REGISTRIES)


def _module_of(obj):
    return sys.modules.get(type(obj).__module__)


def _unchanged(obj, owner_module, class_name: str) -> bool:
    """obj 가 owner_module 의 class_name 이고 그 클래스 forward 를 그대로 쓰는가(인스턴스 forward 덮어쓰기 없음)."""
    klass = getattr(owner_module, class_name, None)
    return (
        klass is not None
        and type(obj) is klass
        and "forward" not in vars(obj)
        and not obj._forward_hooks
        and not obj._forward_pre_hooks
    )


def supports(connector) -> bool:
    """이 커넥터에 run 캐시를 써도 되는가 — upstream 코드와 연산이 같다는 전제를 모양으로 확인한다."""
    if not isinstance(connector, torch.nn.Module) or not global_hooks_clear():
        return False
    v2 = _module_of(connector)
    if not _unchanged(connector, v2, "QualityAnchoredSemanticConnectorV2"):
        return False
    try:
        native = connector.native_adapter
        anchor = connector.quality_anchor
        resampler = connector.semantic_resampler
        blocks = list(native.blocks)
    except AttributeError:
        return False
    if not blocks or not _unchanged(resampler, v2, "TimestepAwareSemanticResampler"):
        return False
    for block in resampler.blocks:
        if not (_unchanged(block, v2, "SemanticResamplerBlock")
                and _unchanged(block.cross_attention, v2, "MultiHeadAttention")):
            return False
    llm = _module_of(blocks[0])
    if llm is None or not all(hasattr(llm, name) for name in ("apply_rotary_pos_emb", "scaled_dot_product_attention")):
        return False
    for index, block in enumerate(blocks):
        if not (_unchanged(block, llm, "TransformerBlock") and _unchanged(block.cross_attn, llm, "Attention")):
            return False
        if not _unchanged(anchor.semantic_attentions[index], llm, "Attention"):
            return False
    # 여기서 본 모듈(캐시 경로가 forward 를 건너뛰는 것)만 훅이 없어야 한다 — 나머지는 캐시 경로도 그대로 부른다
    return True


class RunCache:
    """run 하나(같은 입력·같은 행 수)의 timestep 무관 텐서. 샘플링 장치에 있다."""

    __slots__ = (
        "count", "x0", "query_rope", "bank_rope", "target_attention_mask", "native_source_mask",
        "semantic_attention_mask", "native_kv", "anchor_kv", "resampler_kv", "queries", "feature_dtype",
    )

    def tensors(self):
        yield self.x0
        yield from self.query_rope
        yield from self.bank_rope
        for mask in (self.target_attention_mask, self.native_source_mask, self.semantic_attention_mask):
            if mask is not None:
                yield mask
        for k, v in self.native_kv + self.anchor_kv:
            yield k
            yield v
        for k, v, mask in self.resampler_kv:
            yield k
            yield v
            if mask is not None:
                yield mask
        yield self.queries

    @property
    def nbytes(self) -> int:
        """이 캐시가 붙잡는 저장 공간(같은 저장소는 한 번) — expand 뷰는 원본 크기만큼."""
        seen, total = set(), 0
        for tensor in self.tensors():
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), tensor.device)
            if key in seen:
                continue
            seen.add(key)
            total += storage.nbytes()
        return total


def _attention_kv(llm, attention, context, rope):
    """backend.nn.anima.Attention.forward 의 K/V 쪽 — 같은 연산·같은 순서."""
    context_shape = context.shape[:-1]
    kv_shape = (*context_shape, attention.n_heads, attention.head_dim)
    key_states = attention.k_norm(attention.k_proj(context).view(kv_shape)).transpose(1, 2)
    value_states = attention.v_proj(context).view(kv_shape).transpose(1, 2)
    if rope is not None:
        cos, sin = rope
        key_states = llm.apply_rotary_pos_emb(key_states, cos, sin)
    return key_states, value_states


def _attention_with_kv(llm, attention, x, key_states, value_states, mask, rope):
    """backend.nn.anima.Attention.forward 의 q·attention·출력 쪽 (K/V 는 캐시)."""
    input_shape = x.shape[:-1]
    q_shape = (*input_shape, attention.n_heads, attention.head_dim)
    query_states = attention.q_norm(attention.q_proj(x).view(q_shape)).transpose(1, 2)
    if rope is not None:
        cos, sin = rope
        query_states = llm.apply_rotary_pos_emb(query_states, cos, sin)
    attn_output = llm.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return attention.o_proj(attn_output)


def _native_block_with_kv(llm, block, x, key_states, value_states, target_attention_mask, source_attention_mask,
                          rope):
    """backend.nn.anima.TransformerBlock.forward (cross-attention K/V 는 캐시). x 는 제자리로 바뀐다(원본과 같음)."""
    if block.use_self_attn:
        normed = block.norm_self_attn(x)
        attn_out = block.self_attn(normed, mask=target_attention_mask, position_embeddings=rope,
                                   position_embeddings_context=rope)
        x.add_(attn_out)

    normed = block.norm_cross_attn(x)
    attn_out = _attention_with_kv(llm, block.cross_attn, normed, key_states, value_states,
                                  source_attention_mask, rope)
    x.add_(attn_out)

    x.add_(block.mlp(block.norm_mlp(x)))
    return x


def _mha_kv(attention, context):
    """semantic_v2.MultiHeadAttention.forward 의 K/V 쪽. batch 는 query 와 같다(커넥터는 같은 행 수로 부른다)."""
    batch, context_tokens = context.shape[0], context.shape[1]

    def heads(value):
        return value.reshape(batch, context_tokens, attention.num_heads, attention.head_dim).transpose(1, 2)

    return heads(attention.k_proj(context)), heads(attention.v_proj(context))


def _mha_mask(context_mask, batch, context_tokens):
    if context_mask is None:
        return None
    return context_mask.to(torch.bool).reshape(batch, 1, 1, context_tokens)


def _mha_with_kv(attention, query, k, v, mask):
    batch, query_tokens, _ = query.shape
    q = attention.q_proj(query).reshape(batch, query_tokens, attention.num_heads, attention.head_dim).transpose(1, 2)
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    attended = attended.transpose(1, 2).reshape(batch, query_tokens, attention.query_dim)
    return attention.o_proj(attended)


def _resampler_block_with_kv(block, queries, timestep_embedding, k, v, mask):
    """semantic_v2.SemanticResamplerBlock.forward (cross-attention K/V 는 캐시)."""
    modulation = block.time_modulation(F.silu(timestep_embedding))
    cross_scale, cross_shift, self_scale, self_shift, mlp_scale, mlp_shift = modulation.chunk(6, dim=-1)
    queries = queries + _mha_with_kv(
        block.cross_attention,
        block.cross_norm(queries, cross_scale, cross_shift),
        k,
        v,
        mask,
    )
    normalized = block.self_norm(queries, self_scale, self_shift)
    queries = queries + block.self_attention(normalized, normalized)
    normalized = block.mlp_norm(queries, mlp_scale, mlp_shift)
    gate, value = block.mlp_in(normalized).chunk(2, dim=-1)
    return queries + block.mlp_out(F.silu(gate) * value)


@torch.no_grad()
def prepare(
    connector,
    native_source: torch.Tensor,
    target_input_ids: torch.Tensor,
    semantic_hidden_states,
    target_attention_mask: torch.Tensor | None = None,
    native_source_mask: torch.Tensor | None = None,
    semantic_source_mask: torch.Tensor | None = None,
) -> RunCache:
    """run 의 timestep 무관 텐서를 계산한다 — forward(...) 와 같은 인자(timesteps 제외)."""
    if len(semantic_hidden_states) != len(connector.layer_indices):
        raise ValueError(
            f"Expected {len(connector.layer_indices)} semantic layers, "
            f"got {len(semantic_hidden_states)}"
        )
    cache = RunCache()
    native = connector.native_adapter
    anchor = connector.quality_anchor
    resampler = connector.semantic_resampler
    llm = _module_of(native.blocks[0])

    cache.target_attention_mask = connector._mask(target_attention_mask)
    cache.native_source_mask = connector._mask(native_source_mask)
    cache.semantic_attention_mask = connector._mask(semantic_source_mask)

    # --- resampler (TimestepAwareSemanticResampler.forward 의 timestep 앞부분)
    hidden_states = semantic_hidden_states
    if len(hidden_states) != resampler.num_layers:
        raise ValueError(f"Expected {resampler.num_layers} Qwen layers, got {len(hidden_states)}")
    batch = hidden_states[0].shape[0]
    layer_streams = [
        hidden
        + resampler.layer_embeddings[index].to(
            device=hidden.device,
            dtype=hidden.dtype,
        ).unsqueeze(0)
        for index, hidden in enumerate(hidden_states)
    ]
    qwen_features = torch.cat(layer_streams, dim=1)
    qwen_mask = None
    if semantic_source_mask is not None:
        qwen_mask = torch.cat([semantic_source_mask] * resampler.num_layers, dim=1)
    cache.feature_dtype = qwen_features.dtype
    cache.queries = resampler.query_tokens.to(
        device=qwen_features.device,
        dtype=qwen_features.dtype,
    ).expand(batch, -1, -1)
    cache.resampler_kv = []
    for block in resampler.blocks:
        context = block.source_norm(qwen_features)
        k, v = _mha_kv(block.cross_attention, context)
        cache.resampler_kv.append((k, v, _mha_mask(qwen_mask, batch, context.shape[1])))
    bank_tokens = resampler.query_tokens.shape[1]   # output_projection 은 토큰 수를 바꾸지 않는다

    # --- native·anchor (QualityAnchoredSemanticConnectorV2.forward 의 준비와 첫 블록)
    x = native.in_proj(native.embed(target_input_ids))
    query_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    native_positions = torch.arange(native_source.shape[1], device=x.device).unsqueeze(0)
    anchor_positions = torch.arange(semantic_hidden_states[0].shape[1], device=x.device).unsqueeze(0)
    bank_positions = torch.arange(bank_tokens, device=x.device).unsqueeze(0)
    cache.bank_rope = native.rotary_emb(x, bank_positions)
    cache.query_rope = native.rotary_emb(x, query_positions)
    native_rope = native.rotary_emb(x, native_positions)
    anchor_rope = native.rotary_emb(x, anchor_positions)
    anchor_mix = anchor.layer_mix_logits.float().softmax(dim=-1).to(device=x.device, dtype=x.dtype)

    cache.native_kv, cache.anchor_kv = [], []
    for index, native_block in enumerate(native.blocks):
        anchor_source = anchor.source_norms[index](
            anchor._mixed_source(semantic_hidden_states, anchor_mix, index)
        )
        if index == 0:
            # 첫 블록과 첫 anchor attention 은 통째로 — 원래 forward 가 부르는 모듈을 그대로 부른다
            x = native_block(
                x,
                native_source,
                target_attention_mask=cache.target_attention_mask,
                source_attention_mask=cache.native_source_mask,
                position_embeddings=cache.query_rope,
                position_embeddings_context=native_rope,
            )
            x = x + anchor.semantic_attentions[index](
                anchor.query_norms[index](x),
                mask=cache.semantic_attention_mask,
                context=anchor_source,
                position_embeddings=cache.query_rope,
                position_embeddings_context=anchor_rope,
            )
            cache.x0 = x
            continue
        cache.native_kv.append(_attention_kv(llm, native_block.cross_attn, native_source, native_rope))
        cache.anchor_kv.append(_attention_kv(llm, anchor.semantic_attentions[index], anchor_source, anchor_rope))
    cache.count = int(target_input_ids.shape[0])
    return cache


@torch.no_grad()
def forward(connector, cache: RunCache, timesteps: torch.Tensor | None) -> torch.Tensor:
    """connector(..., timesteps=timesteps) 와 같은 값 — prepare 에 준 입력으로, include_inserted_blocks·include_v2 기본값."""
    if timesteps is None:
        raise ValueError("Semantic Connector v2 requires diffusion timesteps")
    v2 = _module_of(connector)
    native = connector.native_adapter
    anchor = connector.quality_anchor
    resampler = connector.semantic_resampler
    llm = _module_of(native.blocks[0])

    # --- resampler (TimestepAwareSemanticResampler.forward 의 timestep 뒷부분)
    time = v2.sinusoidal_timestep_embedding(timesteps, resampler.model_dim).to(dtype=cache.feature_dtype)
    time = resampler.time_mlp(time)
    queries = cache.queries
    for block, (k, v, mask) in zip(resampler.blocks, cache.resampler_kv):
        queries = _resampler_block_with_kv(block, queries, time, k, v, mask)
    semantic_bank = resampler.output_projection(resampler.output_norm(queries))

    # --- 블록 (첫 블록·첫 anchor 는 캐시한 x0 에서 시작)
    query_rope = cache.query_rope
    x = cache.x0
    for index, native_block in enumerate(native.blocks):
        if index > 0:
            key_states, value_states = cache.native_kv[index - 1]
            x = _native_block_with_kv(
                llm, native_block, x, key_states, value_states,
                cache.target_attention_mask, cache.native_source_mask, query_rope,
            )
            key_states, value_states = cache.anchor_kv[index - 1]
            x = x + _attention_with_kv(
                llm,
                anchor.semantic_attentions[index],
                anchor.query_norms[index](x),
                key_states,
                value_states,
                cache.semantic_attention_mask,
                query_rope,
            )
        # x0 는 여기서 새 텐서가 된다(x + ...) — 다음 블록의 제자리 add_ 가 캐시를 건드리지 않는다
        x = x + connector.v2_attentions[index](
            connector.v2_query_norms[index](x),
            context=connector.v2_semantic_norms[index](semantic_bank),
            position_embeddings=query_rope,
            position_embeddings_context=cache.bank_rope,
        )
    return native.norm(native.out_proj(x))
