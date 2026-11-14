# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted for MLX from Transformers' Inkling2 implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..cache import (
    ArraysCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from ..inkling.language import InklingShortConvolution
from ..inkling.language import LanguageModel as InklingLanguageModel
from ..inkling.language import (
    _cache_padding_mask,
    _clone_cache_tree,
    _next_conv_state,
    banded_additive_mask,
)
from ..switch_layers import SwitchGLU
from .config import TextConfig, validate_sharding


def l2_clip(x, threshold, scale=None):
    dtype = x.dtype
    x = x.astype(mx.float32)
    if scale is not None:
        x = x * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return (x * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(dtype)


class ShortConvolution(InklingShortConvolution):
    def __call__(self, x, cache=None, mask=None):
        # Keep the convolution, residual, and scaling in FP32. The checkpoint
        # stores BF16 taps, but rounding the convolution before adding the
        # residual changes both reference parity and prefill/decode consistency.
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]))
        context = mx.concatenate([state, inputs], axis=1)
        if cache is not None:
            cache[self.conv_idx] = _next_conv_state(state, inputs, mask)
        out = mx.conv1d(
            context, self.conv.weight.astype(mx.float32), groups=x.shape[-1]
        )
        return ((out + xf) / math.sqrt(self.kernel_size + 1)).astype(x.dtype)


class DynamicShortConvolution(nn.Module):
    def __init__(self, config, channels, num_heads, conv_idx):
        super().__init__()
        self.kernel_size = config.sconv_kernel_size
        self.num_heads = num_heads
        self.conv_idx = conv_idx
        self.static_weight = mx.zeros((self.kernel_size, channels))
        self.weight_proj = nn.Linear(
            config.hidden_size, num_heads * self.kernel_size, bias=False
        )

    def __call__(self, x, dynamic_input, cache=None, mask=None):
        b, length, channels = x.shape
        k = self.kernel_size
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((b, k - 1, channels), mx.float32)
        context = mx.concatenate([state, inputs], axis=1)
        if cache is not None:
            cache[self.conv_idx] = _next_conv_state(state, inputs, mask)
        dynamic = (
            self.weight_proj(dynamic_input)
            .astype(mx.float32)
            .reshape(b, length, self.num_heads, 1, k)
        )
        weights = self.static_weight.T.reshape(self.num_heads, -1, k) + dynamic
        out = mx.zeros_like(xf).reshape(b, length, self.num_heads, -1)
        for lag in range(k):
            values = context[:, k - 1 - lag : k - 1 - lag + length].reshape(out.shape)
            out = out + values * weights[..., lag]
        return ((out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = self.sliding or config.rel_extent
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.r_proj = nn.Linear(
            config.hidden_size, self.n_heads * config.d_rel, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.sink_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.q_scale = mx.ones((self.n_heads, self.head_dim))
        self.rel_scale = mx.ones((self.n_heads, config.d_rel))
        self.rel_proj = mx.zeros((config.d_rel, self.rel_extent))
        for idx, name in enumerate(("k_sconv", "v_sconv")):
            channels = self.n_kv * self.head_dim
            self[name] = (
                ShortConvolution(channels, config.sconv_kernel_size, idx)
                if self.is_sliding
                else DynamicShortConvolution(config, channels, self.n_kv, idx)
            )

    def __call__(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        b, length, _ = x.shape
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        k, v = self.k_proj(key_input), self.v_proj(value_input)
        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)
        q = l2_clip(
            self.q_proj(x).reshape(b, length, self.n_heads, self.head_dim),
            cfg.qk_l2_clip_threshold,
            self.q_scale,
        )
        k = l2_clip(
            k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold
        )
        v = v.reshape(b, length, self.n_kv, self.head_dim)
        if cfg.use_value_norm_no_gamma:
            v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(
                x.dtype
            )
        r = l2_clip(
            self.r_proj(x).reshape(b, length, self.n_heads, cfg.d_rel),
            cfg.rel_l2_clip_threshold,
            self.rel_scale,
        )
        if not self.is_sliding and cfg.log_scaling_n_floor is not None:
            offset = kv.offset if kv is not None else 0
            positions = mx.arange(length) + mx.array(offset)[..., None] + 1
            tau = 1 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / cfg.log_scaling_n_floor, 1)
            )
            q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
            r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
            # Relative logits index backward distance, so ring buffers must be
            # read in chronological order (ordinary RoPE attention does not care).
            if isinstance(kv, RotatingKVCache):
                k, v = kv._temporal_order(k), kv._temporal_order(v)
            elif isinstance(kv, BatchRotatingKVCache) and kv.rotated:
                k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
        size = k.shape[2]
        padding = getattr(kv, "left_padding", None)
        mask = banded_additive_mask(
            r,
            self.rel_proj.astype(x.dtype),
            size - length,
            size,
            self.sliding,
            self.rel_extent,
            left_padding=padding,
            shape_ref=k,
        )
        # A query-dependent sink is represented by a zero K/V and its own logit.
        sinks = self.sink_proj(x).transpose(0, 2, 1)[..., None]
        mask = mx.concatenate([mask, sinks], axis=-1)
        k = mx.concatenate(
            [k, mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)], axis=2
        )
        v = mx.concatenate(
            [v, mx.zeros((b, self.n_kv, 1, self.head_dim), v.dtype)], axis=2
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        out = (
            out.astype(mx.float32) * mx.sigmoid(self.gate_proj(x).astype(mx.float32))
        ).astype(x.dtype)
        return self.o_proj(out)


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap

    def __call__(self, up, gate):
        dtype = gate.dtype
        gate, up = gate.astype(mx.float32), up.astype(mx.float32)
        return (
            mx.tanh(gate / self.gate_cap)
            * self.gate_cap
            * mx.sigmoid(gate)
            * mx.tanh(up / self.up_cap)
            * self.up_cap
        ).astype(dtype)


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.sharding_group = None
        self.expert_start = 0
        self.expert_end = config.n_routed_experts
        self.gate_weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.intermediate_size,
            config.n_routed_experts,
            activation=SoftcappedSwiGLU(config),
        )

    def __call__(self, x):
        scores = mx.sigmoid(
            x.astype(mx.float32) @ self.gate_weight.astype(mx.float32).T
        )
        indices = mx.argpartition(
            -(scores + self.e_score_correction_bias), self.top_k - 1, axis=-1
        )[..., : self.top_k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = (
            weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * self.route_scale
        )
        if self.sharding_group is None:
            out = self.switch_mlp(x, indices)
        else:
            owned = (indices >= self.expert_start) & (indices < self.expert_end)
            local = mx.where(owned, indices - self.expert_start, 0)
            out = self.switch_mlp(x, local)
            out = mx.where(owned[..., None], out, 0)
            # Exactly one rank owns each route. Reduce route outputs before
            # weighting to retain the unsharded FP32 accumulation order.
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        out = out.astype(mx.float32)
        return (out * weights[..., None]).sum(axis=-2).astype(x.dtype)


class AttentionResidual(nn.Module):
    def __init__(self, config, num_queries=1):
        super().__init__()
        self.query = mx.zeros(
            (config.hidden_size,)
            if num_queries == 1
            else (num_queries, config.hidden_size)
        )
        self.normalize_values = config.attn_res_normalize_values

    def __call__(self, blocks, partial):
        states = mx.stack([*blocks, partial], axis=-2).astype(mx.float32)
        keys = mx.fast.rms_norm(states, None, 1e-6)
        scores = (keys @ self.query.reshape(-1, states.shape[-1]).T) * states.shape[
            -1
        ] ** -0.5
        weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
        return weights @ (keys if self.normalize_values else states)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.attention_input = (
            AttentionResidual(config, 1 if self.is_sliding else 3)
            if layer_idx
            else None
        )
        self.mlp_input = AttentionResidual(config)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SparseMoE(config)
        names = (
            ("input_layernorm",)
            if self.is_sliding
            else ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm")
        )
        for name in (*names, "post_attention_layernorm"):
            self[name] = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_sconv = ShortConvolution(
            config.hidden_size, config.sconv_kernel_size, 2
        )
        self.mlp_sconv = ShortConvolution(
            config.hidden_size, config.sconv_kernel_size, 3
        )

    def __call__(self, x, blocks, cache=None):
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        mixed = (
            self.attention_input(blocks, x)
            if self.attention_input is not None
            else x[..., None, :]
        )
        if self.is_sliding:
            attn = self.self_attn(
                self.input_layernorm(mixed[..., 0, :]).astype(x.dtype),
                cache=cache,
                conv_mask=mask,
            )
        else:
            parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
            parts = [
                norm(part).astype(x.dtype)
                for norm, part in zip(
                    (
                        self.q_input_layernorm,
                        self.k_input_layernorm,
                        self.v_input_layernorm,
                    ),
                    parts,
                )
            ]
            attn = self.self_attn(*parts, cache=cache, conv_mask=mask)
        residual = self.attn_sconv(attn, conv, mask).astype(mx.float32)
        if self.is_block_boundary:
            blocks.append(x)
        else:
            residual = residual + x.astype(mx.float32)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        hidden = self.mlp(self.post_attention_layernorm(hidden).astype(x.dtype))
        out = (residual + self.mlp_sconv(hidden, conv, mask).astype(mx.float32)).astype(
            x.dtype
        )
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.final_input = AttentionResidual(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, inputs):
        return self.embed_norm(self.embed_tokens(inputs))

    def __call__(
        self, inputs, cache=None, input_embeddings=None, skip_final_norm=False
    ):
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        blocks = []
        for layer, c in zip(self.layers, cache or [None] * len(self.layers)):
            x = layer(x, blocks, c)
        h = self.final_input(blocks, x)[..., 0, :]
        return h if skip_final_norm else self.norm(h).astype(x.dtype)


class LanguageModel(InklingLanguageModel):
    def __init__(self, config: TextConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sharding_group = None

    def shard(self, group=None):
        group = group or mx.distributed.init()
        size, rank = group.size(), group.rank()
        count = self.config.n_routed_experts
        validate_sharding(vars(self.config), size)
        if not 0 <= rank < size:
            raise ValueError("Each distributed rank must own at least one expert")
        if self.sharding_group is not None:
            raise ValueError("Inkling 2 is already sharded")
        if size == 1:
            return
        start, end = count * rank // size, count * (rank + 1) // size
        scalar_scales = {"weight_scale_2", "weight_global_scale"}
        # Check every layer before changing any weights, including native
        # packed weights, group scales, and per-expert FP32 outer scales.
        for layer in self.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(layer.mlp.switch_mlp, name)
                for key, value in projection.parameters().items():
                    if not isinstance(value, mx.array):
                        continue
                    if key in scalar_scales and value.size == 1:
                        continue
                    if not value.ndim or value.shape[0] != count:
                        raise ValueError(
                            f"Invalid expert partition for {name}.{key}: "
                            f"expected {count} experts, got {value.shape}"
                        )
        for layer in self.layers:
            moe = layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(moe.switch_mlp, name)
                projection.update(
                    {
                        key: mx.contiguous(value[start:end])
                        for key, value in projection.parameters().items()
                        if isinstance(value, mx.array)
                        and not (key in scalar_scales and value.size == 1)
                    }
                )
            # Materialize one layer at a time and release the full expert
            # buffers; lazy slices alone retain the unpartitioned allocation.
            mx.eval(moe.switch_mlp.parameters())
            mx.clear_cache()
            moe.expert_start, moe.expert_end = start, end
            moe.sharding_group = group
        self.sharding_group = group

    def _logits_from_norm(self, h):
        # hf_quant_config.json stores BF16 unembedding weights but requires
        # fwd_output_dtype=float32. Promote the product before multiplication;
        # casting BF16 scores afterwards loses the accumulator's low bits.
        h = h.astype(self.model.embed_norm.weight.dtype).astype(mx.float32)
        logits = self.lm_head(h)
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.config.unpadded_vocab_size]

    def make_cache(self):
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=self.config.sliding_window_size, keep=0)
                    if layer.is_sliding
                    else KVCache()
                ),
                ArraysCache(4),
            )
            for layer in self.model.layers
        ]

    def speculative_verify_hidden(self, inputs, cache):
        # Ring-buffer offsets and convolution padding advance with verification;
        # both belong to the snapshot, in addition to the actual K/V tensors.
        snapshot = [[_clone_cache_tree(vars(part)) for part in c.caches] for c in cache]
        mx.eval([a for _, a in tree_flatten(snapshot) if isinstance(a, mx.array)])
        out = self(inputs, cache=cache, return_hidden=True, skip_logits=True)
        return out.hidden_states[-1], {}, (snapshot, inputs)

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        if isinstance(accepted, mx.array):
            accepted = int(accepted.max().item()) if accepted.size else 0
        elif not isinstance(accepted, int):
            accepted = max(map(int, accepted))
        snapshot, inputs = gdn_states
        for cache, states in zip(caches, snapshot):
            for part, state in zip(cache.caches, states):
                part.__dict__.clear()
                part.__dict__.update(_clone_cache_tree(state))
        self(inputs[:, : accepted + 1], cache=caches, skip_logits=True)
        return accepted
