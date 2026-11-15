import os

import pytest

# float32 matmul runs at TF32 precision on hardware with matrix units, which is
# looser than the float32 references these tests compare against. Set rather than
# setdefault, so an inherited MLX_ENABLE_TF32=1 doesn't leak into the test run.
os.environ["MLX_ENABLE_TF32"] = "0"


@pytest.fixture
def native_mistral_params():
    """Small native metadata shared by loading, processor, and preflight tests."""
    return dict(
        dim=16,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=64,
        norm_eps=1e-6,
        max_position_embeddings=512,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        rope_theta=10000.0,
        moe=dict(
            expert_hidden_dim=16,
            num_experts=8,
            num_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=0,
        ),
        dsa=dict(topk=4),
        vision_encoder=dict(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            image_size=56,
            patch_size=14,
            rope_theta=10000.0,
            image_token_id=3,
            spatial_merge_size=2,
            adapter_bias=False,
        ),
    )


def pytest_sessionfinish(session, exitstatus):
    """Release thread-local MLX resources before Python finalization."""
    del session, exitstatus

    import mlx.core as mx

    clear_streams = getattr(mx, "clear_streams", None)
    if clear_streams is not None:
        clear_streams()
