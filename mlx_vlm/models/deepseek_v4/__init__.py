import mlx_vlm.models.deepseek_v4.processing_deepseek_v4  # noqa: F401 (installs processor patch)

from .config import ModelConfig, validate_sharding
from .deepseek_v4 import Model

# Capabilities of the default native encoder, before any checkpoint-provided
# chat template overrides it. Clients can inspect this without loading weights.
NATIVE_CHAT_CAPABILITIES = frozenset({"tools", "reasoning"})

__all__ = ["Model", "ModelConfig", "validate_sharding", "NATIVE_CHAT_CAPABILITIES"]
