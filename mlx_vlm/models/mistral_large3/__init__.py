from ..base import install_auto_processor_patch
from ..mistral3.processing_mistral3 import Mistral3Processor
from ..pixtral import VisionConfig
from .config import ModelConfig, TextConfig
from .mistral_large3 import Model

install_auto_processor_patch("mistral_large3", Mistral3Processor)

__all__ = ["Model", "ModelConfig", "TextConfig", "VisionConfig", "Mistral3Processor"]
