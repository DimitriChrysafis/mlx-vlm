"""Regenerate the small WeMM audio fixture from the reviewed upstream source.

Run in an isolated environment with torch==2.15.0, transformers==5.2.0 and
numpy. Pass a directory containing h5_modeling_whisper.py and h5_modeling_bert.py
from tencent/WeMM-Video-2B at 3f4faa80d947769f032b04f105317e1f25844082.
This script executes those upstream files. It does not download or modify them.
"""

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, WhisperConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("wemm_video_audio_reference.npz"),
    )
    args = parser.parse_args()
    package = types.ModuleType("wemm_reference")
    package.__path__ = [str(args.reference_dir)]
    sys.modules[package.__name__] = package
    reference = importlib.import_module("wemm_reference.h5_modeling_whisper")
    config = WhisperConfig(
        d_model=8,
        num_mel_bins=4,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        max_source_positions=20,
    )
    config.qformer_config = BertConfig(
        hidden_size=12,
        encoder_width=8,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        add_cross_attention=True,
        cross_attention_freq=1,
        query_length=2,
        out_hidden_size=16,
    )
    config.second_per_window = 0.1
    config.second_stride = 0.06
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = reference.WhisperEncoder(config).float().eval()
    inputs = np.random.default_rng(7).normal(size=(2, 4, 40)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.tensor(inputs)).numpy()
    np.savez_compressed(
        args.output,
        config=json.dumps(config.to_dict()),
        x=inputs,
        expected=expected,
        **{key: value.detach().numpy() for key, value in model.state_dict().items()},
    )


if __name__ == "__main__":
    main()
