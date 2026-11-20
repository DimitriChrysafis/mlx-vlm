"""WeMM-specific loading, audio parity, and multimodal ordering contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLImageProcessor,
    Qwen3VLVideoProcessor,
)
from mlx_vlm.models.wemm_video import AudioConfig, Model, ModelConfig
from mlx_vlm.models.wemm_video.audio import AudioEncoder, sanitize_audio_weights
from mlx_vlm.models.wemm_video.processing_wemm_video import (
    PLACEHOLDERS,
    WeMMVideoProcessor,
    audio_timestamp_groups,
    video_token_ids,
)
from mlx_vlm.utils import get_model_type, load_processor, prepare_inputs


def small_config(moe=False):
    cases = json.loads(Path(__file__).with_name("model_cases.json").read_text())[
        "cases"
    ]
    name = "wemm_video_moe" if moe else "wemm_video"
    return ModelConfig.from_dict(next(c["config"] for c in cases if c["id"] == name))


@pytest.mark.parametrize("raw_type", ["qwen3_5", "wemm_qwen3_5_moe", "wemm_video"])
def test_wemm_architecture_routing(raw_type):
    assert (
        get_model_type(
            {
                "model_type": raw_type,
                "architectures": ["WeMMVideoForConditionalGeneration"],
            }
        )
        == "wemm_video"
    )
    assert get_model_type({"model_type": "qwen3_5"}) == "qwen3_5"


@pytest.mark.parametrize("remote", [False, True])
def test_processor_dispatch_uses_architecture(tmp_path, monkeypatch, remote):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["WeMMVideoForConditionalGeneration"],
            }
        )
    )
    expected = SimpleNamespace(tokenizer=SimpleNamespace())
    tokenizer_paths = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **kw: str(path))
    monkeypatch.setattr(
        WeMMVideoProcessor, "from_pretrained", lambda *a, **kw: expected
    )

    def tokenizer_loader(path, **kwargs):
        tokenizer_paths.append(path)
        return lambda tokenizer: None

    monkeypatch.setattr("mlx_vlm.utils.load_tokenizer", tokenizer_loader)
    monkeypatch.setattr("mlx_vlm.utils.StoppingCriteria", lambda *a, **kw: None)
    result = load_processor("tencent/WeMM-Video-2B" if remote else tmp_path)
    assert result is expected
    assert tokenizer_paths == [tmp_path]


def test_audio_matches_tencent_reference():
    # Generated with Tencent's unmodified WhisperEncoder and Q-Former at
    # 3f4faa80d947769f032b04f105317e1f25844082, Transformers 5.2.0 / Torch 2.15.
    # Two segments, overlapping windows, and two queries per window exercise
    # the unfold/reshape ordering as well as every attention/FFN/normalization.
    fixture = Path(__file__).with_name("fixtures") / "wemm_video_audio_reference.npz"
    with np.load(fixture) as reference:
        config = AudioConfig.from_dict(json.loads(str(reference["config"])))
        model = AudioEncoder(config)
        weights = sanitize_audio_weights(
            {
                f"model.audio.{key}": mx.array(reference[key])
                for key in reference.files
                if key not in ("config", "x", "expected")
            }
        )
        model.load_weights(
            [
                (key.removeprefix("audio_tower."), value)
                for key, value in weights.items()
            ]
        )
        actual = model(mx.array(reference["x"]))
        np.testing.assert_allclose(
            np.asarray(actual), reference["expected"], rtol=3e-5, atol=3e-5
        )


@pytest.mark.parametrize("moe", [False, True])
def test_dense_and_moe_forward_and_sanitize(moe):
    model = Model(small_config(moe))
    assert model(mx.array([[1, 2, 3]])).logits.shape == (1, 3, 32)
    original = {
        "model.audio.conv1.weight": mx.zeros((8, 4, 3)),
        "model.audio.qformer.bert.embeddings.LayerNorm.gamma": mx.ones((12,)),
        "model.audio.qformer.bert.embeddings.LayerNorm.beta": mx.zeros((12,)),
        "model.audio.qformer.bert.embeddings.position_ids": mx.arange(8)[None],
        "model.language_model.norm.weight": mx.zeros((16,)),
    }
    converted = model.sanitize(original)
    assert converted["audio_tower.conv1.weight"].shape == (8, 3, 4)
    assert mx.all(converted["audio_tower.qformer.norm.weight"] == 1)
    assert mx.all(converted["language_model.model.norm.weight"] == 1)
    assert len(converted) == 4
    again = model.sanitize(converted)
    for key in converted:
        np.testing.assert_array_equal(
            np.asarray(converted[key]), np.asarray(again[key])
        )


def test_image_video_audio_embeddings_are_inserted_separately():
    model = Model(small_config())

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=mx.zeros(1)))

        def __call__(self, pixels, grid):
            return mx.full((1, 16), pixels.item()), None

    class Audio(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SimpleNamespace(weight=mx.zeros(1))

        def __call__(self, features):
            return mx.full((1, 2, 16), 7)

    model.vision_tower, model.audio_tower = Vision(), Audio()
    ids = mx.array([[1, 28, 25, 29, 28, 26, 27, 27, 29]])
    inputs = dict(
        pixel_values=mx.array([3]),
        image_grid_thw=mx.array([[1, 2, 2]]),
        pixel_values_videos=mx.array([5]),
        video_grid_thw=mx.array([[1, 2, 2]]),
        audio_feature=mx.zeros((1, 4, 20)),
    )
    result = model.get_input_embeddings(ids, **inputs)
    np.testing.assert_array_equal(
        np.asarray(result.inputs_embeds[0, 2]), np.full(16, 3)
    )
    np.testing.assert_array_equal(
        np.asarray(result.inputs_embeds[0, 5]), np.full(16, 5)
    )
    np.testing.assert_array_equal(
        np.asarray(result.inputs_embeds[0, 6:8]), np.full((2, 16), 7)
    )
    with pytest.raises(ValueError, match="Audio features and tokens"):
        model.get_input_embeddings(ids[:, :-2], **inputs)


@pytest.fixture
def processor():
    specials = [
        "[UNK]",
        "[PAD]",
        "[EOS]",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_pad|>",
    ]
    backend = Tokenizer(
        WordLevel({s: i for i, s in enumerate(specials)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=specials[3:],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{m['role']}}: {{m['content']}}{% endfor %}"
    )
    return WeMMVideoProcessor(
        image_processor=Qwen3VLImageProcessor(
            patch_size=2, min_pixels=16, max_pixels=16
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=2, min_pixels=16, max_pixels=64
        ),
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
        audio_config={"num_mel_bins": 4},
    )


@pytest.mark.parametrize(
    "duration,expected",
    [(0.05, [300]), (1.2, [6, 294]), (2.3, [8, 7, 285]), (30, [10] * 30)],
)
def test_audio_timestamps_keep_padded_tail(duration, expected):
    groups, times = audio_timestamp_groups(300, 0.1, duration)
    assert groups == expected
    assert times == list(range(len(expected)))


def test_processor_preserves_audio_order_across_video(processor, monkeypatch):
    monkeypatch.setattr(
        processor,
        "_audio_features",
        lambda source, **kw: (
            np.full((1, 4, 3000), float(source), dtype=np.float32),
            1.5,
        ),
    )
    monkeypatch.setattr(
        processor,
        "_video",
        lambda *a, **kw: (
            {
                "pixel_values_videos": np.zeros((8, 24)),
                "video_grid_thw": np.array([[2, 2, 2]]),
            },
            {"frames_indices": [0, 2, 4, 6], "fps": 4},
        ),
    )
    prompt = PLACEHOLDERS["audio"] + PLACEHOLDERS["video"] + PLACEHOLDERS["audio"]
    inputs = processor(
        text=prompt,
        audio=[1, 3],
        videos=[{"video": "unused.mp4", "audio": 2, "use_audio_in_video": True}],
    )
    assert inputs["audio_feature"].shape == (3, 4, 3000)
    np.testing.assert_array_equal(
        np.asarray(inputs["audio_feature"][:, 0, 0]), [1, 2, 3]
    )
    pad = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    assert int((inputs["input_ids"] == pad).sum()) == 900


def test_prepared_video_keeps_path_for_soundtrack(processor, monkeypatch):
    sources = []

    def video(source, *a, **kw):
        sources.append(source)
        return (
            {
                "pixel_values_videos": np.zeros((8, 24)),
                "video_grid_thw": np.array([[2, 2, 2]]),
            },
            {"frames_indices": [0, 1, 2, 3], "fps": 2},
        )

    monkeypatch.setattr(processor, "_video", video)
    monkeypatch.setattr(processor, "_soundtrack", lambda source: np.zeros(16000))
    monkeypatch.setattr(
        processor, "_audio_features", lambda *a, **kw: (np.zeros((1, 4, 3000)), 1)
    )
    data = prepare_inputs(
        processor,
        videos=["clip.mp4"],
        prompts=PLACEHOLDERS["video"],
        use_audio_in_video=True,
    )
    assert sources == ["clip.mp4"]
    assert "audio_feature" in data


def test_processor_rejects_missing_media_and_batch(processor):
    with pytest.raises(ValueError, match="Extra audio"):
        processor(text=PLACEHOLDERS["audio"])
    with pytest.raises(ValueError, match="placeholders differ"):
        processor(text="Describe.", images=[Image.new("RGB", (4, 4))])
    with pytest.raises(ValueError, match="one conversation"):
        processor(text=["first", "second"])


def test_video_tokens_use_source_frame_timestamps(processor):
    tokenizer = processor.tokenizer
    ids = video_token_ids(
        tokenizer,
        [2, 2, 2],
        {"frames_indices": [0, 3, 6, 9], "fps": 3},
        2,
        2,
        audio_count=300,
        audio_duration=2,
        window=0.1,
    )
    start, end, video, audio = [
        tokenizer.convert_tokens_to_ids(t)
        for t in (
            "<|vision_start|>",
            "<|vision_end|>",
            "<|video_pad|>",
            "<|audio_pad|>",
        )
    ]
    expected = []
    for timestamp, count in [(0.5, 10), (2.5, 290)]:
        expected += tokenizer.encode(
            f"<{timestamp:.1f} seconds>", add_special_tokens=False
        )
        expected += [start, video] + [audio] * count + [end]
    assert ids == expected
