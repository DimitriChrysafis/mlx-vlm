# Inkling 2 Small

MLX-VLM supports text and image inference with
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small).
Load the native checkpoint directly; no separate weight conversion is required.

```python
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

# Avoid retaining large temporary buffers while unpacking the native weights.
mx.set_cache_limit(0)
model, processor = load("thinkingmachines/Inkling-2-Small", lazy=True)
mx.eval(model.vision_tower.parameters())
for layer in model.layers:
    mx.eval(layer.parameters())
mx.eval(model.parameters())
prompt = apply_chat_template(
    processor, model.config, "Explain why the sky is blue in one sentence.",
    reasoning_effort="none",
)
result = generate(
    model, processor, prompt, max_tokens=128, temperature=0,
    prefill_step_size=128, skip_special_tokens=True,
)
print(result.text)
```

For an image, pass `num_images=1` to `apply_chat_template` and
`image=["/path/to/image.png"]` to `generate`, with an image-related prompt.
Multiple images and the standard streaming, batch, and prefix-cache paths are supported.

The native target requires about 390 GB for its weights. On a 512 GB M3 Ultra,
loading and short text/image smoke tests peaked at 395 GB with the allocation
cache disabled. Loading layers sequentially, as above, bounds temporary buffers
and pending reads. The default allocation cache can retain enough temporary
buffers to exhaust memory during loading.

The native download is approximately 394 GB, including a separate draft checkpoint.
The target uses 48 layers, 384 experts with eight selected per token, learned
relative-position attention, short convolutions, and block attention residuals.
Its vision encoder is a 23-layer ViT. The loader keeps the first 40 layers' experts
in NVFP4 and the final eight layers' experts in MXFP8, including the original
per-expert scales. Keep `hf_quant_config.json` with the checkpoint: it describes
this mixed format. Inference uses floating-point activations with the native
quantized weights; it does not reproduce the reference runtime's activation
quantization.

## Distributed inference

Use the standard `sharded_load` API or distributed generation example:

```sh
mlx.launch --backend jaccl --hostfile /path/to/hosts.json \
  examples/sharded_generate.py \
  --backend jaccl --model thinkingmachines/Inkling-2-Small \
  --cache-limit 0 --max-tokens 128 --prompt "Explain why the sky is blue."
```

Each host needs this checkout and access to the checkpoint. Use `ring` instead
of `jaccl` on both flags for a TCP cluster. Add `--image /path/to/image.png` for
image input, with the image available at that path on every host.

The model partitions whole experts across ranks, including packed NVFP4/MXFP8
weights, group scales, and FP32 outer scales. Attention, routing, vision, and
KV/convolution caches remain replicated. Each rank must own at least one expert;
uneven partitions are supported. Route outputs are combined before the FP32
weighted sum, and sharding preserves the quantized kernel's original dispatch
threshold. This avoids introducing extra BF16 rounding differences. Pipeline
parallelism is not implemented.

For Python, call `mx.set_cache_limit(0)` before
`model, processor = sharded_load(path, tensor_group=group)`, where `group` comes
from `mx.distributed.init(...)`. Use the normal generation APIs on every rank.
Expert buffers are materialized and partitioned one layer at a time to release
the full allocations during loading; lazy slices alone retain those allocations.

Validation on MLX 0.32.4 includes real ring collectives with two and four local
ranks on tiny dense and mixed-quantization models, with exact BF16 logit agreement
against unsharded inference. The released checkpoint was also tested with two
local ranks: all 1,025 positions match the retained single-device MLX logits
exactly, for both full prefill and independent cached decoding (zero RMS error,
zero top-token disagreements). Both ranks generated `4` for `2 + 2` and `Red`
for the red-image smoke test. Each rank stores 192 experts per layer and about
202 GB of weights; peak allocation was 213.6 GB per rank.

The full-size local test serialized loading and warmup and used host barriers
around inference steps because both processes shared one GPU. Concurrent loading
on that GPU hit a Metal timeout. These are correctness results; multi-Mac
JACCL/RDMA operation and throughput have not been measured. Exact agreement with
single-device MLX does not remove the CPU-reference differences described below.

The dense parameters retain the checkpoint's mixed dtypes: 96 attention-residual
queries and 48 router biases are FP32; the other 1,274 unquantized tensors are
BF16. NVFP4 outer scales remain FP32. Conversion to BF16 preserves those FP32
parameters. Convolutions, routing, and attention-residual mixing compute in FP32
where required, without changing the stored BF16 weights. The vocabulary
projection uses BF16 input values and weights with FP32 accumulation and output,
as specified by `fwd_output_dtype` in the checkpoint metadata; its scores are
not rounded to BF16 before softcapping.

This port supports the target model. The bundled `mtp.safetensors` contains an
Inkling 2 DSpark drafter with a recurrent head, which is not yet supported. It is
not an Inkling 1 MTP checkpoint. Audio input is not supported by this model.

Validation includes tiny-model language/vision forwards, mixed-precision native
loading and converted-checkpoint reloads, cached decoding across sliding-window
boundaries, padded batches, speculative target rollback, multi-image isolation,
and in-memory/on-disk prefix-cache reuse. Small random-weight language and vision
checks match the Transformers implementation within `8e-7` in float32. Independent
decoding of released NVFP4 and MXFP8 expert slices matches within `3e-7`.
The full released checkpoint also passes text (`2 + 2` → `4`) and image
(an 80×80 red image → `Red`) generation smoke tests on an M3 Ultra with 512 GB RAM.
These short checks establish loading and inference, not a quality benchmark.

A released-weight, teacher-forced comparison across all 48 language layers and
27 positions gives FP32 logit RMS error `9.3e-6` and maximum error `7.9e-5` against
Transformers with independently decoded experts. A native mixed-dtype rerun on
two prompts (27 and 33 positions), using the metadata's FP32 vocabulary output
on both sides, gives logit RMS errors `0.21586` and `0.16443`, with maximum errors
`2.38001` and `2.06870`. All 60 top-token choices agree, including after FP32
log-softmax, but strict logit parity still fails (RMS limit `0.15`, maximum limit
`1.0`, relative RMS limit `0.02`). Correcting the head's intermediate BF16
rounding improves RMS slightly on both prompts; in isolation on identical
hidden states, its maximum error falls from `0.0476` to `3.8e-6`. The dtype audit
checks all 1,418 unquantized tensors against the native metadata with no
mismatches, and conversion/reload tests protect the FP32 values.

These comparisons use an adapted CPU reference: dense attention projections
retain BF16 weights and outputs, while decoded expert products, convolutions,
routing, residual mixing, and the vocabulary output compute in FP32 with
explicit activation casts. Convolution computes in FP32 in both implementations.
Widening completed outputs for error measurement preserves their values exactly;
compute-time promotions can change kernel selection and rounding. The results
therefore measure agreement with this reference's arithmetic as well as the
weight-loading and architecture implementation.

The adapted reference keeps decoded NVFP4 expert weights in FP32. In MLX
0.32.4, the long-prefill matrix kernel instead rounds scaled decoded weights
to BF16, while vector kernels apply the outer scale after FP32 accumulation.
Fixed-input expert checks confirm this arithmetic difference; preserving the
checkpoint's mixed storage dtypes does not remove it. See MLX's
[matrix weight loader](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/fp_quantized.h#L188)
and [vector output scaling](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/fp_quantized.h#L406).

Cached decoding versus prefill on these prompts has RMS errors `0.12374` and
`0.11573`, with no top-token changes, but still exceeds the maximum-error limit
of `1.0`. The reference uses floating activations and FP32 expert accumulation;
this does not validate the CUDA runtime's activation quantization. Static
convolutions accumulate, add their residual, and scale in FP32 before rounding
to the activation dtype for both prefill and decoding.

A longer comparison covers all 1,025 teacher-forced positions across all 48
language layers, using the same native mixed dtypes and pinned reference.
Prefill has 7/1,025 top-token disagreements and logit RMS error `0.76623`.
Independent chunked decoding has 11/1,025 disagreements against the reference
and 11/1,025 against prefill. All logits are finite, and both paths agree with
the reference at context lengths 1,024 and 1,025, but these results do not
establish zero BF16 disagreement or strict numerical parity over the full
sequence. The short-prompt agreement above must not be generalized to longer
contexts.

A reference self-consistency control uses the same 1,025 inputs and chunk
boundaries on the CPU reference. Its cached path differs from its own full
prefill at 6/1,025 positions, with RMS error `0.76752`. Comparing MLX cached
outputs against those cached reference outputs gives 11/1,025 disagreements
and RMS error `0.75742`. These controls expose execution-shape sensitivity in
the adapted BF16 reference; they do not make the strict parity checks pass.

A separate reference control matches the expert-weight precision and outer-scale
placement of MLX's matrix and vector paths. It gives 6/1,025 disagreements against
the unchanged MLX prefill, with logit RMS error `0.77935`, versus 7/1,025 and
`0.76623` for the original reference. Both short prompts retain full top-token
agreement. Matching expert precision alone therefore does not establish strict
parity. These diagnostic results do not replace the original baseline or validate
the CUDA runtime.

Architecture reference:
[Transformers Inkling2](https://github.com/huggingface/transformers/blob/6a9193deabe4ffcf495726897e3f2e651b8bbead/src/transformers/models/inkling2/modeling_inkling2.py).
