# Inkling 2 Small

MLX-VLM supports text and image inference with
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small).
Load the native checkpoint directly; no separate weight conversion is required.

```python
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

# Avoid retaining large temporary buffers while unpacking the native weights.
allocation_cache_limit = mx.set_cache_limit(0)
model, processor = load("thinkingmachines/Inkling-2-Small", lazy=True)
mx.eval(model.vision_tower.parameters())
for layer in model.layers:
    mx.eval(layer.parameters())
mx.eval(model.parameters())
mx.set_cache_limit(allocation_cache_limit)
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
weights, group scales, and FP32 outer scales. Compatible two-rank Metal models
also partition attention heads, complete output-projection rows, and the
corresponding KV/convolution state. Heads are gathered before the output
projection so each dot product retains its original reduction. The small sink
projection stays replicated because splitting its rows changes prefill rounding.
For matrix projections with more than 15 input rows, zero padding preserves the
original output width and kernel dispatch before selecting the owned rows. This
fallback contains no remote weights; it protects prefill and larger verification
blocks from shape-dependent rounding changes.
Routing, vision, embeddings, and relative projections remain replicated.
Other group sizes and unsupported attention layouts retain replicated attention.
Each rank must own at least one expert;
uneven partitions are supported. Route outputs are combined before the FP32
weighted sum, and sharding preserves the quantized kernel's original dispatch
threshold. This avoids introducing extra BF16 rounding differences. Pipeline
parallelism is not implemented.

For Python, call `mx.set_cache_limit(0)` before
`model, processor = sharded_load(path, tensor_group=group)`, where `group` comes
from `mx.distributed.init(...)`. Use the normal generation APIs on every rank.
Expert buffers are materialized and partitioned one layer at a time to release
the full allocations during loading; lazy slices alone retain those allocations.
The sharding path also clears temporary allocations after each layer.

The standard [cluster APIs](../../../docs/cluster-api.md) support this model.
Configure matching source and MLX versions on each host, then use
`POST /v1/cluster/models/prepare` to deliver the native checkpoint and
`POST /v1/cluster/start` to launch and load a stopped group. For direct RDMA
delivery, configure the selected peer's `ssh.rdma` link and request
`transport: "jaccl_rdma"`. Send generation requests to `/v1/responses` with
`distributed: true` and `reasoning_effort: "none"` for direct answers.
Each destination needs room for the full checkpoint on disk even though each
of two inference ranks keeps only half the experts in memory. Cluster member
validation checks expert ownership; attention-head divisibility is not required.

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
on that GPU hit a Metal timeout. These are correctness results. Exact agreement
with single-device MLX does not remove the CPU-reference differences described
below.

The cluster APIs were also tested on two physical 512 GB M3 Ultras connected
through JACCL and a direct Thunderbolt RDMA link, using MLX
`0.32.5.dev20261030+1ba7f96`. Verified delivery of the full 393.51 GB checkpoint
took 71.9 seconds. Managed launch/load, text and image streaming, a 1,025-token
input, retained decode continuation, concurrent request admission, and managed
shutdown passed. Resumed text matched uninterrupted generation, and both
concurrent requests matched their individual controls. Both workers remained
healthy and reported matching inference fingerprints.

Measured single-request results for the same 1,025-token input (temperature 0,
no thinking, no draft, zero cached input tokens):

| Prefill chunk | Prompt tokens/s | Generation tokens/s | Client time to first text | Output tokens | Peak GB/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 141.1 | 12.72 | 8.05 s | 32 | 209.1 |
| 1,024 | 319.6 | 12.45 | 3.78 s | 28 | 209.1 |

These are individual functional runs, not a repeated throughput benchmark.
Changing the prefill chunk changed the greedy answer on this long prompt,
consistent with the execution-shape sensitivity discussed below. The API tests
check rank consistency and generated text; they do not establish full-logit
parity with the CPU reference or rerun the retained single-device logit comparison
under this newer MLX build.

The dense parameters retain the checkpoint's mixed dtypes: 96 attention-residual
queries and 48 router biases are FP32; the other 1,274 unquantized tensors are
BF16. NVFP4 outer scales remain FP32. Conversion to BF16 preserves those FP32
parameters. Convolutions, routing, and attention-residual mixing compute in FP32
where required, without changing the stored BF16 weights. The vocabulary
projection uses BF16 input values and weights with FP32 accumulation and output,
as specified by `fwd_output_dtype` in the checkpoint metadata; its scores are
not rounded to BF16 before softcapping.

## Bundled speculative decoding

The bundled `mtp.safetensors` is supported through the standard drafter loader.
It contains a five-layer DSpark transformer and an `rnn_hidden` Markov head;
its trained block contains one anchor and seven proposed tokens. The 3.42 GB
sidecar remains BF16 and shares the target's normalized embeddings and vocabulary
head. No extraction or separate conversion is needed:

```python
from mlx_vlm.speculative.drafters import load_drafter

draft, kind = load_drafter("thinkingmachines/Inkling-2-Small")
result = generate(
    model, processor, prompt, max_tokens=128, temperature=0,
    draft_model=draft, draft_kind=kind, prefill_step_size=128,
    skip_special_tokens=True,
)
```

For the CLI, add `--draft-model thinkingmachines/Inkling-2-Small` to the normal
generation command; the loader selects `dflash` automatically. The sharded
generation example accepts the same flag and replicates the small drafter on
each rank while partitioning the target experts. Greedy batches distribute
whole draft requests across those ranks and combine only their proposed token
IDs. Each request's draft-cache owner remains fixed through batch filtering,
admission and resume, preserving the original per-request arithmetic. Fresh
single requests and stochastic drafting retain replicated execution.
Distributed **cluster API**
workers currently reject all speculative drafters; use the sharded Python/example
path for distributed DSpark.

The target captures partial residual blocks entering layers 5, 11, 17, 23, 29,
35 and 41, plus the normalized output at layer 48. Each draft layer learns its
own fusion of these features. Attention sees the complete draft block and, on
local layers, the 1,024 target positions preceding its anchor. The recurrent
head uses the last committed target features and the preceding proposed token.
Rejected proposals roll back KV and all four convolution states through shared
cache transactions, including unequal acceptance lengths in a batch. Chunked
prefill preserves the requested target chunk size with and without drafting.

Draft logits follow the bundled head's BF16 rounding and softcap after adding
the recurrent correction. This differs from the target's FP32 vocabulary output.
The folded recurrent projection is derived once in FP64, in bounded vocabulary
tiles, then stored in the draft dtype. The implementation follows the
[pinned DSpark reference](https://github.com/WoosukKwon/vllm/blob/d664cd1e96052857aea2aa844eeba571c24758b4/vllm/models/inkling2/dspark.py).

Tiny-model regressions cover native and standalone loading, captured-state
indices, FP32/BF16 greedy and positioned sampling, prefill boundaries, short
output budgets, request resets, per-row rollback/abort, continuous admission,
cancellation and pause/resume. An independent CPU transcription using released
BF16 weights and fixed target features gives zero top-token disagreements at
all seven draft positions for contexts of 33 and 1,025 tokens. Draft-logit RMS
errors are 0.06992 and 0.08619 respectively; maximum errors are 0.5 and 0.4375.
These checks establish agreement on those teacher-forced inputs, not bitwise
parity with the CUDA runtime or a model-quality benchmark.

Released-weight validation on the selected two 512 GB M3 Ultras over JACCL
(`0.32.5.dev20261030+1ba7f96`) covers arithmetic, a 48-token explanation, a red
image and a 1,025-token prompt. All 90 generated token IDs, including stop tokens,
match drafting-disabled controls with the same 128-token prefill chunk size;
both ranks agree in every run. These are functional checks through the sharded
Python API, separate from the cluster API measurements above.

The speculative verifier reuses Mistral Large 4's masked NVFP4 gate/up kernel
through a shared module and evaluates its graph every four layers. It retains
Inkling's routing, activation and FP32 accumulation.
For two or four requests with eight expert routes, sorted pairs selecting the same
NVFP4 expert reuse its packed gate/up weights. Expert boundaries and unowned
routes preserve independent native reductions. With four requests, the down
projection also reuses weights within sorted expert groups, using the native
256- or 512-element reduction block for its input width. The final eight MXFP8
layers pair matching routes and skip unowned experts in all three projections.
Other shapes, precisions and quantization layouts use the existing paths.
Masked convolution rollback selects all prefix states in one operation, with
a compiled portable fallback. This replaces seven separate prefix-state
calculations per convolution cache for the trained eight-token block.

Short decoding also caches the target vocabulary weights' exact FP32 conversion.
With two expert ranks, each computes complete vocabulary rows and gathers scores
in vocabulary order; no dot product is split across devices. The checkpoint
weights remain BF16. Routing, residual mixing and activation graphs are compiled
without changing their FP32 boundaries. A short-convolution kernel reads the
state and masked input directly, preserving ordered FP32 arithmetic while avoiding
an intermediate concatenation. Copy-only cache kernels commit ragged accepted
prefixes; growing windows, dtype changes and custom cache updates use the native
fallback. Locally owned draft requests share dense projection work through the
same native-reduction kernel used by Mistral Large 4, retaining separate attention
and recurrent state per request.

An earlier repeated native-scheduler benchmark on the same two M3 Ultras used 1,025
input tokens per request, 128 output tokens, greedy sampling, a per-row prefill
chunk of 128, fresh KV caches and the normal allocator cache after loading.
The draft remained resident in every mode. Medians from three measured runs
after warmup, alternating order and using the slower rank's timing:

| Concurrent requests | DSpark off tokens/s | DSpark block 8 tokens/s | DSpark block 2 tokens/s |
| --- | ---: | ---: | ---: |
| 1 | 13.59 | 19.13 | Not measured |
| 4 | 35.67 | 30.04 | 39.17 |

Rates for four requests are aggregate throughput. The original eight-token
verification block improves from 26.53 to 30.04 tokens/s (13.2%) after distributing
draft requests and pairing expert projections. It still runs slower than
ordinary batch-four decoding. Setting `draft_block_size=2` in the native batch
generator verifies one proposal at a time and reaches 1.10x ordinary decode
throughput, or 1.06x including prefill. The draft transformer still evaluates its
full trained block of eight positions before selecting the requested proposals.
The default block size remains eight; the smaller block is explicit tuning.

Batch-four block-eight throughput ranged from 30.00 to 30.04 tokens/s, and
block-two throughput ranged from 39.13 to 39.45. Single-request DSpark ranged
from 19.11 to 19.16 tokens/s, retaining its 1.41x decode speedup and 1.17x speedup
including prefill. Peak MLX allocation was 209.2 GB/rank for one request and
210.8 GB/rank for four, with no swap growth. Batch-four prefill was about 335
aggregate tokens/s, with time to first token around 12.2 seconds with DSpark.
These measurements do not establish a 1.5–2x speedup over ordinary decoding.

Both ranks and all repetitions agreed. All 20 logical runs preserved the prior
tokens and acceptance records for their respective block size. Released-weight
checks cover 240 paired expert comparisons across the two ranks with bitwise
agreement, plus exact draft proposals and owned-cache states across ragged
contexts, row removal, admission and a resumed singleton owned by rank one.
That revision passed 258 relevant tests. Naive dense draft batching was not
retained because it changed aligned proposal logits; an earlier convolution
kernel that still materialized the FP32 input also failed to improve throughput.

All 128 single-request tokens match DSpark off. Existing batch-four differences
from ordinary decoding remain: first differing positions are 94, 43, 68 and 18
(zero-based) with block eight. With block two they are 83, 6 and 16 in the first
three requests; the fourth matches all 128 tokens. Changing verification depth
changes execution geometry and can change greedy output. These are retained
MLX comparisons, not evidence of zero BF16 disagreement with ordinary decoding
or the CPU reference. Keep DSpark and the shorter verification block opt-in.

The subsequent shared-model baseline reached 49.97 aggregate tokens/s with
DSpark block three, versus 40.66 without DSpark. A separate two-replica result
of 63.19 tokens/s did not satisfy the shared-model requirement and is excluded
from that target.

The retained implementation now partitions attention heads and expert weights
across the same two M3 Ultras. Neither Mac holds a complete target model.
A fresh benchmark uses the same requests, native mixed precision, 1,025 input
and 128 output tokens per request, 128-token per-row prefill chunks, fresh caches,
and a resident bundled draft in both modes. Three measured repetitions follow
warmup, with alternating mode/case order and the slower rank's timing:

| Concurrent requests | DSpark off tokens/s | DSpark on tokens/s | Draft block | Decode speedup | Peak GB/Mac |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 43.28 | 62.54 | 3 | 1.45x | 204.93 |
| 1 | 16.87 | 24.78 | 8 | 1.47x | 203.41 |

The batch-four DSpark repetitions were 62.54, 62.80 and 61.39 tokens/s: all three
exceeded 60. These are aggregate decode rates for one shared model, counting
each logical token once. The 62.54 median improves on the previous shared-model
49.97 median by 25.1%. Batch-four throughput including prefill is 25.06 tokens/s
with DSpark versus 21.15 without; median time to first token is 12.34 versus
12.45 seconds. The matrix-shape fallback protects numerical parity but increases
DSpark TTFT from the prior 11.67 seconds. Single-request DSpark ranges from
24.72 to 24.80 tokens/s. Swap usage stayed unchanged on both Macs, with no new
swap-outs. This remains a native distributed BatchGenerator benchmark; cluster
API speculative serving is not supported.

Both ranks completed all 16 logical runs with matching source fingerprints.
All generated tokens and acceptance records match their respective previous
shared-model controls. Released-weight validation covers 4,182 aligned positions
per rank, including the 1,025-token input, larger verification blocks, ragged
acceptance, abort and row filtering. Maximum, mean, RMS and relative RMS logit
errors are zero, with zero raw or normalized top-token disagreements. Captured
states, owned target-cache slices, batched draft logits and owned draft caches
match exactly. Another 144 real-weight expert checks pass on each rank, covering
all 48 layers at lengths one, three and eight; 248 focused regression tests pass.
An initially split matrix path failed the eight-token check and was replaced by
the zero-padding fallback described above before these final measurements.

These checks preserve the previous MLX arithmetic and native mixed BF16/FP32
boundaries. They do not remove the existing ON/OFF or CPU-reference differences.
Batch-four block-three ON/OFF first divergences remain 33, 43, 98 and 30
(zero-based); the single-request ON/OFF outputs match all 128 tokens.

Audio input is not supported by this model.

## Target numerical validation

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
