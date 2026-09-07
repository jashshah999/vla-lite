# OpenVLA-7B on a small GPU — and a silent bug you should know about

Two things, both measured on one laptop GPU:

1. **OpenVLA-7B is currently broken on modern `transformers`, and it fails silently.**
   It loads, it runs, it returns plausible 7-DoF actions — and the actions are
   *identical for every image*, including a pure black frame and a pure white
   frame. The vision tower is never consumed. If you followed the official README
   on a 2026 stack, your policy has been blind.
2. **Once fixed, it runs in 4.6 GB of VRAM instead of 15.3 GB** — so a 7B
   vision-language-action model fits on an 8 GB consumer card. But quantization is
   *not* free, and the numbers below say exactly how much it costs.

## The bug

`openvla/openvla-7b` pins `torch==2.2.0`, `transformers==4.40.1`, `timm==0.9.10`
in its own `pyproject.toml`. On a current stack there are three failures. The
first two are loud; the third is the dangerous one.

| # | Symptom | Cause |
|---|---|---|
| 1 | `AttributeError: ... has no attribute '_supports_sdpa'` | the checkpoint declares `_supports_sdpa` as a **property** that reads `self.language_model`; transformers now resolves attention inside `PreTrainedModel.__init__`, before that submodule exists |
| 2 | `AttributeError: ... has no attribute 'generate'` | transformers ≥4.50 stopped having `PreTrainedModel` inherit `GenerationMixin`, and the checkpoint never inherits it explicitly |
| 3 | **no error — constant actions** | `prepare_inputs_for_generation` slices the prompt whenever `past_key_values is not None`. Modern transformers passes an **empty `DynamicCache`** on step 0 instead of `None`, so the 35-token prompt collapses to 1 token, `forward` takes its `input_ids.shape[1] == 1` cached-generation branch, and `pixel_values` is never read |

Bug 3, measured on `openvla-7b-finetuned-libero-spatial` before the fix — token
`31872` emitted seven times for *every* input:

| input | action |
|---|---|
| real LIBERO frame 0 | `[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]` |
| real LIBERO frame 9 | `[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]` |
| pure black image | `[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]` |
| pure white image | `[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]` |
| random noise | `[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]` |

After the fix, the same frames produce genuinely different actions:

| input | action |
|---|---|
| real frame 0 | `[0.3336, 0.4689, -0.252, 0.0, -0.021, -0.0387, 0.9961]` |
| real frame 5 | `[-0.003, -0.0014, 0.0926, 0.0, 0.0, -0.0206, 0.0]` |
| real frame 9 | `[-0.2076, -0.1401, -0.9338, 0.0149, -0.1459, 0.0001, 0.9961]` |

Note the gripper channel changing between 0.0 and 0.996 — the policy is
responding to what it sees.

There is a one-command check for this, because "it returned an action" is not
evidence that a VLA works:

```bash
pip install "openvla-compat[quant] @ git+https://github.com/jashshah999/vla-lite"
openvla-compat-check         # black vs white must not match
```

The fix is ~40 lines in [`openvla_compat.py`](openvla_compat.py) and needs no
changes to the checkpoint. It also resolves the `.to()` crash on 4-bit loading
(issues [#286](https://github.com/openvla/openvla/issues/286),
[#287](https://github.com/openvla/openvla/issues/287)) — a bitsandbytes model is
placed by accelerate via `device_map` and must never have `.to()` called on it —
and [#311](https://github.com/openvla/openvla/issues/311) ("can't run on rigs
with ≤11 GB cards", open with zero replies since Oct 2025).

## Measurements

`openvla/openvla-7b-finetuned-libero-spatial`, 16 real LIBERO frames from
`HuggingFaceVLA/libero` episode 0, greedy decoding, all three fixes applied.
Fidelity is measured against the **bf16** run on byte-identical inputs.

| config | peak VRAM | host RAM | p50 latency | action L1 vs bf16 | rel. err (÷ action σ) | cosine | gripper flips |
|---|---|---|---|---|---|---|---|
| bf16 (GPU) | 15.29 GB | 1.8 GB | 260 ms | — (reference) | — | — | — |
| int8 (GPU) | 8.05 GB | 2.3 GB | 422 ms | 0.071 | 0.51 | 0.867 | 1 / 16 |
| **nf4 (GPU)** | **4.60 GB** | 2.4 GB | **245 ms** | 0.132 | 0.83 | 0.743 | 2 / 16 |
| fp32 (**CPU only**) | none | 31.2 GB | 10,523 ms | **0.000** | 0.000 | 1.000 | 0 / 4 |

The CPU row uses 4 frames (it takes ~10.5 s per action); the GPU rows use 16.

Rig: RTX 5000 Ada Laptop (16.8 GB, compute 8.9), driver 535.183.01,
torch 2.10.0+cu128, transformers 4.56.2, timm 0.9.16, bitsandbytes 0.50.0,
Python 3.11. Raw JSON for every row is in [`results/`](results/).

### What the numbers actually say

**The memory win is real and it is the point.** 15.29 GB → 4.60 GB is 3.3×.
bf16 does not fit a 12 GB card at all; nf4 fits an 8 GB card with room for the
rest of a robot stack. That is the difference between "needs a workstation" and
"runs on the laptop you own".

**The speed win is not real.** nf4 is 245 ms vs bf16's 260 ms — within noise of
each other, and int8 is *slower* (422 ms) because bitsandbytes' int8 path
dequantizes per matmul. Weight-only quantization shrinks footprint; it does not
make VLA inference fast. Anyone promising both is selling something. (bnb also
warns that the fused vision hidden dim, 4304, is unaligned for its fast kernel
and falls back to a slower path — a real optimization opportunity, untouched here.)

**It runs with no GPU at all** — 31.2 GB of system RAM, no CUDA — but at
10.5 s per action (0.1 Hz) it is not a controller. It *is* useful for debugging,
CI, and offline evaluation on machines without a GPU.

**Why fp32-on-CPU matches bf16-on-GPU exactly** (L1 = 0.000, cosine = 1.000,
4/4 distinct actions, so this is not the degenerate case): OpenVLA discretizes
each action dimension into 256 bins and decodes them as tokens. fp32-vs-bf16
numerical differences are far too small to move an `argmax` across a bin
boundary, so the *same* tokens come out. That is also the mechanism behind the
quantization damage above — nf4 error *is* large enough to flip bins, and one
flipped bin is a visible jump in the action, not a rounding error. It explains
why a metric like "L1 = 0.13" understates the problem and why the gripper-flip
count is the number to watch.

**Quantization changes the policy's behaviour, and 4-bit changes it a lot.**
A relative error of 0.83 means the average action deviates by ~83% of that action
dimension's own standard deviation, and cosine similarity of 0.743 means the
action *direction* often differs. Most seriously, the gripper channel disagrees
on 2 of 16 frames — an open-vs-closed flip is not a rounding error, it is a
different behaviour at the moment that decides whether a grasp happens.
int8 is roughly half as damaging on every metric and is the better default if
you have 8 GB.

## Prior art, and what is actually new here

An adversarial novelty audit (3 agents, 111 verified lookups; the OpenVLA
portion of its notes is in [`novelty-audit.json`](novelty-audit.json)) was run
against this work. Most of it
is **not** new, and the specifics matter:

- **4-bit/8-bit OpenVLA is not new.** The OpenVLA paper itself
  (arXiv:2406.09246, Table 2) quantized this exact model to int4/int8 and
  reported VRAM alongside *real-robot success rates* — a more meaningful damage
  measure than the action-space deltas here.
- **The `_supports_sdpa` fix is not new.** It was already published in
  `moojink/openvla-oft` issue #108, over a year before this work.
- **A better quantization-damage study already exists.**
  `github.com/parastoopil/openvla-1bit` does Hessian-guided 1-bit GPTQ/BiLLM with
  gripper-specific error breakdowns, published before this.
- **CPU-only OpenVLA is already a reference benchmark** in Tenstorrent's public
  repo.
- The fp32-CPU ≡ bf16-GPU byte-identical result is a clean check but a
  *predictable* consequence of the paper's published 256-bin action
  discretization, not a discovery.

**The one item with no prior report: the silent vision bypass.** The audit
searched every openvla/openvla issue and PR (all states), every HuggingFace
discussion on openvla-7b and its four LIBERO checkpoints, ~30 forks'
`modeling_prismatic.py`, HN and arXiv, and found nothing describing this failure
mode. The closest are issue #148 ("Cached Generation vs Multimodal Forward",
closed with zero replies — it points at the exact code region but frames it as a
design question) and #62 (constant actions after fine-tuning, attributed to
training data). No fork has patched it.

It is worth being clear about the genre, though: silent breakage of custom
multimodal `generate()`/cache handling across transformers upgrades has hit
Florence2, DeepSeek-OCR and MiMo-Audio too. An ML infra engineer would recognize
the *family* instantly. This is a useful unreported bug in a widely-used
checkpoint, not a new idea.

### The mechanism, corrected

The audit challenged my original explanation on the grounds that
`_prepare_cache_for_generation` only injects an empty `DynamicCache` when
`_supports_cache_class` is True, which OpenVLA never sets. That reading is
correct about the attribute (verified: `_supports_cache_class = ABSENT`) but the
conclusion does not hold on transformers 4.56.2. Measured on the **official**
load path (`AutoModelForVision2Seq` + `trust_remote_code=True`, with only the
unavoidable `_supports_sdpa` patch, and transformers' own auto-injected
`GenerationMixin` confirmed present in the MRO):

- black, white and noise images all yield the identical action
  `[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]`
- `vision_backbone` forward hook: **0 calls**. `projector`: **0 calls**.
- the language model receives `input_ids` of shape **(1, 1) on all 7 steps**,
  including the first

The prompt is therefore sliced on step 0. That slice occurs in exactly one place
in the checkpoint's code, gated on `past_key_values is not None` — so
`past_key_values` *was* non-None on the first call under 4.56.2, whatever the
gating logic reads like in other versions. The bug is real on the official path
and is not an artifact of custom loading; I verified that specifically, because my
first implementation bypassed the Auto path and could have caused it.

## Honest limitations

- **No closed-loop success rate.** These are open-loop action deltas on 16 frames
  from one episode. Action error is a *regression gate*, not proof of task
  competence — HuggingFace's own writeups and lerobot#2853 document that
  validation loss/MSE does not track success rate for robot policies. The
  literature standard is LIBERO, 50 trials × 10 tasks × 4 suites, reported
  per-task with raw counts across ≥3 seeds. **That has not been run here**, so no
  claim about task success is made, in either direction.
- 16 samples, one episode, one instruction. The gripper-flip rate (2/16) has a
  wide confidence interval; treat it as a flag, not a measured rate.
- The frames come from a LIBERO episode whose task is a long-horizon two-mug
  placement, while the checkpoint is the `libero_spatial` finetune — a
  deliberately mild distribution mismatch, since the goal was to compare
  precisions on identical inputs, not to score the policy.
- The CPU row is 4 frames, not 16, and fp32 CPU needed an explicit `.float()`
  because the checkpoint config's `torch_dtype: bfloat16` does not always reach
  the nested language model via the `dtype=` kwarg (it surfaces as
  "expected m1 and m2 to have the same dtype").
- nf4/int8 on CPU is not covered: bitsandbytes' low-bit kernels are CUDA-side,
  so the CPU path here is full fp32 and needs 31 GB of RAM.

## Use it

```bash
pip install "openvla-compat[quant] @ git+https://github.com/jashshah999/vla-lite"
```

`transformers` 4.50–4.x and `timm<1.0` are required (the checkpoint's remote code
rejects `timm>=1.0`; `transformers` 5.x is untested). Then:

```python
from openvla_compat import load_openvla, predict_action
from PIL import Image

model, processor = load_openvla(
    "openvla/openvla-7b-finetuned-libero-spatial", precision="nf4"
)
action = predict_action(
    model, processor, Image.open("frame.png"),
    "pick up the black bowl", unnorm_key="libero_spatial",
)
```

Reproduce the table:

```bash
cd bench
for p in bf16 int8 nf4; do
  python openvla_runner.py \
    --model openvla/openvla-7b-finetuned-libero-spatial \
    --precision $p --device cuda --real-frames --n-fidelity 16 \
    --label fix-$p --save-actions ../results/actions/fix_$p.npy
done
```

`bench/measure.py` holds the measurement primitives (peak VRAM via
`max_memory_allocated` *and* the allocator's reserved high-water mark, latency
percentiles rather than means because a controller lives on the tail, and the
fidelity metrics above). `bench/measure.py --selftest` verifies the harness
itself on synthetic tensors.

## Licensing

OpenVLA weights are MIT (`openvla/openvla-7b`), so quantized derivatives are
redistributable. Code here is MIT. Note that pi0-family checkpoints, by
contrast, carry Gemma Terms of Use obligations — if you extend this to those
models, the terms must travel with the weights.
