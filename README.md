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
python openvla_compat.py     # black vs white must not match
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

| config | peak VRAM | weights | p50 latency | p99 | action L1 vs bf16 | rel. err (÷ action σ) | cosine | gripper flips |
|---|---|---|---|---|---|---|---|---|
| bf16 | 15.29 GB | 15.08 GB | 260 ms | 263 ms | — (reference) | — | — | — |
| int8 | 8.05 GB | 7.81 GB | 422 ms | 423 ms | 0.071 | 0.51 | 0.867 | 1 / 16 |
| **nf4** | **4.60 GB** | **4.17 GB** | **245 ms** | 247 ms | 0.132 | 0.83 | 0.743 | 2 / 16 |

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

**Quantization changes the policy's behaviour, and 4-bit changes it a lot.**
A relative error of 0.83 means the average action deviates by ~83% of that action
dimension's own standard deviation, and cosine similarity of 0.743 means the
action *direction* often differs. Most seriously, the gripper channel disagrees
on 2 of 16 frames — an open-vs-closed flip is not a rounding error, it is a
different behaviour at the moment that decides whether a grasp happens.
int8 is roughly half as damaging on every metric and is the better default if
you have 8 GB.

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
- CPU-only inference is not reported: the fp32 CPU path needs a consistent-dtype
  load and was still being fixed when these numbers were taken.

## Use it

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
