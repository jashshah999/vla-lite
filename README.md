# openvla-compat

Run OpenVLA-7B on a current `transformers` stack, and on an 8 GB GPU.

**Why this exists:** on `transformers` ≥4.50, OpenVLA-7B loads, runs, and returns
plausible 7-DoF actions — but the actions are **identical for every image**,
including a pure black frame and a pure white frame. The vision tower is never
consumed. No error is raised. If you followed the official README on a 2026
stack, your policy has been blind. Upstream issue:
[openvla/openvla#346](https://github.com/openvla/openvla/issues/346).

## Install and check

```bash
pip install "openvla-compat[quant] @ git+https://github.com/jashshah999/vla-lite"
openvla-compat-check     # black frame vs white frame must NOT give the same action
```

Requires `transformers` 4.50–4.x and `timm<1.0` (the checkpoint's remote code
rejects `timm>=1.0`). Verified on transformers 4.56.2 / torch 2.10 / timm 0.9.16 /
bitsandbytes 0.50.

```python
from openvla_compat import load_openvla, predict_action
from PIL import Image

model, processor = load_openvla("openvla/openvla-7b-finetuned-libero-spatial", precision="nf4")
action = predict_action(model, processor, Image.open("frame.png"),
                        "pick up the black bowl", unnorm_key="libero_spatial")
```

`precision` is one of `bf16 | fp32 | int8 | nf4 | fp4`. No changes to the
checkpoint are needed; the fixes are applied to the model class at load time.

## The bug

`openvla/openvla-7b` pins `transformers==4.40.1`. On a modern stack there are
three failures. The first two are loud; the third is silent.

| # | Symptom | Cause |
|---|---|---|
| 1 | `AttributeError: ... no attribute '_supports_sdpa'` | declared as a property reading `self.language_model`, which transformers now touches inside `PreTrainedModel.__init__`, before that submodule exists |
| 2 | `AttributeError: ... no attribute 'generate'` | `PreTrainedModel` no longer inherits `GenerationMixin`; the checkpoint never inherits it explicitly |
| 3 | **no error — constant actions** | `prepare_inputs_for_generation` slices the prompt whenever `past_key_values is not None`. Modern transformers passes an **empty `DynamicCache`** on step 0, so the 35-token prompt collapses to 1 token, `forward` takes its cached-generation branch, and `pixel_values` is never read |

Bug 3, measured on `openvla-7b-finetuned-libero-spatial`: token `31872` emitted
seven times for a real LIBERO frame, a different real frame, pure black, pure
white, and random noise — every input unnormalizing to
`[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]`.

After the fix, the same frames give distinct, image-dependent actions, with the
gripper channel switching between 0.0 and 0.996:

| input | action |
|---|---|
| real frame 0 | `[0.3336, 0.4689, -0.252, 0.0, -0.021, -0.0387, 0.9961]` |
| real frame 5 | `[-0.003, -0.0014, 0.0926, 0.0, 0.0, -0.0206, 0.0]` |
| real frame 9 | `[-0.2076, -0.1401, -0.9338, 0.0149, -0.1459, 0.0001, 0.9961]` |

The fix (in [`openvla_compat.py`](openvla_compat.py)) gates the slice on actual
cache *length* and forwards an empty cache as `None`, since the multimodal
branch asserts `past_key_values is None`. It was verified on the official
`AutoModelForVision2Seq` load path: with only the unavoidable `_supports_sdpa`
patch, `vision_backbone` and `projector` forward hooks fire **0 times** and the
language model receives `input_ids` of shape `(1, 1)` on every step, including
the first.

The 4-bit `.to()` crash from
[#286](https://github.com/openvla/openvla/issues/286)/[#287](https://github.com/openvla/openvla/issues/287)
is also handled: a bitsandbytes model is placed by accelerate via `device_map`
and must never have `.to()` called on it.

## Measurements

`openvla-7b-finetuned-libero-spatial`, 16 real LIBERO frames, greedy decoding,
all fixes applied. Fidelity is against the bf16 run on identical inputs.
RTX 5000 Ada Laptop (16 GB). Raw JSON per row in [`results/`](results/).

| config | peak VRAM | p50 latency | action L1 vs bf16 | cosine | gripper flips |
|---|---|---|---|---|---|
| bf16 | 15.29 GB | 260 ms | — | — | — |
| int8 | 8.05 GB | 422 ms | 0.071 | 0.867 | 1 / 16 |
| **nf4** | **4.60 GB** | 245 ms | 0.132 | 0.743 | 2 / 16 |
| fp32, CPU only | 31 GB RAM | 10,500 ms | 0.000 | 1.000 | 0 / 4 |

- **The memory win is real and is the point.** 15.3 → 4.6 GB (3.3×). bf16 does not
  fit a 12 GB card; nf4 fits an 8 GB card with room for the rest of a robot stack.
- **The speed win is not real.** nf4 is within noise of bf16, and int8 is slower
  because bitsandbytes dequantizes per matmul. Weight-only quantization shrinks
  footprint; it does not make VLA inference fast.
- **4-bit changes behaviour.** OpenVLA discretizes each action dimension into 256
  bins, so a small numerical error either does nothing or flips a whole bin.
  fp32-on-CPU is therefore *byte-identical* to bf16-on-GPU, while nf4 flips the
  gripper on 2 of 16 frames — a different grasp decision, not a rounding error.
  **int8 is the safer default if it fits.**
- CPU-only works (no CUDA, 31 GB RAM) at ~0.1 Hz — useful for CI and offline
  evaluation, not control.

Reproduce:

```bash
cd bench
for p in bf16 int8 nf4; do
  python openvla_runner.py --model openvla/openvla-7b-finetuned-libero-spatial \
    --precision $p --device cuda --real-frames --n-fidelity 16 --label fix-$p
done
```

## Limitations

- **No closed-loop success rate.** These are open-loop action deltas on 16 frames
  from one episode. Action error is a regression gate, not proof of task
  competence; the standard is LIBERO (50 trials × 10 tasks × 4 suites) and it has
  not been run here.
- 16 samples, one instruction. The 2/16 gripper-flip rate has a wide interval;
  treat it as a flag, not a rate.
- nf4/int8 on CPU is not covered (bitsandbytes low-bit kernels are CUDA-side).

## Prior art

Quantized OpenVLA is not new — the OpenVLA paper itself reports int4/int8 with
real-robot success rates, and [`parastoopil/openvla-1bit`](https://github.com/parastoopil/openvla-1bit)
has a more thorough quantization-damage study. The `_supports_sdpa` fix was
first published in [`moojink/openvla-oft#108`](https://github.com/moojink/openvla-oft/issues/108).
The silent vision bypass (bug 3) had no prior report in any openvla issue, HF
discussion, or fork as of Sep 2026 (search notes in
[`novelty-audit.json`](novelty-audit.json)). It is an instance of a common bug
family — custom multimodal `generate()`/cache handling breaking silently across
transformers upgrades — not a new idea.

## License

MIT. OpenVLA weights are MIT, so quantized derivatives are redistributable.
