"""Measure OpenVLA-7B under bf16 / int8 / nf4 on GPU, and on CPU.

Two things make OpenVLA-7B hard to run on cheap hardware today, and this script
targets both:

1. Dependency wall — the checkpoint's remote code was written against
   torch 2.2 / transformers 4.40 / timm 0.9.10 (see openvla/openvla
   pyproject.toml). Modern stacks are 2 years newer.
2. Broken low-bit path — openvla/openvla issues #286 and #287 (open since
   2025-07-30) report 4-bit loading crashing on a `.to()` call, and #311
   ("can't run on multi-GPU rigs with <=11 GB cards", open, zero replies)
   is exactly the cheap-hardware case.

The `.to()` crash is not a bitsandbytes bug: a 4-bit model is already placed on
its device by accelerate, so moving it afterwards is invalid. The published
OpenVLA snippet does `.from_pretrained(...).to("cuda:0")`, which is why every
user following the README hits it. Here we pass device_map instead and never
call .to() on a quantized model.

Usage:
  PYTHONPATH= ~/miniconda3/envs/vla/bin/python bench/openvla_runner.py \
      --model openvla/openvla-7b --precision nf4
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from measure import (
    Result,
    VramTracker,
    action_fidelity,
    host_rss_gb,
    markdown_table,
    save,
    time_calls,
)

PROMPT = "In: What action should the robot take to pick up the black bowl?\nOut:"


def build_quant_config(precision: str):
    """bitsandbytes config, or None for plain dtype loading."""
    if precision not in {"int8", "nf4", "fp4"}:
        return None
    from transformers import BitsAndBytesConfig

    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4" if precision == "nf4" else "fp4",
        # compute in bf16: the weights are 4-bit but matmuls run in bf16, which
        # is what keeps action quality from collapsing
        bnb_4bit_compute_dtype=torch.bfloat16,
        # second-level quantization of the scales; buys a few hundred MB
        bnb_4bit_use_double_quant=True,
    )


def patch_openvla_remote_code(model_id: str):
    """Make the 2024-era checkpoint code importable on a modern transformers.

    openvla's modeling_prismatic.py declares:

        @property
        def _supports_sdpa(self) -> bool:
            return self.language_model._supports_sdpa

    transformers >=4.5x resolves the attention implementation inside
    PreTrainedModel.__init__, which runs *before* self.language_model is
    assigned. The property therefore raises, and nn.Module.__getattr__ reports
    it as "'OpenVLAForActionPrediction' object has no attribute
    '_supports_sdpa'" — the real cause is ordering, not a missing attribute.

    The Llama backbone does support SDPA, so replacing the property with a
    plain class attribute is behaviour-preserving.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.generation.utils import GenerationMixin

    # The finetuned LIBERO checkpoints ship weights only and point at the base
    # repo's modeling code via auto_map, so fall back to it for the class.
    try:
        cls = get_class_from_dynamic_module(
            "modeling_prismatic.OpenVLAForActionPrediction", model_id
        )
    except OSError:
        cls = get_class_from_dynamic_module(
            "modeling_prismatic.OpenVLAForActionPrediction", "openvla/openvla-7b"
        )
    patched = []
    for klass in cls.__mro__:
        attr = klass.__dict__.get("_supports_sdpa")
        if isinstance(attr, property):
            setattr(klass, "_supports_sdpa", True)
            patched.append(f"{klass.__name__}._supports_sdpa")

    # Second incompatibility: transformers >=4.50 stopped having PreTrainedModel
    # inherit GenerationMixin. OpenVLA defines prepare_inputs_for_generation and
    # calls self.generate() inside predict_action, but never inherits the mixin,
    # so .generate vanished. Re-mix it in *after* cls so the checkpoint's own
    # overrides still take precedence in the MRO.
    # Third and worst incompatibility — silent, not a crash.
    #
    # OpenVLA's prepare_inputs_for_generation does:
    #     if past_key_values is not None: input_ids = input_ids[:, -1:]
    # Under transformers 4.40 the first generate() step passed past_key_values=None.
    # Modern transformers passes an *empty* Cache object instead, which is not
    # None, so the 35-token prompt is sliced to its last token on step 0.
    #
    # forward() then routes on input_ids.shape[1] == 1 into the "cached
    # generation" branch, which never touches pixel_values. Vision is therefore
    # dropped entirely and the model emits one constant token forever
    # (measured: token 31872 x7 for black, white, noise and real frames alike),
    # which unnormalizes into a plausible-looking but input-independent action.
    #
    # The multimodal branch additionally asserts past_key_values is None, so it
    # is not enough to restore the full prompt: an empty cache must be passed
    # through as None on the first step.
    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
    ):
        cache_len = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                cache_len = past_key_values.get_seq_length() or 0
            else:
                try:
                    cache_len = past_key_values[0][0].shape[2]
                except Exception:
                    cache_len = 0

        if cache_len > 0:
            model_inputs = {"input_ids": input_ids[:, -1:], "past_key_values": past_key_values}
        elif inputs_embeds is not None:
            model_inputs = {"input_embeds": inputs_embeds, "past_key_values": None}
        else:
            # first step: full prompt, and no cache object at all
            model_inputs = {"input_ids": input_ids, "past_key_values": None}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        return model_inputs

    overrides = {"prepare_inputs_for_generation": prepare_inputs_for_generation}
    if not issubclass(cls, GenerationMixin):
        bases = (cls, GenerationMixin)
        patched.append("re-mixed GenerationMixin")
    else:
        bases = (cls,)
    cls = type(f"{cls.__name__}Modernized", bases, overrides)
    patched.append("empty-Cache-aware prepare_inputs_for_generation")

    return cls, patched


def load_openvla(model_id: str, precision: str, device: str):
    from transformers import AutoProcessor

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_cls, patched = patch_openvla_remote_code(model_id)
    if patched:
        print(f"[patch] replaced _supports_sdpa property on: {', '.join(patched)}")

    quant_cfg = build_quant_config(precision)
    kwargs: dict = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
        # Critical: let accelerate place the model. Calling .to() afterwards is
        # what raises the error in issues #286/#287.
        kwargs["device_map"] = {"": 0} if device.startswith("cuda") else "cpu"
    else:
        kwargs["dtype"] = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model = model_cls.from_pretrained(model_id, **kwargs)

    if quant_cfg is None:
        # dtype= does not always reach the nested language_model config
        model = model.float() if precision == "fp32" else model.to(torch.bfloat16)
        model = model.to(device)

    model.eval()
    return model, processor, time.perf_counter() - t0


_REAL_FRAMES: list[tuple[Image.Image, str]] = []


def load_real_frames(n: int, repo: str = "HuggingFaceVLA/libero", stride: int = 17):
    """Real LIBERO observations + their real task instructions.

    Random-noise images are unusable for fidelity work: the policy emits one
    constant action regardless of input, so every precision "agrees" perfectly
    and the comparison is vacuous (measured: action std 0.0 across samples).
    Frames are strided across the episode so the samples differ from each other.
    """
    global _REAL_FRAMES
    if _REAL_FRAMES:
        return _REAL_FRAMES

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo, episodes=[0])
    key = "observation.images.image"
    out = []
    for i in range(n):
        s = ds[(i * stride) % ds.num_frames]
        arr = (s[key].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        out.append((Image.fromarray(arr).resize((224, 224)), str(s["task"])))
    _REAL_FRAMES = out
    return out


def vision_dtype(model) -> torch.dtype:
    """dtype the (unquantized) vision tower actually holds.

    With bitsandbytes the LLM is 4-bit but the vision backbone stays in the
    checkpoint's native dtype — fp16 for these OpenVLA releases — so feeding
    bf16 pixels raises "Input type (BFloat16) and bias type (Half) should be
    the same". Read it off the model instead of assuming.
    """
    try:
        return next(model.vision_backbone.parameters()).dtype
    except Exception:
        return torch.float32


def make_inputs(processor, device: str, precision: str, seed: int = 0, real: bool = False,
                img_dtype: torch.dtype | None = None):
    if real:
        frames = load_real_frames(max(16, seed % 1000 + 1))
        img, task = frames[seed % len(frames)]
        prompt = f"In: What action should the robot take to {task.lower().rstrip('.')}?\nOut:"
        inputs = processor(prompt, img)
    else:
        rng = np.random.default_rng(seed)
        img = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = processor(PROMPT, img)
    dtype = img_dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
        else:
            out[k] = v
    return out


def predict(model, inputs, unnorm_key: str):
    with torch.no_grad():
        return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)


def collect_actions(model, processor, device, precision, unnorm_key, n, real=False) -> np.ndarray:
    acts = []
    for i in range(n):
        inp = make_inputs(processor, device, precision, seed=i, real=real, img_dtype=vision_dtype(model))
        a = predict(model, inp, unnorm_key)
        acts.append(np.asarray(a, dtype=np.float64).reshape(1, 1, -1))
    return np.concatenate(acts, axis=0)


def run(args) -> Result:
    torch.manual_seed(0)
    model, processor, load_s = load_openvla(args.model, args.precision, args.device)

    # figure out a valid unnorm key from the checkpoint itself
    unnorm_key = args.unnorm_key
    stats = getattr(model, "norm_stats", None)
    if unnorm_key is None:
        if isinstance(stats, dict) and stats:
            unnorm_key = sorted(stats)[0]
        else:
            unnorm_key = "bridge_orig"

    with VramTracker(args.device) as vt:
        inputs = make_inputs(processor, args.device, args.precision, real=args.real_frames, img_dtype=vision_dtype(model))
        first = predict(model, inputs, unnorm_key)
        lat = time_calls(
            lambda: predict(model, inputs, unnorm_key), args.warmup, args.iters, args.device
        )

    res = Result(
        label=args.label or f"openvla-{args.precision}-{args.device}",
        model=args.model,
        dtype=args.precision,
        device=args.device,
        load_seconds=round(load_s, 2),
        weights_gb=round(
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9, 3
        ),
        peak_vram_gb=vt.peak_allocated_gb,
        peak_host_rss_gb=host_rss_gb(),
        latency_ms=lat,
        control_hz=round(1000.0 / lat["p50"], 2) if lat.get("p50") else None,
        notes=[
            f"unnorm_key={unnorm_key}; action dim={np.asarray(first).shape}",
            f"peak VRAM reserved {vt.peak_reserved_gb} GB",
            "single-step action prediction (OpenVLA emits one action per forward)",
            ("real LIBERO frames" if args.real_frames else "synthetic image: valid for memory/latency only"),
        ],
    )

    if args.save_actions:
        acts = collect_actions(model, processor, args.device, args.precision, unnorm_key, args.n_fidelity, args.real_frames)
        Path(args.save_actions).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_actions, acts)
        res.notes.append(f"saved actions {acts.shape} -> {args.save_actions}")

    if args.compare_actions:
        ref = np.load(args.compare_actions)
        acts = collect_actions(model, processor, args.device, args.precision, unnorm_key, args.n_fidelity, args.real_frames)
        res.fidelity = action_fidelity(ref, acts)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "int8", "nf4", "fp4"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default=None)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--n-fidelity", type=int, default=8)
    ap.add_argument("--save-actions", default=None)
    ap.add_argument("--compare-actions", default=None)
    ap.add_argument("--real-frames", action="store_true", help="use real LIBERO frames (required for meaningful fidelity)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    res = run(args)
    print(res.to_json())
    print(markdown_table([res]))
    print("saved ->", save(res, Path(args.out)))


if __name__ == "__main__":
    main()
