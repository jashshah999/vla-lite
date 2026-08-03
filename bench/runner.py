"""Load a LeRobot VLA policy under a given precision/quantization and measure it.

Usage:
  PYTHONPATH= ~/miniconda3/envs/vla/bin/python bench/runner.py \
      --model lerobot/smolvla_libero --policy smolvla --dtype bf16 --device cuda

Fidelity note: synthetic inputs are valid for memory/latency, and valid for a
*relative* fp-vs-quantized action comparison because both models see byte-identical
inputs. They are NOT a substitute for real frames when judging whether a policy
still completes tasks — that needs the LIBERO closed-loop eval.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from measure import (
    Result,
    VramTracker,
    action_fidelity,
    host_rss_gb,
    markdown_table,
    save,
    time_calls,
)

POLICY_CLASSES = {
    "smolvla": ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
    "pi0": ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
    "pi05": ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
    "pi0fast": ("lerobot.policies.pi0_fast.modeling_pi0_fast", "PI0FASTPolicy"),
}

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_policy(policy: str, model_id: str, dtype: str, device: str):
    """Load the policy plus its own pre/post processor pipelines.

    The processors are not optional decoration: they tokenize the task string
    into observation.language.tokens and apply the checkpoint's normalization
    stats, so running without them measures a different (and wrong) model.
    """
    import importlib

    from lerobot.policies.factory import make_pre_post_processors

    mod_name, cls_name = POLICY_CLASSES[policy]
    cls = getattr(importlib.import_module(mod_name), cls_name)

    t0 = time.perf_counter()
    pol = cls.from_pretrained(model_id)
    if dtype in DTYPES:
        pol = pol.to(dtype=DTYPES[dtype])
    pol = pol.to(device=device)
    pol.eval()
    load_s = time.perf_counter() - t0

    pre, post = make_pre_post_processors(pol.config, pretrained_path=model_id)
    return pol, load_s, pre, post


def weights_gb(pol) -> float:
    """Actual bytes held by parameters + buffers, whatever the dtypes are."""
    total = sum(p.numel() * p.element_size() for p in pol.parameters())
    total += sum(b.numel() * b.element_size() for b in pol.buffers())
    return round(total / 1e9, 3)


def make_batch(pol, device: str, dtype: str, seed: int = 0) -> dict[str, torch.Tensor]:
    """Build one correctly-shaped observation from the policy's own feature spec."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cfg = pol.config
    batch: dict[str, torch.Tensor] = {}
    tdt = DTYPES.get(dtype, torch.float32)

    for key, ft in (cfg.input_features or {}).items():
        shape = tuple(ft.shape)
        if "image" in key:
            # images in [0,1]; policy-side processors handle resize/normalize
            t = torch.rand((1, *shape), generator=g, dtype=torch.float32)
        else:
            t = torch.randn((1, *shape), generator=g, dtype=torch.float32) * 0.1
        batch[key] = t.to(device=device, dtype=tdt if "image" in key else torch.float32)

    # language instruction: most VLAs require a task string
    batch["task"] = ["pick up the black bowl and place it on the plate"]
    return batch


def infer_once(pol, pre, post, batch):
    with torch.no_grad():
        obs = pre(dict(batch))
        chunk = pol.predict_action_chunk(obs)
        return post(chunk)


def collect_actions(pol, pre, post, device: str, dtype: str, n: int) -> np.ndarray:
    """Actions over n distinct fixed-seed observations, for fidelity comparison.

    Flow-matching policies sample noise internally, so the noise is pinned per
    sample here; otherwise the fp-vs-quantized delta would be dominated by
    sampling randomness rather than by numerical error.
    """
    outs = []
    for i in range(n):
        torch.manual_seed(1000 + i)
        b = make_batch(pol, device, dtype, seed=1000 + i)
        with torch.no_grad():
            a = post(pol.predict_action_chunk(pre(dict(b))))
        outs.append(a.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def run(args) -> Result:
    torch.manual_seed(0)
    pol, load_s, pre, post = load_policy(args.policy, args.model, args.dtype, args.device)

    with VramTracker(args.device) as vt:
        batch = make_batch(pol, args.device, args.dtype)
        infer_once(pol, pre, post, batch)  # includes any lazy init
        lat = time_calls(lambda: infer_once(pol, pre, post, batch), args.warmup, args.iters, args.device)

    chunk = infer_once(pol, pre, post, batch)
    n_steps = int(getattr(pol.config, "n_action_steps", chunk.shape[1]))
    # A chunk of N actions consumed at one action per control tick means the
    # policy only has to run every N ticks — that is the number a robot cares about.
    per_call_s = lat["p50"] / 1e3
    control_hz = (n_steps / per_call_s) if per_call_s > 0 else None

    res = Result(
        label=args.label or f"{args.policy}-{args.dtype}-{args.device}",
        model=args.model,
        dtype=args.dtype,
        device=args.device,
        load_seconds=round(load_s, 2),
        weights_gb=weights_gb(pol),
        peak_vram_gb=vt.peak_allocated_gb,
        peak_host_rss_gb=host_rss_gb(),
        latency_ms=lat,
        control_hz=round(control_hz, 1) if control_hz else None,
        notes=[
            f"action chunk shape {tuple(chunk.shape)}; n_action_steps={n_steps}",
            f"peak VRAM reserved {vt.peak_reserved_gb} GB (allocator high-water mark)",
            "synthetic inputs: valid for memory/latency, not a task-success claim",
        ],
    )

    if args.save_actions:
        acts = collect_actions(pol, pre, post, args.device, args.dtype, args.n_fidelity)
        Path(args.save_actions).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_actions, acts)
        res.notes.append(f"saved {acts.shape} actions -> {args.save_actions}")

    if args.compare_actions:
        ref = np.load(args.compare_actions)
        acts = collect_actions(pol, pre, post, args.device, args.dtype, args.n_fidelity)
        res.fidelity = action_fidelity(ref, acts)

    del pol
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", required=True, choices=sorted(POLICY_CLASSES))
    ap.add_argument("--dtype", default="bf16", choices=[*DTYPES, "int8", "nf4"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--n-fidelity", type=int, default=8)
    ap.add_argument("--save-actions", default=None)
    ap.add_argument("--compare-actions", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    res = run(args)
    print(res.to_json())
    print(markdown_table([res]))
    print("saved ->", save(res, Path(args.out)))


if __name__ == "__main__":
    main()
