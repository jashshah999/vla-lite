"""Measurement harness for VLA policies: peak VRAM, latency/control-rate, action fidelity.

Every claim this project publishes has to come from here, so the numbers are
produced the same way for every configuration (fp32 / bf16 / int8 / int4 / cpu).

Run with the project env and PYTHONPATH cleared (ROS leaks numpy otherwise):
  PYTHONPATH= ~/miniconda3/envs/vla/bin/python bench/measure.py --help
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# environment capture — a benchmark without the exact rig is not reproducible
# --------------------------------------------------------------------------- #

def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": _cpu_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env |= {
            "gpu": props.name,
            "gpu_total_gb": round(props.total_memory / 1e9, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "driver": _nvidia_smi("driver_version"),
        }
    return env


def _cpu_count() -> int:
    try:
        return len(__import__("os").sched_getaffinity(0))
    except Exception:
        return __import__("os").cpu_count() or 0


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    """One measured configuration."""

    label: str
    model: str
    dtype: str
    device: str
    load_seconds: float | None = None
    weights_gb: float | None = None
    peak_vram_gb: float | None = None
    peak_host_rss_gb: float | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    control_hz: float | None = None
    fidelity: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=environment)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, default=str)


# --------------------------------------------------------------------------- #
# memory + timing primitives
# --------------------------------------------------------------------------- #

class VramTracker:
    """Peak *device* memory across a block, measured from a clean baseline.

    Uses max_memory_allocated (tensor bytes) and also reports the reserved
    figure, because the caching allocator's reserved number is what actually
    has to fit on a smaller card.
    """

    def __init__(self, device: str):
        self.device = device
        self.enabled = device.startswith("cuda") and torch.cuda.is_available()
        self.peak_allocated_gb: float | None = None
        self.peak_reserved_gb: float | None = None

    def __enter__(self) -> "VramTracker":
        if self.enabled:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.enabled:
            torch.cuda.synchronize()
            self.peak_allocated_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            self.peak_reserved_gb = round(torch.cuda.max_memory_reserved() / 1e9, 3)


def host_rss_gb() -> float | None:
    """Resident set size — the number that matters for a CPU-only claim."""
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1e9, 3)
    except Exception:
        return None


def time_calls(fn, warmup: int, iters: int, device: str) -> dict[str, float]:
    """Latency distribution. Reports p50/p90/p99, not just the mean.

    A robot controller lives or dies on the tail: a policy that averages 20 ms
    but spikes to 200 ms drops frames at a fixed control rate, so the mean
    alone would be a misleading number to publish.
    """
    sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") and torch.cuda.is_available() else (lambda: None)

    for _ in range(warmup):
        fn()
    sync()

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    def pct(p: float) -> float:
        idx = min(len(samples) - 1, int(round(p / 100 * (len(samples) - 1))))
        return round(samples[idx], 2)

    return {
        "mean": round(statistics.fmean(samples), 2),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "min": round(samples[0], 2),
        "max": round(samples[-1], 2),
        "iters": float(iters),
    }


# --------------------------------------------------------------------------- #
# fidelity — the claim that actually matters for a robot policy
# --------------------------------------------------------------------------- #

def action_fidelity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compare a quantized policy's actions against the full-precision teacher.

    Both arrays are (n_samples, horizon, action_dim) in *unnormalized* action
    units. Absolute error is reported in those units and also relative to the
    reference's own spread, since an L1 of 0.01 means nothing without knowing
    whether the action range is 0.1 or 100.
    """
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")

    err = np.abs(reference - candidate)
    spread = reference.std(axis=(0, 1)) + 1e-8
    rel = (err / spread).mean()

    # Cosine similarity per action vector: catches direction flips that a
    # small L1 can hide.
    a = reference.reshape(-1, reference.shape[-1])
    b = candidate.reshape(-1, candidate.shape[-1])
    denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-8
    cos = float(np.mean((a * b).sum(axis=1) / denom))

    return {
        "l1_mean": float(err.mean()),
        "l1_p99": float(np.percentile(err, 99)),
        "l1_max": float(err.max()),
        "rmse": float(np.sqrt(((reference - candidate) ** 2).mean())),
        "rel_err_vs_action_std": float(rel),
        "cosine_sim": cos,
        "n_samples": int(reference.shape[0]),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def save(result: Result, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.label}.json"
    path.write_text(result.to_json())
    return path


def markdown_table(results: list[Result]) -> str:
    head = (
        "| config | device | dtype | peak VRAM | host RSS | p50 ms | p99 ms | Hz | "
        "action L1 vs fp | rel err | cos |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        def g(d: dict[str, float], k: str) -> str:
            v = d.get(k)
            return "—" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))

        rows.append(
            f"| {r.label} | {r.device} | {r.dtype} | "
            f"{'—' if r.peak_vram_gb is None else f'{r.peak_vram_gb:.2f} GB'} | "
            f"{'—' if r.peak_host_rss_gb is None else f'{r.peak_host_rss_gb:.2f} GB'} | "
            f"{g(r.latency_ms, 'p50')} | {g(r.latency_ms, 'p99')} | "
            f"{'—' if r.control_hz is None else f'{r.control_hz:.1f}'} | "
            f"{g(r.fidelity, 'l1_mean')} | {g(r.fidelity, 'rel_err_vs_action_std')} | "
            f"{g(r.fidelity, 'cosine_sim')} |"
        )
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="self-check the harness")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print(json.dumps(environment(), indent=1))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        with VramTracker(dev) as vt:
            x = torch.randn(2048, 2048, device=dev)
            for _ in range(3):
                x = x @ x.T / 2048.0
        lat = time_calls(lambda: torch.randn(512, 512, device=dev) @ torch.randn(512, 512, device=dev), 3, 20, dev)
        ref = np.random.randn(8, 10, 7)
        fid = action_fidelity(ref, ref + np.random.randn(*ref.shape) * 0.01)
        r = Result(
            label="selftest", model="matmul", dtype="fp32", device=dev,
            peak_vram_gb=vt.peak_allocated_gb, peak_host_rss_gb=host_rss_gb(),
            latency_ms=lat, fidelity=fid,
            notes=["synthetic self-test of the harness, not a model measurement"],
        )
        print(r.to_json())
        print(markdown_table([r]))


if __name__ == "__main__":
    main()
