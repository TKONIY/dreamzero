#!/usr/bin/env python3
"""Profiling wrapper for DreamZero inference.

Parses server stdout to extract CUDA timing breakdown and generates a summary report.

Usage:
    # Terminal 1: Start server with output redirected
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
        --model-path ./checkpoints/dreamzero 2>&1 | tee server_log.txt

    # Terminal 2: Run test client with more chunks for profiling
    python test_client_AR.py --port 5000 --num-chunks 10 2>&1 | tee client_log.txt

    # After test completes, parse the logs:
    python profile_inference.py --server-log server_log.txt --client-log client_log.txt
"""

import argparse
import re
import sys
from dataclasses import dataclass, field


@dataclass
class InferenceStep:
    step: int = 0
    # From action_head (CUDA events, GPU time)
    total: float = 0.0
    text_encoder: float = 0.0
    image_encoder: float = 0.0
    vae: float = 0.0
    kv_cache: float = 0.0
    diffusion: float = 0.0
    dit_compute_steps: int = 0
    scheduler: float = 0.0
    # From sim_policy (wall clock)
    transform: float = 0.0
    model: float = 0.0
    untransform: float = 0.0
    # From server handler
    forward_time: float = 0.0
    wait_time: float = 0.0


def parse_server_log(log_path: str) -> list[InferenceStep]:
    """Parse server log to extract timing information."""
    steps = []
    current_step = None

    # Patterns from wan_flow_matching_action_tf.py line 1264
    action_head_pattern = re.compile(
        r"Time taken: Total ([\d.]+) seconds, "
        r"Text Encoder ([\d.]+) seconds, "
        r"Image Encoder ([\d.]+) seconds, "
        r"VAE ([\d.]+) seconds, "
        r"KV Cache Creation ([\d.]+) seconds, "
        r"Diffusion ([\d.]+) seconds, "
        r"DIT Compute Steps (\d+) steps, "
        r"Scheduler ([\d.]+) seconds"
    )

    # Pattern from sim_policy.py line 702
    sim_policy_pattern = re.compile(
        r"Inference Time: Total ([\d.]+) seconds, "
        r"Transform: ([\d.]+) seconds, "
        r"Model: ([\d.]+) seconds, "
        r"Untransform: ([\d.]+) seconds"
    )

    # Pattern from server handler
    forward_pattern = re.compile(r"Forward Time: ([\d.]+) seconds")
    wait_pattern = re.compile(r"Wait Time: ([\d.]+) seconds")

    with open(log_path) as f:
        for line in f:
            m = action_head_pattern.search(line)
            if m:
                current_step = InferenceStep(step=len(steps))
                current_step.total = float(m.group(1))
                current_step.text_encoder = float(m.group(2))
                current_step.image_encoder = float(m.group(3))
                current_step.vae = float(m.group(4))
                current_step.kv_cache = float(m.group(5))
                current_step.diffusion = float(m.group(6))
                current_step.dit_compute_steps = int(m.group(7))
                current_step.scheduler = float(m.group(8))
                continue

            m = sim_policy_pattern.search(line)
            if m and current_step is not None:
                current_step.transform = float(m.group(2))
                current_step.model = float(m.group(3))
                current_step.untransform = float(m.group(4))
                steps.append(current_step)
                current_step = None
                continue

            m = forward_pattern.search(line)
            if m and steps:
                steps[-1].forward_time = float(m.group(1))

            m = wait_pattern.search(line)
            if m:
                # wait_time is logged before inference, associate with next step
                if current_step is None:
                    current_step = InferenceStep(step=len(steps))
                current_step.wait_time = float(m.group(1))

    return steps


def parse_client_log(log_path: str) -> list[dict]:
    """Parse client log for per-step timing and action stats."""
    results = []
    pattern = re.compile(
        r"Action shape: \((\d+), (\d+)\), "
        r"range: \[([-\d.]+), ([-\d.]+)\], "
        r"time: ([\d.]+)s"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                results.append({
                    "shape": (int(m.group(1)), int(m.group(2))),
                    "min": float(m.group(3)),
                    "max": float(m.group(4)),
                    "client_time": float(m.group(5)),
                })
    return results


def print_report(steps: list[InferenceStep], client_results: list[dict] | None = None):
    """Print formatted performance breakdown report."""
    if not steps:
        print("No inference steps found in log.")
        return

    print("=" * 100)
    print("DreamZero Inference Performance Breakdown")
    print("=" * 100)

    # Per-step table
    header = (
        f"{'Step':>4} | {'Total':>7} | {'TextEnc':>7} | {'ImgEnc':>7} | {'VAE':>7} | "
        f"{'KV Cache':>8} | {'Diffus':>7} | {'DiTStps':>7} | {'Sched':>7} | "
        f"{'Xform':>7} | {'Unxform':>7}"
    )
    if client_results:
        header += f" | {'Client':>7}"
    print(header)
    print("-" * len(header))

    for i, s in enumerate(steps):
        row = (
            f"{s.step:>4} | {s.total:>6.2f}s | {s.text_encoder:>6.2f}s | "
            f"{s.image_encoder:>6.2f}s | {s.vae:>6.2f}s | {s.kv_cache:>7.2f}s | "
            f"{s.diffusion:>6.2f}s | {s.dit_compute_steps:>7} | {s.scheduler:>6.2f}s | "
            f"{s.transform:>6.3f}s | {s.untransform:>6.3f}s"
        )
        if client_results and i < len(client_results):
            row += f" | {client_results[i]['client_time']:>6.2f}s"
        print(row)

    # Summary statistics (skip first step as warmup)
    if len(steps) > 1:
        warm_steps = steps[1:]  # skip warmup
    else:
        warm_steps = steps

    n = len(warm_steps)
    avg = lambda attr: sum(getattr(s, attr) for s in warm_steps) / n

    print()
    print("=" * 100)
    print(f"Summary (steps {1 if len(steps) > 1 else 0}-{len(steps)-1}, "
          f"excluding step 0 warmup, N={n})")
    print("=" * 100)

    total_avg = avg("total")
    components = [
        ("Text Encoder", avg("text_encoder")),
        ("Image Encoder", avg("image_encoder")),
        ("VAE Encode", avg("vae")),
        ("KV Cache Creation", avg("kv_cache")),
        ("Diffusion (DiT)", avg("diffusion")),
        ("Scheduler/Other", avg("scheduler")),
    ]

    print(f"\n{'Component':<25} {'Avg Time':>10} {'% of Total':>12}")
    print("-" * 50)
    for name, t in components:
        pct = (t / total_avg * 100) if total_avg > 0 else 0
        print(f"{name:<25} {t:>9.3f}s {pct:>11.1f}%")
    print("-" * 50)
    print(f"{'Action Head Total':<25} {total_avg:>9.3f}s {'100.0':>11}%")

    print(f"\n{'Data Transform':<25} {avg('transform'):>9.3f}s")
    print(f"{'Action Untransform':<25} {avg('untransform'):>9.3f}s")
    print(f"{'E2E (transform+model+un)':<25} {avg('transform') + total_avg + avg('untransform'):>9.3f}s")

    avg_dit_steps = sum(s.dit_compute_steps for s in warm_steps) / n
    avg_diffusion = avg("diffusion")
    per_step = avg_diffusion / avg_dit_steps if avg_dit_steps > 0 else 0
    print(f"\nAvg DiT compute steps: {avg_dit_steps:.1f}")
    print(f"Avg time per DiT step: {per_step*1000:.1f}ms")

    if client_results and len(client_results) > 1:
        warm_client = client_results[1:]
        avg_client = sum(r["client_time"] for r in warm_client) / len(warm_client)
        print(f"\nAvg client-side E2E time: {avg_client:.3f}s (includes network)")

    # Optimization suggestions
    print("\n" + "=" * 100)
    print("Bottleneck Analysis")
    print("=" * 100)
    sorted_components = sorted(components, key=lambda x: x[1], reverse=True)
    print(f"\nTop bottleneck: {sorted_components[0][0]} ({sorted_components[0][1]:.3f}s, "
          f"{sorted_components[0][1]/total_avg*100:.1f}%)")
    if len(sorted_components) > 1:
        print(f"2nd bottleneck: {sorted_components[1][0]} ({sorted_components[1][1]:.3f}s, "
              f"{sorted_components[1][1]/total_avg*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Parse DreamZero inference logs and generate performance report")
    parser.add_argument("--server-log", required=True, help="Path to server log file")
    parser.add_argument("--client-log", default=None, help="Path to client log file (optional)")
    args = parser.parse_args()

    steps = parse_server_log(args.server_log)
    client_results = parse_client_log(args.client_log) if args.client_log else None
    print_report(steps, client_results)


if __name__ == "__main__":
    main()
