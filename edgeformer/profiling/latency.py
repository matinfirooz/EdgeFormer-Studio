"""Synchronized forward-only timings with mode and RNG restoration."""

import math
import statistics
import time
import torch


def benchmark_latency(model, sample, warmup=10, runs=50, *, return_stats=False):
    if runs < 1 or warmup < 0 or sample.shape[0] < 1:
        raise ValueError("runs and batch size must be positive; warmup must be nonnegative")
    modes = {m: m.training for m in model.modules()}
    devices = [sample.device.index or 0] if sample.is_cuda else []
    timings = []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            for _ in range(warmup):
                model(sample)
            for _ in range(runs):
                if sample.is_cuda:
                    torch.cuda.synchronize(sample.device)
                start = time.perf_counter_ns()
                model(sample)
                if sample.is_cuda:
                    torch.cuda.synchronize(sample.device)
                timings.append((time.perf_counter_ns() - start) / 1e6)
    finally:
        for m, mode in modes.items():
            m.training = mode
    ordered = sorted(timings)
    mean = statistics.mean(timings)
    result = dict(
        mean_ms=mean,
        median_ms=statistics.median(timings),
        p95_ms=ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        std_ms=statistics.pstdev(timings),
        min_ms=min(timings),
        batch_size=sample.shape[0],
        runs=runs,
        warmup=warmup,
        images_per_second=1000 * sample.shape[0] / mean,
        scope="forward only; excludes loading, preprocessing and host/device transfers",
    )
    return result if return_stats else mean
