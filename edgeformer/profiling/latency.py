import time
import torch


def benchmark_latency(model, sample, warmup=10, runs=50):
    model.eval()
    device = sample.device
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / runs
