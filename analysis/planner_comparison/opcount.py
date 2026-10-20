"""FLOPs, op-level bytes and peak memory of one call, and measured GPU ceilings (P2c roofline).

Shared by scripts/planner_comparison/bench_lewam.py (lewam env) and bench_pg.py (swm env); needs only torch.
"""

import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import FlopCounterMode


class ByteCounter(TorchDispatchMode):
    """Bytes read and written per op: sum of its input and output tensor sizes, view ops skipped.

    No reuse between ops is assumed (per-op memory-access model, as in LLM-Viewer), so this overestimates DRAM
    traffic when tensors stay in L2 between ops.
    """

    def __init__(self):
        super().__init__()
        self.bytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if not func.is_view:
            tensors = [t for t in tree_flatten((args, kwargs, out))[0] if isinstance(t, torch.Tensor)]
            self.bytes += sum(t.numel() * t.element_size() for t in tensors)
        return out


def count_call(call):
    """Run call() once under the FLOP and byte counters; also its peak and working memory (MiB)."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with FlopCounterMode(display=False) as flops, ByteCounter() as traffic:
        call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return dict(gflop=flops.get_total_flops() / 1e9, gbyte=traffic.bytes / 1e9,
                peak_mib=peak / 2**20, working_mib=(peak - before) / 2**20)


def time_calls(call, budget_s=30.0, max_calls=20, min_calls=3, label=None):
    """GPU ms of repeated calls (CUDA events), after warmup; fewer calls when one call is slow.

    label: NVTX range name around each timed call (for the nsys pass).
    """
    first = time.perf_counter()
    call()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - first
    for _ in range(0 if first_s > 2.0 else 2):  # warmup: the first call plus up to two more
        call()
    n_calls = int(min(max_calls, max(min_calls, budget_s // max(first_s, 1e-3))))
    times = []
    for _ in range(n_calls):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if label:
            torch.cuda.nvtx.range_push(label)
        start.record()
        call()
        end.record()
        if label:
            torch.cuda.nvtx.range_pop()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def measure_ceilings(size=8192, copy_gib=1.0, l2_copy_mib=32,l2_copies=100, repeats=10):
    """Achievable FP32 matmul TFLOP/s (TF32 as configured), DRAM copy GB/s and L2-resident copy GB/s on this GPU.

    The L2 copy (source + destination well under the 96 MiB L2) is repeated l2_copies times per timed call.
    """
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    gemm_ms = min(time_calls(lambda: a @ b, max_calls=repeats, min_calls=repeats))
    src = torch.empty(int(copy_gib * 2**30) // 4, device='cuda')
    dst = torch.empty_like(src)
    copy_ms = min(time_calls(lambda: dst.copy_(src), max_calls=repeats, min_calls=repeats))
    l2_src = torch.empty(l2_copy_mib * 2**20 // 4, device='cuda')
    l2_dst = torch.empty_like(l2_src)

    def copy_in_l2():
        for _ in range(l2_copies):
            l2_dst.copy_(l2_src)

    l2_ms = min(time_calls(copy_in_l2, max_calls=repeats, min_calls=repeats)) / l2_copies
    return dict(gemm_tflops=2 * size**3 / gemm_ms / 1e9, copy_gbps=2 * src.numel() * 4 / copy_ms / 1e6,
                l2_copy_gbps=2 * l2_src.numel() * 4 / l2_ms / 1e6, tf32_matmul=torch.backends.cuda.matmul.allow_tf32)


def is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error)
