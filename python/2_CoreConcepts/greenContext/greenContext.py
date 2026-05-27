# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Green Context Sample using CUDA Core API.

Three scenarios are timed with CUDA events and compared:

  1. Reference: the critical kernel alone on the primary context,
     with no competing work. Establishes the pure compute time of
     the critical kernel with access to every SM on the device.
  2. Baseline: both kernels run on the device's primary context,
     on two non-blocking streams. They contend for all SMs.
  3. Green contexts: SMs are split into two disjoint groups; each
     kernel runs on a stream belonging to its own green context.

The headline metric is the total wall time of the critical kernel
from launch to completion on its stream. In the baseline it is
dominated by waiting behind the long-running kernel; with green
contexts it reflects only the kernel's own compute time on a
smaller SM partition. The reference row separates those effects.

Note: Parallel execution on the GPU is never guaranteed. Green
contexts remove one common source of contention (shared SMs) but
they are not a hard scheduling promise.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from cuda.core import (
    ContextOptions,
    Device,
    EventOptions,
    LaunchConfig,
    Program,
    ProgramOptions,
    SMResourceOptions,
    launch,
)

# Two CUDA kernels:
# 1. The delay kernel spins until `cycles` SM clock ticks have elapsed.
# 2. The critical kernel does a small amount of useful work.

KERNEL_SRC = r"""
extern "C" __global__ void delay_kernel(unsigned long long cycles)
{
    unsigned long long start = clock64();
    while ((unsigned long long)(clock64() - start) < cycles) { }
}

extern "C" __global__ void critical_kernel(float *out, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Two dependent accumulators so the compiler cannot collapse the
        // loop into a closed-form expression. `iters` is a runtime argument
        // for the same reason.
        float v = (float)i * 1e-6f + 1.0f;
        float u = (float)i * 1e-7f + 0.5f;
        for (int k = 0; k < iters; ++k) {
            v = v * 1.000001f + u;
            u = u * 0.999999f + v * 1e-7f;
        }
        out[i] = v + u;
    }
}
"""


@dataclass
class ScenarioResult:
    name: str
    critical_total_ms: float
    critical_sm_count: int
    long_ms: Optional[float] = None
    critical_offset_ms: Optional[float] = None
    long_sm_count: Optional[int] = None


def print_sm_topology(device: Device) -> None:
    sm = device.resources.sm
    print("[Green Context Sample using CUDA Core API]")
    print(f"Device: {device.name}")
    print(f"Compute Capability: sm_{device.arch}")
    print(f"Total SMs:                 {sm.sm_count}")
    print(f"Min. SM partition size:    {sm.min_partition_size}")
    print(f"SM co-scheduled alignment: {sm.coscheduled_alignment}")


def _align_down(n: int, k: int) -> int:
    if k <= 0:
        return n
    return (n // k) * k


def _driver_accepts_split(sm, long_count: int, critical_count: int) -> bool:
    if long_count <= 0 or critical_count <= 0:
        return False
    try:
        groups, _ = sm.split(
            SMResourceOptions(count=(long_count, critical_count)),
            dry_run=True,
        )
    except Exception:
        return False
    actual = tuple(g.sm_count for g in groups)
    return actual == (long_count, critical_count)


def _find_working_split(
    sm, prefer_critical: Optional[int] = None
) -> Optional[Tuple[int, int, int]]:
    """
    Probe the driver for a (long, critical) split it actually accepts.

    Escalates the alignment granularity from `min_partition_size` upward in
    powers of two, requiring BOTH sides to be multiples of the current
    granularity. This handles architectures where the driver's true
    allocation granularity is larger than the reported
    `min_partition_size` (e.g. TPC/GPC-pair alignment on Blackwell: on a
    188-SM part `min_partition_size` is 8 but the driver actually requires
    each side to be a multiple of 16).

    Returns (long_count, critical_count, granularity) or None. The
    granularity is the smallest power-of-two multiple of
    `min_partition_size` at which both sides are aligned and the driver
    accepts the pair.
    """
    total = sm.sm_count
    min_part = sm.min_partition_size
    if min_part <= 0:
        return None

    if prefer_critical is None or prefer_critical <= 0:
        prefer_critical = max(min_part, min(16, total // 8))

    # Escalate granularity in powers of two. The upper bound is half of
    # `total` because below that we cannot fit two partitions of size
    # >= granularity.
    granularity = min_part
    while granularity * 2 <= total:
        base = max(granularity, _align_down(prefer_critical, granularity))

        candidates: List[int] = []
        seen = set()

        def push(c: int) -> None:
            if c >= granularity and c <= total - granularity and c not in seen:
                seen.add(c)
                candidates.append(c)

        # Walk outward from `base` (the preferred critical size, aligned
        # down to the current granularity) in steps of granularity.
        push(base)
        max_steps = max(total // granularity, 1)
        for step in range(1, max_steps + 1):
            push(base + step * granularity)
            push(base - step * granularity)

        for critical in candidates:
            long_count = _align_down(total - critical, granularity)
            if long_count < granularity:
                continue
            if _driver_accepts_split(sm, long_count, critical):
                return long_count, critical, granularity

        granularity *= 2

    return None


def _format_suggestion(sm, prefer_critical: Optional[int]) -> Optional[str]:
    """
    Return a '--split A,B' string the driver is known to accept, or None
    if we couldn't find one.
    """
    found = _find_working_split(sm, prefer_critical=prefer_critical)
    if found is None:
        return None
    long_count, critical_count, _granularity = found
    return f"--split {long_count},{critical_count}"


def parse_split(arg: Optional[str], device: Device) -> Tuple[int, int]:
    """
    Parse the --split "A,B" CLI argument and validate it against the device.

    Returns (long_count, critical_count).
    """
    sm = device.resources.sm
    total = sm.sm_count
    min_part = sm.min_partition_size

    if arg is None:
        # Auto: reserve a small aligned slice for the critical kernel and
        # hand the rest (also aligned) to the long-running kernel. We
        # can't trust `min_partition_size` alone: on some GPUs (e.g.
        # 188-SM Blackwell) the driver requires stricter alignment than
        # it reports. Escalate the granularity until the driver accepts
        # a pair.
        prefer_critical = max(min_part, min(16, total // 8))
        found = _find_working_split(sm, prefer_critical=prefer_critical)
        if found is None:
            print(
                "Error: could not find an SM split that the driver accepts "
                f"on this device (total SMs={total}, "
                f"min_partition_size={min_part})."
            )
            print(
                "       The driver enforces architecture-specific alignment "
                "rules beyond min_partition_size; try passing an explicit "
                "--split."
            )
            sys.exit(1)
        long_count, critical_count, granularity = found
        if granularity > min_part:
            print(
                f"Note: driver required stricter alignment than "
                f"min_partition_size={min_part}; selected split uses "
                f"granularity={granularity} SMs."
            )
        return long_count, critical_count

    # User-provided split.
    try:
        parts = [int(x.strip()) for x in arg.split(",")]
    except ValueError:
        print(f"Error: --split must look like 'A,B', got: {arg!r}")
        sys.exit(1)
    if len(parts) != 2:
        print(
            "Error: --split must contain exactly two comma-separated "
            f"integers, got: {arg!r}"
        )
        sys.exit(1)
    long_count, critical_count = parts

    errors = []
    if long_count <= 0 or critical_count <= 0:
        errors.append("both partition sizes must be positive")
    if long_count % min_part != 0 or critical_count % min_part != 0:
        errors.append(f"each size must be a multiple of min_partition_size={min_part}")
    if long_count + critical_count > total:
        errors.append(
            f"sum {long_count + critical_count} exceeds device total of {total} SMs"
        )

    if errors:
        print("Error: invalid --split value:")
        for e in errors:
            print(f"  - {e}")
        suggestion = _format_suggestion(
            sm, prefer_critical=critical_count if critical_count > 0 else None
        )
        if suggestion is not None:
            print(f"Tip: a driver-accepted split on this device is {suggestion}")
        sys.exit(1)

    # Confirm the driver itself accepts the split. The well-known alignment
    # checks above are necessary but not sufficient on every architecture.
    try:
        groups, _ = sm.split(
            SMResourceOptions(count=(long_count, critical_count)),
            dry_run=True,
        )
    except Exception as e:
        print(f"Error: driver rejected the requested split: {e}")
        print(
            "       The sample's own alignment checks are not exhaustive on "
            "every architecture; the driver enforces additional hardware "
            "constraints (for example TPC/partition-grid alignment)."
        )
        suggestion = _format_suggestion(sm, prefer_critical=critical_count)
        if suggestion is not None:
            print(f"Tip: a driver-accepted split on this device is {suggestion}")
        sys.exit(1)

    actual = tuple(g.sm_count for g in groups)
    if actual != (long_count, critical_count):
        print(f"Error: driver adjusted the requested split to {actual}.")
        suggestion = _format_suggestion(sm, prefer_critical=critical_count)
        if suggestion is not None:
            print(f"Tip: a driver-accepted split on this device is {suggestion}")
        else:
            print("       Pick a different --split, or omit it for the auto default.")
        sys.exit(1)

    return long_count, critical_count


def compile_kernels(device: Device):
    options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
    program = Program(KERNEL_SRC, code_type="c++", options=options)
    module = program.compile(
        "cubin",
        name_expressions=("delay_kernel", "critical_kernel"),
    )
    return module.get_kernel("delay_kernel"), module.get_kernel("critical_kernel")


def microseconds_to_cycles(device: Device, microseconds: float) -> int:
    """
    Convert microseconds to SM clock cycles, using the reported GPU clock rate.
    clock_rate is in kHz, so 1 us = clock_rate_kHz / 1000 cycles.
    """
    clock_khz = device.properties.clock_rate
    return int(microseconds * clock_khz / 1000.0)


def _run_one(
    device: Device,
    name: str,
    long_stream,
    critical_stream,
    long_sm_count: int,
    critical_sm_count: int,
    delay_kernel,
    critical_kernel,
    delay_cycles: int,
    delay_blocks: int,
    critical_out_ptr: int,
    critical_n: int,
    critical_iters: int,
    launch_gap_s: float,
) -> ScenarioResult:
    """
    Launch the delay kernel on `long_stream`, wait `launch_gap_s` on the host,
    launch the critical kernel on `critical_stream`, and time both with events.
    """

    # Create events with timing enabled.
    opts = EventOptions(timing_enabled=True)
    e_long_start = device.create_event(opts)
    e_long_end = device.create_event(opts)
    e_crit_start = device.create_event(opts)
    e_crit_end = device.create_event(opts)

    # 1024 threads/block ensures at most one delay block is resident per SM
    # on current architectures, so grid size directly controls the number of
    # waves: delay_blocks / sm_count_visible_to_stream.
    delay_block = 1024
    delay_cfg = LaunchConfig(grid=delay_blocks, block=delay_block)
    critical_block = 256
    critical_grid = (critical_n + critical_block - 1) // critical_block
    critical_cfg = LaunchConfig(grid=critical_grid, block=critical_block)

    # Start of timed region
    long_stream.record(e_long_start)
    launch(long_stream, delay_cfg, delay_kernel, np.uint64(delay_cycles))
    long_stream.record(e_long_end)

    time.sleep(launch_gap_s)

    critical_stream.record(e_crit_start)
    launch(
        critical_stream,
        critical_cfg,
        critical_kernel,
        critical_out_ptr,
        np.int32(critical_n),
        np.int32(critical_iters),
    )
    critical_stream.record(e_crit_end)

    # Sync both streams so every event has completed and is measurable.
    long_stream.sync()
    critical_stream.sync()
    # End of timed region

    return ScenarioResult(
        name=name,
        long_ms=e_long_end - e_long_start,
        critical_total_ms=e_crit_end - e_crit_start,
        critical_offset_ms=e_crit_start - e_long_start,
        long_sm_count=long_sm_count,
        critical_sm_count=critical_sm_count,
    )


def run_critical_alone(
    device: Device,
    critical_kernel,
    critical_n: int,
    critical_iters: int,
) -> ScenarioResult:
    """
    Critical kernel alone on the primary context, no competing work.
    Establishes the pure compute time with every SM on the device available.
    """
    stream = device.create_stream()
    out = device.allocate(critical_n * 4)
    total_sm = device.resources.sm.sm_count
    try:
        opts = EventOptions(timing_enabled=True)
        e_start = device.create_event(opts)
        e_end = device.create_event(opts)
        block = 256
        grid = (critical_n + block - 1) // block
        cfg = LaunchConfig(grid=grid, block=block)

        stream.record(e_start)
        launch(
            stream,
            cfg,
            critical_kernel,
            int(out.handle),
            np.int32(critical_n),
            np.int32(critical_iters),
        )
        stream.record(e_end)
        stream.sync()

        return ScenarioResult(
            name="crit alone (primary ctx)",
            critical_total_ms=e_end - e_start,
            critical_sm_count=total_sm,
        )
    finally:
        out.close()


def run_baseline(
    device: Device,
    delay_kernel,
    critical_kernel,
    delay_cycles: int,
    delay_blocks: int,
    critical_n: int,
    critical_iters: int,
    launch_gap_s: float,
) -> ScenarioResult:
    """Both kernels on the primary context, two non-blocking streams."""
    long_stream = device.create_stream()
    critical_stream = device.create_stream()
    out = device.allocate(critical_n * 4)
    total_sm = device.resources.sm.sm_count
    try:
        return _run_one(
            device,
            name="baseline (primary ctx)",
            long_stream=long_stream,
            critical_stream=critical_stream,
            long_sm_count=total_sm,
            critical_sm_count=total_sm,
            delay_kernel=delay_kernel,
            critical_kernel=critical_kernel,
            delay_cycles=delay_cycles,
            delay_blocks=delay_blocks,
            critical_out_ptr=int(out.handle),
            critical_n=critical_n,
            critical_iters=critical_iters,
            launch_gap_s=launch_gap_s,
        )
    finally:
        out.close()


def run_green_context(
    device: Device,
    split: Tuple[int, int],
    delay_kernel,
    critical_kernel,
    delay_cycles: int,
    delay_blocks: int,
    critical_n: int,
    critical_iters: int,
    launch_gap_s: float,
) -> ScenarioResult:
    """Each kernel on its own green context, with disjoint SM partitions."""
    long_count, critical_count = split
    sm = device.resources.sm
    groups, _remainder = sm.split(SMResourceOptions(count=(long_count, critical_count)))
    assert len(groups) == 2
    long_group, critical_group = groups

    # Create the large ctx last so it's closed first: order matters only for
    # ensuring we never try to close a ctx that's currently the thread's
    # active ctx.
    ctx_long = device.create_context(ContextOptions(resources=[long_group]))
    ctx_crit = None
    out = None
    try:
        ctx_crit = device.create_context(ContextOptions(resources=[critical_group]))

        long_stream = ctx_long.create_stream()
        critical_stream = ctx_crit.create_stream()
        out = device.allocate(critical_n * 4)

        return _run_one(
            device,
            name=f"green ctx ({long_count}+{critical_count} SMs)",
            long_stream=long_stream,
            critical_stream=critical_stream,
            long_sm_count=ctx_long.resources.sm.sm_count,
            critical_sm_count=ctx_crit.resources.sm.sm_count,
            delay_kernel=delay_kernel,
            critical_kernel=critical_kernel,
            delay_cycles=delay_cycles,
            delay_blocks=delay_blocks,
            critical_out_ptr=int(out.handle),
            critical_n=critical_n,
            critical_iters=critical_iters,
            launch_gap_s=launch_gap_s,
        )
    finally:
        if out is not None:
            out.close()
        # Streams must be released before their owning ctx; letting them go out
        # of scope here is sufficient since no references escape this frame.
        if ctx_crit is not None:
            ctx_crit.close()
        ctx_long.close()


def _fmt_ms(value: Optional[float], width: int) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.3f}"


def print_results(results: List[ScenarioResult]) -> None:
    print()
    header = (
        f"{'scenario':<32}{'SMs (long/crit)':>20}"
        f"{'long (ms)':>14}{'crit total (ms)':>18}{'crit offset (ms)':>19}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        long_sm = "-" if r.long_sm_count is None else str(r.long_sm_count)
        sms = f"{long_sm}/{r.critical_sm_count}"
        print(
            f"{r.name:<32}{sms:>20}"
            f"{_fmt_ms(r.long_ms, 14)}{_fmt_ms(r.critical_total_ms, 18)}"
            f"{_fmt_ms(r.critical_offset_ms, 19)}"
        )
    print()
    print("long (ms)        : wall time of the delay kernel")
    print("crit total (ms)  : launch-to-complete wall time of the critical kernel")
    print(
        "crit offset (ms) : when the critical stream started, relative to the"
        " long stream start"
    )


def report_speedup(
    alone: ScenarioResult,
    baseline: ScenarioResult,
    green: ScenarioResult,
) -> None:
    """
    Print three headline numbers that put the raw scenario timings in context:
    """
    if baseline.critical_total_ms <= 0 or alone.critical_total_ms <= 0:
        return
    latency_speedup = baseline.critical_total_ms / max(green.critical_total_ms, 1e-9)
    compute_cost = green.critical_total_ms / alone.critical_total_ms
    wait_ms = max(0.0, baseline.critical_total_ms - alone.critical_total_ms)
    print()
    print(
        f"Critical-kernel latency speedup (baseline vs green ctx): "
        f"{latency_speedup:.2f}x"
    )
    print(
        f"Green-ctx compute cost vs unconstrained (crit alone):    {compute_cost:.2f}x"
    )
    print(f"Baseline time spent waiting for SMs (not computing):     ~{wait_ms:.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Green Context sample using CUDA Core API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="SM split as 'LONG,CRITICAL', e.g. '112,16'. Default: auto.",
    )
    parser.add_argument(
        "--delay-us",
        type=int,
        default=2000,
        help=(
            "Per-block busy-wait duration of the delay kernel, "
            "in microseconds (default: 2000)"
        ),
    )
    parser.add_argument(
        "--delay-waves",
        type=int,
        default=16,
        help=(
            "Number of waves of the delay kernel on the long partition. "
            "Drives the default --delay-blocks (default: 16)."
        ),
    )
    parser.add_argument(
        "--delay-blocks",
        type=int,
        default=None,
        help=(
            "Number of blocks launched for the delay kernel. "
            "Overrides --delay-waves if set. "
            "Default: --delay-waves * device SM count."
        ),
    )
    parser.add_argument(
        "--critical-n",
        type=int,
        default=1 << 22,
        help="Work size of the critical kernel (default: 4194304)",
    )
    parser.add_argument(
        "--critical-iters",
        type=int,
        default=1024,
        help=(
            "Iterations of the inner math loop inside the critical kernel. "
            "Higher values make the critical kernel's compute time more "
            "substantial (default: 1024)."
        ),
    )
    parser.add_argument(
        "--launch-gap-ms",
        type=float,
        default=1.0,
        help=(
            "Host delay between launching the long and critical kernels, "
            "in ms (default: 1.0)"
        ),
    )
    args = parser.parse_args()

    try:
        device = Device(args.device)
        device.set_current()
    except Exception as e:
        print(f"Error: failed to initialize CUDA device {args.device}: {e}")
        return 1

    print_sm_topology(device)

    long_count, critical_count = parse_split(args.split, device)
    print(f"SM split (long/critical):  {long_count} / {critical_count}")

    sm_count = device.resources.sm.sm_count
    delay_blocks = args.delay_blocks or args.delay_waves * sm_count
    delay_cycles = microseconds_to_cycles(device, args.delay_us)
    launch_gap_s = max(0.0, args.launch_gap_ms / 1000.0)

    # Rough estimate of the long kernel's duration on the full device. Mostly
    # informational; the real value is reported after the run.
    est_long_ms = (delay_blocks / sm_count) * (args.delay_us / 1000.0)

    print("Workload parameters:")
    print(
        f"  delay kernel: {delay_blocks} blocks, {args.delay_us} us/block "
        f"(~{est_long_ms:.1f} ms on {sm_count} SMs)"
    )
    print(
        f"  critical kernel: {args.critical_n} elements, "
        f"{args.critical_iters} inner iterations"
    )
    print(f"  host launch gap: {args.launch_gap_ms} ms")

    print()
    print("Compiling kernels ...")
    delay_k, crit_k = compile_kernels(device)

    try:
        print("Running reference scenario (critical kernel alone) ...")
        alone = run_critical_alone(
            device,
            crit_k,
            args.critical_n,
            args.critical_iters,
        )

        print("Running baseline scenario (primary context) ...")
        baseline = run_baseline(
            device,
            delay_k,
            crit_k,
            delay_cycles,
            delay_blocks,
            args.critical_n,
            args.critical_iters,
            launch_gap_s,
        )

        print("Running green context scenario ...")
        green = run_green_context(
            device,
            (long_count, critical_count),
            delay_k,
            crit_k,
            delay_cycles,
            delay_blocks,
            args.critical_n,
            args.critical_iters,
            launch_gap_s,
        )
    except Exception as e:
        print(f"Error: scenario failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print_results([alone, baseline, green])
    report_speedup(alone, baseline, green)

    print("\nDone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
