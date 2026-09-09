# simpleAtomicIntrinsics - Atomic vs. Non-Atomic Operations

## Description

A CUDA sample that demonstrates **why atomic operations are needed** by running the same work with and without atomics under heavy contention. 1,000,000 threads write into an array of just 10 integers, so roughly 100,000 threads collide on every element. Three operations — add, max, and compare-and-swap — are each shown four ways: a non-atomic kernel, the CUDA atomic intrinsic, `cuda::std::atomic_ref` from CCCL (the `std::atomic` API in device code), and `cuda::atomic_ref` from CCCL (the CUDA-specific variant that natively supports thread scopes):

Both CCCL variants use `atomic_ref` rather than `atomic` so they act on the *existing* plain-`int` array — this matches the behavior of the intrinsic atomics, which also adds atomic access to memory that already exists. `atomic<int>` would instead require the array elements themselves to be declared separately as the atomic type.

| Operation | Non-atomic | Intrinsic | cuda::std::atomic_ref | cuda::atomic_ref |
|---|---|---|---|---|
| Add 1 to an element | `increment` | `increment_atomic` | `increment_atomic_std` | `increment_atomic_cuda` |
| Keep the maximum value | `max` | `max_atomic` | `max_atomic_std` | `max_atomic_cuda` |
| Increment via compare-and-swap | `cas` | `cas_atomic` | `cas_atomic_std` | `cas_atomic_cuda` |

The non-atomic kernels perform the read-modify-write as separate steps, so concurrent threads interleave and lose updates. The atomic kernels perform it as one indivisible hardware operation and produce the exact expected result every time.

## What You'll Learn

- What a race condition looks like: the non-atomic results are dramatically (add, CAS) or subtly (max) wrong
- Using the atomic intrinsics `atomicAdd`, `atomicMax`, and `atomicCAS` on global memory
- Building an atomic operation from an `atomicCAS` retry loop — the pattern that can implement any read-modify-write atomically
- Using `cuda::std::atomic_ref` (libcu++/CCCL) to write the same atomics with the standard `std::atomic` API in device code
- Using `cuda::atomic_ref` (CCCL) — the CUDA-specific variant that shares the same API as `cuda::std::atomic_ref` but natively supports CUDA thread scopes
- Why dedicated intrinsics beat CAS loops under contention (compare the `cas_atomic` timing against `increment_atomic`)
- Timing GPU work with CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`)
- Resetting device buffers between runs with `cudaMemset`

## Key Concepts

- **Race condition** — a plain `g[i] = g[i] + 1` is three steps (read, modify, write); two threads can read the same old value and one increment is lost
- **Atomic read-modify-write** — the hardware serializes atomic updates to the same address (performed at the L2 cache), so no update is lost
- **Compare-and-swap (CAS)** — `atomicCAS(addr, expected, desired)` swaps only if the current value equals `expected` and returns the value it found; looping until the swap succeeds makes any operation atomic
- **Contention cost** — atomics are correct but serialize colliding threads; the CAS retry loop shows this at its most extreme

## Key APIs

### CUDA Device Intrinsics
- `atomicAdd` — atomically add a value to a memory location
- `atomicMax` — atomically store the maximum of the current and a proposed value
- `atomicCAS` — atomically compare-and-swap; returns the previous value

### libcu++ (CCCL)
- `cuda::std::atomic_ref<int>` — wraps plain memory with the standard `std::atomic` interface, usable in device code; header: `<cuda/std/atomic>`
- `cuda::atomic_ref<int>` — CUDA-specific version of `std::atomic_ref`; same API but natively supports CUDA thread scopes (e.g. `cuda::thread_scope_device`); header: `<cuda/atomic>`
- `fetch_add` — atomic add, `std::atomic` style
- `load` / `compare_exchange_weak` — the standard CAS retry loop; used to build max (which `std::atomic` lacks) and the CAS increment

### CUDA Runtime
- `cudaMalloc` / `cudaFree` — allocate and release device memory
- `cudaMemset` — fill device memory with a byte value (zero the array between runs)
- `cudaMemcpy` — copy results back to the host
- `cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime` / `cudaEventDestroy` — GPU-timeline timing of each kernel

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5 or higher

### Software
- CUDA Toolkit
- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

```bash
./simpleAtomicIntrinsics
```

No command-line arguments are required. The sample always runs on device 0.

## Expected Output

```text
=== Atomic vs. non-atomic operations (intrinsics, cuda::std::atomic_ref, cuda::atomic_ref) ===

GPU Device 0: with compute capability X.Y and Number of SMs <smCount>

1000000 total threads in 1000 blocks writing into 10 array elements

[add] expected: every element = 100000
non-atomic                             (0.19 ms): { 13 13 13 13 13 13 13 13 13 13 }
atomicAdd()                            (0.19 ms): { 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 }
cuda::std::atomic_ref::fetch_add()     (0.26 ms): { 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 }
cuda::atomic_ref::fetch_add()          (0.26 ms): { 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 }

[max] expected: element i = 999990 + i
non-atomic                                         (0.013 ms): { 998070 998071 998072 998073 998064 998065 998066 998067 998068 998069 }
atomicMax()                                        (0.18 ms):  { 999990 999991 999992 999993 999994 999995 999996 999997 999998 999999 }
cuda::std::atomic_ref::compare_exchange_weak()     (1.0 ms):   { 999990 999991 999992 999993 999994 999995 999996 999997 999998 999999 }
cuda::atomic_ref::compare_exchange_weak()          (1.0 ms):   { 999990 999991 999992 999993 999994 999995 999996 999997 999998 999999 }

[CAS] expected: every element = 100000
non-atomic                                         (0.014 ms): { 13 13 13 13 13 13 13 13 13 13 }
atomicCAS()                                        (1716 ms):  { 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 }
cuda::std::atomic_ref::compare_exchange_weak()     (9934 ms):  { 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 }
cuda::atomic_ref::compare_exchange_weak()          (9934 ms):  { 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 }

The non-atomic kernels lose updates when threads race on the same
element; the atomic kernels match the expected values exactly.
```

**Reading the numbers:**
- The non-atomic values vary from run to run — that nondeterminism is the race condition itself
- `max` is the sneakiest failure: 998070 looks plausible next to the correct 999990
- `cas_atomic` is orders of magnitude slower than `increment_atomic` for the same result: with ~100,000 threads contending per element, almost every CAS attempt fails and retries — use the dedicated intrinsic when one exists
- The `_std` and `_cuda` kernels are correct but slower than the intrinsics: both `cuda::std::atomic_ref` and `cuda::atomic_ref` default to sequentially-consistent ordering at system scope — a stronger guarantee than the relaxed device-scope intrinsics (which is why their timings match each other in every section above)

## Files

- `simpleAtomicIntrinsics.cu` — the twelve kernels and the host driver
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA Programming Guide — Atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#atomics)
- [CUDA Programming Guide — Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions)
- [CUDA Core Compute Libraries — cuda::atomic](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html)
