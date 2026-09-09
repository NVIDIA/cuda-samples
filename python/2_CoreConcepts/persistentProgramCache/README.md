# Persistent Program Cache (Python)

## Description

This sample demonstrates how to persist and reuse compiled CUDA artifacts
with `cuda.core`. It generates a specialized matmul + epilogue kernel,
compiles it to CUBIN on a cache miss, stores the artifact in
`FileStreamProgramCache`, and reloads the CUBIN on later runs.

## Core Cache Flow

The central cache logic is in `compile_or_load_kernel`:

```python
options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
program = Program(source, code_type="c++", options=options)
key = make_program_cache_key(...)

with FileStreamProgramCache(cache_dir) as cache:
    cached = cache.get(key)

    if cached is None:
        module = program.compile("cubin")
        cache[key] = module
        status = "MISS"
    else:
        module = ObjectCode.from_cubin(cached, name="persistentProgramCache")
        status = "HIT"

kernel = module.get_kernel(KERNEL_NAME)
```

## What You'll Learn

- Generating a CUDA source string from workload configuration
- Building a persistent cache key with `make_program_cache_key`
- Storing compiled CUBIN bytes with `FileStreamProgramCache`
- Reconstructing a loadable `ObjectCode` with `ObjectCode.from_cubin`
- Measuring cache lookup, compile, module load, and kernel execution time
- Validating that cached and freshly compiled artifacts produce the same result

## Key Libraries

- [`cuda.core`][cuda-core] - Pythonic access to CUDA programs, object code, launches, and events
- `cupy` - input and output buffers on the GPU
- `numpy` - deterministic input generation and host reference computation

## Key APIs

### From `cuda.core`

- `Program(...).compile("cubin")` - compile generated CUDA source to CUBIN
- `ObjectCode.from_cubin(...)` - reconstruct loadable object code from cached bytes
- `ObjectCode.get_kernel(name)` - fetch the kernel from the compiled artifact
- `LaunchConfig` and `launch(...)` - configure and launch the generated kernel
- `EventOptions(timing_enabled=True)` - time repeated kernel launches

### From `cuda.core.utils`

- `FileStreamProgramCache` - disk-backed, process-safe program cache
- `make_program_cache_key(...)` - derive a cache key from source, options, and target type

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)
- `numpy` (>=2.3.2)

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/persistentProgramCache
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)
- `numpy` (>=2.3.2)

## How to Run

### Basic usage

Run once with a cleared cache to force compilation:

```bash
cd cuda-samples/python/2_CoreConcepts/persistentProgramCache
python persistentProgramCache.py --clear-cache
```

Run again with the same configuration to reuse the cached artifact:

```bash
python persistentProgramCache.py
```

### With custom parameters

```bash
# Compile a different generated kernel variant
python persistentProgramCache.py --tile-size 32 --epilogue identity

# Use a custom matrix size
python persistentProgramCache.py --m 1024 --n 1024 --k 1024

# Use a specific GPU
python persistentProgramCache.py --device 1
```

Changing the tile size, epilogue, source code, compile options, or target
GPU architecture changes the cache key and produces a cache miss.

## Expected Output

The output includes the run configuration, cache status, timings, and
validation result:

```text
Persistent Program Cache
Device:             <Your GPU Name>
Compute Capability: <X.Y>
Cache directory:    <cache path>
Workload:           C = relu(A @ B + bias)
Matrix sizes:       M=512, N=512, K=512
Tile size:          16
Timed launches:     warmup=5, iterations=20

Cache status:       MISS | HIT
Cache key:          <key prefix>...
Artifact size:      <size> KiB
Cache lookup:       <time> ms
Compile time:       <time> ms | skipped
Cache store:        <time> ms          (MISS only)
Module load:        <time> ms
Host prep time:     <time> ms
Kernel time:        <time> ms
Max error:          <error>
Validation:         PASSED
```

Run with `--clear-cache` first to force a miss, then run again with the same
configuration to get a hit. Compare `Host prep time`: the miss path compiles
and stores the CUBIN, while the hit path reloads the cached CUBIN.

**Note:** Device name, timing, artifact size, and cache key will vary based
on GPU, driver, CUDA Toolkit, and host system.

## Files

- `persistentProgramCache.py` - Python implementation using `cuda.core` program cache utilities
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` program caches API](
  https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#program-caches)
- [`cuda.core` examples](https://github.com/NVIDIA/cuda-python/tree/main/cuda_core/examples)

[cuda-core]: https://nvidia.github.io/cuda-python/cuda-core/latest/
