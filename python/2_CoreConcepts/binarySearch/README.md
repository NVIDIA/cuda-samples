# binarySearch (Python)

## Description

This sample demonstrates the parallel binary-search algorithms
exposed by **cuda.compute** (from the `cuda-cccl` package). Given
a sorted `d_data` array and a batch of `d_values` to locate, one
device-wide call returns the insertion index for every value:

- `cuda.compute.lower_bound` writes, for each value, the lowest index
  where it could be inserted into `d_data` without breaking the sort
  order. Equivalent to `numpy.searchsorted(..., side="left")`.
- `cuda.compute.upper_bound` is the analogous upper form, equivalent
  to `numpy.searchsorted(..., side="right")`.

The sample runs both algorithms on two curated inputs: one with
distinct elements (where `lower_bound` and `upper_bound` agree on
any value not in the data) and one with duplicates (where they
diverge on present values). Results are verified against
`numpy.searchsorted`.

## What You'll Learn

- How to call `cuda.compute.lower_bound` / `upper_bound` with CuPy
  arrays
- The semantic difference between `lower_bound` and `upper_bound`,
  especially for inputs containing duplicates
- How the output dtype (`np.uintp`) is used for indices

## Key Libraries

- [`cuda.compute`](https://nvidia.github.io/cccl/python.html) (from the `cuda-cccl` package) - device algorithms
- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - device setup
- `cupy` - device buffers
- `numpy` - host-side reference via `numpy.searchsorted`

## Key APIs

### From `cuda.compute`

- `cuda.compute.lower_bound(d_data, num_items, d_values, num_values, d_out)`
- `cuda.compute.upper_bound(d_data, num_items, d_values, num_values, d_out)`

### From `cuda_samples_utils`

- `print_gpu_info()` - print device name and compute capability

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- Minimum GPU memory: 512 MB

### Software

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-cccl` (>=1.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)

If the CUDA toolkit is not on your `PATH`, set `CUDA_HOME` so that
cuda.compute's JIT path can locate its dependencies:

```bash
export CUDA_HOME=/usr/local/cuda
```

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/binarySearch
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-cccl` (>=1.0.0) - ships the `cuda.compute` module
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)
- `numpy` (>=1.24.0)

## How to Run

### Basic usage

```bash
cd cuda-samples/python/2_CoreConcepts/binarySearch
python binarySearch.py
```

### With custom parameters

```bash
python binarySearch.py --device 1
```

## Expected Output

```
Device: <Your GPU Name>
Compute Capability: <X.Y>

Case 1: distinct data, mixed queries
  data    = [1, 3, 5, 7, 9]
  values  = [0, 3, 4, 10]
  lower_bound: got [0, 1, 2, 5]  expected [0, 1, 2, 5]  OK
  upper_bound: got [0, 2, 2, 5]  expected [0, 2, 2, 5]  OK

Case 2: duplicates in data
  data    = [1, 3, 3, 5, 7, 9]
  values  = [3, 3, 5, 8]
  lower_bound: got [1, 1, 3, 5]  expected [1, 1, 3, 5]  OK
  upper_bound: got [3, 3, 4, 5]  expected [3, 3, 4, 5]  OK

Done
```

**Note:** Device name and compute capability will vary based on your GPU.

## Files

- `binarySearch.py` - Python implementation
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_samples_utils.py` - Common utilities (imported by this sample)
