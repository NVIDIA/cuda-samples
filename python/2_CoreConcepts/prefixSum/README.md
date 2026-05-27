# Prefix Sum (Scan)

Demonstrates parallel prefix sum (scan) algorithms using cuda.compute with cuda.core stream management.

## Overview

- Inclusive scan: `output[i] = [init_value] + input[0] + input[1] + ... + input[i]`
- Exclusive scan: `output[i] = init_value + input[0] + input[1] + ... + input[i-1]`
- Uses cuda.compute APIs for optimized CUB-based implementations
- Uses cuda.core APIs for device and stream management
- Demonstrates CuPy integration via `ExternalStream`

## Requirements

### Hardware

- NVIDIA GPU with CUDA support

### Software

- CUDA Toolkit 13.0+
- Python 3.10+
- `cuda-python` (13.0.0+)
- `cuda-core` (>=0.6.0)
- `cuda-cccl` (1.0.0+)
- `cupy-cuda13x` (13.0.0+)
- `numpy` (>=2.3.2)

## Usage

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run sample
python prefixSum.py
```

## Key Concepts

| Scan Type | Formula | First Element |
|-----------|---------|---------------|
| Inclusive | `output[i] = [init_value] + Σ input[0..i]` | `[init_value] + input[0]` |
| Exclusive | `output[i] = init_value + Σ input[0..i-1]` | `init_value` (typically `0`, the identity for sum) |

### Stream Management

This sample demonstrates proper stream usage across libraries:

```python
# Create stream with cuda.core
stream = device.create_stream()

# Wrap for CuPy compatibility (requires int handle)
cp_stream = cp.cuda.ExternalStream(int(stream.handle))

# Use with CuPy operations
with cp_stream:
    d_input = cp.asarray(data)
    d_output = cp.empty_like(d_input)

# Pass to cuda.compute
inclusive_scan(
    d_in=d_input,
    d_out=d_output,
    op=OpKind.PLUS,
    init_value=None,
    num_items=len(d_input),
    stream=stream,
)
```

## Applications

- Stream compaction
- Radix sort
- Histogram computation
- Polynomial evaluation
