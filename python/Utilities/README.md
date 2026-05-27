# CUDA Python Utilities

Common utilities for CUDA Python samples using the `cuda.core` API.

## Overview

This module provides reusable utility functions for CUDA samples to reduce code duplication. Samples import from `cuda_samples_utils.py` using simple path-based imports (no package structure needed).

## Installation Requirements

Install from the Python samples directory:

```bash
cd /path/to/cuda-samples/Python
pip install -r requirements.txt
```

This installs a common CUDA 13 stack (see `python/requirements.txt`):

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)
- `numpy` (>=2.3.2)

## How to Use in Samples

Import utilities using path-based import:

```python
import sys
from pathlib import Path

# Add Utilities directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result

# Use the utility
if verify_array_result(result, expected):
    print("Success!")
```

## Available Functions

### Result Verification

#### `verify_array_result(result, expected, rtol=1e-5, atol=1e-8, verbose=True)`

Verify computed results match expected values. The helper detects whether both
arguments are NumPy arrays or both are CuPy arrays and uses the matching
library's `allclose` (no unnecessary cross-device transfers).

**Parameters:**
- `result`: NumPy or CuPy array with computed results
- `expected`: NumPy or CuPy array with expected values (same kind as `result`)
- `rtol`: Relative tolerance (default: 1e-5)
- `atol`: Absolute tolerance (default: 1e-8)
- `verbose`: Print test result (default: True)

**Returns:**
- `True` if results match within tolerance, `False` otherwise

**Example:**
```python
expected = a + b
if verify_array_result(c, expected):
    print("Computation correct!")
```

### Package Check

#### `check_cuda_requirements()`

Check if required CUDA packages are available.

**Returns:**
- `True` if requirements are met, `False` otherwise

**Example:**
```python
if not check_cuda_requirements():
    sys.exit(1)
```

## Design Philosophy

These utilities focus on common operations that are **not** part of `cuda.core` API:
- Result verification for NumPy or CuPy arrays
- Package requirements checking

For CUDA operations like device initialization, kernel compilation, and grid size calculations, samples should use `cuda.core` API directly to demonstrate the proper usage patterns.

## Complete Example

See `../1_GettingStarted/vectorAdd/vectorAdd.py` for a complete example:

```python
import sys
from pathlib import Path

# Import utility
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result

import cupy as cp
from cuda.core import Device, Program, ProgramOptions, LaunchConfig, launch

# Use cuda.core directly for device and kernel operations
device = Device(0)
device.set_current()

program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
program = Program(kernel_source, code_type="c++", options=program_options)
module = program.compile("cubin", name_expressions=("kernel_name",))
kernel = module.get_kernel("kernel_name")

# Calculate grid size inline
threads_per_block = 256
blocks_per_grid = (num_elements + threads_per_block - 1) // threads_per_block

# Launch kernel - pass cupy arrays directly
config = LaunchConfig(grid=blocks_per_grid, block=threads_per_block)
launch(stream, config, kernel, a, b, c, cp.int32(num_elements))

# Verify results using utility
verify_array_result(c, expected)
```

## Benefits

- **Code Reuse**: Write common functionality once
- **Consistency**: All samples use the same patterns
- **Maintainability**: Bug fixes benefit all samples
- **Transparency**: Samples show cuda.core API usage directly
- **Simplicity**: No complex package structure needed
