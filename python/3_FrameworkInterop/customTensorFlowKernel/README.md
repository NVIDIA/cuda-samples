# Sample: TensorFlow Custom GPU Operator

## Description

Learn how to add a custom GPU operation to TensorFlow using `cuda.core` with `tf.py_function`. This sample implements a custom **ReLU operation** (y = max(0, x)) for rapid prototyping of GPU operations.

## Key Question Answered

**Q: How do I add a custom GPU op to TensorFlow?**

**A:** Use `tf.py_function` to wrap cuda.core kernels:
1. Write CUDA kernels (forward + backward) with grid-stride loops
2. Compile them with cuda.core
3. Wrap in Python functions
4. Use `tf.py_function` to call from TensorFlow
5. Register gradients with `@tf.custom_gradient`

## Requirements

- NVIDIA GPU with Compute Capability 7.0+
- CUDA Toolkit 13.0+
- Python 3.10+
- TensorFlow 2.10+
- cuda-python >= 13.0.0
- cuda-core >= 0.6.0 (required for LEGACY_DEFAULT_STREAM)
- numpy >= 2.3.2
- CuPy (for device pointer access)

## Installation

```bash
cd python/3_FrameworkInterop/customTensorFlowKernel
pip install -r requirements.txt
```

## How to Run

```bash
python customTensorFlowKernel.py
python customTensorFlowKernel.py --size 1000000
```

## Usage Example

```python
import tensorflow as tf
from customTensorFlowKernel import custom_relu

# Simple usage
x = tf.random.normal([100], dtype=tf.float32)
y = custom_relu(x)

# In a Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.Lambda(custom_relu),
    tf.keras.layers.Dense(10)
])
```

## Key Concepts

- **tf.py_function**: Bridges TensorFlow and Python code using cuda.core (has overhead, not XLA-compatible)
- **@tf.custom_gradient**: Registers custom backward pass
- **cuda.core**: Primary GPU manager (device, stream, kernel compilation)
- **CuPy**: Internal helper for device pointer access only

## Production Alternatives

This sample is for rapid prototyping. For production:
- **TensorFlow C++ Custom Op**: Full performance, XLA compatible
- **XLA Custom Calls**: For XLA-compiled models
- See TensorFlow documentation for details

## See Also

- [cuda.core Documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [TensorFlow tf.py_function](https://www.tensorflow.org/api_docs/python/tf/py_function)
- [TensorFlow @custom_gradient](https://www.tensorflow.org/api_docs/python/tf/custom_gradient)
- [TensorFlow C++ Custom Op Guide](https://www.tensorflow.org/guide/create_op)
- [CuPy Documentation](https://docs.cupy.dev/)
