# Sample: FFT Signal Analysis (Python)

## Description

Analyze signal frequencies using Fast Fourier Transform (FFT) on the GPU. This sample demonstrates CuPy's cuFFT for GPU-accelerated frequency analysis: generating composite signals, computing magnitude spectrum, detecting dominant frequencies via peak detection, and comparing GPU vs CPU FFT performance.

## What You'll Learn

- Using CuPy's `cp.fft.rfft()` for real-to-complex FFT on GPU
- Computing magnitude spectrum from FFT results
- Peak detection to identify dominant frequencies
- Comparing GPU (cuFFT) vs CPU (NumPy) FFT performance
- Uses cuda.core APIs for device management and CUDA event timing

## Key Concepts

- **FFT (Fast Fourier Transform)**: Efficiently computes the Discrete Fourier Transform
- **Magnitude Spectrum**: `|FFT(signal)| * 2 / N` gives amplitude at each frequency
- **rfft**: Real FFT - optimized for real-valued input signals
- **Peak Detection**: Finding local maxima to identify dominant frequencies

### Stream Interop

This sample demonstrates CuPy integration with cuda.core streams:

```python
# Create stream with cuda.core
stream = device.create_stream()

# Use with CuPy operations
cp.cuda.ExternalStream(int(stream.handle)).use()
```

## Key APIs

### From `cuda.core`:

- `Device` - Device management and context
- `EventOptions` - Configure events for GPU timing
- `stream.record()` - Record events for timing

### From CuPy:

- `cp.fft.rfft()` - Real-to-complex FFT (GPU-accelerated via cuFFT)
- `cp.fft.rfftfreq()` - Generate frequency bins for rfft
- `cp.cuda.ExternalStream()` - Interop with cuda.core streams

### From NumPy:

- `np.fft.rfft()` - CPU FFT for comparison

## Requirements

### Hardware:

- NVIDIA GPU with CUDA support

### Software:

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- See `requirements.txt` for Python packages

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python fftSignalAnalysis.py
```

## Expected Output

```
============================================================
FFT Signal Analysis
============================================================

Device: <Your GPU>
Compute Capability: sm_XX

Signal Parameters:
  Samples: 1,048,576
  Sample Rate: 44,100 Hz
  ...

------------------------------------------------------------
GPU FFT (cuFFT)
------------------------------------------------------------
Time: X.XXX ms

Detected Frequencies:
     440.0 Hz (magnitude: X.XXXX)
     ...

------------------------------------------------------------
CPU FFT (NumPy)
------------------------------------------------------------
Time: XX.XXX ms

------------------------------------------------------------
PERFORMANCE SUMMARY
------------------------------------------------------------
GPU (cuFFT): X.XXX ms
CPU (NumPy): XX.XXX ms
Speedup: XXx

------------------------------------------------------------
VERIFICATION
------------------------------------------------------------
GPU vs CPU FFT magnitude: Test PASSED

Frequency Detection Accuracy:
     440 Hz: ✓
     ...

Done
```

**Note:** Times and speedup vary by hardware.

## Files

- `fftSignalAnalysis.py` - Main sample using cuda.core and CuPy
- `README.md` - This file
- `requirements.txt` - Dependencies

## See Also

- [cuda.core Documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CuPy FFT Documentation](https://docs.cupy.dev/en/stable/reference/fft.html)
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
