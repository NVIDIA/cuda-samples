# Sample: Kernel Nsys Profiling - CUDA C++ Kernel Profiling with cuda.core (Python)

## Description

This sample demonstrates how to profile custom CUDA C++ kernels compiled and launched with `cuda.core` using NVIDIA Nsight Systems. It implements three GPU operations (vector addition, SAXPY, vector transform) as custom kernels and shows how to instrument code with NVTX markers for profiling analysis.

## What you will learn

- How to write and compile CUDA C++ kernels with `cuda.core.Program`
- How to launch kernels with `LaunchConfig` and manage CUDA streams
- How to use NVTX markers (`nvtx.annotate()`) to annotate code sections
- How to profile kernels with Nsight Systems and analyze performance
- Modern CUDA Python workflow with `cuda.core.Device` and proper resource cleanup

## Requirements

- NVIDIA GPU with Compute Capability 7.0+
- CUDA Toolkit 13.0+
- Python 3.10+
- Packages: `numpy`, `cuda-python`, `cuda-core`, `cupy-cuda13x`, `nvtx` (see `requirements.txt`; NumPy >=2.3.2)

**Install:**
```bash
pip install -r requirements.txt
```

## How to run

```bash
python kernelNsysProfile.py
python kernelNsysProfile.py --array-size 10000000  # Custom size
```

## Nsys Profiling

**Basic profile:**
```bash
nsys profile -o gpu_profile python kernelNsysProfile.py
nsys-ui gpu_profile.nsys-rep  # View results
```

The program uses color-coded NVTX markers:
- **Purple**: Phase 2 (cuda.core Custom Kernels - main focus)
- **Yellow/Blue/Green**: Other phases
- **Cyan**: Nested operations

Focus on Phase 2 to analyze kernel execution times, launch overhead, and GPU utilization.

**For detailed Nsys usage and analysis techniques, see the [NVIDIA Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/).**

## Troubleshooting

**Missing packages:**
```bash
pip install -r requirements.txt
```

**Out of memory:**
```bash
python kernelNsysProfile.py -n 10000000  # Reduce array size
```

**Nsys not found:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CuPy Documentation](https://docs.cupy.dev/)
