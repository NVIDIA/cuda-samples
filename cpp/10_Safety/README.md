# CUDA Safety Samples

Safety-specific samples live in this directory. They demonstrate CUDA safety workflows.

Build a sample from its directory:

```bash
mkdir -p build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../../../cmake/toolchains/toolchain-aarch64-qnx-safe.cmake \
  -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=110
cmake --build .
```

Set `CMAKE_CUDA_ARCHITECTURES` to `110` for Thor. See the top-level [README](../../README.md#qnx-safety-cuda-safe-toolkit) for the QNX Safety toolchain setup.
