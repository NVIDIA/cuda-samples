# Building CUDA Samples as Container Images

Individual CUDA samples can be built and run as OCI container images.  This
requires no CUDA toolkit or GPU driver on the build host — all compilation
happens inside the container using the CUDA devel base image.

## Prerequisites

- An OCI-compatible container CLI: [Docker](https://docs.docker.com/engine/install/),
  [Podman](https://podman.io/docs/installation), or
  [nerdctl](https://github.com/containerd/nerdctl)
- CMake 3.20 or later

## Configure

From the repository root, configure the container build project:

```bash
cmake -B build-container -S container/
```

CMake will automatically detect `docker`, `podman`, or `nerdctl` (in that
order).  To use a specific tool, pass it explicitly:

```bash
cmake -B build-container -S container/ -DCONTAINER_EXECUTABLE=/usr/bin/podman
```

## Build a sample image

```bash
cmake --build build-container --target container_build_deviceQuery
cmake --build build-container --target container_build_conjugateGradient
```

Each target runs the equivalent of:

```bash
docker build \
    --file Samples/<category>/<name>/Containerfile \
    --tag cuda-samples/<name>:latest \
    .
```

using the repo root as the build context.

## Run a sample

Running a sample requires a GPU on the current machine and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
installed on the host.

```bash
cmake --build build-container --target container_run_deviceQuery
```

Or directly:

```bash
docker run --rm --runtime=nvidia --gpus=all cuda-samples/devicequery:latest
```

## CUDA image version

The Containerfiles default to the version set by `CUDA_IMAGE_VERSION` in
[CMakeLists.txt](CMakeLists.txt).  Override it at configure time to build
against a different CUDA base image:

```bash
cmake -B build-container -S container/ -DCUDA_IMAGE_VERSION=13.1.0
```

To tag the image to reflect the version, pass `--tag` directly:

```bash
docker build \
    --build-arg CUDA_IMAGE_VERSION=13.1.0 \
    --file Samples/1_Utilities/deviceQuery/Containerfile \
    --tag cuda-samples/devicequery:cuda13.1.0 \
    .
```

## Adding container support for a new sample

1. Add a `Containerfile` to the sample's directory (see an existing sample for
   reference).  The build context is always the repository root, so the
   directory structure under `/cuda-samples/` inside the container must mirror
   the repository layout so that relative paths in the sample's `CMakeLists.txt`
   resolve correctly.

2. Register the sample in [CMakeLists.txt](CMakeLists.txt):

   ```cmake
   add_container_sample(Samples/<category>/<name>)
   ```

## Samples with container support

| Sample | Category |
|--------|----------|
| [deviceQuery](../Samples/1_Utilities/deviceQuery/) | Utilities |
| [conjugateGradient](../Samples/4_CUDA_Libraries/conjugateGradient/) | CUDA Libraries |
