cmake_minimum_required(VERSION 3.20)

# This project generates container build and run targets for individual CUDA
# samples.  It does NOT require a CUDA toolkit or GPU on the host; all
# compilation happens inside the container.
#
# Configure and build from the repo root:
#
#   cmake -B build-container -S container/
#   cmake --build build-container --target container_build_deviceQuery

project(cuda-samples-container LANGUAGES NONE)

set(CUDA_IMAGE_VERSION "12.6.0" CACHE STRING
    "CUDA image version used as the base for container builds (e.g. 12.6.2, 13.1.0)")

include(../cmake/ContainerSamples.cmake)

# ---------------------------------------------------------------------------
# Register samples that have a Containerfile / Dockerfile.
# Add further samples here as container support is extended.
# ---------------------------------------------------------------------------
add_container_sample(Samples/1_Utilities/deviceQuery)
add_container_sample(Samples/4_CUDA_Libraries/conjugateGradient)
