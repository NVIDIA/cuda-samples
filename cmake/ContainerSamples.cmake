# ContainerSamples.cmake
#
# Provides add_container_sample() for creating container build and run targets
# for individual CUDA samples.  No CUDA toolkit is required on the host; all
# compilation happens inside the container using the devel base image.
#
# Any OCI-compatible CLI (docker, podman, nerdctl, …) is supported.  The tool
# is located automatically, or can be overridden by setting CONTAINER_EXECUTABLE
# before including this module.
#
# Usage (from a CUDA-free CMakeLists.txt):
#
#   include(cmake/ContainerSamples.cmake)
#   add_container_sample(Samples/1_Utilities/deviceQuery)
#
# This creates two targets per sample:
#   container_build_<name>  -- builds the image from the sample's Containerfile
#   container_run_<name>    -- runs the image with GPU access

find_program(CONTAINER_EXECUTABLE
    NAMES docker podman nerdctl
    REQUIRED
    DOC "OCI-compatible container CLI used to build and run sample images"
)

# CMAKE_CURRENT_LIST_DIR is the directory of this file (cmake/), so the repo
# root is one level up regardless of where the including CMakeLists.txt lives.
set(_CUDA_SAMPLES_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

function(add_container_sample SAMPLE_REL_PATH)
    get_filename_component(SAMPLE_NAME "${SAMPLE_REL_PATH}" NAME)

    # Accept either 'Containerfile' (preferred) or 'Dockerfile' (legacy name).
    set(_CONTAINERFILE "${_CUDA_SAMPLES_ROOT}/${SAMPLE_REL_PATH}/Containerfile")
    if(NOT EXISTS "${_CONTAINERFILE}")
        set(_CONTAINERFILE "${_CUDA_SAMPLES_ROOT}/${SAMPLE_REL_PATH}/Dockerfile")
    endif()

    if(NOT EXISTS "${_CONTAINERFILE}")
        message(WARNING "add_container_sample: no Containerfile found for '${SAMPLE_NAME}' "
                        "(expected ${_CUDA_SAMPLES_ROOT}/${SAMPLE_REL_PATH}/Containerfile)")
        return()
    endif()

    string(TOLOWER "${SAMPLE_NAME}" _SAMPLE_NAME_LOWER)
    set(_IMAGE_TAG "cuda-samples/${_SAMPLE_NAME_LOWER}:latest")

    set(_BUILD_ARGS --file "${_CONTAINERFILE}" --tag "${_IMAGE_TAG}")
    if(CUDA_IMAGE_VERSION)
        list(APPEND _BUILD_ARGS --build-arg "CUDA_IMAGE_VERSION=${CUDA_IMAGE_VERSION}")
    endif()

    add_custom_target(container_build_${SAMPLE_NAME}
        COMMAND "${CONTAINER_EXECUTABLE}" build
                ${_BUILD_ARGS}
                "${_CUDA_SAMPLES_ROOT}"
        WORKING_DIRECTORY "${_CUDA_SAMPLES_ROOT}"
        COMMENT "Building container image ${_IMAGE_TAG}"
        VERBATIM
    )

    add_custom_target(container_run_${SAMPLE_NAME}
        COMMAND "${CONTAINER_EXECUTABLE}" run --rm --runtime=nvidia --gpus all "${_IMAGE_TAG}"
        COMMENT "Running ${_IMAGE_TAG}"
        VERBATIM
    )
endfunction()
