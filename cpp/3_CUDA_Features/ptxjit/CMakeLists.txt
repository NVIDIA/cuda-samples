cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(ptxjit LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(ptxjit ptxjit.cpp)

target_compile_options(ptxjit PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(ptxjit PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(ptxjit PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ptxjit PUBLIC
    CUDA::cuda_driver
    CUDA::cudart
)

set(CUDA_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/ptxjit_kernel64.ptx")
set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptxjit_kernel.cu")

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(INCLUDES_LIST)
    foreach(dir ${CUDAToolkit_INCLUDE_DIRS})
        list(APPEND INCLUDES_LIST "-I${dir}")
    endforeach()
    string(JOIN " " INCLUDES "${INCLUDES_LIST}")
endif()

add_custom_command(
    OUTPUT ${CUDA_PTX_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} -Wno-deprecated-gpu-targets -o ${CUDA_PTX_FILE} -ptx ${CUDA_KERNEL_SOURCE}
    DEPENDS ${CUDA_KERNEL_SOURCE}
    COMMENT "Building CUDA PTX: ${CUDA_PTX_FILE}"
)

# Create a dummy target for fatbin generation
add_custom_target(generate_ptxjit_ptx ALL DEPENDS ${CUDA_PTX_FILE})

# Ensure ptxjit depends on the fatbin
add_dependencies(ptxjit generate_ptxjit_ptx)

include(InstallSamples)
setup_samples_install()
