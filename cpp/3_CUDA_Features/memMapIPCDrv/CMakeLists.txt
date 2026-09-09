cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(memMapIPCDrv LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(memMapIPCDrv memMapIpc.cpp ../../../Common/helper_multiprocess.cpp)

target_compile_options(memMapIPCDrv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(memMapIPCDrv PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(memMapIPCDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(memMapIPCDrv PUBLIC
    CUDA::cuda_driver
)

if(UNIX AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    target_link_libraries(memMapIPCDrv PUBLIC rt)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    target_link_libraries(memMapIPCDrv PUBLIC socket)
endif()

set(CUDA_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/memMapIpc_kernel64.ptx")
set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/memMapIpc_kernel.cu")

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
add_custom_target(generate_memMapIpc_ptx ALL DEPENDS ${CUDA_PTX_FILE})

# Ensure memMapIPCDrv depends on the fatbin
add_dependencies(memMapIPCDrv generate_memMapIpc_ptx)

include(InstallSamples)
setup_samples_install()
