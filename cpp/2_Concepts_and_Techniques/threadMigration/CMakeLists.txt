cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(threadMigration LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(threadMigration threadMigration.cpp)

target_compile_options(threadMigration PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(threadMigration PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(threadMigration PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(threadMigration PUBLIC
    CUDA::cuda_driver
)

if(UNIX AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    target_link_libraries(threadMigration PUBLIC pthread)
endif()

set(CUDA_FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/threadMigration_kernel64.fatbin")
set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/threadMigration_kernel.cu")

# Construct GENCODE_FLAGS explicitly from CUDA architectures
set(GENCODE_FLAGS "")
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
    list(APPEND GENCODE_FLAGS "-gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(INCLUDES_LIST)
    foreach(dir ${CUDAToolkit_INCLUDE_DIRS})
        list(APPEND INCLUDES_LIST "-I${dir}")
    endforeach()
    string(JOIN " " INCLUDES "${INCLUDES_LIST}")
endif()

add_custom_command(
    OUTPUT ${CUDA_FATBIN_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} -Wno-deprecated-gpu-targets ${GENCODE_FLAGS} -o ${CUDA_FATBIN_FILE} -fatbin ${CUDA_KERNEL_SOURCE}
    DEPENDS ${CUDA_KERNEL_SOURCE}
    COMMENT "Building CUDA fatbin: ${CUDA_FATBIN_FILE}"
)

# Create a dummy target for fatbin generation
add_custom_target(generate_fatbin_threadMigration ALL DEPENDS ${CUDA_FATBIN_FILE})

# Ensure matrixMulDrv depends on the fatbin
add_dependencies(threadMigration generate_fatbin_threadMigration)

include(InstallSamples)
setup_samples_install()
