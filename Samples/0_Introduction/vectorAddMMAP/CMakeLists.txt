cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(vectorAddMMAP LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for vectorAddMMAP
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample vectorAddMMAP - not supported on aarch64")
else()
    add_executable(vectorAddMMAP vectorAddMMAP.cpp multidevicealloc_memmap.cpp)

    target_compile_options(vectorAddMMAP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(vectorAddMMAP PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(vectorAddMMAP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(vectorAddMMAP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(vectorAddMMAP PUBLIC
        CUDA::cuda_driver
    )

    set(CUDA_FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/vectorAdd_kernel64.fatbin")
    set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd_kernel.cu")

    add_custom_command(
        OUTPUT ${CUDA_FATBIN_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} ${ALL_CCFLAGS} -Wno-deprecated-gpu-targets  ${GENCODE_FLAGS} -o ${CUDA_FATBIN_FILE} -fatbin ${CUDA_KERNEL_SOURCE}
        DEPENDS ${CUDA_KERNEL_SOURCE}
        COMMENT "Building CUDA fatbin: ${CUDA_FATBIN_FILE}"
    )

    # Create a dummy target for fatbin generation
    add_custom_target(generate_fatbin_vectorAddMMAP ALL DEPENDS ${CUDA_FATBIN_FILE})

    # Ensure matrixMulDrv depends on the fatbin
    add_dependencies(vectorAddMMAP generate_fatbin_vectorAddMMAP)
endif()
