cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(vectorAddMMAP LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample vectorAddMMAP - not supported on aarch64")
else()
    add_executable(vectorAddMMAP vectorAddMMAP.cpp multidevicealloc_memmap.cpp)

    target_compile_options(vectorAddMMAP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(vectorAddMMAP PRIVATE cxx_std_17 cuda_std_17)

    target_include_directories(vectorAddMMAP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(vectorAddMMAP PUBLIC
        CUDA::cuda_driver
    )

    set(CUDA_FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/vectorAdd_kernel64.fatbin")
    set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd_kernel.cu")

    # Construct GENCODE_FLAGS explicitly from CUDA architectures
    set(GENCODE_FLAGS "")
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        list(APPEND GENCODE_FLAGS "-gencode=arch=compute_${arch},code=sm_${arch}")
    endforeach()

    add_custom_command(
        OUTPUT ${CUDA_FATBIN_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} -Wno-deprecated-gpu-targets ${GENCODE_FLAGS} -o ${CUDA_FATBIN_FILE} -fatbin ${CUDA_KERNEL_SOURCE}
        DEPENDS ${CUDA_KERNEL_SOURCE}
        COMMENT "Building CUDA fatbin: ${CUDA_FATBIN_FILE}"
    )

    # Create a dummy target for fatbin generation
    add_custom_target(generate_fatbin_vectorAddMMAP ALL DEPENDS ${CUDA_FATBIN_FILE})

    # Ensure matrixMulDrv depends on the fatbin
    add_dependencies(vectorAddMMAP generate_fatbin_vectorAddMMAP)
endif()

include(InstallSamples)
setup_samples_install()
