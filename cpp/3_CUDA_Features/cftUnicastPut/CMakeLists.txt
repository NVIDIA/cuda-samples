cmake_minimum_required(VERSION 3.20)

# The fabric.try_put instructions and the mbarrier layout::v1 form used by this
# sample require __CUDA_ARCH__ >= 1000 (Blackwell and later).
set(SAMPLE_DISALLOW_ARCHS 75 80 86 87 89 90)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(cftUnicastPut LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

find_package(MPI)

if(${MPI_FOUND})
    add_executable(cftUnicastPut cft_unicast_put.cu)

    target_compile_options(cftUnicastPut PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(cftUnicastPut PRIVATE cxx_std_17 cuda_std_17)

    target_include_directories(cftUnicastPut PUBLIC
        ${MPI_INCLUDE_PATH}
    )

    target_link_libraries(cftUnicastPut PUBLIC
        ${MPI_C_LIBRARIES}
        ${MPI_CXX_LIBRARIES}
        CUDA::cuda_driver
    )

    include(InstallSamples)
    setup_samples_install()
else()
    message(STATUS "MPI not found - will not build sample 'cftUnicastPut'")
endif()
