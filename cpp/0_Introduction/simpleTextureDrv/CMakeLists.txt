cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleTextureDrv LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(simpleTextureDrv simpleTextureDrv.cpp)

target_compile_options(simpleTextureDrv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleTextureDrv PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(simpleTextureDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simpleTextureDrv PUBLIC
    CUDA::cuda_driver
)

set(CUDA_FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/simpleTexture_kernel64.fatbin")
set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/simpleTexture_kernel.cu")

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

# Copy data files to output directory
add_custom_command(TARGET simpleTextureDrv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/teapot512.pgm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET simpleTextureDrv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_rotated.pgm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Create a dummy target for fatbin generation
add_custom_target(generate_fatbin_textureDrv ALL DEPENDS ${CUDA_FATBIN_FILE})

# Ensure simpleTextureDrv depends on the fatbin
add_dependencies(simpleTextureDrv generate_fatbin_textureDrv)

include(InstallSamples)
setup_samples_install()
