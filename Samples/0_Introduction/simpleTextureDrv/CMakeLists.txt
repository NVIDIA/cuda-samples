cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleTextureDrv LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for simpleTextureDrv
add_executable(simpleTextureDrv simpleTextureDrv.cpp)

target_compile_options(simpleTextureDrv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleTextureDrv PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleTextureDrv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(simpleTextureDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simpleTextureDrv PUBLIC
    CUDA::cuda_driver
)

set(CUDA_FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/simpleTexture_kernel64.fatbin")
set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/simpleTexture_kernel.cu")

add_custom_command(
    OUTPUT ${CUDA_FATBIN_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o ${CUDA_FATBIN_FILE} -fatbin ${CUDA_KERNEL_SOURCE}
    DEPENDS ${CUDA_KERNEL_SOURCE}
    COMMENT "Building CUDA fatbin: ${CUDA_FATBIN_FILE}"
)

# Create a dummy target for fatbin generation
add_custom_target(generate_fatbin_textureDrv ALL DEPENDS ${CUDA_FATBIN_FILE})

# Ensure simpleTextureDrv depends on the fatbin
add_dependencies(simpleTextureDrv generate_fatbin_textureDrv)
