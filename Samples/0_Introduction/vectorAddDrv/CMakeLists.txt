cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(vectorAddDrv LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    vectorAddDrv.cpp
)

# Add target for vectorAddDrv
add_executable(vectorAddDrv ${SRC_FILES})
set_target_properties(vectorAddDrv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(vectorAddDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(vectorAddDrv PUBLIC
    CUDA::cuda_driver
)

set(CUDA_FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/vectorAdd_kernel64.fatbin")
set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd_kernel.cu")

add_custom_command(
    OUTPUT ${CUDA_FATBIN_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o ${CUDA_FATBIN_FILE} -fatbin ${CUDA_KERNEL_SOURCE}
    DEPENDS ${CUDA_KERNEL_SOURCE}
    COMMENT "Building CUDA fatbin: ${CUDA_FATBIN_FILE}"
)

# Create a dummy target for fatbin generation
add_custom_target(generate_fatbin_vectorAdd ALL DEPENDS ${CUDA_FATBIN_FILE})

# Ensure matrixMulDrv depends on the fatbin
add_dependencies(vectorAddDrv generate_fatbin_vectorAdd)
