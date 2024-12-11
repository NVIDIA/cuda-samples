# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    matrixMul.cpp
)

# Add sample target executable
add_executable(matrixMul_nvrtc ${SRC_FILES})
target_link_libraries(matrixMul_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy clock_kernel.cu to the output directory
add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/matrixMul_kernel.cu ${CUDAToolkit_BIN_DIR}/../include/cooperative_groups.h ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CUDAToolkit_BIN_DIR}/../include/cooperative_groups ${CMAKE_CURRENT_BINARY_DIR}/cooperative_groups
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CUDAToolkit_BIN_DIR}/../include/nv ${CMAKE_CURRENT_BINARY_DIR}/nv
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CUDAToolkit_BIN_DIR}/../include/cuda ${CMAKE_CURRENT_BINARY_DIR}/cuda
)
