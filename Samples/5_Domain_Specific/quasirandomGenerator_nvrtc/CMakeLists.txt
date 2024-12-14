# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    quasirandomGenerator.cpp
    quasirandomGenerator_gold.cpp
)

# Add target for quasirandomGenerator_nvrtc
add_executable(quasirandomGenerator_nvrtc ${SRC_FILES})
set_target_properties(quasirandomGenerator_nvrtc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(quasirandomGenerator_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy kernel to the output directory
add_custom_command(TARGET quasirandomGenerator_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/quasirandomGenerator_kernel.cu ${CMAKE_CURRENT_SOURCE_DIR}/quasirandomGenerator_gpu.cuh ${CMAKE_CURRENT_SOURCE_DIR}/quasirandomGenerator_common.h ${CMAKE_CURRENT_BINARY_DIR}
)
