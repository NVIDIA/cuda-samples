# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    binomialOptions.cpp
    binomialOptions_gold.cpp
    binomialOptions_gpu.cpp
)

# Add target for binomialOptions_nvrtc
add_executable(binomialOptions_nvrtc ${SRC_FILES})
set_target_properties(binomialOptions_nvrtc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(binomialOptions_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy kernel to the output directory
add_custom_command(TARGET binomialOptions_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/binomialOptions_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)
