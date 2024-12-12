# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    inlinePTX.cpp
)

# Add target for inlinePTX_nvrtc
add_executable(inlinePTX_nvrtc ${SRC_FILES})
set_target_properties(inlinePTX_nvrtc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(inlinePTX_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy clock_kernel.cu to the output directory
add_custom_command(TARGET inlinePTX_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/inlinePTX_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)
