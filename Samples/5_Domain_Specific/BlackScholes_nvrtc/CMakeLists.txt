# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    BlackScholes.cpp
    BlackScholes_gold.cpp
)

# Add target for BlackScholes_nvrtc
add_executable(BlackScholes_nvrtc ${SRC_FILES})
set_target_properties(BlackScholes_nvrtc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(BlackScholes_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy kernel to the output directory
add_custom_command(TARGET BlackScholes_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/BlackScholes_kernel.cuh ${CMAKE_CURRENT_BINARY_DIR}
)
