# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    stereoDisparity.cu
)

# Add target for stereoDisparity
add_executable(stereoDisparity ${SRC_FILES})
set_target_properties(stereoDisparity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_custom_command(TARGET stereoDisparity POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)
