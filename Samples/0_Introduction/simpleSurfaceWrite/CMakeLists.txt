# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleSurfaceWrite.cu
)

# Add target for simpleSurfaceWrite
add_executable(simpleSurfaceWrite ${SRC_FILES})
set_target_properties(simpleSurfaceWrite PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy data to the output directory
add_custom_command(TARGET simpleSurfaceWrite POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}
)
