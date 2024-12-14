# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    dxtc.cu
)

# Add target for dxtc
add_executable(dxtc ${SRC_FILES})
set_target_properties(dxtc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_custom_command(TARGET dxtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)
