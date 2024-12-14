# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    dwtHaar1D.cu
)

# Add target for dwtHaar1D
add_executable(dwtHaar1D ${SRC_FILES})
set_target_properties(dwtHaar1D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_custom_command(TARGET dwtHaar1D POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)
