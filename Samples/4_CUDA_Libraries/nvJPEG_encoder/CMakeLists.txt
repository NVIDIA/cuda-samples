# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    nvJPEG_encoder.cpp
)

# Add target for nvJPEG_encoder
add_executable(nvJPEG_encoder ${SRC_FILES})
set_target_properties(nvJPEG_encoder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(nvJPEG_encoder PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nvJPEG_encoder PRIVATE
    CUDA::cudart
    CUDA::nvjpeg
)

# Copy data to the output directory
add_custom_command(TARGET nvJPEG_encoder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/images
    ${CMAKE_CURRENT_BINARY_DIR}/images
)

add_custom_command(TARGET nvJPEG_encoder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/encode_output
    ${CMAKE_CURRENT_BINARY_DIR}/encode_output
)
