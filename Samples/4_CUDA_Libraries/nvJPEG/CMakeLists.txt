# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    nvJPEG.cpp
)

# Add target for nvJPEG
add_executable(nvJPEG ${SRC_FILES})
set_target_properties(nvJPEG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(nvJPEG PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nvJPEG PRIVATE
    CUDA::cudart
    CUDA::nvjpeg
)

# Copy data to the output directory
add_custom_command(TARGET nvJPEG POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/images
    ${CMAKE_CURRENT_BINARY_DIR}/images
)
