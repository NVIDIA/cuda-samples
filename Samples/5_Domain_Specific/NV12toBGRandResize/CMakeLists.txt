# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    bgr_resize.cu
    nv12_resize.cu
    nv12_to_bgr_planar.cu
    resize_convert_main.cpp
    utils.cu
)

# Add target for NV12toBGRandResize
add_executable(NV12toBGRandResize ${SRC_FILES})
set_target_properties(NV12toBGRandResize PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(NV12toBGRandResize PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_custom_command(TARGET NV12toBGRandResize POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)
