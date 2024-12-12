# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    dct8x8.cu
    BmpUtil.cpp
    DCT8x8_Gold.cpp
)

# Add target for dct8x8
add_executable(dct8x8 ${SRC_FILES})
set_target_properties(dct8x8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(dct8x8 PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

file(GLOB REF_DATA "teapot512*")
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${REF_DATA}
    ${CMAKE_CURRENT_BINARY_DIR}
)
