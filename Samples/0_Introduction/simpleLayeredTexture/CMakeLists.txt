# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleLayeredTexture.cu
)

# Add target for simpleLayeredTexture
add_executable(simpleLayeredTexture ${SRC_FILES})
set_target_properties(simpleLayeredTexture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
