# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleTexture.cu
)

# Add target for simpleTexture
add_executable(simpleTexture ${SRC_FILES})
set_target_properties(simpleTexture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
