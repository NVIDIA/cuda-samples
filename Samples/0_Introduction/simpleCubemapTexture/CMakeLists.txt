# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCubemapTexture.cu
)

# Add target for simpleCubemapTexture
add_executable(simpleCubemapTexture ${SRC_FILES})
set_target_properties(simpleCubemapTexture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
