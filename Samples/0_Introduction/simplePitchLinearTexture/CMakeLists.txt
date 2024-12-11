# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simplePitchLinearTexture.cu
)

# Add target for simplePitchLinearTexture
add_executable(simplePitchLinearTexture ${SRC_FILES})
set_target_properties(simplePitchLinearTexture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
