# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCUFFT_2d_MGPU.cu
)

# Add target for simpleCUFFT_2d_MGPU
add_executable(simpleCUFFT_2d_MGPU ${SRC_FILES})
set_target_properties(simpleCUFFT_2d_MGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(simpleCUFFT_2d_MGPU PRIVATE
    CUDA::cufft
)
