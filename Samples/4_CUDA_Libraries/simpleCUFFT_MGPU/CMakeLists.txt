# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCUFFT_MGPU.cu
)

# Add target for simpleCUFFT_MGPU
add_executable(simpleCUFFT_MGPU ${SRC_FILES})
set_target_properties(simpleCUFFT_MGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(simpleCUFFT_MGPU PRIVATE
    CUDA::cufft
)
