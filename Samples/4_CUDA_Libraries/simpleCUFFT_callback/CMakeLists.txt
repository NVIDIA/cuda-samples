# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCUFFT_callback.cu
)

# Add target for simpleCUFFT_callback
add_executable(simpleCUFFT_callback ${SRC_FILES})
set_target_properties(simpleCUFFT_callback PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(simpleCUFFT_callback PRIVATE
    CUDA::cufft
)
