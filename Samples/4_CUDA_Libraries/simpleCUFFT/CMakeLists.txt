# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCUFFT.cu
)

# Add target for simpleCUFFT
add_executable(simpleCUFFT ${SRC_FILES})
set_target_properties(simpleCUFFT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(simpleCUFFT PRIVATE
    CUDA::cufft
)
