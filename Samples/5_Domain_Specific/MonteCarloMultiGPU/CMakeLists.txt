# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    MonteCarlo_gold.cpp
    MonteCarlo_kernel.cu
    multithreading.cpp
    MonteCarloMultiGPU.cpp
)

# Add target for MonteCarloMultiGPU
add_executable(MonteCarloMultiGPU ${SRC_FILES})
set_target_properties(MonteCarloMultiGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MonteCarloMultiGPU PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(MonteCarloMultiGPU PRIVATE
    CUDA::curand
)
