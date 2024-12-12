# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    compMalloc.cpp
    saxpy.cu
)

# Add target for cudaCompressibleMemory
add_executable(cudaCompressibleMemory ${SRC_FILES})
set_target_properties(cudaCompressibleMemory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cudaCompressibleMemory PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cudaCompressibleMemory PRIVATE
    CUDA::cuda_driver
)
