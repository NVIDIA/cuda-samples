# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    batchCUBLAS.cpp
)

# Add target for batchCUBLAS
add_executable(batchCUBLAS ${SRC_FILES})
set_target_properties(batchCUBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(batchCUBLAS PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(batchCUBLAS PRIVATE
    CUDA::cublas
    CUDA::cudart
)
