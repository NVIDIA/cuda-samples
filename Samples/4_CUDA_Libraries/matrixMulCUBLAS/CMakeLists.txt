# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    matrixMulCUBLAS.cpp
)

# Add target for matrixMulCUBLAS
add_executable(matrixMulCUBLAS ${SRC_FILES})
set_target_properties(matrixMulCUBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(matrixMulCUBLAS PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(matrixMulCUBLAS PRIVATE
    CUDA::cudart
    CUDA::cublas
)
