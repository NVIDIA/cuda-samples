# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCUBLAS_LU.cpp
)

# Add target for simpleCUBLAS_LU
add_executable(simpleCUBLAS_LU ${SRC_FILES})
set_target_properties(simpleCUBLAS_LU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(simpleCUBLAS_LU PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simpleCUBLAS_LU PRIVATE
    CUDA::cudart
    CUDA::cublas
)
