# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCUBLAS.cpp
)

# Add target for simpleCUBLAS
add_executable(simpleCUBLAS ${SRC_FILES})
set_target_properties(simpleCUBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(simpleCUBLAS PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simpleCUBLAS PRIVATE
    CUDA::cudart
    CUDA::cublas
)
