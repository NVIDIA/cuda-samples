# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    jacobi.cu
    main.cpp
)

# Add target for jacobiCudaGraphs
add_executable(jacobiCudaGraphs ${SRC_FILES})
set_target_properties(jacobiCudaGraphs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(jacobiCudaGraphs PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)
