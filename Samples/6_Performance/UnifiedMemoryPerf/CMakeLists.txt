# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    helperFunctions.cpp
    matrixMultiplyPerf.cu
    commonKernels.cu
)

# Add target for UnifiedMemoryPerf
add_executable(UnifiedMemoryPerf ${SRC_FILES})
set_target_properties(UnifiedMemoryPerf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(UnifiedMemoryPerf PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)
