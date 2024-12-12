# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    bisect_large.cu
    bisect_small.cu
    bisect_util.cu
    gerschgorin.cpp
    main.cu
    matlab.cpp
)

# Add target for eigenvalues
add_executable(eigenvalues ${SRC_FILES})
set_target_properties(eigenvalues PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
