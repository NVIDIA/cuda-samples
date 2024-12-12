# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    histogram256.cu
    histogram64.cu
    histogram_gold.cpp
    main.cpp
)

# Add target for histogram
add_executable(histogram ${SRC_FILES})
set_target_properties(histogram PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(histogram PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
