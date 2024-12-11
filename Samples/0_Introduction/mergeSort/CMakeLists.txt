# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    mergeSort.cu
    main.cpp
    mergeSort_host.cpp
    mergeSort_validate.cpp
)

# Add target for mergeSort
add_executable(mergeSort ${SRC_FILES})
set_target_properties(mergeSort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(mergeSort PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)
