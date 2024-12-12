# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    main.cpp
    bitonicSort.cu
    sortingNetworks_validate.cpp
)

# Add target for sortingNetworks
add_executable(sortingNetworks ${SRC_FILES})
set_target_properties(sortingNetworks PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(sortingNetworks PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
