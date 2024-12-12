# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    deviceQuery.cpp
)

# Add target for deviceQuery
add_executable(deviceQuery ${SRC_FILES})
set_target_properties(deviceQuery PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(deviceQuery PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(deviceQuery PUBLIC
    CUDA::cudart
)
