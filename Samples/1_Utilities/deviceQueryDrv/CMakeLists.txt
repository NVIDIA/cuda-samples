# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    deviceQueryDrv.cpp
)

# Add target for deviceQueryDrv
add_executable(deviceQueryDrv ${SRC_FILES})
set_target_properties(deviceQueryDrv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(deviceQueryDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(deviceQueryDrv PUBLIC
    CUDA::cuda_driver
)
