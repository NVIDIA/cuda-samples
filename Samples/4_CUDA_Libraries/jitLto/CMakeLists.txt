# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    jitLto.cpp
)

# Add target for jitLto
add_executable(jitLto ${SRC_FILES})
set_target_properties(jitLto PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(jitLto PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(jitLto PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
)
