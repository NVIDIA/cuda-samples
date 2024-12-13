# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCUBLASXT.cpp
)

# Add target for simpleCUBLASXT
add_executable(simpleCUBLASXT ${SRC_FILES})
set_target_properties(simpleCUBLASXT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(simpleCUBLASXT PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simpleCUBLASXT PRIVATE
    CUDA::cudart
    CUDA::cublas
)
