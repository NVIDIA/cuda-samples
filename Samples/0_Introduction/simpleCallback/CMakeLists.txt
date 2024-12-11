# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCallback.cu
    multithreading.cpp
)

# Add target for simpleCallback
add_executable(simpleCallback ${SRC_FILES})
set_target_properties(simpleCallback PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
