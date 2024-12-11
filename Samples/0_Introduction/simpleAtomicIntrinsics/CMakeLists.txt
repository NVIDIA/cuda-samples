# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleAtomicIntrinsics.cu
    simpleAtomicIntrinsics_cpu.cpp
)

# Add target for simpleAtomicIntrinsics
add_executable(simpleAtomicIntrinsics ${SRC_FILES})
set_target_properties(simpleAtomicIntrinsics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
