# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleIPC.cu
    ../../../Common/helper_multiprocess.cpp
)

# Add target for simpleIPC
add_executable(simpleIPC ${SRC_FILES})
set_target_properties(simpleIPC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
