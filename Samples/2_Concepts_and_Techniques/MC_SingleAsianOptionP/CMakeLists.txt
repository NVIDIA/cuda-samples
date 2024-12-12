# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    MC_SingleAsianOptionP.cu
)

# Add target for MC_SingleAsianOptionP
add_executable(MC_SingleAsianOptionP ${SRC_FILES})
set_target_properties(MC_SingleAsianOptionP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
