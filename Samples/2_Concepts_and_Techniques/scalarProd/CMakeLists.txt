# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    scalarProd_cpu.cpp
    scalarProd.cu
)

# Add target for scalarProd
add_executable(scalarProd ${SRC_FILES})
set_target_properties(scalarProd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
