# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    BlackScholes.cu
    BlackScholes_gold.cpp
)

# Add target for BlackScholes
add_executable(BlackScholes ${SRC_FILES})
set_target_properties(BlackScholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
