# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cdpSimplePrint.cu
)

# Add target for cdpSimplePrint
add_executable(cdpSimplePrint ${SRC_FILES})
set_target_properties(cdpSimplePrint PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
