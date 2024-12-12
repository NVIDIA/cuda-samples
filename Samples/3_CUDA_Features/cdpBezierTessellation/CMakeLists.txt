# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    BezierLineCDP.cu
)

# Add target for cdpBezierTessellation
add_executable(cdpBezierTessellation ${SRC_FILES})
set_target_properties(cdpBezierTessellation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
