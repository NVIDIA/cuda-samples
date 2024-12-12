# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cdpQuadtree.cu
)

# Add target for cdpQuadtree
add_executable(cdpQuadtree ${SRC_FILES})
set_target_properties(cdpQuadtree PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
