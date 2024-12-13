# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    watershedSegmentationNPP.cpp
)

# Add target for watershedSegmentationNPP
add_executable(watershedSegmentationNPP ${SRC_FILES})
set_target_properties(watershedSegmentationNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(watershedSegmentationNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(watershedSegmentationNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppif
        CUDA::cudart
)
