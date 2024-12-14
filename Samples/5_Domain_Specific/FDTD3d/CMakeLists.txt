# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    src/FDTD3d.cpp
    src/FDTD3dGPU.cu
    src/FDTD3dReference.cpp
)

# Add target for FDTD3d
add_executable(FDTD3d ${SRC_FILES})
set_target_properties(FDTD3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(FDTD3d PRIVATE
    inc
)
