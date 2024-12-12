# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    convolutionSeparable.cu
    convolutionSeparable_gold.cpp
    main.cpp
)

# Add target for convolutionSeparable
add_executable(convolutionSeparable ${SRC_FILES})
set_target_properties(convolutionSeparable PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(convolutionSeparable PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
