# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    convolutionTexture.cu
    convolutionTexture_gold.cpp
    main.cpp
)

# Add target for convolutionTexture
add_executable(convolutionTexture ${SRC_FILES})
set_target_properties(convolutionTexture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(convolutionTexture PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
