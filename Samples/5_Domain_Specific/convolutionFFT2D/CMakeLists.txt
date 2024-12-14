# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    convolutionFFT2D.cu
    convolutionFFT2D_gold.cpp
    main.cpp
)

# Add target for convolutionFFT2D
add_executable(convolutionFFT2D ${SRC_FILES})
set_target_properties(convolutionFFT2D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(convolutionFFT2D PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(convolutionFFT2D PUBLIC
    CUDA::cufft
)
