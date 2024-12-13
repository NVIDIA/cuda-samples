# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    main.cpp
)

# Add target for conjugateGradientPrecond
add_executable(conjugateGradientPrecond ${SRC_FILES})
set_target_properties(conjugateGradientPrecond PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(conjugateGradientPrecond PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(conjugateGradientPrecond PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)
