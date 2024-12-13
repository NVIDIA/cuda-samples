# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    main.cpp
)

# Add target for conjugateGradientUM
add_executable(conjugateGradientUM ${SRC_FILES})
set_target_properties(conjugateGradientUM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(conjugateGradientUM PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(conjugateGradientUM PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)
