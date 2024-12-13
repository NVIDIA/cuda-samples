# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    main.cpp
)

# Add target for conjugateGradient
add_executable(conjugateGradient ${SRC_FILES})
set_target_properties(conjugateGradient PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(conjugateGradient PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(conjugateGradient PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)
