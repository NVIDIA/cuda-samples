# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    quasirandomGenerator.cpp
    quasirandomGenerator_gold.cpp
    quasirandomGenerator_kernel.cu
)

# Add target for quasirandomGenerator
add_executable(quasirandomGenerator ${SRC_FILES})
set_target_properties(quasirandomGenerator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(quasirandomGenerator PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)
