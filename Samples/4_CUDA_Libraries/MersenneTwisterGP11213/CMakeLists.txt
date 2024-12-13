# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    MersenneTwister.cpp
)

# Add target for MersenneTwisterGP11213
add_executable(MersenneTwisterGP11213 ${SRC_FILES})
set_target_properties(MersenneTwisterGP11213 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MersenneTwisterGP11213 PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(MersenneTwisterGP11213 PRIVATE
    CUDA::cudart
    CUDA::curand
)
