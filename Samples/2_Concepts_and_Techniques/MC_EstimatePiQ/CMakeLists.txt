# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    src/main.cpp
    src/piestimator.cu
    src/test.cpp
)

# Add target for MC_EstimatePiQ
add_executable(MC_EstimatePiQ ${SRC_FILES})
set_target_properties(MC_EstimatePiQ PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MC_EstimatePiQ PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(MC_EstimatePiQ PUBLIC
    CUDA::curand
)
