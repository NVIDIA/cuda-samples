# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    src/main.cpp
    src/piestimator.cu
    src/test.cpp
)

# Add target for MC_EstimatePiInlineQ
add_executable(MC_EstimatePiInlineQ ${SRC_FILES})
set_target_properties(MC_EstimatePiInlineQ PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MC_EstimatePiInlineQ PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(MC_EstimatePiInlineQ PUBLIC
    CUDA::curand
)
