# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    src/main.cpp
    src/piestimator.cu
    src/test.cpp
)

# Add target for MC_EstimatePiInlineP
add_executable(MC_EstimatePiInlineP ${SRC_FILES})
set_target_properties(MC_EstimatePiInlineP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MC_EstimatePiInlineP PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CUDAToolkit_INCLUDE_DIRS}
)
