# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    src/main.cpp
    src/pricingengine.cu
    src/test.cpp
)

# Add target for MC_SingleAsianOptionP
add_executable(MC_SingleAsianOptionP ${SRC_FILES})
set_target_properties(MC_SingleAsianOptionP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MC_SingleAsianOptionP PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CUDAToolkit_INCLUDE_DIRS}
)
