# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    main.cpp
    scan_gold.cpp
    scan.cu
)

# Add target for scan
add_executable(scan ${SRC_FILES})
set_target_properties(scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(scan PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
