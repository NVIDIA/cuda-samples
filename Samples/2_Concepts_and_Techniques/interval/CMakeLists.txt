# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    interval.cu
)

# Add target for interval
add_executable(interval ${SRC_FILES})
set_target_properties(interval PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(interval PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
