# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleAtomicIntrinsics_cpu.cpp
    simpleAtomicIntrinsics.cpp
)

# Add sample target executable
add_executable(simpleAtomicIntrinsics_nvrtc ${SRC_FILES})
target_link_libraries(simpleAtomicIntrinsics_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy clock_kernel.cu to the output directory
add_custom_command(TARGET simpleAtomicIntrinsics_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/simpleAtomicIntrinsics_kernel.cuh
    ${CMAKE_CURRENT_BINARY_DIR}
)
