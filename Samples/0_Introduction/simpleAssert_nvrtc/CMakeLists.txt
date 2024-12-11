# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleAssert.cpp
)

# Add sample target executable
add_executable(simpleAssert_nvrtc ${SRC_FILES})
target_link_libraries(simpleAssert_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy clock_kernel.cu to the output directory
add_custom_command(TARGET simpleAssert_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/simpleAssert_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)
