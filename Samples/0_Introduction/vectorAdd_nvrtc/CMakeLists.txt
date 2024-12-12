# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    vectorAdd.cpp
)

# Add sample target executable
add_executable(vectorAdd_nvrtc ${SRC_FILES})
target_link_libraries(vectorAdd_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy vectorAdd_kernel.cu to the output directory
add_custom_command(TARGET vectorAdd_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)
