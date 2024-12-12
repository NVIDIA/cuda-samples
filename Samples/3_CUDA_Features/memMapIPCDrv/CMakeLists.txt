# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    memMapIpc.cpp
    ../../../Common/helper_multiprocess.cpp
)

# Add target for memMapIPCDrv
add_executable(memMapIPCDrv ${SRC_FILES})
set_target_properties(memMapIPCDrv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(memMapIPCDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(memMapIPCDrv PUBLIC
    CUDA::cuda_driver
)

set(CUDA_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/memMapIpc_kernel64.ptx")
set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/memMapIpc_kernel.cu")

add_custom_command(
    OUTPUT ${CUDA_PTX_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o ${CUDA_PTX_FILE} -ptx ${CUDA_KERNEL_SOURCE}
    DEPENDS ${CUDA_KERNEL_SOURCE}
    COMMENT "Building CUDA PTX: ${CUDA_PTX_FILE}"
)


# Create a dummy target for fatbin generation
add_custom_target(generate_memMapIpc_ptx ALL DEPENDS ${CUDA_PTX_FILE})

# Ensure matrixMulDrv depends on the fatbin
add_dependencies(matrixMulDrv generate_memMapIpc_ptx)
