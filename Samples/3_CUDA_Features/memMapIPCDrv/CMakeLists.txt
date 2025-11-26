cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(memMapIPCDrv LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for memMapIPCDrv
add_executable(memMapIPCDrv memMapIpc.cpp ../../../Common/helper_multiprocess.cpp)

target_compile_options(memMapIPCDrv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(memMapIPCDrv PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(memMapIPCDrv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(memMapIPCDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(memMapIPCDrv PUBLIC
    CUDA::cuda_driver
)

if(UNIX AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    target_link_libraries(memMapIPCDrv PUBLIC rt)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    target_link_libraries(memMapIPCDrv PUBLIC socket)
endif()

set(CUDA_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/memMapIpc_kernel64.ptx")
set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/memMapIpc_kernel.cu")

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(INCLUDES_LIST)
    foreach(dir ${CUDAToolkit_INCLUDE_DIRS})
        list(APPEND INCLUDES_LIST "-I${dir}")
    endforeach()
    string(JOIN " " INCLUDES "${INCLUDES_LIST}")
endif()

add_custom_command(
    OUTPUT ${CUDA_PTX_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} ${ALL_CCFLAGS} -Wno-deprecated-gpu-targets -o ${CUDA_PTX_FILE} -ptx ${CUDA_KERNEL_SOURCE}
    DEPENDS ${CUDA_KERNEL_SOURCE}
    COMMENT "Building CUDA PTX: ${CUDA_PTX_FILE}"
)


# Create a dummy target for fatbin generation
add_custom_target(generate_memMapIpc_ptx ALL DEPENDS ${CUDA_PTX_FILE})

# Ensure memMapIPCDrv depends on the fatbin
add_dependencies(memMapIPCDrv generate_memMapIpc_ptx)
