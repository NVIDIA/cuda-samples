set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Need to set the QNX_HOST and QNX_TARGET environment variables
set(QNX_HOST $ENV{QNX_HOST})
set(QNX_TARGET $ENV{QNX_TARGET})

message(STATUS "QNX_HOST = ${QNX_HOST}")
message(STATUS "QNX_TARGET = ${QNX_TARGET}")

find_program(QNX_QCC   NAMES qcc   PATHS "${QNX_HOST}/usr/bin")
find_program(QNX_QPLUS NAMES q++   PATHS "${QNX_HOST}/usr/bin")

if(NOT QNX_QCC OR NOT QNX_QPLUS)
    message(FATAL_ERROR "Could not find qcc or q++ in QNX_HOST=${QNX_HOST}/usr/bin")
endif()

# Specify the cross-compilers
set(CMAKE_C_COMPILER ${QNX_QCC})
set(CMAKE_CXX_COMPILER ${QNX_QPLUS})

set(CMAKE_C_COMPILER_TARGET aarch64)
set(CMAKE_CXX_COMPILER_TARGET aarch64)

# Set compiler flags
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER_ID_TEST_FLAGS_FIRST "-nodlink -L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -I${CUDA_ROOT}/include")

set(CMAKE_C_FLAGS " \"-V${__qnx_gcc_ver},gcc_ntoaarch64le\"")
set(CMAKE_CXX_FLAGS " \"-V${__qnx_gcc_ver},gcc_ntoaarch64le\"")
set(CMAKE_CUDA_FLAGS " --qpp-config=${__qnx_gcc_ver},gcc_ntoaarch64le")
set(AUTOMAGIC_NVCC_FLAGS --qpp-config=${__qnx_gcc_ver},gcc_ntoaarch64le CACHE STRING "automagic feature detection flags for cross build")
add_link_options("-V${__qnx_gcc_ver},gcc_ntoaarch64le")

set(CROSS_COMPILE_FOR_QNX ON CACHE BOOL "Cross compiling for QNX platforms")
string(APPEND CMAKE_CXX_FLAGS " -D_QNX_SOURCE")
string(APPEND CMAKE_CUDA_FLAGS " -D_QNX_SOURCE")
