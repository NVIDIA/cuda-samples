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

# cudaNvSci needs NvSci headers/libs from the QNX rootfs; point TARGET_FS at it. This pre-seeds
# FindNVSCI's cache vars and auto-detects the header/lib dirs across rootfs layouts.
if(DEFINED TARGET_FS)
    # Expand a leading ~ so paths reach the linker absolute.
    if(TARGET_FS MATCHES "^~")
        string(REGEX REPLACE "^~" "$ENV{HOME}" TARGET_FS "${TARGET_FS}")
    endif()
    get_filename_component(_nvsci_sdk_root "${TARGET_FS}" DIRECTORY)

    # Header dir varies by rootfs layout. Both headers are required, so accept a dir only if it
    # holds both.
    foreach(_inc "${TARGET_FS}/include" "${_nvsci_sdk_root}/include" "${TARGET_FS}/usr/include")
        if(NOT _nvsci_inc AND EXISTS "${_inc}/nvscibuf.h" AND EXISTS "${_inc}/nvscisync.h")
            set(_nvsci_inc "${_inc}")
        endif()
    endforeach()

    # Lib dir varies by rootfs layout. FindNVSCI needs both libraries, so accept a dir only if it
    # holds both.
    foreach(_lib "${TARGET_FS}/lib-target" "${TARGET_FS}/usr/libnvidia" "${TARGET_FS}/usr/lib")
        if(NOT _nvsci_lib AND EXISTS "${_lib}/libnvscibuf.so" AND EXISTS "${_lib}/libnvscisync.so")
            set(_nvsci_lib "${_lib}")
        endif()
    endforeach()

    # A rootfs without NvSci is not an error: FindNVSCI then reports NvSci as missing and only the
    # NvSci samples are skipped.
    if(_nvsci_inc AND _nvsci_lib)
        set(NVSCIBUF_INCLUDE_DIR  "${_nvsci_inc}"                  CACHE PATH     "NvSciBuf headers")
        set(NVSCISYNC_INCLUDE_DIR "${_nvsci_inc}"                  CACHE PATH     "NvSciSync headers")
        set(NVSCIBUF_LIBRARY      "${_nvsci_lib}/libnvscibuf.so"   CACHE FILEPATH "NvSciBuf library")
        set(NVSCISYNC_LIBRARY     "${_nvsci_lib}/libnvscisync.so"  CACHE FILEPATH "NvSciSync library")

        # rpath-link the rootfs lib dirs so NvSci's transitive deps resolve at link time.
        # NvSci pulls in NvRm/NvOs/NvSciIpc, which can sit in a lib dir of their own.
        foreach(_dir "${_nvsci_lib}" "${TARGET_FS}/lib-target" "${_nvsci_sdk_root}/lib-target"
                     "${TARGET_FS}/usr/libnvidia" "${TARGET_FS}/usr/lib")
            if(IS_DIRECTORY "${_dir}")
                set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${_dir}")
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link,${_dir}")
            endif()
        endforeach()
    else()
        message(STATUS "NvSci headers/libs not found in TARGET_FS='${TARGET_FS}'")
    endif()
endif()
