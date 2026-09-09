# Toolchain for QNX Safety builds with the CUDA Safe toolkit (QNX SAFE stack).
# Use this file instead of toolchain-aarch64-qnx.cmake for safe-toolkit installs.

set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(QNX_HOST $ENV{QNX_HOST})
set(QNX_TARGET $ENV{QNX_TARGET})

message(STATUS "QNX_HOST = ${QNX_HOST}")
message(STATUS "QNX_TARGET = ${QNX_TARGET}")

find_program(QNX_QCC   NAMES qcc   PATHS "${QNX_HOST}/usr/bin")
find_program(QNX_QPLUS NAMES q++   PATHS "${QNX_HOST}/usr/bin")

if(NOT QNX_QCC OR NOT QNX_QPLUS)
    message(FATAL_ERROR "Could not find qcc or q++ in QNX_HOST=${QNX_HOST}/usr/bin")
endif()

set(CMAKE_C_COMPILER ${QNX_QCC})
set(CMAKE_CXX_COMPILER ${QNX_QPLUS})

set(CMAKE_C_COMPILER_TARGET aarch64)
set(CMAKE_CXX_COMPILER_TARGET aarch64)

# Toolkit root: required so CUDA compiler ID test does not expand empty CUDA_ROOT into -I/include, etc.
if(CMAKE_CUDA_COMPILER)
    get_filename_component(_cuda_nvcc_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT "${_cuda_nvcc_dir}" DIRECTORY)
elseif(DEFINED ENV{CUDA_PATH})
    set(CUDA_TOOLKIT_ROOT "$ENV{CUDA_PATH}")
endif()
if(CUDA_TOOLKIT_ROOT)
    set(CUDA_TOOLKIT_ROOT "${CUDA_TOOLKIT_ROOT}" CACHE PATH "CUDA Toolkit root")
    set(CUDA_ROOT "${CUDA_TOOLKIT_ROOT}" CACHE PATH "CUDA Toolkit root (nvcc / legacy CMake)" FORCE)
endif()

if(NOT __qnx_gcc_ver)
    set(__qnx_gcc_ver "12.2.0" CACHE STRING "QNX qcc/q++ toolchain version for -V / --qpp-config (match your SDP)")
endif()

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)

set(_cuda_safe_target "aarch64-qnx")

# CUDA Safe toolkit flags: -safety-compat keeps the kernel-launch ABI consistent; the safe toolkit
# ships only a shared cudart (no static/cudadevrt); -I/-L point at its headers/libs;
# --unresolved-symbols defers libcudart's rootfs-resident deps (nvdvms_*, NvOs*) to runtime.
if(CUDA_TOOLKIT_ROOT)
    set(_cuda_safe_inc "${CUDA_TOOLKIT_ROOT}/targets/${_cuda_safe_target}/include")
    set(_cuda_safe_lib "${CUDA_TOOLKIT_ROOT}/targets/${_cuda_safe_target}/lib")
    set(_cuda_safe_stubs "${CUDA_TOOLKIT_ROOT}/targets/${_cuda_safe_target}/lib/stubs")
endif()

set(_cuda_safe_flags "-safety-compat --cudart=shared --cudadevrt=none -target-dir ${_cuda_safe_target} --qpp-config=${__qnx_gcc_ver},gcc_ntoaarch64le")
if(CUDA_TOOLKIT_ROOT)
    string(APPEND _cuda_safe_flags " -I${_cuda_safe_inc} -L${_cuda_safe_lib} -L${_cuda_safe_stubs}")
endif()
string(APPEND _cuda_safe_flags " -Xlinker --unresolved-symbols=ignore-in-shared-libs")

set(CMAKE_C_FLAGS " \"-V${__qnx_gcc_ver},gcc_ntoaarch64le\"")
set(CMAKE_CXX_FLAGS " \"-V${__qnx_gcc_ver},gcc_ntoaarch64le\"")
# Host .cpp TUs are built by q++ directly, so give them the CUDA include too.
if(CUDA_TOOLKIT_ROOT)
    string(APPEND CMAKE_C_FLAGS   " -I${_cuda_safe_inc}")
    string(APPEND CMAKE_CXX_FLAGS " -I${_cuda_safe_inc}")
endif()
set(CMAKE_CUDA_FLAGS " ${_cuda_safe_flags}")
separate_arguments(_automagic_nvcc_flags UNIX_COMMAND "${_cuda_safe_flags}")
set(AUTOMAGIC_NVCC_FLAGS ${_automagic_nvcc_flags}
    CACHE STRING "automagic feature detection flags for QNX Safe cross build")

# Safe toolkit has no cudadevrt/static cudart, so disable CMake's implicit runtime libs and link the
# shared cudart ourselves. Also link the driver (-lcuda): the shared libcudart.so references safety-
# stack symbols (nvdvms_*, NvOs*) that are transitive deps of libcuda.so.1, so the exe must pull in
# libcuda for them to resolve/load on the target. STANDARD_LIBRARIES_INIT covers ABI checks; the
# HOST_LINK options cover real targets.
set(CMAKE_CUDA_RUNTIME_LIBRARY "None" CACHE STRING "CUDA runtime library (safe toolkit: linked manually)")
set(CMAKE_CUDA_STANDARD_LIBRARIES_INIT "-lcudart -lcuda")
if(CUDA_TOOLKIT_ROOT)
    add_link_options("$<HOST_LINK:-L${_cuda_safe_lib}>" "$<HOST_LINK:-L${_cuda_safe_stubs}>"
                     "$<HOST_LINK:-lcudart>" "$<HOST_LINK:-lcuda>")
endif()

# -V on the host link only; nvcc would comma-split it at the device link (which uses --qpp-config).
add_link_options("$<HOST_LINK:-V${__qnx_gcc_ver}$<COMMA>gcc_ntoaarch64le>")
# Defer libcudart's rootfs-resident deps to runtime.
add_link_options("-Wl,--unresolved-symbols=ignore-in-shared-libs")

set(CROSS_COMPILE_FOR_QNX ON CACHE BOOL "Cross compiling for QNX platforms")
string(APPEND CMAKE_CXX_FLAGS " -D_QNX_SOURCE")
string(APPEND CMAKE_CUDA_FLAGS " -D_QNX_SOURCE")

# cudaNvSci needs NvSci headers/libs from the QNX safety rootfs; point TARGET_FS at it. This
# pre-seeds FindNVSCI's cache vars and auto-detects the header/lib dirs across rootfs layouts.
if(DEFINED TARGET_FS)
    # Expand a leading ~ so paths reach the linker absolute.
    if(TARGET_FS MATCHES "^~")
        string(REGEX REPLACE "^~" "$ENV{HOME}" TARGET_FS "${TARGET_FS}")
    endif()
    get_filename_component(_nvsci_sdk_root "${TARGET_FS}" DIRECTORY)

    # Header dir varies by rootfs layout.
    foreach(_inc "${TARGET_FS}/include" "${_nvsci_sdk_root}/include" "${TARGET_FS}/usr/include")
        if(NOT _nvsci_inc AND EXISTS "${_inc}/nvscibuf.h")
            set(_nvsci_inc "${_inc}")
        endif()
    endforeach()

    # Lib dir varies by rootfs layout.
    foreach(_lib "${TARGET_FS}/lib-target" "${TARGET_FS}/usr/libnvidia" "${TARGET_FS}/usr/lib")
        if(NOT _nvsci_lib AND EXISTS "${_lib}/libnvscibuf.so")
            set(_nvsci_lib "${_lib}")
        endif()
    endforeach()

    if(NOT _nvsci_inc OR NOT _nvsci_lib)
        message(FATAL_ERROR
            "Could not locate NvSci headers/libs from TARGET_FS='${TARGET_FS}'. "
            "Checked include roots for nvscibuf.h and lib roots for libnvscibuf.so. "
            "Please set TARGET_FS (or NVSCIBUF/NVSCISYNC cache vars) to a valid QNX safety rootfs.")
    endif()

    set(NVSCIBUF_INCLUDE_DIR  "${_nvsci_inc}"                  CACHE PATH     "NvSciBuf headers (QNX safe)")
    set(NVSCISYNC_INCLUDE_DIR "${_nvsci_inc}"                  CACHE PATH     "NvSciSync headers (QNX safe)")
    set(NVSCIBUF_LIBRARY      "${_nvsci_lib}/libnvscibuf.so"   CACHE FILEPATH "NvSciBuf library (QNX safe)")
    set(NVSCISYNC_LIBRARY     "${_nvsci_lib}/libnvscisync.so"  CACHE FILEPATH "NvSciSync library (QNX safe)")

    # rpath-link the rootfs lib dirs so NvSci's transitive deps resolve at link time.
    foreach(_dir "${_nvsci_lib}" "${TARGET_FS}/lib-target" "${TARGET_FS}/usr/libnvidia" "${TARGET_FS}/usr/lib")
        if(IS_DIRECTORY "${_dir}")
            set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${_dir}")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link,${_dir}")
        endif()
    endforeach()
endif()
