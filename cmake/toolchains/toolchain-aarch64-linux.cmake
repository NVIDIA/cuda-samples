set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross-compilers
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_AR aarch64-linux-gnu-ar)
set(CMAKE_RANLIB aarch64-linux-gnu-ranlib)

# Indicate cross-compiling.
set(CMAKE_CROSSCOMPILING TRUE)

# Set CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)

# Use a local sysroot copy
if(DEFINED TARGET_FS)
    # The aarch64/sbsa_aarch64 CUDA toolkit are support on Tegra since 13.0, so need to check which version of the toolkit is installed
    set(CUDA_AARCH64_TARGET "aarch64-linux")
    if(NOT EXISTS "/usr/local/cuda/targets/${CUDA_AARCH64_TARGET}")
        set(CUDA_AARCH64_TARGET "sbsa-linux")
    endif()

    set(CMAKE_SYSROOT "${TARGET_FS}")
    list(APPEND CMAKE_FIND_ROOT_PATH
        "/usr/local/cuda/targets/${CUDA_AARCH64_TARGET}"
    )

    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler --sysroot=${TARGET_FS}")

    set(LIB_PATHS
        "${TARGET_FS}/usr/lib/"
        "${TARGET_FS}/usr/lib/aarch64-linux-gnu"
        "${TARGET_FS}/usr/lib/aarch64-linux-gnu/nvidia"
    )
    # Add rpath-link flags for all library paths
    foreach(lib_path ${LIB_PATHS})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${lib_path}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link,${lib_path}")
    endforeach()

    # Add the real path of CUDA installation on TARGET_FS for nvvm
    find_program(TARGET_CUDA_NVCC_PATH nvcc 
    PATH "${TARGET_FS}/usr/local/cuda/bin"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    )
    if(TARGET_CUDA_NVCC_PATH)
        # Get the real path of CUDA installation on TARGET_FS
        get_filename_component(TARGET_CUDA_PATH "${TARGET_CUDA_NVCC_PATH}" REALPATH)
        get_filename_component(TARGET_CUDA_ROOT "${TARGET_CUDA_PATH}" DIRECTORY)
        get_filename_component(TARGET_CUDA_ROOT "${TARGET_CUDA_ROOT}" DIRECTORY)
    endif()

    if (DEFINED TARGET_CUDA_ROOT)
        list(APPEND CMAKE_LIBRARY_PATH "${TARGET_CUDA_ROOT}/targets/${CUDA_AARCH64_TARGET}/lib")
        # Define NVVM paths for build and runtime
        set(ENV{LIBNVVM_HOME} "${TARGET_CUDA_ROOT}")
        set(RUNTIME_LIBNVVM_PATH "${TARGET_CUDA_ROOT}/nvvm/lib64")
    endif()
endif()
