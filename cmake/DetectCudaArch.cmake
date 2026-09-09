if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES is not set.\n"
        "For standalone sample builds, specify a target architecture explicitly:\n"
        "  cmake -DCMAKE_CUDA_ARCHITECTURES=<arch> <path-to-sample>\n"
        "Example: cmake -DCMAKE_CUDA_ARCHITECTURES=90 .")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/CudaSampleArchs.cmake")

# Master list of all CUDA architectures supported by this repo and CUDA 13.x.
# Each sample may define SAMPLE_DISALLOW_ARCHS to exclude architectures it does not support.
set(_master_archs ${CUDA_SAMPLES_MASTER_ARCHS})

# Compute this sample's effective arch list: master minus any disallowed archs.
set(_effective_archs ${_master_archs})
if(DEFINED SAMPLE_DISALLOW_ARCHS)
    foreach(_arch ${SAMPLE_DISALLOW_ARCHS})
        list(REMOVE_ITEM _effective_archs "${_arch}")
    endforeach()
endif()

get_filename_component(_sample_dir "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

# CMAKE_CUDA_ARCHITECTURES is set by the top-level CMakeLists.txt (defaulting to the
# full master list if the user did not specify). Build only for the intersection with
# the effective list; skip the sample if no requested arch is supported.
# CMake special keywords (all, all-major, native) are passed through as-is.
set(_passthrough_values "all" "all-major" "native")
if(CMAKE_CUDA_ARCHITECTURES IN_LIST _passthrough_values)
    set(_passthrough TRUE)
else()
    set(_passthrough FALSE)
endif()

if(NOT _passthrough)
    set(_build_archs)
    foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
        if(_arch IN_LIST _effective_archs)
            list(APPEND _build_archs "${_arch}")
        endif()
    endforeach()

    if(NOT _build_archs)
        if(NOT CUDA_SAMPLES_ARCHS_DEFAULTED)
            message(WARNING
                "Sample in directory '${_sample_dir}' skipped: "
                "user specified '${CMAKE_CUDA_ARCHITECTURES}' but this sample only supports '${_effective_archs}'")
        endif()
        set(SAMPLE_SKIP_BUILD TRUE)
    elseif(NOT "${_build_archs}" STREQUAL "${CMAKE_CUDA_ARCHITECTURES}" AND NOT CUDA_SAMPLES_ARCHS_DEFAULTED)
        message(WARNING
            "Sample in directory '${_sample_dir}': user requested some architecture(s) that are not supported. "
            "Only building for '${_build_archs}'")
        set(CMAKE_CUDA_ARCHITECTURES "${_build_archs}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "${_build_archs}")
    endif()
    unset(_build_archs)
endif()

unset(_master_archs)
unset(_effective_archs)
unset(_sample_dir)
unset(_passthrough)
unset(_passthrough_values)
