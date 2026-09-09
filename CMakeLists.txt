cmake_minimum_required(VERSION 3.20)

include(cmake/CudaSampleArchs.cmake)

# Default to all supported architectures if the user does not specify.
# Gated on CUDA_SAMPLES_ARCHS_DEFAULTED rather than CMAKE_CUDA_ARCHITECTURES so this
# decision is made once (on the first configure) and does not flip on reconfigure,
# when CMAKE_CUDA_ARCHITECTURES is already defined from our own cached default.
# The decision is revisited when the list itself moves, which the user can do with -D
# on any later configure of the same build directory. Keep the list each configure
# ends up with, so that such a change is told apart from our own cached default.
if(NOT DEFINED CUDA_SAMPLES_ARCHS_DEFAULTED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        list(JOIN CUDA_SAMPLES_MASTER_ARCHS ";" _default_archs)
        set(CMAKE_CUDA_ARCHITECTURES "${_default_archs}" CACHE STRING "CUDA architectures")
        unset(_default_archs)
        set(CUDA_SAMPLES_ARCHS_DEFAULTED TRUE CACHE INTERNAL "")
    else()
        set(CUDA_SAMPLES_ARCHS_DEFAULTED FALSE CACHE INTERNAL "")
    endif()
elseif(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "${CUDA_SAMPLES_LAST_CONFIGURED_ARCHS}")
    set(CUDA_SAMPLES_ARCHS_DEFAULTED FALSE CACHE INTERNAL "")
endif()

set(CUDA_SAMPLES_LAST_CONFIGURED_ARCHS "${CMAKE_CUDA_ARCHITECTURES}" CACHE INTERNAL "")

project(cuda-samples LANGUAGES C CXX CUDA)

include(cmake/CudaSampleCommon.cmake)

add_subdirectory(cpp)
