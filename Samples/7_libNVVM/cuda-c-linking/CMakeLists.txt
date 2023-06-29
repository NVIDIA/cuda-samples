# Copyright (c) 1993-2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

message(STATUS "Found LLVM Version ${LLVM_PACKAGE_VERSION}")
if (LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "15" OR
    LLVM_PACKAGE_VERSION VERSION_LESS "7")
  message(STATUS "The cuda-c-linking sample is expected to build with "
                 "LLVM development libraries v7 to v14, opaque pointers are "
                 "not yet supported in libNVVM.")
  return()
endif ()

add_executable(cuda-c-linking cuda-c-linking.cpp)
add_custom_target(clobber)
add_custom_target(testrun)
add_test(NAME cuda-c-linking
   COMMAND cuda-c-linking
   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(cuda-c-linking ${NVVM_LIB} ${CUDA_LIB})

# See https://llvm.org/docs/CMake.html#developing-llvm-passes-out-of-source
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(llvm_libs core support)
target_link_libraries(cuda-c-linking ${llvm_libs})

if (UNIX)
  set_target_properties(cuda-c-linking PROPERTIES
                        LINK_FLAGS "-Wl,-rpath,${LIBNVVM_RPATH}")
endif()

##############################
### Math Lib
##############################

### The math library can be built to support a number of GPU platforms using the
### common compute_50 architecture.
### For a collection of specific architecutres, the compiler options
### "-gencode=compute_XX,code=sm_XX"... can be used multiple times in the same
### nvcc invocation.
### The result is bundled into a system library using the 'nvcc -lib' feature.
### >> nvcc -m64 -arch=compute_50 -dc math-funcs.cu -o math-funcs64.o
### >> nvcc -m64 -lib math-funcs64.o -o libmathfuncs64.a

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc") # -dc: Build for device only.
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18.0)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif ()
if (WIN32)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
endif ()
add_library(mathfuncs64 STATIC math-funcs.cu)
set_target_properties(mathfuncs64 PROPERTIES PREFIX "lib"
                      OUTPUT_NAME "mathfuncs64"
                      SUFFIX ".a" CUDA_SEPERABLE_COMPILATION ON)
install(TARGETS cuda-c-linking mathfuncs64 DESTINATION bin)
