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

set(CMAKE_INSTALL_RPATH ${LIBNVVM_HOME})
set(CMAKE_INCLUDE_CURRENT_DIR YES)

add_executable(simple simple.c)
add_custom_target(clobber)
add_custom_target(testrun)
add_test(NAME simple COMMAND simple WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(simple ${NVVM_LIB} ${CUDA_LIB})

if (WIN32)
  set_target_properties(simple PROPERTIES COMPILE_FLAGS "/wd4996")
else (WIN32)
  set_target_properties(simple PROPERTIES
                        LINK_FLAGS "-Wl,-rpath,${LIBNVVM_RPATH}")
endif (WIN32)

install(TARGETS simple DESTINATION bin)
install(FILES simple-gpu64.ll DESTINATION bin)

# 'simple' will load simple-gpu64.ll from the current working directory. That
# .ll file should be present where tests are executed (the build directory).
file(COPY simple-gpu64.ll DESTINATION "${CMAKE_BINARY_DIR}")
