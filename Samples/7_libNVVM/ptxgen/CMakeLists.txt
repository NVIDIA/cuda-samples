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

add_executable(ptxgen ptxgen.c)

add_test(NAME ptxgenTest
   COMMAND ptxgen "${CMAKE_CURRENT_SOURCE_DIR}/test.ll"
   WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(ptxgenTest PROPERTIES FIXTURES_SETUP PTXGENTEST)

target_link_libraries(ptxgen ${NVVM_LIB})

if (DEFINED RUNTIME_LIBNVVM_PATH)
  get_filename_component(_LIBNVVM_HOME "${RUNTIME_LIBNVVM_PATH}" DIRECTORY)
else ()
  set(_LIBNVVM_HOME "${LIBNVVM_HOME}")
endif ()

if (WIN32)
  set_target_properties(ptxgen PROPERTIES
                        COMPILE_FLAGS "/wd4996 -DLIBNVVM_HOME=\"${LIBNVVM_HOME}\"")
else (WIN32)
  set_target_properties(ptxgen PROPERTIES
                        COMPILE_FLAGS "-DLIBNVVM_HOME=\"${_LIBNVVM_HOME}\""
                        LINK_FLAGS    "-Wl,-rpath,${LIBNVVM_RPATH}")
endif (WIN32)

install(TARGETS ptxgen DESTINATION bin)

add_custom_command(
    TARGET ptxgen
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/test.ll" "$<TARGET_FILE_DIR:ptxgen>"
)
if (WIN32)
  add_custom_command(
      TARGET ptxgen
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CMAKE_BINARY_DIR}/nvvm64_40_0.dll" "$<TARGET_FILE_DIR:ptxgen>"
  )
endif ()
