include_guard(GLOBAL)

get_filename_component(_cuda_sample_cmake_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)

list(APPEND CMAKE_MODULE_PATH "${_cuda_sample_cmake_dir}")
list(APPEND CMAKE_MODULE_PATH "${_cuda_sample_cmake_dir}/Modules")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

unset(_cuda_sample_cmake_dir)
