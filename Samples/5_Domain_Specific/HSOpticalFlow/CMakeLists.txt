# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    flowCUDA.cu
    flowGold.cpp
    main.cpp
)

# Add target for HSOpticalFlow
add_executable(HSOpticalFlow ${SRC_FILES})
set_target_properties(HSOpticalFlow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(HSOpticalFlow PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_custom_command(TARGET HSOpticalFlow POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)
