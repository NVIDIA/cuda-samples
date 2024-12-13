# Include directories and libraries
include_directories(
    ../../../Common
    ../../../Common/UtilNPP
)

# Source file
set(SRC_FILES
    FilterBorderControlNPP.cpp
)


find_package(FreeImage)

if(${FreeImage_FOUND})
    # Add target for FilterBorderControlNPP
    add_executable(FilterBorderControlNPP ${SRC_FILES})
    set_target_properties(FilterBorderControlNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(FilterBorderControlNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FreeImage_INCLUDE_DIRS}
    )

    target_link_libraries(FilterBorderControlNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppif
        CUDA::nppitc
        CUDA::nppidei
        CUDA::nppial
        CUDA::cudart
        ${FreeImage_LIBRARIES}
    )

    # Copy data files to output directory
    add_custom_command(TARGET FilterBorderControlNPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    )
else()
    message(STATUS "FreeImage not found - will not build sample 'FilterBorderControlNPP'")
endif()
