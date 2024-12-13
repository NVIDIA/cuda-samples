# Include directories and libraries
include_directories(
    ../../../Common
    ../../../Common/UtilNPP
)

# Source file
set(SRC_FILES
    boxFilterNPP.cpp
)

find_package(FreeImage)

if(${FreeImage_FOUND})
    # Add target for boxFilterNPP
    add_executable(boxFilterNPP ${SRC_FILES})
    set_target_properties(boxFilterNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(boxFilterNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FreeImage_INCLUDE_DIRS}
    )

    target_link_libraries(boxFilterNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppif
        CUDA::cudart
        ${FreeImage_LIBRARIES}
    )

    # Copy data files to output directory
    add_custom_command(TARGET boxFilterNPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/*.pgm
        ${CMAKE_CURRENT_BINARY_DIR}
    )
else()
    message(STATUS "FreeImage not found - will not build sample 'boxFilterNPP'")
endif()
