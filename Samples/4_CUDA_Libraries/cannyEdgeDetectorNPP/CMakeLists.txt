# Include directories and libraries
include_directories(
    ../../../Common
    ../../../Common/UtilNPP
)

# Source file
set(SRC_FILES
    cannyEdgeDetectorNPP.cpp
)

find_package(FreeImage)

if(${FreeImage_FOUND})
    # Add target for cannyEdgeDetectorNPP
    add_executable(cannyEdgeDetectorNPP ${SRC_FILES})
    set_target_properties(cannyEdgeDetectorNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(cannyEdgeDetectorNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FreeImage_INCLUDE_DIRS}
    )

    target_link_libraries(cannyEdgeDetectorNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppif
        CUDA::cudart
        ${FreeImage_LIBRARIES}
    )

    # Copy data files to output directory
    add_custom_command(TARGET cannyEdgeDetectorNPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/*.pgm
        ${CMAKE_CURRENT_BINARY_DIR}
    )
else()
    message(STATUS "FreeImage not found - will not build sample 'cannyEdgeDetectorNPP'")
endif()
