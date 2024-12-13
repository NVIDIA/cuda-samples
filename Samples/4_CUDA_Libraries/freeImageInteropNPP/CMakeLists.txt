# Include directories and libraries
include_directories(
    ../../../Common
    ../../../Common/UtilNPP
)

# Source file
set(SRC_FILES
    freeImageInteropNPP.cpp
)

find_package(FreeImage)

if(${FreeImage_FOUND})
    # Add target for freeImageInteropNPP
    add_executable(freeImageInteropNPP ${SRC_FILES})
    set_target_properties(freeImageInteropNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(freeImageInteropNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FreeImage_INCLUDE_DIRS}
    )

    target_link_libraries(freeImageInteropNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppif
        CUDA::cudart
        ${FreeImage_LIBRARIES}
    )
else()
    message(STATUS "FreeImage not found - will not build sample 'freeImageInteropNPP'")
endif()
