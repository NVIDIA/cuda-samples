# Include directories and libraries
include_directories(
    ../../../Common
    ../../../Common/UtilNPP
)

# Source file
set(SRC_FILES
    histEqualizationNPP.cpp
)

find_package(FreeImage)

if(${FreeImage_FOUND})
    # Add target for histEqualizationNPP
    add_executable(histEqualizationNPP ${SRC_FILES})
    set_target_properties(histEqualizationNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(histEqualizationNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FreeImage_INCLUDE_DIRS}
    )

    target_link_libraries(histEqualizationNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppist
        CUDA::nppicc
        CUDA::cudart
        ${FreeImage_LIBRARIES}
    )
else()
    message(STATUS "FreeImage not found - will not build sample 'histEqualizationNPP'")
endif()
