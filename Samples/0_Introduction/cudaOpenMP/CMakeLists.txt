# Include directories and libraries
include_directories(../../../Common)

find_package(OpenMP)

# Source file
set(SRC_FILES
    cudaOpenMP.cu
)

if(OpenMP_CXX_FOUND)
# Add target for asyncAPI
    add_executable(cudaOpenMP ${SRC_FILES})
    set_target_properties(cudaOpenMP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(cudaOpenMP PUBLIC
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "OpenMP not found - will not build sample 'cudaOpenMP'")
endif()
