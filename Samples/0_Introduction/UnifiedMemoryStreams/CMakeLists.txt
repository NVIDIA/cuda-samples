# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    UnifiedMemoryStreams.cu
)

find_package(OpenMP REQUIRED)

if(${OpenMP_FOUND})
    # Add target for UnifiedMemoryStreams
    add_executable(UnifiedMemoryStreams ${SRC_FILES})
    set_target_properties(UnifiedMemoryStreams PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(UnifiedMemoryStreams PUBLIC
        CUDA::cublas
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "OpenMP not found - will not build sample 'UnifiedMemoryStreams'")
endif()
