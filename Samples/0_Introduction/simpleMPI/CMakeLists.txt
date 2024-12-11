# Include directories and libraries
include_directories(../../../Common)

find_package(MPI)

# Source file
set(SRC_FILES
    simpleMPI.cpp
    simpleMPI.cu
)

if(${MPI_FOUND})
    # Add target for simpleMPI
    add_executable(simpleMPI ${SRC_FILES})
    set_target_properties(simpleMPI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(simpleMPI PUBLIC
        ${MPI_INCLUDE_PATH}
    )

    target_link_libraries(simpleMPI PUBLIC
        ${MPI_CXX_LIBRARIES}
    )

else()
    message(STATUS "MPI not found - will not build sample 'simpleMPI'")
endif()
