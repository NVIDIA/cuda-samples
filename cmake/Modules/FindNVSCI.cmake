# Find the NVSCI libraries and headers
#
# This module defines the following variables:
#  NVSCI_FOUND        - True if NVSCI was found
#  NVSCI_INCLUDE_DIRS - NVSCI include directories
#  NVSCI_LIBRARIES    - NVSCI libraries
#  NVSCIBUF_LIBRARY   - NVSCI buffer library
#  NVSCISYNC_LIBRARY  - NVSCI sync library

# Find the libraries
find_library(NVSCIBUF_LIBRARY
    NAMES nvscibuf libnvscibuf
    PATHS 
        /usr/lib
        /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
    PATH_SUFFIXES nvidia
)

find_library(NVSCISYNC_LIBRARY
    NAMES nvscisync libnvscisync
    PATHS 
        /usr/lib
        /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
    PATH_SUFFIXES nvidia
)

# Find the header files
find_path(NVSCIBUF_INCLUDE_DIR
    NAMES nvscibuf.h
    PATHS 
        /usr/include
        /usr/local/include
)

find_path(NVSCISYNC_INCLUDE_DIR
    NAMES nvscisync.h
    PATHS 
        /usr/include
        /usr/local/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSCI
    REQUIRED_VARS 
        NVSCIBUF_LIBRARY
        NVSCISYNC_LIBRARY
        NVSCIBUF_INCLUDE_DIR
        NVSCISYNC_INCLUDE_DIR
)

if(NVSCI_FOUND)
    set(NVSCI_LIBRARIES ${NVSCIBUF_LIBRARY} ${NVSCISYNC_LIBRARY})
    set(NVSCI_INCLUDE_DIRS ${NVSCIBUF_INCLUDE_DIR} ${NVSCISYNC_INCLUDE_DIR})
endif()

mark_as_advanced(
    NVSCIBUF_LIBRARY
    NVSCISYNC_LIBRARY
    NVSCIBUF_INCLUDE_DIR
    NVSCISYNC_INCLUDE_DIR
) 
