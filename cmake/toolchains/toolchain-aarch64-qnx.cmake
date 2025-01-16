#==============================================================================
# Toolchain file for cross-compiling to aarch64 QNX
#==============================================================================

# Cross-compiling, so tell CMake that we are not building for the host system
set(CMAKE_SYSTEM_NAME QNX)

# Target processor architecture
set(CMAKE_SYSTEM_PROCESSOR aarch64)

#------------------------------------------------------------------------------
# QNX host and target come from environment
# Adjust these or hard-code paths as needed:
#
#    set(QNX_HOST "/path/to/qnx/host")      # e.g. /qnx/qnx710/host/linux/x86_64
#    set(QNX_TARGET "/path/to/qnx/target")  # e.g. /qnx/qnx710/target/qnx7
#
# You can also pass them on the cmake command line:
#    cmake -D QNX_HOST=/path/to/qnx/host \
#          -D QNX_TARGET=/path/to/qnx/target \
#          -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64-qnx.cmake ..
#------------------------------------------------------------------------------

#----------------------------------------------------------------------------
# C/C++ Compilers from QNX
#----------------------------------------------------------------------------
find_program(QNX_QCC   NAMES qcc   PATHS "${QNX_HOST}/usr/bin")
find_program(QNX_QPLUS NAMES q++   PATHS "${QNX_HOST}/usr/bin")

if(NOT QNX_QCC OR NOT QNX_QPLUS)
    message(FATAL_ERROR "Could not find qcc or q++ in QNX_HOST=${QNX_HOST}/usr/bin")
endif()

set(CMAKE_C_COMPILER   "${QNX_QCC}")
set(CMAKE_CXX_COMPILER "${QNX_QPLUS}")

#----------------------------------------------------------------------------
# Sysroot (if you want CMake to know the default sysroot)
#----------------------------------------------------------------------------
# This is optional, but convenient if the QNX headers/libraries must be found:
#----------------------------------------------------------------------------
if(DEFINED QNX_TARGET)
    set(CMAKE_SYSROOT "${QNX_TARGET}")
endif()

#----------------------------------------------------------------------------
# Additional preprocessor definitions & include paths
#----------------------------------------------------------------------------
add_compile_options(
    -D_QNX_SOURCE
    -DWIN_INTERFACE_CUSTOM
)

# Add an include path to /usr/include/aarch64-qnx-gnu:
include_directories("/usr/include/aarch64-qnx-gnu")

#----------------------------------------------------------------------------
# Linker flags
#----------------------------------------------------------------------------
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/lib/aarch64-qnx-gnu")

# Because the Makefile also adds -Wl,-rpath-link,/usr/lib/aarch64-qnx-gnu:
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,/usr/lib/aarch64-qnx-gnu")

# If you have a “target filesystem” (TARGET_FS) to link with:
#   -L$(TARGET_FS)/usr/lib
#   -L$(TARGET_FS)/usr/libnvidia
# etc., you can optionally extend the link flags.  For example:
#
# if(DEFINED TARGET_FS)
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} \
#         -L${TARGET_FS}/usr/lib -Wl,-rpath-link,${TARGET_FS}/usr/lib \
#         -L${TARGET_FS}/usr/libnvidia -Wl,-rpath-link,${TARGET_FS}/usr/libnvidia")
#     include_directories("${TARGET_FS}/../include")
# endif()

# If you need to link additional libraries, e.g. -lslog2 under certain conditions:
#   list(APPEND EXTRA_LIBS "slog2")
#   ...
#----------------------------------------------------------------------------
