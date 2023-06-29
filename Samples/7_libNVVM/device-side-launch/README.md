Device-Side Launch From NVVM IR
===============================

This document is for the programming language and compiler implementers who
target NVVM IR and plan to support Dynamic Parallelism in their langauge.
It provides the low-level details related to supporting kernel launches at
the NVVM IR level.

This document assumes the CUDA runtime is used. The method for device-side
launch using the OpenCL runtime is similar but different.

This document is written after the "Device-Side Launch from PTX"
section from CUDA C Programming Guide
(http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-side-launch-from-ptx).

Kernel Launch APIs
------------------

Device-side kernel launches can be implemented using the following two APIs 
in an NVVM IR program: cudaLaunchDevice() and cudaGetParameterBuffer().
cudaLaunchDevice() launches the specified kernel with the parameter buffer
that is obtained by calling cudaGetParameterBuffer() and filled with the
parameters to the launched kernel. The parameter buffer can be NULL, i.e.,
no need to invoke cudaGetParameterBuffer(), if the launched kernel does not
take any parameters.

cudaLaunchDevice
----------------

At the NVVM IR level, cudaLaunchDeviceV2() needs to be declared in the
form shown below before it is used.

    ; NVVM IR level declaration of cudaLaunchDeviceV2
    declare i32 @cudaLaunchDeviceV2(i8*, %struct.CUstream_st*)

The CUDA-level declaration below is mapped to one of the aftorementioned NVVM
IR level declarations and is found in the system header file 
cuda_device_runtime_api.h. The function is defined in the cudadevrt system 
library, which must be linked with a program in order to use device-side
kernel launch functionality.

    extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
    cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream);

The first parameter is a pointer to the parameter buffer, and the
second parameter is the stream associated with the launch. The layout
of the parameter buffer is explained in "Parameter Buffer Layout"
below.

cudaGetParameterBuffer
----------------------

cudaGetParameterBufferV2() needs to be declared at the NVVM IR level
before it's used. The NVVM IR level declaration must be in the form
given below:

    ; NVVM IR level declaration of cudaGetParameterBufferV2
    declare i8* @cudaGetParameterBufferV2(i8*, %struct.dim3, %struct.dim3, i32)

The following CUDA-level declaration of cudaGetParameterBufferV2() is
mapped to the aforementioned NVVM IR level declaration:

    extern __device__ __cudart_builtin__ void * CUDARTAPI 
    cudaGetParameterBufferV2(void *func, dim3 gridDimension, 
                             dim3 blockDimension,
                             unsigned int sharedMemSize);

The first parameter is a pointer to the kernel to be launched, and the
other parameters specify the launch configuration, i.e., as grid
dimension, block dimension, and shared memory size. 

Parameter Buffer Layout
-----------------------

Parameter reordering in the parameter buffer is prohibited, and each individual 
parameter placed in the parameter buffer is required to be aligned. That is, 
each parameter must be placed at the n-th byte in the parameter buffer, where n
is the smallest multiple of the parameter size that is greater than the offset 
of the last byte taken by the preceding parameter. The maximum size of the 
parameter buffer is 4KB.
