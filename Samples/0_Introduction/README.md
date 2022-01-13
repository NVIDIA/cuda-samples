# 0. Introduction


### [asyncAPI](./asyncAPI)
This sample illustrates the usage of CUDA events for both GPU timing and overlapping CPU and GPU execution. Events are inserted into a stream of CUDA calls. Since CUDA stream calls are asynchronous, the CPU can perform computations while GPU is executing (including DMA memcopies between the host and device). CPU can query CUDA events to determine whether GPU has completed tasks.

### [c++11_cuda](./c++11_cuda)
This sample demonstrates C++11 feature support in CUDA. It scans a input text file and prints no. of occurrences of x, y, z, w characters. 

### [clock](./clock)
This example shows how to use the clock function to measure the performance of block of threads of a kernel accurately.

### [clock_nvrtc](./clock_nvrtc)
This example shows how to use the clock function using libNVRTC to measure the performance of block of threads of a kernel accurately.

### [concurrentKernels](./concurrentKernels)
This sample demonstrates the use of CUDA streams for concurrent execution of several kernels on GPU device. It also illustrates how to introduce dependencies between CUDA streams with the new cudaStreamWaitEvent function.

### [cppIntegration](./cppIntegration)
This example demonstrates how to integrate CUDA into an existing C++ application, i.e. the CUDA entry point on host side is only a function which is called from C++ code and only the file containing this function is compiled with nvcc. It also demonstrates that vector types can be used from cpp.

### [cppOverload](./cppOverload)
This sample demonstrates how to use C++ function overloading on the GPU.

### [cudaOpenMP](./cudaOpenMP)
This sample demonstrates how to use OpenMP API to write an application for multiple GPUs.

### [fp16ScalarProduct](./fp16ScalarProduct)
Calculates scalar product of two vectors of FP16 numbers.

### [matrixMul](./matrixMul)
This sample implements matrix multiplication and is exactly the same as Chapter 6 of the programming guide. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication.  To illustrate GPU performance for matrix multiply, this sample also shows how to use the new CUDA 4.0 interface for CUBLAS to demonstrate high-performance performance for matrix multiplication.

### [matrixMul_nvrtc](./matrixMul_nvrtc)
This sample implements matrix multiplication and is exactly the same as Chapter 6 of the programming guide. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication.  To illustrate GPU performance for matrix multiply, this sample also shows how to use the new CUDA 4.0 interface for CUBLAS to demonstrate high-performance performance for matrix multiplication.

### [matrixMulDrv](./matrixMulDrv)
This sample implements matrix multiplication and uses the new CUDA 4.0 kernel launch Driver API. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication. CUBLAS provides high-performance matrix multiplication.

### [matrixMulDynlinkJIT](./matrixMulDynlinkJIT)
This sample revisits matrix multiplication using the CUDA driver API. It demonstrates how to link to CUDA driver at runtime and how to use JIT (just-in-time) compilation from PTX code. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication. CUBLAS provides high-performance matrix multiplication.

### [mergeSort](./mergeSort)
This sample implements a merge sort (also known as Batcher's sort), algorithms belonging to the class of sorting networks. While generally subefficient on large sequences compared to algorithms with better asymptotic algorithmic complexity (i.e. merge sort or radix sort), may be the algorithms of choice for sorting batches of short- to mid-sized (key, value) array pairs. Refer to the excellent tutorial by H. W. Lang http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm

### [simpleAssert](./simpleAssert)
This CUDA Runtime API sample is a very basic sample that implements how to use the assert function in the device code. Requires Compute Capability 2.0 .

### [simpleAssert_nvrtc](./simpleAssert_nvrtc)
This CUDA Runtime API sample is a very basic sample that implements how to use the assert function in the device code. Requires Compute Capability 2.0 .

### [simpleAtomicIntrinsics](./simpleAtomicIntrinsics)
A simple demonstration of global memory atomic instructions.

### [simpleAtomicIntrinsics_nvrtc](./simpleAtomicIntrinsics_nvrtc)
A simple demonstration of global memory atomic instructions.This sample makes use of NVRTC for Runtime Compilation.

### [simpleAttributes](./simpleAttributes)
This CUDA Runtime API sample is a very basic example that implements how to use the stream attributes that affect L2 locality. Performance improvement due to use of L2 access policy window can only be noticed on Compute capability 8.0 or higher.

### [simpleAWBarrier](./simpleAWBarrier)
A simple demonstration of arrive wait barriers.

### [simpleCallback](./simpleCallback)
This sample implements multi-threaded heterogeneous computing workloads with the new CPU callbacks for CUDA streams and events introduced with CUDA 5.0.

### [simpleCooperativeGroups](./simpleCooperativeGroups)
This sample is a simple code that illustrates basic usage of cooperative groups within the thread block.

### [simpleCubemapTexture](./simpleCubemapTexture)
Simple example that demonstrates how to use a new CUDA 4.1 feature to support cubemap Textures in CUDA C.

### [simpleCUDA2GL](./simpleCUDA2GL)
This sample shows how to copy CUDA image back to OpenGL using the most efficient methods.

### [simpleDrvRuntime](./simpleDrvRuntime)
A simple example which demonstrates how CUDA Driver and Runtime APIs can work together to load cuda fatbinary of vector add kernel and performing vector addition.

### [simpleHyperQ](./simpleHyperQ)
This sample demonstrates the use of CUDA streams for concurrent execution of several kernels on devices which provide HyperQ (SM 3.5).  Devices without HyperQ (SM 2.0 and SM 3.0) will run a maximum of two kernels concurrently.

### [simpleIPC](./simpleIPC)
This CUDA Runtime API sample is a very basic sample that demonstrates Inter Process Communication with one process per GPU for computation.  Requires Compute Capability 3.0 or higher and a Linux Operating System, or a Windows Operating System with TCC enabled GPUs

### [simpleLayeredTexture](./simpleLayeredTexture)
Simple example that demonstrates how to use a new CUDA 4.0 feature to support layered Textures in CUDA C.

### [simpleMPI](./simpleMPI)
Simple example demonstrating how to use MPI in combination with CUDA.

### [simpleMultiCopy](./simpleMultiCopy)
Supported in GPUs with Compute Capability 1.1, overlapping compute with one memcopy is possible from the host system.  For Quadro and Tesla GPUs with Compute Capability 2.0, a second overlapped copy operation in either direction at full speed is possible (PCI-e is symmetric).  This sample illustrates the usage of CUDA streams to achieve overlapping of kernel execution with data copies to and from the device.

### [simpleMultiGPU](./simpleMultiGPU)
This application demonstrates how to use the new CUDA 4.0 API for CUDA context management and multi-threaded access to run CUDA kernels on multiple-GPUs.

### [simpleOccupancy](./simpleOccupancy)
This sample demonstrates the basic usage of the CUDA occupancy calculator and occupancy-based launch configurator APIs by launching a kernel with the launch configurator, and measures the utilization difference against a manually configured launch.

### [simpleP2P](./simpleP2P)
This application demonstrates CUDA APIs that support Peer-To-Peer (P2P) copies, Peer-To-Peer (P2P) addressing, and Unified Virtual Memory Addressing (UVA) between multiple GPUs. In general, P2P is supported between two same GPUs with some exceptions, such as some Tesla and Quadro GPUs.

### [simplePitchLinearTexture](./simplePitchLinearTexture)
Use of Pitch Linear Textures

### [simplePrintf](./simplePrintf)
This basic CUDA Runtime API sample demonstrates how to use the printf function in the device code.

### [simpleSeparateCompilation](./simpleSeparateCompilation)
This sample demonstrates a CUDA 5.0 feature, the ability to create a GPU device static library and use it within another CUDA kernel.  This example demonstrates how to pass in a GPU device function (from the GPU device static library) as a function pointer to be called.  This sample requires devices with compute capability 2.0 or higher.

### [simpleStreams](./simpleStreams)
This sample uses CUDA streams to overlap kernel executions with memory copies between the host and a GPU device.  This sample uses a new CUDA 4.0 feature that supports pinning of generic host memory.  Requires Compute Capability 2.0 or higher.

### [simpleSurfaceWrite](./simpleSurfaceWrite)
Simple example that demonstrates the use of 2D surface references (Write-to-Texture)

### [simpleTemplates](./simpleTemplates)
This sample is a templatized version of the template project. It also shows how to correctly templatize dynamically allocated shared memory arrays.

### [simpleTemplates_nvrtc](./simpleTemplates_nvrtc)
This sample is a templatized version of the template project. It also shows how to correctly templatize dynamically allocated shared memory arrays.

### [simpleTexture](./simpleTexture)
Simple example that demonstrates use of Textures in CUDA.

### [simpleTexture3D](./simpleTexture3D)
Simple example that demonstrates use of 3D Textures in CUDA.

### [simpleTextureDrv](./simpleTextureDrv)
Simple example that demonstrates use of Textures in CUDA.  This sample uses the new CUDA 4.0 kernel launch Driver API.

### [simpleVoteIntrinsics](./simpleVoteIntrinsics)
Simple program which demonstrates how to use the Vote (__any_sync, __all_sync) intrinsic instruction in a CUDA kernel.

### [simpleVoteIntrinsics_nvrtc](./simpleVoteIntrinsics_nvrtc)
Simple program which demonstrates how to use the Vote (any, all) intrinsic instruction in a CUDA kernel with runtime compilation using NVRTC APIs. Requires Compute Capability 2.0 or higher.

### [simpleZeroCopy](./simpleZeroCopy)
This sample illustrates how to use Zero MemCopy, kernels can read and write directly to pinned system memory.

### [systemWideAtomics](./systemWideAtomics)
A simple demonstration of system wide atomic instructions.

### [template](./template)
A trivial template project that can be used as a starting point to create new CUDA projects.

### [UnifiedMemoryStreams](./UnifiedMemoryStreams)
This sample demonstrates the use of OpenMP and streams with Unified Memory on a single GPU.

### [vectorAdd](./vectorAdd)
This CUDA Runtime API sample is a very basic sample that implements element by element vector addition. It is the same as the sample illustrating Chapter 3 of the programming guide with some additions like error checking.

### [vectorAdd_nvrtc](./vectorAdd_nvrtc)
This CUDA Driver API sample uses NVRTC for runtime compilation of vector addition kernel. Vector addition kernel demonstrated is the same as the sample illustrating Chapter 3 of the programming guide.

### [vectorAddDrv](./vectorAddDrv)
This Vector Addition sample is a basic sample that is implemented element by element.  It is the same as the sample illustrating Chapter 3 of the programming guide with some additions like error checking.   This sample also uses the new CUDA 4.0 kernel launch Driver API.

### [vectorAddMMAP](./vectorAddMMAP)
This sample replaces the device allocation in the vectorAddDrv sample with cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap api allows the user to specify the physical properties of their memory while retaining the contiguous nature of their access, thus not requiring a change in their program structure.

