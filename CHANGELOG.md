## Changelog

### CUDA 11.5
* Added `cuDLAHybridMode`. Demonstrate usage of cuDLA in hybrid mode.
* Added `cuDLAStandaloneMode`. Demonstrate usage of cuDLA in standalone mode.
* Added `cuDLAErrorReporting`. Demonstrate DLA error detection via CUDA.
* Added `graphMemoryNodes`. Demonstrates memory allocations and frees within CUDA graphs using Graph APIs and Stream Capture APIs.
* Added `graphMemoryFootprint`. Demonstrates how graph memory nodes re-use virtual addresses and physical memory.
* All samples from CUDA toolkit are now available on [GitHub](https://github.com/nvidia/cuda-samples).

### CUDA 11.4 update 1
* Added support for VS Code on linux platform.

### CUDA 11.4
* Added `cdpQuadtree`. Demonstrates Quad Trees implementation using CUDA Dynamic Parallelism.
* Updated `simpleVulkan`, `simpleVulkanMMAP` and `vulkanImageCUDA`. Demonstrates use of SPIR-V shaders.

### CUDA 11.3
*  Added `streamOrderedAllocationIPC`. Demonstrates Inter Process Communication using one process per GPU for computation.
*  Added `simpleCUBLAS_LU`. Demonstrates batched matrix LU decomposition using cuBLAS API `cublas<t>getrfBatched()`
*  Updated `simpleVulkan`. Demonstrates use of timeline semaphore.
*  Updated multiple samples to use pinned memory using `cudaMallocHost()`.

### CUDA 11.2
*  Added `streamOrderedAllocation`. Demonstrates stream ordered memory allocation on a GPU using cudaMallocAsync and cudaMemPool family of APIs.
*  Added `streamOrderedAllocationP2P`. Demonstrates peer-to-peer access of stream ordered memory allocated using cudaMallocAsync and cudaMemPool family of APIs.
*  Dropped Visual Studio 2015 support from all the windows supported samples.
*  FreeImage is no longer distributed with the CUDA Samples. On Windows, see the [Dependencies](./README.md#freeimage) section for more details on how to set up FreeImage. On Linux, it is recommended to install FreeImage with your distribution's package manager.
*  All the samples using CUDA Pipeline & Arrive-wait barriers are been updated to use new `cuda::pipeline` and `cuda::barrier` interfaces.
*  Updated all the samples to build with parallel build option `--threads` of `nvcc` cuda compiler.
*  Added `cudaNvSciNvMedia`. Demonstrates CUDA-NvMedia interop via NvSciBuf/NvSciSync APIs.
*  Added `simpleGL`. Demonstrates interoperability between CUDA and OpenGL.

### CUDA 11.1
*  Added `watershedSegmentationNPP`. Demonstrates how to use the NPP watershed segmentation function.
*  Added `batchedLabelMarkersAndLabelCompressionNPP`. Demonstrates how to use the NPP label markers generation and label compression functions based on a Union Find (UF) algorithm including both single image and batched image versions.
*  Dropped Visual Studio 2012, 2013 support from all the windows supported samples.
*  Added kernel performing warp aggregated atomic max in multi buckets using cg::labeled_partition & cg::reduce in `warpAggregatedAtomicsCG`.
*  Added extended CG shuffle mechanics to `shfl_scan` sample.
*  Added `cudaOpenMP`. Demonstrates how to use OpenMP API to write an application for multiple GPUs.
*  Added `simpleZeroCopy`. Demonstrates how to use zero copy, kernels can read and write directly to pinned system memory.

### CUDA 11.0
*  Added `dmmaTensorCoreGemm`. Demonstrates double precision GEMM computation using the Double precision Warp Matrix Multiply and Accumulate (WMMA) API introduced with CUDA 11 in Ampere chip family tensor cores.
*  Added `bf16TensorCoreGemm`. Demonstrates __nv_bfloat16 (e8m7) GEMM computation using the __nv_bfloat16 WMMA API introduced with CUDA 11 in Ampere chip family tensor cores.
*  Added `tf32TensorCoreGemm`. Demonstrates tf32 (e8m10) GEMM computation using the tf32 WMMA API introduced with CUDA 11 in Ampere chip family tensor cores.
*  Added `globalToShmemAsyncCopy`. Demonstrates async copy of data from global to shared memory when on compute capability 8.0 or higher. Also demonstrates arrive-wait barrier for synchronization.
*  Added `simpleAWBarrier`. Demonstrates arrive wait barriers.
*  Added `simpleAttributes`. Demonstrates the stream attributes that affect L2 locality.
*  Added warp aggregated atomic multi bucket increments kernel using labeled_partition cooperative groups in `warpAggregatedAtomicsCG` which can be used on compute capability 7.0 and above GPU architectures.
*  Added `binaryPartitionCG`. Demonstrates  binary partition cooperative groups and reduction within the thread block.
*  Added two new reduction kernels in `reduction` one which demonstrates reduce_add_sync intrinstic supported on compute capability 8.0 and another which uses cooperative_groups::reduce function which does thread_block_tile level reduction introduced from CUDA 11.0.
*  Added `cudaCompressibleMemory`. Demonstrates compressible memory allocation using cuMemMap API.
*  Added `simpleVulkanMMAP`. Demonstrates Vulkan CUDA Interop via cuMemMap APIs.
*  Added `concurrentKernels`. Demonstrates the use of CUDA streams for concurrent execution of several kernels on a GPU.
*  Dropped Mac OSX support from all samples.

### CUDA 10.2
*  Added `simpleD3D11`. Demonstrates CUDA-D3D11 External Resource Interoperability APIs for updating D3D11 buffers from CUDA and synchronization between D3D11 and CUDA with Keyed Mutexes.
*  Added `simpleDrvRuntime`. Demonstrates CUDA Driver and Runtime APIs working together to load fatbinary of a CUDA kernel.
*  Added `vectorAddMMAP`. Demonstrates how cuMemMap API allows the user to specify the physical properties of their memory while retaining the contiguous nature of their access.
*  Added `memMapIPCDrv`. Demonstrates Inter Process Communication using cuMemMap APIs.
*  Added `cudaNvSci`. Demonstrates CUDA-NvSciBuf/NvSciSync Interop.
*  Added `jacobiCudaGraphs`. Demonstrates Instantiated CUDA Graph Update with Jacobi Iterative Method using different approaches.
*  Added `cuSolverSp_LinearSolver`. Demonstrates cuSolverSP's LU, QR and Cholesky factorization.
*  Added `MersenneTwisterGP11213`. Demonstrates the Mersenne Twister random number generator GP11213 in cuRAND.

### CUDA 10.1 Update 2
*  Added `vulkanImageCUDA`. Demonstrates how to perform Vulkan image - CUDA Interop.
*  Added `nvJPEG_encoder`. Demonstrates encoding of jpeg images using NVJPEG Library.
*  Added Windows OS support to `nvJPEG` sample.
*  Added `boxFilterNPP`. Demonstrates how to use NPP FilterBox function to perform a box filter.
*  Added `cannyEdgeDetectorNPP`. Demonstrates the nppiFilterCannyBorder_8u_C1R Canny Edge Detection image filter function.

### CUDA 10.1 Update 1
*  Added `NV12toBGRandResize`. Demonstrates how to convert and resize NV12 frames to BGR planars frames using CUDA in batch.
*  Added `EGLStream_CUDA_Interop`. Demonstrates data exchange between CUDA and EGL Streams.
*  Added `cuSolverDn_LinearSolver`. Demonstrates cuSolverDN's LU, QR and Cholesky factorization.
*  Added support of Visual Studio 2019 to all samples supported on [Windows](./README.md#windows-1).

### CUDA 10.1
*  Added `immaTensorCoreGemm`. Demonstrates integer GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API for integers employing the Tensor Cores.
*  Added `simpleIPC`. Demonstrates Inter Process Communication with one process per GPU for computation.
*  Added `nvJPEG`. Demonstrates single and batched decoding of jpeg images using NVJPEG Library.
*  Added `bandwidthTest`. It measures the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e.
*  Added `reduction`. Demonstrates several important optimization strategies for Data-Parallel Algorithms like reduction.
*  Update all the samples to support CUDA 10.1.


### CUDA 10.0
*  Added `simpleCudaGraphs`. Demonstrates CUDA Graphs creation, instantiation and launch using Graphs APIs and Stream Capture APIs.
*  Added `conjugateGradientCudaGraphs`. Demonstrates conjugate gradient solver on GPU using CUBLAS and CUSPARSE library calls captured and called using CUDA Graph APIs.
*  Added `simpleVulkan`. Demonstrates Vulkan - CUDA Interop.
*  Added `simpleD3D12`. Demonstrates DX12 - CUDA Interop.
*  Added `UnifiedMemoryPerf`. Demonstrates performance comparision of various memory types involved in system.
*  Added `p2pBandwidthLatencyTest`. Demonstrates Peer-To-Peer (P2P) data transfers between pairs of GPUs and computes latency and bandwidth.
*  Added `systemWideAtomics`. Demonstrates system wide atomic instructions.
*  Added `simpleCUBLASXT`. Demonstrates CUBLAS-XT library which performs GEMM operations over multiple GPUs.
*  Added Windows OS support to `conjugateGradientMultiDeviceCG` sample.
*  Removed support of Visual Studio 2010 from all samples.


### CUDA 9.2

This is the first release of CUDA Samples on GitHub:
*  Added `vectorAdd_nvrtc`. Demonstrates runtime compilation library using NVRTC of a simple vectorAdd kernel.
*  Added `warpAggregatedAtomicsCG`. Demonstrates warp aggregated atomics using Cooperative Groups.
*  Added `deviceQuery`. Enumerates the properties of the CUDA devices present in the system.
*  Added `matrixMul`. Demonstrates a matrix multiplication using shared memory through tiled approach.
*  Added `matrixMulDrv`. Demonstrates a matrix multiplication using shared memory through tiled approach, uses CUDA Driver API.
*  Added `cudaTensorCoreGemm`. Demonstrates a GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in CUDA 9, as well as the new Tensor Cores introduced in the Volta chip family.
*  Added `simpleVoteIntrinsics` which uses *_sync equivalent of the vote intrinsics _any, _all added since CUDA 9.0.
*  Added `shfl_scan` which uses *_sync equivalent of the shfl intrinsics added since CUDA 9.0.
*  Added `conjugateGradientMultiBlockCG`. Demonstrates a conjugate gradient solver on GPU using Multi Block Cooperative Groups.
*  Added `conjugateGradientMultiDeviceCG`. Demonstrates a conjugate gradient solver on multiple GPUs using Multi Device Cooperative Groups, also uses unified memory prefetching and usage hints APIs.
*  Added `simpleCUBLAS`. Demonstrates how perform GEMM operations using CUBLAS library.
*  Added `simpleCUFFT`. Demonstrates how perform FFT operations using CUFFT library.