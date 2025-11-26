## Changelog

### CUDA 13.0
* Updated the samples using the cudaDeviceProp fields which are deprecated and removed in CUDA 13.0, replacing the fields with the equivalents in "cudaDeviceGetAttribute":
    * Deprecated "cudaDeviceProp" fields
        `int clockRate; // - Replaced with "cudaDevAttrClockRate"`
        `int deviceOverlap; // - Replaced with "cudaDevAttrGpuOverlap */`
        `int kernelExecTimeoutEnabled; // - Replaced with "cudaDevAttrKernelExecTimeout`
        `int computeMode; // - Replaced with "cudaDevAttrComputeMode" */`
        `int memoryClockRate; // - Replaced with "cudaDevAttrMemoryClockRate"`
        `int cooperativeMultiDeviceLaunch; // - Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.`
    * `0_Introduction`
        * `UnifiedMemoryStreams`
        * `simpleHyperQ`
        * `simpleIPC`
        * `simpleMultiCopy`
        * `systemWideAtomics`
    * `1_Utilitie`
        * `deviceQuery`
    * `2_Concepts_and_Techniques`
        * `streamOrderedAllocationIPC`
    * `4_CUDA_Libraries`
        * `simpleCUBLASXT`
    * `5_Domain_Specific`
        * `simpleVulkan`
        * `vulkanImageCUDA`
* Updated the samples using the CUDA driver API "cuCtxCreate" with adding the parameter "CUctxCreateParams" as "cuCtxCreate" is updated to "cuCtxCreate_v4" by default in CUDA 13.0:
    * `Common`
        * `nvrtc_helper.h`
    * `0_Introduction`
        * `UnifiedMemoryStreams`
        * `matrixMulDrv`
        * `simpleTextureDrv`
        * `vectorAddDrv`
        * `vectorAddMMAP`
    * `2_Concepts_and_Techniques`
        * `EGLStream_CUDA_CrossGPU`
        * `EGLStream_CUDA_Interop`
        * `threadMigration`
    * `3_CUDA_Features`
        * `graphMemoryFootprint`
        * `memMapIPCDrv`
    * `4_CUDA_Libraries`
        * `jitLto`
    * `7_libNVVM`
        * `cuda-c-linking`
        * `device-side-launch`
        * `simple`
        * `uvmlite`
    * `8_Platform_Specific/Tegra`
        * `EGLSync_CUDAEvent_Interop`
* Updated the sample using CUDA API "cudaGraphAddNode"/"cudaStreamGetCaptureInfo" with adding "cudaGraphEdgeData" pointer parameter as they are updated to "cudaGraphAddNode_v2"/"cudaStreamGetCaptureInfo_v3" by default in CUDA 13.0:
    * `3_CUDA_Features`
        * `graphConditionalNodes`
* Updated the samples using CUDA API "cudaMemAdvise"/"cudaMemPrefetchAsync" with changing the parameter "int device" to "cudaMemLocation location" as they are updated to "cudaMemAdvise_v2"/"cudaMemPrefetchAsyn_v2" by default in CUDA 13.0.
    * `4_CUDA_Libraries`
        * `conjugateGradientMultiDeviceCG`
    * `6_Performance`
        * `UnifiedMemoryPerf`
* Replaced "thrust::identity<uint>()" with "cuda::std::identity()" as it is deprecated in CUDA 13.0.
    * `2_Concepts_and_Techniques`
        * `segmentationTreeThrust`
* Updated the the headers file and samples for CUFFT error codes update.
    * Deprecated CUFFT errors:
        * `CUFFT_INCOMPLETE_PARAMETER_LIST`
        * `CUFFT_PARSE_ERROR`
        * `CUFFT_LICENSE_ERROR`
    * New added CUFFT errors:
        * `CUFFT_MISSING_DEPENDENCY`
        * `CUFFT_NVRTC_FAILURE`
        * `CUFFT_NVJITLINK_FAILURE`
        * `CUFFT_NVSHMEM_FAILURE`
    * Header files and samples that are related with this change:
        * `Common/helper_cuda.h`
        * `4_CUDA_Libraries`
            * `simpleCUFFT`
            * `simpleCUFFT_2d_MGPU`
            * `simpleCUFFT_MGPU`
            * `simpleCUFFT_callback`
* Updated toolchain for cross-compilation for Tegra QNX platforms.

### CUDA 12.9
* Updated toolchain for cross-compilation for Tegra Linux platforms.
* Added `run_tests.py` utility to exercise all samples. See README.md for details
* Repository has been updated with consistent code formatting across all samples
* Many small code tweaks and bug fixes (see commit history for details)
* Removed the following outdated samples:
  * `1_Utilities`
    * `bandwidthTest` - this sample was out of date and did not produce accurate results. For bandwidth
    testing of NVIDIA GPU platforms, please refer to [NVBandwidth](https://github.com/NVIDIA/nvbandwidth)

### CUDA 12.8
* Updated build system across the repository to CMake. Removed Visual Studio project files and Makefiles.
* Removed the following outdated samples:
    * `0_Introduction`
        * `c++11_cuda` demonstrating CUDA and C++ 11 interoperability (reason: obsolete)
        * `concurrentKernels` demonstrating the ability to run multiple kernels simultaneously (reason: obsolete)
        * `cppIntegration` demonstrating calling between .cu and .cpp files (reason: obsolete)
        * `cppOverload` demonstrating C++ function overloading (reason: obsolete)
        * `simpleSeparateCompilation` demonstrating NVCC compilation to a static library (reason: trivial)
        * `simpleTemplates_nvrtc` demonstrating NVRTC usage for `simpleTemplates` sample (reason: redundant)
        * `simpleVoteIntrinsics_nvrtc` demonstrating NVRTC usage for `simpleVoteIntrinsics` sample (reason: redundant)
    * `2_Concepts_and_Techniques`
        * `cuHook` demonstrating dlsym hooks. (reason: incompatible with modern `glibc`)
    * `4_CUDA_Libraries`
        * `batchedLabelMarkersAndLabelCompressionNPP` demonstrating NPP features (reason: some functionality removed from library)
    * `5_Domain_Specific`
        * Legacy Direct3D 9 and 10 interoperability samples:
            * `fluidsD3D9`
            * `simpleD3D10`
            * `simpleD3D10RenderTarget`
            * `simpleD3D10Texture`
            * `simpleD3D9`
            * `simpleD3D9Texture`
            * `SLID3D10Texture`
            * `VFlockingD3D10`
    * `8_Platform_Specific/Tegra`
        * Temporarily removed the following two samples pending updates:
            * `nbody_screen` demonstrating the nbody sample in QNX
            * `simpleGLES_screen` demonstrating GLES interop in QNX
* Moved the following Tegra-specific samples to a dedicated subdirectory: `8_Platform_Specific/Tegra`
    * `EGLSync_CUDAEvent_Interop`
    * `cuDLAErrorReporting`
    * `cuDLAHybridMode`
    * `cuDLALayerwiseStatsHybrid`
    * `cuDLALayerwiseStatsStandalone`
    * `cuDLAStandaloneMode`
    * `cudaNvSciBufMultiplanar`
    * `cudaNvSciNvMedia`
    * `fluidsGLES`
    * `nbody_opengles`
    * `simpleGLES`
    * `simpleGLES_EGLOutput`



### CUDA 12.5

### CUDA 12.4
* Added graphConditionalNodes Sample

### CUDA 12.3
* Added cuDLA samples
* Fixed jitLto regression

### CUDA 12.2
* libNVVM samples received updates
* Fixed jitLto Case issues
* Enabled HOST_COMPILER flag to the makefiles for GCC which is untested but may still work.

### CUDA 12.1
* Added new sample for Large Kernels

### CUDA 12.0
* Added new flags for JIT compiling
* Removed deprecated APIs in Hopper Architecture

### CUDA 11.6
* Added new folder structure for samples
* Added support of Visual Studio 2022 to all samples supported on [Windows](#windows-1).
* All CUDA samples are now only available on [GitHub](https://github.com/nvidia/cuda-samples). They are no longer available via CUDA toolkit.

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
