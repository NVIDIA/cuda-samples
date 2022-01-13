# 2. Concepts and Techniques


### [boxFilter](./boxFilter)
Fast image box filter using CUDA with OpenGL rendering.

### [convolutionSeparable](./convolutionSeparable)
This sample implements a separable convolution filter of a 2D signal with a gaussian kernel.

### [convolutionTexture](./convolutionTexture)
Texture-based implementation of a separable 2D convolution with a gaussian kernel. Used for performance comparison against convolutionSeparable.

### [cuHook](./cuHook)
This sample demonstrates how to build and use an intercept library with CUDA. The library has to be loaded via LD_PRELOAD, e.g. LD_PRELOAD=<full_path>/libcuhook.so.1 ./cuHook

### [dct8x8](./dct8x8)
This sample demonstrates how Discrete Cosine Transform (DCT) for blocks of 8 by 8 pixels can be performed using CUDA: a naive implementation by definition and a more traditional approach used in many libraries. As opposed to implementing DCT in a fragment shader, CUDA allows for an easier and more efficient implementation.

### [EGLStream_CUDA_CrossGPU](./EGLStream_CUDA_CrossGPU)
Demonstrates CUDA and EGL Streams interop, where consumer's EGL Stream is on one GPU and producer's on other and both consumer-producer are different processes.

### [EGLStreams_CUDA_Interop](./EGLStreams_CUDA_Interop)
Demonstrates data exchange between CUDA and EGL Streams.

### [EGLSync_CUDAEvent_Interop](./EGLSync_CUDAEvent_Interop)
Demonstrates interoperability between CUDA Event and EGL Sync/EGL Image using which one can achieve synchronization on GPU itself for GL-EGL-CUDA operations instead of blocking CPU for synchronization.

### [eigenvalues](./eigenvalues)
The computation of all or a subset of all eigenvalues is an important problem in Linear Algebra, statistics, physics, and many other fields. This sample demonstrates a parallel implementation of a bisection algorithm for the computation of all eigenvalues of a tridiagonal symmetric matrix of arbitrary size with CUDA.

### [FunctionPointers](./FunctionPointers)
This sample illustrates how to use function pointers and implements the Sobel Edge Detection filter for 8-bit monochrome images.

### [histogram](./histogram)
This sample demonstrates efficient implementation of 64-bin and 256-bin histogram.

### [imageDenoising](./imageDenoising)
This sample demonstrates two adaptive image denoising techniques: KNN and NLM, based on computation of both geometric and color distance between texels. While both techniques are implemented in the DirectX SDK using shaders, massively speeded up variation of the latter technique, taking advantage of shared memory, is implemented in addition to DirectX counterparts.

### [inlinePTX](./inlinePTX)
A simple test application that demonstrates a new CUDA 4.0 ability to embed PTX in a CUDA kernel.

### [inlinePTX_nvrtc](./inlinePTX_nvrtc)
A simple test application that demonstrates a new CUDA 4.0 ability to embed PTX in a CUDA kernel.

### [interval](./interval)
Interval arithmetic operators example.  Uses various C++ features (templates and recursion).  The recursive mode requires Compute SM 2.0 capabilities.

### [MC_EstimatePiInlineP](./MC_EstimatePiInlineP)
This sample uses Monte Carlo simulation for Estimation of Pi (using inline PRNG).  This sample also uses the NVIDIA CURAND library.

### [MC_EstimatePiInlineQ](./MC_EstimatePiInlineQ)
This sample uses Monte Carlo simulation for Estimation of Pi (using inline QRNG).  This sample also uses the NVIDIA CURAND library.

### [MC_EstimatePiP](./MC_EstimatePiP)
This sample uses Monte Carlo simulation for Estimation of Pi (using batch PRNG).  This sample also uses the NVIDIA CURAND library.

### [MC_EstimatePiQ](./MC_EstimatePiQ)
This sample uses Monte Carlo simulation for Estimation of Pi (using batch QRNG).  This sample also uses the NVIDIA CURAND library.

### [MC_SingleAsianOptionP](./MC_SingleAsianOptionP)
This sample uses Monte Carlo to simulate Single Asian Options using the NVIDIA CURAND library.

### [particles](./particles)
This sample uses CUDA to simulate and visualize a large set of particles and their physical interaction.  Adding "-particles=<N>" to the command line will allow users to set # of particles for simulation.  This example implements a uniform grid data structure using either atomic operations or a fast radix sort from the Thrust library

### [radixSortThrust](./radixSortThrust)
This sample demonstrates a very fast and efficient parallel radix sort uses Thrust library. The included RadixSort class can sort either key-value pairs (with float or unsigned integer keys) or keys only.

### [reduction](./reduction)
A parallel sum reduction that computes the sum of a large arrays of values. This sample demonstrates several important optimization strategies for Data-Parallel Algorithms like reduction using shared memory, __shfl_down_sync, __reduce_add_sync and cooperative_groups reduce.

### [reductionMultiBlockCG](./reductionMultiBlockCG)
This sample demonstrates single pass reduction using Multi Block Cooperative Groups.  This sample requires devices with compute capability 6.0 or higher having compute preemption.

### [scalarProd](./scalarProd)
This sample calculates scalar products of a given set of input vector pairs.

### [scan](./scan)
This example demonstrates an efficient CUDA implementation of parallel prefix sum, also known as "scan".  Given an array of numbers, scan computes a new array in which each element is the sum of all the elements before it in the input array.

### [segmentationTreeThrust](./segmentationTreeThrust)
This sample demonstrates an approach to the image segmentation trees construction.  This method is based on Boruvka's MST algorithm.

### [shfl_scan](./shfl_scan)
This example demonstrates how to use the shuffle intrinsic __shfl_up_sync to perform a scan operation across a thread block. 

### [sortingNetworks](./sortingNetworks)
This sample implements bitonic sort and odd-even merge sort (also known as Batcher's sort), algorithms belonging to the class of sorting networks. While generally subefficient, for large sequences compared to algorithms with better asymptotic algorithmic complexity (i.e. merge sort or radix sort), this may be the preferred algorithms of choice for sorting batches of short-sized to mid-sized (key, value) array pairs. Refer to an excellent tutorial by H. W. Lang http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm

### [streamOrderedAllocation](./streamOrderedAllocation)
This sample demonstrates stream ordered memory allocation on a GPU using cudaMallocAsync and cudaMemPool family of APIs.

### [streamOrderedAllocationIPC](./streamOrderedAllocationIPC)
This sample demonstrates IPC pools of stream ordered memory allocated using cudaMallocAsync and cudaMemPool family of APIs.

### [streamOrderedAllocationP2P](./streamOrderedAllocationP2P)
This sample demonstrates peer-to-peer access of stream ordered memory allocated using cudaMallocAsync and cudaMemPool family of APIs.

### [threadFenceReduction](./threadFenceReduction)
This sample shows how to perform a reduction operation on an array of values using the thread Fence intrinsic to produce a single value in a single kernel (as opposed to two or more kernel calls as shown in the "reduction" CUDA Sample).  Single-pass reduction requires global atomic instructions (Compute Capability 2.0 or later) and the _threadfence() intrinsic (CUDA 2.2 or later).

### [threadMigration](./threadMigration)
Simple program illustrating how to the CUDA Context Management API and uses the new CUDA 4.0 parameter passing and CUDA launch API.  CUDA contexts can be created separately and attached independently to different threads.

