# 3. CUDA Features


### [bf16TensorCoreGemm](./bf16TensorCoreGemm)
A CUDA sample demonstrating __nv_bfloat16 (e8m7) GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API introduced with CUDA 11 in Ampere chip family tensor cores for faster matrix operations. This sample also uses async copy provided by cuda pipeline interface for gmem to shmem async loads which improves kernel performance and reduces register presssure.

### [binaryPartitionCG](./binaryPartitionCG)
This sample is a simple code that illustrates binary partition cooperative groups and reduce within the thread block.

### [bindlessTexture](./bindlessTexture)
This example demonstrates use of cudaSurfaceObject, cudaTextureObject, and MipMap support in CUDA.  A GPU with Compute Capability SM 3.0 is required to run the sample.

### [cdpAdvancedQuicksort](./cdpAdvancedQuicksort)
This sample demonstrates an advanced quicksort implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

### [cdpBezierTessellation](./cdpBezierTessellation)
This sample demonstrates bezier tessellation of lines implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

### [cdpQuadtree](./cdpQuadtree)
This sample demonstrates Quad Trees implemented using CUDA Dynamic Parallelism. This sample requires devices with compute capability 3.5 or higher.

### [cdpSimplePrint](./cdpSimplePrint)
This sample demonstrates simple printf implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

### [cdpSimpleQuicksort](./cdpSimpleQuicksort)
This sample demonstrates simple quicksort implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

### [cudaCompressibleMemory](./cudaCompressibleMemory)
This sample demonstrates the compressible memory allocation using cuMemMap API.

### [cudaTensorCoreGemm](./cudaTensorCoreGemm)
CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in CUDA 9.

This sample demonstrates the use of the new CUDA WMMA API employing the Tensor Cores introduced in the Volta chip family for faster matrix operations.

In addition to that, it demonstrates the use of the new CUDA function attribute cudaFuncAttributeMaxDynamicSharedMemorySize that allows the application to reserve an extended amount of shared memory than it is available by default.

### [dmmaTensorCoreGemm](./dmmaTensorCoreGemm)
CUDA sample demonstrates double precision GEMM computation using the Double precision Warp Matrix Multiply and Accumulate (WMMA) API introduced with CUDA 11 in Ampere chip family tensor cores for faster matrix operations. This sample also uses async copy provided by cuda pipeline interface for gmem to shmem async loads which improves kernel performance and reduces register presssure. Further, this sample also demonstrates how to use cooperative groups async copy interface over a group for performing gmem to shmem async loads.

### [globalToShmemAsyncCopy](./globalToShmemAsyncCopy)
This sample implements matrix multiplication which uses asynchronous copy of data from global to shared memory when on compute capability 8.0 or higher. Also demonstrates arrive-wait barrier for synchronization.

### [graphMemoryFootprint](./graphMemoryFootprint)
This sample demonstrates how graph memory nodes re-use virtual addresses and physical memory.

### [graphMemoryNodes](./graphMemoryNodes)
A demonstration of memory allocations and frees within CUDA graphs using Graph APIs and Stream Capture APIs.

### [immaTensorCoreGemm](./immaTensorCoreGemm)
CUDA sample demonstrating a integer GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API for integer introduced in CUDA 10. This sample demonstrates the use of the CUDA WMMA API employing the Tensor Cores introduced in the Volta chip family for faster matrix operations. In addition to that, it demonstrates the use of the new CUDA function attribute cudaFuncAttributeMaxDynamicSharedMemorySize that allows the application to reserve an extended amount of shared memory than it is available by default.

### [jacobiCudaGraphs](./jacobiCudaGraphs)
Demonstrates Instantiated CUDA Graph Update with Jacobi Iterative Method using cudaGraphExecKernelNodeSetParams() and cudaGraphExecUpdate() approach.

### [memMapIPCDrv](./memMapIPCDrv)
This CUDA Driver API sample is a very basic sample that demonstrates Inter Process Communication using cuMemMap APIs with one process per GPU for computation. Requires Compute Capability 3.0 or higher and a Linux Operating System, or a Windows Operating System

### [newdelete](./newdelete)
This sample demonstrates dynamic global memory allocation through device C++ new and delete operators and virtual function declarations available with CUDA 4.0.

### [ptxjit](./ptxjit)
This sample uses the Driver API to just-in-time compile (JIT) a Kernel from PTX code. Additionally, this sample demonstrates the seamless interoperability capability of the CUDA Runtime and CUDA Driver API calls.  For CUDA 5.5, this sample shows how to use cuLink* functions to link PTX assembly using the CUDA driver at runtime.

### [simpleCudaGraphs](./simpleCudaGraphs)
A demonstration of CUDA Graphs creation, instantiation and launch using Graphs APIs and Stream Capture APIs.

### [StreamPriorities](./StreamPriorities)
This sample demonstrates basic use of stream priorities.

### [tf32TensorCoreGemm](./tf32TensorCoreGemm)
A CUDA sample demonstrating tf32 (e8m10) GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API introduced with CUDA 11 in Ampere chip family tensor cores for faster matrix operations. This sample also uses async copy provided by cuda pipeline interface for gmem to shmem async loads which improves kernel performance and reduces register presssure.

### [warpAggregatedAtomicsCG](./warpAggregatedAtomicsCG)
This sample demonstrates how using Cooperative Groups (CG) to perform warp aggregated atomics to single and multiple counters, a useful technique to improve performance when many threads atomically add to a single or multiple counters.

