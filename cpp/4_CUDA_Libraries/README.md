# 4. CUDA Libraries


## [conjugateGradientCudaGraphs](./conjugateGradientCudaGraphs)
This sample implements a conjugate gradient solver on GPU using CUBLAS and CUSPARSE library calls captured and called using CUDA Graph APIs.

## [conjugateGradientMultiBlockCG](./conjugateGradientMultiBlockCG)
This sample implements a conjugate gradient solver on GPU using Multi Block Cooperative Groups, also uses Unified Memory.

## [conjugateGradientMultiDeviceCG](./conjugateGradientMultiDeviceCG)
This sample implements a conjugate gradient solver on multiple GPUs using Multi Device Cooperative Groups, also uses Unified Memory optimized using prefetching and usage hints.

## [cudaNvSci](./cudaNvSci)
This sample demonstrates CUDA-NvSciBuf/NvSciSync Interop. Two CPU threads import the NvSciBuf and NvSciSync into CUDA to perform two image processing algorithms on a ppm image - image rotation in 1st thread & rgba to grayscale conversion of rotated image in 2nd thread. Currently only supported on Ubuntu 18.04

## [lineOfSight](./lineOfSight)
This sample is an implementation of a simple line-of-sight algorithm: Given a height map and a ray originating at some observation point, it computes all the points along the ray that are visible from the observation point. The implementation is based on the Thrust library.

## [oceanFFT](./oceanFFT)
This sample simulates an Ocean height field using CUFFT Library and renders the result using OpenGL.

## [randomFog](./randomFog)
This sample illustrates pseudo- and quasi- random numbers produced by CURAND.

