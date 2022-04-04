# 4. CUDA Libraries


### [batchCUBLAS](./batchCUBLAS)
A CUDA Sample that demonstrates how using batched CUBLAS API calls to improve overall performance.

### [batchedLabelMarkersAndLabelCompressionNPP](./batchedLabelMarkersAndLabelCompressionNPP)
An NPP CUDA Sample that demonstrates how to use the NPP label markers generation and label compression functions based on a Union Find (UF) algorithm including both single image and batched image versions.

### [boxFilterNPP](./boxFilterNPP)
A NPP CUDA Sample that demonstrates how to use NPP FilterBox function to perform a Box Filter.

### [cannyEdgeDetectorNPP](./cannyEdgeDetectorNPP)
An NPP CUDA Sample that demonstrates the recommended parameters to use with the nppiFilterCannyBorder_8u_C1R Canny Edge Detection image filter function. This function expects a single channel 8-bit grayscale input image. You can generate a grayscale image from a color image by first calling nppiColorToGray() or nppiRGBToGray(). The Canny Edge Detection function combines and improves on the techniques required to produce an edge detection image using multiple steps.

### [conjugateGradient](./conjugateGradient)
This sample implements a conjugate gradient solver on GPU using CUBLAS and CUSPARSE library.

### [conjugateGradientCudaGraphs](./conjugateGradientCudaGraphs)
This sample implements a conjugate gradient solver on GPU using CUBLAS and CUSPARSE library calls captured and called using CUDA Graph APIs.

### [conjugateGradientMultiBlockCG](./conjugateGradientMultiBlockCG)
This sample implements a conjugate gradient solver on GPU using Multi Block Cooperative Groups, also uses Unified Memory.

### [conjugateGradientMultiDeviceCG](./conjugateGradientMultiDeviceCG)
This sample implements a conjugate gradient solver on multiple GPUs using Multi Device Cooperative Groups, also uses Unified Memory optimized using prefetching and usage hints.

### [conjugateGradientPrecond](./conjugateGradientPrecond)
This sample implements a preconditioned conjugate gradient solver on GPU using CUBLAS and CUSPARSE library.

### [conjugateGradientUM](./conjugateGradientUM)
This sample implements a conjugate gradient solver on GPU using CUBLAS and CUSPARSE library, using Unified Memory

### [cudaNvSci](./cudaNvSci)
This sample demonstrates CUDA-NvSciBuf/NvSciSync Interop. Two CPU threads import the NvSciBuf and NvSciSync into CUDA to perform two image processing algorithms on a ppm image - image rotation in 1st thread & rgba to grayscale conversion of rotated image in 2nd thread. Currently only supported on Ubuntu 18.04

### [cudaNvSciNvMedia](./cudaNvSciNvMedia)
This sample demonstrates CUDA-NvMedia interop via NvSciBuf/NvSciSync APIs. Note that this sample only supports cross build from x86_64 to aarch64, aarch64 native build is not supported. For detailed workflow of the sample please check cudaNvSciNvMedia_Readme.pdf in the sample directory.

### [cuDLAErrorReporting](./cuDLAErrorReporting)
This sample demonstrates how DLA errors can be detected via CUDA.

### [cuDLAHybridMode](./cuDLAHybridMode)
This sample demonstrates cuDLA hybrid mode wherein DLA can be programmed using CUDA.

### [cuDLAStandaloneMode](./cuDLAStandaloneMode)
This sample demonstrates cuDLA standalone mode wherein DLA can be programmed without using CUDA.

### [cuSolverDn_LinearSolver](./cuSolverDn_LinearSolver)
A CUDA Sample that demonstrates cuSolverDN's LU, QR and Cholesky factorization.

### [cuSolverRf](./cuSolverRf)
A CUDA Sample that demonstrates cuSolver's refactorization library - CUSOLVERRF.

### [cuSolverSp_LinearSolver](./cuSolverSp_LinearSolver)
A CUDA Sample that demonstrates cuSolverSP's LU, QR and Cholesky factorization.

### [cuSolverSp_LowlevelCholesky](./cuSolverSp_LowlevelCholesky)
A CUDA Sample that demonstrates Cholesky factorization using cuSolverSP's low level APIs.

### [cuSolverSp_LowlevelQR](./cuSolverSp_LowlevelQR)
A CUDA Sample that demonstrates QR factorization using cuSolverSP's low level APIs.

### [FilterBorderControlNPP](./FilterBorderControlNPP)
This sample demonstrates how any border version of an NPP filtering function can be used in the most common mode, with border control enabled. Mentioned functions can be used to duplicate the results of the equivalent non-border version of the NPP functions. They can be also used for enabling and disabling border control on various source image edges depending on what portion of the source image is being used as input.

### [freeImageInteropNPP](./freeImageInteropNPP)
A simple CUDA Sample demonstrate how to use FreeImage library with NPP.

### [histEqualizationNPP](./histEqualizationNPP)
This CUDA Sample demonstrates how to use NPP for histogram equalization for image data.

### [lineOfSight](./lineOfSight)
This sample is an implementation of a simple line-of-sight algorithm: Given a height map and a ray originating at some observation point, it computes all the points along the ray that are visible from the observation point. The implementation is based on the Thrust library.

### [matrixMulCUBLAS](./matrixMulCUBLAS)
This sample implements matrix multiplication from Chapter 3 of the programming guide. To illustrate GPU performance for matrix multiply, this sample also shows how to use the new CUDA 4.0 interface for CUBLAS to demonstrate high-performance performance for matrix multiplication.

### [MersenneTwisterGP11213](./MersenneTwisterGP11213)
This sample demonstrates the Mersenne Twister random number generator GP11213 in cuRAND.

### [nvJPEG](./nvJPEG)
A CUDA Sample that demonstrates single and batched decoding of jpeg images using NVJPEG Library.

### [nvJPEG_encoder](./nvJPEG_encoder)
A CUDA Sample that demonstrates single encoding of jpeg images using NVJPEG Library.

### [oceanFFT](./oceanFFT)
This sample simulates an Ocean height field using CUFFT Library and renders the result using OpenGL.

### [randomFog](./randomFog)
This sample illustrates pseudo- and quasi- random numbers produced by CURAND.

### [simpleCUBLAS](./simpleCUBLAS)
Example of using CUBLAS API interface to perform GEMM operations.

### [simpleCUBLAS_LU](./simpleCUBLAS_LU)
CUDA sample demonstrating cuBLAS API cublasDgetrfBatched() for lower-upper (LU) decomposition of a matrix.

### [simpleCUBLASXT](./simpleCUBLASXT)
Example of using CUBLAS-XT library which performs GEMM operations over Multiple GPUs.

### [simpleCUFFT](./simpleCUFFT)
Example of using CUFFT. In this example, CUFFT is used to compute the 1D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain. cuFFT plans are created using simple and advanced API functions.

### [simpleCUFFT_2d_MGPU](./simpleCUFFT_2d_MGPU)
Example of using CUFFT. In this example, CUFFT is used to compute the 2D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain on Multiple GPU.

### [simpleCUFFT_callback](./simpleCUFFT_callback)
Example of using CUFFT. In this example, CUFFT is used to compute the 1D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain. The difference between this example and the Simple CUFFT example is that the multiplication step is done by the CUFFT kernel with a user-supplied CUFFT callback routine, rather than by a separate kernel call.

### [simpleCUFFT_MGPU](./simpleCUFFT_MGPU)
Example of using CUFFT. In this example, CUFFT is used to compute the 1D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain on Multiple GPU.

### [watershedSegmentationNPP](./watershedSegmentationNPP)
An NPP CUDA Sample that demonstrates how to use the NPP watershed segmentation function.

