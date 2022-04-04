# 5. Domain Specific


### [bicubicTexture](./bicubicTexture)
This sample demonstrates how to efficiently implement a Bicubic B-spline interpolation filter with CUDA texture.

### [bilateralFilter](./bilateralFilter)
Bilateral filter is an edge-preserving non-linear smoothing filter that is implemented with CUDA with OpenGL rendering. It can be used in image recovery and denoising. Each pixel is weight by considering both the spatial distance and color distance between its neighbors. Reference:"C. Tomasi, R. Manduchi, Bilateral Filtering for Gray and Color Images, proceeding of the ICCV, 1998, http://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf"

### [binomialOptions](./binomialOptions)
This sample evaluates fair call price for a given set of European options under binomial model.

### [binomialOptions_nvrtc](./binomialOptions_nvrtc)
This sample evaluates fair call price for a given set of European options under binomial model. This sample makes use of NVRTC for Runtime Compilation.

### [BlackScholes](./BlackScholes)
This sample evaluates fair call and put prices for a given set of European options by Black-Scholes formula.

### [BlackScholes_nvrtc](./BlackScholes_nvrtc)
This sample evaluates fair call and put prices for a given set of European options by Black-Scholes formula, compiling the CUDA kernels involved at runtime using NVRTC.
    

### [convolutionFFT2D](./convolutionFFT2D)
This sample demonstrates how 2D convolutions with very large kernel sizes can be efficiently implemented using FFT transformations.

### [dwtHaar1D](./dwtHaar1D)
Discrete Haar wavelet decomposition for 1D signals with a length which is a power of 2.

### [dxtc](./dxtc)
High Quality DXT Compression using CUDA. This example shows how to implement an existing computationally-intensive CPU compression algorithm in parallel on the GPU, and obtain an order of magnitude performance improvement.

### [fastWalshTransform](./fastWalshTransform)
Naturally(Hadamard)-ordered Fast Walsh Transform for batching vectors of arbitrary eligible lengths that are power of two in size.

### [FDTD3d](./FDTD3d)
This sample applies a finite differences time domain progression stencil on a 3D surface.

### [fluidsD3D9](./fluidsD3D9)
An example of fluid simulation using CUDA and CUFFT, with Direct3D 9 rendering.  A Direct3D Capable device is required.

### [fluidsGL](./fluidsGL)
An example of fluid simulation using CUDA and CUFFT, with OpenGL rendering.

### [fluidsGLES](./fluidsGLES)
An example of fluid simulation using CUDA and CUFFT, with OpenGLES rendering.

### [HSOpticalFlow](./HSOpticalFlow)
Variational optical flow estimation example.  Uses textures for image operations. Shows how simple PDE solver can be accelerated with CUDA.

### [Mandelbrot](./Mandelbrot)
This sample uses CUDA to compute and display the Mandelbrot or Julia sets interactively. It also illustrates the use of "double single" arithmetic to improve precision when zooming a long way into the pattern. This sample uses double precision.  Thanks to Mark Granger of NewTek who submitted this code sample.!

### [marchingCubes](./marchingCubes)
This sample extracts a geometric isosurface from a volume dataset using the marching cubes algorithm. It uses the scan (prefix sum) function from the Thrust library to perform stream compaction.

### [MonteCarloMultiGPU](./MonteCarloMultiGPU)
This sample evaluates fair call price for a given set of European options using the Monte Carlo approach, taking advantage of all CUDA-capable GPUs installed in the system. This sample use double precision hardware if a GTX 200 class GPU is present.  The sample also takes advantage of CUDA 4.0 capability to supporting using a single CPU thread to control multiple GPUs

### [nbody](./nbody)
This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA.  This sample accompanies the GPU Gems 3 chapter "Fast N-Body Simulation with CUDA".  With CUDA 5.5, performance on Tesla K20c has increased to over 1.8TFLOP/s single precision.  Double Performance has also improved on all Kepler and Fermi GPU architectures as well.  Starting in CUDA 4.0, the nBody sample has been updated to take advantage of new features to easily scale the n-body simulation across multiple GPUs in a single PC.  Adding "-numbodies=<bodies>" to the command line will allow users to set # of bodies for simulation.  Adding “-numdevices=<N>” to the command line option will cause the sample to use N devices (if available) for simulation.  In this mode, the position and velocity data for all bodies are read from system memory using “zero copy” rather than from device memory.  For a small number of devices (4 or fewer) and a large enough number of bodies, bandwidth is not a bottleneck so we can achieve strong scaling across these devices.

### [nbody_opengles](./nbody_opengles)
This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA. Unlike the OpenGL nbody sample, there is no user interaction.

### [nbody_screen](./nbody_screen)
This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA. Unlike the OpenGL nbody sample, there is no user interaction.

### [NV12toBGRandResize](./NV12toBGRandResize)
This code shows two ways to convert and resize NV12 frames to BGR 3 planars frames using CUDA in batch. Way-1, Convert NV12 Input to BGR @ Input Resolution-1, then Resize to Resolution#2. Way-2, resize NV12 Input to Resolution#2 then convert it to BGR Output. NVIDIA HW Decoder, both dGPU and Tegra, normally outputs NV12 pitch format frames. For the inference using TensorRT, the input frame needs to be BGR planar format with possibly different size. So, conversion and resizing from NV12 to BGR planar is usually required for the inference following decoding. This CUDA code provides a reference implementation for conversion and resizing.

### [p2pBandwidthLatencyTest](./p2pBandwidthLatencyTest)
This application demonstrates the CUDA Peer-To-Peer (P2P) data transfers between pairs of GPUs and computes latency and bandwidth.  Tests on GPU pairs using P2P and without P2P are tested.

### [postProcessGL](./postProcessGL)
This sample shows how to post-process an image rendered in OpenGL using CUDA.

### [quasirandomGenerator](./quasirandomGenerator)
This sample implements Niederreiter Quasirandom Sequence Generator and Inverse Cumulative Normal Distribution functions for the generation of Standard Normal Distributions.

### [quasirandomGenerator_nvrtc](./quasirandomGenerator_nvrtc)
This sample implements Niederreiter Quasirandom Sequence Generator and Inverse Cumulative Normal Distribution functions for the generation of Standard Normal Distributions, compiling the CUDA kernels involved at runtime using NVRTC.

### [recursiveGaussian](./recursiveGaussian)
This sample implements a Gaussian blur using Deriche's recursive method. The advantage of this method is that the execution time is independent of the filter width.

### [simpleD3D10](./simpleD3D10)
Simple program which demonstrates interoperability between CUDA and Direct3D10. The program generates a vertex array with CUDA and uses Direct3D10 to render the geometry.  A Direct3D Capable device is required.

### [simpleD3D10RenderTarget](./simpleD3D10RenderTarget)
Simple program which demonstrates interop of rendertargets between Direct3D10 and CUDA. The program uses RenderTarget positions with CUDA and generates a histogram with visualization.  A Direct3D10 Capable device is required.

### [simpleD3D10Texture](./simpleD3D10Texture)
Simple program which demonstrates how to interoperate CUDA with Direct3D10 Texture.  The program creates a number of D3D10 Textures (2D, 3D, and CubeMap) which are generated from CUDA kernels. Direct3D then renders the results on the screen.  A Direct3D10 Capable device is required.

### [simpleD3D11](./simpleD3D11)
Simple program which demonstrates  how to use the CUDA D3D11 External Resource Interoperability APIs to update D3D11 buffers from CUDA and synchronize between D3D11 and CUDA with Keyed Mutexes.


### [simpleD3D11Texture](./simpleD3D11Texture)
Simple program which demonstrates Direct3D11 Texture interoperability with CUDA.  The program creates a number of D3D11 Textures (2D, 3D, and CubeMap) which are written to from CUDA kernels. Direct3D then renders the results on the screen.  A Direct3D Capable device is required.

### [simpleD3D12](./simpleD3D12)
A program which demonstrates Direct3D12 interoperability with CUDA.  The program creates a sinewave in DX12 vertex buffer which is created using CUDA kernels. DX12 and CUDA synchronizes using DirectX12 Fences. Direct3D then renders the results on the screen.  A DirectX12 Capable NVIDIA GPU is required on Windows10 or higher OS.

### [simpleD3D9](./simpleD3D9)
Simple program which demonstrates interoperability between CUDA and Direct3D9. The program generates a vertex array with CUDA and uses Direct3D9 to render the geometry.  A Direct3D capable device is required.

### [simpleD3D9Texture](./simpleD3D9Texture)
Simple program which demonstrates Direct3D9 Texture interoperability with CUDA.  The program creates a number of D3D9 Textures (2D, 3D, and CubeMap) which are written to from CUDA kernels. Direct3D then renders the results on the screen.  A Direct3D capable device is required.

### [simpleGL](./simpleGL)
Simple program which demonstrates interoperability between CUDA and OpenGL. The program modifies vertex positions with CUDA and uses OpenGL to render the geometry.

### [simpleGLES](./simpleGLES)
Demonstrates data exchange between CUDA and OpenGL ES (aka Graphics interop). The program modifies vertex positions with CUDA and uses OpenGL ES to render the geometry.

### [simpleGLES_EGLOutput](./simpleGLES_EGLOutput)
Demonstrates data exchange between CUDA and OpenGL ES (aka Graphics interop). The program modifies vertex positions with CUDA and uses OpenGL ES to render the geometry, and shows how to render directly to the display using the EGLOutput mechanism and the DRM library.

### [simpleGLES_screen](./simpleGLES_screen)
Demonstrates data exchange between CUDA and OpenGL ES (aka Graphics interop). The program modifies vertex positions with CUDA and uses OpenGL ES to render the geometry.

### [simpleVulkan](./simpleVulkan)
This sample demonstrates Vulkan CUDA Interop. CUDA imports the Vulkan vertex buffer and operates on it to create sinewave, and synchronizes with Vulkan through vulkan semaphores imported by CUDA. This sample depends on Vulkan SDK, GLFW3 libraries, for building this sample please refer to "Build_instructions.txt" provided in this sample's directory

### [simpleVulkanMMAP](./simpleVulkanMMAP)
 This sample demonstrates Vulkan CUDA Interop via cuMemMap APIs. CUDA exports buffers that Vulkan imports as vertex buffer. CUDA invokes kernels to operate on vertices and synchronizes with Vulkan through vulkan semaphores imported by CUDA. This sample depends on Vulkan SDK, GLFW3 libraries, for building this sample please refer to "Build_instructions.txt" provided in this sample's directory

### [SLID3D10Texture](./SLID3D10Texture)
Simple program which demonstrates SLI with Direct3D10 Texture interoperability with CUDA.  The program creates a D3D10 Texture which is written to from a CUDA kernel. Direct3D then renders the results on the screen.  A Direct3D Capable device is required.

### [smokeParticles](./smokeParticles)
Smoke simulation with volumetric shadows using half-angle slicing technique. Uses CUDA for procedural simulation, Thrust Library for sorting algorithms, and OpenGL for graphics rendering.

### [SobelFilter](./SobelFilter)
This sample implements the Sobel edge detection filter for 8-bit monochrome images.

### [SobolQRNG](./SobolQRNG)
This sample implements Sobol Quasirandom Sequence Generator.

### [stereoDisparity](./stereoDisparity)
A CUDA program that demonstrates how to compute a stereo disparity map using SIMD SAD (Sum of Absolute Difference) intrinsics.  Requires Compute Capability 2.0 or higher.

### [VFlockingD3D10](./VFlockingD3D10)
The sample models formation of V-shaped flocks by big birds, such as geese and cranes. The algorithms of such flocking are borrowed from the paper "V-like formations in flocks of artificial birds" from Artificial Life, Vol. 14, No. 2, 2008. The sample has CPU- and GPU-based implementations. Press 'g' to toggle between them. The GPU-based simulation works many times faster than the CPU-based one. The printout in the console window reports the simulation time per step. Press 'r' to reset the initial distribution of birds.

### [volumeFiltering](./volumeFiltering)
This sample demonstrates 3D Volumetric Filtering using 3D Textures and 3D Surface Writes.

### [volumeRender](./volumeRender)
This sample demonstrates basic volume rendering using 3D Textures.

### [vulkanImageCUDA](./vulkanImageCUDA)
This sample demonstrates Vulkan Image - CUDA Interop. CUDA imports the Vulkan image buffer, performs box filtering over it, and synchronizes with Vulkan through vulkan semaphores imported by CUDA. This sample depends on Vulkan SDK, GLFW3 libraries, for building this sample please refer to "Build_instructions.txt" provided in this sample's directory

