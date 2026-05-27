# 8. Platform_Specific/Tegra


### [cudaNvSciNvMedia](./cudaNvSciNvMedia)
This sample demonstrates CUDA-NvMedia interop via NvSciBuf/NvSciSync APIs. Note that this sample only supports cross build from x86_64 to aarch64, aarch64 native build is not supported. For detailed workflow of the sample please check cudaNvSciNvMedia_Readme.pdf in the sample directory.

### [cudaNvSciBufMultiplanar](./cudaNvSciBufMultiplanar)
This sample demonstrates CUDA-NvSciBuf Interop for Multiplanar images. A YUV 420 multiplanar image is flipped and allocated using NvSciBuf APIs and imported into CUDA with CUDA External Resource Interoperability. A CUDA surface is created from the corresponding mapped CUDA array and again bit flipping is performed on the surface. The result is copied back to a YUV image which is compared against the input.

### [cuDLAErrorReporting](./cuDLAErrorReporting)
This sample demonstrates how DLA errors can be detected via CUDA.

### [cuDLAHybridMode](./cuDLAHybridMode)
This sample demonstrates cuDLA hybrid mode wherein DLA can be programmed using CUDA.

### [cuDLALayerwiseStatsHybrid](./cuDLALayerwiseStatsHybrid)
This sample is used to provide layerwise statistics to the application in the cuDLA hybrid mode wherein DLA is programmed using CUDA.

### [cuDLALayerwiseStatsStandalone](./cuDLALayerwiseStatsStandalone)
This sample is used to provide layerwise statistics to the application in cuDLA standalone mode where DLA is programmed without using CUDA.

### [cuDLAStandaloneMode](./cuDLAStandaloneMode)
This sample demonstrates cuDLA standalone mode wherein DLA can be programmed without using CUDA.

### [EGLSync_CUDAEvent_Interop](./EGLSync_CUDAEvent_Interop)
Demonstrates interoperability between CUDA Event and EGL Sync/EGL Image using which one can achieve synchronization on GPU itself for GL-EGL-CUDA operations instead of blocking CPU for synchronization.

### [fluidsGLES](./fluidsGLES)
An example of fluid simulation using CUDA and CUFFT, with OpenGLES rendering.

### [nbody_opengles](./nbody_opengles)
This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA. Unlike the OpenGL nbody sample, there is no user interaction.

### [simpleGLES](./simpleGLES)
Demonstrates data exchange between CUDA and OpenGL ES (aka Graphics interop). The program modifies vertex positions with CUDA and uses OpenGL ES to render the geometry.

### [simpleGLES_EGLOutput](./simpleGLES_EGLOutput)
Demonstrates data exchange between CUDA and OpenGL ES (aka Graphics interop). The program modifies vertex positions with CUDA and uses OpenGL ES to render the geometry, and shows how to render directly to the display using the EGLOutput mechanism and the DRM library.
