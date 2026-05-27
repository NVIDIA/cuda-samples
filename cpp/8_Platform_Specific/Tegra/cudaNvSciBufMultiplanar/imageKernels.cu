/* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <cuda.h>
#include <helper_cuda.h>

static __global__ void flipSurfaceBits(cudaSurfaceObject_t surfObj, int width, int height)
{
    char         data;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        // Read from input surface
        surf2Dread(&data, surfObj, x, y);
        // Write to output surface
        data = ~data;
        surf2Dwrite(data, surfObj, x, y);
    }
}

// Copy cudaArray to surface memory and launch the CUDA kernel
void launchFlipSurfaceBitsKernel(cudaArray_t *levelArray,
                                 int32_t     *multiPlanarWidth,
                                 int32_t     *multiPlanarHeight,
                                 int          numPlanes)
{

    cudaSurfaceObject_t surfObject[numPlanes] = {0};
    cudaResourceDesc    resDesc;

    for (int i = 0; i < numPlanes; i++) {
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = levelArray[i];
        checkCudaErrors(cudaCreateSurfaceObject(&surfObject[i], &resDesc));
        dim3 threadsperBlock(16, 16);
        dim3 numBlocks((multiPlanarWidth[i] + threadsperBlock.x - 1) / threadsperBlock.x,
                       (multiPlanarHeight[i] + threadsperBlock.y - 1) / threadsperBlock.y);
        flipSurfaceBits<<<numBlocks, threadsperBlock>>>(surfObject[i], multiPlanarWidth[i], multiPlanarHeight[i]);
    }
}
