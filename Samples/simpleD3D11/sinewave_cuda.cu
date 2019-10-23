/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
#include "ShaderStructs.h"
#include "helper_cuda.h"
#include "sinewave_cuda.h"

__global__ void sinewave_gen_kernel(Vertex *vertices, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    if (y < height && x < width)
    {
        // write output vertex
        vertices[y*width+x].position.x = u;
        vertices[y*width+x].position.y = w;
        vertices[y*width+x].position.z = v;
        vertices[y*width+x].color.x = 1.0f;
        vertices[y*width+x].color.y = 0.0f;
        vertices[y*width+x].color.z = 0.0f;
        vertices[y*width + x].color.w = 0.0f;
    }
}

Vertex* cudaImportVertexBuffer(void*sharedHandle, cudaExternalMemory_t &externalMemory, int meshWidth, int meshHeight)
{
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D11ResourceKmt;
    externalMemoryHandleDesc.size = sizeof(Vertex) * meshHeight * meshWidth;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;

    checkCudaErrors(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));

    cudaExternalMemoryBufferDesc externalMemoryBufferDesc;
    memset(&externalMemoryBufferDesc, 0, sizeof(externalMemoryBufferDesc));
    externalMemoryBufferDesc.offset = 0;
    externalMemoryBufferDesc.size = sizeof(Vertex) * meshHeight * meshWidth;
    externalMemoryBufferDesc.flags = 0;

    Vertex* cudaDevVertptr = NULL;
    checkCudaErrors(cudaExternalMemoryGetMappedBuffer((void**)&cudaDevVertptr, externalMemory, &externalMemoryBufferDesc));

    return cudaDevVertptr;
}

void cudaImportKeyedMutex(void*sharedHandle, cudaExternalSemaphore_t &extSemaphore)
{
    cudaExternalSemaphoreHandleDesc extSemaDesc;
    memset(&extSemaDesc, 0, sizeof(extSemaDesc));
    extSemaDesc.type = cudaExternalSemaphoreHandleTypeKeyedMutexKmt;
    extSemaDesc.handle.win32.handle = sharedHandle;
    extSemaDesc.flags = 0;

    checkCudaErrors(cudaImportExternalSemaphore(&extSemaphore, &extSemaDesc));
}

void cudaAcquireSync(cudaExternalSemaphore_t &extSemaphore, uint64_t key, unsigned int timeoutMs, cudaStream_t streamToRun)
{
    cudaExternalSemaphoreWaitParams extSemWaitParams;
    memset(&extSemWaitParams, 0, sizeof(extSemWaitParams));
    extSemWaitParams.params.keyedMutex.key = key;
    extSemWaitParams.params.keyedMutex.timeoutMs = timeoutMs;

    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&extSemaphore, &extSemWaitParams, 1, streamToRun));
}

void cudaReleaseSync(cudaExternalSemaphore_t &extSemaphore, uint64_t key, cudaStream_t streamToRun)
{
    cudaExternalSemaphoreSignalParams extSemSigParams;
    memset(&extSemSigParams, 0, sizeof(extSemSigParams));
    extSemSigParams.params.keyedMutex.key = key;

    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&extSemaphore, &extSemSigParams, 1, streamToRun));
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void RunSineWaveKernel(cudaExternalSemaphore_t &extSemaphore, uint64_t &key, unsigned int timeoutMs, 
                        size_t mesh_width, size_t mesh_height, Vertex *cudaDevVertptr, cudaStream_t streamToRun)
{
    static float t = 0.0f;
    cudaAcquireSync(extSemaphore, key++, timeoutMs, streamToRun);

    dim3 block(16, 16, 1);
    dim3 grid(mesh_width / 16, mesh_height / 16, 1);
    sinewave_gen_kernel<<< grid, block, 0, streamToRun >>>(cudaDevVertptr, mesh_width, mesh_height, t);
    getLastCudaError("sinewave_gen_kernel execution failed.\n");

    cudaReleaseSync(extSemaphore, key, streamToRun);
    t += 0.01f;
}

