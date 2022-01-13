/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include <string.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

cudaError_t setProp(CUmemAllocationProp *prop, bool UseCompressibleMemory)
{
    CUdevice currentDevice;
    if (cuCtxGetDevice(&currentDevice) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    memset(prop, 0, sizeof(CUmemAllocationProp));
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = currentDevice;

    if (UseCompressibleMemory)
        prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    return cudaSuccess;
}

cudaError_t allocateCompressible(void **adr, size_t size, bool UseCompressibleMemory)
{
    CUmemAllocationProp prop = {};
    cudaError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != cudaSuccess)
        return err;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;
    CUdeviceptr dptr;
    if (cuMemAddressReserve(&dptr, size, 0, 0, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    CUmemGenericAllocationHandle allocationHandle;
    if (cuMemCreate(&allocationHandle, size, &prop, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    // Check if cuMemCreate was able to allocate compressible memory.
    if (UseCompressibleMemory) {
        CUmemAllocationProp allocationProp = {};
        cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
        if (allocationProp.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
            printf("Could not allocate compressible memory... so waiving execution\n");
            exit(EXIT_WAIVED);
        }
    }

    if (cuMemMap(dptr, size, 0, allocationHandle, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    if (cuMemRelease(allocationHandle) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    if (cuMemSetAccess(dptr, size, &accessDescriptor, 1) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    *adr = (void *)dptr;
    return cudaSuccess;
}

cudaError_t freeCompressible(void *ptr, size_t size, bool UseCompressibleMemory)
{
    CUmemAllocationProp prop = {};
    cudaError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != cudaSuccess)
        return err;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;

    if (ptr == NULL)
        return cudaSuccess;
    if (cuMemUnmap((CUdeviceptr)ptr, size) != CUDA_SUCCESS ||
        cuMemAddressFree((CUdeviceptr)ptr, size) != CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    return cudaSuccess;
}
