/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/*
 * This sample demonstrates CUDA locality domains using the Driver API.
 * It creates one green context and stream per reported locality domain and
 * maps one virtual-address chunk to each locality domain, then queries the
 * placement back from both the stream resource and the device pointer.
 */

#include <cstring>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    std::cout << "CUDA Driver API locality domains sample" << std::endl;

    cuInit(0);

    CUdevice dev;
    cuDeviceGet(&dev, 0);

    CUctxCreateParams ctxCreateParams = {};
    CUcontext         context;
    cuCtxCreate(&context, &ctxCreateParams, 0, dev);

    // Device information
    int localityDomainCount = 0;
    {
        cuDeviceGetAttribute(&localityDomainCount, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, dev);
        int localityDomainSmCount = 0;
        cuDeviceGetAttribute(&localityDomainSmCount, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT, dev);

        std::cout << "Locality domain count: " << localityDomainCount << std::endl;
        std::cout << "SMs per locality domain: " << localityDomainSmCount << std::endl;
    }

    // Construct localized green contexts and streams
    std::vector<CUgreenCtx> greenContexts(localityDomainCount);
    std::vector<CUstream>   streams(localityDomainCount);
    {
        CUdevResource smResource;
        cuDeviceGetDevResource(dev, &smResource, CU_DEV_RESOURCE_TYPE_SM);

        std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(localityDomainCount);
        for (size_t i = 0; i < localityDomainCount; i++) {
            std::memset(&params[i], 0, sizeof(params[i]));
            params[i].flags            = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
            params[i].localityDomainId = static_cast<unsigned int>(i);
        }

        std::vector<CUdevResource> result(localityDomainCount);
        CUdevResource              remainder = {};

        cuDevSmResourceSplit(result.data(), localityDomainCount, &smResource, &remainder, 0, params.data());
        std::cout << "Device SMs split into " << localityDomainCount << " partitions" << std::endl;
        for (size_t i = 0; i < localityDomainCount; i++) {
            std::cout << " - result[" << i << "]: .localityDomainId = " << result[i].sm.localityDomainId
                      << ", .smCount = " << result[i].sm.smCount << std::endl;
        }
        if (remainder.sm.smCount > 0) {
            std::cout << " - remainder: .localityDomainId = none, .smCount = " << remainder.sm.smCount << std::endl;
        }

        for (size_t i = 0; i < localityDomainCount; i++) {
            CUdevResourceDesc desc;
            cuDevResourceGenerateDesc(&desc, &result[i], 1);
            cuGreenCtxCreate(&greenContexts[i], desc, dev, CU_GREEN_CTX_DEFAULT_STREAM);
            cuGreenCtxStreamCreate(&streams[i], greenContexts[i], CU_STREAM_NON_BLOCKING, 0);
        }
    }

    // Inspect locality of streams
    {
        for (size_t i = 0; i < localityDomainCount; i++) {
            CUdevResource streamSmResource;
            cuStreamGetDevResource(streams[i], &streamSmResource, CU_DEV_RESOURCE_TYPE_SM);
            bool isStreamLocalized = (streamSmResource.sm.flags & CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID) != 0;
            std::cout << "Green context stream " << i << " ";
            if (isStreamLocalized) {
                unsigned int streamLocalityDomain = streamSmResource.sm.localityDomainId;
                std::cout << "is localized to locality domain " << streamLocalityDomain << std::endl;
            }
            else {
                std::cout << "is not localized" << std::endl;
            }
        }
    }

    // Allocate localized memory
    CUdeviceptr              allocationBase        = 0;
    size_t                   allocationGranularity = 0;
    size_t                   allocationSize        = 0;
    std::vector<CUdeviceptr> localizedPtrs(localityDomainCount);
    {
        {
            CUmemAllocationProp prop                 = {};
            prop.type                                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type                       = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
            prop.location.localized.deviceId         = static_cast<unsigned char>(dev);
            prop.location.localized.localityDomainId = 0;
            cuMemGetAllocationGranularity(&allocationGranularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        }
        std::cout << "Allocation granularity per chunk: " << allocationGranularity << " bytes" << std::endl;
        allocationSize = allocationGranularity * localityDomainCount;

        cuMemAddressReserve(&allocationBase, allocationSize, 0, 0, 0);
        std::cout << "Reserved VA range size: " << allocationSize << " bytes" << std::endl;

        for (size_t i = 0; i < localityDomainCount; i++) {
            localizedPtrs[i] = allocationBase + i * allocationGranularity;

            CUmemGenericAllocationHandle handle;
            CUmemAllocationProp          prop        = {};
            prop.type                                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type                       = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
            prop.location.localized.deviceId         = static_cast<unsigned char>(dev);
            prop.location.localized.localityDomainId = static_cast<unsigned char>(i);
            cuMemCreate(&handle, allocationGranularity, &prop, 0);
            cuMemMap(localizedPtrs[i], allocationGranularity, 0, handle, 0);

            CUmemAccessDesc desc                     = {};
            desc.location.type                       = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
            desc.location.localized.deviceId         = static_cast<unsigned char>(dev);
            desc.location.localized.localityDomainId = static_cast<unsigned char>(i);
            desc.flags                               = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(localizedPtrs[i], allocationGranularity, &desc, 1);
            cuMemRelease(handle);
        }
    }

    // Inspect locality of allocations
    {
        for (CUdeviceptr ptr : localizedPtrs) {
            unsigned int localityDomainId = 0;
            cuPointerGetAttribute(&localityDomainId, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, ptr);
            std::cout << "allocation 0x" << std::hex << ptr << std::dec;
            if (localityDomainId == -1) {
                std::cout << " is not localized." << std::endl;
            }
            else {
                std::cout << " is localized to locality domain " << localityDomainId << std::endl;
            }
        }
    }

    // Clean up
    for (CUstream stream : streams) {
        cuStreamDestroy(stream);
    }
    for (CUgreenCtx greenContext : greenContexts) {
        cuGreenCtxDestroy(greenContext);
    }
    for (CUdeviceptr localizedPtr : localizedPtrs) {
        cuMemUnmap(localizedPtr, allocationGranularity);
    }
    cuMemAddressFree(allocationBase, allocationSize);
    cuCtxDestroy(context);

    return 0;
}
