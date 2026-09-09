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
 * This sample demonstrates CUDA locality domains using the Runtime API.
 * It creates one green context and stream per reported locality domain and
 * allocates one localized memory-pool chunk per locality domain, then queries the
 * placement back from both the stream resource and the device pointer.
 */

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    std::cout << "CUDA Runtime API locality domains sample" << std::endl;

    int dev = 0;
    cudaSetDevice(dev);

    int memoryPoolsSupported = 0;
    cudaDeviceGetAttribute(&memoryPoolsSupported, cudaDevAttrMemoryPoolsSupported, dev);

    if (!memoryPoolsSupported) {
        std::cout << "Waiving execution as the device does not support memory pools." << std::endl;
        return 0;
    }

    // Device information
    int localityDomainCount = 0;
    {
        int localityDomainSmCount = 0;
        cudaDeviceGetAttribute(&localityDomainCount, cudaDevAttrLocalityDomainCount, dev);
        cudaDeviceGetAttribute(&localityDomainSmCount, cudaDevAttrLocalityDomainMultiprocessorCount, dev);

        std::cout << "Locality domain count: " << localityDomainCount << std::endl;
        std::cout << "SMs per locality domain: " << localityDomainSmCount << std::endl;
    }

    // Construct localized green contexts and streams
    std::vector<cudaExecutionContext_t> greenContexts(localityDomainCount);
    std::vector<cudaStream_t>           streams(localityDomainCount);
    {
        cudaDevResource smResource = {};
        cudaDeviceGetDevResource(dev, &smResource, cudaDevResourceTypeSm);

        std::vector<cudaDevSmResourceGroupParams> params(localityDomainCount);
        for (size_t i = 0; i < localityDomainCount; i++) {
            std::memset(&params[i], 0, sizeof(params[i]));
            params[i].flags            = cudaDevSmResourceGroupLocalityDomainId;
            params[i].localityDomainId = static_cast<unsigned int>(i);
        }

        std::vector<cudaDevResource> result(localityDomainCount);
        cudaDevResource              remainder = {};

        cudaDevSmResourceSplit(result.data(), localityDomainCount, &smResource, &remainder, 0, params.data());
        std::cout << "Device SMs split into " << localityDomainCount << " partitions" << std::endl;
        for (size_t i = 0; i < localityDomainCount; i++) {
            std::cout << " - result[" << i << "]: .localityDomainId = " << result[i].sm.localityDomainId
                      << ", .smCount = " << result[i].sm.smCount << std::endl;
        }
        if (remainder.sm.smCount > 0) {
            std::cout << " - remainder: .localityDomainId = none, .smCount = " << remainder.sm.smCount << std::endl;
        }

        for (size_t i = 0; i < localityDomainCount; i++) {
            cudaDevResourceDesc_t desc = nullptr;
            cudaDevResourceGenerateDesc(&desc, &result[i], 1);
            cudaGreenCtxCreate(&greenContexts[i], desc, dev, 0);
            cudaExecutionCtxStreamCreate(&streams[i], greenContexts[i], cudaStreamNonBlocking, 0);
        }
    }

    // Inspect locality of streams
    {
        for (size_t i = 0; i < localityDomainCount; i++) {
            cudaDevResource streamSmResource = {};
            cudaStreamGetDevResource(streams[i], &streamSmResource, cudaDevResourceTypeSm);
            bool isStreamLocalized = (streamSmResource.sm.flags & cudaDevSmResourceGroupLocalityDomainId) != 0;
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
    size_t                     allocationSize = 1 << 20;
    std::vector<void *>        localizedPtrs(localityDomainCount);
    std::vector<cudaMemPool_t> pools(localityDomainCount);
    {
        for (size_t i = 0; i < localityDomainCount; i++) {
            cudaMemPoolProps poolProps                    = {};
            poolProps.allocType                           = cudaMemAllocationTypePinned;
            poolProps.location.type                       = cudaMemLocationTypeDeviceLocalityDomain;
            poolProps.location.localized.deviceId         = static_cast<unsigned char>(dev);
            poolProps.location.localized.localityDomainId = static_cast<unsigned char>(i);

            cudaMemPoolCreate(&pools[i], &poolProps);
            cudaMallocFromPoolAsync(&localizedPtrs[i], allocationSize, pools[i], streams[i]);
            cudaStreamSynchronize(streams[i]);
        }
    }

    // Inspect locality of allocations
    {
        for (void *ptr : localizedPtrs) {
            cudaPointerAttributes pointerAttributes = {};
            cudaPointerGetAttributes(&pointerAttributes, ptr);

            std::cout << "allocation 0x" << std::hex << reinterpret_cast<std::uintptr_t>(ptr) << std::dec;
            if (pointerAttributes.localityDomainOrdinal == -1) {
                std::cout << " is not localized." << std::endl;
            }
            else {
                std::cout << " is localized to locality domain " << pointerAttributes.localityDomainOrdinal
                          << std::endl;
            }
        }
    }

    // Clean up
    for (size_t i = 0; i < localizedPtrs.size(); i++) {
        cudaFreeAsync(localizedPtrs[i], streams[i]);
    }
    for (cudaStream_t stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    for (cudaExecutionContext_t greenContext : greenContexts) {
        cudaExecutionCtxDestroy(greenContext);
    }
    for (cudaMemPool_t pool : pools) {
        cudaMemPoolDestroy(pool);
    }

    return 0;
}
