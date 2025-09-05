/* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime_api.h>

#include <chrono>
#include <mutex>
#include <span>
#include <stdio.h>
#include <vector>

constexpr uint32_t memblockSize = 40 * 1024 * 1024;
constexpr uint32_t stepTime     = 10; // The time to wait between allocation steps in ms
constexpr uint64_t maxSteps     = 500;

// This sample uses the cudaDeviceRegisterAsyncNotification API to allocate as much video memory as it can.
// A notification to free some video memory is sent whenever the application goes over its budget. If multiple
// instances of this program are launched, all instances will likely end up with similar budgets and in 
// steady-state will have similar amounts of video memory allocated.

int main()
{
    // This structure will be our 'userData' to pass into the callback. This will give the
    // callback state that it can affect without needing to use globals.
    struct MemoryInfo {
        std::vector<int*> memblocks;
        std::mutex mutex;
        bool allowedToGrow = true;
    } memoryInfo;

    // The function that we will register as a callback. This could also be a standalone function
    // instead of a lambda, but, since it is only referenced here, we can use a lambda to avoid
    // using the global namespace.
    cudaAsyncCallback basicUserCallback = [](cudaAsyncNotificationInfo_t* notificationInfo, void* userData, cudaAsyncCallbackHandle_t callback)
    {
        MemoryInfo* memoryInfo = (MemoryInfo*)userData;

        // Must check the type before accessing the info member of cudaAsyncNotificationInfo_t.
        // Otherwise, we could misinterpret notificationInfo->info if a different type of
        // notification is sent.
        if (notificationInfo->type == cudaAsyncNotificationTypeOverBudget) {
            printf("Asked to free %lld bytes of video memory\n", notificationInfo->info.overBudget.bytesOverBudget);
            uint64_t numBlocksToFree = (notificationInfo->info.overBudget.bytesOverBudget / memblockSize) + 1;

            // The async notification will be on a separate thread, so shared data should be synchronized.
            std::scoped_lock lock(memoryInfo->mutex);
            memoryInfo->allowedToGrow = false;
            
            // Free the required number of memblocks
            std::span<int*> blocksToFree(memoryInfo->memblocks.end() - numBlocksToFree, numBlocksToFree);

            for (auto& block : blocksToFree) {
                cudaFree(block);
                block = nullptr;
            }

            // Shrink the vector to remove the freed blocks
            std::erase(memoryInfo->memblocks, nullptr);
        }
    };

    // Initialize CUDA
    const int cudaDevice = 0;
    cudaSetDevice(cudaDevice);

    // This callback handle is an opaque object that can be used to unregister the notification via
    // cudaDeviceUnregisterAsyncNotification and to identify which callback registration a given
    // notification corresponds to.
    cudaAsyncCallbackHandle_t callback;
    cudaDeviceRegisterAsyncNotification(cudaDevice, basicUserCallback, (void*)&memoryInfo, &callback);

    // Attempt to allocate a block of memory, then sleep before repeating.
    uint64_t stepCounter = 0;
    while (stepCounter < maxSteps)
    {
        {
            // The async notification will be on a separate thread, so shared data should be synchronized.
            std::scoped_lock lock(memoryInfo.mutex);

            if (memoryInfo.allowedToGrow)
            {
                int* newMemblock;
                cudaError_t cudaStatus = cudaMalloc((void**)&newMemblock, memblockSize);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                }
                memoryInfo.memblocks.push_back(newMemblock);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(stepTime));
        stepCounter++;
    }

    printf("Current memory allocated: %lld MB\n", memoryInfo.memblocks.size() * memblockSize / 1024 / 1024);

    // Unregister callback handle in application cleanup
    cudaDeviceUnregisterAsyncNotification(cudaDevice, callback);

    return 0;
}
