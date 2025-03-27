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

#include <cuda_runtime.h>
#include <helper_image.h>
#include <iostream>

#include "cuda_consumer.h"
#include "nvmedia_image_nvscibuf.h"
#include "nvmedia_utils/cmdline.h"

// Enable this to 1 if require cuda processed output to ppm file.
#define WRITE_OUTPUT_IMAGE 0

#define checkNvSciErrors(call)                                   \
    do {                                                         \
        NvSciError _status = call;                               \
        if (NvSciError_Success != _status) {                     \
            printf("NVSCI call in file '%s' in line %i returned" \
                   " %d, expected %d\n",                         \
                   __FILE__,                                     \
                   __LINE__,                                     \
                   _status,                                      \
                   NvSciError_Success);                          \
            fflush(stdout);                                      \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

__global__ static void
yuvToGrayscale(cudaSurfaceObject_t surfaceObject, unsigned int *dstImage, int32_t imageWidth, int32_t imageHeight)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 *dstImageUchar4 = (uchar4 *)dstImage;
    for (; x < imageWidth && y < imageHeight; x += gridDim.x * blockDim.x, y += gridDim.y * blockDim.y) {
        int           colInBytes   = x * sizeof(unsigned char);
        unsigned char luma         = surf2Dread<unsigned char>(surfaceObject, colInBytes, y);
        uchar4        grayscalePix = make_uchar4(luma, luma, luma, 0);

        dstImageUchar4[y * imageWidth + x] = grayscalePix;
    }
}

static void cudaImportNvSciSync(cudaExternalSemaphore_t &extSem, NvSciSyncObj &syncObj)
{
    cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type                = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = (void *)syncObj;

    checkCudaErrors(cudaImportExternalSemaphore(&extSem, &extSemDesc));
}

static void waitExternalSemaphore(cudaExternalSemaphore_t &waitSem, NvSciSyncFence *fence, cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    // For cross-process signaler-waiter applications need to use NvSciIpc
    // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
    // across process. This step is optional in single-process.
    waitParams.params.nvSciSync.fence = (void *)fence;
    waitParams.flags                  = 0;

    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&waitSem, &waitParams, 1, stream));
}

static void signalExternalSemaphore(cudaExternalSemaphore_t &signalSem, NvSciSyncFence *fence, cudaStream_t stream)
{
    cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    // For cross-process signaler-waiter applications need to use NvSciIpc
    // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
    // across process. This step is optional in single-process.
    signalParams.params.nvSciSync.fence = (void *)fence;
    signalParams.flags                  = 0;

    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&signalSem, &signalParams, 1, stream));
}

static void yuvToGrayscaleCudaKernel(cudaExternalResInterop &cudaExtResObj, int32_t imageWidth, int32_t imageHeight)
{
#if WRITE_OUTPUT_IMAGE
    unsigned int *h_dstImage;
    checkCudaErrors(cudaMallocHost(&h_dstImage, sizeof(unsigned int) * imageHeight * imageWidth));
#endif
    dim3 block(16, 16, 1);
    dim3 grid((imageWidth / block.x) + 1, (imageHeight / block.y) + 1, 1);

    yuvToGrayscale<<<grid, block, 0, cudaExtResObj.stream>>>(
        cudaExtResObj.cudaSurfaceNvmediaBuf[0], cudaExtResObj.d_outputImage, imageWidth, imageHeight);

#if WRITE_OUTPUT_IMAGE
    checkCudaErrors(cudaMemcpyAsync(h_dstImage,
                                    cudaExtResObj.d_outputImage,
                                    sizeof(unsigned int) * imageHeight * imageWidth,
                                    cudaMemcpyDeviceToHost,
                                    cudaExtResObj.stream));
    checkCudaErrors(cudaStreamSynchronize(cudaExtResObj.stream));
    char        outputFilename[1024];
    std::string image_filename = "Grayscale";
    strcpy(outputFilename, image_filename.c_str());
    strcpy(outputFilename + image_filename.length(), "_nvsci_out.ppm");
    sdkSavePPM4ub(outputFilename, (unsigned char *)h_dstImage, imageWidth, imageHeight);
    printf("Wrote '%s'\n", outputFilename);
    checkCudaErrors(cudaFreeHost(h_dstImage));
#endif
}

static void cudaImportNvSciImage(cudaExternalResInterop &cudaExtResObj, NvSciBufObj &inputBufObj)
{
    NvSciBufModule           module   = NULL;
    NvSciBufAttrList         attrlist = NULL;
    NvSciBufAttrKeyValuePair pairArrayOut[10];

    checkNvSciErrors(NvSciBufModuleOpen(&module));
    checkNvSciErrors(NvSciBufAttrListCreate(module, &attrlist));
    checkNvSciErrors(NvSciBufObjGetAttrList(inputBufObj, &attrlist));

    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);

    int numAttrs                 = 0;
    pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_Size;
    pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneChannelCount;
    pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneCount;
    pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneWidth;
    pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneHeight;
    pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_Layout;
    pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneBitsPerPixel;
    pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneOffset;

    checkNvSciErrors(NvSciBufAttrListGetAttrs(attrlist, pairArrayOut, numAttrs));

    uint64_t size             = *(uint64_t *)pairArrayOut[0].value;
    uint8_t  channelCount     = *(uint8_t *)pairArrayOut[1].value;
    cudaExtResObj.planeCount  = *(int32_t *)pairArrayOut[2].value;
    cudaExtResObj.imageWidth  = (int32_t *)malloc(sizeof(int32_t) * cudaExtResObj.planeCount);
    cudaExtResObj.imageHeight = (int32_t *)malloc(sizeof(int32_t) * cudaExtResObj.planeCount);
    cudaExtResObj.planeOffset = (uint64_t *)malloc(sizeof(uint64_t) * cudaExtResObj.planeCount);

    memcpy(cudaExtResObj.imageWidth, (int32_t *)pairArrayOut[3].value, cudaExtResObj.planeCount * sizeof(int32_t));
    memcpy(cudaExtResObj.imageHeight, (int32_t *)pairArrayOut[4].value, cudaExtResObj.planeCount * sizeof(int32_t));
    memcpy(cudaExtResObj.planeOffset, (uint64_t *)pairArrayOut[7].value, cudaExtResObj.planeCount * sizeof(uint64_t));

    NvSciBufAttrValImageLayoutType layout       = *(NvSciBufAttrValImageLayoutType *)pairArrayOut[5].value;
    uint32_t                       bitsPerPixel = *(uint32_t *)pairArrayOut[6].value;

    if (layout != NvSciBufImage_BlockLinearType) {
        printf("Image layout is not block linear.. waiving execution\n");
        exit(EXIT_WAIVED);
    }

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type                  = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = inputBufObj;
    memHandleDesc.size                  = size;
    checkCudaErrors(cudaImportExternalMemory(&cudaExtResObj.extMemImageBuf, &memHandleDesc));

    cudaExtResObj.d_mipmapArray =
        (cudaMipmappedArray_t *)malloc(sizeof(cudaMipmappedArray_t) * cudaExtResObj.planeCount);

    for (int i = 0; i < cudaExtResObj.planeCount; i++) {
        cudaExtent extent = {};
        memset(&extent, 0, sizeof(extent));
        extent.width  = cudaExtResObj.imageWidth[i];
        extent.height = cudaExtResObj.imageHeight[i];
        extent.depth  = 0;
        cudaChannelFormatDesc desc;
        switch (channelCount) {
        case 1:
        default:
            desc = cudaCreateChannelDesc(bitsPerPixel, 0, 0, 0, cudaChannelFormatKindUnsigned);
            break;
        case 2:
            desc = cudaCreateChannelDesc(bitsPerPixel, bitsPerPixel, 0, 0, cudaChannelFormatKindUnsigned);
            break;
        case 3:
            desc = cudaCreateChannelDesc(bitsPerPixel, bitsPerPixel, bitsPerPixel, 0, cudaChannelFormatKindUnsigned);
            break;
        case 4:
            desc = cudaCreateChannelDesc(
                bitsPerPixel, bitsPerPixel, bitsPerPixel, bitsPerPixel, cudaChannelFormatKindUnsigned);
            break;
        }

        cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {0};
        mipmapDesc.offset                               = cudaExtResObj.planeOffset[i];
        mipmapDesc.formatDesc                           = desc;
        mipmapDesc.extent                               = extent;
        mipmapDesc.flags                                = 0;
        mipmapDesc.numLevels                            = 1;
        checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(
            &cudaExtResObj.d_mipmapArray[i], cudaExtResObj.extMemImageBuf, &mipmapDesc));
    }
}

static cudaSurfaceObject_t createCudaSurface(cudaArray_t &d_mipLevelArray)
{
    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType         = cudaResourceTypeArray;
    resourceDesc.res.array.array = d_mipLevelArray;

    cudaSurfaceObject_t surfaceObject;
    checkCudaErrors(cudaCreateSurfaceObject(&surfaceObject, &resourceDesc));
    return surfaceObject;
}

static cudaStream_t createCudaStream(int deviceId)
{
    checkCudaErrors(cudaSetDevice(deviceId));
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return stream;
}

// CUDA setup buffers/synchronization objects for interop via NvSci API.
void setupCuda(cudaExternalResInterop &cudaExtResObj,
               NvSciBufObj            &inputBufObj,
               NvSciSyncObj           &syncObj,
               NvSciSyncObj           &cudaSignalerSyncObj,
               int                     deviceId)
{
    checkCudaErrors(cudaSetDevice(deviceId));
    cudaImportNvSciSync(cudaExtResObj.waitSem, syncObj);
    cudaImportNvSciSync(cudaExtResObj.signalSem, cudaSignalerSyncObj);

    cudaImportNvSciImage(cudaExtResObj, inputBufObj);
    cudaExtResObj.d_mipLevelArray = (cudaArray_t *)malloc(sizeof(cudaArray_t) * cudaExtResObj.planeCount);
    cudaExtResObj.cudaSurfaceNvmediaBuf =
        (cudaSurfaceObject_t *)malloc(sizeof(cudaSurfaceObject_t) * cudaExtResObj.planeCount);

    for (int i = 0; i < cudaExtResObj.planeCount; ++i) {
        uint32_t mipLevelId = 0;
        checkCudaErrors(
            cudaGetMipmappedArrayLevel(&cudaExtResObj.d_mipLevelArray[i], cudaExtResObj.d_mipmapArray[i], mipLevelId));
        cudaExtResObj.cudaSurfaceNvmediaBuf[i] = createCudaSurface(cudaExtResObj.d_mipLevelArray[i]);
    }

    cudaExtResObj.stream = createCudaStream(deviceId);
    checkCudaErrors(cudaMalloc(&cudaExtResObj.d_outputImage,
                               sizeof(unsigned int) * cudaExtResObj.imageWidth[0] * cudaExtResObj.imageHeight[0]));
}

// CUDA clean up buffers used **with** NvSci API.
void cleanupCuda(cudaExternalResInterop &cudaExtResObj)
{
    for (int i = 0; i < cudaExtResObj.planeCount; i++) {
        checkCudaErrors(cudaDestroySurfaceObject(cudaExtResObj.cudaSurfaceNvmediaBuf[i]));
        checkCudaErrors(cudaFreeMipmappedArray(cudaExtResObj.d_mipmapArray[i]));
    }
    free(cudaExtResObj.d_mipmapArray);
    free(cudaExtResObj.d_mipLevelArray);
    free(cudaExtResObj.cudaSurfaceNvmediaBuf);
    free(cudaExtResObj.imageWidth);
    free(cudaExtResObj.imageHeight);
    checkCudaErrors(cudaDestroyExternalSemaphore(cudaExtResObj.waitSem));
    checkCudaErrors(cudaDestroyExternalSemaphore(cudaExtResObj.signalSem));
    checkCudaErrors(cudaDestroyExternalMemory(cudaExtResObj.extMemImageBuf));
    checkCudaErrors(cudaStreamDestroy(cudaExtResObj.stream));
    checkCudaErrors(cudaFree(cudaExtResObj.d_outputImage));
}

void runCudaOperation(cudaExternalResInterop &cudaExtResObj,
                      NvSciSyncFence         *cudaWaitFence,
                      NvSciSyncFence         *cudaSignalFence,
                      int                     deviceId,
                      int                     iterations)
{
    checkCudaErrors(cudaSetDevice(deviceId));
    static int64_t launch = 0;

    waitExternalSemaphore(cudaExtResObj.waitSem, cudaWaitFence, cudaExtResObj.stream);

    // run cuda kernel over surface object of the LUMA surface part to extract
    // grayscale.
    yuvToGrayscaleCudaKernel(cudaExtResObj, cudaExtResObj.imageWidth[0], cudaExtResObj.imageHeight[0]);

    // signal fence till the second last iterations for NvMedia2DBlit to wait for
    // cuda signal and for final iteration as there is no corresponding NvMedia
    // operation pending therefore we end with cudaStreamSynchronize()
    if (launch < iterations - 1) {
        signalExternalSemaphore(cudaExtResObj.signalSem, cudaSignalFence, cudaExtResObj.stream);
    }
    else {
        checkCudaErrors(cudaStreamSynchronize(cudaExtResObj.stream));
    }
    launch++;
}

// CUDA imports and operates on NvSci buffer/synchronization objects
void setupCuda(Blit2DTest *ctx, cudaResources &cudaResObj, int deviceId)
{
    checkCudaErrors(cudaSetDevice(deviceId));
    cudaResObj.d_yuvArray            = (cudaArray_t *)malloc(sizeof(cudaArray_t) * ctx->numSurfaces);
    cudaResObj.cudaSurfaceNvmediaBuf = (cudaSurfaceObject_t *)malloc(sizeof(cudaSurfaceObject_t) * ctx->numSurfaces);
    cudaChannelFormatDesc channelDesc;
    switch (ctx->bytesPerPixel) {
    case 1:
    default:
        channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        break;
    }

    for (int k = 0; k < ctx->numSurfaces; k++) {
        checkCudaErrors(cudaMallocArray(&cudaResObj.d_yuvArray[k],
                                        &channelDesc,
                                        ctx->widthSurface * ctx->xScalePtr[k] * ctx->bytesPerPixel,
                                        ctx->heightSurface * ctx->yScalePtr[k]));
        cudaResObj.cudaSurfaceNvmediaBuf[k] = createCudaSurface(cudaResObj.d_yuvArray[k]);
    }
    checkCudaErrors(
        cudaMalloc(&cudaResObj.d_outputImage, sizeof(unsigned int) * ctx->widthSurface * ctx->heightSurface));

    cudaResObj.stream = createCudaStream(deviceId);
}

// CUDA clean up buffers used **without** NvSci API.
void cleanupCuda(Blit2DTest *ctx, cudaResources &cudaResObj)
{
    for (int k = 0; k < ctx->numSurfaces; k++) {
        checkCudaErrors(cudaDestroySurfaceObject(cudaResObj.cudaSurfaceNvmediaBuf[k]));
        checkCudaErrors(cudaFreeArray(cudaResObj.d_yuvArray[k]));
    }

    free(cudaResObj.cudaSurfaceNvmediaBuf);

    checkCudaErrors(cudaStreamDestroy(cudaResObj.stream));
    checkCudaErrors(cudaFree(cudaResObj.d_outputImage));
}

static void
yuvToGrayscaleCudaKernelNonNvSci(cudaResources &cudaResObj, int deviceId, int32_t imageWidth, int32_t imageHeight)
{
#if WRITE_OUTPUT_IMAGE
    unsigned int *h_dstImage;
    checkCudaErrors(cudaMallocHost(&h_dstImage, sizeof(unsigned int) * imageHeight * imageWidth));
#endif
    dim3 block(16, 16, 1);
    dim3 grid((imageWidth / block.x) + 1, (imageHeight / block.y) + 1, 1);

    yuvToGrayscale<<<grid, block, 0, cudaResObj.stream>>>(
        cudaResObj.cudaSurfaceNvmediaBuf[0], cudaResObj.d_outputImage, imageWidth, imageHeight);

#if WRITE_OUTPUT_IMAGE
    checkCudaErrors(cudaMemcpyAsync(h_dstImage,
                                    cudaResObj.d_outputImage,
                                    sizeof(unsigned int) * imageHeight * imageWidth,
                                    cudaMemcpyDeviceToHost,
                                    cudaResObj.stream));
    checkCudaErrors(cudaStreamSynchronize(cudaResObj.stream));
    char        outputFilename[1024];
    std::string image_filename = "Grayscale";
    strcpy(outputFilename, image_filename.c_str());
    strcpy(outputFilename + image_filename.length(), "_non-nvsci_out.ppm");
    sdkSavePPM4ub(outputFilename, (unsigned char *)h_dstImage, imageWidth, imageHeight);
    printf("Wrote '%s'\n", outputFilename);
    checkCudaErrors(cudaFreeHost(h_dstImage));
#else
    checkCudaErrors(cudaStreamSynchronize(cudaResObj.stream));
#endif
}

// CUDA operates **without** NvSci APIs buffer/synchronization objects.
void runCudaOperation(Blit2DTest *ctx, cudaResources &cudaResObj, int deviceId)
{
    for (int k = 0; k < ctx->numSurfaces; k++) {
        checkCudaErrors(cudaMemcpy2DToArray(cudaResObj.d_yuvArray[k],
                                            0,
                                            0,
                                            ctx->dstBuff[k],
                                            ctx->widthSurface * ctx->xScalePtr[k] * ctx->bytesPerPixel,
                                            ctx->widthSurface * ctx->xScalePtr[k] * ctx->bytesPerPixel,
                                            ctx->heightSurface * ctx->yScalePtr[k],
                                            cudaMemcpyHostToDevice));
    }
    // run cuda kernel over surface object of the LUMA surface part to extract
    // grayscale.
    yuvToGrayscaleCudaKernelNonNvSci(cudaResObj, deviceId, ctx->widthSurface, ctx->heightSurface);
}
