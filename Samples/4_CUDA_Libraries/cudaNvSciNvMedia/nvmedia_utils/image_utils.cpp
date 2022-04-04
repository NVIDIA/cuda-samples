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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "image_utils.h"
#include "misc_utils.h"
#include "nvmedia_surface.h"

#define MAXM_NUM_SURFACES 6

typedef struct {
    float heightFactor[6];
    float widthFactor[6];
    unsigned int numSurfaces;
} ImgUtilSurfParams;

ImgUtilSurfParams ImgSurfParamsTable_RGBA  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

ImgUtilSurfParams ImgSurfParamsTable_RAW  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

ImgUtilSurfParams ImgSurfParamsTable_YUV[][4] = {
    { /* PLANAR */
        { /* 420 */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 0.5, 0.5, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 422 */
            .heightFactor = {1, 1, 1, 0, 0, 0},
            .widthFactor = {1, 0.5, 0.5, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 444 */
            .heightFactor = {1, 1, 1, 0, 0, 0},
            .widthFactor = {1, 1, 1, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 422R */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 1, 1, 0, 0, 0},
            .numSurfaces = 3,
        },
    },
    { /* SEMI_PLANAR */
        { /* 420 */
            .heightFactor = {1, 0.5, 0, 0, 0, 0},
            .widthFactor = {1, 0.5, 0, 0, 0, 0},
            .numSurfaces = 2,
        },
        { /* 422 */
            .heightFactor = {1, 1, 0, 0, 0, 0},
            .widthFactor = {1, 0.5, 0, 0, 0, 0},
            .numSurfaces = 2,
        },
        { /* 444 */
            .heightFactor = {1, 1, 0.5, 0, 0, 0},
            .widthFactor = {1, 1, 0.5, 0, 0, 0},
            .numSurfaces = 2,
        },
        { /* 422R */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 1, 0.5, 0, 0, 0},
            .numSurfaces = 2,
        },
    },
    { /* PACKED */
        { /* 420 */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 0.5, 0.5, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 422 */
            .heightFactor = {1, 1, 1, 0, 0, 0},
            .widthFactor = {1, 0.5, 0.5, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 444 */
            .heightFactor = {1, 1, 1, 0, 0, 0},
            .widthFactor = {1, 1, 1, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 422R */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 1, 1, 0, 0, 0},
            .numSurfaces = 3,
        },
    },
};

ImgUtilSurfParams ImgSurfParamsTable_Packed  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};


unsigned int ImgBytesPerPixelTable_RGBA[][6] = {
    {4, 0, 0, 0, 0, 0}, /* 8 */
};

unsigned int ImgBytesPerPixelTable_RGBA16[][6] = {
    {8, 0, 0, 0, 0, 0}, /* 16 */
};

unsigned int ImgBytesPerPixelTable_RG16[6] =
    {4, 0, 0, 0, 0, 0};

unsigned int ImgBytesPerPixelTable_Alpha[][6] = {
    {1, 0, 0, 0, 0, 0}, /* 8 */
    {2, 0, 0, 0, 0, 0}, /* 10 */
    {2, 0, 0, 0, 0, 0}, /* 12 */
    {2, 0, 0, 0, 0, 0}, /* 14 */
    {2, 0, 0, 0, 0, 0}, /* 16 */
    {4, 0, 0, 0, 0, 0}, /* 32 */
};

unsigned int ImgBytesPerPixelTable_RAW[][6] = {
    {1, 0, 0, 0, 0, 0}, /* 8 */
    {2, 0, 0, 0, 0, 0}, /* 10 */
    {2, 0, 0, 0, 0, 0}, /* 12 */
    {2, 0, 0, 0, 0, 0}, /* 14 */
    {2, 0, 0, 0, 0, 0}, /* 16 */
    {4, 0, 0, 0, 0, 0}, /* 32 */
    {4, 0, 0, 0, 0, 0}, /* 16_8_8 */
    {4, 0, 0, 0, 0, 0}, /* 10_8_8 */
    {4, 0, 0, 0, 0, 0}, /* 2_10_10_10 */
    {4, 0, 0, 0, 0, 0}, /* 20 */
};

unsigned int ImgBytesPerPixelTable_YUV[][9][6] = {
    { /* PLANAR */
        {1, 1, 1, 0, 0, 0}, /* 8 */
        {2, 2, 2, 0, 0, 0}, /* 10 */
        {2, 2, 2, 0, 0, 0}, /* 12 */
        {2, 2, 2, 0, 0, 0}, /* 14 */
        {2, 2, 2, 0, 0, 0}, /* 16 */
        {4, 4, 4, 0, 0, 0}, /* 32 */
        {2, 1, 1, 0, 0, 0}, /* 16_8_8 */
        {2, 1, 1, 0, 0, 0}, /* 10_8_8 */
        {4, 0, 0, 0, 0, 0}, /* 2_10_10_10 */
    },
    { /* SEMI_PLANAR */
        {1, 2, 0, 0, 0, 0}, /* 8 */
        {2, 4, 0, 0, 0, 0}, /* 10 */
        {2, 4, 0, 0, 0, 0}, /* 12 */
        {2, 4, 0, 0, 0, 0}, /* 14 */
        {2, 4, 0, 0, 0, 0}, /* 16 */
        {4, 8, 0, 0, 0, 0}, /* 32 */
        {2, 2, 0, 0, 0, 0}, /* 16_8_8 */
        {2, 2, 0, 0, 0, 0}, /* 10_8_8 */
        {4, 0, 0, 0, 0, 0}, /* 2_10_10_10 */
    }
};

static NvMediaStatus
GetBytesPerCompForPackedYUV(unsigned int surfBPCidx,
                unsigned int *bytespercomp
)
{
    switch(surfBPCidx) {
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_8:
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_LAYOUT_2_10_10_10:
        *bytespercomp = 1;
        break;
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_10:
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_12:
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_14:
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_16:
        *bytespercomp = 2;
        break;
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_20:
        *bytespercomp = 3;
        break;
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_32:
        *bytespercomp = 4;
        break;
    default:
        return NVMEDIA_STATUS_ERROR;
    }
    return NVMEDIA_STATUS_OK;

}

static NvMediaStatus
GetSurfParams(unsigned int surfaceType,
             float **xScale,
             float **yScale,
             unsigned int **bytePerPixel,
             uint32_t *numSurfacesVal)
{
    NvMediaStatus status;
    unsigned int surfType, surfMemoryType, surfSubSamplingType, surfBPC, surfCompOrder;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);
    uint32_t numSurfaces = 1;
    static unsigned int yuvpackedtbl[6] = {1, 0, 0, 0, 0, 0};
    unsigned int numcomps = 1;

    status = NvMediaSurfaceFormatGetAttrs(surfaceType,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status != NVMEDIA_STATUS_OK) {
        printf("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return NVMEDIA_STATUS_ERROR;
    }

    surfType = srcAttr[NVM_SURF_ATTR_SURF_TYPE].value;
    surfMemoryType = srcAttr[NVM_SURF_ATTR_MEMORY].value;
    surfSubSamplingType = srcAttr[NVM_SURF_ATTR_SUB_SAMPLING_TYPE].value;
    surfBPC = srcAttr[NVM_SURF_ATTR_BITS_PER_COMPONENT].value;
    surfCompOrder = srcAttr[NVM_SURF_ATTR_COMPONENT_ORDER].value;

    switch(surfType) {
        case NVM_SURF_ATTR_SURF_TYPE_YUV:
            if (surfSubSamplingType == NVM_SURF_ATTR_SUB_SAMPLING_TYPE_NONE &&
                surfMemoryType == NVM_SURF_ATTR_MEMORY_PACKED) {

                xScalePtr =  &ImgSurfParamsTable_Packed.widthFactor[0];
                yScalePtr = &ImgSurfParamsTable_Packed.heightFactor[0];
                numSurfaces = ImgSurfParamsTable_Packed.numSurfaces;

                if (NVMEDIA_STATUS_OK != GetBytesPerCompForPackedYUV(surfBPC, &yuvpackedtbl[0])) {
                    printf("Invalid Bits per component and Packed YUV combination\n");
                    return NVMEDIA_STATUS_ERROR;
                }

                switch(surfCompOrder) {
                    case NVM_SURF_ATTR_COMPONENT_ORDER_VUYX:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_XYUV:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_XUYV:
                        numcomps = 4;
                        break;
                    case NVM_SURF_ATTR_COMPONENT_ORDER_UYVY:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_VYUY:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_YVYU:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_YUYV:
                        numcomps = 2;
                        break;
                    case NVM_SURF_ATTR_COMPONENT_ORDER_LUMA:
                        numcomps = 1;
                        break;
                    default:
                        printf("Invalid component Order  and Packed YUV combination\n");
                        return NVMEDIA_STATUS_ERROR;
                }
                yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
                bytePerPixelPtr = &yuvpackedtbl[0];

            } else {
                xScalePtr = &ImgSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].widthFactor[0];
                yScalePtr = &ImgSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].heightFactor[0];
                numSurfaces = ImgSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].numSurfaces;
                bytePerPixelPtr = &ImgBytesPerPixelTable_YUV[0][surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            }

            break;
        case NVM_SURF_ATTR_SURF_TYPE_RGBA:
            if (surfCompOrder == NVM_SURF_ATTR_COMPONENT_ORDER_ALPHA) {
                bytePerPixelPtr = &ImgBytesPerPixelTable_Alpha[surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            } else if (surfCompOrder == NVM_SURF_ATTR_COMPONENT_ORDER_RG) {
                if(surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_16) {
                    bytePerPixelPtr = &ImgBytesPerPixelTable_RG16[0];
                } else {
                    printf("Invalid RGorder & Bitspercomp combination.Only RG16 is supported\n");
                    return NVMEDIA_STATUS_ERROR;
                }
            } else { /* RGBA, ARGB, BGRA */
                if (surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_16) {
                    bytePerPixelPtr = &ImgBytesPerPixelTable_RGBA16[0][0];
                } else if (surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_8) {
                    bytePerPixelPtr = &ImgBytesPerPixelTable_RGBA[0][0];
                } else {
                    printf("RGBA orders with 8 and 16bits only is supported \n");
                    return NVMEDIA_STATUS_ERROR;
                }
            }
            xScalePtr = &ImgSurfParamsTable_RGBA.widthFactor[0];
            yScalePtr = &ImgSurfParamsTable_RGBA.heightFactor[0];
            numSurfaces =  ImgSurfParamsTable_RGBA.numSurfaces;
            break;
        case NVM_SURF_ATTR_SURF_TYPE_RAW:
            bytePerPixelPtr = &ImgBytesPerPixelTable_RAW[surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            xScalePtr = &ImgSurfParamsTable_RAW.widthFactor[0];
            yScalePtr = &ImgSurfParamsTable_RAW.heightFactor[0];
            numSurfaces =  ImgSurfParamsTable_RAW.numSurfaces;
            break;
        default:
            printf("%s: Unsupported Pixel Format %d", __func__, surfType);
            return NVMEDIA_STATUS_ERROR;
    }

    if (xScale) {
        *xScale = xScalePtr;
    }
    if (yScale) {
        *yScale = yScalePtr;
    }
    if (bytePerPixel) {
        *bytePerPixel = bytePerPixelPtr;
    }
    if (numSurfacesVal) {
        *numSurfacesVal = numSurfaces;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
AllocateBufferToWriteImage(
    Blit2DTest *ctx,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    NvMediaBool appendFlag)
{
    uint32_t imageSize = 0;
    unsigned int size[3] ={0};
    uint8_t *buffer = NULL;
    uint32_t i, k, newk = 0;
    unsigned int *bytePerPixelPtr = NULL;
    ctx->numSurfaces = 1;
    NvMediaImageSurfaceMap surfaceMap;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    uint32_t lineWidth, numRows, startOffset;

    if(!image) {
        printf("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaImageLock() failed\n", __func__);
        return status;
    }
    NvMediaImageUnlock(image);

    ctx->dstBuff = (uint8_t**) malloc(sizeof(uint8_t*)*MAXM_NUM_SURFACES);
    if(!ctx->dstBuff) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    ctx->dstBuffPitches = (uint32_t*) calloc(1,sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!ctx->dstBuffPitches) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    ctx->heightSurface = surfaceMap.height;
    ctx->widthSurface  = surfaceMap.width;

    status = GetSurfParams(image->type,
                           &ctx->xScalePtr,
                           &ctx->yScalePtr,
                           &bytePerPixelPtr,
                           &ctx->numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: GetSurfParams() failed\n", __func__);
        goto done;
    }

    imageSize = 0;
    for(i = 0; i < ctx->numSurfaces; i++) {
        size[i] = (ctx->widthSurface * ctx->xScalePtr[i] * ctx->heightSurface * ctx->yScalePtr[i] * bytePerPixelPtr[i]);
        imageSize += size[i];
        ctx->dstBuffPitches[i] = (uint32_t)((float)ctx->widthSurface * ctx->xScalePtr[i]) * bytePerPixelPtr[i];
    }

    // Embedded data size needs to be included for RAW surftype
    size[0] += image->embeddedDataTopSize;
    size[0] += image->embeddedDataBottomSize;
    imageSize += image->embeddedDataTopSize;
    imageSize += image->embeddedDataBottomSize;

    buffer = (uint8_t *) calloc(1, imageSize);
    if(!buffer) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    ctx->dstBuffer = buffer;
    memset(buffer, 0xFF, imageSize);
    for(i = 0; i < ctx->numSurfaces; i++) {
        ctx->dstBuff[i] = buffer;
        buffer = buffer + (uint32_t)(ctx->heightSurface * ctx->yScalePtr[i] * ctx->dstBuffPitches[i]);
    }

done:
    return status;
}

NvMediaStatus
WriteImageToAllocatedBuffer(
    Blit2DTest *ctx,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    NvMediaBool appendFlag,
    uint32_t bytesPerPixel)
{
    NvMediaImageSurfaceMap surfaceMap;

    NvMediaStatus status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaImageLock() failed\n", __func__);
        goto done;
    }
    status = NvMediaImageGetBits(image, NULL, (void **)ctx->dstBuff, ctx->dstBuffPitches);
    NvMediaImageUnlock(image);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaVideoSurfaceGetBits() failed \n", __func__);
        goto done;
    }

done:

    return status;
}


static NvMediaStatus
ReadImageNew(
    char *fileName,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    uint32_t bytesPerPixel,
    uint32_t pixelAlignment)
{
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t imageSize = 0,surfaceSize = 0;
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    uint32_t i, j, k, newk = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    unsigned int uHeightSurface, uWidthSurface;
    NvMediaImageSurfaceMap surfaceMap;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    FILE *file = NULL;
    unsigned int count, index;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);
    unsigned int surfType, surfBPC;

    if(!image || !fileName) {
        printf("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaImageLock() failed\n", __func__);
        return status;
    }
    NvMediaImageUnlock(image);

    uHeightSurface = surfaceMap.height;
    uWidthSurface  = surfaceMap.width;

    if(width > uWidthSurface || height > uHeightSurface) {
        printf("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    pBuff = (uint8_t **) malloc(sizeof(uint8_t*)*MAXM_NUM_SURFACES);
    if(!pBuff) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = (uint32_t *)calloc(1,sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffPitches) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    status = GetSurfParams(image->type,
                           &xScalePtr,
                           &yScalePtr,
                           &bytePerPixelPtr,
                           &numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: GetSurfParams() failed\n", __func__);
        goto done;
    }

    status = NvMediaSurfaceFormatGetAttrs(image->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status != NVMEDIA_STATUS_OK) {
        printf("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        goto done;
    }
    surfType = srcAttr[NVM_SURF_ATTR_SURF_TYPE].value;
    surfBPC = srcAttr[NVM_SURF_ATTR_BITS_PER_COMPONENT].value;

    surfaceSize = 0;
    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        surfaceSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        imageSize += (width * xScalePtr[i] * height * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    buffer = (uint8_t *)calloc(1, surfaceSize);
    if(!buffer) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer = buffer;
    memset(buffer,0x10,surfaceSize);
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        if (i) {
            memset(pBuff[i], 0x80, (uHeightSurface * yScalePtr[i] * pBuffPitches[i]));
        }
        buffer = buffer + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    file = fopen(fileName, "rb");
    if(!file) {
        printf("%s: Error opening file: %s\n", __func__, fileName);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    if(frameNum > 0) {
        if(fseeko(file, frameNum * (off_t)imageSize, SEEK_SET)) {
            printf("ReadImage: Error seeking file: %s\n", fileName);
            status = NVMEDIA_STATUS_ERROR;
            goto done;
        }
    }

    if((surfType == NVM_SURF_ATTR_SURF_TYPE_RGBA ) && strstr(fileName, ".png")) {
        printf("ReadImage: Does not support png format\n");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }
    for(k = 0; k < numSurfaces; k++) {
        for(j = 0; j < height*yScalePtr[k]; j++) {
            newk = (!uvOrderFlag && k ) ? (numSurfaces - k) : k;
            index = j * pBuffPitches[newk];
            count = width * xScalePtr[newk] * bytePerPixelPtr[newk];
            if (fread(pBuff[newk] + index, count, 1, file) != 1) {
                status = NVMEDIA_STATUS_ERROR;
                printf("ReadImage: Error reading file: %s\n", fileName);
                goto done;
            }
            if((surfType == NVM_SURF_ATTR_SURF_TYPE_YUV) && (pixelAlignment == LSB_ALIGNED)) {
                uint16_t *psrc = (uint16_t*)(pBuff[newk] + index);
                switch(surfBPC) {
                    case NVM_SURF_ATTR_BITS_PER_COMPONENT_10:
                        for(i = 0; i < count/2; i++) {
                            *(psrc + i) = (*(psrc + i)) << (16 - 10);
                        }
                        break;
                    case NVM_SURF_ATTR_BITS_PER_COMPONENT_12:
                        for(i = 0; i < count/2; i++) {
                            *(psrc + i) = (*(psrc + i)) << (16 - 12);
                        }
                        break;
                    case NVM_SURF_ATTR_BITS_PER_COMPONENT_14:
                        for(i = 0; i < count/2; i++) {
                            *(psrc + i) = (*(psrc + i)) << (16 - 14);
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaImageLock() failed\n", __func__);
        goto done;
    }
    status = NvMediaImagePutBits(image, NULL, (void **)pBuff, pBuffPitches);
    NvMediaImageUnlock(image);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: Failed to put bits\n", __func__);
    }

done:
    if(pBuff) {
        free(pBuff);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    if(file) {
        fclose(file);
    }

    return status;
}

NvMediaStatus
ReadImage(
    char *fileName,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    uint32_t bytesPerPixel,
    uint32_t pixelAlignment)
{
    NvMediaStatus status;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    status = NvMediaSurfaceFormatGetAttrs(image->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status == NVMEDIA_STATUS_OK) {
        return ReadImageNew(
                        fileName,
                        frameNum,
                        width,
                        height,
                        image,
                        uvOrderFlag,
                        bytesPerPixel,
                        pixelAlignment);
    } else {
        printf("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return status;
    }
}

NvMediaStatus
InitImage(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height)
{
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t imageSize = 0,surfaceSize = 0;
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    uint32_t i;
    unsigned int uHeightSurface, uWidthSurface;
    NvMediaImageSurfaceMap surfaceMap;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    if(!image) {
        printf("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaImageLock failed\n", __func__);
        return status;
    }
    NvMediaImageUnlock(image);


    uHeightSurface = surfaceMap.height;
    uWidthSurface  = surfaceMap.width;

    if(width > uWidthSurface || height > uHeightSurface) {
        printf("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    pBuff = (uint8_t **) calloc(1,sizeof(uint8_t*)*MAXM_NUM_SURFACES);
    if(!pBuff) {
        printf("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = (uint32_t *) calloc(1,sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffPitches) {
        printf("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    status = GetSurfParams(image->type,
                           &xScalePtr,
                           &yScalePtr,
                           &bytePerPixelPtr,
                           &numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: GetSurfParams failed\n", __func__);
        goto done;
    }

    status = NvMediaSurfaceFormatGetAttrs(image->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        goto done;
    }

    surfaceSize = 0;
    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        surfaceSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        imageSize += (width * xScalePtr[i] * height * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    buffer = (uint8_t *)calloc(1, surfaceSize);
    if(!buffer) {
        printf("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer = buffer;
    memset(buffer,0x00,surfaceSize);
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        buffer = buffer + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaImageLock failed\n", __func__);
        goto done;
    }
    status = NvMediaImagePutBits(image, NULL, (void **)pBuff, pBuffPitches);
    NvMediaImageUnlock(image);
    if(status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMediaImagePutBits failed\n", __func__);
    }

done:
    if(pBuff) {
        free(pBuff);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    return status;
}

