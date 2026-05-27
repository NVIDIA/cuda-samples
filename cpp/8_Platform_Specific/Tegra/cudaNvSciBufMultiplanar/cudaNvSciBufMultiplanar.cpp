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
#include "cudaNvSciBufMultiplanar.h"

NvSciBufModule module;
NvSciBufObj    buffObj;
CUuuid         uuid;

void flipBits(uint8_t *pBuff, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++) {
        pBuff[i] = (~pBuff[i]);
    }
}

// Compare input and generated image files
void compareFiles(std::string &path1, std::string &path2)
{
    bool  result = true;
    FILE *fp1, *fp2;
    int   ch1, ch2;

    fp1 = fopen(path1.c_str(), "rb");
    fp2 = fopen(path2.c_str(), "rb");
    if (!fp1) {
        result = false;
        printf("File %s open failed in %s line %d\n", path1.c_str(), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    if (!fp2) {
        result = false;
        printf("File %s open failed in %s line %d\n", path2.c_str(), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    do {
        ch1 = getc(fp1);
        ch2 = getc(fp2);

        if (ch1 != ch2) {
            result = false;
            break;
        }
    } while (ch1 != EOF && ch2 != EOF);

    if (result) {
        printf("Input file : %s and output file : %s match SUCCESS\n", path1.c_str(), path2.c_str());
    }
    else {
        printf("Input file : %s and output file : %s match FAILURE\n", path1.c_str(), path2.c_str());
    }

    if (fp1) {
        fclose(fp1);
    }
    if (fp2) {
        fclose(fp2);
    }
}

void Caller::init()
{
    checkNvSciErrors(NvSciBufAttrListCreate(module, &attrList));
    attrListOut = NULL;
}

void Caller::deinit()
{
    NvSciBufAttrListFree(attrList);
    checkCudaErrors(cudaDestroyExternalMemory(extMem));
}

// Set NvSciBufImage attribute values in the attribute list
void Caller::setAttrListImageMultiPlanes(int imageWidth, int imageHeight)
{
    NvSciBufType                   bufType       = NvSciBufType_Image;
    NvSciBufAttrValImageLayoutType layout        = NvSciBufImage_BlockLinearType;
    bool                           cpuAccessFlag = false;
    NvSciBufAttrValAccessPerm      perm          = NvSciBufAccessPerm_ReadWrite;
    NvSciRmGpuId                   gpuid;
    bool                           vpr        = false;
    int32_t                        planeCount = PLANAR_NUM_PLANES;
    int                            drvVersion;
    // Dimensions of the imported image in the YUV 420 planar format
    int32_t                  planeWidths[]  = {imageWidth, imageWidth / 2, imageWidth / 2};
    int32_t                  planeHeights[] = {imageHeight, imageHeight / 2, imageHeight / 2};
    NvSciBufAttrKeyValuePair keyPair;
    NvSciBufAttrKeyValuePair pairArray[ATTR_SIZE];

    NvSciBufAttrValColorFmt      planeColorFmts[] = {NvSciColor_Y8, NvSciColor_V8, NvSciColor_U8};
    NvSciBufAttrValImageScanType planeScanType[]  = {NvSciBufScan_ProgressiveType};

    memcpy(&gpuid.bytes, &uuid.bytes, sizeof(uuid.bytes));

    NvSciBufAttrKeyValuePair imgBuffAttrsArr[] = {
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag, sizeof(cpuAccessFlag)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {NvSciBufGeneralAttrKey_GpuId, &gpuid, sizeof(gpuid)},
        {NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
        {NvSciBufImageAttrKey_VprFlag, &vpr, sizeof(vpr)},
        {NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
        {NvSciBufImageAttrKey_PlaneColorFormat, planeColorFmts, sizeof(planeColorFmts)},
        {NvSciBufImageAttrKey_PlaneWidth, planeWidths, sizeof(planeWidths)},
        {NvSciBufImageAttrKey_PlaneHeight, planeHeights, sizeof(planeHeights)},
        {NvSciBufImageAttrKey_PlaneScanType, planeScanType, sizeof(planeScanType)},
    };

    std::vector<NvSciBufAttrKeyValuePair> imgBuffAttrsVec(
        imgBuffAttrsArr, imgBuffAttrsArr + (sizeof(imgBuffAttrsArr) / sizeof(imgBuffAttrsArr[0])));

    memset(pairArray, 0, sizeof(NvSciBufAttrKeyValuePair) * imgBuffAttrsVec.size());
    std::copy(imgBuffAttrsVec.begin(), imgBuffAttrsVec.end(), pairArray);
    checkNvSciErrors(NvSciBufAttrListSetAttrs(attrList, pairArray, imgBuffAttrsVec.size()));
}

cudaNvSciBufMultiplanar::cudaNvSciBufMultiplanar(size_t width, size_t height, std::vector<int> &deviceIds)
    : imageWidth(width)
    , imageHeight(height)
{
    mCudaDeviceId      = deviceIds[0];
    attrListReconciled = NULL;
    attrListConflict   = NULL;
    checkNvSciErrors(NvSciBufModuleOpen(&module));
    initCuda(mCudaDeviceId);
}

void cudaNvSciBufMultiplanar::initCuda(int devId)
{
    int          major = 0, minor = 0, drvVersion;
    NvSciRmGpuId gpuid;

    checkCudaErrors(cudaSetDevice(mCudaDeviceId));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, mCudaDeviceId));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, mCudaDeviceId));
    printf("[cudaNvSciBufMultiplanar] GPU Device %d: \"%s\" with compute capability "
           "%d.%d\n\n",
           mCudaDeviceId,
           _ConvertSMVer2ArchName(major, minor),
           major,
           minor);

    checkCudaDrvErrors(cuDriverGetVersion(&drvVersion));

    if (drvVersion <= 11030) {
        checkCudaDrvErrors(cuDeviceGetUuid(&uuid, devId));
    }
    else {
        checkCudaDrvErrors(cuDeviceGetUuid_v2(&uuid, devId));
    }
}

/*
Caller1 flips a YUV image which is allocated to nvscibuf APIs and copied into CUDA Array.
It is mapped to CUDA surface and bit flip is done. Caller2 in the same thread copies
CUDA Array to a YUV image file. The original image is compared with the double bit
flipped image.
*/
void cudaNvSciBufMultiplanar::runCudaNvSciBufPlanar(std::string &imageFilename, std::string &imageFilenameOut)
{
    cudaArray_t levelArray1[PLANAR_NUM_PLANES];
    cudaArray_t levelArray2[PLANAR_NUM_PLANES];
    Caller      caller1;
    Caller      caller2;

    int numPlanes = PLANAR_NUM_PLANES;
    caller1.init();
    caller2.init();

    // Set NvSciBufImage attribute values in the attribute list
    caller1.setAttrListImageMultiPlanes(imageWidth, imageHeight);
    caller2.setAttrListImageMultiPlanes(imageWidth, imageHeight);

    // Reconcile attribute lists and allocate NvSciBuf object
    reconcileAttrList(&caller1.attrList, &caller2.attrList);
    caller1.copyExtMemToMultiPlanarArrays();
    for (int i = 0; i < numPlanes; i++) {
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelArray1[i], caller1.multiPlanarArray[i], 0));
    }
    caller1.copyYUVToCudaArrayAndFlipBits(imageFilename, levelArray1);

    caller2.copyExtMemToMultiPlanarArrays();
    for (int i = 0; i < numPlanes; i++) {
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelArray2[i], caller2.multiPlanarArray[i], 0));
    }
    // Maps cudaArray to surface memory and launches a kernel to flip bits
    launchFlipSurfaceBitsKernel(levelArray2, caller2.multiPlanarWidth, caller2.multiPlanarHeight, numPlanes);

    // Synchronization can be done using nvSciSync when non CUDA callers and cross-process signaler-waiter
    // applications are involved. Please refer to the cudaNvSci sample library for more details.
    checkCudaDrvErrors(cuCtxSynchronize());
    printf("Bit flip of the surface memory done\n");

    caller2.copyCudaArrayToYUV(imageFilenameOut, levelArray2);
    compareFiles(imageFilename, imageFilenameOut);

    // Release memory
    printf("Releasing memory\n");
    for (int i = 0; i < numPlanes; i++) {
        checkCudaErrors(cudaFreeMipmappedArray(caller1.multiPlanarArray[i]));
        checkCudaErrors(cudaFreeMipmappedArray(caller2.multiPlanarArray[i]));
    }
    tearDown(&caller1, &caller2);
}

// Map NvSciBufObj to cudaMipmappedArray
void Caller::copyExtMemToMultiPlanarArrays()
{
    checkNvSciErrors(NvSciBufObjGetAttrList(buffObj, &attrListOut));
    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * PLANE_ATTR_SIZE);
    cudaExternalMemoryHandleDesc         memHandleDesc;
    cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {0};
    cudaChannelFormatDesc                desc       = {0};
    cudaExtent                           extent     = {0};

    pairArrayOut[PLANE_SIZE].key           = NvSciBufImageAttrKey_Size;              // Datatype: @c uint64_t
    pairArrayOut[PLANE_ALIGNED_SIZE].key   = NvSciBufImageAttrKey_PlaneAlignedSize;  // Datatype: @c uint64_t[]
    pairArrayOut[PLANE_OFFSET].key         = NvSciBufImageAttrKey_PlaneOffset;       // Datatype: @c uint64_t[]
    pairArrayOut[PLANE_HEIGHT].key         = NvSciBufImageAttrKey_PlaneHeight;       // Datatype: @c uint32_t[]
    pairArrayOut[PLANE_WIDTH].key          = NvSciBufImageAttrKey_PlaneWidth;        // Datatype: @c int32_t[]
    pairArrayOut[PLANE_CHANNEL_COUNT].key  = NvSciBufImageAttrKey_PlaneChannelCount; // Datatype: @c uint8_t
    pairArrayOut[PLANE_BITS_PER_PIXEL].key = NvSciBufImageAttrKey_PlaneBitsPerPixel; // Datatype: @c uint32_t[]
    pairArrayOut[PLANE_COUNT].key          = NvSciBufImageAttrKey_PlaneCount;        // Datatype: @c uint32_t
    checkNvSciErrors(NvSciBufAttrListGetAttrs(attrListOut, pairArrayOut, (PLANE_ATTR_SIZE)));

    uint64_t  size              = *(uint64_t *)pairArrayOut[PLANE_SIZE].value;
    uint64_t *planeAlignedSize  = (uint64_t *)pairArrayOut[PLANE_ALIGNED_SIZE].value;
    int32_t  *planeWidth        = (int32_t *)pairArrayOut[PLANE_WIDTH].value;
    int32_t  *planeHeight       = (int32_t *)pairArrayOut[PLANE_HEIGHT].value;
    uint64_t *planeOffset       = (uint64_t *)pairArrayOut[PLANE_OFFSET].value;
    uint8_t   planeChannelCount = *(uint8_t *)pairArrayOut[PLANE_CHANNEL_COUNT].value;
    uint32_t *planeBitsPerPixel = (uint32_t *)pairArrayOut[PLANE_BITS_PER_PIXEL].value;
    uint32_t  planeCount        = *(uint32_t *)pairArrayOut[PLANE_COUNT].value;

    numPlanes = planeCount;

    for (int i = 0; i < numPlanes; i++) {
        multiPlanarWidth[i]  = planeWidth[i];
        multiPlanarHeight[i] = planeHeight[i];
    }

    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type                  = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = buffObj;
    memHandleDesc.size                  = size;
    checkCudaErrors(cudaImportExternalMemory(&extMem, &memHandleDesc));

    desc = cudaCreateChannelDesc(planeBitsPerPixel[0], 0, 0, 0, cudaChannelFormatKindUnsigned);
    memset(&mipmapDesc, 0, sizeof(mipmapDesc));
    mipmapDesc.numLevels = 1;

    for (int i = 0; i < numPlanes; i++) {
        memset(&extent, 0, sizeof(extent));
        extent.width          = planeWidth[i];
        extent.height         = planeHeight[i];
        extent.depth          = 0;
        mipmapDesc.offset     = planeOffset[i];
        mipmapDesc.formatDesc = desc;
        mipmapDesc.extent     = extent;
        mipmapDesc.flags      = cudaArraySurfaceLoadStore;
        ;
        checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&multiPlanarArray[i], extMem, &mipmapDesc));
    }
}

void cudaNvSciBufMultiplanar::reconcileAttrList(NvSciBufAttrList *attrList1, NvSciBufAttrList *attrList2)
{
    attrList[0]       = *attrList1;
    attrList[1]       = *attrList2;
    bool isReconciled = false;

    checkNvSciErrors(NvSciBufAttrListReconcile(attrList, 2, &attrListReconciled, &attrListConflict));
    checkNvSciErrors(NvSciBufAttrListIsReconciled(attrListReconciled, &isReconciled));
    checkNvSciErrors(NvSciBufObjAlloc(attrListReconciled, &buffObj));
    printf("NvSciBufAttrList reconciled\n");
}

// YUV 420 image is flipped and copied to cuda Array which is mapped to nvsciBuf
void Caller::copyYUVToCudaArrayAndFlipBits(std::string &path, cudaArray_t *cudaArr)
{
    FILE    *fp = NULL;
    uint8_t *pYBuff, *pUBuff, *pVBuff, *pChroma;
    uint8_t *pBuff               = NULL;
    uint32_t uvOffset[numPlanes] = {0}, copyWidthInBytes[numPlanes] = {0}, copyHeight[numPlanes] = {0};
    uint32_t width  = multiPlanarWidth[0];
    uint32_t height = multiPlanarHeight[0];

    fp = fopen(path.c_str(), "rb");
    if (!fp) {
        printf("CudaProducer: Error opening file: %s in %s line %d\n", path.c_str(), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    pBuff = (uint8_t *)malloc((width * height * PLANAR_CHROMA_WIDTH_ORDER * PLANAR_CHROMA_HEIGHT_ORDER)
                              * sizeof(unsigned char));
    if (!pBuff) {
        printf("CudaProducer: Failed to allocate image buffer in %s line %d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    // Y V U order in the buffer. Fully planar formats use
    // three planes to store the Y, Cb and Cr components separately.
    pYBuff = pBuff;
    pVBuff = pYBuff + width * height;
    pUBuff = pVBuff + (width / PLANAR_CHROMA_WIDTH_ORDER) * (height / PLANAR_CHROMA_HEIGHT_ORDER);
    for (uint32_t i = 0; i < height; i++) {
        if (fread(pYBuff, width, 1, fp) != 1) {
            printf("ReadYUVFrame: Error reading file: %s in %s line %d\n", path.c_str(), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        flipBits(pYBuff, width);
        pYBuff += width;
    }

    pChroma = pVBuff;
    for (uint32_t i = 0; i < height / PLANAR_CHROMA_HEIGHT_ORDER; i++) {
        if (fread(pChroma, width / PLANAR_CHROMA_WIDTH_ORDER, 1, fp) != 1) {
            printf("ReadYUVFrame: Error reading file: %s in %s line %d\n", path.c_str(), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        flipBits(pChroma, width);
        pChroma += width / PLANAR_CHROMA_WIDTH_ORDER;
    }

    pChroma = pUBuff;
    for (uint32_t i = 0; i < height / PLANAR_CHROMA_HEIGHT_ORDER; i++) {
        if (fread(pChroma, width / PLANAR_CHROMA_WIDTH_ORDER, 1, fp) != 1) {
            printf("ReadYUVFrame: Error reading file: %s in %s line %d\n", path.c_str(), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        flipBits(pChroma, width);
        pChroma += width / PLANAR_CHROMA_WIDTH_ORDER;
    }
    uvOffset[0]         = 0;
    copyHeight[0]       = height;
    copyHeight[1]       = height / PLANAR_CHROMA_HEIGHT_ORDER;
    copyHeight[2]       = height / PLANAR_CHROMA_HEIGHT_ORDER;
    copyWidthInBytes[0] = width;
    // Width of the second and third planes is half of the first plane.
    copyWidthInBytes[1] = width / PLANAR_CHROMA_WIDTH_ORDER;
    copyWidthInBytes[2] = width / PLANAR_CHROMA_WIDTH_ORDER;
    uvOffset[1]         = width * height;
    uvOffset[2]         = uvOffset[1] + (width / PLANAR_CHROMA_WIDTH_ORDER) * (height / PLANAR_CHROMA_HEIGHT_ORDER);
    for (int i = 0; i < numPlanes; i++) {
        checkCudaDrvErrors(cuCtxSynchronize());
        checkCudaErrors(cudaMemcpy2DToArray(cudaArr[i],
                                            0,
                                            0,
                                            (void *)(pBuff + uvOffset[i]),
                                            copyWidthInBytes[i],
                                            copyWidthInBytes[i],
                                            copyHeight[i],
                                            cudaMemcpyHostToDevice));
    }

    if (fp) {
        fclose(fp);
        fp = NULL;
    }
    if (pBuff) {
        free(pBuff);
        pBuff = NULL;
    }
    printf("Image %s copied to CUDA Array and bit flip done\n", path.c_str());
}

// Copy Cuda Array in YUV 420 format to a file
void Caller::copyCudaArrayToYUV(std::string &path, cudaArray_t *cudaArr)
{
    FILE    *fp = NULL;
    int      bufferSize;
    uint32_t width            = multiPlanarWidth[0];
    uint32_t height           = multiPlanarHeight[0];
    uint32_t copyWidthInBytes = 0, copyHeight = 0;
    uint8_t *pCudaCopyMem = NULL;

    fp = fopen(path.c_str(), "wb+");
    if (!fp) {
        printf("WriteFrame: file open failed %s in %s line %d\n", path.c_str(), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numPlanes; i++) {
        if (i == 0) {
            bufferSize       = width * height;
            copyWidthInBytes = width;
            copyHeight       = height;

            pCudaCopyMem = (uint8_t *)malloc(bufferSize);
            if (pCudaCopyMem == NULL) {
                printf("pCudaCopyMem malloc failed in %s line %d\n", __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }
        }
        else {
            bufferSize       = ((height / PLANAR_CHROMA_HEIGHT_ORDER) * (width / PLANAR_CHROMA_WIDTH_ORDER));
            copyWidthInBytes = width / PLANAR_CHROMA_WIDTH_ORDER;
            copyHeight       = height / PLANAR_CHROMA_HEIGHT_ORDER;
        }
        memset(pCudaCopyMem, 0, bufferSize);

        checkCudaErrors(cudaMemcpy2DFromArray((void *)pCudaCopyMem,
                                              copyWidthInBytes,
                                              cudaArr[i],
                                              0,
                                              0,
                                              copyWidthInBytes,
                                              copyHeight,
                                              cudaMemcpyDeviceToHost));

        checkCudaDrvErrors(cuCtxSynchronize());

        if (fwrite(pCudaCopyMem, bufferSize, 1, fp) != 1) {
            printf("Cuda consumer: output file write failed in %s line %d\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    }
    printf("Output file : %s saved\n", path.c_str());

    if (fp) {
        fclose(fp);
        fp = NULL;
    }
}

void cudaNvSciBufMultiplanar::tearDown(Caller *caller1, Caller *caller2)
{
    caller1->deinit();
    caller2->deinit();
    NvSciBufObjFree(buffObj);
}
