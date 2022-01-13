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


#ifndef _NVMEDIA_2D_CMD_LINE_H_
#define _NVMEDIA_2D_CMD_LINE_H_

/* Include header containing NvMedia2D declarations */
#include "nvmedia_2d.h"

/* Maximum length of the path including file name */
#define FILE_NAME_SIZE 1024

/* TestArgs contains all arguments required to run the 2D test */
typedef struct _TestArgs {
    char inputFileName[FILE_NAME_SIZE];

    NvMediaSurfAllocAttr srcSurfAllocAttrs[NVM_SURF_ALLOC_ATTR_MAX];
    NvMediaSurfAllocAttr dstSurfAllocAttrs[NVM_SURF_ALLOC_ATTR_MAX];
    uint32_t numSurfAllocAttrs;

    NvMediaSurfFormatAttr srcSurfFormatAttrs[NVM_SURF_FMT_ATTR_MAX];
    NvMediaSurfFormatAttr dstSurfFormatAttrs[NVM_SURF_FMT_ATTR_MAX];

    NvMediaRect srcRect;
    NvMediaRect dstRect;
    NvMedia2DBlitParameters     blitParams;
    size_t iterations;
} TestArgs;

typedef struct {
    NvMediaDevice               *device;
    /* I2D for 2D blit processing */
    NvMedia2D                   *i2d;
    NvMediaImage                *srcImage;
    NvMediaImage                *dstImage;
    NvMediaRect                 *srcRect;
    NvMediaRect                 *dstRect;
    uint8_t                     **dstBuff;
    uint32_t                    *dstBuffPitches;
    uint8_t                     *dstBuffer;
    uint32_t                    numSurfaces;
    uint32_t                    bytesPerPixel;
    uint32_t                    heightSurface;
    uint32_t                    widthSurface;
    float                       *xScalePtr;
    float                       *yScalePtr;

} Blit2DTest;

/* Prints application usage options */
void PrintUsage (void);

/* Parses command line arguments.
* Also parses any configuration files supplied in the command line arguments.
* Arguments:
* argc
*    (in) Number of tokens in the command line
* argv
*    (in) Command line tokens
* args
*    (out) Pointer to test arguments structure
*/
int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif /* _NVMEDIA_2D_CMD_LINE_H_ */
