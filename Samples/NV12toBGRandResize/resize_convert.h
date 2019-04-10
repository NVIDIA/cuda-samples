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


#ifndef __H_RESIZE_CONVERT__
#define __H_RESIZE_CONVERT__

#include <iostream>
#include <helper_cuda.h>

// nv12 resize
extern "C"
void resizeNV12Batch(
    uint8_t *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
    uint8_t *dpDst, int nDstPitch, int nDstWidth, int nDstHeight,
    int nBatchSize, cudaStream_t stream = 0);

// bgr resize
extern "C"
void resizeBGRplanarBatch(
    float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
    float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight,
    int nBatchSize, cudaStream_t stream = 0,
    int cropX = 0, int cropY = 0, int cropW = 0, int cropH = 0,
    bool whSameResizeRatio = false);

//NV12 to bgr planar
extern "C"
void nv12ToBGRplanarBatch(uint8_t *pNv12, int nNv12Pitch,
    float *pRgb, int nRgbPitch, int nWidth, int nHeight,
    int nBatchSize, cudaStream_t stream=0);
#endif
