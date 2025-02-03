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

#ifndef _NVMEDIA_TEST_MISC_UTILS_H_
#define _NVMEDIA_TEST_MISC_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvmedia_common.h"

#ifndef __INTEGRITY
#define MIN(a,b)             (((a) < (b)) ? (a) : (b))
#define MAX(a,b)             (((a) > (b)) ? (a) : (b))
#endif

typedef enum {
    LSB_ALIGNED,
    MSB_ALIGNED
} PixelAlignment;


//  u32
//
//    u32()  Reads 4 bytes from buffer and returns the read value
//
//  Arguments:
//
//   ptr
//      (in) Input buffer

uint32_t   u32(const uint8_t* ptr);

//  GetTimeMicroSec
//
//    GetTimeMicroSec()  Returns current time in microseconds
//
//  Arguments:
//
//   uTime
//      (out) Pointer to current time in microseconds

NvMediaStatus
GetTimeMicroSec(
    uint64_t *uTime);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_MISC_UTILS_H_ */
