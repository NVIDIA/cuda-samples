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

/**
**************************************************************************
* \file Common.h
* \brief Common includes header.
*
* This file contains includes of all libraries used by the project.
*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>  // helper functions for CUDA timing and initialization
#include <helper_functions.h>  // helper functions for timing, string parsing

/**
*  The dimension of pixels block
*/
#define BLOCK_SIZE 8

/**
*  Square of dimension of pixels block
*/
#define BLOCK_SIZE2 64

/**
*  log_2{BLOCK_SIZE), used for quick multiplication or division by the
*  pixels block dimension via shifting
*/
#define BLOCK_SIZE_LOG2 3

/**
*  log_2{BLOCK_SIZE*BLOCK_SIZE), used for quick multiplication or division by
* the
*  square of pixels block via shifting
*/
#define BLOCK_SIZE2_LOG2 6

/**
*  This macro states that __mul24 operation is performed faster that traditional
*  multiplication for two integers on CUDA. Please undefine if it appears to be
*  wrong on your system
*/
#define __MUL24_FASTER_THAN_ASTERIX

/**
*  Wrapper to the fastest integer multiplication function on CUDA
*/
#ifdef __MUL24_FASTER_THAN_ASTERIX
#define FMUL(x, y) (__mul24(x, y))
#else
#define FMUL(x, y) ((x) * (y))
#endif

/**
*  This macro allows using aligned memory management
*/
//#define __ALLOW_ALIGNED_MEMORY_MANAGEMENT
