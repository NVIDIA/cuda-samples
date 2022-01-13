/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#if !defined(__COMMON_GPU_HEADER_H)
#define __COMMON_GPU_HEADER_H

////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side constants and data structures
////////////////////////////////////////////////////////////////////////////////

#define  TIME_STEPS 16

#define CACHE_DELTA (2 * TIME_STEPS)

#define  CACHE_SIZE (256)

#define  CACHE_STEP (CACHE_SIZE - CACHE_DELTA)

#if NUM_STEPS % CACHE_DELTA
#error Bad constants
#endif

#endif