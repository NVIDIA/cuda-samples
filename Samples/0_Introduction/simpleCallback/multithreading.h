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

#ifndef MULTITHREADING_H
#define MULTITHREADING_H

// Simple portable thread library.

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Windows threads.
#include <windows.h>

typedef HANDLE CUTThread;
typedef unsigned(WINAPI *CUT_THREADROUTINE)(void *);

struct CUTBarrier {
  CRITICAL_SECTION criticalSection;
  HANDLE barrierEvent;
  int releaseCount;
  int count;
};

#define CUT_THREADPROC unsigned WINAPI
#define CUT_THREADEND return 0

#else
// POSIX threads.
#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void *
#define CUT_THREADEND return 0

struct CUTBarrier {
  pthread_mutex_t mutex;
  pthread_cond_t conditionVariable;
  int releaseCount;
  int count;
};

#endif

#ifdef __cplusplus
extern "C" {
#endif

// Create thread.
CUTThread cutStartThread(CUT_THREADROUTINE, void *data);

// Wait for thread to finish.
void cutEndThread(CUTThread thread);

// Wait for multiple threads.
void cutWaitForThreads(const CUTThread *threads, int num);

// Create barrier.
CUTBarrier cutCreateBarrier(int releaseCount);

// Increment barrier. (execution continues)
void cutIncrementBarrier(CUTBarrier *barrier);

// Wait for barrier release.
void cutWaitForBarrier(CUTBarrier *barrier);

// Destroy barrier
void cutDestroyBarrier(CUTBarrier *barrier);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MULTITHREADING_H
