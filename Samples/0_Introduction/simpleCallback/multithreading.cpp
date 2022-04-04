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

#include "multithreading.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data) {
  return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
}

// Wait for thread to finish
void cutEndThread(CUTThread thread) {
  WaitForSingleObject(thread, INFINITE);
  CloseHandle(thread);
}

// Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num) {
  WaitForMultipleObjects(num, threads, true, INFINITE);

  for (int i = 0; i < num; i++) {
    CloseHandle(threads[i]);
  }
}

// Create barrier.
CUTBarrier cutCreateBarrier(int releaseCount) {
  CUTBarrier barrier;

  InitializeCriticalSection(&barrier.criticalSection);
  barrier.barrierEvent = CreateEvent(NULL, TRUE, FALSE, TEXT("BarrierEvent"));
  barrier.count = 0;
  barrier.releaseCount = releaseCount;

  return barrier;
}

// Increment barrier. (execution continues)
void cutIncrementBarrier(CUTBarrier *barrier) {
  int myBarrierCount;
  EnterCriticalSection(&barrier->criticalSection);
  myBarrierCount = ++barrier->count;
  LeaveCriticalSection(&barrier->criticalSection);

  if (myBarrierCount >= barrier->releaseCount) {
    SetEvent(barrier->barrierEvent);
  }
}

// Wait for barrier release.
void cutWaitForBarrier(CUTBarrier *barrier) {
  WaitForSingleObject(barrier->barrierEvent, INFINITE);
}

// Destroy barrier
void cutDestroyBarrier(CUTBarrier *barrier) {}

#else
// Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data) {
  pthread_t thread;
  pthread_create(&thread, NULL, func, data);
  return thread;
}

// Wait for thread to finish
void cutEndThread(CUTThread thread) { pthread_join(thread, NULL); }

// Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num) {
  for (int i = 0; i < num; i++) {
    cutEndThread(threads[i]);
  }
}

// Create barrier.
CUTBarrier cutCreateBarrier(int releaseCount) {
  CUTBarrier barrier;

  barrier.count = 0;
  barrier.releaseCount = releaseCount;

  pthread_mutex_init(&barrier.mutex, 0);
  pthread_cond_init(&barrier.conditionVariable, 0);

  return barrier;
}

// Increment barrier. (execution continues)
void cutIncrementBarrier(CUTBarrier *barrier) {
  int myBarrierCount;
  pthread_mutex_lock(&barrier->mutex);
  myBarrierCount = ++barrier->count;
  pthread_mutex_unlock(&barrier->mutex);

  if (myBarrierCount >= barrier->releaseCount) {
    pthread_cond_signal(&barrier->conditionVariable);
  }
}

// Wait for barrier release.
void cutWaitForBarrier(CUTBarrier *barrier) {
  pthread_mutex_lock(&barrier->mutex);

  while (barrier->count < barrier->releaseCount) {
    pthread_cond_wait(&barrier->conditionVariable, &barrier->mutex);
  }

  pthread_mutex_unlock(&barrier->mutex);
}

// Destroy barrier
void cutDestroyBarrier(CUTBarrier *barrier) {
  pthread_mutex_destroy(&barrier->mutex);
  pthread_cond_destroy(&barrier->conditionVariable);
}

#endif
