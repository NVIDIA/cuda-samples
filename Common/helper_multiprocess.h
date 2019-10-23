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

#ifndef HELPER_MULTIPROCESS_H
#define HELPER_MULTIPROCESS_H

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <tchar.h>
#include <strsafe.h>
#include <sddl.h>
#include <aclapi.h>
#include <winternl.h>
#else
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#endif
#include <vector>

typedef struct sharedMemoryInfo_st {
    void *addr;
    size_t size;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    HANDLE shmHandle;
#else
    int shmFd;
#endif
} sharedMemoryInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);

void sharedMemoryClose(sharedMemoryInfo *info);


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef PROCESS_INFORMATION Process;
#else
typedef pid_t Process;
#endif

int spawnProcess(Process *process, const char *app, char * const *args);

int waitProcess(Process *process);

#define checkIpcErrors(ipcFuncResult) \
    if (ipcFuncResult == -1) { fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(EXIT_FAILURE); }

#if defined(__linux__)
struct ipcHandle_st {
    int socket;
    char *socketName;
};
typedef int ShareableHandle;
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
struct ipcHandle_st {
    std::vector<HANDLE> hMailslot; // 1 Handle in case of child and `num children` Handles for parent.
};
typedef HANDLE ShareableHandle;
#endif

typedef struct ipcHandle_st ipcHandle;

int
ipcCreateSocket(ipcHandle *&handle, const char *name, const std::vector<Process>& processes);

int
ipcOpenSocket(ipcHandle *&handle);

int
ipcCloseSocket(ipcHandle *handle);

int
ipcRecvShareableHandles(ipcHandle *handle, std::vector<ShareableHandle>& shareableHandles);

int
ipcSendShareableHandles(ipcHandle *handle, const std::vector<ShareableHandle>& shareableHandles, const std::vector<Process>& processes);

int
ipcCloseShareableHandle(ShareableHandle shHandle);

#endif // HELPER_MULTIPROCESS_H
