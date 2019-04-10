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

#include "helper_multiprocess.h"
#include <cstdlib>
#include <string>

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    info->size = sz;
    info->shmHandle = CreateFileMapping(INVALID_HANDLE_VALUE,
                                        NULL,
                                        PAGE_READWRITE,
                                        0,
                                        (DWORD)sz,
                                        name);
    if (info->shmHandle == 0) {
        return GetLastError();
    }

    info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
    if (info->addr == NULL) {
        return GetLastError();
    }

    return 0;
#else
    int status = 0;

    info->size = sz;

    info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
    if (info->shmFd < 0) {
        return errno;
    }

    status = ftruncate(info->shmFd, sz);
    if (status != 0) {
        return status;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == NULL) {
        return errno;
    }

    return 0;
#endif
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    info->size = sz;

    info->shmHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (info->shmHandle == 0) {
        return GetLastError();
    }

    info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
    if (info->addr == NULL) {
        return GetLastError();
    }

    return 0;
#else
    info->size = sz;

    info->shmFd = shm_open(name, O_RDWR, 0777);
    if (info->shmFd < 0) {
        return errno;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == NULL) {
        return errno;
    }

    return 0;
#endif
}

void sharedMemoryClose(sharedMemoryInfo *info)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    if (info->addr) {
        UnmapViewOfFile(info->addr);
    }
    if (info->shmHandle) {
        CloseHandle(info->shmHandle);
    }
#else
    if (info->addr) {
        munmap(info->addr, info->size);
    }
    if (info->shmFd) {
        close(info->shmFd);
    }
#endif
}

int spawnProcess(Process *process, const char *app, char * const *args)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    STARTUPINFO si = {0};
    BOOL status;
    size_t arglen = 0;
    size_t argIdx = 0;
    std::string arg_string;
	memset(process, 0, sizeof(*process));

    while (*args) {
		arg_string.append(*args).append(1, ' ');
		args++;
	}

    status = CreateProcess(app, LPSTR(arg_string.c_str()), NULL, NULL, FALSE, 0, NULL, NULL, &si, process);

    return status ? 0 : GetLastError();
#else
    *process = fork();
    if (*process == 0) {
        if (0 > execvp(app, args)) {
            return errno;
        }
    }
    else if (*process < 0) {
        return errno;
    }
    return 0;
#endif
}

int waitProcess(Process *process)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	DWORD exitCode;
    WaitForSingleObject(process->hProcess, INFINITE);
    GetExitCodeProcess(process->hProcess, &exitCode);
    CloseHandle(process->hProcess);
    CloseHandle(process->hThread);
	return (int)exitCode;
#else
    int status = 0;
    do {
        if (0 > waitpid(*process, &status, 0)) {
            return errno;
        }
    } while (!WIFEXITED(status));
    return WEXITSTATUS(status);
#endif
}
