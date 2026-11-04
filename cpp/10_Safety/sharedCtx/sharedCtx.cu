/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>
#include <cuda_runtime_api_safe_ex.h>
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static const unsigned int kDevice    = 0;
static const int          kTimeoutMs = 30000;
static const int          kElements  = 256;
// Inherited by the process this sample spawns. A normal launch does not have it open.
static const int kIpcFd = 100;

__global__ void incrementKernel(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += 1;
    }
}

static bool checkRt(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        printf("sharedCtx: %s failed: %s\n", what, cudaGetErrorString(err));
        return false;
    }
    return true;
}

static bool isIpcSocket(int fd)
{
    int       soType = 0;
    socklen_t len    = sizeof(soType);
    return getsockopt(fd, SOL_SOCKET, SO_TYPE, &soType, &len) == 0 && soType == SOCK_STREAM;
}

static bool setRecvTimeout(int fd)
{
    struct timeval tv;
    tv.tv_sec  = kTimeoutMs / 1000;
    tv.tv_usec = (kTimeoutMs % 1000) * 1000;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
        printf("sharedCtx: setsockopt SO_RCVTIMEO failed: %s\n", strerror(errno));
        return false;
    }
    return true;
}

static bool writeAll(int fd, const void *data, size_t bytes)
{
    const char *pos  = static_cast<const char *>(data);
    size_t      left = bytes;
    while (left > 0) {
        ssize_t n = send(fd, pos, left, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            printf("sharedCtx: send failed: %s\n", strerror(errno));
            return false;
        }
        if (n == 0) {
            printf("sharedCtx: socket closed during send\n");
            return false;
        }
        pos += n;
        left -= static_cast<size_t>(n);
    }
    return true;
}

static bool readAll(int fd, void *data, size_t bytes)
{
    char  *pos  = static_cast<char *>(data);
    size_t left = bytes;
    while (left > 0) {
        ssize_t n = recv(fd, pos, left, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            printf("sharedCtx: recv failed: %s\n", strerror(errno));
            return false;
        }
        if (n == 0) {
            printf("sharedCtx: socket closed during recv\n");
            return false;
        }
        pos += n;
        left -= static_cast<size_t>(n);
    }
    return true;
}

// Launch a kernel on the current primary context and check the result.
static bool runIncrement(const char *role)
{
    int *host = static_cast<int *>(malloc(sizeof(int) * kElements));
    if (host == nullptr) {
        printf("sharedCtx: %s host allocation failed\n", role);
        return false;
    }
    for (int i = 0; i < kElements; ++i) {
        host[i] = i;
    }

    int         *device = nullptr;
    cudaStream_t stream = nullptr;
    if (!checkRt(cudaMalloc(&device, sizeof(int) * kElements), "cudaMalloc")
        || !checkRt(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")
        || !checkRt(cudaMemcpyAsync(device, host, sizeof(int) * kElements, cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync HtoD")) {
        if (device != nullptr) {
            checkRt(cudaFree(device), "cudaFree");
        }
        if (stream != nullptr) {
            checkRt(cudaStreamDestroy(stream), "cudaStreamDestroy");
        }
        free(host);
        return false;
    }

    incrementKernel<<<1, kElements, 0, stream>>>(device, kElements);
    if (!checkRt(cudaGetLastError(), "incrementKernel launch")
        || !checkRt(cudaStreamSynchronize(stream), "cudaStreamSynchronize")
        || !checkRt(cudaMemcpyAsync(host, device, sizeof(int) * kElements, cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync DtoH")
        || !checkRt(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
        checkRt(cudaStreamDestroy(stream), "cudaStreamDestroy");
        checkRt(cudaFree(device), "cudaFree");
        free(host);
        return false;
    }

    bool released = checkRt(cudaStreamDestroy(stream), "cudaStreamDestroy");
    released      = checkRt(cudaFree(device), "cudaFree") && released;

    bool ok = true;
    for (int i = 0; i < kElements; ++i) {
        if (host[i] != i + 1) {
            printf("sharedCtx: %s mismatch at %d (got %d)\n", role, i, host[i]);
            ok = false;
            break;
        }
    }

    free(host);
    if (ok && released) {
        printf("sharedCtx: %s kernel passed\n", role);
    }
    return ok && released;
}

static int queryProcess(int fd)
{
    int                                rc = EXIT_FAILURE;
    cudaSafeExSharedDeviceCtxDeviceKey deviceKey;
    cudaSafeExSharedDeviceCtxShareKey  shareKey;

    printf("sharedCtx: querying device key\n");
    if (!checkRt(cudaSafeExQueryDeviceKey(&deviceKey, kDevice), "cudaSafeExQueryDeviceKey")
        || !writeAll(fd, &deviceKey, sizeof(deviceKey)) || !readAll(fd, &shareKey, sizeof(shareKey))) {
        goto done;
    }

    printf("sharedCtx: importing share key\n");
    if (!checkRt(cudaSafeExSharedDeviceCtxKeyImport(&shareKey, kDevice), "cudaSafeExSharedDeviceCtxKeyImport")
        || !checkRt(cudaFree(0), "cudaFree(0) after import") || !runIncrement("import process")) {
        goto done;
    }
    rc = EXIT_SUCCESS;

done:
    close(fd);
    return rc;
}

static int exportProcess(pid_t peer, int fd)
{
    int  rc         = EXIT_FAILURE;
    int  status     = 0;
    bool peerReaped = false;

    cudaSafeExSharedDeviceCtxDeviceKey deviceKey;
    cudaSafeExSharedDeviceCtxShareKey  shareKey;

    if (!readAll(fd, &deviceKey, sizeof(deviceKey))) {
        goto done;
    }
    printf("sharedCtx: received device key\n");

    if (!checkRt(cudaFree(0), "cudaFree(0) before export")
        || !checkRt(cudaSafeExSharedDeviceCtxKeyExport(&shareKey, deviceKey, kDevice),
                    "cudaSafeExSharedDeviceCtxKeyExport")) {
        goto done;
    }
    printf("sharedCtx: exported share key\n");

    if (!runIncrement("export process") || !writeAll(fd, &shareKey, sizeof(shareKey))) {
        goto done;
    }

    if (waitpid(peer, &status, 0) < 0) {
        printf("sharedCtx: waitpid failed: %s\n", strerror(errno));
        peerReaped = true;
        goto done;
    }
    peerReaped = true;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
        int code = WIFEXITED(status) ? WEXITSTATUS(status) : status;
        printf("sharedCtx: peer failed with status %d\n", code);
        goto done;
    }

    printf("sharedCtx: Result = PASS\n");
    rc = EXIT_SUCCESS;

done:
    close(fd);
    if (!peerReaped) {
        kill(peer, SIGKILL);
        waitpid(peer, nullptr, 0);
    }
    return rc;
}

int main(int argc, char **argv)
{
    (void)argc;

    // The spawned process inherits this socket. The program takes no arguments.
    if (isIpcSocket(kIpcFd)) {
        return queryProcess(kIpcFd);
    }

    int sv[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        printf("sharedCtx: socketpair failed: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    if (!setRecvTimeout(sv[0]) || !setRecvTimeout(sv[1])) {
        close(sv[0]);
        close(sv[1]);
        return EXIT_FAILURE;
    }

    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        printf("sharedCtx: posix_spawn_file_actions_init failed: %s\n", strerror(errno));
        close(sv[0]);
        close(sv[1]);
        return EXIT_FAILURE;
    }
    // CUDA does not survive fork() without exec(). Spawn a new process image and
    // hand it one end of the socket. The other end stays in this process.
    int spawnErr = posix_spawn_file_actions_adddup2(&actions, sv[1], kIpcFd);
    if (spawnErr == 0 && sv[0] != kIpcFd) {
        spawnErr = posix_spawn_file_actions_addclose(&actions, sv[0]);
    }
    if (spawnErr == 0 && sv[1] != kIpcFd) {
        spawnErr = posix_spawn_file_actions_addclose(&actions, sv[1]);
    }
    if (spawnErr != 0) {
        printf("sharedCtx: posix_spawn file action failed: %s\n", strerror(spawnErr));
        posix_spawn_file_actions_destroy(&actions);
        close(sv[0]);
        close(sv[1]);
        return EXIT_FAILURE;
    }

    char *childArgv[] = {argv[0], nullptr};
    pid_t peer        = 0;
    spawnErr          = posix_spawnp(&peer, argv[0], &actions, nullptr, childArgv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(sv[1]);
    if (spawnErr != 0) {
        printf("sharedCtx: posix_spawnp failed: %s\n", strerror(spawnErr));
        close(sv[0]);
        return EXIT_FAILURE;
    }
    return exportProcess(peer, sv[0]);
}
