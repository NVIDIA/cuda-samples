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

/**
 * CUDA dma-buf interoperability.
 *
 * Three demos exercise CUDA's ability to export a host allocation as a Linux
 * dma-buf file descriptor and to import a dma-buf fd back as a CUDA-mappable
 * buffer: (1) same-process round-trip, (2) cross-process IPC via fork() and
 * SCM_RIGHTS with both sides on one GPU, and (3) cross-GPU sharing across
 * processes -- the same fork+SCM_RIGHTS path as demo 2, but the producer
 * child runs on one GPU and the consumer parent imports the fd into a
 * different GPU's context. Demo 3 self-skips when the system does not have
 * two GPUs that satisfy the dma-buf capability attributes.
 *
 * Producer allocations are always pinned host memory; using a single
 * allocator keeps the producer path identical across all three demos.
 *
 * See README.md for build instructions, expected output, and the full list
 * of APIs the sample exercises.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "dmabufInterop_helpers.h"

#define NUM_ELEMENTS  4096
#define BLOCK_SIZE    256
#define PATTERN_BASE  0x1000

/* Kernel: writes ptr[i] = base + i. */
__global__ void writePatternKernel(int *ptr, int base, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ptr[idx] = base + idx;
    }
}

/* Kernel: counts elements that do not match base + i. */
__global__ void verifyPatternKernel(int *ptr, int base, int count, int *errors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        if (ptr[idx] != base + idx) {
            atomicAdd(errors, 1);
        }
    }
}

/* Align v up to the next multiple of a. */
static size_t alignUp(size_t v, size_t a)
{
    size_t r = v % a;
    return (r == 0) ? v : v + (a - r);
}

/* Result of a producer-side export: the allocation base (for cuMemFreeHost),
 * the page-aligned pointer usable as both host and device pointer, the
 * dma-buf fd, and the aligned size. */
struct DmabufExport
{
    void  *allocBase;
    void  *alignedPtr;
    int    fd;
    size_t size;
};

/* Allocate page-aligned pinned host memory and export it as a dma-buf fd.
 * cuMemGetHandleForAddressRange requires a page-aligned base, so we allocate
 * an extra page of headroom and round up. */
static DmabufExport exportHostAllocAsDmabuf(size_t size)
{
    long pageSize = sysconf(_SC_PAGESIZE);

    DmabufExport e = {};
    e.size = alignUp(size, (size_t)pageSize);

    cuMemAllocHost(&e.allocBase, e.size + (size_t)pageSize);
    e.alignedPtr = (void *)alignUp((size_t)e.allocBase, (size_t)pageSize);

    cuMemGetHandleForAddressRange(&e.fd,
                                  (CUdeviceptr)e.alignedPtr,
                                  e.size,
                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                  0ULL);
    return e;
}

/* Import a dma-buf fd into the current CUDA context and obtain a CUdeviceptr
 * that kernels can read and write. */
static void importDmabuf(int fd, size_t size, CUexternalMemory *outExtMem, CUdeviceptr *outDevPtr)
{
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = {};
    memDesc.type      = CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD;
    memDesc.handle.fd = fd;
    memDesc.size      = size;
    cuImportExternalMemory(outExtMem, &memDesc);

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufDesc = {};
    bufDesc.size = size;
    cuExternalMemoryGetMappedBuffer(outDevPtr, *outExtMem, &bufDesc);
}

/* Same-process round-trip: one context exports a host allocation as a
 * dma-buf fd and imports it back. Producer writes the pattern; consumer
 * reads via the imported mapping and verifies. */
static int runSameProcess(int device)
{
    cudaSetDevice(device);

    DmabufExport e = exportHostAllocAsDmabuf(NUM_ELEMENTS * sizeof(int));
    printf("  producer: allocated %zu bytes, exported as dma-buf fd %d\n", e.size, e.fd);

    int blocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    writePatternKernel<<<blocks, BLOCK_SIZE>>>((int *)e.alignedPtr, PATTERN_BASE, NUM_ELEMENTS);
    cudaDeviceSynchronize();
    printf("  producer: wrote pattern\n");

    CUexternalMemory extMem = NULL;
    CUdeviceptr      importedPtr = 0;
    importDmabuf(e.fd, e.size, &extMem, &importedPtr);
    printf("  consumer: imported dma-buf as device pointer\n");

    int *errors = NULL;
    cudaMallocHost(&errors, sizeof(int));
    *errors = 0;
    verifyPatternKernel<<<blocks, BLOCK_SIZE>>>(
        (int *)importedPtr, PATTERN_BASE, NUM_ELEMENTS, errors);
    cudaDeviceSynchronize();

    int rc = (*errors == 0) ? 0 : -1;
    printf("  consumer: %s (%d mismatches)\n", rc == 0 ? "verified" : "Error", *errors);

    /* Free up resources. */
    cudaFreeHost(errors);
    cuDestroyExternalMemory(extMem);
    close(e.fd);
    cuMemFreeHost(e.allocBase);
    return rc;
}

/* Cross-process child: producer. Allocates, exports, writes the pattern,
 * sends the fd to the parent, waits for a one-byte ack so it does not free
 * the underlying memory before the parent finishes reading.
 *
 * The child waits for a one-byte "go" from the parent before doing any
 * CUDA work, so that the parent can serialise output across the same-process
 * demo and the two cross-process demos. If the parent closes the socket
 * without writing (because some upstream gate failed), the child exits
 * cleanly. */
static int runCrossProcessChild(int device, int sock)
{
    cuInit(0);
    cudaSetDevice(device);

    char go = 0;
    if (read(sock, &go, 1) <= 0) return 0;

    DmabufExport e = exportHostAllocAsDmabuf(NUM_ELEMENTS * sizeof(int));
    printf("  child:  allocated %zu bytes, exported as dma-buf fd %d\n", e.size, e.fd);

    int blocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    writePatternKernel<<<blocks, BLOCK_SIZE>>>((int *)e.alignedPtr, PATTERN_BASE, NUM_ELEMENTS);
    cudaDeviceSynchronize();
    printf("  child:  wrote pattern\n");

    sendFd(sock, e.fd);
    printf("  child:  sent fd to parent\n");

    char ack = 0;
    [[maybe_unused]] ssize_t r = read(sock, &ack, 1);

    /* Free up resources. */
    close(e.fd);
    cuMemFreeHost(e.allocBase);
    return 0;
}

/* Cross-process parent: consumer. Signals the child to start producing,
 * receives the fd, imports it, verifies the pattern, and acks the child so
 * it can clean up. */
static int runCrossProcessParent(int device, int sock)
{
    cudaSetDevice(device);

    char go = 'G';
    [[maybe_unused]] ssize_t r_go = write(sock, &go, 1);

    /* fd is the dma-buf that the producer child just wrote to; it refers to
     * the same underlying memory the child exported, just a different fd
     * number in this process. */
    int fd = recvFd(sock);
    printf("  parent: received dma-buf fd %d from child\n", fd);

    const size_t size = alignUp(NUM_ELEMENTS * sizeof(int), (size_t)sysconf(_SC_PAGESIZE));

    CUexternalMemory extMem = NULL;
    CUdeviceptr      importedPtr = 0;
    importDmabuf(fd, size, &extMem, &importedPtr);
    printf("  parent: imported dma-buf as device pointer\n");

    int *errors = NULL;
    cudaMallocHost(&errors, sizeof(int));
    *errors = 0;
    int blocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    verifyPatternKernel<<<blocks, BLOCK_SIZE>>>(
        (int *)importedPtr, PATTERN_BASE, NUM_ELEMENTS, errors);
    cudaDeviceSynchronize();

    int rc = (*errors == 0) ? 0 : -1;
    printf("  parent: %s (%d mismatches)\n", rc == 0 ? "verified" : "Error", *errors);

    char ack = 'A';
    [[maybe_unused]] ssize_t r_ack = write(sock, &ack, 1);

    /* Free up resources. */
    cudaFreeHost(errors);
    cuDestroyExternalMemory(extMem);
    close(fd);
    return rc;
}

/* Cross-process orchestration. The fork() must happen BEFORE either side
 * calls cuInit: the CUDA driver does not survive fork-after-init, so child
 * and parent each initialise CUDA independently after the fork. main()
 * spawns the child here and drives the parent side later. */
static pid_t spawnCrossProcessChild(int device, int *outParentSocket)
{
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
        perror("socketpair");
        return -1;
    }

    /* Flush stdio before fork() so the child does not inherit a buffer that
     * would then be flushed twice. */
    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(sockets[0]);
        close(sockets[1]);
        return -1;
    }
    if (pid == 0) {
        close(sockets[1]);
        int rc = runCrossProcessChild(device, sockets[0]);
        close(sockets[0]);
        fflush(stdout);
        fflush(stderr);
        _exit(rc == 0 ? 0 : 1);
    }

    close(sockets[0]);
    *outParentSocket = sockets[1];
    return pid;
}

/* Find any pair of GPUs where one supports producing a dma-buf fd from a
 * host allocation and the other supports importing a dma-buf fd. Iteration
 * is device-order so the result is deterministic -- child and parent each
 * call this independently after the fork and arrive at the same (producer,
 * consumer) pair. Both outputs are -1 if no qualifying pair is present. */
static void findCrossGpuPair(int *outProducer, int *outConsumer)
{
    *outProducer = -1;
    *outConsumer = -1;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) return;

    for (int p = 0; p < deviceCount; ++p) {
        int hostAllocOk = 0;
        cuDeviceGetAttribute(&hostAllocOk,
                             CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED, p);
        if (!hostAllocOk) continue;
        for (int c = 0; c < deviceCount; ++c) {
            if (c == p) continue;
            int importOk = 0;
            cuDeviceGetAttribute(&importOk,
                                 CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, c);
            if (importOk) { *outProducer = p; *outConsumer = c; return; }
        }
    }
}

/* Cross-GPU cross-process child: producer on the discovered producer GPU.
 * Mirrors runCrossProcessChild but runs on a specific device chosen by
 * findCrossGpuPair. The parent is the arbiter for skip decisions: child
 * waits for a "go" byte from the parent (or an EOF if the parent decided
 * to skip the demo) before allocating anything. */
static int runCrossProcCrossGpuChild(int sock)
{
    cuInit(0);

    char go = 0;
    if (read(sock, &go, 1) <= 0) return 0;

    int producer = -1, consumer = -1;
    findCrossGpuPair(&producer, &consumer);
    if (producer < 0) return 0;

    cudaSetDevice(producer);

    DmabufExport e = exportHostAllocAsDmabuf(NUM_ELEMENTS * sizeof(int));
    printf("  child:  on device %d, allocated %zu bytes, exported as dma-buf fd %d\n",
           producer, e.size, e.fd);

    int blocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    writePatternKernel<<<blocks, BLOCK_SIZE>>>((int *)e.alignedPtr, PATTERN_BASE, NUM_ELEMENTS);
    cudaDeviceSynchronize();
    printf("  child:  wrote pattern\n");

    sendFd(sock, e.fd);
    printf("  child:  sent fd to parent\n");

    char ack = 0;
    [[maybe_unused]] ssize_t r = read(sock, &ack, 1);

    /* Free up resources. */
    close(e.fd);
    cuMemFreeHost(e.allocBase);
    return 0;
}

/* Cross-GPU cross-process parent: consumer on the discovered consumer GPU.
 * Decides whether the demo proceeds: signals "go" to the child if a
 * qualifying pair exists, otherwise prints a skip message and returns
 * (closing the socket without writing, which the child sees as EOF). */
static int runCrossProcCrossGpuParent(int sock)
{
    int producer = -1, consumer = -1;
    findCrossGpuPair(&producer, &consumer);
    if (producer < 0) {
        printf("  no GPU pair satisfies the dma-buf capability attributes. Skipping.\n");
        return 0;
    }

    cudaSetDevice(consumer);

    char go = 'G';
    [[maybe_unused]] ssize_t r_go = write(sock, &go, 1);

    /* fd is the dma-buf that the producer child just wrote to on its own GPU;
     * import it into this parent's context on the consumer GPU below to
     * access the same underlying memory. */
    int fd = recvFd(sock);
    printf("  parent: on device %d, received dma-buf fd %d from child\n", consumer, fd);

    const size_t size = alignUp(NUM_ELEMENTS * sizeof(int), (size_t)sysconf(_SC_PAGESIZE));

    CUexternalMemory extMem = NULL;
    CUdeviceptr      importedPtr = 0;
    importDmabuf(fd, size, &extMem, &importedPtr);
    printf("  parent: imported dma-buf as device pointer\n");

    int *errors = NULL;
    cudaMallocHost(&errors, sizeof(int));
    *errors = 0;
    int blocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    verifyPatternKernel<<<blocks, BLOCK_SIZE>>>(
        (int *)importedPtr, PATTERN_BASE, NUM_ELEMENTS, errors);
    cudaDeviceSynchronize();

    int rc = (*errors == 0) ? 0 : -1;
    printf("  parent: %s (%d mismatches)\n", rc == 0 ? "verified" : "Error", *errors);

    char ack = 'A';
    [[maybe_unused]] ssize_t r_ack = write(sock, &ack, 1);

    /* Free up resources. */
    cudaFreeHost(errors);
    cuDestroyExternalMemory(extMem);
    close(fd);
    return rc;
}

/* Cross-GPU cross-process orchestration: spawn the producer child the same
 * way as spawnCrossProcessChild. Like the single-GPU IPC demo, this fork
 * must happen before any cuInit in the parent process. */
static pid_t spawnCrossProcCrossGpuChild(int *outParentSocket)
{
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
        perror("socketpair");
        return -1;
    }

    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(sockets[0]);
        close(sockets[1]);
        return -1;
    }
    if (pid == 0) {
        close(sockets[1]);
        int rc = runCrossProcCrossGpuChild(sockets[0]);
        close(sockets[0]);
        fflush(stdout);
        fflush(stderr);
        _exit(rc == 0 ? 0 : 1);
    }

    close(sockets[0]);
    *outParentSocket = sockets[1];
    return pid;
}

int main()
{
    /* Make stdout line-buffered so printf output from the parent and the
     * fork()ed children appears in temporal order even when stdout is
     * redirected to a pipe (e.g. via ssh) rather than a terminal. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const int device = 0;

    /* Spawn both children before any cuInit in this process. Each child runs
     * a producer-side role and blocks waiting for its parent to drive the
     * import. fork() must precede cuInit -- the CUDA driver does not survive
     * fork-after-init, so each child does its own cuInit independently. */
    int   sockIpc = -1, sockCrossGpu = -1;
    pid_t childIpcPid      = spawnCrossProcessChild(device, &sockIpc);
    if (childIpcPid < 0) return 1;
    pid_t childCrossGpuPid = spawnCrossProcCrossGpuChild(&sockCrossGpu);
    if (childCrossGpuPid < 0) {
        close(sockIpc);
        reapCrossProcessChild(childIpcPid);
        return 1;
    }

    cuInit(0);

    int dmabufExportSupported = 0;
    cuDeviceGetAttribute(&dmabufExportSupported,
                         CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED, device);
    if (!dmabufExportSupported) {
        printf("Device %d does not report CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED. "
               "Skipping sample.\n", device);
        close(sockIpc);
        close(sockCrossGpu);
        reapCrossProcessChild(childIpcPid);
        reapCrossProcessChild(childCrossGpuPid);
        return 0;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("Device %d: %s (Compute Capability %d.%d)\n\n",
           device, props.name, props.major, props.minor);

    printf("Same-process round-trip\n");
    int rc = runSameProcess(device);

    printf("\nCross-process IPC (fork + SCM_RIGHTS)\n");
    int ipcParentRc = (rc == 0) ? runCrossProcessParent(device, sockIpc) : -1;
    close(sockIpc);
    int ipcChildRc = reapCrossProcessChild(childIpcPid);
    if (rc == 0 && (ipcParentRc != 0 || ipcChildRc != 0)) rc = -1;

    printf("\nCross-GPU sharing across processes\n");
    int crossGpuParentRc = (rc == 0) ? runCrossProcCrossGpuParent(sockCrossGpu) : -1;
    close(sockCrossGpu);
    int crossGpuChildRc = reapCrossProcessChild(childCrossGpuPid);
    if (rc == 0 && (crossGpuParentRc != 0 || crossGpuChildRc != 0)) rc = -1;

    if (rc == 0) printf("\nDone\n");
    return rc == 0 ? 0 : 1;
}
