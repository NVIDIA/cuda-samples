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

/*
 * Unicast point-to-point data movement with Compute Fabric Transport (CFT)
 * logical endpoints.
 *
 * MPI example on a ring: rank r sends to (r+1)%N and receives from (r-1+N)%N.
 * Default mode uses plain fabric puts; a separate one-thread kernel then flags the
 * send neighbor over the same endpoint and waits on the receive neighbor's flag.
 * Pass --counted for counted fabric puts and a counter which counts bytes arrival
 * for receive completion.
 */

#include <cassert>
#include <climits>
#include <cstddef>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda_awbarrier_helpers.h>
#include <cuda_runtime.h>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define MPI_CHECK(cmd)                                                     \
    do {                                                                   \
        int e = cmd;                                                       \
        if (e != MPI_SUCCESS) {                                            \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

static void checkCuda(CUresult result, const char *func, const char *file, int line)
{
    if (result != CUDA_SUCCESS) {
        const char *name = nullptr;
        cuGetErrorName(result, &name);
        fprintf(stderr,
                "CUDA driver error at %s:%d code=%d(%s) \"%s\"\n",
                file,
                line,
                static_cast<int>(result),
                name ? name : "<unknown>",
                func);
        exit(EXIT_FAILURE);
    }
}

static void checkCuda(cudaError_t result, const char *func, const char *file, int line)
{
    if (result != cudaSuccess) {
        fprintf(stderr,
                "CUDA runtime error at %s:%d code=%d(%s) \"%s\"\n",
                file,
                line,
                static_cast<int>(result),
                cudaGetErrorName(result),
                func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

// Round value up to the next multiple of alignment (alignment need not be a power of two).
constexpr size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

void waitForEndpointReady(CUlogicalEndpointId leId)
{
    while (true) {
        int ready = 0;
        checkCudaErrors(cuLogicalEndpointQuery(leId, 1, &ready));
        if (ready > 0)
            break;
    }
}

static constexpr uint32_t FABRIC_CHUNK_SIZE = 8192;
// A b128 fabric put moves 16 bytes at a time, and mbarrier complete_tx::16B
// counts transferred bytes in 16-byte units. So an expected-tx count is always
// the byte count divided by this granularity.
static constexpr uint32_t FABRIC_TX_GRANULARITY_BYTES = 16;
// Sync region placed after the payload on a 256-byte boundary, reused per mode:
//   default : a 16-byte inter-GPU flag that the receive neighbor sets with a
//             fabric put (b128 puts move 16 bytes at a time).
//   --counted : an 8-byte delivered-byte counter.
static constexpr size_t INTER_GPU_FLAG_BYTES = 16;
static constexpr size_t SYNC_REGION_BYTES    = 16; // max of counter (8B) and flag (16B)
// The counter targeted by counted fabric puts must be 256-byte aligned, so the
// whole sync region is placed on a 256-byte boundary.
static constexpr size_t SYNC_REGION_ALIGN_BYTES = 256;
static constexpr int    WARP_SIZE               = 32;

#ifndef NDEBUG
// Debug builds: cap spin-waits at 30 s so a stuck fabric op prints instead of hanging.
static constexpr uint64_t CFT_WAIT_TIMEOUT_NS = 30ULL * 1000 * 1000 * 1000;

inline __device__ uint64_t readGlobalTimerNs()
{
    uint64_t t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#endif

// Spin while pred() is true. Returns true when pred() becomes false; in debug
// builds returns false after CFT_WAIT_TIMEOUT_NS and prints `what` to identify
// the stall. Release builds spin forever. Caller must handle false (e.g. __trap()).
template <typename Predicate>
inline __device__ bool waitWithDebugTimeout(Predicate &&pred, const char *what)
{
#ifndef NDEBUG
    uint64_t start = readGlobalTimerNs();
    while (pred()) {
        uint64_t now = readGlobalTimerNs();
        if (now - start > CFT_WAIT_TIMEOUT_NS) {
            printf("%s timeout after %llu ns (block=%u)\n", what, (unsigned long long)(now - start), blockIdx.x);
            return false;
        }
    }
#else
    (void)what;
    while (pred()) {
    }
#endif
    return true;
}

// Decode and print the fabric errors captured in the mbarrier's report value.
// Decode it with cudaFabricOpStatusSourceMbarrierV1: first query the error count, then
// retrieve each decoded error. Call this only when the report predicate is set.
inline __device__ void reportFabricError(uint8_t *reportValue, uint64_t offset)
{
    const unsigned long long off      = offset;
    unsigned int             errCount = 0;
    if (cudaFabricOpErrorStatusCount(reportValue, cudaFabricOpStatusSourceMbarrierV1, &errCount) != cudaSuccess) {
        printf("cudaFabricOpErrorStatusCount failed at offset %llu\n", off);
        return;
    }
    for (unsigned int i = 0; i < errCount; ++i) {
        cudaFabricOpStatusInfo info = cudaFabricOpStatusInfoSuccess;
        if (cudaFabricOpErrorStatusGet(reportValue, cudaFabricOpStatusSourceMbarrierV1, i, &info) == cudaSuccess) {
            printf("fabric put error[%u/%u] at offset %llu: status=%d\n", i, errCount, off, (int)info);
        }
        else {
            printf("cudaFabricOpErrorStatusGet failed for error[%u/%u] at offset %llu\n", i, errCount, off);
        }
    }
}

// Each block sends one FABRIC_CHUNK_SIZE slice to the peer's logical endpoint (putLeId).
// blockIdx.x selects the chunk: offset = blockIdx.x * FABRIC_CHUNK_SIZE. Launch
// gridDim.x = totalBytes / FABRIC_CHUNK_SIZE so every chunk is covered by one block.
//
// expectedBytes is how many bytes this rank expects to receive (tracked in our
// local counter by the peer's puts).
__global__ void cftTryPutCountedKernel(uint32_t  putLeId,
                                       uint64_t  counterOffset,
                                       int       myRank,
                                       uint64_t *counterPtr,
                                       uint64_t  expectedBytes)
{
    constexpr int elems = FABRIC_CHUNK_SIZE / sizeof(int);
    __shared__ alignas(16) int smemBuf[elems];
    __shared__ __mbarrier_t putBar;

    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(cuda::ptx::layout_v1, &putBar, 1);
    }

    // Fill shared memory with a rank-tagged pattern before streaming it to the peer.
    for (int i = threadIdx.x; i < elems; i += blockDim.x) {
        smemBuf[i] = myRank * elems + i;
    }
    __syncthreads();

    // Thread 0 issues one counted fabric put for this block's chunk.
    if (threadIdx.x == 0) {
        const uint64_t offset = blockIdx.x * FABRIC_CHUNK_SIZE;
        // Make the staged shared-memory source visible to the async proxy before the put reads it.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        cuda::ptx::fabric_try_put_counted(cuda::ptx::space_shared,
                                          cuda::ptx::sem_relaxed,
                                          cuda::ptx::scope_sys,
                                          putLeId,
                                          offset,
                                          counterOffset,
                                          smemBuf,
                                          FABRIC_CHUNK_SIZE,
                                          &putBar);
        cuda::ptx::fabric_submit();
        cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed,
                                             cuda::ptx::scope_cta,
                                             cuda::ptx::space_shared,
                                             &putBar,
                                             (FABRIC_CHUNK_SIZE / FABRIC_TX_GRANULARITY_BYTES));
        // Spin while the barrier's phase hasn't flipped yet. The report outputs
        // are only valid once the barrier completes, so inspect them afterward.
        bool    isReportSeen = false;
        uint8_t reportValue  = 0;
        bool    complete     = waitWithDebugTimeout(
            [&]() {
                return !cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::mbarrier_phase_primary,
                                                            cuda::ptx::sem_relaxed,
                                                            cuda::ptx::scope_cta,
                                                            isReportSeen,
                                                            reportValue,
                                                            &putBar,
                                                            /*phaseParity=*/0);
            },
            "cftTryPutCountedKernel: mbarrier try_wait parity");
        if (!complete) {
            __trap();
        }
        if (isReportSeen) {
            reportFabricError(&reportValue, offset);
        }
    }
    // Block 0 waits until our receive byte counter reaches expectedBytes. System
    // scope is required because the peer's counted fabric puts update this
    // counter in our fabric-bound memory, not via threads in this CTA.
    // A relaxed load suffices: the fabric-proxy is a strong proxy in the PTX
    // memory model, so fabric-delivered writes are eventually observable
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(*counterPtr);
        if (!waitWithDebugTimeout([&]() { return ref.load(cuda::memory_order_relaxed) != expectedBytes; },
                                  "cftTryPutCountedKernel: receive byte counter")) {
            __trap();
        }
    }
}

// Non-counted variant: each block sends one FABRIC_CHUNK_SIZE chunk via plain
// fabric.try_put (no byte counter). This kernel only moves payload; the inter-GPU
// completion handshake happens in cftSignalFlagKernel, launched next on the same
// stream. The kernel boundary is the cross-block sync point -- once this kernel
// returns, every block's put has been issued and its source SMEM consumed.
__global__ void cftTryPutKernel(uint32_t putLeId, int myRank)
{
    constexpr int elems = FABRIC_CHUNK_SIZE / sizeof(int);
    __shared__ alignas(16) int smemBuf[elems];
    __shared__ __mbarrier_t putBar;

    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(cuda::ptx::layout_v1, &putBar, 1);
    }

    for (int i = threadIdx.x; i < elems; i += blockDim.x) {
        smemBuf[i] = myRank * elems + i;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint64_t offset = blockIdx.x * FABRIC_CHUNK_SIZE;
        // Make the staged shared-memory source visible to the async proxy before the put reads it.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        cuda::ptx::fabric_try_put(cuda::ptx::space_shared,
                                  cuda::ptx::sem_relaxed,
                                  cuda::ptx::scope_sys,
                                  putLeId,
                                  offset,
                                  smemBuf,
                                  FABRIC_CHUNK_SIZE,
                                  &putBar);
        cuda::ptx::fabric_submit();
        cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed,
                                             cuda::ptx::scope_cta,
                                             cuda::ptx::space_shared,
                                             &putBar,
                                             (FABRIC_CHUNK_SIZE / FABRIC_TX_GRANULARITY_BYTES));
        // Spin while the barrier's phase hasn't flipped yet. The report outputs
        // are only valid once the barrier completes, so inspect them afterward.
        bool    isReportSeen = false;
        uint8_t reportValue  = 0;
        bool    complete     = waitWithDebugTimeout(
            [&]() {
                return !cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::mbarrier_phase_primary,
                                                            cuda::ptx::sem_relaxed,
                                                            cuda::ptx::scope_cta,
                                                            isReportSeen,
                                                            reportValue,
                                                            &putBar,
                                                            /*phaseParity=*/0);
            },
            "cftTryPutKernel: mbarrier try_wait parity");
        if (!complete) {
            __trap();
        }
        if (isReportSeen) {
            reportFabricError(&reportValue, offset);
        }
    }
}

// Inter-GPU completion signal for the default (non-counted) path, launched with one
// after cftTryPutKernel. Because it is ordered on the same stream, the payload
// kernel has already finished, so we issue a single flag put to signal completion.
//
// Note: this kernel could be overlapped with the tail of cftTryPutKernel using
// Programmatic Dependent Launch (PDL) to hide its launch latency. The prologue
// (barrier init, staging smemFlag) may run early, but the flag put must still be
// gated on payload completion via cudaGridDependencySynchronize(), so we keep the
// kernels plainly stream-ordered here for clarity.
__global__ void cftSignalFlagKernel(uint32_t putLeId, uint64_t interGpuFlagOffset, uint32_t *interGpuFlag)
{
    __shared__ alignas(16) uint32_t smemFlag[INTER_GPU_FLAG_BYTES / sizeof(uint32_t)];
    __shared__ __mbarrier_t putBar;

    // This kernel is a single-thread, single-block signal: it is launched <<<1, 1>>>
    // and its one thread issues the flag put and waits on the neighbor's flag.
    assert(blockDim.x == 1 && gridDim.x == 1);

    cuda::ptx::mbarrier_init(cuda::ptx::layout_v1, &putBar, 1);
    // A b128 fabric put moves 16 bytes. Initialize the full source even though
    // the receiver treats a non-zero first word as the signal.
    for (size_t i = 0; i < INTER_GPU_FLAG_BYTES / sizeof(uint32_t); ++i) {
        smemFlag[i] = 1;
    }
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    cuda::ptx::fabric_try_put(cuda::ptx::space_shared,
                              cuda::ptx::sem_relaxed,
                              cuda::ptx::scope_sys,
                              putLeId,
                              interGpuFlagOffset,
                              smemFlag,
                              INTER_GPU_FLAG_BYTES,
                              &putBar);
    cuda::ptx::fabric_submit();
    cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed,
                                         cuda::ptx::scope_cta,
                                         cuda::ptx::space_shared,
                                         &putBar,
                                         (INTER_GPU_FLAG_BYTES / FABRIC_TX_GRANULARITY_BYTES));
    // Spin while the barrier's phase hasn't flipped yet. The report outputs are
    // only valid once the barrier completes, so inspect them afterward.
    bool    isReportSeen = false;
    uint8_t reportValue  = 0;
    bool    complete     = waitWithDebugTimeout(
        [&]() {
            return !cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::mbarrier_phase_primary,
                                                        cuda::ptx::sem_relaxed,
                                                        cuda::ptx::scope_cta,
                                                        isReportSeen,
                                                        reportValue,
                                                        &putBar,
                                                        /*phaseParity=*/0);
        },
        "cftSignalFlagKernel: mbarrier try_wait parity");
    if (!complete) {
        __trap();
    }
    if (isReportSeen) {
        reportFabricError(&reportValue, interGpuFlagOffset);
    }

    // Wait for the receive neighbor to set our flag via a fabric put.
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref(*interGpuFlag);
    if (!waitWithDebugTimeout([&]() { return ref.load(cuda::memory_order_relaxed) == 0; },
                              "cftSignalFlagKernel: inter-GPU flag")) {
        __trap();
    }
}

// Confirm the device exposes the capabilities this example relies on: fabric
// handles are always required; counted logical-endpoint ops only in --counted mode.
// Returns false when the sample is not supported on this device (caller should
// exit with EXIT_WAIVED, not EXIT_FAILURE).
static bool requireDeviceCapabilities(CUdevice dev, bool needCounted)
{
    int vmmSupported = 0;
    checkCudaErrors(cuDeviceGetAttribute(&vmmSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev));
    if (!vmmSupported) {
        fprintf(stderr, "device does not support virtual memory management\n");
        return false;
    }

    int unicastSupported = 0;
    checkCudaErrors(cuDeviceGetAttribute(&unicastSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED, dev));
    if (!unicastSupported) {
        fprintf(stderr, "device does not support unicast logical endpoints\n");
        return false;
    }

    int fabricSupported = 0;
    checkCudaErrors(cuDeviceGetAttribute(&fabricSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev));
    if (!fabricSupported) {
        fprintf(stderr, "device does not support fabric handles\n");
        return false;
    }

    // Bitmask of CUlogicalEndpointIpcHandleType.
    int leHandleTypes = 0;
    checkCudaErrors(cuDeviceGetAttribute(&leHandleTypes, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_SUPPORTED_HANDLE_TYPES, dev));
    if ((leHandleTypes & CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC) == 0) {
        fprintf(stderr, "device does not support fabric IPC for logical endpoints\n");
        return false;
    }

    if (needCounted) {
        int countedSupported = 0;
        checkCudaErrors(
            cuDeviceGetAttribute(&countedSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED, dev));
        if (!countedSupported) {
            fprintf(stderr, "device does not support counted logical-endpoint ops\n");
            return false;
        }
    }
    return true;
}

static constexpr int DEFAULT_CHUNK_CNT = 100;

static void printUsage(const char *prog)
{
    fprintf(stderr,
            "Usage: mpirun -np <N> %s [chunkCnt] [--counted]\n"
            "  <N>         number of MPI ranks (must be >= 2)\n"
            "  [chunkCnt]  number of %u-byte chunks each rank sends (default: %d)\n"
            "  --counted   use counted fabric puts with a byte-arrival counter\n"
            "Example: mpirun -np 2 %s\n",
            prog,
            FABRIC_CHUNK_SIZE,
            DEFAULT_CHUNK_CNT,
            prog);
}

int main(int argc, char *argv[])
{
    int  chunkCnt   = DEFAULT_CHUNK_CNT;
    bool useCounted = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--counted") == 0) {
            useCounted = true;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        else {
            char *end = nullptr;
            const long parsed = strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
                fprintf(stderr, "error: invalid argument '%s'\n", argv[i]);
                printUsage(argv[0]);
                return EXIT_FAILURE;
            }
            chunkCnt = static_cast<int>(parsed);
        }
    }
    const size_t totalBytes = static_cast<size_t>(FABRIC_CHUNK_SIZE) * chunkCnt;

    // ---- MPI setup: ring topology, rank i sends to (i+1)%N. ----
    int myRank   = 0;
    int numRanks = 0;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));
    if (numRanks < 2) {
        if (myRank == 0) {
            fprintf(stderr, "error: this example needs at least 2 MPI ranks (got %d)\n", numRanks);
        }
        MPI_CHECK(MPI_Finalize());
        return EXIT_FAILURE;
    }
    if (myRank == 0) {
        printf("mode = %s, chunkCnt = %d, totalBytes = %zu, ranks = %d\n",
               useCounted ? "counted" : "put",
               chunkCnt,
               totalBytes,
               numRanks);
    }
    // Ring neighbors: put to sendRank; receive completion from recvRank.
    const int sendRank = (myRank + 1) % numRanks;
    const int recvRank = (myRank + numRanks - 1) % numRanks;

    // ---- Device context and stream. ----
    // Device ordinals are node-local, so select a device using this process's
    // rank among the MPI processes on the same node.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, myRank, MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));

    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (localRank >= deviceCount) {
        fprintf(stderr,
                "error: local MPI rank %d cannot be assigned a unique device; "
                "only %d device(s) are visible on this node\n",
                localRank,
                deviceCount);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const int deviceOrdinal = localRank;

    checkCudaErrors(cudaSetDevice(deviceOrdinal));
    CUdevice cuDevice;
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&cuDevice, deviceOrdinal));
    int localSupported = requireDeviceCapabilities(cuDevice, useCounted) ? 1 : 0;
    int allSupported   = 0;
    MPI_CHECK(MPI_Allreduce(&localSupported, &allSupported, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (!allSupported) {
        MPI_CHECK(MPI_Comm_free(&localComm));
        MPI_CHECK(MPI_Finalize());
        return EXIT_WAIVED;
    }
    CUstream stream;
    checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    // Query how many cliques this device is a member of:
    size_t cliqueCount = 0; // not modified if there are errors
    checkCudaErrors(cuDeviceGetCliqueCount(&cliqueCount, cuDevice));
    if (cliqueCount == 0) {
        fprintf(stderr, "This GPU is not a member of any clique.");
    }

    // Query the clique info for all cliques
    std::vector<CUcliqueInfo> cliqueInfos(cliqueCount);
    if (cliqueCount > 0) {
        checkCudaErrors(cuDeviceGetCliqueInfo(cliqueInfos.data(), &cliqueCount, cuDevice));
    }
    unsigned int localCliqueId = 0;
    bool         found         = false;
    for (size_t idx = 0; idx < cliqueCount; ++idx) {
        if (cliqueInfos[idx].type == CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT) {
            localCliqueId = cliqueInfos[idx].id;
            found         = true;
            break;
        }
    }
    if (cliqueCount != 0 && !found) {
        fprintf(stderr, "This GPU is not a member of a clique with unicast logical endpoint support.");
    }
    int localCliqueOk = found ? 1 : 0;
    int allCliqueOk   = 0;
    MPI_CHECK(MPI_Allreduce(&localCliqueOk, &allCliqueOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (!allCliqueOk) {
        checkCudaErrors(cuStreamDestroy(stream));
        MPI_CHECK(MPI_Comm_free(&localComm));
        MPI_CHECK(MPI_Finalize());
        return EXIT_WAIVED;
    }

    // Verify that all MPI ranks found the same clique id:
    std::vector<unsigned int> allCliqueIds(numRanks);
    MPI_CHECK(MPI_Allgather(&localCliqueId,
                            sizeof(unsigned int),
                            MPI_BYTE,
                            allCliqueIds.data(),
                            sizeof(unsigned int),
                            MPI_BYTE,
                            MPI_COMM_WORLD));
    for (int i = 0; i < numRanks; i++) {
        if (allCliqueIds[i] != localCliqueId) {
            fprintf(stderr,
                    "Rank %d: Clique id %u does not match expected %u\n",
                    myRank,
                    allCliqueIds[i],
                    localCliqueId);
            return EXIT_FAILURE;
        }
    }

    // Reserve enough local ids for this rank's endpoint and all imported peers.
    CUlogicalEndpointId leId = 0;
    checkCudaErrors(cuLogicalEndpointIdReserve(&leId, (uint32_t)numRanks));

    // ---- Describe the fabric-shareable allocation and query its granularity. ----
    CUmemAllocationProp exportAllocProp  = {};
    exportAllocProp.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    exportAllocProp.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    exportAllocProp.location.id          = cuDevice;
    exportAllocProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t granularity                   = 0;
    checkCudaErrors(cuMemGetAllocationGranularity(&granularity, &exportAllocProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    // ---- Describe the logical endpoint (unicast; counted ops optional). ----
    CUlogicalEndpointProp endpointProp{};
    endpointProp.type           = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
    endpointProp.unicast        = {.device = cuDevice};
    endpointProp.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
    endpointProp.flags          = useCounted ? CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS : CU_LOGICAL_ENDPOINT_FLAG_NONE;
    cuuint64_t bindAlignment;
    cuuint64_t maxSize;
    checkCudaErrors(cuLogicalEndpointGetLimits(&bindAlignment, &maxSize, &endpointProp));
    // Make the allocation and endpoint size a multiple of both independent
    // alignment requirements.
    const size_t requiredAlignment = std::lcm(granularity, static_cast<size_t>(bindAlignment));

    // ---- Allocate payload, sync region, and alignment padding. ----
    size_t exportSize = alignUp(totalBytes + SYNC_REGION_ALIGN_BYTES + SYNC_REGION_BYTES, requiredAlignment);
    if (exportSize > maxSize) {
        fprintf(stderr,
                "error: required endpoint size %zu exceeds maximum %llu\n",
                exportSize,
                static_cast<unsigned long long>(maxSize));
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
        return EXIT_FAILURE;
    }
    endpointProp.size = exportSize;

    // ---- Create the local endpoint, then exchange handles with send neighbor. ----
    checkCudaErrors(cuLogicalEndpointCreate(leId + myRank, &endpointProp));
    // Export our own endpoint to a fabric IPC handle, hand it to the peers that
    // need it, and import each peer's handle to a locally reserved id.
    CUlogicalEndpointFabricHandle localHandle;
    CUlogicalEndpointFabricHandle sendRankHandle;
    checkCudaErrors(cuLogicalEndpointExport(&localHandle, leId + myRank, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
    MPI_CHECK(MPI_Sendrecv(&localHandle,
                           sizeof(CUlogicalEndpointFabricHandle),
                           MPI_BYTE,
                           recvRank,
                           0,
                           &sendRankHandle,
                           sizeof(CUlogicalEndpointFabricHandle),
                           MPI_BYTE,
                           sendRank,
                           0,
                           MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
    checkCudaErrors(cuLogicalEndpointImport(leId + sendRank, &sendRankHandle, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
    // Create and import are non-blocking, so wait until both endpoints are ready.
    waitForEndpointReady(leId + myRank);
    waitForEndpointReady(leId + sendRank);

    // ---- Allocate and bind memory after sharing the endpoint. ----
    CUmemGenericAllocationHandle exportHandle;
    CUdeviceptr                  exportPtr = 0;
    checkCudaErrors(cuMemCreate(&exportHandle, exportSize, &exportAllocProp, 0));
    checkCudaErrors(cuMemAddressReserve(&exportPtr, exportSize, granularity, 0, 0));
    checkCudaErrors(cuMemMap(exportPtr, exportSize, 0, exportHandle, 0));
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id     = exportAllocProp.location.id;
    accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkCudaErrors(cuMemSetAccess(exportPtr, exportSize, &accessDesc, 1));
    // Zero the whole allocation before the first transfer or barrier handshake.
    checkCudaErrors(cuMemsetD8(exportPtr, 0, exportSize));
    // 256-byte-aligned sync region after the payload (required for counted fabric puts).
    std::byte *const basePtr   = reinterpret_cast<std::byte *>(exportPtr);
    void            *syncPtr   = basePtr + totalBytes;
    size_t           syncSpace = exportSize - totalBytes;
    syncPtr                    = std::align(SYNC_REGION_ALIGN_BYTES, SYNC_REGION_BYTES, syncPtr, syncSpace);
    assert(syncPtr != nullptr); // guaranteed by the SYNC_REGION_ALIGN_BYTES slack reserved above
    const size_t syncRegionOffset = static_cast<std::byte *>(syncPtr) - basePtr;
    // counted mode (--counted): local byte counter the peer's counted puts accumulate into.
    uint64_t *const counterPtr = reinterpret_cast<uint64_t *>(syncPtr);
    // default mode: local inter-GPU flag, set by the receive neighbor's fabric put.
    uint32_t *const interGpuFlag = reinterpret_cast<uint32_t *>(syncPtr);
    // Bind the whole backing allocation at endpoint offset 0, by allocation handle.
    checkCudaErrors(cuLogicalEndpointBindMem(leId + myRank, cuDevice, 0, exportHandle, 0, exportSize, 0));
    // No rank may issue an operation until every destination has been bound.
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ---- Push data to sendRank. The kernel writes into the neighbor's endpoint
    //      (leId + sendRank); the receive neighbor's puts land in our own buffer. ----
    const uint32_t sendLe             = leId + sendRank;
    const uint64_t counterOffset      = syncRegionOffset;
    const uint64_t interGpuFlagOffset = syncRegionOffset;

    // Launch one block per chunk: gridDim.x == totalBytes / FABRIC_CHUNK_SIZE.
    if (useCounted) {
        cftTryPutCountedKernel<<<chunkCnt, WARP_SIZE, 0, stream>>>(sendLe,
                                                                   counterOffset,
                                                                   myRank,
                                                                   counterPtr,
                                                                   /*expectedBytes=*/totalBytes);
    }
    else {
        // Payload first, then a one-thread flag kernel; same-stream ordering guarantees
        // the payload puts are done before the inter-GPU flag handshake runs.
        cftTryPutKernel<<<chunkCnt, WARP_SIZE, 0, stream>>>(sendLe, myRank);
        cftSignalFlagKernel<<<1, 1, 0, stream>>>(sendLe, interGpuFlagOffset, interGpuFlag);
    }
    checkCudaErrors(cuStreamSynchronize(stream));

    // ---- Check the payload the receive neighbor pushed into our endpoint. ----
    // Every sending block stages the same rank-tagged pattern, so the expected
    // value at int index i is recvRank * elemsPerChunk + (i % elemsPerChunk).
    constexpr size_t elemsPerChunk = FABRIC_CHUNK_SIZE / sizeof(int);
    const size_t     totalElems    = totalBytes / sizeof(int);
    std::vector<int> received(totalElems);
    checkCudaErrors(cuMemcpyDtoH(received.data(), exportPtr, totalBytes));

    size_t mismatches = 0;
    for (size_t i = 0; i < totalElems; ++i) {
        const int expected = static_cast<int>(recvRank * elemsPerChunk + i % elemsPerChunk);
        if (received[i] != expected) {
            if (mismatches == 0) {
                fprintf(stderr,
                        "rank %d: first mismatch at int index %zu: got %d, expected %d\n",
                        myRank,
                        i,
                        received[i],
                        expected);
            }
            ++mismatches;
        }
    }

    int localFailed = (mismatches != 0) ? 1 : 0;
    int anyFailed   = 0;
    MPI_CHECK(MPI_Allreduce(&localFailed, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (myRank == 0) {
        printf("Result = %s\n", anyFailed ? "FAIL" : "PASS");
    }

    // ---- Clean up endpoints and free the backing allocation. ----
    // Once every rank's stream has synchronized, all fabric puts
    // have completed and no rank will issue another operation. Making that global
    // with a barrier ensures no peer still targets an endpoint we clean up below.
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Destroy the imported endpoint (the peer target our puts addressed). An import
    // is only a local reference to the peer's endpoint id and maps no peer memory,
    // so once our puts have drained this just drops that reference.
    checkCudaErrors(cuLogicalEndpointDestroy(leId + sendRank));

    // Unbind and destroy our owned endpoint, then release the reserved id range.
    checkCudaErrors(cuLogicalEndpointUnbind(leId + myRank, cuDevice, 0, exportSize));
    checkCudaErrors(cuLogicalEndpointDestroy(leId + myRank));
    checkCudaErrors(cuLogicalEndpointIdRelease(leId, numRanks));

    // Free the backing allocation in VMM cleanup order: unmap the range, release
    // the physical handle, then free the reserved virtual address range.
    checkCudaErrors(cuMemUnmap(exportPtr, exportSize));
    checkCudaErrors(cuMemRelease(exportHandle));
    checkCudaErrors(cuMemAddressFree(exportPtr, exportSize));

    checkCudaErrors(cuStreamDestroy(stream));
    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());

    return anyFailed ? EXIT_FAILURE : EXIT_SUCCESS;
}
