/* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
 * This is a simple application showing the performance characteristics of cudaGraphs.
 */

#define USE_NVTX

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

typedef volatile int LatchType;

std::chrono::time_point<std::chrono::high_resolution_clock> getCpuTime()
{
    return std::chrono::high_resolution_clock::now();
}

template <typename T> float getMicroSecondDuration(T start, T end)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * .001f;
}

float getAsyncMicroSecondDuration(cudaEvent_t start, cudaEvent_t end)
{
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    return ms * 1000;
}

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>

class Tracer
{
public:
    Tracer(const char *name) { nvtxRangePushA(name); }
    ~Tracer() { nvtxRangePop(); }
};
#define RANGE(name)      Tracer uniq_name_using_macros(name);
#define RANGE_PUSH(name) nvtxRangePushA(name)
#define RANGE_POP()      nvtxRangePop();
#else
#define RANGE(name)
#endif

std::vector<cudaStream_t> stream;
cudaEvent_t               event[1];
cudaEvent_t               timingEvent[2];

struct hostData
{
    long long timeElapsed;
    bool      timeoutDetected;
    long long timeElapsed2;
    bool      timeoutDetected2;
    LatchType latch;
    LatchType latch2;
};

struct hostData *hostData;

__global__ void empty() {}

// Function to read the GPU nanosecond timer in a kernel
__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

__global__ void delay(long long ticks)
{
    long long endTime = clock64() + ticks;
    while (clock64() < endTime)
        ;
}

__global__ void waitWithTimeout(long long nanoseconds, bool *timeoutDetected, long long *timeElapsed, LatchType *latch)
{
    long long startTime = __globaltimer();
    long long endTime   = startTime + nanoseconds;
    long long time      = 0;
    do {
        time = __globaltimer();
    } while (time < endTime && (latch == NULL || *latch == 0));
    if (timeElapsed != NULL) {
        *timeElapsed = time - startTime;
    }
    if (timeoutDetected) {
        // report timeout if latch not detected
        *timeoutDetected = (latch == NULL || *latch == 0);
    }
}

__global__ void preUploadAnnotation() {}

__global__ void postUploadAnnotation() {}

cudaGraph_t createParallelChain(int length, int width, bool singleEntry = false)
{
    RANGE_PUSH(__func__);
    RANGE("capture");
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeGlobal);
    int streamIdx = 0;
    if (singleEntry) {
        empty<<<1, 1, 0, stream[streamIdx]>>>();
    }

    cudaEventRecord(event[0], stream[0]);
    for (int i = 1; i < width; i++) {
        cudaStreamWaitEvent(stream[i], event[0]);
    }

    for (int i = 0; i < width; i++) {
        streamIdx = i;
        for (int j = 0; j < length; j++) {
            empty<<<1, 1, 0, stream[streamIdx]>>>();
        }
    }

    for (int i = 1; i < width; i++) {
        cudaEventRecord(event[0], stream[i]);
        cudaStreamWaitEvent(stream[0], event[0]);
    }

    cudaStreamEndCapture(stream[0], &graph);
    return graph;
}

std::vector<const char *> metricName;
std::vector<float>        metricValue;

int  counter2 = 0;
void runDemo(cudaGraph_t graph, int length, int width)
{
    cudaGraphExec_t graphExec;
    {
        auto start = getCpuTime();
        cudaGraphInstantiateWithFlags(&graphExec, graph, 0);
        auto end = getCpuTime();
        metricName.push_back("instantiation");
        metricValue.push_back(getMicroSecondDuration(start, end));
    }
    {
        RANGE("launch including upload");
        auto start = getCpuTime();
        cudaGraphLaunch(graphExec, stream[0]);
        auto apiReturn = getCpuTime();
        cudaStreamSynchronize(stream[0]);
        auto streamSync = getCpuTime();
        metricName.push_back("first_launch_api");
        metricValue.push_back(getMicroSecondDuration(start, apiReturn));
        metricName.push_back("first_launch_total");
        metricValue.push_back(getMicroSecondDuration(start, streamSync));
    }
    {
        RANGE("repeat lauch in empty stream");
        auto start = getCpuTime();
        cudaGraphLaunch(graphExec, stream[0]);
        auto apiReturn = getCpuTime();
        cudaStreamSynchronize(stream[0]);
        auto streamSync = getCpuTime();
        metricName.push_back("repeat_launch_api");
        metricValue.push_back(getMicroSecondDuration(start, apiReturn));
        metricName.push_back("repeat_launch_total");
        metricValue.push_back(getMicroSecondDuration(start, streamSync));
    }
    {
        // re-instantiating the exec to simulate first launch into a busy stream.
        cudaGraphExecDestroy(graphExec);
        cudaGraphInstantiateWithFlags(&graphExec, graph, 0);

        long long maxTimeoutNanoSeconds = 4000 + 500 * length * width;
        waitWithTimeout<<<1, 1, 0, stream[0]>>>(
            maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);

        RANGE("launch including upload in busy stream");
        cudaEventRecord(timingEvent[0], stream[0]);
        cudaGraphLaunch(graphExec, stream[0]);
        cudaEventRecord(timingEvent[1], stream[0]);

        hostData->latch = 1;
        cudaStreamSynchronize(stream[0]);

        metricName.push_back("first_launch_device");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));
        metricName.push_back("blockingKernelTimeoutDetected");
        metricValue.push_back(hostData->timeoutDetected);
        hostData->latch           = 0;
        hostData->timeoutDetected = 0;
    }
    {
        RANGE("repeat lauch in busy stream");
        long long maxTimeoutNanoSeconds = 4000 + 500 * length * width;
        waitWithTimeout<<<1, 1, 0, stream[0]>>>(
            maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);
        cudaEventRecord(timingEvent[0], stream[0]);
        cudaGraphLaunch(graphExec, stream[0]);
        cudaEventRecord(timingEvent[1], stream[0]);

        hostData->latch = 1;
        cudaStreamSynchronize(stream[0]);

        metricName.push_back("repeat_launch_device");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));
        metricName.push_back("blockingKernelTimeoutDetected");
        metricValue.push_back(hostData->timeoutDetected);
        hostData->latch           = 0;
        hostData->timeoutDetected = 0;
    }
    {
        // re-instantiating the exec to provide upload with work to do.
        cudaGraphExecDestroy(graphExec);
        cudaGraphInstantiateWithFlags(&graphExec, graph, 0);
        long long maxTimeoutNanoSeconds = 4000 + 1000 * length * width;
        waitWithTimeout<<<1, 1, 0, stream[0]>>>(
            maxTimeoutNanoSeconds, &hostData->timeoutDetected2, &hostData->timeElapsed2, &hostData->latch2);
        maxTimeoutNanoSeconds = 2000 + 500 * length * width;
        waitWithTimeout<<<1, 1, 0, stream[1]>>>(
            maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);

        RANGE("uploading a graph off of the critical path");
        preUploadAnnotation<<<1, 1, 0, stream[1]>>>();
        cudaEventRecord(timingEvent[0], stream[0]);
        auto start = getCpuTime();
        cudaGraphUpload(graphExec, stream[1]);
        auto apiReturn = getCpuTime();
        cudaEventRecord(event[0], stream[1]);
        cudaEventRecord(timingEvent[1], stream[0]);
        postUploadAnnotation<<<1, 1, 0, stream[1]>>>();

        hostData->latch = 1; // release the blocking kernel for the upload
        cudaStreamWaitEvent(stream[0], event[0]);
        cudaGraphLaunch(graphExec, stream[0]);
        cudaEventSynchronize(event[0]); // upload done, similuate critical path being ready for the graph to run by the
                                        // release of the second latch

        hostData->latch2 = 1; // release the work
        cudaStreamSynchronize(stream[0]);

        metricName.push_back("upload_api_time");
        metricValue.push_back(getMicroSecondDuration(start, apiReturn));
        metricName.push_back("updoad_device_time");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));
        metricName.push_back("blockingKernelTimeoutDetected");
        metricValue.push_back(hostData->timeoutDetected);

        hostData->latch            = 0;
        hostData->latch2           = 0;
        hostData->timeoutDetected  = 0;
        hostData->timeoutDetected2 = 0;
    }
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    RANGE_POP();
}

void usage()
{
    printf("programName [outputFmt] [numTrials] [length] [width] [pattern] [stride] [maxLength] \n");
    printf("\toutputFmt - program output, default=3 (see below)\n");
    printf("\tnumTrials (per length)\n");
    printf("\tstarting length of the topology\n");
    printf("\twidth - width of the graph topology\n");
    printf("\tpattern - Structure of graph, default=0 (see below)\n");
    printf("\tstride - how to grow the length between each set of trials \n");
    printf("\tmaxLength - maximum lenght to try \n");
    printf("\n");
    printf("outputFmt can be:\n");
    printf("\t0: this help message\n");
    printf("\t1: csv data headers\n");
    printf("\t2: per trial csv data\n");
    printf("\t3: csv data & headers\n");
    printf("\t4: csv data is printed and trials are averaged for each length\n");
    printf("\t5: csv data is printed and trials are averaged for each length and headers are printed\n");
    printf("\n");
    printf("Pattern can be:\n");
    printf("\t0: No interconnect between branches\n");
    printf("\t1: Adds an extra root node before the initial fork\n");
}

int main(int argc, char **argv)
{
    if (argc < 1) {
        usage();
        return 0;
    }

    int numTrials = 1, length = 20, width = 1, outputFmt = 3, pattern = 0, stride = 1;
    if (argc > 1)
        outputFmt = atoi(argv[1]);
    if (argc > 2)
        numTrials = atoi(argv[2]);
    if (argc > 3)
        length = atoi(argv[3]);
    if (argc > 4)
        width = atoi(argv[4]);
    if (argc > 5)
        pattern = atoi(argv[5]);
    if (argc > 6)
        stride = atoi(argv[6]);
    int maxLength = length;
    if (argc > 7)
        maxLength = atoi(argv[7]);
    if (maxLength < length) {
        maxLength = length;
    }

    if ((outputFmt & 4) && (outputFmt & 2)) {
        printf("printing average and all samples doesn't make sense\n");
    }

    if (length == 0 || width == 0 || outputFmt == 0 || outputFmt > 5 || pattern > 1) {
        usage();
        return 0;
    }

    bool singleEntry = (pattern == 1);

    cudaGraph_t graph;

    cudaFree(0);
    cudaMallocHost(&hostData, sizeof(*hostData));
    int numStreams = width;
    if (numStreams == 1)
        numStreams = 2; // demo needs two streams even if capture only needs 1.
    stream.resize(numStreams);
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    cudaEventCreate(&event[0], cudaEventDisableTiming);
    cudaEventCreate(&timingEvent[0], 0);
    cudaEventCreate(&timingEvent[1], 0);

    {
        RANGE("warmup");
        for (int i = 0; i < width; i++) {
            empty<<<1, 1, 0, stream[i]>>>();
        }
        cudaStreamSynchronize(stream[0]);

        auto start = getCpuTime();
        graph      = createParallelChain(length, width, singleEntry);
        auto end   = getCpuTime();
        metricValue.push_back(getMicroSecondDuration(start, end));
        metricName.push_back("capture");
        runDemo(graph, length, width);
    }

    if (outputFmt & 1) {
        printf("length, width, pattern, ");
        for (int i = 0; i < metricName.size(); i++) {
            printf("%s, ", metricName[i]);
        }
        printf("\r\n");
    }

    if (!(outputFmt & 6)) {
        printf("skipping trials since no output is expected\n");
        return 1;
    }

    std::vector<double> metricTotal;
    metricTotal.resize(metricValue.size());

    while (length <= maxLength) {
        for (int i = 0; i < numTrials; i++) {
            metricName.clear();
            metricValue.clear();
            auto start = getCpuTime();
            graph      = createParallelChain(length, width, singleEntry);
            auto end   = getCpuTime();
            metricValue.push_back(getMicroSecondDuration(start, end));

            runDemo(graph, length, width);

            if (outputFmt & 2) {
                printf("%d, %d, %d, ", length, width, pattern);
                for (int i = 0; i < metricValue.size(); i++) {
                    printf("%0.3f, ", metricValue[i]);
                }
                printf("\r\n");
            }
            if (outputFmt & 4) {
                for (int i = 0; i < metricTotal.size(); i++) {
                    metricTotal[i] += metricValue[i];
                }
            }
        }

        if (outputFmt & 4) {
            printf("%d, %d, %d, ", length, width, pattern);
            for (int i = 0; i < metricTotal.size(); i++) {
                printf("%0.3f, ", metricTotal[i] / numTrials);
                metricTotal[i] = 0;
            }
            printf("\r\n");
        }

        length += stride;
    }

    cudaFreeHost(hostData);

    printf("\n");
    printf("Test passed\n");
    return 0;
}
