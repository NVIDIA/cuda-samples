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

/*
 * Various kernels and functors used throughout the algorithm. For details
 * on usage see "SegmentationTreeBuilder::invokeStep()".
 */

#ifndef _KERNELS_H_
#define _KERNELS_H_

#include <stdio.h>
#include <thrust/functional.h>

#include "common.cuh"

// Functors used with thrust library.
template <typename Input>
struct IsGreaterEqualThan : public thrust::unary_function<Input, bool>
{
    __host__ __device__ IsGreaterEqualThan(uint upperBound) :
        upperBound_(upperBound) {}

    __host__ __device__ bool operator()(const Input &value) const
    {
        return value >= upperBound_;
    }

    uint upperBound_;
};

// CUDA kernels.
__global__ void addScalar(uint *array, int scalar, uint size)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        array[tid] += scalar;
    }
}

__global__ void markSegments(const uint *verticesOffsets,
                             uint *flags,
                             uint verticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        flags[verticesOffsets[tid]] = 1;
    }
}

__global__ void getVerticesMapping(const uint *clusteredVerticesIDs,
                                   const uint *newVerticesIDs,
                                   uint *verticesMapping,
                                   uint verticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        uint vertexID = clusteredVerticesIDs[tid];
        verticesMapping[vertexID] = newVerticesIDs[tid];
    }
}

__global__ void getSuccessors(const uint *verticesOffsets,
                              const uint *minScannedEdges,
                              uint *successors,
                              uint verticesCount,
                              uint edgesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        uint successorPos = (tid < verticesCount - 1) ?
                            (verticesOffsets[tid + 1] - 1) :
                            (edgesCount - 1);

        successors[tid] = minScannedEdges[successorPos];
    }
}

__global__ void removeCycles(uint *successors,
                             uint verticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        uint successor = successors[tid];
        uint nextSuccessor = successors[successor];

        if (tid == nextSuccessor)
        {
            if (tid < successor)
            {
                successors[tid] = tid;
            }
            else
            {
                successors[successor] = successor;
            }
        }
    }
}

__global__ void getRepresentatives(const uint *successors,
                                   uint *representatives,
                                   uint verticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        uint successor = successors[tid];
        uint nextSuccessor = successors[successor];

        while (successor != nextSuccessor)
        {
            successor = nextSuccessor;
            nextSuccessor = successors[nextSuccessor];
        }

        representatives[tid] = successor;
    }
}

__global__ void invalidateLoops(const uint *startpoints,
                                const uint *verticesMapping,
                                uint *edges,
                                uint edgesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < edgesCount)
    {
        uint startpoint = startpoints[tid];
        uint &endpoint = edges[tid];

        uint newStartpoint = verticesMapping[startpoint];
        uint newEndpoint = verticesMapping[endpoint];

        if (newStartpoint == newEndpoint)
        {
            endpoint = UINT_MAX;
        }
    }
}

__global__ void calculateEdgesInfo(const uint *startpoints,
                                   const uint *verticesMapping,
                                   const uint *edges,
                                   const float *weights,
                                   uint *newStartpoints,
                                   uint *survivedEdgesIDs,
                                   uint edgesCount,
                                   uint newVerticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < edgesCount)
    {
        uint startpoint = startpoints[tid];
        uint endpoint = edges[tid];

        newStartpoints[tid] = endpoint < UINT_MAX ?
                              verticesMapping[startpoint] :
                              newVerticesCount + verticesMapping[startpoint];

        survivedEdgesIDs[tid] = endpoint < UINT_MAX ?
                                tid :
                                UINT_MAX;
    }
}

__global__ void makeNewEdges(const uint *survivedEdgesIDs,
                             const uint *verticesMapping,
                             const uint *edges,
                             const float *weights,
                             uint *newEdges,
                             float *newWeights,
                             uint edgesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < edgesCount)
    {
        uint edgeID = survivedEdgesIDs[tid];
        uint oldEdge = edges[edgeID];

        newEdges[tid] = verticesMapping[oldEdge];
        newWeights[tid] = weights[edgeID];
    }
}

#endif // #ifndef _KERNELS_H_
