/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include "ShaderStructs.h"
#include "helper_cuda.h"

__global__ void sinewave_gen_kernel(Vertex *vertices, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    if (y < height && x < width)
    {
        // write output vertex
        vertices[y*width+x].position.x = u;
        vertices[y*width+x].position.y = w;
        vertices[y*width+x].position.z = v;
        //vertices[y*width+x].position[3] = 1.0f;
        vertices[y*width+x].color.x = 1.0f;
        vertices[y*width+x].color.y = 0.0f;
        vertices[y*width+x].color.z = 0.0f;
		vertices[y*width + x].color.w = 0.0f;
    }
}

// The host CPU Sinewave thread spawner
void RunSineWaveKernel(size_t mesh_width, size_t mesh_height, Vertex *cudaDevVertptr, cudaStream_t streamToRun, float AnimTime)
{
	dim3 block(16, 16, 1);
	dim3 grid(mesh_width / 16, mesh_height / 16, 1);
	Vertex *vertices = (Vertex*)cudaDevVertptr;
	sinewave_gen_kernel<<< grid, block, 0, streamToRun >>>(vertices, mesh_width, mesh_height, AnimTime);

	getLastCudaError("sinewave_gen_kernel execution failed.\n");
}

