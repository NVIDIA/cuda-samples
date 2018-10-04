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

#pragma once

#include "stdafx.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

using namespace DirectX;

struct Vertex
{
	XMFLOAT3 position;
	XMFLOAT4 color;
};

#if 0
// Constant buffer used to send MVP matrices to the vertex shader.
struct ModelViewProjectionConstantBuffer
{
	XMFLOAT4X4 model;
	XMFLOAT4X4 view;
	XMFLOAT4X4 projection;
};
#endif

void RunSineWaveKernel(size_t mesh_width, size_t mesh_height, Vertex *cudaDevVertptr, cudaStream_t streamToRun, float AnimTime);