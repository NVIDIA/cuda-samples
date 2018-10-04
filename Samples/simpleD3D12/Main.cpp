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

#include "stdafx.h"
#include "simpleD3D12.h"

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	DX12CudaInterop sample(1280, 720, "D3D12 CUDA Interop");
	return Win32Application::Run(&sample, hInstance, nCmdShow);
}
