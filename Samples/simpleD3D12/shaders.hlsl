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


struct PSInput
{
	float4 position : SV_POSITION;
	float4 color : COLOR;
};

PSInput VSMain(float3 position : POSITION, float4 color : COLOR)
{
	PSInput result;

	result.position = float4(position, 1.0f);

	// Pass the color through without modification.
	result.color = color;

	return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
	return input.color;
}
