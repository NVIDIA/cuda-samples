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

#include "DXSampleHelper.h"
#include "Win32Application.h"

class DX12CudaSample
{
public:
	DX12CudaSample(UINT width, UINT height, std::string name);
	virtual ~DX12CudaSample();

	virtual void OnInit() = 0;
	virtual void OnRender() = 0;
	virtual void OnDestroy() = 0;

	// Samples override the event handlers to handle specific messages.
	virtual void OnKeyDown(UINT8 /*key*/)   {}
	virtual void OnKeyUp(UINT8 /*key*/)     {}

	// Accessors.
	UINT GetWidth() const           { return m_width; }
	UINT GetHeight() const          { return m_height; }
	const CHAR* GetTitle() const   { return m_title.c_str(); }

	void ParseCommandLineArgs(_In_reads_(argc) WCHAR* argv[], int argc);

protected:
	std::wstring GetAssetFullPath(const char* assetName);
	void GetHardwareAdapter(_In_ IDXGIFactory2* pFactory, _Outptr_result_maybenull_ IDXGIAdapter1** ppAdapter);
	void SetCustomWindowText(const char* text);
	std::wstring string2wstring(const std::string& s);

	// Viewport dimensions.
	UINT m_width;
	UINT m_height;
	float m_aspectRatio;

	// Adapter info.
	bool m_useWarpDevice;

private:
	// Root assets path.
	std::string m_assetsPath;

	// Window title.
	std::string m_title;
};
