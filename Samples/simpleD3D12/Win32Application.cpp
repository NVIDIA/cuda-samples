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
#include "Win32Application.h"

HWND Win32Application::m_hwnd = nullptr;

int Win32Application::Run(DX12CudaSample* pSample, HINSTANCE hInstance, int nCmdShow)
{
	// Parse the command line parameters
	int argc;
	LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	pSample->ParseCommandLineArgs(argv, argc);
	LocalFree(argv);

	// Initialize the window class.
	WNDCLASSEX windowClass = { 0 };
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = WindowProc;
	windowClass.hInstance = hInstance;
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.lpszClassName = "DX12CudaSampleClass";
	RegisterClassEx(&windowClass);

	RECT windowRect = { 0, 0, static_cast<LONG>(pSample->GetWidth()), static_cast<LONG>(pSample->GetHeight()) };
	AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

	// Create the window and store a handle to it.
	m_hwnd = CreateWindow(
		windowClass.lpszClassName,
		pSample->GetTitle(),
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		nullptr,		// We have no parent window.
		nullptr,		// We aren't using menus.
		hInstance,
		pSample);

	// Initialize the sample. OnInit is defined in each child-implementation of DXSample.
	pSample->OnInit();

	ShowWindow(m_hwnd, nCmdShow);

	// Main sample loop.
	MSG msg = {};
	while (msg.message != WM_QUIT)
	{
		// Process any messages in the queue.
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	pSample->OnDestroy();

	// Return this part of the WM_QUIT message to Windows.
	return static_cast<char>(msg.wParam);
}

// Main message handler for the sample.
LRESULT CALLBACK Win32Application::WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	DX12CudaSample* pSample = reinterpret_cast<DX12CudaSample*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

	switch (message)
	{
	case WM_CREATE:
		{
			// Save the DXSample* passed in to CreateWindow.
			LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
			SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
		}
		return 0;

	case WM_KEYDOWN:
		if (pSample)
		{
			pSample->OnKeyDown(static_cast<UINT8>(wParam));
		}
		return 0;

	case WM_KEYUP:
		if (pSample)
		{
			pSample->OnKeyUp(static_cast<UINT8>(wParam));
		}
		return 0;

	case WM_PAINT:
		if (pSample)
		{
			pSample->OnRender();
		}
		return 0;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}

	// Handle any messages the switch statement didn't.
	return DefWindowProc(hWnd, message, wParam, lParam);
}
