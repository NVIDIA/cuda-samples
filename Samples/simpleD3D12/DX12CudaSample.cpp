/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

/*Includes code from DirectX-Graphics-Samples/Samples/Desktop/D3D12HelloWorld/src/HelloTexture,
  which is licensed as follows:

The MIT License (MIT)
    Copyright (c) 2015 Microsoft

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include "stdafx.h"
#include "DX12CudaSample.h"
#include <helper_string.h>

using namespace Microsoft::WRL;

DX12CudaSample::DX12CudaSample(UINT width, UINT height, std::string name)
    : m_width(width), m_height(height), m_title(name), m_useWarpDevice(false) {
  m_aspectRatio = static_cast<float>(width) / static_cast<float>(height);
}

DX12CudaSample::~DX12CudaSample() {}

std::wstring DX12CudaSample::string2wstring(const std::string& s) {
  int len;
  int slength = (int)s.length() + 1;
  len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
  wchar_t* buf = new wchar_t[len];
  MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
  std::wstring r(buf);
  delete[] buf;
  return r;
}
// Helper function for resolving the full path of assets.
std::wstring DX12CudaSample::GetAssetFullPath(const char* assetName) {
  LPTSTR lpBuffer = new char[4096];
  GetCurrentDirectory(FILENAME_MAX, lpBuffer);
  char* tmp = sdkFindFilePath((const char*)assetName, "simpleD3D12");
  if (tmp == NULL) {
    throw std::exception("File not found");
  }
  for (int i = 0; i < strlen(tmp); i++) {
    if (tmp[i] == '/') {
      tmp[i] = '\\';
    }
  }
  m_assetsPath = lpBuffer;
  m_assetsPath = m_assetsPath + "\\" + tmp;

  std::wstring stemp = string2wstring(m_assetsPath);

  return stemp;
}

// Helper function for acquiring the first available hardware adapter that
// supports Direct3D 12. If no such adapter can be found, *ppAdapter will be set
// to nullptr.
_Use_decl_annotations_ void DX12CudaSample::GetHardwareAdapter(
    IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter) {
  ComPtr<IDXGIAdapter1> adapter;
  *ppAdapter = nullptr;

  for (UINT adapterIndex = 0;
       DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter);
       ++adapterIndex) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      // Don't select the Basic Render Driver adapter.
      // If you want a software adapter, pass in "/warp" on the command line.
      continue;
    }

    // Check to see if the adapter supports Direct3D 12, but don't create the
    // actual device yet.
    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                    _uuidof(ID3D12Device), nullptr))) {
      break;
    }
  }

  *ppAdapter = adapter.Detach();
}

// Helper function for setting the window's title text.
void DX12CudaSample::SetCustomWindowText(const char* text) {
  std::string windowText = m_title + text;
  SetWindowText(Win32Application::GetHwnd(), windowText.c_str());
}

// Helper function for parsing any supplied command line args.
_Use_decl_annotations_ void DX12CudaSample::ParseCommandLineArgs(WCHAR* argv[],
                                                                 int argc) {
  for (int i = 1; i < argc; ++i) {
    if (_wcsnicmp(argv[i], L"-warp", wcslen(argv[i])) == 0 ||
        _wcsnicmp(argv[i], L"/warp", wcslen(argv[i])) == 0) {
      m_useWarpDevice = true;
      m_title = m_title + " (WARP)";
    }
  }
}
