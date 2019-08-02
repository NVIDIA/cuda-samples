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

#pragma once

#include "DXSampleHelper.h"
#include "Win32Application.h"

class DX12CudaSample {
 public:
  DX12CudaSample(UINT width, UINT height, std::string name);
  virtual ~DX12CudaSample();

  virtual void OnInit() = 0;
  virtual void OnRender() = 0;
  virtual void OnDestroy() = 0;

  // Samples override the event handlers to handle specific messages.
  virtual void OnKeyDown(UINT8 /*key*/) {}
  virtual void OnKeyUp(UINT8 /*key*/) {}

  // Accessors.
  UINT GetWidth() const { return m_width; }
  UINT GetHeight() const { return m_height; }
  const CHAR* GetTitle() const { return m_title.c_str(); }

  void ParseCommandLineArgs(_In_reads_(argc) WCHAR* argv[], int argc);

 protected:
  std::wstring GetAssetFullPath(const char* assetName);
  void GetHardwareAdapter(_In_ IDXGIFactory2* pFactory,
                          _Outptr_result_maybenull_ IDXGIAdapter1** ppAdapter);
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
