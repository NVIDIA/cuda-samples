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

#include <stdexcept>
// Note that while ComPtr is used to manage the lifetime of resources on the
// CPU, it has no understanding of the lifetime of resources on the GPU. Apps
// must account for the GPU lifetime of resources to avoid destroying objects
// that may still be referenced by the GPU.
using Microsoft::WRL::ComPtr;

inline std::string HrToString(HRESULT hr) {
  char s_str[64] = {};
  sprintf_s(s_str, "HRESULT of 0x%08X", static_cast<UINT>(hr));
  return std::string(s_str);
}

class HrException : public std::runtime_error {
 public:
  HrException(HRESULT hr) : std::runtime_error(HrToString(hr)), m_hr(hr) {}
  HRESULT Error() const { return m_hr; }

 private:
  const HRESULT m_hr;
};

#define SAFE_RELEASE(p) \
  if (p) (p)->Release()

inline void ThrowIfFailed(HRESULT hr) {
  if (FAILED(hr)) {
    throw HrException(hr);
  }
}

inline HRESULT ReadDataFromFile(LPCWSTR filename, byte** data, UINT* size) {
  using namespace Microsoft::WRL;

  CREATEFILE2_EXTENDED_PARAMETERS extendedParams = {};
  extendedParams.dwSize = sizeof(CREATEFILE2_EXTENDED_PARAMETERS);
  extendedParams.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
  extendedParams.dwFileFlags = FILE_FLAG_SEQUENTIAL_SCAN;
  extendedParams.dwSecurityQosFlags = SECURITY_ANONYMOUS;
  extendedParams.lpSecurityAttributes = nullptr;
  extendedParams.hTemplateFile = nullptr;

  Wrappers::FileHandle file(CreateFile2(filename, GENERIC_READ, FILE_SHARE_READ,
                                        OPEN_EXISTING, &extendedParams));
  if (file.Get() == INVALID_HANDLE_VALUE) {
    throw std::exception();
  }

  FILE_STANDARD_INFO fileInfo = {};
  if (!GetFileInformationByHandleEx(file.Get(), FileStandardInfo, &fileInfo,
                                    sizeof(fileInfo))) {
    throw std::exception();
  }

  if (fileInfo.EndOfFile.HighPart != 0) {
    throw std::exception();
  }

  *data = reinterpret_cast<byte*>(malloc(fileInfo.EndOfFile.LowPart));
  *size = fileInfo.EndOfFile.LowPart;

  if (!ReadFile(file.Get(), *data, fileInfo.EndOfFile.LowPart, nullptr,
                nullptr)) {
    throw std::exception();
  }

  return S_OK;
}

// Assign a name to the object to aid with debugging.
#if defined(_DEBUG) || defined(DBG)
inline void SetName(ID3D12Object* pObject, LPCWSTR name) {
  pObject->SetName(name);
}
inline void SetNameIndexed(ID3D12Object* pObject, LPCWSTR name, UINT index) {
  WCHAR fullName[50];
  if (swprintf_s(fullName, L"%s[%u]", name, index) > 0) {
    pObject->SetName(fullName);
  }
}
#else
inline void SetName(ID3D12Object*, LPCWSTR) {}
inline void SetNameIndexed(ID3D12Object*, LPCWSTR, UINT) {}
#endif

// Naming helper for ComPtr<T>.
// Assigns the name of the variable as the name of the object.
// The indexed variant will include the index in the name of the object.
#define NAME_D3D12_OBJECT(x) SetName((x).Get(), L#x)
#define NAME_D3D12_OBJECT_INDEXED(x, n) SetNameIndexed((x)[n].Get(), L#x, n)

inline UINT CalculateConstantBufferByteSize(UINT byteSize) {
  // Constant buffer size is required to be aligned.
  return (byteSize + (D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1)) &
         ~(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1);
}

#ifdef D3D_COMPILE_STANDARD_FILE_INCLUDE
inline Microsoft::WRL::ComPtr<ID3DBlob> CompileShader(
    const std::wstring& filename, const D3D_SHADER_MACRO* defines,
    const std::string& entrypoint, const std::string& target) {
  UINT compileFlags = 0;
#if defined(_DEBUG) || defined(DBG)
  compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

  HRESULT hr;

  Microsoft::WRL::ComPtr<ID3DBlob> byteCode = nullptr;
  Microsoft::WRL::ComPtr<ID3DBlob> errors;
  hr = D3DCompileFromFile(filename.c_str(), defines,
                          D3D_COMPILE_STANDARD_FILE_INCLUDE, entrypoint.c_str(),
                          target.c_str(), compileFlags, 0, &byteCode, &errors);

  if (errors != nullptr) {
    OutputDebugStringA((char*)errors->GetBufferPointer());
  }
  ThrowIfFailed(hr);

  return byteCode;
}
#endif

// Resets all elements in a ComPtr array.
template <class T>
void ResetComPtrArray(T* comPtrArray) {
  for (auto& i : *comPtrArray) {
    i.Reset();
  }
}

// Resets all elements in a unique_ptr array.
template <class T>
void ResetUniquePtrArray(T* uniquePtrArray) {
  for (auto& i : *uniquePtrArray) {
    i.reset();
  }
}
