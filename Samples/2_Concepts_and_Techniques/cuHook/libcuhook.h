/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _CUHOOK_H_
#define _CUHOOK_H_

typedef enum HookTypesEnum {
  PRE_CALL_HOOK,
  POST_CALL_HOOK,
  CU_HOOK_TYPES,
} HookTypes;

typedef enum HookSymbolsEnum {
  CU_HOOK_MEM_ALLOC,
  CU_HOOK_MEM_FREE,
  CU_HOOK_CTX_GET_CURRENT,
  CU_HOOK_CTX_SET_CURRENT,
  CU_HOOK_CTX_DESTROY,
  CU_HOOK_SYMBOLS,
} HookSymbols;

// One and only function to call to register a callback
// You need to dlsym this symbol in your application and call it to register
// callbacks
typedef void (*fnCuHookRegisterCallback)(HookSymbols symbol, HookTypes type,
                                         void* callback);
extern "C" {
void cuHookRegisterCallback(HookSymbols symbol, HookTypes type, void* callback);
}

// In case you want to intercept, the callbacks need the same type/parameters as
// the real functions
typedef CUresult CUDAAPI (*fnMemAlloc)(CUdeviceptr* dptr, size_t bytesize);
typedef CUresult CUDAAPI (*fnMemFree)(CUdeviceptr dptr);
typedef CUresult CUDAAPI (*fnCtxGetCurrent)(CUcontext* pctx);
typedef CUresult CUDAAPI (*fnCtxSetCurrent)(CUcontext ctx);
typedef CUresult CUDAAPI (*fnCtxDestroy)(CUcontext ctx);

#endif /* _CUHOOK_H_ */
