; Copyright (c) 2014-2023, NVIDIA CORPORATION. All rights reserved.
;
; Redistribution and use in source and binary forms, with or without
; modification, are permitted provided that the following conditions
; are met:
;  * Redistributions of source code must retain the above copyright
;    notice, this list of conditions and the following disclaimer.
;  * Redistributions in binary form must reproduce the above copyright
;    notice, this list of conditions and the following disclaimer in the
;    documentation and/or other materials provided with the distribution.
;  * Neither the name of NVIDIA CORPORATION nor the names of its
;    contributors may be used to endorse or promote products derived
;    from this software without specific prior written permission.
;
; THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
; EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
; IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
; PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
; CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
; EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
; PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
; PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
; OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
; (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
; OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

; This NVVM IR program shows how to call cudaGetParameterBuffer and cudaLaunchDevice functions.
; What it does is similar to the following CUDA C code.
;
; __global__ void kernel(int depth)
; {
;   if (threadIdx.x == 0) {
;     printf("kernel launched, depth = %d\n", depth);
;   }
;
;   __syncthreads();
;
;   if (++depth > 3)
;     return;
;
;   kernel<<<1,1>>>(depth);
;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

@"$str" = private addrspace(1) constant [29 x i8] c"kernel launched, depth = %d\0A\00"

define void @kernel(i32 %depth) {
entry:
  %tmp31 = alloca i32, align 8
  %gen2local = addrspacecast i32* %tmp31 to i32 addrspace(5)*
  %tmp31.sub = bitcast i32* %tmp31 to i8*
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = addrspacecast i8 addrspace(1)* getelementptr inbounds ([29 x i8], [29 x i8] addrspace(1)* @"$str", i64 0, i64 0) to i8 addrspace(0)*
  store i32 %depth, i32 addrspace(5)* %gen2local, align 8
  %call = call i32 @vprintf(i8* %1, i8* %tmp31.sub)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.cuda.syncthreads()
  %inc = add nsw i32 %depth, 1
  %cmp5 = icmp sgt i32 %inc, 3
  br i1 %cmp5, label %return, label %if.end7

if.end7:                                          ; preds = %if.end
  %call15 = call i8* @cudaGetParameterBufferV2(i8* bitcast (void (i32)* @kernel to i8*), %struct.dim3 { i32 1, i32 1, i32 1 }, %struct.dim3 { i32 1, i32 1, i32 1 }, i32 0)
  %tobool = icmp eq i8* %call15, null
  br i1 %tobool, label %return, label %cond.true

cond.true:                                        ; preds = %if.end7
  %conv = bitcast i8* %call15 to i32*
  store i32 %inc, i32* %conv, align 4
  %call20 = call i32 @cudaLaunchDeviceV2(i8* %call15, %struct.CUstream_st* null)
  br label %return

return:                                           ; preds = %cond.true, %if.end7, %if.end
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

declare i32 @vprintf(i8* nocapture, i8*) nounwind

declare void @llvm.cuda.syncthreads() nounwind

declare i8* @cudaGetParameterBufferV2(i8*, %struct.dim3, %struct.dim3, i32)

declare i32 @cudaLaunchDeviceV2(i8*, %struct.CUstream_st*)

!nvvm.annotations = !{!0}
!0 = !{void (i32)* @kernel, !"kernel", i32 1}

!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0}
