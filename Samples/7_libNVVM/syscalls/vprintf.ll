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

; This NVVM IR programs shows how to call the vprintf function.
; What it does is similar to the following CUDA C code.
;
; __global__  void foo(char c, short s, int i, long l,
;                      float f, double d, char* p)
; {
;   printf("Hello world from %c %hi %d %ld %f %f %s\n", c, s, i, l, f, d, p);
; }
;
; There is no direct printf() support. In order to use
; vprintf(), a local buffer is allocated. Integer types that
; are shorter than int need to be extended to int and float
; needs to be extended to double before being pushed into
; the local buffer.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@"$str" = private addrspace(4) constant [41 x i8] c"Hello world from %c %hi %d %ld %f %f %s\0A\00"

define void @foo(i8 signext %c, i16 signext %s, i32 %i, i64 %l, float %f, double %d, i8* %p) {
entry:
  %tmp = alloca [12 x i32], align 8
  %tmp2 = getelementptr inbounds [12 x i32], [12 x i32]* %tmp, i64 0, i64 0
  %gen2local = addrspacecast i32* %tmp2 to i32 addrspace(5)*

  %conv = sext i8 %c to i32
  store i32 %conv, i32 addrspace(5)* %gen2local, align 8

  %conv2 = sext i16 %s to i32
  %getElem11 = getelementptr i32, i32 addrspace(5)* %gen2local, i64 1
  store i32 %conv2, i32 addrspace(5)* %getElem11, align 4

  %getElem12 = getelementptr i32, i32 addrspace(5)* %gen2local, i64 2
  store i32 %i, i32 addrspace(5)* %getElem12, align 8

  %getElem13 = getelementptr i32, i32 addrspace(5)* %gen2local, i64 4
  %bitCast = bitcast i32 addrspace(5)* %getElem13 to i64 addrspace(5)*
  store i64 %l, i64 addrspace(5)* %bitCast, align 8

  %getElem14 = getelementptr i32, i32 addrspace(5)* %gen2local, i64 6
  %conv6 = fpext float %f to double
  %bitCast15 = bitcast i32 addrspace(5)* %getElem14 to double addrspace(5)*
  store double %conv6, double addrspace(5)* %bitCast15, align 8

  %getElem16 = getelementptr i32, i32 addrspace(5)* %gen2local, i64 8
  %bitCast17 = bitcast i32 addrspace(5)* %getElem16 to double addrspace(5)*
  store double %d, double addrspace(5)* %bitCast17, align 8

  %getElem18 = getelementptr i32, i32 addrspace(5)* %gen2local, i64 10
  %bitCast19 = bitcast i32 addrspace(5)* %getElem18 to i8* addrspace(5)*
  store i8* %p, i8* addrspace(5)* %bitCast19, align 8

  %0 = addrspacecast i8 addrspace(4)* getelementptr inbounds ([41 x i8], [41 x i8] addrspace(4)* @"$str", i64 0, i64 0) to i8*
  %1 = bitcast [12 x i32]* %tmp to i8*
  %call = call i32 @vprintf(i8* %0, i8* %1)
  ret void
}

declare i32 @vprintf(i8* nocapture, i8*) nounwind

!nvvm.annotations = !{!0}
!0 = !{void (i8, i16, i32, i64, float, double, i8*)* @foo, !"kernel", i32 1}

!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0}
