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

; This NVVM IR program shows how to use unsize external shared memory.
; What it does is similar to the following CUDA C code.
;
; extern __shared__ int a[];
;
; __global__ void foo()
; {
;    int *p1 = a;
;    float *p2 = (float *)&a[10];
;
;   *p1 = 10;
;   *p2 = 21.0f;
; }
;

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@a = external addrspace(3) global [0 x i32]

define void @foo() {
entry:
  store i32 10, i32 addrspace(3)* getelementptr inbounds ([0 x i32], [0 x i32] addrspace(3)* @a, i64 0, i64 0), align 4
  store float 2.100000e+01, float addrspace(3)* bitcast (i32 addrspace(3)* getelementptr inbounds ([0 x i32], [0 x i32] addrspace(3)* @a, i64 0, i64 10) to float addrspace(3)*), align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{void ()* @foo, !"kernel", i32 1}

!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0}
