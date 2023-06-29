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

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; the initial value of xxx is 10
@xxx = internal addrspace(1) global i32 10, align 4

; the initial value of yyy is 100
@yyy = internal addrspace(1) global i32 100, align 4

@llvm.used = appending global [3 x i8*] [i8* bitcast (i8* addrspacecast (i32 addrspace(1)* @xxx to i8*) to i8*), i8* bitcast (i8* addrspacecast (i32 addrspace(1)* @yyy to i8*) to i8*), i8* bitcast (void (i32*)* @test_kernel to i8*)], section "llvm.metadata"

; %ptr can be in the managed space, and its address can be directly used in the host and device.
; See the uvmlite.c, which passes the device pointer of xxx as the kernel parameter.
; This kernel also directly accesses @yyy, which is also managed.
define void @test_kernel(i32* nocapture %ptr) nounwind alwaysinline {
  ; *%ptr = *%ptr + 20
  %gen2other = addrspacecast i32* %ptr to i32 addrspace(1)*
  %tmp1 = load i32, i32 addrspace(1)* %gen2other, align 4
  %add = add nsw i32 %tmp1, 20
  store i32 %add, i32 addrspace(1)* %gen2other, align 4

  ; @yyy = @yyy + 30
  %tmp2 = load i32, i32 addrspace(1)* @yyy, align 4
  %add3 = add nsw i32 %tmp2, 30
  store i32 %add3, i32 addrspace(1)* @yyy, align 4
  ret void
}

!nvvm.annotations = !{!7, !8, !9}
!nvvmir.version = !{!6}

!6 = !{i32 2, i32 0}
!7 = !{i32 addrspace(1)* @xxx, !"managed", i32 1}
!8 = !{i32 addrspace(1)* @yyy, !"managed", i32 1}
!9 = !{void (i32*)* @test_kernel, !"kernel", i32 1}
