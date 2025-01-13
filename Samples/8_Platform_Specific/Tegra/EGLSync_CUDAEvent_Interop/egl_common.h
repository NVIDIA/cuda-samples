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

//
// DESCRIPTION:   Common EGL functions header file
//

#ifndef _EGL_COMMON_H_
#define _EGL_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <signal.h>
#include "cuda.h"
#include "cudaEGL.h"

EGLImageKHR eglImage;

#define EXTENSION_LIST(T)                                \
    T(PFNEGLCREATEIMAGEKHRPROC, eglCreateImageKHR)       \
    T(PFNEGLDESTROYIMAGEKHRPROC, eglDestroyImageKHR)     \
    T(PFNEGLCREATESYNCKHRPROC, eglCreateSyncKHR)         \
    T(PFNEGLDESTROYSYNCKHRPROC, eglDestroySyncKHR)       \
    T(PFNEGLCLIENTWAITSYNCKHRPROC, eglClientWaitSyncKHR) \
    T(PFNEGLGETSYNCATTRIBKHRPROC, eglGetSyncAttribKHR)   \
    T(PFNEGLCREATESYNC64KHRPROC, eglCreateSync64KHR)     \
    T(PFNEGLWAITSYNCKHRPROC, eglWaitSyncKHR)

#define eglCreateImageKHR my_eglCreateImageKHR
#define eglDestroyImageKHR my_eglDestroyImageKHR
#define eglCreateSyncKHR my_eglCreateSyncKHR
#define eglDestroySyncKHR my_eglDestroySyncKHR
#define eglClientWaitSyncKHR my_eglClientWaitSyncKHR
#define eglGetSyncAttribKHR my_eglGetSyncAttribKHR
#define eglCreateSync64KHR my_eglCreateSync64KHR
#define eglWaitSyncKHR my_eglWaitSyncKHR

#define EXTLST_DECL(tx, x) tx my_##x = NULL;
#define EXTLST_EXTERN(tx, x) extern tx my_##x;
#define EXTLST_ENTRY(tx, x) {(extlst_fnptr_t *)&my_##x, #x},

int eglSetupExtensions(void);
#endif
