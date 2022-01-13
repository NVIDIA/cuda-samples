/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 Copyright (c) 2005,
  Aaron Lefohn (lefohn@cs.ucdavis.edu)
  Adam Moerschell (atmoerschell@ucdavis.edu)
 All rights reserved.

 This software is licensed under the BSD open-source license. See
 http://www.opensource.org/licenses/bsd-license.php for more detail.

 *************************************************************
 Redistribution and use in source and binary forms, with or
 without modification, are permitted provided that the following
 conditions are met:

 Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 Neither the name of the University of California, Davis nor the names of
 the contributors may be used to endorse or promote products derived
 from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 OF SUCH DAMAGE.
*/

#ifndef UCDAVIS_RENDER_BUFFER_H
#define UCDAVIS_RENDER_BUFFER_H

#include "framebufferObject.h"

/*!
Renderbuffer Class. This class encapsulates the Renderbuffer OpenGL
object described in the FramebufferObject (FBO) OpenGL spec.
See the official spec at:
    http://oss.sgi.com/projects/ogl-sample/registry/EXT/framebuffer_object.txt
for complete details.

A "Renderbuffer" is a chunk of GPU memory used by FramebufferObjects to
represent "traditional" framebuffer memory (depth, stencil, and color buffers).
By "traditional," we mean that the memory cannot be bound as a texture.
With respect to GPU shaders, Renderbuffer memory is "write-only." Framebuffer
operations such as alpha blending, depth test, alpha test, stencil test, etc.
read from this memory in post-fragment-shader (ROP) operations.

The most common use of Renderbuffers is to create depth and stencil buffers.
Note that as of 7/1/05, NVIDIA drivers to do not support stencil Renderbuffers.

Usage Notes:
  1) "internalFormat" can be any of the following:
      Valid OpenGL internal formats beginning with:
        RGB, RGBA, DEPTH_COMPONENT

      or a stencil buffer format (not currently supported
      in NVIDIA drivers as of 7/1/05).
        STENCIL_INDEX1_EXT
        STENCIL_INDEX4_EXT
        STENCIL_INDEX8_EXT
        STENCIL_INDEX16_EXT
*/
class Renderbuffer
{
    public:
        /// Ctors/Dtors
        Renderbuffer();
        Renderbuffer(GLenum internalFormat, int width, int height);
        ~Renderbuffer();

        void   Bind();
        void   Unbind();
        void   Set(GLenum internalFormat, int width, int height);
        GLuint GetId() const;

        static GLint GetMaxSize();

    private:
        GLuint m_bufId;
        static GLuint _CreateBufferId();
};

#endif

