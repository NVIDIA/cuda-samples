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
 
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_buffer_reference2 : enable

layout(binding = 0) uniform UniformBufferObject {
    float frame;
} ubo;

layout(location = 0) in float pointInsideCircle;
layout(location = 1) in vec2 xyPos;
 
layout(location = 0) out vec3 fragColor;
 
const float PI = 3.1415926;
 
out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufferFloat {
  float f[];
};

layout(push_constant, std430) uniform Registers
{
  uint64_t base_address;
} registers;

layout(std430, binding = 0) buffer CUDA_SSBO {
  float coords[2];
} cuda_ssbo;

void main() {
    gl_PointSize = 1.0;
    gl_Position = vec4(xyPos.xy, 0.0f, 1.0f);

#if 0
    gl_Position.x += cuda_ssbo.coords[0];
    gl_Position.y += cuda_ssbo.coords[1];
#endif
#if 0
    gl_Position.x += BufferFloat(registers.base_address).f[0];
    gl_Position.y += BufferFloat(registers.base_address).f[1];
#endif

    float color_r = 1.0f + 0.5f * sin(ubo.frame / 100.0f);
    float color_g = 1.0f + 0.5f * sin((ubo.frame / 100.0f) + (2.0f*PI/3.0f));
    float color_b = 1.0f;
    fragColor = vec3(pointInsideCircle.x * color_r, pointInsideCircle.x * color_g, color_b);
}
