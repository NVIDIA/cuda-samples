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

// GLSL shaders for particle rendering
#define STRINGIFY(A) #A

// particle vertex shader
const char *particleVS = STRINGIFY(
  uniform float pointRadius;  // point size in world space         \n
  uniform float pointScale;   // scale to calculate size in pixels \n
  uniform vec4 eyePos;                                             \n
  void main()                                                      \n
  {
    \n
    vec4 wpos = vec4(gl_Vertex.xyz, 1.0);                          \n
    gl_Position = gl_ModelViewProjectionMatrix *wpos;              \n
    
    // calculate window-space point size                           \n
    vec4 eyeSpacePos = gl_ModelViewMatrix *wpos;                   \n
    float dist = length(eyeSpacePos.xyz);                          \n
    gl_PointSize = pointRadius * (pointScale / dist);              \n
    
    gl_TexCoord[0] = gl_MultiTexCoord0; // sprite texcoord         \n
    gl_TexCoord[1] = eyeSpacePos;                                  \n
    
    gl_FrontColor = gl_Color;                                      \n
  }                                                                \n
  );

// motion blur shaders
const char *mblurVS = STRINGIFY(
  uniform float timestep;                                    \n
  void main()                                                \n
  {
    \n
    vec3 pos    = gl_Vertex.xyz;                             \n
    vec3 vel    = gl_MultiTexCoord0.xyz;                     \n
    vec3 pos2   = (pos - vel*timestep).xyz; // previous position \n

    gl_Position    = gl_ModelViewMatrix * vec4(pos, 1.0);  \n // eye space
    gl_TexCoord[0] = gl_ModelViewMatrix * vec4(pos2, 1.0); \n

    // aging                                                 \n
    float lifetime = gl_MultiTexCoord0.w;                    \n
    float age = gl_Vertex.w;                                 \n
    float phase = (lifetime > 0.0) ? (age / lifetime) : 1.0; \n // [0, 1]

    gl_TexCoord[1].x = phase;                                \n
    float fade = 1.0 - phase;                                \n
    //  float fade = 1.0;                                        \n

    //    gl_FrontColor = gl_Color;                              \n
    gl_FrontColor = vec4(gl_Color.xyz, gl_Color.w*fade);     \n
  }                                                          \n
  );

// motion blur geometry shader
// - outputs stretched quad between previous and current positions
const char *mblurGS =
  "#version 120\n"
  "#extension GL_EXT_geometry_shader4 : enable\n"
  STRINGIFY(
  uniform float pointRadius;  // point size in world space       \n
  void main()                                                    \n
  {
    \n
    // aging                                                   \n
    float phase = gl_TexCoordIn[0][1].x;                       \n
    float radius = pointRadius;                                \n

    // eye space                                               \n
    vec3 pos = gl_PositionIn[0].xyz;                           \n
    vec3 pos2 = gl_TexCoordIn[0][0].xyz;                       \n
    vec3 motion = pos - pos2;                                  \n
    vec3 dir = normalize(motion);                              \n
    float len = length(motion);                                \n

    vec3 x = dir *radius;                                     \n
    vec3 view = normalize(-pos);                               \n
    vec3 y = normalize(cross(dir, view)) * radius;             \n
    float facing = dot(view, dir);                             \n

    // check for very small motion to avoid jitter             \n
    float threshold = 0.01;                                    \n

    if ((len < threshold) || (facing > 0.95) || (facing < -0.95))
    {
      \n
      pos2 = pos;
      \n
      x = vec3(radius, 0.0, 0.0);
      \n
      y = vec3(0.0, -radius, 0.0);
      \n
    }                                                          \n

    // output quad                                             \n
    gl_FrontColor = gl_FrontColorIn[0];                        \n
    gl_TexCoord[0] = vec4(0, 0, 0, phase);                     \n
    gl_TexCoord[1] = gl_PositionIn[0];                         \n
    gl_Position = gl_ProjectionMatrix * vec4(pos + x + y, 1);  \n
    EmitVertex();                                              \n

    gl_TexCoord[0] = vec4(0, 1, 0, phase);                     \n
    gl_TexCoord[1] = gl_PositionIn[0];                         \n
    gl_Position = gl_ProjectionMatrix * vec4(pos + x - y, 1);  \n
    EmitVertex();                                              \n

    gl_TexCoord[0] = vec4(1, 0, 0, phase);                     \n
    gl_TexCoord[1] = gl_PositionIn[0];                         \n
    gl_Position = gl_ProjectionMatrix * vec4(pos2 - x + y, 1); \n
    EmitVertex();                                              \n

    gl_TexCoord[0] = vec4(1, 1, 0, phase);                     \n
    gl_TexCoord[1] = gl_PositionIn[0];                         \n
    gl_Position = gl_ProjectionMatrix * vec4(pos2 - x - y, 1); \n
    EmitVertex();                                              \n
  }                                                            \n
  );


const char *simplePS = STRINGIFY(
  void main()                                                    \n
  {
    \n
    gl_FragColor = gl_Color;                                   \n
  }                                                              \n
  );

// render particle without shadows
const char *particlePS = STRINGIFY(
  uniform float pointRadius;                                         \n
  void main()                                                        \n
  {
    \n
    // calculate eye-space sphere normal from texture coordinates  \n
    vec3 N;                                                        \n
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    float r2 = dot(N.xy, N.xy);                                    \n

    if (r2 > 1.0) discard;   // kill pixels outside circle         \n
    N.z = sqrt(1.0-r2);                                            \n

    //  float alpha = saturate(1.0 - r2);                              \n
    float alpha = clamp((1.0 - r2), 0.0, 1.0);                     \n
    alpha *= gl_Color.w;                                           \n

    gl_FragColor = vec4(gl_Color.xyz * alpha, alpha);              \n
  }                                                                  \n
  );

// render particle including shadows
const char *particleShadowPS = STRINGIFY(
  uniform float pointRadius;                                         \n
  uniform sampler2D shadowTex;                                       \n
  uniform sampler2D depthTex;                                        \n
  void main()                                                        \n
  { 
    \n
    // calculate eye-space sphere normal from texture coordinates  \n
    vec3 N;                                                        \n
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    float r2 = dot(N.xy, N.xy);                                    \n

    if (r2 > 1.0) discard;                                         \n // kill pixels outside circle
    N.z = sqrt(1.0-r2);                                            \n
    vec4 eyeSpacePos = gl_TexCoord[1];                             \n
    vec4 eyeSpaceSpherePos = vec4(eyeSpacePos.xyz + N*pointRadius, 1.0); \n // point on sphere
    vec4 shadowPos = gl_TextureMatrix[0] * eyeSpaceSpherePos;      \n
    vec3 shadow = vec3(1.0) - texture2DProj(shadowTex, shadowPos.xyw).xyz;  \n
    //  float alpha = saturate(1.0 - r2);                              \n
    float alpha = clamp((1.0 - r2), 0.0, 1.0);                     \n
    alpha *= gl_Color.w;                                           \n

    gl_FragColor = vec4(gl_Color.xyz *shadow * alpha, alpha);     \n  // premul alpha
  }
  );

// render particle as lit sphere
const char *particleSpherePS = STRINGIFY(
  uniform float pointRadius;                                         \n
  uniform vec3 lightDir = vec3(0.577, 0.577, 0.577);                 \n
  void main()                                                        \n
  {
    \n
    // calculate eye-space sphere normal from texture coordinates  \n
    vec3 N;                                                        \n
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    float r2 = dot(N.xy, N.xy);                                    \n

    if (r2 > 1.0) discard;   // kill pixels outside circle         \n
    N.z = sqrt(1.0-r2);                                            \n

    // calculate depth                                             \n
    vec4 eyeSpacePos = vec4(gl_TexCoord[1].xyz + N*pointRadius, 1.0);   // position of this pixel on sphere in eye space \n
    vec4 clipSpacePos = gl_ProjectionMatrix *eyeSpacePos;         \n
    gl_FragDepth = (clipSpacePos.z / clipSpacePos.w)*0.5+0.5;      \n

    float diffuse = max(0.0, dot(N, lightDir));                    \n

    gl_FragColor = diffuse *gl_Color;                               \n
  }                                                                  \n
  );

const char *passThruVS = STRINGIFY(
  void main()                                                        \n
  {
    \n
    gl_Position = gl_Vertex;                                       \n
    gl_TexCoord[0] = gl_MultiTexCoord0;                            \n
    gl_FrontColor = gl_Color;                                      \n
  }                                                                  \n
  );

const char *texture2DPS = STRINGIFY(
  uniform sampler2D tex;                                             \n
  void main()                                                        \n
  {
    \n
    gl_FragColor = texture2D(tex, gl_TexCoord[0].xy);              \n
  }                                                                  \n
  );

// 4 tap 3x3 gaussian blur
const char *blurPS = STRINGIFY(
  uniform sampler2D tex;                                    \n
  uniform vec2 texelSize;                                   \n
  uniform float blurRadius;                                 \n
  void main()                                               \n
  {
    \n
    vec4 c;                                                                        \n
    c = texture2D(tex, gl_TexCoord[0].xy + vec2(-0.5, -0.5)*texelSize*blurRadius); \n
    c += texture2D(tex, gl_TexCoord[0].xy + vec2(0.5, -0.5)*texelSize*blurRadius); \n
    c += texture2D(tex, gl_TexCoord[0].xy + vec2(0.5, 0.5)*texelSize*blurRadius);  \n
    c += texture2D(tex, gl_TexCoord[0].xy + vec2(-0.5, 0.5)*texelSize*blurRadius); \n
    c *= 0.25;                                                                     \n

    gl_FragColor = c;                                \n
  }                                                  \n
  );

// floor shader
const char *floorVS = STRINGIFY(
  varying vec4 vertexPosEye;  // vertex position in eye space  \n
  varying vec3 normalEye;                                      \n
  void main()                                                  \n
  {
    \n
    gl_Position = gl_ModelViewProjectionMatrix *gl_Vertex;  \n
    gl_TexCoord[0] = gl_MultiTexCoord0;                      \n
    vertexPosEye = gl_ModelViewMatrix *gl_Vertex;           \n
    normalEye = gl_NormalMatrix *gl_Normal;                 \n
    gl_FrontColor = gl_Color;                                \n
  }                                                            \n
  );

const char *floorPS = STRINGIFY(
  uniform vec3 lightPosEye; // light position in eye space           \n
  uniform vec3 lightColor;                                           \n
  uniform sampler2D tex;                                             \n
  uniform sampler2D shadowTex;                                       \n
  varying vec4 vertexPosEye;  // vertex position in eye space        \n
  varying vec3 normalEye;                                            \n
  void main()                                                        \n
  {
    \n
    vec4 shadowPos = gl_TextureMatrix[0] * vertexPosEye;                    \n
    vec4 colorMap  = texture2D(tex, gl_TexCoord[0].xy);                     \n

    vec3 N = normalize(normalEye);                                          \n
    vec3 L = normalize(lightPosEye - vertexPosEye.xyz);                     \n
    float diffuse = max(0.0, dot(N, L));                                    \n

    vec3 shadow = vec3(1.0) - texture2DProj(shadowTex, shadowPos.xyw).xyz;  \n

    if (shadowPos.w < 0.0) shadow = lightColor;   \n // avoid back projections
    gl_FragColor = vec4(gl_Color.xyz *colorMap.xyz *diffuse * shadow, 1.0); \n
  }                                                                         \n
);
