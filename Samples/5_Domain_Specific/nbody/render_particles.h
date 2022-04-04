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

#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

class ParticleRenderer {
 public:
  ParticleRenderer();
  ~ParticleRenderer();

  void setPositions(float *pos, int numParticles);
  void setPositions(double *pos, int numParticles);
  void setBaseColor(float color[4]);
  void setColors(float *color, int numParticles);
  void setPBO(unsigned int pbo, int numParticles, bool fp64);

  enum DisplayMode {
    PARTICLE_POINTS,
    PARTICLE_SPRITES,
    PARTICLE_SPRITES_COLOR,
    PARTICLE_NUM_MODES
  };

  void display(DisplayMode mode = PARTICLE_POINTS);

  void setPointSize(float size) { m_pointSize = size; }
  void setSpriteSize(float size) { m_spriteSize = size; }

  void resetPBO();

 protected:  // methods
  void _initGL();
  void _createTexture(int resolution);
  void _drawPoints(bool color = false);

 protected:  // data
  float *m_pos;
  double *m_pos_fp64;
  int m_numParticles;

  float m_pointSize;
  float m_spriteSize;

  unsigned int m_vertexShader;
  unsigned int m_vertexShaderPoints;
  unsigned int m_pixelShader;
  unsigned int m_programPoints;
  unsigned int m_programSprites;
  unsigned int m_texture;
  unsigned int m_pbo;
  unsigned int m_vboColor;

  float m_baseColor[4];

  bool m_bFp64Positions;
};

#endif  //__ RENDER_PARTICLES__
