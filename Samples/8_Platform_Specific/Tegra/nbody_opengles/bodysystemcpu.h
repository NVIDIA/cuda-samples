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

#ifndef __BODYSYSTEMCPU_H__
#define __BODYSYSTEMCPU_H__

#include "bodysystem.h"

// CPU Body System
template <typename T>
class BodySystemCPU : public BodySystem<T> {
 public:
  BodySystemCPU(int numBodies);
  virtual ~BodySystemCPU();

  virtual void loadTipsyFile(const std::string &filename);

  virtual void update(T deltaTime);

  virtual void setSoftening(T softening) {
    m_softeningSquared = softening * softening;
  }
  virtual void setDamping(T damping) { m_damping = damping; }

  virtual T *getArray(BodyArray array);
  virtual void setArray(BodyArray array, const T *data);

  virtual unsigned int getCurrentReadBuffer() const { return 0; }

  virtual unsigned int getNumBodies() const { return m_numBodies; }

 protected:           // methods
  BodySystemCPU() {}  // default constructor

  virtual void _initialize(int numBodies);
  virtual void _finalize();

  void _computeNBodyGravitation();
  void _integrateNBodySystem(T deltaTime);

 protected:  // data
  int m_numBodies;
  bool m_bInitialized;

  T *m_pos;
  T *m_vel;
  T *m_force;

  T m_softeningSquared;
  T m_damping;
};

#include "bodysystemcpu_impl.h"

#endif  // __BODYSYSTEMCPU_H__
