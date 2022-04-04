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

#ifndef __BODYSYSTEMCUDA_H__
#define __BODYSYSTEMCUDA_H__

#include "bodysystem.h"

template <typename T>
struct DeviceData {
  T *dPos[2];  // mapped host pointers
  T *dVel;
  cudaEvent_t event;
  unsigned int offset;
  unsigned int numBodies;
};

// CUDA BodySystem: runs on the GPU
template <typename T>
class BodySystemCUDA : public BodySystem<T> {
 public:
  BodySystemCUDA(unsigned int numBodies, unsigned int numDevices,
                 unsigned int blockSize, bool usePBO, bool useSysMem = false);
  virtual ~BodySystemCUDA();

  virtual void loadTipsyFile(const std::string &filename);

  virtual void update(T deltaTime);

  virtual void setSoftening(T softening);
  virtual void setDamping(T damping);

  virtual T *getArray(BodyArray array);
  virtual void setArray(BodyArray array, const T *data);

  virtual unsigned int getCurrentReadBuffer() const {
    return m_pbo[m_currentRead];
  }

  virtual unsigned int getNumBodies() const { return m_numBodies; }

 protected:  // methods
  BodySystemCUDA() {}

  virtual void _initialize(int numBodies);
  virtual void _finalize();

 protected:  // data
  unsigned int m_numBodies;
  unsigned int m_numDevices;
  bool m_bInitialized;

  // Host data
  T *m_hPos[2];
  T *m_hVel;

  DeviceData<T> *m_deviceData;

  bool m_bUsePBO;
  bool m_bUseSysMem;
  unsigned int m_SMVersion;

  T m_damping;

  unsigned int m_pbo[2];
  cudaGraphicsResource *m_pGRes[2];
  unsigned int m_currentRead;
  unsigned int m_currentWrite;

  unsigned int m_blockSize;
};

#include "bodysystemcuda_impl.h"

#endif  // __BODYSYSTEMCUDA_H__
