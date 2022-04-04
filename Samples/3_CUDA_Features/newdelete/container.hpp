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

////////////////////////////////////////////////////////////////////////////////
//
// Container parent class.
//
////////////////////////////////////////////////////////////////////////////////

template <class T>
class Container {
 public:
  __device__ Container() { ; }

  __device__ virtual ~Container() { ; }

  __device__ virtual void push(T e) = 0;

  __device__ virtual bool pop(T& e) = 0;
};

////////////////////////////////////////////////////////////////////////////////
//
// Vector class derived from Container class using linear memory as data storage
// NOTE: This education purpose implementation has restricted functionality.
//       For example, concurrent push and pop operations will not work
//       correctly.
//
////////////////////////////////////////////////////////////////////////////////

template <class T>
class Vector : public Container<T> {
 public:
  // Constructor, data is allocated on the heap
  // NOTE: This must be called from only one thread
  __device__ Vector(int max_size) : m_top(-1) { m_data = new T[max_size]; }

  // Constructor, data uses preallocated buffer via placement new
  __device__ Vector(int max_size, T* preallocated_buffer) : m_top(-1) {
    m_data = new (preallocated_buffer) T[max_size];
  }

  // Destructor, data is freed
  // NOTE: This must be called from only one thread
  __device__ ~Vector() {
    if (m_data) delete[] m_data;
  }

  __device__ virtual void push(T e) {
    if (m_data) {
      // Atomically increment the top idx
      int idx = atomicAdd(&(this->m_top), 1);
      m_data[idx + 1] = e;
    }
  }

  __device__ virtual bool pop(T& e) {
    if (m_data && m_top >= 0) {
      // Atomically decrement the top idx
      int idx = atomicAdd(&(this->m_top), -1);
      if (idx >= 0) {
        e = m_data[idx];
        return true;
      }
    }
    return false;
  }

 private:
  int m_size;
  T* m_data;

  int m_top;
};
