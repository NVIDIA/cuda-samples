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

/*
 Simple parameter system
 sgreen@nvidia.com 4/2001
*/

#ifndef PARAM_H
#define PARAM_H

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// base class for named parameter
class ParamBase {
 public:
  ParamBase(const char *name) : m_name(name) {}
  virtual ~ParamBase() {}

  std::string &GetName() { return m_name; }

  virtual float GetFloatValue() = 0;
  virtual int GetIntValue() = 0;
  virtual std::string GetValueString() = 0;

  virtual void Reset() = 0;
  virtual void Increment() = 0;
  virtual void Decrement() = 0;

  virtual float GetPercentage() = 0;
  virtual void SetPercentage(float p) = 0;

  virtual void Write(std::ostream &stream) = 0;
  virtual void Read(std::istream &stream) = 0;

  virtual bool IsList() = 0;

 protected:
  std::string m_name;
};

// derived class for single-valued parameter
template <class T>
class Param : public ParamBase {
 public:
  Param(const char *name, T value = 0, T min = 0, T max = 10000, T step = 1,
        T *ptr = 0)
      : ParamBase(name),
        m_default(value),
        m_min(min),
        m_max(max),
        m_step(step),
        m_precision(3) {
    if (ptr) {
      m_ptr = ptr;
    } else {
      m_ptr = &m_value;
    }

    *m_ptr = value;
  }
  ~Param() {}

  T GetValue() const { return *m_ptr; }
  T SetValue(const T value) { *m_ptr = value; }

  float GetFloatValue() { return (float)*m_ptr; }
  int GetIntValue() { return (int)*m_ptr; }

  std::string GetValueString() {
    std::ostringstream ost;
    ost << std::setprecision(m_precision) << std::fixed;
    ost << *m_ptr;
    return ost.str();
  }

  void SetPrecision(int x) { m_precision = x; }

  float GetPercentage() { return (*m_ptr - m_min) / (float)(m_max - m_min); }

  void SetPercentage(float p) { *m_ptr = (T)(m_min + p * (m_max - m_min)); }

  void Reset() { *m_ptr = m_default; }

  void Increment() {
    *m_ptr += m_step;

    if (*m_ptr > m_max) {
      *m_ptr = m_max;
    }
  }

  void Decrement() {
    *m_ptr -= m_step;

    if (*m_ptr < m_min) {
      *m_ptr = m_min;
    }
  }

  void Write(std::ostream &stream) {
    stream << m_name << " " << *m_ptr << '\n';
  }
  void Read(std::istream &stream) { stream >> m_name >> *m_ptr; }

  bool IsList() { return false; }

 private:
  T m_value;
  T *m_ptr;  // pointer to value declared elsewhere
  T m_default, m_min, m_max, m_step;
  int m_precision;  // number of digits after decimal point in string output
};

const Param<int> dummy("error");

// list of parameters
class ParamList : public ParamBase {
 public:
  ParamList(const char *name = "") : ParamBase(name) { active = true; }
  ~ParamList() {}

  float GetFloatValue() { return 0.0f; }
  int GetIntValue() { return 0; }

  void AddParam(ParamBase *param) {
    m_params.push_back(param);
    m_map[param->GetName()] = param;
    m_current = m_params.begin();
  }

  // look-up parameter based on name
  ParamBase *GetParam(char *name) {
    ParamBase *p = m_map[name];

    if (p) {
      return p;
    } else {
      return (ParamBase *)&dummy;
    }
  }

  ParamBase *GetParam(int i) { return m_params[i]; }

  ParamBase *GetCurrent() { return *m_current; }

  int GetSize() { return (int)m_params.size(); }

  std::string GetValueString() { return m_name; }

  // functions to traverse list
  void Reset() { m_current = m_params.begin(); }

  void Increment() {
    m_current++;

    if (m_current == m_params.end()) {
      m_current = m_params.begin();
    }
  }

  void Decrement() {
    if (m_current == m_params.begin()) {
      m_current = m_params.end() - 1;
    } else {
      m_current--;
    }
  }

  float GetPercentage() { return 0.0f; }
  void SetPercentage(float /*p*/) {}

  void Write(std::ostream &stream) {
    stream << m_name << '\n';

    for (std::vector<ParamBase *>::const_iterator p = m_params.begin();
         p != m_params.end(); ++p) {
      (*p)->Write(stream);
    }
  }

  void Read(std::istream &stream) {
    stream >> m_name;

    for (std::vector<ParamBase *>::const_iterator p = m_params.begin();
         p != m_params.end(); ++p) {
      (*p)->Read(stream);
    }
  }

  bool IsList() { return true; }

  void ResetAll() {
    for (std::vector<ParamBase *>::const_iterator p = m_params.begin();
         p != m_params.end(); ++p) {
      (*p)->Reset();
    }
  }

 protected:
  bool active;
  std::vector<ParamBase *> m_params;
  std::map<std::string, ParamBase *> m_map;
  std::vector<ParamBase *>::const_iterator m_current;
};

#endif
