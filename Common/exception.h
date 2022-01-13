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

/* CUda UTility Library */
#ifndef COMMON_EXCEPTION_H_
#define COMMON_EXCEPTION_H_

// includes, system
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

//! Exception wrapper.
//! @param Std_Exception Exception out of namespace std for easy typing.
template <class Std_Exception>
class Exception : public Std_Exception {
 public:
  //! @brief Static construction interface
  //! @return Alwayss throws ( Located_Exception<Exception>)
  //! @param file file in which the Exception occurs
  //! @param line line in which the Exception occurs
  //! @param detailed details on the code fragment causing the Exception
  static void throw_it(const char *file, const int line,
                       const char *detailed = "-");

  //! Static construction interface
  //! @return Alwayss throws ( Located_Exception<Exception>)
  //! @param file file in which the Exception occurs
  //! @param line line in which the Exception occurs
  //! @param detailed details on the code fragment causing the Exception
  static void throw_it(const char *file, const int line,
                       const std::string &detailed);

  //! Destructor
  virtual ~Exception() throw();

 private:
  //! Constructor, default (private)
  Exception();

  //! Constructor, standard
  //! @param str string returned by what()
  explicit Exception(const std::string &str);
};

////////////////////////////////////////////////////////////////////////////////
//! Exception handler function for arbitrary exceptions
//! @param ex exception to handle
////////////////////////////////////////////////////////////////////////////////
template <class Exception_Typ>
inline void handleException(const Exception_Typ &ex) {
  std::cerr << ex.what() << std::endl;

  exit(EXIT_FAILURE);
}

//! Convenience macros

//! Exception caused by dynamic program behavior, e.g. file does not exist
#define RUNTIME_EXCEPTION(msg) \
  Exception<std::runtime_error>::throw_it(__FILE__, __LINE__, msg)

//! Logic exception in program, e.g. an assert failed
#define LOGIC_EXCEPTION(msg) \
  Exception<std::logic_error>::throw_it(__FILE__, __LINE__, msg)

//! Out of range exception
#define RANGE_EXCEPTION(msg) \
  Exception<std::range_error>::throw_it(__FILE__, __LINE__, msg)

////////////////////////////////////////////////////////////////////////////////
//! Implementation

// includes, system
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
//! Static construction interface.
//! @param  Exception causing code fragment (file and line) and detailed infos.
////////////////////////////////////////////////////////////////////////////////
/*static*/ template <class Std_Exception>
void Exception<Std_Exception>::throw_it(const char *file, const int line,
                                        const char *detailed) {
  std::stringstream s;

  // Quiet heavy-weight but exceptions are not for
  // performance / release versions
  s << "Exception in file '" << file << "' in line " << line << "\n"
    << "Detailed description: " << detailed << "\n";

  throw Exception(s.str());
}

////////////////////////////////////////////////////////////////////////////////
//! Static construction interface.
//! @param  Exception causing code fragment (file and line) and detailed infos.
////////////////////////////////////////////////////////////////////////////////
/*static*/ template <class Std_Exception>
void Exception<Std_Exception>::throw_it(const char *file, const int line,
                                        const std::string &msg) {
  throw_it(file, line, msg.c_str());
}

////////////////////////////////////////////////////////////////////////////////
//! Constructor, default (private).
////////////////////////////////////////////////////////////////////////////////
template <class Std_Exception>
Exception<Std_Exception>::Exception() : Std_Exception("Unknown Exception.\n") {}

////////////////////////////////////////////////////////////////////////////////
//! Constructor, standard (private).
//! String returned by what().
////////////////////////////////////////////////////////////////////////////////
template <class Std_Exception>
Exception<Std_Exception>::Exception(const std::string &s) : Std_Exception(s) {}

////////////////////////////////////////////////////////////////////////////////
//! Destructor
////////////////////////////////////////////////////////////////////////////////
template <class Std_Exception>
Exception<Std_Exception>::~Exception() throw() {}

  // functions, exported

#endif  // COMMON_EXCEPTION_H_
