/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NV_UTIL_NPP_EXCEPTIONS_H
#define NV_UTIL_NPP_EXCEPTIONS_H


#include <string>
#include <sstream>
#include <iostream>

/// All npp related C++ classes are put into the npp namespace.
namespace npp
{

    /// Exception base class.
    ///     This exception base class will be used for everything C++ throught
    /// the NPP project.
    ///     The exception contains a string message, as well as data fields for a string
    /// containing the name of the file as well as the line number where the exception was thrown.
    ///     The easiest way of throwing exceptions and providing filename and line number is
    /// to use one of the ASSERT macros defined for that purpose.
    class Exception
    {
        public:
            /// Constructor.
            /// \param rMessage A message with information as to why the exception was thrown.
            /// \param rFileName The name of the file where the exception was thrown.
            /// \param nLineNumber Line number in the file where the exception was thrown.
            explicit
            Exception(const std::string &rMessage = "", const std::string &rFileName = "", unsigned int nLineNumber = 0)
                : sMessage_(rMessage), sFileName_(rFileName), nLineNumber_(nLineNumber)
            { };

            Exception(const Exception &rException)
                : sMessage_(rException.sMessage_), sFileName_(rException.sFileName_), nLineNumber_(rException.nLineNumber_)
            { };

            virtual
            ~Exception()
            { };

            /// Get the exception's message.
            const
            std::string &
            message()
            const
            {
                return sMessage_;
            }

            /// Get the exception's file info.
            const
            std::string &
            fileName()
            const
            {
                return sFileName_;
            }

            /// Get the exceptions's line info.
            unsigned int
            lineNumber()
            const
            {
                return nLineNumber_;
            }


            /// Create a clone of this exception.
            ///      This creates a new Exception object on the heap. It is
            /// the responsibility of the user of this function to free this memory
            /// (delete x).
            virtual
            Exception *
            clone()
            const
            {
                return new Exception(*this);
            }

            /// Create a single string with all the exceptions information.
            ///     The virtual toString() method is used by the operator<<()
            /// so that all exceptions derived from this base-class can print
            /// their full information correctly even if a reference to their
            /// exact type is not had at the time of printing (i.e. the basic
            /// operator<<() is used).
            virtual
            std::string
            toString()
            const
            {
                std::ostringstream oOutputString;
                oOutputString << fileName() << ":" << lineNumber() << ": " << message();
                return oOutputString.str();
            }

        private:
            std::string sMessage_;      ///< Message regarding the cause of the exception.
            std::string sFileName_;     ///< Name of the file where the exception was thrown.
            unsigned int nLineNumber_;  ///< Line number in the file where the exception was thrown
    };

    /// Output stream inserter for Exception.
    /// \param rOutputStream The stream the exception information is written to.
    /// \param rException The exception that's being written.
    /// \return Reference to the output stream being used.
    std::ostream &
    operator << (std::ostream &rOutputStream, const Exception &rException)
    {
        rOutputStream << rException.toString();
        return rOutputStream;
    }

    /// Basic assert macro.
    ///     This macro should be used to enforce any kind of pre or post conditions.
    /// Unlike the C-runtime assert macro, this macro does not abort execution, but throws
    /// a C++ exception. The exception is automatically filled with information about the failing
    /// condition, the filename and line number where the exception was thrown.
    /// \note The macro is written in such a way that omitting a semicolon after its usage
    ///     causes a compiler error. The correct way to invoke this macro is:
    /// NPP_ASSERT(n < MAX);
#define NPP_ASSERT(C) do {if (!(C)) throw npp::Exception(#C " assertion faild!", __FILE__, __LINE__);} while(false)

    // ASSERT macro.
    //  Same functionality as the basic assert macro with the added ability to pass
    //  a message M. M should be a string literal.
    //  Note: Never use code inside ASSERT() that causes a side-effect ASSERT macros may get compiled
    //      out in release mode.
#define NPP_ASSERT_MSG(C, M) do {if (!(C)) throw npp::Exception(#C " assertion faild! Message: " M, __FILE__, __LINE__);} while(false)

#ifdef _DEBUG
    /// Basic debug assert macro.
    ///     This macro is identical in every respect to NPP_ASSERT(C) but it does get compiled to a
    /// no-op in release builds. It is therefor of utmost importance to not put statements into
    /// this macro that cause side effects required for correct program execution.
#define NPP_DEBUG_ASSERT(C) do {if (!(C)) throw npp::Exception(#C " debug assertion faild!", __FILE__, __LINE__);} while(false)
#else
#define NPP_DEBUG_ASSERT(C)
#endif

    /// ASSERT for null-pointer test.
    /// It is safe to put code with side effects into this macro. Also: This macro never
    /// gets compiled to a no-op because resource allocation may fail based on external causes not under
    /// control of a software developer.
#define NPP_ASSERT_NOT_NULL(P) do {if ((P) == 0) throw npp::Exception(#P " not null assertion faild!", __FILE__, __LINE__);} while(false)

    /// Macro for flagging methods as not implemented.
    /// The macro throws an exception with a message that an implementation was missing
#define NPP_NOT_IMPLEMENTED() do {throw npp::Exception("Implementation missing!", __FILE__, __LINE__);} while(false)

    /// Macro for checking error return code of CUDA (runtime) calls.
    /// This macro never gets disabled.
#define NPP_CHECK_CUDA(S) do {cudaError_t eCUDAResult; \
        eCUDAResult = S; \
        if (eCUDAResult != cudaSuccess) std::cout << "NPP_CHECK_CUDA - eCUDAResult = " << eCUDAResult << std::endl; \
        NPP_ASSERT(eCUDAResult == cudaSuccess);} while (false)

    /// Macro for checking error return code for NPP calls.
#define NPP_CHECK_NPP(S) do {NppStatus eStatusNPP; \
        eStatusNPP = S; \
        if (eStatusNPP != NPP_SUCCESS) std::cout << "NPP_CHECK_NPP - eStatusNPP = " << _cudaGetErrorEnum(eStatusNPP) << "("<< eStatusNPP << ")" << std::endl; \
        NPP_ASSERT(eStatusNPP == NPP_SUCCESS);} while (false)

    /// Macro for checking error return codes from cuFFT calls.
#define NPP_CHECK_CUFFT(S) do {cufftResult eCUFFTResult; \
        eCUFFTResult = S; \
        if (eCUFFTResult != NPP_SUCCESS) std::cout << "NPP_CHECK_CUFFT - eCUFFTResult = " << eCUFFTResult << std::endl; \
        NPP_ASSERT(eCUFFTResult == CUFFT_SUCCESS);} while (false)

} // npp namespace

#endif // NV_UTIL_NPP_EXCEPTIONS_H
