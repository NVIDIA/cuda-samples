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


#ifndef NV_UTIL_NPP_SIGNAL_H
#define NV_UTIL_NPP_SIGNAL_H

#include <cstring>

namespace npp
{
    class Signal
    {
        public:
            Signal() : nSize_(0)
            { };

            explicit
            Signal(size_t nSize) : nSize_(nSize)
            { };

            Signal(const Signal &rSignal) : nSize_(rSignal.nSize_)
            { };

            virtual
            ~Signal()
            { }

            Signal &
            operator= (const Signal &rSignal)
            {
                nSize_ = rSignal.nSize_;
                return *this;
            }

            size_t
            size()
            const
            {
                return nSize_;
            }

            void
            swap(Signal &rSignal)
            {
                size_t nTemp = nSize_;
                nSize_ = rSignal.nSize_;
                rSignal.nSize_ = nTemp;
            }


        private:
            size_t nSize_;
    };

    template<typename D, class A>
    class SignalTemplate: public Signal
    {
        public:
            typedef D tData;

            SignalTemplate(): aValues_(0)
            {
                ;
            }

            SignalTemplate(size_t nSize): Signal(nSize)
                , aValues_(0)
            {
                aValues_ = A::Malloc1D(size());
            }

            SignalTemplate(const SignalTemplate<D, A> &rSignal): Signal(rSignal)
                , aValues_(0)
            {
                aValues_ = A::Malloc1D(size());
                A::Copy1D(aValues_, rSignal.values(), size());
            }

            virtual
            ~SignalTemplate()
            {
                A::Free1D(aValues_);
            }

            SignalTemplate &
            operator= (const SignalTemplate<D, A> &rSignal)
            {
                // in case of self-assignment
                if (&rSignal == this)
                {
                    return *this;
                }

                A::Free1D(aValues_);
                this->aPixels_ = 0;

                // assign parent class's data fields (width, height)
                Signal::operator =(rSignal);

                aValues_ = A::Malloc1D(size());
                A::Copy1D(aValues_, rSignal.value(), size());

                return *this;
            }

            /// Get a pointer to the pixel array.
            ///     The result pointer can be offset to pixel at position (x, y) and
            /// even negative offsets are allowed.
            /// \param nX Horizontal pointer/array offset.
            /// \param nY Vertical pointer/array offset.
            /// \return Pointer to the pixel array (or first pixel in array with coordinates (nX, nY).
            tData *
            values(int i = 0)
            {
                return aValues_ + i;
            }

            const
            tData *
            values(int i = 0)
            const
            {
                return aValues_ + i;
            }

            void
            swap(SignalTemplate<D, A> &rSignal)
            {
                Signal::swap(rSignal);

                tData *aTemp       = this->aValues_;
                this->aValues_      = rSignal.aValues_;
                rSignal.aValues_    = aTemp;
            }

        private:
            D *aValues_;
    };

} // npp namespace


#endif // NV_UTIL_NPP_SIGNAL_H
