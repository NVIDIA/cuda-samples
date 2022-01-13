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


#ifndef NV_UTIL_NPP_SIGNALS_CPU_H
#define NV_UTIL_NPP_SIGNALS_CPU_H

#include "Signal.h"

#include "SignalAllocatorsCPU.h"
#include "Exceptions.h"

#include <npp.h>


namespace npp
{

    template<typename D, class A>
    class SignalCPU: public npp::SignalTemplate<D, A>
    {
        public:
            typedef typename npp::SignalTemplate<D, A>::tData tData;

            SignalCPU()
            {
                ;
            }

            SignalCPU(size_t nSize): SignalTemplate<D, A>(nSize)
            {
                ;
            }

            SignalCPU(const SignalCPU<D, A> &rSignal): SignalTemplate<D, A>(rSignal)
            {
                ;
            }

            virtual
            ~SignalCPU()
            {
                ;
            }

            SignalCPU &
            operator= (const SignalCPU<D,A> &rSignal)
            {
                SignalTemplate<D, A>::operator= (rSignal);

                return *this;
            }

            tData &
            operator [](unsigned int i)
            {
                return *SignalTemplate<D, A>::values(i);
            }

            tData
            operator [](unsigned int i)
            const
            {
                return *SignalTemplate<D, A>::values(i);
            }

    };

    typedef SignalCPU<Npp8u,   npp::SignalAllocatorCPU<Npp8u>   >   SignalCPU_8u;
    typedef SignalCPU<Npp32s,  npp::SignalAllocatorCPU<Npp32s>  >   SignalCPU_32s;
    typedef SignalCPU<Npp16s,  npp::SignalAllocatorCPU<Npp16s>  >   SignalCPU_16s;
    typedef SignalCPU<Npp16sc, npp::SignalAllocatorCPU<Npp16sc> >   SignalCPU_16sc;
    typedef SignalCPU<Npp32sc, npp::SignalAllocatorCPU<Npp32sc> >   SignalCPU_32sc;
    typedef SignalCPU<Npp32f,  npp::SignalAllocatorCPU<Npp32f>  >   SignalCPU_32f;
    typedef SignalCPU<Npp32fc, npp::SignalAllocatorCPU<Npp32fc> >   SignalCPU_32fc;
    typedef SignalCPU<Npp64s,  npp::SignalAllocatorCPU<Npp64s>  >   SignalCPU_64s;
    typedef SignalCPU<Npp64sc, npp::SignalAllocatorCPU<Npp64sc> >   SignalCPU_64sc;
    typedef SignalCPU<Npp64f,  npp::SignalAllocatorCPU<Npp64f>  >   SignalCPU_64f;
    typedef SignalCPU<Npp64fc, npp::SignalAllocatorCPU<Npp64fc> >   SignalCPU_64fc;

} // npp namespace

#endif // NV_UTIL_NPP_SIGNALS_CPU_H
