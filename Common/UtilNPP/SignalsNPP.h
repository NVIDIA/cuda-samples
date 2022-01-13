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


#ifndef NV_UTIL_NPP_SIGNALS_NPP_H
#define NV_UTIL_NPP_SIGNALS_NPP_H

#include "Exceptions.h"
#include "Signal.h"

#include "SignalAllocatorsNPP.h"
#include <cuda_runtime.h>

namespace npp
{
    // forward declaration
    template<typename D, class A> class SignalCPU;

    template<typename D>
    class SignalNPP: public npp::SignalTemplate<D, npp::SignalAllocator<D> >
    {
        public:
            SignalNPP()
            {
                ;
            }

            explicit
            SignalNPP(size_t nSize): SignalTemplate<D, npp::SignalAllocator<D> >(nSize)
            {
                ;
            }

            SignalNPP(const SignalNPP<D> &rSignal): SignalTemplate<D, npp::SignalAllocator<D> >(rSignal)
            {
                ;
            }

            template<class X>
            explicit
            SignalNPP(const SignalCPU<D, X> &rSignal): SignalTemplate<D, npp::SignalAllocator<D> >(rSignal.size())
            {
                npp::SignalAllocator<D>::HostToDeviceCopy1D(SignalTemplate<D, npp::SignalAllocator<D> >::values(),
                                                            rSignal.values(), SignalTemplate<D, npp::SignalAllocator<D> >::size());
            }

            virtual
            ~SignalNPP()
            {
                ;
            }

            SignalNPP &
            operator= (const SignalNPP<D> &rSignal)
            {
                SignalTemplate<D, npp::SignalAllocator<D> >::operator= (rSignal);

                return *this;
            }

            void
            copyTo(D *pValues)
            const
            {
                npp::SignalAllocator<D>::DeviceToHostCopy1D(pValues, SignalTemplate<D, npp::SignalAllocator<D> >::values(), SignalTemplate<D, npp::SignalAllocator<D> >::size());
            }

            void
            copyFrom(D *pValues)
            {
                npp::SignalAllocator<D>::HostToDeviceCopy1D(SignalTemplate<D, npp::SignalAllocator<D> >::values(), pValues, SignalTemplate<D, npp::SignalAllocator<D> >::size());
            }
    };

    typedef SignalNPP<Npp8u>    SignalNPP_8u;
    typedef SignalNPP<Npp16s>   SignalNPP_16s;
    typedef SignalNPP<Npp16sc>  SignalNPP_16sc;
    typedef SignalNPP<Npp32s>   SignalNPP_32s;
    typedef SignalNPP<Npp32sc>  SignalNPP_32sc;
    typedef SignalNPP<Npp32f>   SignalNPP_32f;
    typedef SignalNPP<Npp32fc>  SignalNPP_32fc;
    typedef SignalNPP<Npp64s>   SignalNPP_64s;
    typedef SignalNPP<Npp64sc>  SignalNPP_64sc;
    typedef SignalNPP<Npp64f>   SignalNPP_64f;
    typedef SignalNPP<Npp64fc>  SignalNPP_64fc;

} // npp namespace

#endif // NV_UTIL_NPP_SIGNALS_NPP_H
