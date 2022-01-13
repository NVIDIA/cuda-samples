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

#ifndef NV_UTIL_NPP_IMAGE_H
#define NV_UTIL_NPP_IMAGE_H

#include <cstddef>

namespace npp
{

    class Image
    {
        public:
            struct Size
            {
                unsigned int nWidth;
                unsigned int nHeight;

                Size() : nWidth(0), nHeight(0)
                { };

                Size(unsigned int nWidthNew, unsigned nHeightNew) : nWidth(nWidthNew), nHeight(nHeightNew)
                { };

                Size(const Size &rSize) : nWidth(rSize.nWidth), nHeight(rSize.nHeight)
                { };

                Size &
                operator= (const Size &rSize)
                {
                    if (&rSize == this)
                    {
                        return *this;
                    }

                    nWidth = rSize.nWidth;
                    nHeight = rSize.nHeight;

                    return *this;
                }

                void
                swap(Size &rSize)
                {
                    unsigned int nTemp;
                    nTemp = nWidth;
                    nWidth = rSize.nWidth;
                    rSize.nWidth = nTemp;

                    nTemp = nHeight;
                    nHeight = rSize.nHeight;
                    rSize.nHeight = nTemp;
                }
            };

            Image()
            { };

            Image(unsigned int nWidth, unsigned int nHeight) : oSize_(nWidth, nHeight)
            { };

            Image(const Image::Size &rSize) : oSize_(rSize)
            { };

            Image(const Image &rImage) : oSize_(rImage.oSize_)
            { };

            virtual
            ~Image()
            { };

            Image &
            operator= (const Image &rImage)
            {
                if (&rImage == this)
                {
                    return *this;
                }

                oSize_  = rImage.oSize_;
                return *this;
            };

            unsigned int
            width()
            const
            {
                return oSize_.nWidth;
            }

            unsigned int
            height()
            const
            {
                return oSize_.nHeight;
            }

            Size
            size()
            const
            {
                return oSize_;
            }

            void
            swap(Image &rImage)
            {
                oSize_.swap(rImage.oSize_);
            }

        private:
            Size oSize_;
    };

    bool
    operator== (const Image::Size &rFirst, const Image::Size &rSecond)
    {
        return rFirst.nWidth == rSecond.nWidth && rFirst.nHeight == rSecond.nHeight;
    }

    bool
    operator!= (const Image::Size &rFirst, const Image::Size &rSecond)
    {
        return rFirst.nWidth != rSecond.nWidth || rFirst.nHeight != rSecond.nHeight;
    }

} // npp namespace


#endif // NV_UTIL_NPP_IMAGE_H
