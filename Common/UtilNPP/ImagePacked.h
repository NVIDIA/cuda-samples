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

#ifndef NV_UTIL_NPP_IMAGE_PACKED_H
#define NV_UTIL_NPP_IMAGE_PACKED_H

#include "Image.h"
#include "Pixel.h"

namespace npp
{
    template<typename D, size_t N, class A>
    class ImagePacked: public npp::Image
    {
        public:
            typedef npp::Pixel<D, N>    tPixel;
            typedef D                   tData;
            static const size_t         gnChannels = N;
            typedef npp::Image::Size    tSize;

            ImagePacked(): aPixels_(0)
                , nPitch_(0)
            {
                ;
            }

            ImagePacked(unsigned int nWidth, unsigned int nHeight): Image(nWidth, nHeight)
                , aPixels_(0)
                , nPitch_(0)
            {
                aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
            }

            ImagePacked(unsigned int nWidth, unsigned int nHeight, bool bTight): Image(nWidth, nHeight)
                , aPixels_(0)
                , nPitch_(0)
            {
                aPixels_ = A::Malloc2D(width(), height(), &nPitch_, bTight);
            }

            ImagePacked(const tSize &rSize): Image(rSize)
                , aPixels_(0)
                , nPitch_(0)
            {
                aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
            }

            ImagePacked(const ImagePacked<D, N, A> &rImage): Image(rImage)
                , aPixels_(0)
                , nPitch_(rImage.pitch())
            {
                aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
                A::Copy2D(aPixels_, nPitch_, rImage.pixels(), rImage.pitch(), width(), height());
            }

            virtual
            ~ImagePacked()
            {
                A::Free2D(aPixels_);
            }

            ImagePacked &
            operator= (const ImagePacked<D, N, A> &rImage)
            {
                // in case of self-assignment
                if (&rImage == this)
                {
                    return *this;
                }

                A::Free2D(aPixels_);
                aPixels_ = 0;
                nPitch_ = 0;

                // assign parent class's data fields (width, height)
                Image::operator =(rImage);

                aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
                A::Copy2D(aPixels_, nPitch_, rImage.data(), rImage.pitch(), width(), height());

                return *this;
            }

            unsigned int
            pitch()
            const
            {
                return nPitch_;
            }

            /// Get a pointer to the pixel array.
            ///     The result pointer can be offset to pixel at position (x, y) and
            /// even negative offsets are allowed.
            /// \param nX Horizontal pointer/array offset.
            /// \param nY Vertical pointer/array offset.
            /// \return Pointer to the pixel array (or first pixel in array with coordinates (nX, nY).
            tPixel *
            pixels(int nX = 0, int nY = 0)
            {
                return reinterpret_cast<tPixel *>(reinterpret_cast<unsigned char *>(aPixels_) + nY * pitch() + nX * gnChannels * sizeof(D));
            }

            const
            tPixel *
            pixels(int nX = 0, int nY = 0)
            const
            {
                return reinterpret_cast<const tPixel *>(reinterpret_cast<unsigned char *>(aPixels_) + nY * pitch() + nX * gnChannels * sizeof(D));
            }

            D *
            data(int nX = 0, int nY = 0)
            {
                return reinterpret_cast<D *>(pixels(nX, nY));
            }

            const
            D *
            data(int nX = 0, int nY = 0)
            const
            {
                return reinterpret_cast<const D *>(pixels(nX, nY));
            }

            void
            swap(ImagePacked<D, N, A> &rImage)
            {
                Image::swap(rImage);

                tData *aTemp   = aPixels_;
                aPixels_        = rImage.aPixels_;
                rImage.aPixels_ = aTemp;

                unsigned int nTemp = nPitch_;
                nPitch_            = rImage.nPitch_;
                rImage.nPitch_     = nTemp;
            }

        private:
            D *aPixels_;
            unsigned int nPitch_;
    };

} // npp namespace


#endif // NV_IMAGE_IPP_H
