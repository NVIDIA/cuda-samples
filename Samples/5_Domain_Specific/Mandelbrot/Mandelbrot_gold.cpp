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

#include "Mandelbrot_gold.h"

#define ABS(n) ((n) < 0 ? -(n) : (n))

/* dfloat class declaration */
class dfloat {
 private:
  float val[2];

 public:
  dfloat() { val[0] = val[1] = 0; }
  dfloat(float a, float b) {
    val[0] = a;
    val[1] = b;
  }
  dfloat(double b);
  inline float operator[](unsigned idx) const { return val[idx]; }
};
inline dfloat operator+(const dfloat &dsa, const dfloat &dsb);
inline dfloat operator-(const dfloat &dsa, const dfloat &dsb);
inline dfloat operator*(const dfloat &dsa, const dfloat &dsb);
inline int operator<(const dfloat &a, float b) { return a[0] < b; }

// The core Mandelbrot calculation function template
template <class T>
inline int CalcMandelbrot(const T xPos, const T yPos, const T xJParam,
                          const T yJParam, const int crunch,
                          const bool isJulia) {
  T x, y, xx, yy, xC, yC;
  int i = crunch;

  if (isJulia) {
    xC = xJParam;
    yC = yJParam;
    y = yPos;
    x = xPos;
    yy = y * y;
    xx = x * x;
  } else {
    xC = xPos;
    yC = yPos;
    x = y = 0;
    xx = yy = 0;
  }

  while (--i && (xx + yy < 4.0f)) {
    y = x * y + x * y + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;
  }

  return i;
}  // CalcMandelbrot

inline void updatePixel(uchar4 &dst, const uchar4 &color, int frame) {
  int frame1 = frame + 1;
  int frame2 = frame1 / 2;
  dst.x = (dst.x * frame + color.x + frame2) / frame1;
  dst.y = (dst.y * frame + color.y + frame2) / frame1;
  dst.z = (dst.z * frame + color.z + frame2) / frame1;
}

inline void setColor(uchar4 &dst, const uchar4 &colors, int &m,
                     const int animationFrame) {
  if (m == 0) {
    dst.x = 0;
    dst.y = 0;
    dst.z = 0;
    return;
  }

  m += animationFrame;
  dst.x = m * colors.x;
  dst.y = m * colors.y;
  dst.z = m * colors.z;
}

template <class T, class T_>
void runMandelbrotGold0(uchar4 *dst, const int imageW, const int imageH,
                        const int crunch, const T xOff, const T yOff,
                        const T xJParam, const T yJParam, const T scale,
                        const uchar4 colors, const int frame,
                        const int animationFrame, const bool isJulia) {
  for (int iy = 0; iy < imageH; iy++)
    for (int ix = 0; ix < imageW; ix++) {
      // Calculate the location
      const T_ xPos = (T)ix * scale + xOff;
      const T_ yPos = (T)iy * scale + yOff;

      // Calculate the Mandelbrot index for the current location
      int m = CalcMandelbrot<T_>(xPos, yPos, xJParam, yJParam, crunch, isJulia);
      m = m > 0 ? crunch - m : 0;

      // Convert the Mandelbrot index into a color
      uchar4 color;

      setColor(color, colors, m, animationFrame);

      // Output the pixel
      int pixel = imageW * iy + ix;

      if (frame == 0) {
        color.w = 0;
        dst[pixel] = color;
      } else
        updatePixel(dst[pixel], color, frame);
    }

}  // runMandelbrotGold0_

// Determine if two pixel colors are within tolerance
inline int CheckColors(const uchar4 &color0, const uchar4 &color1) {
  int x = color1.x - color0.x;
  int y = color1.y - color0.y;
  int z = color1.z - color0.z;
  return (ABS(x) > 10) || (ABS(y) > 10) || (ABS(z) > 10);
}  // CheckColors

template <class T, class T_>
void runMandelbrotGold1(uchar4 *dst, const int imageW, const int imageH,
                        const int crunch, const T xOff, const T yOff,
                        const T xJParam, const T yJParam, const T scale,
                        const uchar4 colors, const int frame,
                        const int animationFrame, const bool isJulia) {
  for (int iy = 0; iy < imageH; iy++)
    for (int ix = 0; ix < imageW; ix++) {
      // Get the current pixel color
      int pixel = imageW * iy + ix;
      uchar4 pixelColor = dst[pixel];
      int count = 0;

      // Search for pixels out of tolerance surrounding the current pixel
      if (ix > 0) count += CheckColors(pixelColor, dst[pixel - 1]);

      if (ix + 1 < imageW) count += CheckColors(pixelColor, dst[pixel + 1]);

      if (iy > 0) count += CheckColors(pixelColor, dst[pixel - imageW]);

      if (iy + 1 < imageH)
        count += CheckColors(pixelColor, dst[pixel + imageW]);

      if (count) {
        // Calculate the location
        const T_ xPos = (T)ix * scale + xOff;
        const T_ yPos = (T)iy * scale + yOff;

        // Calculate the Mandelbrot index for the current location
        int m =
            CalcMandelbrot<T_>(xPos, yPos, xJParam, yJParam, crunch, isJulia);
        m = m > 0 ? crunch - m : 0;

        // Convert the Mandelbrot index into a color
        uchar4 color;

        setColor(color, colors, m, animationFrame);

        // Output the pixel
        updatePixel(dst[pixel], color, frame);
      }
    }
}  // RunMandelbrotGold1_

/* Implementation of exported functions */
void RunMandelbrotGold1(uchar4 *dst, const int imageW, const int imageH,
                        const int crunch, const float xOff, const float yOff,
                        const float xJParam, const float yJParam,
                        const float scale, const uchar4 colors, const int frame,
                        const int animationFrame, const bool isJulia) {
  runMandelbrotGold1<float, float>(dst, imageW, imageH, crunch, xOff, yOff,
                                   xJParam, yJParam, scale, colors, frame,
                                   animationFrame, isJulia);

}  // RunMandelbrotGold1

void RunMandelbrotDSGold1(uchar4 *dst, const int imageW, const int imageH,
                          const int crunch, const double xOff,
                          const double yOff, const double xJParam,
                          const double yJParam, const double scale,
                          const uchar4 colors, const int frame,
                          const int animationFrame, const bool isJulia) {
  runMandelbrotGold1<double, dfloat>(dst, imageW, imageH, crunch, xOff, yOff,
                                     xJParam, yJParam, scale, colors, frame,
                                     animationFrame, isJulia);

}  // RunMandelbrotDSGold1

void RunMandelbrotGold0(uchar4 *dst, const int imageW, const int imageH,
                        const int crunch, const float xOff, const float yOff,
                        const float xJParam, const float yJParam,
                        const float scale, const uchar4 colors, const int frame,
                        const int animationFrame, const bool isJulia) {
  runMandelbrotGold0<float, float>(dst, imageW, imageH, crunch, xOff, yOff,
                                   xJParam, yJParam, scale, colors, frame,
                                   animationFrame, isJulia);
}  // RunMandelbrotGold0

void RunMandelbrotDSGold0(uchar4 *dst, const int imageW, const int imageH,
                          const int crunch, const double xOff,
                          const double yOff, const double xJParam,
                          const double yJParam, const double scale,
                          const uchar4 colors, const int frame,
                          const int animationFrame, const bool isJulia) {
  runMandelbrotGold0<double, dfloat>(dst, imageW, imageH, crunch, xOff, yOff,
                                     xJParam, yJParam, scale, colors, frame,
                                     animationFrame, isJulia);
}  // RunMandelbrotDSGold0

/*dfloat operations implementation */

/* Construct a DS number equal to a double precision floating point number b*/
dfloat::dfloat(double b) {
  val[0] = (float)b;
  val[1] = (float)(b - val[0]);
}

inline dfloat operator+(const dfloat &dsa, const dfloat &dsb) {
  // Compute dsa + dsb using Knuth's trick.
  float t1 = dsa[0] + dsb[0];
  float e = t1 - dsa[0];
  float t2 = ((dsb[0] - e) + (dsa[0] - (t1 - e))) + dsa[1] + dsb[1];

  // The result is t1 + t2, after normalization.
  e = t1 + t2;
  return dfloat(e, t2 - (e - t1));
}

inline dfloat operator-(const dfloat &dsa, const dfloat &dsb) {
  // Compute dsa - dsb using Knuth's trick.
  float t1 = dsa[0] - dsb[0];
  float e = t1 - dsa[0];
  float t2 = ((-dsb[0] - e) + (dsa[0] - (t1 - e))) + dsa[1] - dsb[1];

  // The result is t1 + t2, after normalization.
  e = t1 + t2;
  return dfloat(e, t2 - (e - t1));
}

inline dfloat operator*(const dfloat &dsa, const dfloat &dsb) {
  // This splits dsa(1) and dsb(1) into high-order and low-order words.
  float c11 = dsa[0] * dsb[0];
  float c21 = dsa[0] * dsb[0] - c11;

  // Compute dsa[0] * dsb[1] + dsa[1] * dsb[0] (only high-order word is needed).
  float c2 = dsa[0] * dsb[1] + dsa[1] * dsb[0];

  // Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.
  float t1 = c11 + c2;
  float e = t1 - c11;
  float t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + dsa[1] * dsb[1];

  // The result is t1 + t2, after normalization.
  e = t1 + t2;
  return dfloat(e, t2 - (e - t1));
}
