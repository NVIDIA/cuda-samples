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

#include "common.h"
#include "flowGold.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief host texture fetch
///
/// read from arbitrary position within image using bilinear interpolation
/// out of range coords are mirrored
/// \param[in]  t   texture raw data
/// \param[in]  w   texture width
/// \param[in]  h   texture height
/// \param[in]  s   texture stride
/// \param[in]  x   x coord of the point to fetch value at
/// \param[in]  y   y coord of the point to fetch value at
/// \return fetched value
///////////////////////////////////////////////////////////////////////////////
inline float Tex2D(const float *t, int w, int h, int s, float x, float y) {
  // integer parts in floating point format
  float intPartX, intPartY;

  // get fractional parts of coordinates
  float dx = fabsf(modff(x, &intPartX));
  float dy = fabsf(modff(y, &intPartY));

  // assume pixels are squares
  // one of the corners
  int ix0 = (int)intPartX;
  int iy0 = (int)intPartY;

  // mirror out-of-range position
  if (ix0 < 0) ix0 = abs(ix0 + 1);

  if (iy0 < 0) iy0 = abs(iy0 + 1);

  if (ix0 >= w) ix0 = w * 2 - ix0 - 1;

  if (iy0 >= h) iy0 = h * 2 - iy0 - 1;

  // corner which is opposite to (ix0, iy0)
  int ix1 = ix0 + 1;
  int iy1 = iy0 + 1;

  if (ix1 >= w) ix1 = w * 2 - ix1 - 1;

  if (iy1 >= h) iy1 = h * 2 - iy1 - 1;

  float res = t[ix0 + iy0 * s] * (1.0f - dx) * (1.0f - dy);
  res += t[ix1 + iy0 * s] * dx * (1.0f - dy);
  res += t[ix0 + iy1 * s] * (1.0f - dx) * dy;
  res += t[ix1 + iy1 * s] * dx * dy;

  return res;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief host texture fetch
///
/// read specific texel value
/// out of range coords are mirrored
/// \param[in]  t   texture raw data
/// \param[in]  w   texture width
/// \param[in]  h   texture height
/// \param[in]  s   texture stride
/// \param[in]  x   x coord of the point to fetch value at
/// \param[in]  y   y coord of the point to fetch value at
/// \return fetched value
///////////////////////////////////////////////////////////////////////////////
inline float Tex2Di(const float *src, int w, int h, int s, int x, int y) {
  if (x < 0) x = abs(x + 1);

  if (y < 0) y = abs(y + 1);

  if (x >= w) x = w * 2 - x - 1;

  if (y >= h) y = h * 2 - y - 1;

  return src[x + y * s];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief resize image
/// \param[in]  src         image to downscale
/// \param[in]  width       image width
/// \param[in]  height      image height
/// \param[in]  stride      image stride
/// \param[in]  newWidth    image new width
/// \param[in]  newHeight   image new height
/// \param[in]  newStride   image new stride
/// \param[out] out         downscaled image data
///////////////////////////////////////////////////////////////////////////////
static void Downscale(const float *src, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, float *out) {
  for (int i = 0; i < newHeight; ++i) {
    for (int j = 0; j < newWidth; ++j) {
      const int srcX = j * 2;
      const int srcY = i * 2;
      // average 4 neighbouring pixels
      float sum;
      sum = Tex2Di(src, width, height, stride, srcX + 0, srcY + 0);
      sum += Tex2Di(src, width, height, stride, srcX + 0, srcY + 1);
      sum += Tex2Di(src, width, height, stride, srcX + 1, srcY + 0);
      sum += Tex2Di(src, width, height, stride, srcX + 1, srcY + 1);
      // normalize
      sum *= 0.25f;
      out[j + i * newStride] = sum;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  stride      field current stride
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  newStride   field new stride
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static void Upscale(const float *src, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride, float scale,
                    float *out) {
  for (int i = 0; i < newHeight; ++i) {
    for (int j = 0; j < newWidth; ++j) {
      // position within smaller image
      float x = ((float)j - 0.5f) * 0.5f;
      float y = ((float)i - 0.5f) * 0.5f;

      out[j + i * newStride] = Tex2D(src, width, height, stride, x, y) * scale;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with provided vector field
///
/// For each output pixel there is a vector which tells which pixel
/// from a source image should be mapped to this particular output
/// pixel.
/// It is assumed that images and the vector field have the same stride and
/// resolution.
/// \param[in]  src source image
/// \param[in]  w   width
/// \param[in]  h   height
/// \param[in]  s   stride
/// \param[in]  u   horizontal displacement
/// \param[in]  v   vertical displacement
/// \param[out] out warped image
///////////////////////////////////////////////////////////////////////////////
static void WarpImage(const float *src, int w, int h, int s, const float *u,
                      const float *v, float *out) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int pos = j + i * s;
      // warped coords
      float x = (float)j + u[pos];
      float y = (float)i + v[pos];

      out[pos] = Tex2D(src, w, h, s, x, y);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief computes image derivatives for a pair of images
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   images width
/// \param[in]  h   images height
/// \param[in]  s   images stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////
static void ComputeDerivatives(const float *I0, const float *I1, int w, int h,
                               int s, float *Ix, float *Iy, float *Iz) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int pos = j + i * s;
      float t0, t1;
      // derivative filter is (1, -8, 0, 8, -1)/12
      // x derivative
      t0 = Tex2Di(I0, w, h, s, j - 2, i);
      t0 -= Tex2Di(I0, w, h, s, j - 1, i) * 8.0f;
      t0 += Tex2Di(I0, w, h, s, j + 1, i) * 8.0f;
      t0 -= Tex2Di(I0, w, h, s, j + 2, i);
      t0 /= 12.0f;

      t1 = Tex2Di(I1, w, h, s, j - 2, i);
      t1 -= Tex2Di(I1, w, h, s, j - 1, i) * 8.0f;
      t1 += Tex2Di(I1, w, h, s, j + 1, i) * 8.0f;
      t1 -= Tex2Di(I1, w, h, s, j + 2, i);
      t1 /= 12.0f;

      // spatial derivatives are averaged
      Ix[pos] = (t0 + t1) * 0.5f;

      // t derivative
      Iz[pos] = I1[pos] - I0[pos];

      // y derivative
      t0 = Tex2Di(I0, w, h, s, j, i - 2);
      t0 -= Tex2Di(I0, w, h, s, j, i - 1) * 8.0f;
      t0 += Tex2Di(I0, w, h, s, j, i + 1) * 8.0f;
      t0 -= Tex2Di(I0, w, h, s, j, i + 2);
      t0 /= 12.0f;

      t1 = Tex2Di(I1, w, h, s, j, i - 2);
      t1 -= Tex2Di(I1, w, h, s, j, i - 1) * 8.0f;
      t1 += Tex2Di(I1, w, h, s, j, i + 1) * 8.0f;
      t1 -= Tex2Di(I1, w, h, s, j, i + 2);
      t1 /= 12.0f;

      Iy[pos] = (t0 + t1) * 0.5f;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief one iteration of classical Horn-Schunck method
///
/// It is one iteration of Jacobi method for a corresponding linear system
/// \param[in]  du0     current horizontal displacement approximation
/// \param[in]  dv0     current vertical displacement approximation
/// \param[in]  Ix      image x derivative
/// \param[in]  Iy      image y derivative
/// \param[in]  Iz      temporal derivative
/// \param[in]  w       width
/// \param[in]  h       height
/// \param[in]  s       stride
/// \param[in]  alpha   degree of smoothness
/// \param[out] du1     new horizontal displacement approximation
/// \param[out] dv1     new vertical displacement approximation
///////////////////////////////////////////////////////////////////////////////
static void SolveForUpdate(const float *du0, const float *dv0, const float *Ix,
                           const float *Iy, const float *Iz, int w, int h,
                           int s, float alpha, float *du1, float *dv1) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int pos = j + i * s;
      int left, right, up, down;

      // handle borders
      if (j != 0)
        left = pos - 1;
      else
        left = pos;

      if (j != w - 1)
        right = pos + 1;
      else
        right = pos;

      if (i != 0)
        down = pos - s;
      else
        down = pos;

      if (i != h - 1)
        up = pos + s;
      else
        up = pos;

      float sumU = (du0[left] + du0[right] + du0[up] + du0[down]) * 0.25f;
      float sumV = (dv0[left] + dv0[right] + dv0[up] + dv0[down]) * 0.25f;

      float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos]) /
                   (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

      du1[pos] = sumU - Ix[pos] * frac;
      dv1[pos] = sumV - Iy[pos] * frac;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocation and control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  nLevels      number of levels in a pyramid
/// \param[in]  nWarpIters   number of warping iterations per pyramid level
/// \param[in]  nSolverIters number of solver iterations (Jacobi iterations)
/// \param[out] u            horizontal displacement
/// \param[out] v            vertical displacement
///////////////////////////////////////////////////////////////////////////////
void ComputeFlowGold(const float *I0, const float *I1, int width, int height,
                     int stride, float alpha, int nLevels, int nWarpIters,
                     int nSolverIters, float *u, float *v) {
  printf("Computing optical flow on CPU...\n");

  float *u0 = u;
  float *v0 = v;

  const float **pI0 = new const float *[nLevels];
  const float **pI1 = new const float *[nLevels];

  int *pW = new int[nLevels];
  int *pH = new int[nLevels];
  int *pS = new int[nLevels];

  const int pixelCountAligned = height * stride;

  float *tmp = new float[pixelCountAligned];
  float *du0 = new float[pixelCountAligned];
  float *dv0 = new float[pixelCountAligned];
  float *du1 = new float[pixelCountAligned];
  float *dv1 = new float[pixelCountAligned];
  float *Ix = new float[pixelCountAligned];
  float *Iy = new float[pixelCountAligned];
  float *Iz = new float[pixelCountAligned];
  float *nu = new float[pixelCountAligned];
  float *nv = new float[pixelCountAligned];

  // prepare pyramid
  int currentLevel = nLevels - 1;
  pI0[currentLevel] = I0;
  pI1[currentLevel] = I1;

  pW[currentLevel] = width;
  pH[currentLevel] = height;
  pS[currentLevel] = stride;

  for (; currentLevel > 0; --currentLevel) {
    int nw = pW[currentLevel] / 2;
    int nh = pH[currentLevel] / 2;
    int ns = iAlignUp(nw);
    pI0[currentLevel - 1] = new float[ns * nh];
    pI1[currentLevel - 1] = new float[ns * nh];

    Downscale(pI0[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI0[currentLevel - 1]);

    Downscale(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI1[currentLevel - 1]);

    pW[currentLevel - 1] = nw;
    pH[currentLevel - 1] = nh;
    pS[currentLevel - 1] = ns;
  }

  // initial approximation
  memset(u, 0, stride * height * sizeof(float));
  memset(v, 0, stride * height * sizeof(float));

  // compute flow
  for (; currentLevel < nLevels; ++currentLevel) {
    for (int warpIter = 0; warpIter < nWarpIters; ++warpIter) {
      memset(du0, 0, pixelCountAligned * sizeof(float));
      memset(dv0, 0, pixelCountAligned * sizeof(float));

      memset(du1, 0, pixelCountAligned * sizeof(float));
      memset(dv1, 0, pixelCountAligned * sizeof(float));

      WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                pS[currentLevel], u, v, tmp);

      // on current level we compute optical flow
      // between frame 0 and warped frame 1
      ComputeDerivatives(pI0[currentLevel], tmp, pW[currentLevel],
                         pH[currentLevel], pS[currentLevel], Ix, Iy, Iz);

      for (int iter = 0; iter < nSolverIters; ++iter) {
        SolveForUpdate(du0, dv0, Ix, Iy, Iz, pW[currentLevel], pH[currentLevel],
                       pS[currentLevel], alpha, du1, dv1);
        Swap(du0, du1);
        Swap(dv0, dv1);
      }

      // update u, v
      for (int i = 0; i < pH[currentLevel] * pS[currentLevel]; ++i) {
        u[i] += du0[i];
        v[i] += dv0[i];
      }
    }  // end for (int warpIter = 0; warpIter < nWarpIters; ++warpIter)

    if (currentLevel != nLevels - 1) {
      // prolongate solution
      float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];

      Upscale(u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleX, nu);

      float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];

      Upscale(v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleY, nv);

      Swap(u, nu);
      Swap(v, nv);
    }
  }  // end for (; currentLevel < nLevels; ++currentLevel)

  if (u != u0) {
    // solution is not in the specified array
    // copy
    memcpy(u0, u, pixelCountAligned * sizeof(float));
    memcpy(v0, v, pixelCountAligned * sizeof(float));
    Swap(u, nu);
    Swap(v, nv);
  }

  // cleanup
  // last level is not being freed here
  // because it refers to input images
  for (int i = 0; i < nLevels - 1; ++i) {
    delete[] pI0[i];
    delete[] pI1[i];
  }

  delete[] pI0;
  delete[] pI1;
  delete[] pW;
  delete[] pH;
  delete[] pS;
  delete[] tmp;
  delete[] du0;
  delete[] dv0;
  delete[] du1;
  delete[] dv1;
  delete[] Ix;
  delete[] Iy;
  delete[] Iz;
  delete[] nu;
  delete[] nv;
}
