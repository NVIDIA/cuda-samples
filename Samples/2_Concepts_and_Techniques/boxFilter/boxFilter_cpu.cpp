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

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" void computeGold(float *id, float *od, int w, int h, int r);

// CPU implementation
void hboxfilter_x(float *id, float *od, int w, int h, int r) {
  float scale = 1.0f / (2 * r + 1);

  for (int y = 0; y < h; y++) {
    float t;
    // do left edge
    t = id[y * w] * r;

    for (int x = 0; x < r + 1; x++) {
      t += id[y * w + x];
    }

    od[y * w] = t * scale;

    for (int x = 1; x < r + 1; x++) {
      int c = y * w + x;
      t += id[c + r];
      t -= id[y * w];
      od[c] = t * scale;
    }

    // main loop
    for (int x = r + 1; x < w - r; x++) {
      int c = y * w + x;
      t += id[c + r];
      t -= id[c - r - 1];
      od[c] = t * scale;
    }

    // do right edge
    for (int x = w - r; x < w; x++) {
      int c = y * w + x;
      t += id[(y * w) + w - 1];
      t -= id[c - r - 1];
      od[c] = t * scale;
    }
  }
}

void hboxfilter_y(float *id, float *od, int w, int h, int r) {
  float scale = 1.0f / (2 * r + 1);

  for (int x = 0; x < w; x++) {
    float t;
    // do left edge
    t = id[x] * r;

    for (int y = 0; y < r + 1; y++) {
      t += id[y * w + x];
    }

    od[x] = t * scale;

    for (int y = 1; y < r + 1; y++) {
      int c = y * w + x;
      t += id[c + r * w];
      t -= id[x];
      od[c] = t * scale;
    }

    // main loop
    for (int y = r + 1; y < h - r; y++) {
      int c = y * w + x;
      t += id[c + r * w];
      t -= id[c - (r * w) - w];
      od[c] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++) {
      int c = y * w + x;
      t += id[(h - 1) * w + x];
      t -= id[c - (r * w) - w];
      od[c] = t * scale;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! @param image      pointer to input data
//! @param temp       pointer to temporary store
//! @param w          width of image
//! @param h          height of image
//! @param r          radius of filter
////////////////////////////////////////////////////////////////////////////////

void computeGold(float *image, float *temp, int w, int h, int r) {
  hboxfilter_x(image, temp, w, h, r);
  hboxfilter_y(temp, image, w, h, r);
}
