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

// This sample needs at least CUDA 10.0.
// It demonstrates usages of the nvJPEG library

#ifndef NV_JPEG_EXAMPLE
#define NV_JPEG_EXAMPLE

#include "cuda_runtime.h"
#include "nvjpeg.h"
#include "helper_cuda.h"
#include "helper_timer.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <string.h>    // strcmpi
#include <sys/time.h>  // timings

#include <dirent.h>  // linux dir traverse
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// write bmp, input - RGB, device
int writeBMP(const char *filename, const unsigned char *d_chanR, int pitchR,
             const unsigned char *d_chanG, int pitchG,
             const unsigned char *d_chanB, int pitchB, int width, int height) {
  unsigned int headers[13];
  FILE *outfile;
  int extrabytes;
  int paddedsize;
  int x;
  int y;
  int n;
  int red, green, blue;

  std::vector<unsigned char> vchanR(height * width);
  std::vector<unsigned char> vchanG(height * width);
  std::vector<unsigned char> vchanB(height * width);
  unsigned char *chanR = vchanR.data();
  unsigned char *chanG = vchanG.data();
  unsigned char *chanB = vchanB.data();
  checkCudaErrors(cudaMemcpy2D(chanR, (size_t)width, d_chanR, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(chanG, (size_t)width, d_chanG, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(chanB, (size_t)width, d_chanB, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));

  extrabytes =
      4 - ((width * 3) % 4);  // How many bytes of padding to add to each
  // horizontal line - the size of which must
  // be a multiple of 4 bytes.
  if (extrabytes == 4) extrabytes = 0;

  paddedsize = ((width * 3) + extrabytes) * height;

  // Headers...
  // Note that the "BM" identifier in bytes 0 and 1 is NOT included in these
  // "headers".

  headers[0] = paddedsize + 54;  // bfSize (whole file size)
  headers[1] = 0;                // bfReserved (both)
  headers[2] = 54;               // bfOffbits
  headers[3] = 40;               // biSize
  headers[4] = width;            // biWidth
  headers[5] = height;           // biHeight

  // Would have biPlanes and biBitCount in position 6, but they're shorts.
  // It's easier to write them out separately (see below) than pretend
  // they're a single int, especially with endian issues...

  headers[7] = 0;           // biCompression
  headers[8] = paddedsize;  // biSizeImage
  headers[9] = 0;           // biXPelsPerMeter
  headers[10] = 0;          // biYPelsPerMeter
  headers[11] = 0;          // biClrUsed
  headers[12] = 0;          // biClrImportant

  if (!(outfile = fopen(filename, "wb"))) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    return 1;
  }

  //
  // Headers begin...
  // When printing ints and shorts, we write out 1 character at a time to avoid
  // endian issues.
  //
  fprintf(outfile, "BM");

  for (n = 0; n <= 5; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  // These next 4 characters are for the biPlanes and biBitCount fields.

  fprintf(outfile, "%c", 1);
  fprintf(outfile, "%c", 0);
  fprintf(outfile, "%c", 24);
  fprintf(outfile, "%c", 0);

  for (n = 7; n <= 12; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  //
  // Headers done, now write the data...
  //

  for (y = height - 1; y >= 0;
       y--)  // BMP image format is written from bottom to top...
  {
    for (x = 0; x <= width - 1; x++) {
      red = chanR[y * width + x];
      green = chanG[y * width + x];
      blue = chanB[y * width + x];

      if (red > 255) red = 255;
      if (red < 0) red = 0;
      if (green > 255) green = 255;
      if (green < 0) green = 0;
      if (blue > 255) blue = 255;
      if (blue < 0) blue = 0;
      // Also, it's written in (b,g,r) format...

      fprintf(outfile, "%c", blue);
      fprintf(outfile, "%c", green);
      fprintf(outfile, "%c", red);
    }
    if (extrabytes)  // See above - BMP lines must be of lengths divisible by 4.
    {
      for (n = 1; n <= extrabytes; n++) {
        fprintf(outfile, "%c", 0);
      }
    }
  }

  fclose(outfile);
  return 0;
}

// write bmp, input - RGB, device
int writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
              int width, int height) {
  unsigned int headers[13];
  FILE *outfile;
  int extrabytes;
  int paddedsize;
  int x;
  int y;
  int n;
  int red, green, blue;

  std::vector<unsigned char> vchanRGB(height * width * 3);
  unsigned char *chanRGB = vchanRGB.data();
  checkCudaErrors(cudaMemcpy2D(chanRGB, (size_t)width * 3, d_RGB, (size_t)pitch,
                               width * 3, height, cudaMemcpyDeviceToHost));

  extrabytes =
      4 - ((width * 3) % 4);  // How many bytes of padding to add to each
  // horizontal line - the size of which must
  // be a multiple of 4 bytes.
  if (extrabytes == 4) extrabytes = 0;

  paddedsize = ((width * 3) + extrabytes) * height;

  // Headers...
  // Note that the "BM" identifier in bytes 0 and 1 is NOT included in these
  // "headers".
  headers[0] = paddedsize + 54;  // bfSize (whole file size)
  headers[1] = 0;                // bfReserved (both)
  headers[2] = 54;               // bfOffbits
  headers[3] = 40;               // biSize
  headers[4] = width;            // biWidth
  headers[5] = height;           // biHeight

  // Would have biPlanes and biBitCount in position 6, but they're shorts.
  // It's easier to write them out separately (see below) than pretend
  // they're a single int, especially with endian issues...

  headers[7] = 0;           // biCompression
  headers[8] = paddedsize;  // biSizeImage
  headers[9] = 0;           // biXPelsPerMeter
  headers[10] = 0;          // biYPelsPerMeter
  headers[11] = 0;          // biClrUsed
  headers[12] = 0;          // biClrImportant

  if (!(outfile = fopen(filename, "wb"))) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    return 1;
  }

  //
  // Headers begin...
  // When printing ints and shorts, we write out 1 character at a time to avoid
  // endian issues.
  //

  fprintf(outfile, "BM");

  for (n = 0; n <= 5; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  // These next 4 characters are for the biPlanes and biBitCount fields.

  fprintf(outfile, "%c", 1);
  fprintf(outfile, "%c", 0);
  fprintf(outfile, "%c", 24);
  fprintf(outfile, "%c", 0);

  for (n = 7; n <= 12; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  //
  // Headers done, now write the data...
  //
  for (y = height - 1; y >= 0;
       y--)  // BMP image format is written from bottom to top...
  {
    for (x = 0; x <= width - 1; x++) {
      red = chanRGB[(y * width + x) * 3];
      green = chanRGB[(y * width + x) * 3 + 1];
      blue = chanRGB[(y * width + x) * 3 + 2];

      if (red > 255) red = 255;
      if (red < 0) red = 0;
      if (green > 255) green = 255;
      if (green < 0) green = 0;
      if (blue > 255) blue = 255;
      if (blue < 0) blue = 0;
      // Also, it's written in (b,g,r) format...

      fprintf(outfile, "%c", blue);
      fprintf(outfile, "%c", green);
      fprintf(outfile, "%c", red);
    }
    if (extrabytes)  // See above - BMP lines must be of lengths divisible by 4.
    {
      for (n = 1; n <= extrabytes; n++) {
        fprintf(outfile, "%c", 0);
      }
    }
  }

  fclose(outfile);
  return 0;
}

int readInput(const std::string &sInputPath,
              std::vector<std::string> &filelist) {
  int error_code = 1;
  struct stat s;

  if (stat(sInputPath.c_str(), &s) == 0) {
    if (s.st_mode & S_IFREG) {
      filelist.push_back(sInputPath);
    } else if (s.st_mode & S_IFDIR) {
      // processing each file in directory
      DIR *dir_handle;
      struct dirent *dir;
      dir_handle = opendir(sInputPath.c_str());
      std::vector<std::string> filenames;
      if (dir_handle) {
        error_code = 0;
        while ((dir = readdir(dir_handle)) != NULL) {
          if (dir->d_type == DT_REG) {
            std::string sFileName = sInputPath + dir->d_name;
            filelist.push_back(sFileName);
          } else if (dir->d_type == DT_DIR) {
            std::string sname = dir->d_name;
            if (sname != "." && sname != "..") {
              readInput(sInputPath + sname + "/", filelist);
            }
          }
        }
        closedir(dir_handle);
      } else {
        std::cout << "Cannot open input directory: " << sInputPath << std::endl;
        return error_code;
      }
    } else {
      std::cout << "Cannot open input: " << sInputPath << std::endl;
      return error_code;
    }
  } else {
    std::cout << "Cannot find input path " << sInputPath << std::endl;
    return error_code;
  }

  return 0;
}

#endif
