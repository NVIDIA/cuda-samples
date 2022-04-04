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

#include <stdio.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable : 4996)  // disable deprecated warning
#endif

#pragma pack(1)

typedef struct {
  short type;
  int size;
  short reserved1;
  short reserved2;
  int offset;
} BMPHeader;

typedef struct {
  int size;
  int width;
  int height;
  short planes;
  short bitsPerPixel;
  unsigned compression;
  unsigned imageSize;
  int xPelsPerMeter;
  int yPelsPerMeter;
  int clrUsed;
  int clrImportant;
} BMPInfoHeader;

// Isolated definition
typedef struct { unsigned char x, y, z, w; } uchar4;

extern "C" void LoadBMPFile(uchar4 **dst, unsigned int *width,
                            unsigned int *height, const char *name) {
  BMPHeader hdr;
  BMPInfoHeader infoHdr;
  int x, y;

  FILE *fd;

  printf("Loading %s...\n", name);

  if (sizeof(uchar4) != 4) {
    printf("***Bad uchar4 size***\n");
    exit(EXIT_SUCCESS);
  }

  if (!(fd = fopen(name, "rb"))) {
    printf("***BMP load error: file access denied***\n");
    exit(EXIT_SUCCESS);
  }

  fread(&hdr, sizeof(hdr), 1, fd);

  if (hdr.type != 0x4D42) {
    printf("***BMP load error: bad file format***\n");
    exit(EXIT_SUCCESS);
  }

  fread(&infoHdr, sizeof(infoHdr), 1, fd);

  if (infoHdr.bitsPerPixel != 24) {
    printf("***BMP load error: invalid color depth***\n");
    exit(EXIT_SUCCESS);
  }

  if (infoHdr.compression) {
    printf("***BMP load error: compressed image***\n");
    exit(EXIT_SUCCESS);
  }

  *width = infoHdr.width;
  *height = infoHdr.height;
  *dst = (uchar4 *)malloc(*width * *height * 4);

  printf("BMP width: %u\n", infoHdr.width);
  printf("BMP height: %u\n", infoHdr.height);

  fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);

  for (y = 0; y < infoHdr.height; y++) {
    for (x = 0; x < infoHdr.width; x++) {
      (*dst)[(y * infoHdr.width + x)].w = 0;
      (*dst)[(y * infoHdr.width + x)].z = fgetc(fd);
      (*dst)[(y * infoHdr.width + x)].y = fgetc(fd);
      (*dst)[(y * infoHdr.width + x)].x = fgetc(fd);
    }

    for (x = 0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++) {
      fgetc(fd);
    }
  }

  if (ferror(fd)) {
    printf("***Unknown BMP load error.***\n");
    free(*dst);
    exit(EXIT_SUCCESS);
  } else {
    printf("BMP file loaded successfully!\n");
  }

  fclose(fd);
}
