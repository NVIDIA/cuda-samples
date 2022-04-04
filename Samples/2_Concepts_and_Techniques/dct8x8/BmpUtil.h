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

/**
**************************************************************************
* \file BmpUtil.h
* \brief Contains basic image operations declaration.
*
* This file contains declaration of basic bitmap loading, saving,
* conversions to different representations and memory management routines.
*/

#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma pack(push)
#endif

#pragma pack(1)

typedef char int8;
typedef short int16;
typedef int int32;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

/**
* \brief Bitmap file header structure
*
*  Bitmap file header structure
*/
typedef struct {
  uint16 _bm_signature;    //!< File signature, must be "BM"
  uint32 _bm_file_size;    //!< File size
  uint32 _bm_reserved;     //!< Reserved, must be zero
  uint32 _bm_bitmap_data;  //!< Bitmap data
} BMPFileHeader;

/**
* \brief Bitmap info header structure
*
*  Bitmap info header structure
*/
typedef struct {
  uint32 _bm_info_header_size;      //!< Info header size, must be 40
  uint32 _bm_image_width;           //!< Image width
  uint32 _bm_image_height;          //!< Image height
  uint16 _bm_num_of_planes;         //!< Amount of image planes, must be 1
  uint16 _bm_color_depth;           //!< Color depth
  uint32 _bm_compressed;            //!< Image compression, must be none
  uint32 _bm_bitmap_size;           //!< Size of bitmap data
  uint32 _bm_hor_resolution;        //!< Horizontal resolution, assumed to be 0
  uint32 _bm_ver_resolution;        //!< Vertical resolution, assumed to be 0
  uint32 _bm_num_colors_used;       //!< Number of colors used, assumed to be 0
  uint32 _bm_num_important_colors;  //!< Number of important colors, assumed to
                                    //!be 0
} BMPInfoHeader;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma pack(pop)
#else
#pragma pack()
#endif

/**
* \brief Simple 2D size / region_of_interest structure
*
*  Simple 2D size / region_of_interest structure
*/
typedef struct {
  int width;   //!< ROI width
  int height;  //!< ROI height
} ROI;

/**
*  One-byte unsigned integer type
*/
typedef unsigned char byte;

extern "C" {
int clamp_0_255(int x);
float round_f(float num);
byte *MallocPlaneByte(int width, int height, int *pStepBytes);
short *MallocPlaneShort(int width, int height, int *pStepBytes);
float *MallocPlaneFloat(int width, int height, int *pStepBytes);
void CopyByte2Float(byte *ImgSrc, int StrideB, float *ImgDst, int StrideF,
                    ROI Size);
void CopyFloat2Byte(float *ImgSrc, int StrideF, byte *ImgDst, int StrideB,
                    ROI Size);
void FreePlane(void *ptr);
void AddFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size);
void MulFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size);
int PreLoadBmp(char *FileName, int *Width, int *Height);
void LoadBmpAsGray(char *FileName, int Stride, ROI ImSize, byte *Img);
void DumpBmpAsGray(char *FileName, byte *Img, int Stride, ROI ImSize);
void DumpBlockF(float *PlaneF, int StrideF, char *Fname);
void DumpBlock(byte *Plane, int Stride, char *Fname);
float CalculateMSE(byte *Img1, byte *Img2, int Stride, ROI Size);
float CalculatePSNR(byte *Img1, byte *Img2, int Stride, ROI Size);
}
