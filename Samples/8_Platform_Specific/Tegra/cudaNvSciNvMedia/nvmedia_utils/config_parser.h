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

#ifndef _NVMEDIA_TEST_CONFIG_PARSER_H_
#define _NVMEDIA_TEST_CONFIG_PARSER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "nvmedia_core.h"
#include "nvmedia_surface.h"

#define MAX_ITEMS_TO_PARSE 10000

typedef enum _ParamType {
    TYPE_UINT = 0,
    TYPE_UINT_HEX,
    TYPE_INT,
    TYPE_DOUBLE,
    TYPE_FLOAT,
    TYPE_UCHAR,
    TYPE_ULLONG,
    TYPE_USHORT,
    TYPE_CHAR_ARR,
    TYPE_UCHAR_ARR,
    TYPE_SHORT
} ParamType;

typedef enum {
    LIMITS_NONE = 0,
    LIMITS_MIN = 1,
    LIMITS_BOTH = 2
} LimitsType;

typedef enum {
    SECTION_NONE,
    SECTION_CAPTURE,
    SECTION_QP,
    SECTION_RC,
    SECTION_ENCODE_PIC,
    SECTION_ENCODE_PIC_H264,
    SECTION_ENCODE_PIC_H265,
    SECTION_MVC,
    SECTION_PAYLOAD,
    SECTION_2DPROCESSOR
} SectionType;

typedef struct {
    SectionType         secType;
    const char         *name;
    unsigned int        lastSectionIndex;
    size_t              sizeOfStruct;
} SectionMap;

typedef struct {
    const char         *paramName;
    void               *mappedLocation;
    ParamType           type;
    double              defaultValue;
    LimitsType          paramLimits;
    double              minLimit;
    double              maxLimit;
    unsigned int        stringLength;     // string param size
    unsigned int       *stringLengthAddr; // address of string param size
    SectionType         sectionType;
} ConfigParamsMap;

NvMediaStatus   ConfigParser_InitParamsMap(ConfigParamsMap *paramsMap);
NvMediaStatus   ConfigParser_ParseFile(ConfigParamsMap *paramsMap, unsigned int numParams, SectionMap *sectionsMap, char *file);
NvMediaStatus   ConfigParser_ValidateParams(ConfigParamsMap *paramsMap, SectionMap *sectionsMap);
NvMediaStatus   ConfigParser_DisplayParams(ConfigParamsMap *paramsMap, SectionMap *sectionsMap);
NvMediaStatus   ConfigParser_GetSectionIndexByName(SectionMap *sectionsMap, char *sectionName, unsigned int *index);
NvMediaStatus   ConfigParser_GetSectionIndexByType(SectionMap *sectionsMap, SectionType sectionType, unsigned int *index);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_CONFIG_PARSER_H_ */
