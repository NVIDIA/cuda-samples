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

#include <stdlib.h>

#include "config_parser.h"
#include "log_utils.h"
#if defined(__QNX__)
#include <strings.h>
#endif

static NvMediaStatus GetParamIndex(ConfigParamsMap *paramsMap, char *paramName, unsigned int *index)
{
    int i = 0;

    while(paramsMap[i].paramName != NULL) {
        if (strcasecmp(paramsMap[i].paramName, paramName) == 0) {
            *index = i;
            return NVMEDIA_STATUS_OK;
        } else {
            i++;
        }
    }

    return NVMEDIA_STATUS_BAD_PARAMETER;
}

NvMediaStatus ConfigParser_GetSectionIndexByName(SectionMap *sectionsMap, char *sectionName, unsigned int *index)
{
    unsigned int i = 0;

    while(sectionsMap[i].secType != SECTION_NONE) {
        if(strcmp(sectionsMap[i].name, sectionName) == 0) {
            *index = i;
            return NVMEDIA_STATUS_OK;
        } else {
            i++;
        }
    }

    return NVMEDIA_STATUS_BAD_PARAMETER;
}

NvMediaStatus ConfigParser_GetSectionIndexByType(SectionMap *sectionsMap, SectionType sectionType, unsigned int *index)
{
    unsigned int i = 0;

    while(sectionsMap[i].secType != SECTION_NONE) {
        if(sectionsMap[i].secType == sectionType) {
            *index = i;
            return NVMEDIA_STATUS_OK;
        } else {
            i++;
        }
    }

    *index = i;
    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus GetFileContent(char *filename, char **fileContentOut)
{
    FILE *file;
    char *fileCotent;
    long fileSize;

    file = fopen(filename, "r");
    if(file == NULL) {
        printf("Parser_GetFileContent: Cannot open configuration file %s\n", filename);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        printf("Parser_GetFileContent: Cannot fseek in configuration file %s\n", filename);
        return NVMEDIA_STATUS_ERROR;
    }

    fileSize = ftell(file);
    if(fileSize < 0 || fileSize > 150000) {
        printf("Parser_GetFileContent: Unreasonable Filesize %ld encountered for file %s\n", fileSize, filename);
        return NVMEDIA_STATUS_ERROR;
    }

    if(fseek (file, 0, SEEK_SET) != 0) {
        printf("Parser_GetFileContent: Cannot fseek in configuration file %s\n", filename);
        return NVMEDIA_STATUS_ERROR;
    }

    fileCotent = (char*)malloc(fileSize + 1);
    if(fileCotent == NULL) {
        printf("Parser_GetFileContent: Failed allocating buffer for file Content\n");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    fileSize = (long)fread(fileCotent, 1, fileSize, file);
    fileCotent[fileSize] = '\0';
    *fileContentOut = fileCotent;

    fclose(file);

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus ConfigParser_ParseFile(ConfigParamsMap *paramsMap, unsigned int numParams, SectionMap *sectionsMap, char *fileName)
{
    char *items[MAX_ITEMS_TO_PARSE] = {NULL};
    int intValue, itemsCount = 0, i = 0, sectionIndex = 0;
    double doubleValue;
    float floatValue;
    unsigned int currItemIndex, uintValue, sectionId = 0, currSectionId = 0, charValue, paramDefaultLength;
    unsigned short ushortValue;
    short shortValue;
    unsigned long long ullValue;
    NvMediaBool isInString = NVMEDIA_FALSE, isInItem = NVMEDIA_FALSE;
    char *buffer, *bufferEnd, *param, *pParamLength;
    char sectionName[100];
    char currDigit;
    char *configContentBuf = NULL;
    unsigned int numSetsInSection = 0;

    if(GetFileContent(fileName, &configContentBuf) != NVMEDIA_STATUS_OK) {
        printf("ConfigParser_ParseFile: Failed reading file %s", fileName);
        return NVMEDIA_STATUS_ERROR;
    }

    buffer = configContentBuf;
    bufferEnd = &configContentBuf[strlen(configContentBuf)];

    // Stage one: Create items mapping in the content using "items" pointers array. For each parameter we have 3 items: param name, '=' char and the param value.
    while(buffer < bufferEnd) {
        if(itemsCount >= MAX_ITEMS_TO_PARSE) {
            LOG_WARN("ConfigParser_ParseFile: Number of items in configuration file exceeded the maximum allowed (%d). Only %d items will be parsed.\n",
                        MAX_ITEMS_TO_PARSE, MAX_ITEMS_TO_PARSE);
            itemsCount = MAX_ITEMS_TO_PARSE;
            break;
        }
        switch(*buffer) {
            // Carriage return
            case 13:
                ++buffer;
                break;
            case '#':
                *buffer = '\0';                           // Replace '#' with '\0' in case of comment immediately following integer or string
                while(*buffer != '\n' && buffer < bufferEnd) { // Skip till EOL or EOF
                    ++buffer;
                }
                isInString = NVMEDIA_FALSE;
                isInItem = NVMEDIA_FALSE;
                break;
            case '\n':
                isInItem = NVMEDIA_FALSE;
                isInString = NVMEDIA_FALSE;
                *buffer++='\0';
                break;
            case ' ':
            case '\t':                          // Skip whitespace, leave state unchanged
                if(isInString)
                    buffer++;
                else {                          // Terminate non-strings once whitespace is found
                    *buffer++ = '\0';
                    isInItem = NVMEDIA_FALSE;
                }
                break;
            case '"':                           // Begin/End of String
                *buffer++ = '\0';
                if(!isInString) {
                    items[itemsCount++] = buffer;
                    isInItem = ~isInItem;
                } else {
                    isInItem = NVMEDIA_FALSE;
                }
                isInString = ~isInString;           // Toggle
                break;
            case '[':
                *(buffer++) = '\0';
                items[itemsCount++] = buffer;
                while(*buffer != ' ' && *buffer != '\n' && buffer < bufferEnd) { // Skip till whitespace (after which is located the parsed section number) or EOL or EOF
                    sectionName[i++] = *(buffer++);
                }
                sectionName[i] = '\0';
                i = 0;
                while(*buffer == ' ') {
                    *(buffer++) = '\0';
                }
                items[itemsCount++] = buffer;
                while(*buffer != ']' && *buffer != '\n' && buffer < bufferEnd) { // Read the section number
                    currDigit = *buffer;
                    sectionIndex = sectionIndex * 10 + (currDigit - '0');
                    buffer++;
                }
                *(buffer++) = '\0';
                sectionIndex--;
                if(ConfigParser_GetSectionIndexByName(sectionsMap, sectionName, &sectionId) != NVMEDIA_STATUS_OK) {
                    printf("ConfigParser_ParseFile: SectionName couldn't be found in section map: '%s'.\n", sectionName);
                }
                numSetsInSection++;
                sectionsMap[sectionId].lastSectionIndex = sectionIndex;
                sectionIndex = 0;
                isInString = NVMEDIA_FALSE;
                isInItem = NVMEDIA_FALSE;
                break;
            default:
                if(!isInItem) {
                    items[itemsCount++] = buffer;
                    isInItem = ~isInItem;
                }
                buffer++;
        }
    }

    itemsCount--;

    if(numSetsInSection > numParams) {
        printf("%s: Not enough buffers allocated for parsing. Number of sets allocated: %d. Number of sets in config file: %d \n",
                __func__, numParams, numSetsInSection);
        if(configContentBuf) {
            free(configContentBuf);
        }
        return NVMEDIA_STATUS_ERROR;
    }

    // Stage 2: Go through the list of items and save their values in parameters map
    for(i = 0; i < itemsCount; i += 3) {
        if(ConfigParser_GetSectionIndexByName(sectionsMap, items[i], &currItemIndex) == NVMEDIA_STATUS_OK) {
            currSectionId = atoi(items[i + 1]);
            currSectionId--;
            LOG_DBG("ConfigParser_ParseFile: Parsing section %s index %d\n", items[i], currSectionId);
            i -= 1;
            continue;
        }

        if(GetParamIndex(paramsMap, items[i], &currItemIndex) != NVMEDIA_STATUS_OK) {
            LOG_WARN("ConfigParser_ParseFile: Parameter Name '%s' is not recognized. Dismissing this parameter.\n", items[i]);
            continue;
        }

        if(strcmp("=", items[i + 1])) {
            printf("ConfigParser_ParseFile: '=' expected as the second token in each line. Error caught while parsing parameter '%s'.\n", items[i]);
            i -= 2;
            continue;
        }

        if(ConfigParser_GetSectionIndexByType(sectionsMap, paramsMap[currItemIndex].sectionType, &sectionId) != NVMEDIA_STATUS_OK) {
            printf("ConfigParser_ParseFile: Section index couldn't be found in section map by type. Param Name: '%s'.\n", paramsMap[currItemIndex].paramName);
        }

        if(sectionsMap[sectionId].lastSectionIndex == 0) {
            // Param is not part of a collection or collection includes only one item
            currSectionId = 0;
        }

        param = (char *)paramsMap[currItemIndex].mappedLocation + currSectionId * sectionsMap[sectionId].sizeOfStruct;
        pParamLength = (char *)paramsMap[currItemIndex].stringLengthAddr + currSectionId * sectionsMap[sectionId].sizeOfStruct;
        paramDefaultLength = paramsMap[currItemIndex].stringLength;

        // Interpret the Value
        LOG_DBG("ConfigParser_ParseFile: Interpreting parameter %s\n", items[i]);
        switch(paramsMap[currItemIndex].type) {
            case TYPE_INT:
                if(sscanf(items[i + 2], "%d", &intValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected numerical value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(int *)(void *)param = intValue;
                break;
            case TYPE_UINT:
                if(sscanf(items[i + 2], "%u", &uintValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected numerical value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(unsigned int *)(void *)param = uintValue;
                break;
            case TYPE_UINT_HEX:
                if(sscanf(items[i + 2], "%x", &uintValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected unsigned char value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(unsigned int *)(void *)param = uintValue;
                break;
            case TYPE_CHAR_ARR:
                if(items[i + 2] == NULL)
                    memset(param, 0, (pParamLength != NULL && *pParamLength != 0) ? *pParamLength : paramDefaultLength);
                else {
                    strncpy(param, items[i + 2], paramsMap[currItemIndex].stringLength);
                    param[strlen(items[i + 2])] = '\0';
                }
                break;
            case TYPE_DOUBLE:
                if(sscanf(items[i + 2], "%lf", &doubleValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected double value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(double *)(void *)param = doubleValue;
                break;
            case TYPE_FLOAT:
                if(sscanf(items[i + 2], "%f", &floatValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected double value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(float *)(void *)param = floatValue;
                break;
            case TYPE_UCHAR:
                if(sscanf(items[i + 2], "%u", &charValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected unsigned char value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(unsigned char *)(void *)param = charValue;
                break;
            case TYPE_USHORT:
                if(sscanf(items[i + 2], "%hu", &ushortValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected unsigned short value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(unsigned short *)(void *)param = ushortValue;
                break;
            case TYPE_SHORT:
                if(sscanf(items[i + 2], "%hd", &shortValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected short value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(short *)(void *)param = shortValue;
                break;
            case TYPE_UCHAR_ARR:
                if(items[i + 2] == NULL)
                    memset(param, 0, (pParamLength != NULL && *pParamLength != 0) ? *pParamLength : paramDefaultLength);
                else {
                    strncpy(param, items[i + 2], paramsMap[currItemIndex].stringLength);
                    param[strlen(items[i + 2])] = '\0';
                }
                break;
            case TYPE_ULLONG:
                if(sscanf(items[i + 2], "%llu", &ullValue) != 1) {
                    printf("ConfigParser_ParseFile: Expected numerical value for Parameter %s, found value '%s'\n", items[i], items[i + 2]);
                }
                *(unsigned long long *)(void *)param = ullValue;
                break;
            default:
                printf("ConfigParser_ParseFile: Encountered unknown value type in the map\n");
        }
    }

    if (configContentBuf)
        free(configContentBuf);

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus ConfigParser_InitParamsMap(ConfigParamsMap *paramsMap)
{
    int i = 0;

    while(paramsMap[i].paramName != NULL) {
        if (paramsMap[i].mappedLocation == NULL) {
           i++;
           continue;
        }

        switch(paramsMap[i].type) {
            case TYPE_UINT:
            case TYPE_UINT_HEX:
                *(unsigned int *)(paramsMap[i].mappedLocation) = (unsigned int)paramsMap[i].defaultValue;
                break;
            case TYPE_INT:
                *(int *)(paramsMap[i].mappedLocation) = (int)paramsMap[i].defaultValue;
                break;
            case TYPE_DOUBLE:
                *(double *)(paramsMap[i].mappedLocation) = (double)paramsMap[i].defaultValue;
                break;
            case TYPE_FLOAT:
                *(float *)(paramsMap[i].mappedLocation) = (float)paramsMap[i].defaultValue;
                break;
            case TYPE_UCHAR:
                *(unsigned char *)(paramsMap[i].mappedLocation) = (NvMediaBool)paramsMap[i].defaultValue;
                break;
            case TYPE_USHORT:
                *(unsigned short *)(paramsMap[i].mappedLocation) = (unsigned short)paramsMap[i].defaultValue;
                break;
            case TYPE_SHORT:
                *(short *)(paramsMap[i].mappedLocation) = (short)paramsMap[i].defaultValue;
                break;
            case TYPE_ULLONG:
                *(unsigned long long *)(paramsMap[i].mappedLocation) = (unsigned long long)paramsMap[i].defaultValue;
                break;
            case TYPE_CHAR_ARR:
            case TYPE_UCHAR_ARR:
            default:
                break;
        }
        i++;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus ConfigParser_ValidateParams(ConfigParamsMap *paramsMap, SectionMap *sectionsMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    unsigned int sectionId = 0, i = 0, j;
    char *param;

    while(paramsMap[i].paramName != NULL) {
        if(ConfigParser_GetSectionIndexByType(sectionsMap, paramsMap[i].sectionType, &sectionId) != NVMEDIA_STATUS_OK) {
            printf("ConfigParser_ValidateParams: Section index couldn't be found in section map. Param Name: '%s'.\n", paramsMap[i].paramName);
        }

        for(j = 0; j <= sectionsMap[sectionId].lastSectionIndex; j++) {
            if(paramsMap[i].paramLimits == 1 || paramsMap[i].paramLimits == 2) {
                param = (char *)paramsMap[i].mappedLocation + j * sectionsMap[sectionId].sizeOfStruct;
                if (param == NULL) {
                    i++;
                    continue;
                }
                switch (paramsMap[i].type) {
                    case TYPE_UINT:
                    case TYPE_UINT_HEX:
                        if(*(unsigned int *)(void *)param < (unsigned int)paramsMap[i].minLimit ||
                            (paramsMap[i].paramLimits == 2 && *(unsigned int *)(void *)param > (unsigned int)paramsMap[i].maxLimit )) {
                            printf("ConfigParser_ValidateParams: Error in input parameter %s\n", paramsMap[i].paramName);
                            printf("Check configuration file for parameter limits\n");
                            status = NVMEDIA_STATUS_BAD_PARAMETER;
                        }
                        break;
                    case TYPE_DOUBLE:
                        if(*(double *)(void *)param < (double)paramsMap[i].minLimit ||
                            (paramsMap[i].paramLimits == 2 && *(double *)(void *)param > (double)paramsMap[i].maxLimit )) {
                            printf("ConfigParser_ValidateParams: Error in input parameter %s\n", paramsMap[i].paramName);
                            printf("Check configuration file for parameter limits\n");
                            status = NVMEDIA_STATUS_BAD_PARAMETER;
                        }
                        break;
                    case TYPE_FLOAT:
                        if(*(float *)(void *)param < (float)paramsMap[i].minLimit ||
                            (paramsMap[i].paramLimits == 2 && *(float *)(void *)param > (float)paramsMap[i].maxLimit )) {
                            printf("ConfigParser_ValidateParams: Error in input parameter %s\n", paramsMap[i].paramName);
                            printf("Check configuration file for parameter limits\n");
                            status = NVMEDIA_STATUS_BAD_PARAMETER;
                        }
                        break;
                    case TYPE_INT:
                        if(*(int *)(void *)param < (int)paramsMap[i].minLimit ||
                            (paramsMap[i].paramLimits == 2 && *(int *)(void *)param > (int)paramsMap[i].maxLimit )) {
                            printf("ConfigParser_ValidateParams: Error in input parameter %s\n", paramsMap[i].paramName);
                            printf("Check configuration file for parameter limits\n");
                            status = NVMEDIA_STATUS_BAD_PARAMETER;
                        }
                        break;
                    case TYPE_USHORT:
                        if(*(unsigned short *)(void *)param < (unsigned short)paramsMap[i].minLimit ||
                            (paramsMap[i].paramLimits == 2 && *(unsigned short *)(void *)param > (unsigned short)paramsMap[i].maxLimit )) {
                            printf("ConfigParser_ValidateParams: Error in input parameter %s\n", paramsMap[i].paramName);
                            printf("Check configuration file for parameter limits\n");
                            status = NVMEDIA_STATUS_BAD_PARAMETER;
                        }
                        break;
                    case TYPE_SHORT:
                        if(*(short *)(void *)param < (short)paramsMap[i].minLimit ||
                            (paramsMap[i].paramLimits == 2 && *(short *)(void *)param > (short)paramsMap[i].maxLimit )) {
                            printf("ConfigParser_ValidateParams: Error in input parameter %s\n", paramsMap[i].paramName);
                            printf("Check configuration file for parameter limits\n");
                            status = NVMEDIA_STATUS_BAD_PARAMETER;
                        }
                        break;
                    case TYPE_ULLONG:
                        if(*(unsigned long long *)(void *)param < (unsigned long long)paramsMap[i].minLimit ||
                            (paramsMap[i].paramLimits == 2 && *(unsigned long long *)(void *)param > (unsigned long long)paramsMap[i].maxLimit )) {
                            printf("ConfigParser_ValidateParams: Error in input parameter %s\n", paramsMap[i].paramName);
                            printf("Check configuration file for parameter limits\n");
                            status = NVMEDIA_STATUS_BAD_PARAMETER;
                        }
                        break;
                    default:
                        break;
                }
            }
        }
        i++;
    }

    return status;
}

NvMediaStatus ConfigParser_DisplayParams(ConfigParamsMap *pParamsMap, SectionMap *pSectionsMap)
{
    unsigned int i = 0, j, sectionId = 0;
    char *param;

    while(pParamsMap[i].paramName != NULL) {
        if(ConfigParser_GetSectionIndexByType(pSectionsMap, pParamsMap[i].sectionType, &sectionId) != NVMEDIA_STATUS_OK) {
            printf("ConfigParser_DisplayParams: Section index couldn't be found in section map by type. Param Name: '%s'.\n", pParamsMap[i].paramName);
        }

        for(j = 0; j <= pSectionsMap[sectionId].lastSectionIndex; j++) {
            param = (char *)pParamsMap[i].mappedLocation + j * pSectionsMap[sectionId].sizeOfStruct;
            if (param == NULL) {
                i++;
                continue;
            }

            switch(pParamsMap[i].type) {
                case TYPE_UINT:
                    printf("(%d) %s = %u\n", j, pParamsMap[i].paramName, *(unsigned int *)(void *)param);
                    break;
                case TYPE_DOUBLE:
                    printf("(%d) %s = %.2lf\n", j, pParamsMap[i].paramName, *(double *)(void *)param);
                    break;
                case TYPE_FLOAT:
                    printf("(%d) %s = %.2f\n", j, pParamsMap[i].paramName, *(float *)(void *)param);
                    break;
                case TYPE_UCHAR:
                    printf("(%d) %s = %d\n", j, pParamsMap[i].paramName, *(unsigned char *)(void *)param);
                    break;
                case TYPE_USHORT:
                    printf("(%d) %s = %hu\n", j, pParamsMap[i].paramName, *(unsigned short *)(void *)param);
                    break;
                case TYPE_SHORT:
                    printf("(%d) %s = %hd\n", j, pParamsMap[i].paramName, *(short *)(void *)param);
                    break;
                case TYPE_ULLONG:
                    printf("(%d) %s = %llu\n", j, pParamsMap[i].paramName, *(unsigned long long *)(void *)param);
                    break;
                case TYPE_CHAR_ARR:
                    printf("(%d) %s = ""%s""\n", j, pParamsMap[i].paramName, param);
                    break;
                case TYPE_UCHAR_ARR:
                    printf("(%d) %s = ""%s""\n", j, pParamsMap[i].paramName, (unsigned char *)(void *)param);
                    break;
                case TYPE_INT:
                    printf("(%d) %s = %d\n", j, pParamsMap[i].paramName, *(int *)(void *)param);
                    break;
                case TYPE_UINT_HEX:
                    printf("(%d) %s = %x\n", j, pParamsMap[i].paramName, *(unsigned int *)(void *)param);
                    break;
                default:
                    // Do nothing
                    break;
            }
        }
        i++;
    }

    return NVMEDIA_STATUS_OK;
}
