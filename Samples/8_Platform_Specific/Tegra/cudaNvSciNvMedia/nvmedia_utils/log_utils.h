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

#ifndef _NVMEDIA_TEST_LOG_UTILS_H_
#define _NVMEDIA_TEST_LOG_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdio.h>

enum LogLevel {
    LEVEL_ERR  = 0,
    LEVEL_WARN = 1,
    LEVEL_INFO = 2,
    LEVEL_DBG  = 3,
};

enum LogStyle {
    LOG_STYLE_NORMAL = 0,
    LOG_STYLE_FUNCTION_LINE
};

#define LINE_INFO       __FUNCTION__, __LINE__
#define LOG_DBG(...)    LogLevelMessage(LEVEL_DBG, LINE_INFO, __VA_ARGS__)
#define LOG_INFO(...)   LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__)
#define LOG_WARN(...)   LogLevelMessage(LEVEL_WARN, LINE_INFO, __VA_ARGS__)

//  SetLogLevel
//
//    SetLogLevel()  Set logging level
//
//  Arguments:
//
//   level
//      (in) Logging level

void
SetLogLevel(
    enum LogLevel level);

//  SetLogStyle
//
//    SetLogStyle()  Set logging print slyle
//
//  Arguments:
//
//   level
//      (in) Logging style

void
SetLogStyle(
    enum LogStyle style);

//  SetLogFile
//
//    SetLogFile()  Set logging file handle
//
//  Arguments:
//
//   level
//      (in) Logging file handle

void
SetLogFile(
    FILE *logFileHandle);

//  LogLevelMessage
//
//    LogLevelMessage()  Print message if logging level is higher than message level
//
//  Arguments:
//
//   LogLevel
//      (in) Message level
//
//   format
//      (in) Message format
//
//   ...
//      (in) Parameters list

void
LogLevelMessage(
    enum LogLevel level,
    const char *functionName,
    int lineNumber,
    const char *format,
    ...);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_LOG_UTILS_H_ */
