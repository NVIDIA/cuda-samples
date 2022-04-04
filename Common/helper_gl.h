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

// These are helper functions for the SDK samples (OpenGL)
#ifndef HELPER_GL_H
#define HELPER_GL_H

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #include <GL/glew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
    #ifdef __linux__
    #include <GL/glx.h>
    #endif /* __linux__ */
#endif

#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <assert.h>


/* Prototypes */
namespace __HelperGL {
    static int isGLVersionSupported(unsigned reqMajor, unsigned reqMinor);
    static int areGLExtensionsSupported(const std::string &);
#ifdef __linux__

    #ifndef HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
    #define USE_GL_FUNC(name, proto) proto name = (proto) glXGetProcAddress ((const GLubyte *)#name)
    #else
    #define USE_GL_FUNC(name, proto) extern proto name
    #endif

    USE_GL_FUNC(glBindBuffer, PFNGLBINDBUFFERPROC);
    USE_GL_FUNC(glDeleteBuffers, PFNGLDELETEBUFFERSPROC);
    USE_GL_FUNC(glBufferData, PFNGLBUFFERDATAPROC);
    USE_GL_FUNC(glBufferSubData, PFNGLBUFFERSUBDATAPROC);
    USE_GL_FUNC(glGenBuffers, PFNGLGENBUFFERSPROC);
    USE_GL_FUNC(glCreateProgram, PFNGLCREATEPROGRAMPROC);
    USE_GL_FUNC(glBindProgramARB, PFNGLBINDPROGRAMARBPROC);
    USE_GL_FUNC(glGenProgramsARB, PFNGLGENPROGRAMSARBPROC);
    USE_GL_FUNC(glDeleteProgramsARB, PFNGLDELETEPROGRAMSARBPROC);
    USE_GL_FUNC(glDeleteProgram, PFNGLDELETEPROGRAMPROC);
    USE_GL_FUNC(glGetProgramInfoLog, PFNGLGETPROGRAMINFOLOGPROC);
    USE_GL_FUNC(glGetProgramiv, PFNGLGETPROGRAMIVPROC);
    USE_GL_FUNC(glProgramParameteriEXT, PFNGLPROGRAMPARAMETERIEXTPROC);
    USE_GL_FUNC(glProgramStringARB, PFNGLPROGRAMSTRINGARBPROC);
    USE_GL_FUNC(glUnmapBuffer, PFNGLUNMAPBUFFERPROC);
    USE_GL_FUNC(glMapBuffer, PFNGLMAPBUFFERPROC);
    USE_GL_FUNC(glGetBufferParameteriv, PFNGLGETBUFFERPARAMETERIVPROC);
    USE_GL_FUNC(glLinkProgram, PFNGLLINKPROGRAMPROC);
    USE_GL_FUNC(glUseProgram, PFNGLUSEPROGRAMPROC);
    USE_GL_FUNC(glAttachShader, PFNGLATTACHSHADERPROC);
    USE_GL_FUNC(glCreateShader, PFNGLCREATESHADERPROC);
    USE_GL_FUNC(glShaderSource, PFNGLSHADERSOURCEPROC);
    USE_GL_FUNC(glCompileShader, PFNGLCOMPILESHADERPROC);
    USE_GL_FUNC(glDeleteShader, PFNGLDELETESHADERPROC);
    USE_GL_FUNC(glGetShaderInfoLog, PFNGLGETSHADERINFOLOGPROC);
    USE_GL_FUNC(glGetShaderiv, PFNGLGETSHADERIVPROC);
    USE_GL_FUNC(glUniform1i, PFNGLUNIFORM1IPROC);
    USE_GL_FUNC(glUniform1f, PFNGLUNIFORM1FPROC);
    USE_GL_FUNC(glUniform2f, PFNGLUNIFORM2FPROC);
    USE_GL_FUNC(glUniform3f, PFNGLUNIFORM3FPROC);
    USE_GL_FUNC(glUniform4f, PFNGLUNIFORM4FPROC);
    USE_GL_FUNC(glUniform1fv, PFNGLUNIFORM1FVPROC);
    USE_GL_FUNC(glUniform2fv, PFNGLUNIFORM2FVPROC);
    USE_GL_FUNC(glUniform3fv, PFNGLUNIFORM3FVPROC);
    USE_GL_FUNC(glUniform4fv, PFNGLUNIFORM4FVPROC);
    USE_GL_FUNC(glUniformMatrix4fv, PFNGLUNIFORMMATRIX4FVPROC);
    USE_GL_FUNC(glSecondaryColor3fv, PFNGLSECONDARYCOLOR3FVPROC);
    USE_GL_FUNC(glGetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC);
    USE_GL_FUNC(glGenFramebuffersEXT, PFNGLGENFRAMEBUFFERSEXTPROC);
    USE_GL_FUNC(glBindFramebufferEXT, PFNGLBINDFRAMEBUFFEREXTPROC);
    USE_GL_FUNC(glDeleteFramebuffersEXT, PFNGLDELETEFRAMEBUFFERSEXTPROC);
    USE_GL_FUNC(glCheckFramebufferStatusEXT, PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC);
    USE_GL_FUNC(glGetFramebufferAttachmentParameterivEXT, PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC);
    USE_GL_FUNC(glFramebufferTexture1DEXT, PFNGLFRAMEBUFFERTEXTURE1DEXTPROC);
    USE_GL_FUNC(glFramebufferTexture2DEXT, PFNGLFRAMEBUFFERTEXTURE2DEXTPROC);
    USE_GL_FUNC(glFramebufferTexture3DEXT, PFNGLFRAMEBUFFERTEXTURE3DEXTPROC);
    USE_GL_FUNC(glGenerateMipmapEXT, PFNGLGENERATEMIPMAPEXTPROC);
    USE_GL_FUNC(glGenRenderbuffersEXT, PFNGLGENRENDERBUFFERSEXTPROC);
    USE_GL_FUNC(glDeleteRenderbuffersEXT, PFNGLDELETERENDERBUFFERSEXTPROC);
    USE_GL_FUNC(glBindRenderbufferEXT, PFNGLBINDRENDERBUFFEREXTPROC);
    USE_GL_FUNC(glRenderbufferStorageEXT, PFNGLRENDERBUFFERSTORAGEEXTPROC);
    USE_GL_FUNC(glFramebufferRenderbufferEXT, PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC);
    USE_GL_FUNC(glClampColorARB, PFNGLCLAMPCOLORARBPROC);
    USE_GL_FUNC(glBindFragDataLocationEXT, PFNGLBINDFRAGDATALOCATIONEXTPROC);

#if !defined(GLX_EXTENSION_NAME) || !defined(GL_VERSION_1_3)
    USE_GL_FUNC(glActiveTexture, PFNGLACTIVETEXTUREPROC);
    USE_GL_FUNC(glClientActiveTexture, PFNGLACTIVETEXTUREPROC);
#endif

    #undef USE_GL_FUNC
#endif /*__linux__ */
}


namespace __HelperGL {
    namespace __Int {
        static std::vector<std::string> split(const std::string &str)
        {
            std::istringstream ss(str);
            std::istream_iterator<std::string> it(ss);
            return std::vector<std::string> (it, std::istream_iterator<std::string>());
        }

        /* Sort the vector passed by reference */
        template<typename T> static inline void sort(std::vector<T> &a)
        {
            std::sort(a.begin(), a.end());
        }

        /* Compare two vectors */
        template<typename T> static int equals(std::vector<T> a, std::vector<T> b)
        {
            if (a.size() != b.size()) return 0;
            sort(a);
            sort(b);

            return std::equal(a.begin(), a.end(), b.begin());
        }

        template<typename T> static std::vector<T> getIntersection(std::vector<T> a, std::vector<T> b)
        {
            sort(a);
            sort(b);

            std::vector<T> rc;
            std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                             std::back_inserter<std::vector<std::string> >(rc));
            return rc;
        }

        static std::vector<std::string> getGLExtensions()
        {
            std::string extensionsStr( (const char *)glGetString(GL_EXTENSIONS));
            return split (extensionsStr);
        }
    }

    static int areGLExtensionsSupported(const std::string &extensions)
    {
        std::vector<std::string> all = __Int::getGLExtensions();

        std::vector<std::string> requested = __Int::split(extensions);
        std::vector<std::string> matched = __Int::getIntersection(all, requested);

        return __Int::equals(matched, requested);
    }

    static int isGLVersionSupported(unsigned reqMajor, unsigned reqMinor)
    {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        if (glewInit() != GLEW_OK)
        {
            std::cerr << "glewInit() failed!" << std::endl;
            return 0;
        }
#endif
        std::string version ((const char *) glGetString (GL_VERSION));
        std::stringstream stream (version);
        unsigned major, minor;
        char dot;

        stream >> major >> dot >> minor;

        assert (dot == '.');
        return major > reqMajor || (major == reqMajor && minor >= reqMinor);
    }

    static inline const char* glErrorToString(GLenum err)
    {
#define CASE_RETURN_MACRO(arg) case arg: return #arg
        switch(err)
        {
            CASE_RETURN_MACRO(GL_NO_ERROR);
            CASE_RETURN_MACRO(GL_INVALID_ENUM);
            CASE_RETURN_MACRO(GL_INVALID_VALUE);
            CASE_RETURN_MACRO(GL_INVALID_OPERATION);
            CASE_RETURN_MACRO(GL_OUT_OF_MEMORY);
            CASE_RETURN_MACRO(GL_STACK_UNDERFLOW);
            CASE_RETURN_MACRO(GL_STACK_OVERFLOW);
#ifdef GL_INVALID_FRAMEBUFFER_OPERATION
            CASE_RETURN_MACRO(GL_INVALID_FRAMEBUFFER_OPERATION);
#endif
            default: break;
        }
#undef CASE_RETURN_MACRO
        return "*UNKNOWN*";
    }

////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return bool if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
//! @note This function should be used via the CHECK_ERROR_GL() macro
////////////////////////////////////////////////////////////////////////////
    inline bool sdkCheckErrorGL(const char *file, const int line)
    {
        bool ret_val = true;

        // check for error
        GLenum gl_error = glGetError();

        if (gl_error != GL_NO_ERROR)
        {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
            char tmpStr[512];
            // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
            // when the user double clicks on the error line in the Output pane. Like any compile error.
            sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, glErrorToString(gl_error));
            fprintf(stderr, "%s", tmpStr);
#endif
            fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
            fprintf(stderr, "%s\n", glErrorToString(gl_error));
            ret_val = false;
        }

        return ret_val;
    }

#define SDK_CHECK_ERROR_GL()                                              \
    if( false == sdkCheckErrorGL( __FILE__, __LINE__)) {                  \
        exit(EXIT_FAILURE);                                               \
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

} /* of namespace __HelperGL*/

using namespace __HelperGL;

#endif /*HELPER_GL_H*/
