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

Display *display;
int screen;
Window win = 0;

#include <GLES3/gl31.h>
//#include <GLES3/gl3ext.h> // not (yet) needed
#include <EGL/egl.h>
#include <EGL/eglext.h>

#define GET_GLERROR(ret)                                                       \
  {                                                                            \
    GLenum err = glGetError();                                                 \
    if (err != GL_NO_ERROR)                                                    \
    {                                                                          \
      fprintf(stderr, "[%s line %d] OpenGL Error: 0x%x\n", __FILE__, __LINE__, \
              err);                                                            \
      fflush(stderr);                                                          \
                                                                               \
      switch (err)                                                             \
      {                                                                        \
      case GL_INVALID_ENUM:                                                    \
        printf("GL_INVALID_ENUM\n");                                           \
        break;                                                                 \
      case GL_INVALID_VALUE:                                                   \
        printf("GL_INVALID_VALUE\n");                                          \
        break;                                                                 \
      case GL_INVALID_OPERATION:                                               \
        printf("GL_INVALID_OPERATION\n");                                      \
        break;                                                                 \
      case GL_OUT_OF_MEMORY:                                                   \
        printf("GL_OUT_OF_MEMORY\n");                                          \
        break;                                                                 \
      case GL_INVALID_FRAMEBUFFER_OPERATION:                                   \
        printf("GL_INVALID_FRAMEBUFFER_OPERATION\n");                          \
        break;                                                                 \
      default:                                                                 \
        printf("UKNOWN OPENGL ERROR CODE 0x%x\n", err);                        \
      };                                                                       \
    }                                                                          \
  }

EGLDisplay eglDisplay = EGL_NO_DISPLAY;
EGLSurface eglSurface = EGL_NO_SURFACE;
EGLContext eglContext = EGL_NO_CONTEXT;

#if 0 // needed for optional API call retrieval (= if libGLESv2.so wouldn't be \
      // linked explicitly) - tedious! consider GLEW. 
typedef GLenum (* glGetErrorTYPE) (void);
glGetErrorTYPE my_glGetError; 

typedef GL_APICALL const GLubyte (*GL_APIENTRY glGetStringTYPE) (GLenum name);
glGetStringTYPE my_glGetString; 

typedef GL_APICALL void (*GL_APIENTRY glClearTYPE) (GLbitfield mask);
glClearTYPE my_glClear; 

typedef GL_APICALL void (*GL_APIENTRY glGetProgramivTYPE) (GLuint program, GLenum pname, GLint *params);
glGetProgramivTYPE my_glGetProgramiv;
#endif

int graphics_setup_window(int xpos, int ypos, int width, int height,
                          const char *windowname)
{
#ifdef USE_GL
  // OpenGL 4.3 Core Profile creation through EGL - would be even available on
  // desktop, but CUDA interop doesn't yet work for OpenGL context established
  // through EGL
  EGLint configAttrs[] = {EGL_RED_SIZE, 1, EGL_GREEN_SIZE, 1, EGL_BLUE_SIZE, 1,
                          // EGL_DEPTH_SIZE,    16,
                          EGL_SAMPLE_BUFFERS, 0, EGL_SAMPLES, 0, EGL_CONFORMANT,
                          EGL_OPENGL_BIT,
                          // EGL_RENDERABLE_TYPE,      EGL_OPENGL_BIT,
                          // EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR, 1,
                          EGL_NONE};
  EGLint contextAttrs[] = {// EGL_CONTEXT_MAJOR_VERSION_KHR, 4,
                           // EGL_CONTEXT_MINOR_VERSION_KHR, 3,
                           EGL_NONE};
#else // OpenGL ES 3.1
  EGLint configAttrs[] = {
      EGL_RED_SIZE, 1, EGL_GREEN_SIZE, 1, EGL_BLUE_SIZE, 1, EGL_DEPTH_SIZE, 16,
      EGL_SAMPLE_BUFFERS, 0, EGL_SAMPLES, 0,
      // EGL_CONFORMANT,      EGL_OPENGL_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, // 3_BIT_KHR,
      EGL_NONE};
  EGLint contextAttrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
#endif

  EGLint windowAttrs[] = {EGL_NONE};
  EGLConfig *configList = NULL;
  EGLint configCount;

  display = XOpenDisplay(NULL);
  if (!display)
    error_exit("Error opening X display.\n");

  screen = DefaultScreen(display);

  eglDisplay = eglGetDisplay(display);
  if (eglDisplay == EGL_NO_DISPLAY)
    error_exit("EGL failed to obtain display\n");

  if (!eglInitialize(eglDisplay, 0, 0))
    error_exit("EGL failed to initialize\n");

  if (!eglChooseConfig(eglDisplay, configAttrs, NULL, 0, &configCount) ||
      !configCount)
    error_exit("EGL failed to return any matching configurations\n");

  configList = (EGLConfig *)malloc(configCount * sizeof(EGLConfig));

  if (!eglChooseConfig(eglDisplay, configAttrs, configList, configCount,
                       &configCount) ||
      !configCount)
    error_exit("EGL failed to populate configuration list\n");

  win = XCreateSimpleWindow(display, RootWindow(display, screen), xpos, ypos,
                            width, height, 0, BlackPixel(display, screen),
                            WhitePixel(display, screen));

  XStoreName(display, win, windowname);

  XSelectInput(display, win,
               ExposureMask | ButtonPressMask | KeyPressMask |
                   StructureNotifyMask | ButtonReleaseMask | KeyReleaseMask |
                   EnterWindowMask | LeaveWindowMask | PointerMotionMask |
                   Button1MotionMask | Button2MotionMask |
                   VisibilityChangeMask | ColormapChangeMask);

  XMapWindow(display, win);

  eglSurface = eglCreateWindowSurface(eglDisplay, configList[0],
                                      (EGLNativeWindowType)win, windowAttrs);
  if (!eglSurface)
    error_exit("EGL couldn't create window\n");

#ifdef USE_GL
  eglBindAPI(EGL_OPENGL_API);
#else
  eglBindAPI(EGL_OPENGL_ES_API);
#endif

  eglContext = eglCreateContext(eglDisplay, configList[0], NULL, contextAttrs);
  if (!eglContext)
    error_exit("EGL couldn't create context\n");

  if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext))
    error_exit("EGL couldn't make context/surface current\n");

  EGLint Context_RendererType;
  eglQueryContext(eglDisplay, eglContext, EGL_CONTEXT_CLIENT_TYPE,
                  &Context_RendererType);

  switch (Context_RendererType)
  {
  case EGL_OPENGL_API:
    printf("Using OpenGL API\n");
    break;
  case EGL_OPENGL_ES_API:
    printf("Using OpenGL ES API");
    break;
  case EGL_OPENVG_API:
    error_exit("Context Query Returned OpenVG. This is Unsupported\n");
  default:
    error_exit("Unknown Context Type. %04X\n", Context_RendererType);
  }

#if 0 // obtain API function pointers _manually_ (see function pointer \
      // declarations above)
    my_glGetError = (glGetErrorTYPE) eglGetProcAddress("glGetError"); 
    my_glGetString = (glGetStringTYPE) eglGetProcAddress("glGetString"); 
    my_glGetProgramiv = (glGetProgramivTYPE) eglGetProcAddress("glGetProgramiv"); 


    GL_APICALL void (* my_glGenBuffers) (GLsizei n, GLuint *buffers);
    my_glGenBuffers = ((*) (GLsizei n, GLuint *buffers)) eglGetProcAddress("glGenBuffers"); 

    GL_APICALL void (* my_glGetShaderInfoLog) (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
    my_glGetShaderInfoLog = ((*) (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog)) eglGetProcAddress("glGetShaderInfoLog"); 

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer);
#endif

  return 1;
}

void graphics_set_windowtitle(const char *windowname)
{
  XStoreName(display, win, windowname);
}

void graphics_swap_buffers() { eglSwapBuffers(eglDisplay, eglSurface); }

void graphics_close_window()
{
  if (eglDisplay != EGL_NO_DISPLAY)
  {
    eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

    if (eglContext != EGL_NO_CONTEXT)
      eglDestroyContext(eglDisplay, eglContext);

    if (eglSurface != EGL_NO_SURFACE)
      eglDestroySurface(eglDisplay, eglSurface);

    eglTerminate(eglDisplay);
  }

  if (win)
    XDestroyWindow(display, win);

  XCloseDisplay(display);
}
