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

#include <GLES3/gl31.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <sys/keycodes.h>

screen_window_t screen_window;
screen_context_t screen_context;
screen_event_t screen_ev;

EGLDisplay eglDisplay = EGL_NO_DISPLAY;
EGLSurface eglSurface = EGL_NO_SURFACE;
EGLContext eglContext = EGL_NO_CONTEXT;

void error_exit(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  exit(1);
}

enum { NvGlDemoKeyCode_Escape = 27 };

typedef void (*GlCloseCB)(void);
typedef void (*GlKeyCB)(char key, int state);
GlCloseCB closeCB = NULL;
GlKeyCB keyCB = NULL;

void CHECK_GLERROR() {
  GLenum err = glGetError();

  if (err != GL_NO_ERROR) {
    fprintf(stderr, "[%s line %d] OpenGL Error: 0x%x ", __FILE__, __LINE__,
            err);

    switch (err) {
      case GL_INVALID_ENUM:
        fprintf(stderr, "(GL_INVALID_ENUM)\n");
        break;
      case GL_INVALID_VALUE:
        fprintf(stderr, "(GL_INVALID_VALUE)\n");
        break;
      case GL_INVALID_OPERATION:
        fprintf(stderr, "(GL_INVALID_OPERATION)\n");
        break;
      case GL_OUT_OF_MEMORY:
        fprintf(stderr, "(GL_OUT_OF_MEMORY)\n");
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        fprintf(stderr, "(GL_INVALID_FRAMEBUFFER_OPERATION)\n");
        break;
      default:
        break;
    }

    fflush(stderr);
  }
}

static void UpdateEventMask(void) {
  static int rc = 1;

  if (rc) {
    rc = screen_create_event(&screen_ev);
  }
}

void SetCloseCB(GlCloseCB cb) {
  // Call the eglQnxScreenConsumer module if option is enabled
  closeCB = cb;
  UpdateEventMask();
}

void SetKeyCB(GlKeyCB cb) {
  keyCB = cb;
  UpdateEventMask();
}

// Add keys here, that are used in demo apps.
static unsigned char GetKeyPress(int *screenKey) {
  unsigned char key = '\0';

  switch (*screenKey) {
    case KEYCODE_ESCAPE:
      key = NvGlDemoKeyCode_Escape;
      break;
    default:
      /* For "normal" keys, Screen KEYCODE is just ASCII. */
      if (*screenKey <= 127) {
        key = *screenKey;
      }
      break;
  }

  return key;
}

void CheckEvents(void) {
  static int vis = 1, val = 1;
  int rc;

  /**
   ** We start the loop by processing any events that might be in our
   ** queue. The only event that is of interest to us are the resize
   ** and close events. The timeout variable is set to 0 (no wait) or
   ** forever depending if the window is visible or invisible.
   **/

  while (!screen_get_event(screen_context, screen_ev, vis ? 0ull : ~0ull)) {
    // Get QNX CAR 2.1 event property
    rc = screen_get_event_property_iv(screen_ev, SCREEN_PROPERTY_TYPE, &val);
    if (rc || val == SCREEN_EVENT_NONE) {
      break;
    }

    switch (val) {
      case SCREEN_EVENT_CLOSE:
        /**
         ** All we have to do when we receive the close event is
         ** exit the application loop.
         **/
        if (closeCB) {
          closeCB();
        }
        break;

      case SCREEN_EVENT_KEYBOARD:
        rc = screen_get_event_property_iv(screen_ev, SCREEN_PROPERTY_FLAGS,
                                          &val);
        if (rc || val == SCREEN_EVENT_NONE) {
          break;
        }
        if (val & KEY_DOWN) {
          rc = screen_get_event_property_iv(screen_ev, SCREEN_PROPERTY_SYM,
                                            &val);
          if (rc || val == SCREEN_EVENT_NONE) {
            break;
          }
          unsigned char key;
          key = GetKeyPress(&val);
          if (key != '\0') {
            keyCB(key, 1);
          }
        }
        break;

      default:
        break;
    }
  }
}

int graphics_setup_window(int xpos, int ypos, int width, int height,
                          const char *windowname, int reqdispno) {
  EGLint configAttrs[] = {
      EGL_RED_SIZE,   8,  EGL_GREEN_SIZE,      8,
      EGL_BLUE_SIZE,  8,  EGL_ALPHA_SIZE,      8,
      EGL_DEPTH_SIZE, 16, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_NONE};

  EGLint contextAttrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};

  EGLint windowAttrs[] = {EGL_NONE};
  EGLConfig *configList = NULL;
  EGLint configCount;

  int displayCount = 0;
  int dispno;

  screen_context = 0;

  screen_display_t *screenDisplayHandle = NULL;

  if (screen_create_context(&screen_context, 0)) {
    error_exit("Error creating screen context.\n");
  }

  eglDisplay = eglGetDisplay(0);

  if (eglDisplay == EGL_NO_DISPLAY) {
    error_exit("EGL failed to obtain display\n");
  }

  if (!eglInitialize(eglDisplay, 0, 0)) {
    error_exit("EGL failed to initialize\n");
  }

  if (!eglChooseConfig(eglDisplay, configAttrs, NULL, 0, &configCount) ||
      !configCount) {
    error_exit("EGL failed to return any matching configurations\n");
  }

  configList = (EGLConfig *)malloc(configCount * sizeof(EGLConfig));

  if (!eglChooseConfig(eglDisplay, configAttrs, configList, configCount,
                       &configCount) ||
      !configCount) {
    error_exit("EGL failed to populate configuration list\n");
  }

  screen_window = 0;
  if (screen_create_window(&screen_window, screen_context)) {
    error_exit("Error creating screen window.\n");
  }

  // query the total no of display avaibale from QNX CAR2 screen
  if (screen_get_context_property_iv(
          screen_context, SCREEN_PROPERTY_DISPLAY_COUNT, &displayCount)) {
    error_exit("Error getting context property\n");
  }

  screenDisplayHandle =
      (screen_display_t *)malloc(displayCount * sizeof(screen_display_t));
  if (!screenDisplayHandle) {
    error_exit("Error allocating screen memory handle is getting failed\n");
  }

  // query the display handle from QNX CAR2 screen
  if (screen_get_context_property_pv(screen_context, SCREEN_PROPERTY_DISPLAYS,
                                     (void **)screenDisplayHandle)) {
    error_exit("Error getting display handle\n");
  }

  for (dispno = 0; dispno < displayCount; dispno++) {
    int active = 0;
    // Query the connected status from QNX CAR2 screen
    screen_get_display_property_iv(screenDisplayHandle[dispno],
                                   SCREEN_PROPERTY_ATTACHED, &active);
    if (active) {
      if (reqdispno == dispno) {
        // Map the window buffer to user requested display port
        screen_set_window_property_pv(screen_window, SCREEN_PROPERTY_DISPLAY,
                                      (void **)&screenDisplayHandle[reqdispno]);
        break;
      }
    }
  }

  if (dispno == displayCount) {
    error_exit("Failed to set the requested display\n");
  }

  free(screenDisplayHandle);

  int format = SCREEN_FORMAT_RGBA8888;
  if (screen_set_window_property_iv(screen_window, SCREEN_PROPERTY_FORMAT,
                                    &format)) {
    error_exit("Error setting SCREEN_PROPERTY_FORMAT\n");
  }

  int usage = SCREEN_USAGE_OPENGL_ES2;
  if (screen_set_window_property_iv(screen_window, SCREEN_PROPERTY_USAGE,
                                    &usage)) {
    error_exit("Error setting SCREEN_PROPERTY_USAGE\n");
  }

  EGLint interval = 1;
  if (screen_set_window_property_iv(screen_window,
                                    SCREEN_PROPERTY_SWAP_INTERVAL, &interval)) {
    error_exit("Error setting SCREEN_PROPERTY_SWAP_INTERVAL\n");
  }

  int windowSize[2];
  windowSize[0] = width;
  windowSize[1] = height;
  if (screen_set_window_property_iv(screen_window, SCREEN_PROPERTY_SIZE,
                                    windowSize)) {
    error_exit("Error setting SCREEN_PROPERTY_SIZE\n");
  }

  int windowOffset[2];
  windowOffset[0] = xpos;
  windowOffset[1] = ypos;
  if (screen_set_window_property_iv(screen_window, SCREEN_PROPERTY_POSITION,
                                    windowOffset)) {
    error_exit("Error setting SCREEN_PROPERTY_POSITION\n");
  }

  if (screen_create_window_buffers(screen_window, 2)) {
    error_exit("Error creating two window buffers.\n");
  }

  eglSurface =
      eglCreateWindowSurface(eglDisplay, configList[0],
                             (EGLNativeWindowType)screen_window, windowAttrs);
  if (!eglSurface) {
    error_exit("EGL couldn't create window\n");
  }

  eglBindAPI(EGL_OPENGL_ES_API);

  eglContext = eglCreateContext(eglDisplay, configList[0], NULL, contextAttrs);
  if (!eglContext) {
    error_exit("EGL couldn't create context\n");
  }

  if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
    error_exit("EGL couldn't make context/surface current\n");
  }

  EGLint Context_RendererType;
  eglQueryContext(eglDisplay, eglContext, EGL_CONTEXT_CLIENT_TYPE,
                  &Context_RendererType);

  switch (Context_RendererType) {
    case EGL_OPENGL_API:
      printf("Using OpenGL API\n");
      break;
    case EGL_OPENGL_ES_API:
      printf("Using OpenGL ES API\n");
      break;
    case EGL_OPENVG_API:
      error_exit("Context Query Returned OpenVG. This is Unsupported\n");
    default:
      error_exit("Unknown Context Type. %04X\n", Context_RendererType);
  }

  return 1;
}

void graphics_set_windowtitle(const char *windowname) {
  // Do nothing on screen
}

void graphics_swap_buffers() { eglSwapBuffers(eglDisplay, eglSurface); }

void graphics_close_window() {
  if (eglDisplay != EGL_NO_DISPLAY) {
    eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

    if (eglContext != EGL_NO_CONTEXT) {
      eglDestroyContext(eglDisplay, eglContext);
    }

    if (eglSurface != EGL_NO_SURFACE) {
      eglDestroySurface(eglDisplay, eglSurface);
    }

    eglTerminate(eglDisplay);
  }

  if (screen_window) {
    screen_destroy_window(screen_window);
    screen_window = NULL;
  }
  if (screen_context) {
    screen_destroy_context(screen_context);
  }
  if (screen_ev) {
    screen_destroy_event(screen_ev);
  }
}
