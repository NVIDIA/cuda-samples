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

// Display *display;
int screen;
// Window win = 0;

#include <assert.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#include <GLES3/gl31.h>
//#include <GLES3/gl3ext.h> // not (yet) needed
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define MAX_DEVICES 16

static PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = NULL;
static PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT = NULL;
static PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = NULL;
static PFNEGLGETOUTPUTLAYERSEXTPROC eglGetOutputLayersEXT = NULL;
static PFNEGLCREATESTREAMKHRPROC eglCreateStreamKHR = NULL;
static PFNEGLDESTROYSTREAMKHRPROC eglDestroyStreamKHR = NULL;
static PFNEGLSTREAMCONSUMEROUTPUTEXTPROC eglStreamConsumerOutputEXT = NULL;
static PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC
    eglCreateStreamProducerSurfaceKHR = NULL;

#define GET_GLERROR(ret)                                                       \
  {                                                                            \
    GLenum err = glGetError();                                                 \
    if (err != GL_NO_ERROR) {                                                  \
      fprintf(stderr, "[%s line %d] OpenGL Error: 0x%x\n", __FILE__, __LINE__, \
              err);                                                            \
      fflush(stderr);                                                          \
                                                                               \
      switch (err) {                                                           \
        case GL_INVALID_ENUM:                                                  \
          printf("GL_INVALID_ENUM\n");                                         \
          break;                                                               \
        case GL_INVALID_VALUE:                                                 \
          printf("GL_INVALID_VALUE\n");                                        \
          break;                                                               \
        case GL_INVALID_OPERATION:                                             \
          printf("GL_INVALID_OPERATION\n");                                    \
          break;                                                               \
        case GL_OUT_OF_MEMORY:                                                 \
          printf("GL_OUT_OF_MEMORY\n");                                        \
          break;                                                               \
        case GL_INVALID_FRAMEBUFFER_OPERATION:                                 \
          printf("GL_INVALID_FRAMEBUFFER_OPERATION\n");                        \
          break;                                                               \
        default:                                                               \
          printf("UKNOWN OPENGL ERROR CODE 0x%x\n", err);                      \
      };                                                                       \
    }                                                                          \
  }

EGLDisplay eglDisplay = EGL_NO_DISPLAY;
EGLSurface eglSurface = EGL_NO_SURFACE;
EGLContext eglContext = EGL_NO_CONTEXT;

#if 0  // needed for optional API call retrieval (= if libGLESv2.so wouldn't be
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

// Extension checking utility
static bool CheckExtension(const char *exts, const char *ext) {
  int extLen = (int)strlen(ext);
  const char *end = exts + strlen(exts);

  while (exts < end) {
    while (*exts == ' ') {
      exts++;
    }
    int n = strcspn(exts, " ");
    if ((extLen == n) && (strncmp(ext, exts, n) == 0)) {
      return true;
    }
    exts += n;
  }
  return false;
}

int graphics_setup_window(int xpos, int ypos, int width, int height,
                          const char *windowname) {
  int device = 0, crtc = -1, plane = -1;
  int xsurfsize = 0, ysurfsize = 0;
  int xoffset = 0, yoffset = 0;
  int xmodesize = 0, ymodesize = 0;
  // int color = 0, duration = 10;
  int fifo = 0;
  int bounce = 0;
  uint32_t fb_id = -1;

  EGLDeviceEXT egl_devs[MAX_DEVICES], egl_dev;
  EGLOutputLayerEXT egl_lyr;
  EGLConfig egl_cfg;
  EGLStreamKHR egl_str;
  EGLint major, minor;

  const char *drm_name;
  int drm_fd;
  uint32_t drm_conn_id, drm_enc_id, drm_crtc_id, drm_plane_id;
  uint32_t crtc_mask;
  drmModeRes *drm_res_info = NULL;
  drmModePlaneRes *drm_plane_res_info = NULL;
  drmModeCrtc *drm_crtc_info = NULL;
  drmModeConnector *drm_conn_info = NULL;
  drmModeEncoder *drm_enc_info = NULL;
  drmModePlane *drm_plane_info = NULL;
  int drm_mode_index = 0;

  bool set_mode = false;
  int i, n;

  // Load extension function pointers.
  eglQueryDevicesEXT =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
  eglQueryDeviceStringEXT = (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress(
      "eglQueryDeviceStringEXT");
  eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
      "eglGetPlatformDisplayEXT");
  eglGetOutputLayersEXT =
      (PFNEGLGETOUTPUTLAYERSEXTPROC)eglGetProcAddress("eglGetOutputLayersEXT");
  eglCreateStreamKHR =
      (PFNEGLCREATESTREAMKHRPROC)eglGetProcAddress("eglCreateStreamKHR");
  eglDestroyStreamKHR =
      (PFNEGLDESTROYSTREAMKHRPROC)eglGetProcAddress("eglDestroyStreamKHR");
  eglStreamConsumerOutputEXT =
      (PFNEGLSTREAMCONSUMEROUTPUTEXTPROC)eglGetProcAddress(
          "eglStreamConsumerOutputEXT");
  eglCreateStreamProducerSurfaceKHR =
      (PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC)eglGetProcAddress(
          "eglCreateStreamProducerSurfaceKHR");
  if (!eglQueryDevicesEXT || !eglQueryDeviceStringEXT ||
      !eglGetPlatformDisplayEXT || !eglGetOutputLayersEXT ||
      !eglCreateStreamKHR || !eglDestroyStreamKHR ||
      !eglStreamConsumerOutputEXT || !eglCreateStreamProducerSurfaceKHR) {
    printf("Missing required function(s)\n");
    exit(2);
  }
  printf("Loaded extension functions\n");

  // Query device
  if (!eglQueryDevicesEXT(device + 1, egl_devs, &n) || (n <= device)) {
    printf("Requested device index (%d) not found\n", device);
    exit(2);
  }
  egl_dev = egl_devs[device];

  // Obtain and open DRM device file
  drm_name = eglQueryDeviceStringEXT(egl_dev, EGL_DRM_DEVICE_FILE_EXT);
  if (!drm_name) {
    printf("Couldn't obtain device file from 0x%p\n",
           (void *)(uintptr_t)egl_dev);
    exit(3);
  }

  if (!strcmp(drm_name, "drm-nvdc")) {
    drm_fd = drmOpen(drm_name, NULL);
  } else {
    drm_fd = open(drm_name, O_RDWR, 0);
  }

  if (drm_fd == -1) {
    printf("Couldn't open device file '%s'\n", drm_name);
    exit(3);
  }
  printf("Device file: %s\n", drm_name);

  // Obtain DRM-KMS resources
  drm_res_info = drmModeGetResources(drm_fd);
  if (!drm_res_info) {
    printf("Couldn't obtain DRM-KMS resources\n");
    exit(3);
  }
  printf("Obtained device information\n");

  // If a specific crtc was requested, make sure it exists
  if (crtc >= drm_res_info->count_crtcs) {
    printf("Requested crtc index (%d) exceeds count (%d)\n", crtc,
           drm_res_info->count_crtcs);
    exit(4);
  }
  crtc_mask =
      (crtc >= 0) ? (1 << crtc) : ((1 << drm_res_info->count_crtcs) - 1);

  // If drawing to a plane is requested, obtain the plane info
  if (plane >= 0) {
    drm_plane_res_info = drmModeGetPlaneResources(drm_fd);
    if (!drm_plane_res_info) {
      printf("Unable to obtain plane resource list\n");
      exit(5);
    }
    if (plane >= drm_plane_res_info->count_planes) {
      printf("Requested plane index (%d) exceeds count (%d)\n", plane,
             drm_plane_res_info->count_planes);
      exit(5);
    }
    drm_plane_id = drm_plane_res_info->planes[plane];
    drm_plane_info = drmModeGetPlane(drm_fd, drm_plane_id);
    if (!drm_plane_info) {
      printf("Unable to obtain info for plane (%d)\n", drm_plane_id);
      exit(5);
    }
    crtc_mask &= drm_plane_info->possible_crtcs;
    if (!crtc_mask) {
      printf("Requested crtc and plane not compatible\n");
      exit(5);
    }
    printf("Obtained plane information\n");
  }

  // Query info for requested connector
  int conn = 0;
  for (conn = 0; conn < drm_res_info->count_connectors; ++conn) {
    drm_conn_id = drm_res_info->connectors[conn];
    drm_conn_info = drmModeGetConnector(drm_fd, drm_conn_id);
    if (drm_conn_info != NULL) {
      printf("connector %d found\n", drm_conn_info->connector_id);
      if (drm_conn_info->connection == DRM_MODE_CONNECTED) {
        break;
      }
      drmModeFreeConnector(drm_conn_info);
    }
  }

  if (conn == drm_res_info->count_connectors) {
    printf("No active connectors found\n");
    exit(6);
  }
  printf("Obtained connector information\n");

  // If there is already an encoder attached to the connector, choose
  //   it unless not compatible with crtc/plane
  drm_enc_id = drm_conn_info->encoder_id;
  drm_enc_info = drmModeGetEncoder(drm_fd, drm_enc_id);
  if (drm_enc_info) {
    if (!(drm_enc_info->possible_crtcs & crtc_mask)) {
      drmModeFreeEncoder(drm_enc_info);
      drm_enc_info = NULL;
    }
  }

  // If we didn't have a suitable encoder, find one
  if (!drm_enc_info) {
    for (i = 0; i < drm_conn_info->count_encoders; ++i) {
      drm_enc_id = drm_conn_info->encoders[i];
      drm_enc_info = drmModeGetEncoder(drm_fd, drm_enc_id);
      if (drm_enc_info) {
        if (crtc_mask & drm_enc_info->possible_crtcs) {
          crtc_mask &= drm_enc_info->possible_crtcs;
          break;
        }
        drmModeFreeEncoder(drm_enc_info);
        drm_enc_info = NULL;
      }
    }
    if (i == drm_conn_info->count_encoders) {
      printf("Unable to find suitable encoder\n");
      exit(7);
    }
  }
  printf("Obtained encoder information\n");

  // Select a suitable crtc. Give preference to any that's already
  //   attached to the encoder. (Could make this more sophisticated
  //   by finding one not already bound to any other encoders. But
  //   this is just a basic test, so we don't really care that much.)
  assert(crtc_mask);
  for (i = 0; i < drm_res_info->count_crtcs; ++i) {
    if (crtc_mask & (1 << i)) {
      drm_crtc_id = drm_res_info->crtcs[i];
      if (drm_res_info->crtcs[i] == drm_enc_info->crtc_id) {
        break;
      }
    }
  }

  // Query info for crtc
  drm_crtc_info = drmModeGetCrtc(drm_fd, drm_crtc_id);
  if (!drm_crtc_info) {
    printf("Unable to obtain info for crtc (%d)\n", drm_crtc_id);
    exit(4);
  }
  printf("Obtained crtc information\n");

  // If dimensions are specified and not using a plane, find closest mode
  if ((xmodesize || ymodesize) && (plane < 0)) {
    // Find best fit among available modes
    int best_index = 0;
    int best_fit = 0x7fffffff;
    for (i = 0; i < drm_conn_info->count_modes; ++i) {
      drmModeModeInfoPtr mode = drm_conn_info->modes + i;
      int fit = 0;

      if (xmodesize) {
        fit += abs((int)mode->hdisplay - xmodesize) * (int)mode->vdisplay;
      }
      if (ymodesize) {
        fit += abs((int)mode->vdisplay - ymodesize) * (int)mode->hdisplay;
      }

      if (fit < best_fit) {
        best_index = i;
        best_fit = fit;
      }
    }

    // Choose this size/mode
    drm_mode_index = best_index;
    xmodesize = (int)drm_conn_info->modes[best_index].hdisplay;
    ymodesize = (int)drm_conn_info->modes[best_index].vdisplay;
  }

  // We'll only set the mode if we have to. This hopefully allows
  //   multiple instances of this application to run, writing to
  //   separate planes of the same display, as long as they don't
  //   specifiy incompatible settings.
  if ((drm_conn_info->encoder_id != drm_enc_id) ||
      (drm_enc_info->crtc_id != drm_crtc_id) || !drm_crtc_info->mode_valid ||
      ((plane < 0) && xmodesize &&
       (xmodesize != (int)drm_crtc_info->mode.hdisplay)) ||
      ((plane < 0) && ymodesize &&
       (ymodesize != (int)drm_crtc_info->mode.vdisplay))) {
    set_mode = true;
  }

  // If dimensions haven't been specified, figure out good values to use
  if (!xmodesize || !ymodesize) {
    // If mode requires reset, just pick the first one available
    //   from the connector
    if (set_mode) {
      xmodesize = (int)drm_conn_info->modes[0].hdisplay;
      ymodesize = (int)drm_conn_info->modes[0].vdisplay;
    }

    // Otherwise get it from the current crtc settings
    else {
      xmodesize = (int)drm_crtc_info->mode.hdisplay;
      ymodesize = (int)drm_crtc_info->mode.vdisplay;
    }
  }
  printf("Determine mode settings\n");

  // If surf size is unspecified, default to fullscreen normally
  // or to 1/4 fullscreen if in animated bounce mode.
  if (!xsurfsize || !ysurfsize) {
    if (bounce) {
      xsurfsize = xmodesize / 2;
      ysurfsize = ymodesize / 2;
    } else {
      xsurfsize = xmodesize;
      ysurfsize = ymodesize;
    }
  }
  printf("Determine surface size\n");

  // create framebuffer (required for nvidia-drm)
  drmVersionPtr version = drmGetVersion(drm_fd);
  if (!version) {
    printf("drmGetVersion() failed..\n");
    exit(1);
  }

  if (!strcmp(version->name, "nvidia-drm")) {
    drm_mode_create_dumb prop;
    memset(&prop, 0, sizeof(drm_mode_create_dumb));
    prop.width = xmodesize;
    prop.height = ymodesize;
    prop.bpp = 32;

    int res = drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &prop);
    if (res) {
      printf("drmIoctl() failed..(%d)\n", res);
      exit(1);
    }

    uint32_t offset = 0;
    res = drmModeAddFB2(drm_fd, xmodesize, ymodesize, DRM_FORMAT_ARGB8888,
                        &(prop.handle), &(prop.pitch), &offset, &fb_id, 0);
    if (res) {
      printf("drmModeAddFB() failed..(%d)\n", res);
      exit(1);
    }
  }

  if (version) {
    drmFreeVersion(version);
    version = NULL;
  }

  // If necessary, set the mode
  if (set_mode) {
    drmModeSetCrtc(drm_fd, drm_crtc_id, fb_id, 0, 0, &drm_conn_id, 1,
                   drm_conn_info->modes + drm_mode_index);
    printf("Set mode\n");
  }

  // If plane is in use, set it
  if (plane >= 0) {
    drmModeSetPlane(drm_fd, drm_plane_id, drm_crtc_id, fb_id, 0, xoffset,
                    yoffset, xsurfsize, ysurfsize, 0, 0, xsurfsize << 16,
                    ysurfsize << 16);
    printf("Set plane configuration\n");
  }

  // Obtain and initialize EGLDisplay
  int attr[] = {EGL_DRM_MASTER_FD_EXT, drm_fd, EGL_NONE};
  eglDisplay =
      eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, (void *)egl_dev, attr);
  if (eglDisplay == EGL_NO_DISPLAY) {
    printf("Couldn't obtain EGLDisplay for device\n");
    exit(8);
  }
  if (!eglInitialize(eglDisplay, &major, &minor)) {
    printf("Couldn't initialize EGLDisplay (error 0x%x)\n", eglGetError());
    exit(8);
  }
  printf("Obtained EGLDisplay\n");

  // Check for stream_consumer_egloutput + output_drm support
  const char *dpy_exts = eglQueryString(eglDisplay, EGL_EXTENSIONS);
  const char *dev_exts = eglQueryDeviceStringEXT(egl_dev, EGL_EXTENSIONS);

  if (!CheckExtension(dpy_exts, "EGL_EXT_output_base")) {
    printf("Missing required extension: EGL_EXT_output_base\n");
    exit(2);
  }

  if (!CheckExtension(dev_exts, "EGL_EXT_device_drm")) {
    printf("Missing required extension: EGL_EXT_device_drm\n");
    exit(2);
  }

  if (!CheckExtension(dpy_exts, "EGL_EXT_output_drm")) {
    printf("Missing required extension: EGL_EXT_output_drm\n");
    exit(2);
  }

  if (!CheckExtension(dpy_exts, "EGL_EXT_stream_consumer_egloutput")) {
    printf("Missing required extension: EGL_EXT_stream_consumer_egloutput\n");
    exit(2);
  }

  // Choose a config and create a context
  EGLint cfg_attr[] = {EGL_SURFACE_TYPE,
                       EGL_STREAM_BIT_KHR,
                       EGL_RENDERABLE_TYPE,
                       EGL_OPENGL_ES2_BIT,
                       EGL_ALPHA_SIZE,
                       1,
                       EGL_NONE};
  if (!eglChooseConfig(eglDisplay, cfg_attr, &egl_cfg, 1, &n) || !n) {
    printf(
        "Unable to obtain config that supports stream rendering (error 0x%x)\n",
        eglGetError());
    exit(9);
  }
  EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};

  eglBindAPI(EGL_OPENGL_ES_API);

  eglContext = eglCreateContext(eglDisplay, egl_cfg, EGL_NO_CONTEXT, ctx_attr);
  if (eglContext == EGL_NO_CONTEXT) {
    printf("Unable to create context (error 0x%x)\n", eglGetError());
    exit(9);
  }
  printf("Obtained EGLConfig and EGLContext\n");

  // Get the layer for this crtc/plane
  EGLAttrib layer_attr[] = {EGL_NONE, EGL_NONE, EGL_NONE};
  if (plane >= 0) {
    layer_attr[0] = EGL_DRM_PLANE_EXT;
    layer_attr[1] = (EGLAttrib)drm_plane_id;
  } else {
    layer_attr[0] = EGL_DRM_CRTC_EXT;
    layer_attr[1] = (EGLAttrib)drm_crtc_id;
  }
  if (!eglGetOutputLayersEXT(eglDisplay, layer_attr, &egl_lyr, 1, &n) || !n) {
    printf("Unable to obtain EGLOutputLayer for %s 0x%x\n",
           (plane >= 0) ? "plane" : "crtc", (int)layer_attr[1]);
    exit(10);
  }
  printf("Obtained EGLOutputLayer\n");

  // Create a stream and connect to the output
  EGLint stream_attr[] = {EGL_STREAM_FIFO_LENGTH_KHR, fifo, EGL_NONE};
  egl_str = eglCreateStreamKHR(eglDisplay, stream_attr);
  if (egl_str == EGL_NO_STREAM_KHR) {
    printf("Unable to create stream (error 0x%x)\n", eglGetError());
    exit(11);
  }
  if (!eglStreamConsumerOutputEXT(eglDisplay, egl_str, egl_lyr)) {
    printf("Unable to connect stream (error 0x%x)\n", eglGetError());
    exit(11);
  }

  // Create a surface to feed the stream
  EGLint srf_attr[] = {EGL_WIDTH, xsurfsize, EGL_HEIGHT, ysurfsize, EGL_NONE};
  eglSurface =
      eglCreateStreamProducerSurfaceKHR(eglDisplay, egl_cfg, egl_str, srf_attr);
  if (eglSurface == EGL_NO_SURFACE) {
    printf("Unable to create rendering surface (error 0x%x)\n", eglGetError());
    exit(12);
  }
  printf("Bound layer to rendering surface\n");

  // Make current
  if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
    printf("Unable to make context/surface current (error 0x%x)\n",
           eglGetError());
    exit(13);
  }

  EGLint Context_RendererType;
  eglQueryContext(eglDisplay, eglContext, EGL_CONTEXT_CLIENT_TYPE,
                  &Context_RendererType);

#if 0     
    switch (Context_RendererType)
      {
      case EGL_OPENGL_API:
	printf("Using OpenGL API\n");
	break;
      case EGL_OPENGL_ES_API:
	printf("Using OpenGL ES API");
	break;
      case 0xB6D185E8:
	printf("Using EGLOutput context.\n");
	break;
      case EGL_OPENVG_API:
	error_exit("Context Query Returned OpenVG. This is Unsupported\n");
      default:
	error_exit("Unknown Context Type. %04X\n", Context_RendererType);
      }
#endif

#if 0  // obtain API function pointers _manually_ (see function pointer
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

void graphics_set_windowtitle(const char *windowname) {
  printf(" Window title would have been: %s\n", windowname);
}

void graphics_swap_buffers() { eglSwapBuffers(eglDisplay, eglSurface); }

void graphics_close_window() {
  if (eglDisplay != EGL_NO_DISPLAY) {
    eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

    if (eglContext != EGL_NO_CONTEXT) eglDestroyContext(eglDisplay, eglContext);

    if (eglSurface != EGL_NO_SURFACE) eglDestroySurface(eglDisplay, eglSurface);

    eglTerminate(eglDisplay);
  }

#if 0 
    if (plane >= 0) 
        drmModeSetPlane(drm_fd, drm_plane_id, drm_crtc_id, 0, 0,
                        xoffset, yoffset, xsurfsize, ysurfsize,
                        0, 0, xsurfsize << 16, ysurfsize << 16);
     else 
        drmModeSetCrtc(drm_fd, drm_crtc_id, 0, xoffset, yoffset, &drm_conn_id, 1,
                       NULL);
#endif

  printf("Released display resources\n");
}
