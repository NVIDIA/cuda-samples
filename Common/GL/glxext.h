#ifndef __glxext_h_
#define __glxext_h_

#ifdef __cplusplus
extern "C" {
#endif

/*
** License Applicability. Except to the extent portions of this file are
** made subject to an alternative license as permitted in the SGI Free
** Software License B, Version 1.1 (the "License"), the contents of this
** file are subject only to the provisions of the License. You may not use
** this file except in compliance with the License. You may obtain a copy
** of the License at Silicon Graphics, Inc., attn: Legal Services, 1600
** Amphitheatre Parkway, Mountain View, CA 94043-1351, or at:
**
** http://oss.sgi.com/projects/FreeB
**
** Note that, as provided in the License, the Software is distributed on an
** "AS IS" basis, with ALL EXPRESS AND IMPLIED WARRANTIES AND CONDITIONS
** DISCLAIMED, INCLUDING, WITHOUT LIMITATION, ANY IMPLIED WARRANTIES AND
** CONDITIONS OF MERCHANTABILITY, SATISFACTORY QUALITY, FITNESS FOR A
** PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
**
** Original Code. The Original Code is: OpenGL Sample Implementation,
** Version 1.2.1, released January 26, 2000, developed by Silicon Graphics,
** Inc. The Original Code is Copyright (c) 1991-2004 Silicon Graphics, Inc.
** Copyright in any portions created by third parties is as indicated
** elsewhere herein. All Rights Reserved.
**
** Additional Notice Provisions: This software was created using the
** OpenGL(R) version 1.2.1 Sample Implementation published by SGI, but has
** not been independently verified as being compliant with the OpenGL(R)
** version 1.2.1 Specification.
*/

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) && !defined(APIENTRY) && !defined(__CYGWIN__) && !defined(__SCITECH_SNAP__)
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif
#ifndef GLAPI
#define GLAPI extern
#endif

/*************************************************************/

/* Header file version number, required by OpenGL ABI for Linux */
/* glxext.h last updated 2005/01/20 */
/* Current version at http://oss.sgi.com/projects/ogl-sample/registry/ */
#define GLX_GLXEXT_VERSION 10

#ifndef GLX_ARB_get_proc_address
#endif

#ifndef GLX_ARB_multisample
#define GLX_SAMPLE_BUFFERS_ARB             100000
#define GLX_SAMPLES_ARB                    100001
#endif

#ifndef GLX_ARB_fbconfig_float
#define GLX_RGBA_FLOAT_TYPE_ARB            0x20B9
#define GLX_RGBA_FLOAT_BIT_ARB             0x00000004
#endif

#ifndef GLX_SGIS_multisample
#define GLX_SAMPLE_BUFFERS_SGIS            100000
#define GLX_SAMPLES_SGIS                   100001
#endif

#ifndef GLX_EXT_visual_info
#define GLX_X_VISUAL_TYPE_EXT              0x22
#define GLX_TRANSPARENT_TYPE_EXT           0x23
#define GLX_TRANSPARENT_INDEX_VALUE_EXT    0x24
#define GLX_TRANSPARENT_RED_VALUE_EXT      0x25
#define GLX_TRANSPARENT_GREEN_VALUE_EXT    0x26
#define GLX_TRANSPARENT_BLUE_VALUE_EXT     0x27
#define GLX_TRANSPARENT_ALPHA_VALUE_EXT    0x28
#define GLX_NONE_EXT                       0x8000
#define GLX_TRUE_COLOR_EXT                 0x8002
#define GLX_DIRECT_COLOR_EXT               0x8003
#define GLX_PSEUDO_COLOR_EXT               0x8004
#define GLX_STATIC_COLOR_EXT               0x8005
#define GLX_GRAY_SCALE_EXT                 0x8006
#define GLX_STATIC_GRAY_EXT                0x8007
#define GLX_TRANSPARENT_RGB_EXT            0x8008
#define GLX_TRANSPARENT_INDEX_EXT          0x8009
#endif

#ifndef GLX_SGI_swap_control
#endif

#ifndef GLX_SGI_video_sync
#endif

#ifndef GLX_SGI_make_current_read
#endif

#ifndef GLX_SGIX_video_source
#endif

#ifndef GLX_EXT_visual_rating
#define GLX_VISUAL_CAVEAT_EXT              0x20
#define GLX_SLOW_VISUAL_EXT                0x8001
#define GLX_NON_CONFORMANT_VISUAL_EXT      0x800D
/* reuse GLX_NONE_EXT */
#endif

#ifndef GLX_EXT_import_context
#define GLX_SHARE_CONTEXT_EXT              0x800A
#define GLX_VISUAL_ID_EXT                  0x800B
#define GLX_SCREEN_EXT                     0x800C
#endif

#ifndef GLX_SGIX_fbconfig
#define GLX_WINDOW_BIT_SGIX                0x00000001
#define GLX_PIXMAP_BIT_SGIX                0x00000002
#define GLX_RGBA_BIT_SGIX                  0x00000001
#define GLX_COLOR_INDEX_BIT_SGIX           0x00000002
#define GLX_DRAWABLE_TYPE_SGIX             0x8010
#define GLX_RENDER_TYPE_SGIX               0x8011
#define GLX_X_RENDERABLE_SGIX              0x8012
#define GLX_FBCONFIG_ID_SGIX               0x8013
#define GLX_RGBA_TYPE_SGIX                 0x8014
#define GLX_COLOR_INDEX_TYPE_SGIX          0x8015
/* reuse GLX_SCREEN_EXT */
#endif

#ifndef GLX_SGIX_pbuffer
#define GLX_PBUFFER_BIT_SGIX               0x00000004
#define GLX_BUFFER_CLOBBER_MASK_SGIX       0x08000000
#define GLX_FRONT_LEFT_BUFFER_BIT_SGIX     0x00000001
#define GLX_FRONT_RIGHT_BUFFER_BIT_SGIX    0x00000002
#define GLX_BACK_LEFT_BUFFER_BIT_SGIX      0x00000004
#define GLX_BACK_RIGHT_BUFFER_BIT_SGIX     0x00000008
#define GLX_AUX_BUFFERS_BIT_SGIX           0x00000010
#define GLX_DEPTH_BUFFER_BIT_SGIX          0x00000020
#define GLX_STENCIL_BUFFER_BIT_SGIX        0x00000040
#define GLX_ACCUM_BUFFER_BIT_SGIX          0x00000080
#define GLX_SAMPLE_BUFFERS_BIT_SGIX        0x00000100
#define GLX_MAX_PBUFFER_WIDTH_SGIX         0x8016
#define GLX_MAX_PBUFFER_HEIGHT_SGIX        0x8017
#define GLX_MAX_PBUFFER_PIXELS_SGIX        0x8018
#define GLX_OPTIMAL_PBUFFER_WIDTH_SGIX     0x8019
#define GLX_OPTIMAL_PBUFFER_HEIGHT_SGIX    0x801A
#define GLX_PRESERVED_CONTENTS_SGIX        0x801B
#define GLX_LARGEST_PBUFFER_SGIX           0x801C
#define GLX_WIDTH_SGIX                     0x801D
#define GLX_HEIGHT_SGIX                    0x801E
#define GLX_EVENT_MASK_SGIX                0x801F
#define GLX_DAMAGED_SGIX                   0x8020
#define GLX_SAVED_SGIX                     0x8021
#define GLX_WINDOW_SGIX                    0x8022
#define GLX_PBUFFER_SGIX                   0x8023
#endif

#ifndef GLX_SGI_cushion
#endif

#ifndef GLX_SGIX_video_resize
#define GLX_SYNC_FRAME_SGIX                0x00000000
#define GLX_SYNC_SWAP_SGIX                 0x00000001
#endif

#ifndef GLX_SGIX_dmbuffer
#define GLX_DIGITAL_MEDIA_PBUFFER_SGIX     0x8024
#endif

#ifndef GLX_SGIX_swap_group
#endif

#ifndef GLX_SGIX_swap_barrier
#endif

#ifndef GLX_SGIS_blended_overlay
#define GLX_BLENDED_RGBA_SGIS              0x8025
#endif

#ifndef GLX_SGIS_shared_multisample
#define GLX_MULTISAMPLE_SUB_RECT_WIDTH_SGIS 0x8026
#define GLX_MULTISAMPLE_SUB_RECT_HEIGHT_SGIS 0x8027
#endif

#ifndef GLX_SUN_get_transparent_index
#endif

#ifndef GLX_3DFX_multisample
#define GLX_SAMPLE_BUFFERS_3DFX            0x8050
#define GLX_SAMPLES_3DFX                   0x8051
#endif

#ifndef GLX_MESA_copy_sub_buffer
#endif

#ifndef GLX_MESA_pixmap_colormap
#endif

#ifndef GLX_MESA_release_buffers
#endif

#ifndef GLX_MESA_set_3dfx_mode
#define GLX_3DFX_WINDOW_MODE_MESA          0x1
#define GLX_3DFX_FULLSCREEN_MODE_MESA      0x2
#endif

#ifndef GLX_SGIX_visual_select_group
#define GLX_VISUAL_SELECT_GROUP_SGIX       0x8028
#endif

#ifndef GLX_OML_swap_method
#define GLX_SWAP_METHOD_OML                0x8060
#define GLX_SWAP_EXCHANGE_OML              0x8061
#define GLX_SWAP_COPY_OML                  0x8062
#define GLX_SWAP_UNDEFINED_OML             0x8063
#endif

#ifndef GLX_OML_sync_control
#endif

#ifndef GLX_NV_float_buffer
#define GLX_FLOAT_COMPONENTS_NV            0x20B0
#endif

#ifndef GLX_SGIX_hyperpipe
#define GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX 80
#define GLX_BAD_HYPERPIPE_CONFIG_SGIX      91
#define GLX_BAD_HYPERPIPE_SGIX             92
#define GLX_HYPERPIPE_DISPLAY_PIPE_SGIX    0x00000001
#define GLX_HYPERPIPE_RENDER_PIPE_SGIX     0x00000002
#define GLX_PIPE_RECT_SGIX                 0x00000001
#define GLX_PIPE_RECT_LIMITS_SGIX          0x00000002
#define GLX_HYPERPIPE_STEREO_SGIX          0x00000003
#define GLX_HYPERPIPE_PIXEL_AVERAGE_SGIX   0x00000004
#define GLX_HYPERPIPE_ID_SGIX              0x8030
#endif

#ifndef GLX_MESA_agp_offset
#endif


/*************************************************************/

#ifndef GLX_ARB_get_proc_address
/*
 * Linux OpenGL ABI specifies glXGetProcAddressARB should be
 * in glx.h moving related defines there as well.
 */
#endif

#ifndef GLX_SGIX_video_source
typedef XID GLXVideoSourceSGIX;
#endif

#ifndef GLX_SGIX_fbconfig
typedef XID GLXFBConfigIDSGIX;
typedef struct __GLXFBConfigRec *GLXFBConfigSGIX;
#endif

#ifndef GLX_SGIX_pbuffer
typedef XID GLXPbufferSGIX;
typedef struct
{
    int type;
    unsigned long serial;     /* # of last request processed by server */
    Bool send_event;          /* true if this came for SendEvent request */
    Display *display;         /* display the event was read from */
    GLXDrawable drawable;     /* i.d. of Drawable */
    int event_type;       /* GLX_DAMAGED_SGIX or GLX_SAVED_SGIX */
    int draw_type;        /* GLX_WINDOW_SGIX or GLX_PBUFFER_SGIX */
    unsigned int mask;    /* mask indicating which buffers are affected*/
    int x, y;
    int width, height;
    int count;        /* if nonzero, at least this many more */
} GLXBufferClobberEventSGIX;
#endif

#ifndef GLX_NV_swap_group
#endif

#ifndef GLX_NV_video_out
/*
 * GLXVideoDeviceNV is an opaque handle to a video device (part of the
 * GLX_NV_video_out extension).
 */
typedef unsigned int GLXVideoDeviceNV;

/* glXBindVideoImageNV iVideoBuffer values (NV_video_out) */
#define GLX_VIDEO_OUT_COLOR_NV           0x20C3
#define GLX_VIDEO_OUT_ALPHA_NV           0x20C4
#define GLX_VIDEO_OUT_DEPTH_NV           0x20C5
#define GLX_VIDEO_OUT_COLOR_AND_ALPHA_NV 0x20C6
#define GLX_VIDEO_OUT_COLOR_AND_DEPTH_NV 0x20C7

/* glXSendPbufferToVideoNV iBufferType values (NV_video_out) */
#define GLX_VIDEO_OUT_FRAME_NV           0x20C8
#define GLX_VIDEO_OUT_FIELD_1_NV         0x20C9
#define GLX_VIDEO_OUT_FIELD_2_NV         0x20CA
#endif

#ifndef GLX_EXT_texture_from_pixmap
/* New glXGetFBConfigAttrib <attrib_list> tokens */
#define GLX_BIND_TO_TEXTURE_RGB_EXT      0x20D0
#define GLX_BIND_TO_TEXTURE_RGBA_EXT     0x20D1
#define GLX_BIND_TO_MIPMAP_TEXTURE_EXT   0x20D2
#define GLX_BIND_TO_TEXTURE_TARGETS_EXT  0x20D3
#define GLX_Y_INVERTED_EXT               0x20D4

/* New glXCreatePixmap attributes and glXQueryDrawable attributes */
#define GLX_TEXTURE_FORMAT_EXT           0x20D5
#define GLX_TEXTURE_TARGET_EXT           0x20D6
#define GLX_MIPMAP_TEXTURE_EXT           0x20D7

/* Values for GLX_TEXTURE_FORMAT_EXT */
#define GLX_TEXTURE_FORMAT_NONE_EXT      0x20D8
#define GLX_TEXTURE_FORMAT_RGB_EXT       0x20D9
#define GLX_TEXTURE_FORMAT_RGBA_EXT      0x20DA

/* Bits for GLX_BIND_TO_TEXTURE_TARGETS_EXT mask */
#define GLX_TEXTURE_1D_BIT_EXT           0x00000001
#define GLX_TEXTURE_2D_BIT_EXT           0x00000002
#define GLX_TEXTURE_RECTANGLE_BIT_EXT    0x00000004

/* Values for GLX_TEXTURE_TARGET_EXT */
#define GLX_TEXTURE_1D_EXT               0x20DB
#define GLX_TEXTURE_2D_EXT               0x20DC
#define GLX_TEXTURE_RECTANGLE_EXT        0x20DD

/*
 * Values for the buffer parameter of glXBindTexImageEXT and
 * glXReleaseTexImageEXT
 */
#define GLX_FRONT_LEFT_EXT               0x20DE
#define GLX_FRONT_RIGHT_EXT              0x20DF
#define GLX_BACK_LEFT_EXT                0x20E0
#define GLX_BACK_RIGHT_EXT               0x20E1
#define GLX_FRONT_EXT                    GLX_FRONT_LEFT_EXT
#define GLX_BACK_EXT                     GLX_BACK_LEFT_EXT
#define GLX_AUX0_EXT                     0x20E2
#define GLX_AUX1_EXT                     0x20E3
#define GLX_AUX2_EXT                     0x20E4
#define GLX_AUX3_EXT                     0x20E5
#define GLX_AUX4_EXT                     0x20E6
#define GLX_AUX5_EXT                     0x20E7
#define GLX_AUX6_EXT                     0x20E8
#define GLX_AUX7_EXT                     0x20E9
#define GLX_AUX8_EXT                     0x20EA
#define GLX_AUX9_EXT                     0x20EB

#endif

/* Define int32_t and int64_t types for UST/MSC */
/* (as used in the GLX_OML_sync_control extension). */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#include <inttypes.h>
#elif defined( __VMS ) || defined(__FreeBSD__)
#include <inttypes.h>
#elif (defined(__sun__) && defined(__svr4__)) || (defined(__sun) && defined(__SVR4))
#include <inttypes.h>
#elif defined(__SCO__) || defined(__USLC__) || defined(__linux__)
#include <stdint.h>
#elif defined(__UNIXOS2__) || defined(__SOL64__)
typedef long int int32_t;
typedef long long int int64_t;
#else
#error "int32_t and int64_t are undefined!"
#endif

#ifndef GLX_ARB_get_proc_address
/* Moved to glx.h */
#endif

#ifndef GLX_ARB_multisample
#define GLX_ARB_multisample 1
#endif

#ifndef GLX_ARB_fbconfig_float
#define GLX_ARB_fbconfig_float 1
#endif

#ifndef GLX_SGIS_multisample
#define GLX_SGIS_multisample 1
#endif

#ifndef GLX_EXT_visual_info
#define GLX_EXT_visual_info 1
#endif

#ifndef GLX_SGI_swap_control
#define GLX_SGI_swap_control 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern int glXSwapIntervalSGI(int);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef int (* PFNGLXSWAPINTERVALSGIPROC)(int interval);
#endif

#ifndef GLX_SGI_video_sync
#define GLX_SGI_video_sync 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern int glXGetVideoSyncSGI(unsigned int *);
extern int glXWaitVideoSyncSGI(int, int, unsigned int *);
extern int glXGetRefreshRateSGI(unsigned int *);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef int (* PFNGLXGETVIDEOSYNCSGIPROC)(unsigned int *count);
typedef int (* PFNGLXWAITVIDEOSYNCSGIPROC)(int divisor, int remainder, unsigned int *count);
typedef int (* PFNGLXGETREFRESHRATESGIPROC)(unsigned int *);
#endif

#ifndef GLX_SGI_make_current_read
#define GLX_SGI_make_current_read 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern Bool glXMakeCurrentReadSGI(Display *, GLXDrawable, GLXDrawable, GLXContext);
extern GLXDrawable glXGetCurrentReadDrawableSGI(void);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef Bool(* PFNGLXMAKECURRENTREADSGIPROC)(Display *dpy, GLXDrawable draw, GLXDrawable read, GLXContext ctx);
typedef GLXDrawable(* PFNGLXGETCURRENTREADDRAWABLESGIPROC)(void);
#endif

#ifndef GLX_SGIX_video_source
#define GLX_SGIX_video_source 1
#ifdef _VL_H
#ifdef GLX_GLXEXT_PROTOTYPES
extern GLXVideoSourceSGIX glXCreateGLXVideoSourceSGIX(Display *, int, VLServer, VLPath, int, VLNode);
extern void glXDestroyGLXVideoSourceSGIX(Display *, GLXVideoSourceSGIX);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef GLXVideoSourceSGIX(* PFNGLXCREATEGLXVIDEOSOURCESGIXPROC)(Display *display, int screen, VLServer server, VLPath path, int nodeClass, VLNode drainNode);
typedef void (* PFNGLXDESTROYGLXVIDEOSOURCESGIXPROC)(Display *dpy, GLXVideoSourceSGIX glxvideosource);
#endif /* _VL_H */
#endif

#ifndef GLX_EXT_visual_rating
#define GLX_EXT_visual_rating 1
#endif

#ifndef GLX_EXT_import_context
#define GLX_EXT_import_context 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern Display *glXGetCurrentDisplayEXT(void);
extern int glXQueryContextInfoEXT(Display *, GLXContext, int, int *);
extern GLXContextID glXGetContextIDEXT(const GLXContext);
extern GLXContext glXImportContextEXT(Display *, GLXContextID);
extern void glXFreeContextEXT(Display *, GLXContext);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef Display *(* PFNGLXGETCURRENTDISPLAYEXTPROC)(void);
typedef int (* PFNGLXQUERYCONTEXTINFOEXTPROC)(Display *dpy, GLXContext context, int attribute, int *value);
typedef GLXContextID(* PFNGLXGETCONTEXTIDEXTPROC)(const GLXContext context);
typedef GLXContext(* PFNGLXIMPORTCONTEXTEXTPROC)(Display *dpy, GLXContextID contextID);
typedef void (* PFNGLXFREECONTEXTEXTPROC)(Display *dpy, GLXContext context);
#endif

#ifndef GLX_SGIX_fbconfig
#define GLX_SGIX_fbconfig 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern int glXGetFBConfigAttribSGIX(Display *, GLXFBConfigSGIX, int, int *);
extern GLXFBConfigSGIX *glXChooseFBConfigSGIX(Display *, int, int *, int *);
extern GLXPixmap glXCreateGLXPixmapWithConfigSGIX(Display *, GLXFBConfigSGIX, Pixmap);
extern GLXContext glXCreateContextWithConfigSGIX(Display *, GLXFBConfigSGIX, int, GLXContext, Bool);
extern XVisualInfo *glXGetVisualFromFBConfigSGIX(Display *, GLXFBConfigSGIX);
extern GLXFBConfigSGIX glXGetFBConfigFromVisualSGIX(Display *, XVisualInfo *);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef int (* PFNGLXGETFBCONFIGATTRIBSGIXPROC)(Display *dpy, GLXFBConfigSGIX config, int attribute, int *value);
typedef GLXFBConfigSGIX *(* PFNGLXCHOOSEFBCONFIGSGIXPROC)(Display *dpy, int screen, int *attrib_list, int *nelements);
typedef GLXPixmap(* PFNGLXCREATEGLXPIXMAPWITHCONFIGSGIXPROC)(Display *dpy, GLXFBConfigSGIX config, Pixmap pixmap);
typedef GLXContext(* PFNGLXCREATECONTEXTWITHCONFIGSGIXPROC)(Display *dpy, GLXFBConfigSGIX config, int render_type, GLXContext share_list, Bool direct);
typedef XVisualInfo *(* PFNGLXGETVISUALFROMFBCONFIGSGIXPROC)(Display *dpy, GLXFBConfigSGIX config);
typedef GLXFBConfigSGIX(* PFNGLXGETFBCONFIGFROMVISUALSGIXPROC)(Display *dpy, XVisualInfo *vis);
#endif

#ifndef GLX_SGIX_pbuffer
#define GLX_SGIX_pbuffer 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern GLXPbufferSGIX glXCreateGLXPbufferSGIX(Display *, GLXFBConfigSGIX, unsigned int, unsigned int, int *);
extern void glXDestroyGLXPbufferSGIX(Display *, GLXPbufferSGIX);
extern int glXQueryGLXPbufferSGIX(Display *, GLXPbufferSGIX, int, unsigned int *);
extern void glXSelectEventSGIX(Display *, GLXDrawable, unsigned long);
extern void glXGetSelectedEventSGIX(Display *, GLXDrawable, unsigned long *);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef GLXPbufferSGIX(* PFNGLXCREATEGLXPBUFFERSGIXPROC)(Display *dpy, GLXFBConfigSGIX config, unsigned int width, unsigned int height, int *attrib_list);
typedef void (* PFNGLXDESTROYGLXPBUFFERSGIXPROC)(Display *dpy, GLXPbufferSGIX pbuf);
typedef int (* PFNGLXQUERYGLXPBUFFERSGIXPROC)(Display *dpy, GLXPbufferSGIX pbuf, int attribute, unsigned int *value);
typedef void (* PFNGLXSELECTEVENTSGIXPROC)(Display *dpy, GLXDrawable drawable, unsigned long mask);
typedef void (* PFNGLXGETSELECTEDEVENTSGIXPROC)(Display *dpy, GLXDrawable drawable, unsigned long *mask);
#endif

#ifndef GLX_SGI_cushion
#define GLX_SGI_cushion 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern void glXCushionSGI(Display *, Window, float);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef void (* PFNGLXCUSHIONSGIPROC)(Display *dpy, Window window, float cushion);
#endif

#ifndef GLX_SGIX_video_resize
#define GLX_SGIX_video_resize 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern int glXBindChannelToWindowSGIX(Display *, int, int, Window);
extern int glXChannelRectSGIX(Display *, int, int, int, int, int, int);
extern int glXQueryChannelRectSGIX(Display *, int, int, int *, int *, int *, int *);
extern int glXQueryChannelDeltasSGIX(Display *, int, int, int *, int *, int *, int *);
extern int glXChannelRectSyncSGIX(Display *, int, int, GLenum);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef int (* PFNGLXBINDCHANNELTOWINDOWSGIXPROC)(Display *display, int screen, int channel, Window window);
typedef int (* PFNGLXCHANNELRECTSGIXPROC)(Display *display, int screen, int channel, int x, int y, int w, int h);
typedef int (* PFNGLXQUERYCHANNELRECTSGIXPROC)(Display *display, int screen, int channel, int *dx, int *dy, int *dw, int *dh);
typedef int (* PFNGLXQUERYCHANNELDELTASSGIXPROC)(Display *display, int screen, int channel, int *x, int *y, int *w, int *h);
typedef int (* PFNGLXCHANNELRECTSYNCSGIXPROC)(Display *display, int screen, int channel, GLenum synctype);
#endif

#ifndef GLX_SGIX_dmbuffer
#define GLX_SGIX_dmbuffer 1
#ifdef _DM_BUFFER_H_
#ifdef GLX_GLXEXT_PROTOTYPES
extern Bool glXAssociateDMPbufferSGIX(Display *, GLXPbufferSGIX, DMparams *, DMbuffer);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef Bool(* PFNGLXASSOCIATEDMPBUFFERSGIXPROC)(Display *dpy, GLXPbufferSGIX pbuffer, DMparams *params, DMbuffer dmbuffer);
#endif /* _DM_BUFFER_H_ */
#endif

#ifndef GLX_SGIX_swap_group
#define GLX_SGIX_swap_group 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern void glXJoinSwapGroupSGIX(Display *, GLXDrawable, GLXDrawable);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef void (* PFNGLXJOINSWAPGROUPSGIXPROC)(Display *dpy, GLXDrawable drawable, GLXDrawable member);
#endif

#ifndef GLX_SGIX_swap_barrier
#define GLX_SGIX_swap_barrier 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern void glXBindSwapBarrierSGIX(Display *, GLXDrawable, int);
extern Bool glXQueryMaxSwapBarriersSGIX(Display *, int, int *);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef void (* PFNGLXBINDSWAPBARRIERSGIXPROC)(Display *dpy, GLXDrawable drawable, int barrier);
typedef Bool(* PFNGLXQUERYMAXSWAPBARRIERSSGIXPROC)(Display *dpy, int screen, int *max);
#endif

#ifndef GLX_SUN_get_transparent_index
#define GLX_SUN_get_transparent_index 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern Status glXGetTransparentIndexSUN(Display *, Window, Window, long *);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef Status(* PFNGLXGETTRANSPARENTINDEXSUNPROC)(Display *dpy, Window overlay, Window underlay, long *pTransparentIndex);
#endif

#ifndef GLX_MESA_copy_sub_buffer
#define GLX_MESA_copy_sub_buffer 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern void glXCopySubBufferMESA(Display *, GLXDrawable, int, int, int, int);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef void (* PFNGLXCOPYSUBBUFFERMESAPROC)(Display *dpy, GLXDrawable drawable, int x, int y, int width, int height);
#endif

#ifndef GLX_MESA_pixmap_colormap
#define GLX_MESA_pixmap_colormap 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern GLXPixmap glXCreateGLXPixmapMESA(Display *, XVisualInfo *, Pixmap, Colormap);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef GLXPixmap(* PFNGLXCREATEGLXPIXMAPMESAPROC)(Display *dpy, XVisualInfo *visual, Pixmap pixmap, Colormap cmap);
#endif

#ifndef GLX_MESA_release_buffers
#define GLX_MESA_release_buffers 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern Bool glXReleaseBuffersMESA(Display *, GLXDrawable);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef Bool(* PFNGLXRELEASEBUFFERSMESAPROC)(Display *dpy, GLXDrawable drawable);
#endif

#ifndef GLX_MESA_set_3dfx_mode
#define GLX_MESA_set_3dfx_mode 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern Bool glXSet3DfxModeMESA(int);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef Bool(* PFNGLXSET3DFXMODEMESAPROC)(int mode);
#endif

#ifndef GLX_SGIX_visual_select_group
#define GLX_SGIX_visual_select_group 1
#endif

#ifndef GLX_OML_swap_method
#define GLX_OML_swap_method 1
#endif

#ifndef GLX_OML_sync_control
#define GLX_OML_sync_control 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern Bool glXGetSyncValuesOML(Display *, GLXDrawable, int64_t *, int64_t *, int64_t *);
extern Bool glXGetMscRateOML(Display *, GLXDrawable, int32_t *, int32_t *);
extern int64_t glXSwapBuffersMscOML(Display *, GLXDrawable, int64_t, int64_t, int64_t);
extern Bool glXWaitForMscOML(Display *, GLXDrawable, int64_t, int64_t, int64_t, int64_t *, int64_t *, int64_t *);
extern Bool glXWaitForSbcOML(Display *, GLXDrawable, int64_t, int64_t *, int64_t *, int64_t *);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef Bool(* PFNGLXGETSYNCVALUESOMLPROC)(Display *dpy, GLXDrawable drawable, int64_t *ust, int64_t *msc, int64_t *sbc);
typedef Bool(* PFNGLXGETMSCRATEOMLPROC)(Display *dpy, GLXDrawable drawable, int32_t *numerator, int32_t *denominator);
typedef int64_t (* PFNGLXSWAPBUFFERSMSCOMLPROC)(Display *dpy, GLXDrawable drawable, int64_t target_msc, int64_t divisor, int64_t remainder);
typedef Bool(* PFNGLXWAITFORMSCOMLPROC)(Display *dpy, GLXDrawable drawable, int64_t target_msc, int64_t divisor, int64_t remainder, int64_t *ust, int64_t *msc, int64_t *sbc);
typedef Bool(* PFNGLXWAITFORSBCOMLPROC)(Display *dpy, GLXDrawable drawable, int64_t target_sbc, int64_t *ust, int64_t *msc, int64_t *sbc);
#endif

#ifndef GLX_NV_float_buffer
#define GLX_NV_float_buffer 1
#endif

#ifndef GLX_SGIX_hyperpipe
#define GLX_SGIX_hyperpipe 1

typedef struct
{
    char    pipeName[GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX];
    int     networkId;
} GLXHyperpipeNetworkSGIX;

typedef struct
{
    char    pipeName[GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX];
    int     channel;
    unsigned int
    participationType;
    int     timeSlice;
} GLXHyperpipeConfigSGIX;

typedef struct
{
    char pipeName[GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX];
    int srcXOrigin, srcYOrigin, srcWidth, srcHeight;
    int destXOrigin, destYOrigin, destWidth, destHeight;
} GLXPipeRect;

typedef struct
{
    char pipeName[GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX];
    int XOrigin, YOrigin, maxHeight, maxWidth;
} GLXPipeRectLimits;

#ifdef GLX_GLXEXT_PROTOTYPES
extern GLXHyperpipeNetworkSGIX *glXQueryHyperpipeNetworkSGIX(Display *, int *);
extern int glXHyperpipeConfigSGIX(Display *, int, int, GLXHyperpipeConfigSGIX *, int *);
extern GLXHyperpipeConfigSGIX *glXQueryHyperpipeConfigSGIX(Display *, int, int *);
extern int glXDestroyHyperpipeConfigSGIX(Display *, int);
extern int glXBindHyperpipeSGIX(Display *, int);
extern int glXQueryHyperpipeBestAttribSGIX(Display *, int, int, int, void *, void *);
extern int glXHyperpipeAttribSGIX(Display *, int, int, int, void *);
extern int glXQueryHyperpipeAttribSGIX(Display *, int, int, int, void *);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef GLXHyperpipeNetworkSGIX *(* PFNGLXQUERYHYPERPIPENETWORKSGIXPROC)(Display *dpy, int *npipes);
typedef int (* PFNGLXHYPERPIPECONFIGSGIXPROC)(Display *dpy, int networkId, int npipes, GLXHyperpipeConfigSGIX *cfg, int *hpId);
typedef GLXHyperpipeConfigSGIX *(* PFNGLXQUERYHYPERPIPECONFIGSGIXPROC)(Display *dpy, int hpId, int *npipes);
typedef int (* PFNGLXDESTROYHYPERPIPECONFIGSGIXPROC)(Display *dpy, int hpId);
typedef int (* PFNGLXBINDHYPERPIPESGIXPROC)(Display *dpy, int hpId);
typedef int (* PFNGLXQUERYHYPERPIPEBESTATTRIBSGIXPROC)(Display *dpy, int timeSlice, int attrib, int size, void *attribList, void *returnAttribList);
typedef int (* PFNGLXHYPERPIPEATTRIBSGIXPROC)(Display *dpy, int timeSlice, int attrib, int size, void *attribList);
typedef int (* PFNGLXQUERYHYPERPIPEATTRIBSGIXPROC)(Display *dpy, int timeSlice, int attrib, int size, void *returnAttribList);
#endif

#ifndef GLX_MESA_agp_offset
#define GLX_MESA_agp_offset 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern unsigned int glXGetAGPOffsetMESA(const void *);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef unsigned int (* PFNGLXGETAGPOFFSETMESAPROC)(const void *pointer);
#endif

/*
 * GLX_NV_vertex_array_range is not a real extension name...
 */
#ifndef GLX_NV_vertex_array_range
#define GLX_NV_vertex_array_range 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern void *glXAllocateMemoryNV(GLsizei size, GLfloat readfreq,
                                 GLfloat writefreq, GLfloat priority);

extern void glXFreeMemoryNV(GLvoid *pointer);
#endif
typedef void *(* PFNGLXALLOCATEMEMORYNVPROC)(GLsizei size,
                                             GLfloat readfreq,
                                             GLfloat writefreq,
                                             GLfloat priority);

typedef void (* PFNGLXFREEMEMORYNVPROC)(GLvoid *pointer);
#endif

#ifndef GLX_NV_swap_group
#define GLX_NV_swap_group 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern Bool glXJoinSwapGroupNV(Display *dpy, GLXDrawable drawable,
                               GLuint group);

extern Bool glXBindSwapBarrierNV(Display *dpy, GLuint group, GLuint barrier);

extern Bool glXQuerySwapGroupNV(Display *dpy, GLXDrawable drawable,
                                GLuint *group, GLuint *barrier);

extern Bool glXQueryMaxSwapGroupsNV(Display *dpy, int screen,
                                    GLuint *maxGroups, GLuint *maxBarriers);

extern Bool glXQueryFrameCountNV(Display *dpy, int screen, GLuint *count);

extern Bool glXResetFrameCountNV(Display *dpy, int screen);
#endif
typedef Bool(* PFNGLXJOINSWAPGROUPNVPROC)(Display *dpy,
                                          GLXDrawable drawable,
                                          GLuint group);

typedef Bool(* PFNGLXBINDSWAPBARRIERNVPROC)(Display *dpy,
                                            GLuint group,
                                            GLuint barrier);

typedef Bool(* PFNGLXQUERYSWAPGROUPNVPROC)(Display *dpy,
                                           GLXDrawable drawable,
                                           GLuint *group,
                                           GLuint *barrier);

typedef Bool(* PFNGLXQUERYMAXSWAPGROUPSNVPROC)(Display *dpy,
                                               int screen,
                                               GLuint *maxGroups,
                                               GLuint *maxBarriers);

typedef Bool(* PFNGLXQUERYFRAMECOUNTNVPROC)(Display *dpy,
                                            int screen,
                                            GLuint *count);

typedef Bool(* PFNGLXRESETFRAMECOUNTNVPROC)(Display *dpy, int screen);
#endif

#ifndef GLX_NV_video_out
#define GLX_NV_video_out 1
#ifdef GLX_GLXEXT_PROTOTYPES
extern int glXGetVideoDeviceNV(Display *dpy, int screen, int numVideoDevices,
                               GLXVideoDeviceNV *pVideoDevice);

extern int glXReleaseVideoDeviceNV(Display *dpy, int screen,
                                   GLXVideoDeviceNV VideoDevice);

extern int glXBindVideoImageNV(Display *dpy, GLXVideoDeviceNV VideoDevice,
                               GLXPbuffer pbuf, int iVideoBuffer);

extern int glXReleaseVideoImageNV(Display *dpy, GLXPbuffer pbuf);

extern int glXSendPbufferToVideoNV(Display *dpy, GLXPbuffer pbuf,
                                   int iBufferType,
                                   unsigned long *pulCounterPbuffer,
                                   GLboolean bBlock);

extern int glXGetVideoInfoNV(Display *dpy, int screen,
                             GLXVideoDeviceNV VideoDevice,
                             unsigned long *pulCounterOutputVideo,
                             unsigned long *pulCounterOutputPbuffer);
#endif
typedef int (* PFNGLXGETVIDEODEVICENVPROC)(Display *dpy,
                                           int screen,
                                           int numVideoDevices,
                                           GLXVideoDeviceNV *pVideoDevice);

typedef int (* PFNGLXRELEASEVIDEODEVICENVPROC)(Display *dpy,
                                               int screen,
                                               GLXVideoDeviceNV VideoDevice);

typedef int (* PFNGLXBINDVIDEOIMAGENVPROC)(Display *dpy,
                                           GLXVideoDeviceNV VideoDevice,
                                           GLXPbuffer pbuf,
                                           int iVideoBuffer);

typedef int (* PFNGLXRELEASEVIDEOIMAGENVPROC)(Display *dpy,
                                              GLXPbuffer pbuf);

typedef int (* PFNGLXSENDPBUFFERTOVIDEONVPROC)(Display *dpy,
                                               GLXPbuffer pbuf,
                                               int iBufferType,
                                               unsigned long *pulCounterPbuffer,
                                               GLboolean bBlock);

typedef int (* PFNGLXGETVIDEOINFONVPROC)(Display *dpy, int screen,
                                         GLXVideoDeviceNV VideoDevice,
                                         unsigned long *pulCounterOutputVideo,
                                         unsigned long *pulCounterOutputPbuffer);
#endif

#ifndef GLX_EXT_texture_from_pixmap
#define GLX_EXT_texture_from_pixmap
#ifdef GLX_GLXEXT_PROTOTYPES
extern void glXBindTexImageEXT(Display *dpy, GLXDrawable drawable,
                               int buffer, const int *attrib_list);
extern void glXReleaseTexImageEXT(Display *dpy, GLXDrawable drawable,
                                  int buffer);
#endif
typedef void (* PFNGLXBINDTEXIMAGEEXTPROC)(Display *dpy,
                                           GLXDrawable drawable,
                                           int buffer,
                                           const int *attrib_list);
typedef void (* PFNGLXRELEASETEXIMAGEEXTPROC)(Display *dpy,
                                              GLXDrawable drawable,
                                              int buffer);
#endif

#ifdef __cplusplus
}
#endif

#endif
