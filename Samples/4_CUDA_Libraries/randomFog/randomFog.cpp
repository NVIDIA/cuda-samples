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

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA Library Headers
#include <curand.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <rendercheck_gl.h>

// System includes
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <math.h>

// Includes
#include "rng.h"

// standard utility and system includes
#include <helper_timer.h>

// SDK information
static const char *printfFile = "randomFog.txt";

// RNG instance
RNG *g_pRng = NULL;

// CheckRender instance (for QA)
CheckRender *g_pCheckRender = NULL;

// Simple struct which contains the position and color of a vertex
struct SVertex {
  GLfloat x, y, z;
  GLfloat r, g, b;
};

// Data for the vertices
SVertex *g_pVertices = NULL;
int g_nVertices;           // Size of the vertex array
int g_nVerticesPopulated;  // Number currently populated

// Control the randomness
int nSkip1 = 0;  // Number of samples to discard between x,y
int nSkip2 = 0;  // Number of samples to discard between y,z
int nSkip3 = 0;  // Number of samples to discard between z,x

// Control the display
enum Shape_t { Sphere, SphericalShell, Cube, Plane };
Shape_t g_currentShape = Sphere;
bool g_bShowAxes = true;
bool g_bTenXZoom = false;
bool g_bAutoRotate = true;
int g_lastShapeX = 1024;
int g_lastShapeY = 1024;
float g_xRotated = 0.0f;
float g_yRotated = 0.0f;

const float PI = 3.14159265359f;

void createCube(void) {
  int startVertex = 0;

  for (int i = startVertex; i < g_nVerticesPopulated; i++) {
    g_pVertices[i].x = (g_pRng->getNextU01() - .5f) * 2;

    for (int j = 0; j < nSkip1; j++) {
      g_pRng->getNextU01();
    }

    g_pVertices[i].y = (g_pRng->getNextU01() - .5f) * 2;

    for (int j = 0; j < nSkip2; j++) {
      g_pRng->getNextU01();
    }

    g_pVertices[i].z = (g_pRng->getNextU01() - .5f) * 2;

    for (int j = 0; j < nSkip3; j++) {
      g_pRng->getNextU01();
    }

    g_pVertices[i].r = 1.0f;
    g_pVertices[i].g = 1.0f;
    g_pVertices[i].b = 1.0f;
  }
}

void createPlane(void) {
  int startVertex = 0;

  for (int i = startVertex; i < g_nVerticesPopulated; i++) {
    g_pVertices[i].x = (g_pRng->getNextU01() - .5f) * 2;

    for (int j = 0; j < nSkip1; j++) {
      g_pRng->getNextU01();
    }

    g_pVertices[i].y = (g_pRng->getNextU01() - .5f) * 2;

    for (int j = 0; j < nSkip2; j++) {
      g_pRng->getNextU01();
    }

    g_pVertices[i].z = 0.0f;

    g_pVertices[i].r = 1.0f;
    g_pVertices[i].g = 1.0f;
    g_pVertices[i].b = 1.0f;
  }
}

void createSphere(void) {
  int startVertex = 0;

  for (int i = startVertex; i < g_nVerticesPopulated; i++) {
    float r;
    float rho;
    float theta;

    if (g_currentShape == Sphere) {
      r = g_pRng->getNextU01();
      r = powf(r, 1.f / 3.f);

      for (int j = 0; j < nSkip3; j++) {
        g_pRng->getNextU01();
      }
    } else {
      r = 1.0f;
    }

    rho = g_pRng->getNextU01() * PI * 2.0f;

    for (int j = 0; j < nSkip1; j++) {
      g_pRng->getNextU01();
    }

    theta = (g_pRng->getNextU01() * 2.0f) - 1.0f;
    theta = asin(theta);

    for (int j = 0; j < nSkip2; j++) {
      g_pRng->getNextU01();
    }

    g_pVertices[i].x = r * fabs(cos(theta)) * cos(rho);
    g_pVertices[i].y = r * fabs(cos(theta)) * sin(rho);
    g_pVertices[i].z = r * sin(theta);

    g_pVertices[i].r = 1.0f;
    g_pVertices[i].g = 1.0f;
    g_pVertices[i].b = 1.0f;
  }
}

void createAxes(void) {
  // z axis:
  g_pVertices[200000].x = 0.0f;
  g_pVertices[200000].y = 0.0f;
  g_pVertices[200000].z = -1.5f;
  g_pVertices[200001].x = 0.0f;
  g_pVertices[200001].y = 0.0f;
  g_pVertices[200001].z = 1.5f;
  g_pVertices[200000].r = 1.0f;
  g_pVertices[200000].g = 0.0f;
  g_pVertices[200000].b = 0.0f;
  g_pVertices[200001].r = 0.0f;
  g_pVertices[200001].g = 1.0f;
  g_pVertices[200001].b = 1.0f;
  // y axis:
  g_pVertices[200002].x = 0.0f;
  g_pVertices[200002].y = -1.5f;
  g_pVertices[200002].z = 0.0f;
  g_pVertices[200003].x = 0.0f;
  g_pVertices[200003].y = 1.5f;
  g_pVertices[200003].z = 0.0f;
  g_pVertices[200002].r = 0.0f;
  g_pVertices[200002].g = 1.0f;
  g_pVertices[200002].b = 0.0f;
  g_pVertices[200003].r = 1.0f;
  g_pVertices[200003].g = 0.0f;
  g_pVertices[200003].b = 1.0f;
  // x axis:
  g_pVertices[200004].x = -1.5f;
  g_pVertices[200004].y = 0.0f;
  g_pVertices[200004].z = 0.0f;
  g_pVertices[200005].x = 1.5f;
  g_pVertices[200005].y = 0.0f;
  g_pVertices[200005].z = 0.0f;
  g_pVertices[200004].r = 0.0f;
  g_pVertices[200004].g = 0.0f;
  g_pVertices[200004].b = 1.0f;
  g_pVertices[200005].r = 1.0f;
  g_pVertices[200005].g = 1.0f;
  g_pVertices[200005].b = 0.0f;
}

void drawPoints(void) {
  if (g_bShowAxes) {
    glDrawArrays(GL_LINE_STRIP, 200000, 2);
    glDrawArrays(GL_LINE_STRIP, 200002, 2);
    glDrawArrays(GL_LINE_STRIP, 200004, 2);
  }

  glDrawArrays(GL_POINTS, 0, g_nVerticesPopulated);
}

void drawText(void) {
  using std::string;
  using std::stringstream;

  glPushMatrix();
  glLoadIdentity();
  glRasterPos2f(-1.2f, 1.2f);

  string infoString;
  stringstream ss;
  g_pRng->getInfoString(infoString);
  ss << " skip1=" << nSkip1;
  ss << " skip2=" << nSkip2;
  ss << " skip3=" << nSkip3;
  ss << " points=" << g_nVerticesPopulated;
  infoString.append(ss.str());

  for (unsigned int i = 0; i < infoString.size(); i++) {
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, infoString[i]);
  }

  glPopMatrix();
}

void reshape(int x, int y) {
  float xScale;
  float yScale;

  g_lastShapeX = x;
  g_lastShapeY = y;

  // Check if shape is visible
  if (x == 0 || y == 0) {
    return;
  }

  // Set a new projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Adjust fit
  if (y > x) {
    xScale = 1.0f;
    yScale = (float)y / x;
  } else {
    xScale = (float)x / y;
    yScale = 1.0f;
  }

  // Angle of view:40 degrees
  // Near clipping plane distance: 10.0 (default)
  // Far clipping plane distance: 10.0 (default)
  if (g_bTenXZoom) {
    glOrtho(-.15f * xScale, .15f * xScale, -.15f * yScale, .15f * yScale, -5.0f,
            5.0f);
  } else {
    glOrtho(-1.5f * xScale, 1.5f * xScale, -1.5f * yScale, 1.5f * yScale,
            -10.0f, 10.0f);
  }

  // Use the whole window for rendering
  glViewport(0, 0, x, y);
  glMatrixMode(GL_MODELVIEW);
}

void display(void) {
  glClear(GL_COLOR_BUFFER_BIT);
  glLoadIdentity();
  glTranslatef(0.0f, 0.0f, -4.0f);
  glRotatef(g_yRotated, 0.0f, 1.0f, 0.0f);
  glRotatef(g_xRotated, 1.0f, 0.0f, 0.0f);
  drawPoints();
  drawText();
  glFlush();
  glutSwapBuffers();
}

void idle(void) {
  if (g_bAutoRotate) {
    g_yRotated += 0.1f;

    if (g_yRotated >= 360.0f) {
      g_yRotated -= 360.0f;
    }

    g_xRotated += 0.05f;

    if (g_xRotated >= 360.0f) {
      g_xRotated -= 360.0f;
    }

    display();
  }
}

void reCreate(void) {
  switch (g_currentShape) {
    case Sphere:
    case SphericalShell:
      createSphere();
      break;

    case Cube:
      createCube();
      break;

    default:
      createPlane();
  }

  display();
}

void cleanup(int code) {
  if (g_pRng) {
    delete g_pRng;
    g_pRng = NULL;
  }

  if (g_pVertices) {
    delete[] g_pVertices;
    g_pVertices = NULL;
  }

  if (g_pCheckRender) {
    delete g_pCheckRender;
    g_pCheckRender = NULL;
  }

  exit(code);
}

void glutClose() { cleanup(EXIT_SUCCESS); }

void keyboard(unsigned char key, int x, int y) {
  switch (key) {
    // Select shape
    case 's':
    case 'S':
      g_currentShape = Sphere;
      createSphere();
      display();
      break;

    case 'e':
    case 'E':
      g_currentShape = SphericalShell;
      createSphere();
      display();
      break;

    case 'b':
    case 'B':
      g_currentShape = Cube;
      createCube();
      display();
      break;

    case 'p':
    case 'P':
      g_currentShape = Plane;
      createPlane();
      display();
      break;

    // Rotation
    case 'a':
    case 'A':
      g_bAutoRotate = !g_bAutoRotate;
      break;

    case 'i':
    case 'I':
      g_xRotated -= 1.0f;

      if (g_xRotated <= 0.0f) {
        g_xRotated += 360.0f;
      }

      display();
      break;

    case ',':
      g_xRotated += 1.0f;

      if (g_xRotated >= 360.0f) {
        g_xRotated -= 360.0f;
      }

      display();
      break;

    case 'j':
    case 'J':
      g_yRotated -= 1.0f;

      if (g_yRotated <= 0.0f) {
        g_yRotated += 360.0f;
      }

      display();
      break;

    case 'l':
    case 'L':
      g_yRotated += 1.0f;

      if (g_yRotated >= 360.0f) {
        g_yRotated -= 360.0f;
      }

      display();
      break;

    // Zoom
    case 't':
    case 'T':
      g_bTenXZoom = !g_bTenXZoom;
      reshape(g_lastShapeX, g_lastShapeY);
      reCreate();
      break;

    // Axes
    case 'z':
    case 'Z':
      g_bShowAxes = !g_bShowAxes;
      reCreate();
      break;

    // RNG
    case 'x':
    case 'X':
      g_pRng->selectRng(RNG::Pseudo);
      reCreate();
      break;

    case 'c':
    case 'C':
      g_pRng->selectRng(RNG::Quasi);
      reCreate();
      break;

    case 'v':
    case 'V':
      g_pRng->selectRng(RNG::ScrambledQuasi);
      reCreate();
      break;

    case 'r':
    case 'R':
      g_pRng->resetSeed();
      reCreate();
      break;

    case ']':
      g_pRng->incrementDimensions();
      reCreate();
      break;

    case '[':
      g_pRng->resetDimensions();
      reCreate();
      break;

    case '1':
      nSkip1++;
      reCreate();
      break;

    case '2':
      nSkip2++;
      reCreate();
      break;

    case '3':
      nSkip3++;
      reCreate();
      break;

    case '!':
      nSkip1 = 0;
      nSkip2 = 0;
      nSkip3 = 0;
      reCreate();
      break;

    // Number of vertices
    case '+':
      g_nVerticesPopulated += 8000;

      if (g_nVerticesPopulated > g_nVertices) {
        g_nVerticesPopulated = g_nVertices;
      }

      reCreate();
      break;

    case '-':
      g_nVerticesPopulated -= 8000;

      if (g_nVerticesPopulated < 8000) {
        g_nVerticesPopulated = 8000;
      }

      reCreate();
      break;

    // Quit
    case 27:
    case 'q':
    case 'Q':
#if defined(__APPLE__) || defined(MACOSX)
      exit(EXIT_SUCCESS);
#else
      glutDestroyWindow(glutGetWindow());
      return;
#endif
  }
}

void showHelp(void) {
  using std::left;
  using std::setw;
  using std::stringstream;

  stringstream ss;

  ss << "\nRandom number visualization\n\n";
  ss << "On creation, randomFog generates 200,000 random coordinates in "
        "spherical coordinate space (radius, angle rho, angle theta) with "
        "curand's XORWOW algorithm. The coordinates are normalized for a "
        "uniform distribution through the sphere.\n\n";
  ss << "The X axis is drawn with blue in the negative direction and yellow "
        "positive.\n"
     << "The Y axis is drawn with green in the negative direction and magenta "
        "positive.\n"
     << "The Z axis is drawn with red in the negative direction and cyan "
        "positive.\n\n";
  ss << "The following keys can be used to control the output:\n\n";
  ss << left;
  ss << "\t" << setw(10) << "s"
     << "Generate a new set of random numbers and display as spherical "
        "coordinates (Sphere)\n";
  ss << "\t" << setw(10) << "e"
     << "Generate a new set of random numbers and display on a spherical "
        "surface (shEll)\n";
  ss << "\t" << setw(10) << "b"
     << "Generate a new set of random numbers and display as cartesian "
        "coordinates (cuBe/Box)\n";
  ss << "\t" << setw(10) << "p"
     << "Generate a new set of random numbers and display on a cartesian plane "
        "(Plane)\n\n";
  ss << "\t" << setw(10) << "i,l,j"
     << "Rotate the negative Z-axis up, right, down and left respectively\n";
  ss << "\t" << setw(10) << "a"
     << "Toggle auto-rotation\n";
  ss << "\t" << setw(10) << "t"
     << "Toggle 10x zoom\n";
  ss << "\t" << setw(10) << "z"
     << "Toggle axes display\n\n";
  ss << "\t" << setw(10) << "x"
     << "Select XORWOW generator (default)\n";
  ss << "\t" << setw(10) << "c"
     << "Select Sobol' generator\n";
  ss << "\t" << setw(10) << "v"
     << "Select scrambled Sobol' generator\n";
  ss << "\t" << setw(10) << "r"
     << "Reset XORWOW (i.e. reset to initial seed) and regenerate\n";
  ss << "\t" << setw(10) << "]"
     << "Increment the number of Sobol' dimensions and regenerate\n";
  ss << "\t" << setw(10) << "["
     << "Reset the number of Sobol' dimensions to 1 and regenerate\n\n";
  ss << "\t" << setw(10) << "+"
     << "Increment the number of displayed points by 8,000 (up to maximum "
        "200,000)\n";
  ss << "\t" << setw(10) << "-"
     << "Decrement the number of displayed points by 8,000 (down to minimum "
        "8,000)\n\n";
  ss << "\t" << setw(10) << "q/[ESC]"
     << "Quit the application.\n\n";
  puts(ss.str().c_str());
}

int main(int argc, char **argv) {
  using std::runtime_error;

  try {
    bool bQA = false;

    // Open the log file
    printf("Random Fog\n");
    printf("==========\n\n");

    // Check QA mode
    if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
      bQA = true;

      findCudaDevice(argc, (const char **)argv);

      g_pCheckRender =
          new CheckBackBuffer(g_lastShapeX, g_lastShapeY, 4, false);
    } else {
#if defined(__linux__)
      setenv("DISPLAY", ":0", 0);
#endif
      // Initialize GL
      glutInit(&argc, argv);
      glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
      // TODO use width/height?
      glutInitWindowSize(1000, 1000);
      // Create a window with rendering context and everything else we need
      glutCreateWindow("Random Fog");

      if (!isGLVersionSupported(2, 0)) {
        fprintf(stderr, "This sample requires at least OpenGL 2.0\n");
        exit(EXIT_WAIVED);
      }

      // Select CUDA device with OpenGL interoperability
      findCudaDevice(argc, (const char **)argv);
    }

    // Create vertices
    g_nVertices = 200000;
    g_nVerticesPopulated = 200000;
    g_pVertices = new SVertex[g_nVertices + 6];

    // Setup the random number generators
    g_pRng = new RNG(12345, 1, 100000);
    printf("CURAND initialized\n");

    // Compute the initial vertices and indices, starting in spherical mode
    createSphere();
    createAxes();

    showHelp();

    if (bQA) {
      g_pCheckRender->setExecPath(argv[0]);
      g_pCheckRender->dumpBin(
          g_pVertices, g_nVerticesPopulated * sizeof(SVertex), "randomFog.bin");

      if (g_pCheckRender->compareBin2BinFloat(
              "randomFog.bin", "ref_randomFog.bin",
              g_nVerticesPopulated * sizeof(SVertex) / sizeof(float), 0.25f,
              0.35f)) {
        cleanup(EXIT_SUCCESS);
      } else {
        cleanup(EXIT_FAILURE);
      }
    } else {
      // As we do not yet use a depth buffer, we cannot fill our sphere
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      // Enable the vertex array functionality:
      glEnableClientState(GL_VERTEX_ARRAY);
      // Enable the color array functionality (so we can specify a color for
      // each vertex)
      glEnableClientState(GL_COLOR_ARRAY);
      // Pass the vertex pointer:
      glVertexPointer(3,  // 3 components per vertex (x,y,z)
                      GL_FLOAT, sizeof(SVertex), g_pVertices);
      // Pass the color pointer
      glColorPointer(3,  // 3 components per vertex (r,g,b)
                     GL_FLOAT, sizeof(SVertex),
                     &g_pVertices[0].r);  // Pointer to the first color
      // Point size for point mode
      glPointSize(1.0f);
      glLineWidth(2.0f);
      glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      // Notify glut which messages we require:
      glutDisplayFunc(display);
      glutReshapeFunc(reshape);
      glutKeyboardFunc(keyboard);
      glutIdleFunc(idle);

#if defined(__APPLE__) || defined(MACOSX)
      atexit(glutClose);
#else
      glutCloseFunc(glutClose);
#endif

      // Let's get started!
      glutMainLoop();
    }
  } catch (runtime_error &e) {
    printf("runtime error (%s)\n", e.what());
  }

  exit(EXIT_SUCCESS);
}