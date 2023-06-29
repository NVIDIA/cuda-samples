// Copyright (c) 1993-2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <assert.h>
#include <nvvm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Two levels of indirection to stringify LIBDEVICE_MAJOR_VERSION and
 * LIBDEVICE_MINOR_VERSION correctly. */
#define getLibDeviceName() \
  _getLibDeviceName(LIBDEVICE_MAJOR_VERSION, LIBDEVICE_MINOR_VERSION)
#define _getLibDeviceName(MAJOR, MINOR) __getLibDeviceName(MAJOR, MINOR)
#define __getLibDeviceName(MAJOR, MINOR) \
  ("/libdevice/libdevice." #MAJOR #MINOR ".bc")

#define getLibnvvmHome() _getLibnvvmHome(LIBNVVM_HOME)
#define _getLibnvvmHome(NVVM_HOME) __getLibnvvmHome(NVVM_HOME)
#define __getLibnvvmHome(NVVM_HOME) (#NVVM_HOME)

typedef enum {
  PTXGEN_SUCCESS = 0x0000,
  PTXGEN_FILE_IO_ERROR = 0x0001,
  PTXGEN_BAD_ALLOC_ERROR = 0x0002,
  PTXGEN_LIBNVVM_COMPILATION_ERROR = 0x0004,
  PTXGEN_LIBNVVM_ERROR = 0x0008,
  PTXGEN_INVALID_USAGE = 0x0010,
  PTXGEN_LIBNVVM_HOME_UNDEFINED = 0x0020,
  PTXGEN_LIBNVVM_VERIFICATION_ERROR = 0x0040
} PTXGenStatus;

typedef enum { PTXGEN_INPUT_PROGRAM, PTXGEN_INPUT_LIBDEVICE } PTXGENInput;

static PTXGenStatus getLibDevicePath(char **buffer) {
  assert(buffer);

  const char *libnvvmPath = getLibnvvmHome();
  if (libnvvmPath == NULL) {
    fprintf(stderr, "The environment variable LIBNVVM_HOME undefined\n");
    return PTXGEN_LIBNVVM_HOME_UNDEFINED;
  }

  const char *libdevice = getLibDeviceName();
  *buffer = malloc(strlen(libnvvmPath) + strlen(libdevice) + 1);
  if (*buffer == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    return PTXGEN_BAD_ALLOC_ERROR;
  }

  // Concatenate libnvvmPath with libdevice.
  *buffer = strcat(strcpy(*buffer, libnvvmPath), libdevice);

  return PTXGEN_SUCCESS;
}

static PTXGenStatus addFileToProgram(const char *filename, nvvmProgram prog,
                                     PTXGENInput inputType) {
  assert(filename);

  // Open the input file.
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "Failed to open %s\n", filename);
    return PTXGEN_FILE_IO_ERROR;
  }

  // Allocate a buffer for the input.
  struct stat fileStat;
  fstat(fileno(f), &fileStat);
  char *buffer = malloc(fileStat.st_size);
  if (buffer == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    return PTXGEN_BAD_ALLOC_ERROR;
  }
  const size_t size = fread(buffer, 1, fileStat.st_size, f);
  if (ferror(f)) {
    fprintf(stderr, "Failed to read %s\n", filename);
    fclose(f);
    free(buffer);
    return PTXGEN_FILE_IO_ERROR;
  }
  fclose(f);

  // Add the input to the libNVVM program instance.
  nvvmResult result;
  if (inputType == PTXGEN_INPUT_LIBDEVICE)
    result = nvvmLazyAddModuleToProgram(prog, buffer, size, filename);
  else
    result = nvvmAddModuleToProgram(prog, buffer, size, filename);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "Failed to add the module %s to the compilation unit\n",
            filename);
    free(buffer);
    return PTXGEN_LIBNVVM_ERROR;
  }

  free(buffer);
  return PTXGEN_SUCCESS;
}

// Read the nvvmProgram compilation log.
static PTXGenStatus dumpCompilationLog(nvvmProgram prog) {
  size_t logSize;
  PTXGenStatus status = PTXGEN_SUCCESS;
  if (nvvmGetProgramLogSize(prog, &logSize) != NVVM_SUCCESS) {
    fprintf(stderr, "Failed to get the compilation log size\n");
    status = PTXGEN_LIBNVVM_ERROR;
  } else {
    char *log = malloc(logSize);
    if (log == NULL) {
      fprintf(stderr, "Failed to allocate memory\n");
      status = PTXGEN_BAD_ALLOC_ERROR;
    } else if (nvvmGetProgramLog(prog, log) != NVVM_SUCCESS) {
      fprintf(stderr, "Failed to get the compilation log\n");
      status = PTXGEN_LIBNVVM_ERROR;
    } else {
      fprintf(stderr, "%s\n", log);
    }
    free(log);
  }
  return status;
}

static PTXGenStatus generatePTX(unsigned numOptions, const char **options,
                                unsigned numFilenames, const char **filenames) {
  // Create the compilation unit (the libNVVM program instance).
  nvvmProgram prog;
  if (nvvmCreateProgram(&prog) != NVVM_SUCCESS) {
    fprintf(stderr, "Failed to create the compilation unit\n");
    return PTXGEN_LIBNVVM_ERROR;
  }

  // Add the module to the compilation unit.
  for (unsigned i = 0; i < numFilenames; ++i) {
    PTXGenStatus status =
        addFileToProgram(filenames[i], prog, PTXGEN_INPUT_PROGRAM);
    if (status != PTXGEN_SUCCESS) {
      nvvmDestroyProgram(&prog);
      return status;
    }
  }

  // Verify the compilation unit.
  if (nvvmVerifyProgram(prog, numOptions, options) != NVVM_SUCCESS) {
    fprintf(stderr, "Failed to verify the compilation unit\n");
    return PTXGEN_LIBNVVM_VERIFICATION_ERROR;
  }

  // Add libdevice to the libNVVM program instance.
  char *libDeviceName;
  PTXGenStatus status = getLibDevicePath(&libDeviceName);
  if (status != PTXGEN_SUCCESS) {
    nvvmDestroyProgram(&prog);
    return status;
  }
  status = addFileToProgram(libDeviceName, prog, PTXGEN_INPUT_LIBDEVICE);
  free(libDeviceName);
  if (status != PTXGEN_SUCCESS) {
    nvvmDestroyProgram(&prog);
    return status;
  }

  // Display the compilation log.
  status |= dumpCompilationLog(prog);
  if (status & PTXGEN_LIBNVVM_VERIFICATION_ERROR) {
    nvvmDestroyProgram(&prog);
    return status;
  }

  // Compile the compilation unit.
  if (nvvmCompileProgram(prog, numOptions, options) != NVVM_SUCCESS) {
    fprintf(stderr, "Failed to generate PTX from the compilation unit\n");
    status |= PTXGEN_LIBNVVM_COMPILATION_ERROR;
  } else {
    size_t ptxSize;
    if (nvvmGetCompiledResultSize(prog, &ptxSize) != NVVM_SUCCESS) {
      fprintf(stderr, "Failed to get the PTX output size\n");
      status |= PTXGEN_LIBNVVM_ERROR;
    } else {
      char *ptx = malloc(ptxSize);
      if (ptx == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        status |= PTXGEN_BAD_ALLOC_ERROR;
      } else if (nvvmGetCompiledResult(prog, ptx) != NVVM_SUCCESS) {
        fprintf(stderr, "Failed to get the PTX output\n");
        status |= PTXGEN_LIBNVVM_ERROR;
      } else {
        fprintf(stdout, "%s\n", ptx);
      }
      free(ptx);
    }
  }

  status |= dumpCompilationLog(prog);

  // Release the resources.
  nvvmDestroyProgram(&prog);

  return status;
}

static void showUsage(void) {
  fprintf(stderr,
          "Usage: ptxgen [OPTION]... [FILE]...\n"
          "  [FILE] could be a .bc file or a .ll file\n");
}

int main(int argc, char *argv[]) {
  PTXGenStatus status = PTXGEN_SUCCESS;

  if (argc == 1) {
    showUsage();
    return PTXGEN_INVALID_USAGE;
  }

  // Extract libNVVM options and the input file names.
  unsigned numOptions = 0, numFilenames = 0;
  const char **options = malloc((argc - 1) * sizeof(char *));
  const char **filenames = malloc((argc - 1) * sizeof(char *));
  assert(options && filenames);
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '-') {
      options[numOptions] = argv[i];
      ++numOptions;
    } else {
      filenames[numFilenames] = argv[i];
      ++numFilenames;
    }
  }

  if (numFilenames == 0) {
    showUsage();
    status = PTXGEN_INVALID_USAGE;
  } else {
    // Use libNVVM to generate PTX.
    status = generatePTX(numOptions, options, numFilenames, filenames);
  }

  free(options);
  free(filenames);
  return status;
}
