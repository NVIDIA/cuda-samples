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

#include <assert.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
                       int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
                       int **aColInd, int extendSymMatrix);

void UsageSP(void) {
  printf("<options>\n");
  printf("-h          : display this help\n");
  printf("-file=<filename> : filename containing a matrix in MM format\n");
  printf("-device=<device_id> : <device_id> if want to run on specific GPU\n");

  exit(0);
}

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts) {
  memset(&opts, 0, sizeof(opts));

  if (checkCmdLineFlag(argc, (const char **)argv, "-h")) {
    UsageSP();
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    char *fileName = 0;
    getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

    if (fileName) {
      opts.sparse_mat_filename = fileName;
    } else {
      printf("\nIncorrect filename passed to -file \n ");
      UsageSP();
    }
  }
}

int main(int argc, char *argv[]) {
  struct testOpts opts;
  cusolverSpHandle_t cusolverSpH =
      NULL;  // reordering, permutation and 1st LU factorization
  cusparseHandle_t cusparseH = NULL;  // residual evaluation
  cudaStream_t stream = NULL;
  cusparseMatDescr_t descrA = NULL;  // A is a base-0 general matrix

  csrqrInfoHost_t h_info =
      NULL;  // opaque info structure for LU with parital pivoting
  csrqrInfo_t d_info =
      NULL;  // opaque info structure for LU with parital pivoting

  int rowsA = 0;  // number of rows of A
  int colsA = 0;  // number of columns of A
  int nnzA = 0;   // number of nonzeros of A
  int baseA = 0;  // base index in CSR format

  // CSR(A) from I/O
  int *h_csrRowPtrA = NULL;  // <int> n+1
  int *h_csrColIndA = NULL;  // <int> nnzA
  double *h_csrValA = NULL;  // <double> nnzA

  double *h_x = NULL;      // <double> n,  x = A \ b
  double *h_b = NULL;      // <double> n, b = ones(m,1)
  double *h_bcopy = NULL;  // <double> n, b = ones(m,1)
  double *h_r = NULL;      // <double> n, r = b - A*x

  size_t size_internal = 0;
  size_t size_chol = 0;     // size of working space for csrlu
  void *buffer_cpu = NULL;  // working space for Cholesky
  void *buffer_gpu = NULL;  // working space for Cholesky

  int *d_csrRowPtrA = NULL;  // <int> n+1
  int *d_csrColIndA = NULL;  // <int> nnzA
  double *d_csrValA = NULL;  // <double> nnzA
  double *d_x = NULL;        // <double> n, x = A \ b
  double *d_b = NULL;        // <double> n, a copy of h_b
  double *d_r = NULL;        // <double> n, r = b - A*x

  // the constants used in residual evaluation, r = b - A*x
  const double minus_one = -1.0;
  const double one = 1.0;
  const double zero = 0.0;
  // the constant used in cusolverSp
  // singularity is -1 if A is invertible under tol
  // tol determines the condition of singularity
  int singularity = 0;
  const double tol = 1.e-14;

  double x_inf = 0.0;  // |x|
  double r_inf = 0.0;  // |r|
  double A_inf = 0.0;  // |A|

  parseCommandLineArguments(argc, argv, opts);

  findCudaDevice(argc, (const char **)argv);

  if (opts.sparse_mat_filename == NULL) {
    opts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n32.mtx", argv[0]);
    if (opts.sparse_mat_filename != NULL)
      printf("Using default input file [%s]\n", opts.sparse_mat_filename);
    else
      printf("Could not find lap2D_5pt_n32.mtx\n");
  } else {
    printf("Using input file [%s]\n", opts.sparse_mat_filename);
  }

  printf("step 1: read matrix market format\n");

  if (opts.sparse_mat_filename) {
    if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true, &rowsA,
                                   &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
                                   &h_csrColIndA, true)) {
      return 1;
    }
    baseA = h_csrRowPtrA[0];  // baseA = {0,1}
  } else {
    fprintf(stderr, "Error: input matrix is not provided\n");
    return 1;
  }

  if (rowsA != colsA) {
    fprintf(stderr, "Error: only support square matrix\n");
    return 1;
  }

  printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA,
         nnzA, baseA);

  checkCudaErrors(cusolverSpCreate(&cusolverSpH));
  checkCudaErrors(cusparseCreate(&cusparseH));
  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cusolverSpSetStream(cusolverSpH, stream));
  checkCudaErrors(cusparseSetStream(cusparseH, stream));

  checkCudaErrors(cusparseCreateMatDescr(&descrA));

  checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));

  if (baseA) {
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
  } else {
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  }

  h_x = (double *)malloc(sizeof(double) * colsA);
  h_b = (double *)malloc(sizeof(double) * rowsA);
  h_bcopy = (double *)malloc(sizeof(double) * rowsA);
  h_r = (double *)malloc(sizeof(double) * rowsA);

  assert(NULL != h_x);
  assert(NULL != h_b);
  assert(NULL != h_bcopy);
  assert(NULL != h_r);

  checkCudaErrors(
      cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
  checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * colsA));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double) * rowsA));
  checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double) * rowsA));

  for (int row = 0; row < rowsA; row++) {
    h_b[row] = 1.0;
  }

  memcpy(h_bcopy, h_b, sizeof(double) * rowsA);

  printf("step 2: create opaque info structure\n");
  checkCudaErrors(cusolverSpCreateCsrqrInfoHost(&h_info));

  printf("step 3: analyze qr(A) to know structure of L\n");
  checkCudaErrors(cusolverSpXcsrqrAnalysisHost(cusolverSpH, rowsA, colsA, nnzA,
                                               descrA, h_csrRowPtrA,
                                               h_csrColIndA, h_info));

  printf("step 4: workspace for qr(A)\n");
  checkCudaErrors(cusolverSpDcsrqrBufferInfoHost(
      cusolverSpH, rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA,
      h_csrColIndA, h_info, &size_internal, &size_chol));

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  buffer_cpu = (void *)malloc(sizeof(char) * size_chol);
  assert(NULL != buffer_cpu);

  printf("step 5: compute A = L*L^T \n");
  checkCudaErrors(cusolverSpDcsrqrSetupHost(cusolverSpH, rowsA, colsA, nnzA,
                                            descrA, h_csrValA, h_csrRowPtrA,
                                            h_csrColIndA, zero, h_info));

  checkCudaErrors(cusolverSpDcsrqrFactorHost(cusolverSpH, rowsA, colsA, nnzA,
                                             NULL, NULL, h_info, buffer_cpu));

  printf("step 6: check if the matrix is singular \n");
  checkCudaErrors(
      cusolverSpDcsrqrZeroPivotHost(cusolverSpH, h_info, tol, &singularity));

  if (0 <= singularity) {
    fprintf(stderr, "Error: A is not invertible, singularity=%d\n",
            singularity);
    return 1;
  }

  printf("step 7: solve A*x = b \n");
  checkCudaErrors(cusolverSpDcsrqrSolveHost(cusolverSpH, rowsA, colsA, h_b, h_x,
                                            h_info, buffer_cpu));

  printf("step 8: evaluate residual r = b - A*x (result on CPU)\n");
  // use GPU gemv to compute r = b - A*x
  checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA,
                             sizeof(int) * (rowsA + 1),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                             cudaMemcpyHostToDevice));

  checkCudaErrors(
      cudaMemcpy(d_r, h_bcopy, sizeof(double) * rowsA, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_x, h_x, sizeof(double) * colsA, cudaMemcpyHostToDevice));

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseSpMatDescr_t matA = NULL;
  if (baseA) {
    checkCudaErrors(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA,
                                      d_csrColIndA, d_csrValA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F));
  } else {
    checkCudaErrors(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA,
                                      d_csrColIndA, d_csrValA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  }

  cusparseDnVecDescr_t vecx = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecx, colsA, d_x, CUDA_R_64F));
  cusparseDnVecDescr_t vecAx = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecAx, rowsA, d_r, CUDA_R_64F));

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx, &one,
      vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkCudaErrors(cudaMalloc(&buffer, bufferSize));

  checkCudaErrors(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkCudaErrors(
      cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));

  x_inf = vec_norminf(colsA, h_x);
  r_inf = vec_norminf(rowsA, h_r);
  A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA,
                          h_csrColIndA);

  printf("(CPU) |b - A*x| = %E \n", r_inf);
  printf("(CPU) |A| = %E \n", A_inf);
  printf("(CPU) |x| = %E \n", x_inf);
  printf("(CPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

  printf("step 9: create opaque info structure\n");
  checkCudaErrors(cusolverSpCreateCsrqrInfo(&d_info));

  checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA,
                             sizeof(int) * (rowsA + 1),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_b, h_bcopy, sizeof(double) * rowsA, cudaMemcpyHostToDevice));

  printf("step 10: analyze qr(A) to know structure of L\n");
  checkCudaErrors(cusolverSpXcsrqrAnalysis(cusolverSpH, rowsA, colsA, nnzA,
                                           descrA, d_csrRowPtrA, d_csrColIndA,
                                           d_info));

  printf("step 11: workspace for qr(A)\n");
  checkCudaErrors(cusolverSpDcsrqrBufferInfo(
      cusolverSpH, rowsA, colsA, nnzA, descrA, d_csrValA, d_csrRowPtrA,
      d_csrColIndA, d_info, &size_internal, &size_chol));

  printf("GPU buffer size = %lld bytes\n", (signed long long)size_chol);
  if (buffer_gpu) {
    checkCudaErrors(cudaFree(buffer_gpu));
  }
  checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char) * size_chol));

  printf("step 12: compute A = L*L^T \n");
  checkCudaErrors(cusolverSpDcsrqrSetup(cusolverSpH, rowsA, colsA, nnzA, descrA,
                                        d_csrValA, d_csrRowPtrA, d_csrColIndA,
                                        zero, d_info));

  checkCudaErrors(cusolverSpDcsrqrFactor(cusolverSpH, rowsA, colsA, nnzA, NULL,
                                         NULL, d_info, buffer_gpu));

  printf("step 13: check if the matrix is singular \n");
  checkCudaErrors(
      cusolverSpDcsrqrZeroPivot(cusolverSpH, d_info, tol, &singularity));

  if (0 <= singularity) {
    fprintf(stderr, "Error: A is not invertible, singularity=%d\n",
            singularity);
    return 1;
  }

  printf("step 14: solve A*x = b \n");
  checkCudaErrors(cusolverSpDcsrqrSolve(cusolverSpH, rowsA, colsA, d_b, d_x,
                                        d_info, buffer_gpu));

  checkCudaErrors(
      cudaMemcpy(d_r, h_bcopy, sizeof(double) * rowsA, cudaMemcpyHostToDevice));

  checkCudaErrors(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkCudaErrors(
      cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));

  r_inf = vec_norminf(rowsA, h_r);

  printf("(GPU) |b - A*x| = %E \n", r_inf);
  printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

  if (cusolverSpH) {
    checkCudaErrors(cusolverSpDestroy(cusolverSpH));
  }
  if (cusparseH) {
    checkCudaErrors(cusparseDestroy(cusparseH));
  }
  if (stream) {
    checkCudaErrors(cudaStreamDestroy(stream));
  }
  if (descrA) {
    checkCudaErrors(cusparseDestroyMatDescr(descrA));
  }
  if (h_info) {
    checkCudaErrors(cusolverSpDestroyCsrqrInfoHost(h_info));
  }
  if (d_info) {
    checkCudaErrors(cusolverSpDestroyCsrqrInfo(d_info));
  }

  if (matA) {
    checkCudaErrors(cusparseDestroySpMat(matA));
  }
  if (vecx) {
    checkCudaErrors(cusparseDestroyDnVec(vecx));
  }
  if (vecAx) {
    checkCudaErrors(cusparseDestroyDnVec(vecAx));
  }

  if (h_csrValA) {
    free(h_csrValA);
  }
  if (h_csrRowPtrA) {
    free(h_csrRowPtrA);
  }
  if (h_csrColIndA) {
    free(h_csrColIndA);
  }

  if (h_x) {
    free(h_x);
  }
  if (h_b) {
    free(h_b);
  }
  if (h_bcopy) {
    free(h_bcopy);
  }
  if (h_r) {
    free(h_r);
  }

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  if (buffer_gpu) {
    checkCudaErrors(cudaFree(buffer_gpu));
  }

  if (d_csrValA) {
    checkCudaErrors(cudaFree(d_csrValA));
  }
  if (d_csrRowPtrA) {
    checkCudaErrors(cudaFree(d_csrRowPtrA));
  }
  if (d_csrColIndA) {
    checkCudaErrors(cudaFree(d_csrColIndA));
  }
  if (d_x) {
    checkCudaErrors(cudaFree(d_x));
  }
  if (d_b) {
    checkCudaErrors(cudaFree(d_b));
  }
  if (d_r) {
    checkCudaErrors(cudaFree(d_r));
  }

  return 0;
}
