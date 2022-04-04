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

/*
 * This sample implements a preconditioned conjugate gradient solver on
 * the GPU using CUBLAS and CUSPARSE.  Relative to the conjugateGradient
 * SDK example, this demonstrates the use of cusparseScsrilu02() for
 * computing the incompute-LU preconditioner and cusparseScsrsv2_solve()
 * for solving triangular systems.  Specifically, the preconditioned
 * conjugate gradient method with an incomplete LU preconditioner is
 * used to solve the Laplacian operator in 2D on a uniform mesh.
 *
 * Note that the code in this example and the specific matrices used here
 * were chosen to demonstrate the use of the CUSPARSE library as simply
 * and as clearly as possible.  This is not optimized code and the input
 * matrices have been chosen for simplicity rather than performance.
 * These should not be used either as a performance guide or for
 * benchmarking purposes.
 */

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cublas_v2.h>
#include <cusparse.h>

// Utilities and system includes
#include <helper_cuda.h>       // CUDA error checking
#include <helper_functions.h>  // shared functions common to CUDA Samples

const char *sSDKname = "conjugateGradientPrecond";

/*
 * Generate a matrix representing a second order regular Laplacian operator
 * on a 2D domain in Compressed Sparse Row format.
 */
void genLaplace(int *row_ptr, int *col_ind, float *val, int M, int N, int nz,
                float *rhs) {
  assert(M == N);
  int n = (int)sqrt((double)N);
  assert(n * n == N);
  printf("laplace dimension = %d\n", n);
  int idx = 0;

  // loop over degrees of freedom
  for (int i = 0; i < N; i++) {
    int ix = i % n;
    int iy = i / n;

    row_ptr[i] = idx;

    // up
    if (iy > 0) {
      val[idx] = 1.0;
      col_ind[idx] = i - n;
      idx++;
    } else {
      rhs[i] -= 1.0;
    }

    // left
    if (ix > 0) {
      val[idx] = 1.0;
      col_ind[idx] = i - 1;
      idx++;
    } else {
      rhs[i] -= 0.0;
    }

    // center
    val[idx] = -4.0;
    col_ind[idx] = i;
    idx++;

    // right
    if (ix < n - 1) {
      val[idx] = 1.0;
      col_ind[idx] = i + 1;
      idx++;
    } else {
      rhs[i] -= 0.0;
    }

    // down
    if (iy < n - 1) {
      val[idx] = 1.0;
      col_ind[idx] = i + n;
      idx++;
    } else {
      rhs[i] -= 0.0;
    }
  }

  row_ptr[N] = idx;
}

/*
 * Solve Ax=b using the conjugate gradient method
 * a) without any preconditioning,
 * b) using an Incomplete Cholesky preconditioner, and
 * c) using an ILU0 preconditioner.
 */
int main(int argc, char **argv) {
  const int max_iter = 1000;
  int k, M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
  int *d_col, *d_row;
  int qatest = 0;
  const float tol = 1e-12f;
  float *x, *rhs;
  float r0, r1, alpha, beta;
  float *d_val, *d_x;
  float *d_zm1, *d_zm2, *d_rm2;
  float *d_r, *d_p, *d_omega, *d_y;
  float *val = NULL;
  float *d_valsILU0;
  void *buffer = NULL;
  float rsum, diff, err = 0.0;
  float qaerr1, qaerr2 = 0.0;
  float dot, numerator, denominator, nalpha;
  const float floatone = 1.0;
  const float floatzero = 0.0;

  int nErrors = 0;

  printf("conjugateGradientPrecond starting...\n");

  /* QA testing mode */
  if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
    qatest = 1;
  }

  /* This will pick the best possible CUDA capable device */
  cudaDeviceProp deviceProp;
  int devID = findCudaDevice(argc, (const char **)argv);
  printf("GPU selected Device ID = %d \n", devID);

  if (devID < 0) {
    printf("Invalid GPU device %d selected,  exiting...\n", devID);
    exit(EXIT_SUCCESS);
  }

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  /* Statistics about the GPU device */
  printf(
      "> GPU device has %d Multi-Processors, "
      "SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a Laplace matrix in CSR (Compressed Sparse Row) format */
  M = N = 16384;
  nz = 5 * N - 4 * (int)sqrt((double)N);
  I = (int *)malloc(sizeof(int) * (N + 1));   // csr row pointers for matrix A
  J = (int *)malloc(sizeof(int) * nz);        // csr column indices for matrix A
  val = (float *)malloc(sizeof(float) * nz);  // csr values for matrix A
  x = (float *)malloc(sizeof(float) * N);
  rhs = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    rhs[i] = 0.0;  // Initialize RHS
    x[i] = 0.0;    // Initial solution approximation
  }

  genLaplace(I, J, val, M, N, nz, rhs);

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  checkCudaErrors(cublasCreate(&cublasHandle));

  /* Create CUSPARSE context */
  cusparseHandle_t cusparseHandle = NULL;
  checkCudaErrors(cusparseCreate(&cusparseHandle));

  /* Description of the A matrix */
  cusparseMatDescr_t descr = 0;
  checkCudaErrors(cusparseCreateMatDescr(&descr));
  checkCudaErrors(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  /* Allocate required memory */
  checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_y, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_omega, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nz * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_zm1, (N) * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_zm2, (N) * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_rm2, (N) * sizeof(float)));

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseSpMatDescr_t matA = NULL;
  checkCudaErrors(cusparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseDnVecDescr_t vecp = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
  cusparseDnVecDescr_t vecomega = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecomega, N, d_omega, CUDA_R_32F));

  /* Initialize problem data */
  checkCudaErrors(
      cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_y, 0, sizeof(float) * N));

  /* Create ILU(0) info object */
  csrilu02Info_t infoILU = NULL;
  checkCudaErrors(cusparseCreateCsrilu02Info(&infoILU));

  /* Create L factor descriptor and triangular solve info */
  cusparseMatDescr_t descrL = NULL;
  checkCudaErrors(cusparseCreateMatDescr(&descrL));
  checkCudaErrors(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
  checkCudaErrors(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
  checkCudaErrors(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT));
  csrsv2Info_t infoL = NULL;
  checkCudaErrors(cusparseCreateCsrsv2Info(&infoL));

  /* Create U factor descriptor and triangular solve info */
  cusparseMatDescr_t descrU = NULL;
  checkCudaErrors(cusparseCreateMatDescr(&descrU));
  checkCudaErrors(cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO));
  checkCudaErrors(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER));
  checkCudaErrors(cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));
  csrsv2Info_t infoU = NULL;
  checkCudaErrors(cusparseCreateCsrsv2Info(&infoU));

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  size_t tmp = 0;
  int stmp = 0;
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA, vecp,
      &floatzero, vecomega, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &tmp));
  if (tmp > bufferSize) {
    bufferSize = stmp;
  }
  checkCudaErrors(cusparseScsrilu02_bufferSize(
      cusparseHandle, N, nz, descr, d_val, d_row, d_col, infoILU, &stmp));
  if (stmp > bufferSize) {
    bufferSize = stmp;
  }
  checkCudaErrors(cusparseScsrsv2_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrL, d_val,
      d_row, d_col, infoL, &stmp));
  if (stmp > bufferSize) {
    bufferSize = stmp;
  }
  checkCudaErrors(cusparseScsrsv2_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val,
      d_row, d_col, infoU, &stmp));
  if (stmp > bufferSize) {
    bufferSize = stmp;
  }
  checkCudaErrors(cudaMalloc(&buffer, bufferSize));

  /* Conjugate gradient without preconditioning.
     ------------------------------------------

     Follows the description by Golub & Van Loan,
     "Matrix Computations 3rd ed.", Section 10.2.6  */

  printf("Convergence of CG without preconditioning: \n");
  k = 0;
  r0 = 0;
  checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

  while (r1 > tol * tol && k <= max_iter) {
    k++;

    if (k == 1) {
      checkCudaErrors(cublasScopy(cublasHandle, N, d_r, 1, d_p, 1));
    } else {
      beta = r1 / r0;
      checkCudaErrors(cublasSscal(cublasHandle, N, &beta, d_p, 1));
      checkCudaErrors(cublasSaxpy(cublasHandle, N, &floatone, d_r, 1, d_p, 1));
    }

    checkCudaErrors(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA, vecp,
        &floatzero, vecomega, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    checkCudaErrors(cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot));
    alpha = r1 / dot;
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));
    nalpha = -alpha;
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
    r0 = r1;
    checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
  }

  printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

  checkCudaErrors(
      cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

  /* check result */
  err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++) {
      rsum += val[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs[i]);

    if (diff > err) {
      err = diff;
    }
  }

  printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
  nErrors += (k > max_iter) ? 1 : 0;
  qaerr1 = err;

  if (0) {
    // output result in matlab-style array
    int n = (int)sqrt((double)N);
    printf("a = [  ");

    for (int iy = 0; iy < n; iy++) {
      for (int ix = 0; ix < n; ix++) {
        printf(" %f ", x[iy * n + ix]);
      }

      if (iy == n - 1) {
        printf(" ]");
      }

      printf("\n");
    }
  }

  /* Preconditioned Conjugate Gradient using ILU.
     --------------------------------------------
     Follows the description by Golub & Van Loan,
     "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

  printf("\nConvergence of CG using ILU(0) preconditioning: \n");

  /* Perform analysis for ILU(0) */
  checkCudaErrors(cusparseScsrilu02_analysis(
      cusparseHandle, N, nz, descr, d_val, d_row, d_col, infoILU,
      CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));

  /* Copy A data to ILU(0) vals as input*/
  checkCudaErrors(cudaMemcpy(d_valsILU0, d_val, nz * sizeof(float),
                             cudaMemcpyDeviceToDevice));

  /* generate the ILU(0) factors */
  checkCudaErrors(cusparseScsrilu02(cusparseHandle, N, nz, descr, d_valsILU0,
                                    d_row, d_col, infoILU,
                                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));

  /* perform triangular solve analysis */
  checkCudaErrors(
      cusparseScsrsv2_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               N, nz, descrL, d_valsILU0, d_row, d_col, infoL,
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
  checkCudaErrors(
      cusparseScsrsv2_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               N, nz, descrU, d_valsILU0, d_row, d_col, infoU,
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));

  /* reset the initial guess of the solution to zero */
  for (int i = 0; i < N; i++) {
    x[i] = 0.0;
  }
  checkCudaErrors(
      cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));

  k = 0;
  checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

  while (r1 > tol * tol && k <= max_iter) {
    // preconditioner application: d_zm1 = U^-1 L^-1 d_r
    checkCudaErrors(cusparseScsrsv2_solve(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &floatone,
        descrL, d_valsILU0, d_row, d_col, infoL, d_r, d_y,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
    checkCudaErrors(cusparseScsrsv2_solve(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &floatone,
        descrU, d_valsILU0, d_row, d_col, infoU, d_y, d_zm1,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));

    k++;

    if (k == 1) {
      checkCudaErrors(cublasScopy(cublasHandle, N, d_zm1, 1, d_p, 1));
    } else {
      checkCudaErrors(
          cublasSdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
      checkCudaErrors(
          cublasSdot(cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator));
      beta = numerator / denominator;
      checkCudaErrors(cublasSscal(cublasHandle, N, &beta, d_p, 1));
      checkCudaErrors(
          cublasSaxpy(cublasHandle, N, &floatone, d_zm1, 1, d_p, 1));
    }

    checkCudaErrors(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA, vecp,
        &floatzero, vecomega, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
    checkCudaErrors(
        cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &denominator));
    alpha = numerator / denominator;
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));
    checkCudaErrors(cublasScopy(cublasHandle, N, d_r, 1, d_rm2, 1));
    checkCudaErrors(cublasScopy(cublasHandle, N, d_zm1, 1, d_zm2, 1));
    nalpha = -alpha;
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
    checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
  }

  printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

  checkCudaErrors(
      cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

  /* check result */
  err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++) {
      rsum += val[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs[i]);

    if (diff > err) {
      err = diff;
    }
  }

  printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
  nErrors += (k > max_iter) ? 1 : 0;
  qaerr2 = err;

  /* Destroy descriptors */
  checkCudaErrors(cusparseDestroyCsrsv2Info(infoU));
  checkCudaErrors(cusparseDestroyCsrsv2Info(infoL));
  checkCudaErrors(cusparseDestroyCsrilu02Info(infoILU));
  checkCudaErrors(cusparseDestroyMatDescr(descrL));
  checkCudaErrors(cusparseDestroyMatDescr(descrU));
  checkCudaErrors(cusparseDestroyMatDescr(descr));
  checkCudaErrors(cusparseDestroySpMat(matA));
  checkCudaErrors(cusparseDestroyDnVec(vecp));
  checkCudaErrors(cusparseDestroyDnVec(vecomega));

  /* Destroy contexts */
  checkCudaErrors(cusparseDestroy(cusparseHandle));
  checkCudaErrors(cublasDestroy(cublasHandle));

  /* Free device memory */
  free(I);
  free(J);
  free(val);
  free(x);
  free(rhs);
  checkCudaErrors(cudaFree(buffer));
  checkCudaErrors(cudaFree(d_col));
  checkCudaErrors(cudaFree(d_row));
  checkCudaErrors(cudaFree(d_val));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_r));
  checkCudaErrors(cudaFree(d_p));
  checkCudaErrors(cudaFree(d_omega));
  checkCudaErrors(cudaFree(d_valsILU0));
  checkCudaErrors(cudaFree(d_zm1));
  checkCudaErrors(cudaFree(d_zm2));
  checkCudaErrors(cudaFree(d_rm2));

  printf("\n");
  printf("Test Summary:\n");
  printf("   Counted total of %d errors\n", nErrors);
  printf("   qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
  exit((nErrors == 0 && fabs(qaerr1) < 1e-5 && fabs(qaerr2) < 1e-5
            ? EXIT_SUCCESS
            : EXIT_FAILURE));
}
