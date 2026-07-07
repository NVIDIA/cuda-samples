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

/**
 * Unit test for the scalarMul kernel.
 *
 * Tests the vector scalar multiplication kernel C[i] = A[i] * scalar
 * across various scalar values and vector sizes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

__global__ void scalarMul(const float *A, float scalar, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] * scalar;
    }
}

// ---------- Test helpers ----------

static int g_test_count  = 0;
static int g_pass_count  = 0;
static int g_fail_count  = 0;

#define TEST_ASSERT(cond, msg)                                       \
    do {                                                             \
        g_test_count++;                                              \
        if (cond) {                                                  \
            g_pass_count++;                                          \
            printf("  [PASS] %s\n", msg);                            \
        } else {                                                     \
            g_fail_count++;                                          \
            fprintf(stderr, "  [FAIL] %s\n", msg);                   \
        }                                                            \
    } while (0)

/**
 * Run one test: allocate vectors, fill input, launch kernel, verify results.
 * Returns 0 on success, non-zero on failure.
 */
static int run_test(int numElements, float scalar, const char *test_name)
{
    printf("\n--- Test: %s (n=%d, scalar=%.4f) ---\n", test_name, numElements, scalar);

    cudaError_t err          = cudaSuccess;
    size_t      size         = numElements * sizeof(float);
    int         threadsPerBlock = 256;
    int         blocksPerGrid  = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    int         verify_pass   = 0;

    // Allocate host vectors
    float *h_A = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_C == NULL) {
        fprintf(stderr, "  Failed to allocate host vectors!\n");
        free(h_A);
        free(h_C);
        return -1;
    }

    // Initialize input with known values
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = (float)(i + 1);  // 1, 2, 3, ...
    }

    // Allocate device vectors
    float *d_A = NULL, *d_C = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) { fprintf(stderr, "  cudaMalloc d_A failed!\n"); goto cleanup_host; }

    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) { fprintf(stderr, "  cudaMalloc d_C failed!\n"); goto cleanup_dA; }

    // H2D copy
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "  cudaMemcpy H2D failed!\n"); goto cleanup_all; }

    // Launch kernel
    scalarMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, scalar, d_C, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "  Kernel launch failed!\n"); goto cleanup_all; }

    // D2H copy
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "  cudaMemcpy D2H failed!\n"); goto cleanup_all; }

    // Verify all elements
    for (int i = 0; i < numElements; ++i) {
        float expected = h_A[i] * scalar;
        if (fabs(expected - h_C[i]) <= 1e-5f) {
            verify_pass++;
        }
    }
    TEST_ASSERT(verify_pass == numElements, test_name);

cleanup_all:
    cudaFree(d_C);
cleanup_dA:
    cudaFree(d_A);
cleanup_host:
    free(h_A);
    free(h_C);

    return (verify_pass == numElements) ? 0 : -1;
}

// ---------- Test cases ----------

/**
 * Test 1: scalar = 1.0 (identity — output should equal input)
 */
static void test_identity(void)
{
    run_test(100, 1.0f, "scalar=1.0 (identity)");
}

/**
 * Test 2: scalar = 0.0 (zero — output should all be 0)
 */
static void test_zero_scalar(void)
{
    run_test(100, 0.0f, "scalar=0.0 (zero output)");
}

/**
 * Test 3: scalar = -1.0 (negation)
 */
static void test_negation(void)
{
    run_test(100, -1.0f, "scalar=-1.0 (negation)");
}

/**
 * Test 4: scalar = 3.0 (positive multiplier)
 */
static void test_positive_scalar(void)
{
    run_test(50000, 3.0f, "scalar=3.0 (positive, large n)");
}

/**
 * Test 5: scalar = 0.001 (small multiplier — precision check)
 */
static void test_small_scalar(void)
{
    run_test(100, 0.001f, "scalar=0.001 (small)");
}

/**
 * Test 6: scalar = 1000000.0 (large multiplier)
 */
static void test_large_scalar(void)
{
    run_test(100, 1000000.0f, "scalar=1000000.0 (large)");
}

/**
 * Test 7: n = 1 (single element edge case)
 */
static void test_single_element(void)
{
    run_test(1, 5.0f, "n=1 (single element)");
}

/**
 * Test 8: n = 256 (exactly one block)
 */
static void test_one_block(void)
{
    run_test(256, 2.0f, "n=256 (exactly one block)");
}

/**
 * Test 9: n = 257 (just over one block — boundary)
 */
static void test_boundary(void)
{
    run_test(257, 2.0f, "n=257 (one block + 1 element)");
}

/**
 * Test 10: scalar = -0.5 (fractional negative)
 */
static void test_fractional_negative(void)
{
    run_test(100, -0.5f, "scalar=-0.5 (fractional negative)");
}

// ---------- Main ----------

int main(void)
{
    printf("========================================\n");
    printf("  scalarMul Unit Test\n");
    printf("========================================\n");

    test_identity();
    test_zero_scalar();
    test_negation();
    test_positive_scalar();
    test_small_scalar();
    test_large_scalar();
    test_single_element();
    test_one_block();
    test_boundary();
    test_fractional_negative();

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed, %d total\n", g_pass_count, g_fail_count, g_test_count);
    printf("========================================\n");

    if (g_fail_count > 0) {
        printf("  *** SOME TESTS FAILED ***\n");
        return EXIT_FAILURE;
    } else {
        printf("  *** ALL TESTS PASSED ***\n");
        return EXIT_SUCCESS;
    }
}
