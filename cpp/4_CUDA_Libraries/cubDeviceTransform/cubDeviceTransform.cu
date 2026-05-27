/* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

/* This sample demonstrates cub::DeviceTransform in its N-input/M-output
 * form (extended in CCCL 3.3). A single device-wide call reads from
 * N input sequences and writes to M output sequences, driven by a
 * user-provided op that returns a tuple of M values. Two cases are
 * shown: N=3 -> 1 and N=2 -> 2. Results are verified against a host
 * reference.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <helper_cuda.h>

/* Includes, cccl */
#include <cub/device/device_transform.cuh>
#include <cuda/iterator>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

static bool run_n_to_one_transform()
{
    /* result[i] = (a[i] + b[i]) * c[i], with c = counting_iterator<int>(100). */
    thrust::device_vector<int>   a        = {0, -2, 5, 3};
    thrust::device_vector<float> b        = {5.2f, 3.1f, -1.1f, 3.0f};
    auto                         counting = cuda::counting_iterator<int>{100};
    thrust::device_vector<int>   result(a.size());

    auto op = [] __host__ __device__(int x, float y, int z) -> int {
        return static_cast<int>((x + y) * z);
    };

    checkCudaErrors(cub::DeviceTransform::Transform(
        cuda::std::tuple{a.begin(), b.begin(), counting}, result.begin(), a.size(), op));
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::host_vector<int>   ha  = a;
    thrust::host_vector<float> hb  = b;
    thrust::host_vector<int>   got = result;
    std::vector<int>           expected(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        expected[i] = static_cast<int>((ha[i] + hb[i]) * static_cast<int>(100 + i));
    }

    bool ok = true;
    printf("cub::DeviceTransform::Transform (N=3 inputs -> 1 output)\n");
    printf("  result = (a + b) * c with c = counting_iterator(100)\n");
    printf("  got      = {");
    for (size_t i = 0; i < got.size(); ++i) {
        printf(" %d", got[i]);
        if (got[i] != expected[i])
            ok = false;
    }
    printf(" }\n  expected = {");
    for (size_t i = 0; i < expected.size(); ++i)
        printf(" %d", expected[i]);
    printf(" }  %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_n_to_m_transform()
{
    /* (sum[i], diff[i]) = (a[i] + b[i], a[i] - b[i]) in one pass. */
    thrust::device_vector<int> a = {1, 5, 10, 7, 3};
    thrust::device_vector<int> b = {4, 2, 8, 1, 9};
    thrust::device_vector<int> sum(a.size());
    thrust::device_vector<int> diff(a.size());

    auto op = [] __host__ __device__(int x, int y) -> cuda::std::tuple<int, int> {
        return {x + y, x - y};
    };

    checkCudaErrors(cub::DeviceTransform::Transform(cuda::std::tuple{a.begin(), b.begin()},
                                                    cuda::std::tuple{sum.begin(), diff.begin()},
                                                    a.size(),
                                                    op));
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::host_vector<int> ha = a, hb = b, got_sum = sum, got_diff = diff;
    std::vector<int>         exp_sum(a.size()), exp_diff(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        exp_sum[i]  = ha[i] + hb[i];
        exp_diff[i] = ha[i] - hb[i];
    }

    bool ok = true;
    printf("cub::DeviceTransform::Transform (N=2 inputs -> M=2 outputs)\n");
    printf("  op returns cuda::std::tuple{a + b, a - b}\n");
    printf("  sum  = {");
    for (size_t i = 0; i < got_sum.size(); ++i) {
        printf(" %d", got_sum[i]);
        if (got_sum[i] != exp_sum[i])
            ok = false;
    }
    printf(" }  expected = {");
    for (auto v : exp_sum)
        printf(" %d", v);
    printf(" }\n  diff = {");
    for (size_t i = 0; i < got_diff.size(); ++i) {
        printf(" %d", got_diff[i]);
        if (got_diff[i] != exp_diff[i])
            ok = false;
    }
    printf(" }  expected = {");
    for (auto v : exp_diff)
        printf(" %d", v);
    printf(" }  %s\n", ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char **argv)
{
    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device: %s (Compute Capability %d.%d)\n\n", props.name, props.major, props.minor);

    bool ok = true;
    ok &= run_n_to_one_transform();
    printf("\n");
    ok &= run_n_to_m_transform();

    printf("\n%s\n", ok ? "Done" : "FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
