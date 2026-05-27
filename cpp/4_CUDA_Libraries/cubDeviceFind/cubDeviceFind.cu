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

/* This sample demonstrates the three device-wide search algorithms
 * introduced in CCCL 3.3: cub::DeviceFind::FindIf for predicate search,
 * and cub::DeviceFind::LowerBound / UpperBound for parallel binary
 * search. Results are verified against std::find_if, std::lower_bound,
 * and std::upper_bound on the host.
 */

/* Includes, system */
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <helper_cuda.h>

/* Includes, cccl */
#include <cub/device/device_find.cuh>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/* Predicate used with cub::DeviceFind::FindIf. */
struct is_greater_than_t
{
    int threshold;
    __host__ __device__ bool operator()(int value) const { return value > threshold; }
};

static bool run_find_if()
{
    /* Input: 0, 1, ..., 15.  Predicate: value > 9.  Expected index: 10. */
    const int                  num_items = 16;
    thrust::device_vector<int> d_in(num_items);
    for (int i = 0; i < num_items; ++i)
        d_in[i] = i;
    thrust::device_vector<int> d_out(1);
    is_greater_than_t          predicate{9};

    size_t temp_bytes = 0;
    checkCudaErrors(
        cub::DeviceFind::FindIf(nullptr, temp_bytes, d_in.begin(), d_out.begin(), predicate, num_items));
    thrust::device_vector<char> temp(temp_bytes);
    checkCudaErrors(cub::DeviceFind::FindIf(thrust::raw_pointer_cast(temp.data()),
                                            temp_bytes,
                                            d_in.begin(),
                                            d_out.begin(),
                                            predicate,
                                            num_items));
    checkCudaErrors(cudaDeviceSynchronize());

    const int got = d_out[0];

    thrust::host_vector<int> h_in       = d_in;
    auto                     host_it    = std::find_if(h_in.begin(), h_in.end(),
                                   [&](int v) { return v > predicate.threshold; });
    const int                expected   = static_cast<int>(host_it - h_in.begin());

    printf("cub::DeviceFind::FindIf(value > %d) over [0..%d)\n", predicate.threshold, num_items);
    printf("  got index = %d, expected = %d  %s\n", got, expected, (got == expected ? "OK" : "FAIL"));
    return got == expected;
}

static bool run_lower_bound()
{
    /* Sorted range: [0, 2, 4, 6, 8].  Values to locate: [1, 3, 5, 7]. */
    thrust::device_vector<int> d_range  = {0, 2, 4, 6, 8};
    thrust::device_vector<int> d_values = {1, 3, 5, 7};
    thrust::device_vector<int> d_out(d_values.size());

    size_t temp_bytes = 0;
    checkCudaErrors(cub::DeviceFind::LowerBound(nullptr,
                                                temp_bytes,
                                                d_range.begin(),
                                                static_cast<int>(d_range.size()),
                                                d_values.begin(),
                                                static_cast<int>(d_values.size()),
                                                d_out.begin(),
                                                cuda::std::less{}));
    thrust::device_vector<char> temp(temp_bytes);
    checkCudaErrors(cub::DeviceFind::LowerBound(thrust::raw_pointer_cast(temp.data()),
                                                temp_bytes,
                                                d_range.begin(),
                                                static_cast<int>(d_range.size()),
                                                d_values.begin(),
                                                static_cast<int>(d_values.size()),
                                                d_out.begin(),
                                                cuda::std::less{}));
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::host_vector<int> h_range  = d_range;
    thrust::host_vector<int> h_values = d_values;
    thrust::host_vector<int> got      = d_out;
    std::vector<int>         expected(h_values.size());
    for (size_t i = 0; i < h_values.size(); ++i) {
        expected[i] = static_cast<int>(
            std::lower_bound(h_range.begin(), h_range.end(), h_values[i]) - h_range.begin());
    }

    bool ok = true;
    printf("cub::DeviceFind::LowerBound\n");
    printf("  range   = { 0, 2, 4, 6, 8 }\n");
    printf("  values  = { 1, 3, 5, 7 }\n");
    printf("  got     = {");
    for (size_t i = 0; i < got.size(); ++i) {
        printf(" %d", got[i]);
        if (got[i] != expected[i])
            ok = false;
    }
    printf(" }\n  expect  = {");
    for (size_t i = 0; i < expected.size(); ++i)
        printf(" %d", expected[i]);
    printf(" }  %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_upper_bound()
{
    /* Range with duplicates so LowerBound and UpperBound differ on values
     * that appear in the range. */
    thrust::device_vector<int> d_range  = {0, 2, 2, 4, 6, 8};
    thrust::device_vector<int> d_values = {2, 2};
    thrust::device_vector<int> d_lb(d_values.size());
    thrust::device_vector<int> d_ub(d_values.size());

    size_t temp_bytes_lb = 0;
    checkCudaErrors(cub::DeviceFind::LowerBound(nullptr,
                                                temp_bytes_lb,
                                                d_range.begin(),
                                                static_cast<int>(d_range.size()),
                                                d_values.begin(),
                                                static_cast<int>(d_values.size()),
                                                d_lb.begin(),
                                                cuda::std::less{}));
    thrust::device_vector<char> temp_lb(temp_bytes_lb);
    checkCudaErrors(cub::DeviceFind::LowerBound(thrust::raw_pointer_cast(temp_lb.data()),
                                                temp_bytes_lb,
                                                d_range.begin(),
                                                static_cast<int>(d_range.size()),
                                                d_values.begin(),
                                                static_cast<int>(d_values.size()),
                                                d_lb.begin(),
                                                cuda::std::less{}));

    size_t temp_bytes_ub = 0;
    checkCudaErrors(cub::DeviceFind::UpperBound(nullptr,
                                                temp_bytes_ub,
                                                d_range.begin(),
                                                static_cast<int>(d_range.size()),
                                                d_values.begin(),
                                                static_cast<int>(d_values.size()),
                                                d_ub.begin(),
                                                cuda::std::less{}));
    thrust::device_vector<char> temp_ub(temp_bytes_ub);
    checkCudaErrors(cub::DeviceFind::UpperBound(thrust::raw_pointer_cast(temp_ub.data()),
                                                temp_bytes_ub,
                                                d_range.begin(),
                                                static_cast<int>(d_range.size()),
                                                d_values.begin(),
                                                static_cast<int>(d_values.size()),
                                                d_ub.begin(),
                                                cuda::std::less{}));
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::host_vector<int> h_range  = d_range;
    thrust::host_vector<int> h_values = d_values;
    thrust::host_vector<int> got_lb   = d_lb;
    thrust::host_vector<int> got_ub   = d_ub;
    std::vector<int>         exp_lb(h_values.size());
    std::vector<int>         exp_ub(h_values.size());
    for (size_t i = 0; i < h_values.size(); ++i) {
        exp_lb[i] =
            static_cast<int>(std::lower_bound(h_range.begin(), h_range.end(), h_values[i]) - h_range.begin());
        exp_ub[i] =
            static_cast<int>(std::upper_bound(h_range.begin(), h_range.end(), h_values[i]) - h_range.begin());
    }

    bool ok = true;
    printf("cub::DeviceFind::UpperBound (with duplicates in range)\n");
    printf("  range   = { 0, 2, 2, 4, 6, 8 }\n");
    printf("  values  = { 2, 2 }\n");
    printf("  lb      = {");
    for (size_t i = 0; i < got_lb.size(); ++i) {
        printf(" %d", got_lb[i]);
        if (got_lb[i] != exp_lb[i])
            ok = false;
    }
    printf(" }  expected = {");
    for (size_t i = 0; i < exp_lb.size(); ++i)
        printf(" %d", exp_lb[i]);
    printf(" }\n  ub      = {");
    for (size_t i = 0; i < got_ub.size(); ++i) {
        printf(" %d", got_ub[i]);
        if (got_ub[i] != exp_ub[i])
            ok = false;
    }
    printf(" }  expected = {");
    for (size_t i = 0; i < exp_ub.size(); ++i)
        printf(" %d", exp_ub[i]);
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
    ok &= run_find_if();
    printf("\n");
    ok &= run_lower_bound();
    printf("\n");
    ok &= run_upper_bound();

    printf("\n%s\n", ok ? "Done" : "FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
