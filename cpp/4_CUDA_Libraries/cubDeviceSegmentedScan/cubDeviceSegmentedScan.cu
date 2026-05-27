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

/* This sample demonstrates cub::DeviceSegmentedScan, added in CCCL 3.3.
 * Two operations are shown: ExclusiveSegmentedSum across three independent
 * segments, and InclusiveSegmentedScan with a custom binary operator
 * (running maximum via cuda::maximum<>). Each is verified against a
 * host reference implementation.
 */

/* Includes, system */
#include <algorithm>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <helper_cuda.h>

/* Includes, cccl */
#include <cub/device/device_segmented_scan.cuh>
#include <cuda/functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T>
static void print_vec(const char *label, const std::vector<T> &v)
{
    printf("  %-24s{", label);
    for (size_t i = 0; i < v.size(); ++i)
        printf(" %d", static_cast<int>(v[i]));
    printf(" }\n");
}

static std::vector<int> host_exclusive_segmented_sum(const std::vector<int> &input, const std::vector<size_t> &offsets)
{
    std::vector<int> out(input.size(), 0);
    for (size_t s = 0; s + 1 < offsets.size(); ++s) {
        int running = 0;
        for (size_t i = offsets[s]; i < offsets[s + 1]; ++i) {
            out[i] = running;
            running += input[i];
        }
    }
    return out;
}

static std::vector<int> host_inclusive_segmented_max(const std::vector<int> &input, const std::vector<size_t> &offsets)
{
    std::vector<int> out(input.size(), 0);
    for (size_t s = 0; s + 1 < offsets.size(); ++s) {
        int running = std::numeric_limits<int>::min();
        for (size_t i = offsets[s]; i < offsets[s + 1]; ++i) {
            running = std::max(running, input[i]);
            out[i]  = running;
        }
    }
    return out;
}

static bool run_exclusive_segmented_sum()
{
    /* 3 segments: [1,2,3] [4,5] [6,7,8] -> [0,1,3] [0,4] [0,6,13]. */
    thrust::device_vector<int>    d_in      = {1, 2, 3, 4, 5, 6, 7, 8};
    thrust::device_vector<size_t> d_offsets = {0, 3, 5, 8};
    thrust::device_vector<int>    d_out(d_in.size());

    const auto num_segments  = d_offsets.size() - 1;
    auto       begin_offsets = d_offsets.begin();
    auto       end_offsets   = d_offsets.begin() + 1;

    size_t temp_bytes = 0;
    checkCudaErrors(cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
        nullptr, temp_bytes, d_in.begin(), d_out.begin(), begin_offsets, end_offsets, num_segments));
    thrust::device_vector<char> temp(temp_bytes);
    checkCudaErrors(cub::DeviceSegmentedScan::ExclusiveSegmentedSum(thrust::raw_pointer_cast(temp.data()),
                                                                    temp_bytes,
                                                                    d_in.begin(),
                                                                    d_out.begin(),
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    num_segments));
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<int>    h_in(d_in.begin(), d_in.end());
    std::vector<size_t> h_off(d_offsets.begin(), d_offsets.end());
    std::vector<int>    got(d_out.begin(), d_out.end());
    std::vector<int>    expected = host_exclusive_segmented_sum(h_in, h_off);

    printf("cub::DeviceSegmentedScan::ExclusiveSegmentedSum\n");
    print_vec("input:", h_in);
    printf("  %-24s{", "offsets:");
    for (auto o : h_off)
        printf(" %zu", o);
    printf(" }\n");
    print_vec("got:", got);
    print_vec("expected:", expected);
    const bool ok = got == expected;
    printf("  %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_inclusive_segmented_max()
{
    /* Same three segments, but compute running max per segment. */
    thrust::device_vector<int>    d_in      = {3, 1, 4, 5, 2, 9, 7, 8};
    thrust::device_vector<size_t> d_offsets = {0, 3, 5, 8};
    thrust::device_vector<int>    d_out(d_in.size());

    const auto num_segments  = d_offsets.size() - 1;
    auto       begin_offsets = d_offsets.begin();
    auto       end_offsets   = d_offsets.begin() + 1;

    auto max_op = [] __host__ __device__(int a, int b) -> int { return cuda::maximum<>{}(a, b); };

    size_t temp_bytes = 0;
    checkCudaErrors(cub::DeviceSegmentedScan::InclusiveSegmentedScan(
        nullptr, temp_bytes, d_in.begin(), d_out.begin(), begin_offsets, end_offsets, num_segments, max_op));
    thrust::device_vector<char> temp(temp_bytes);
    checkCudaErrors(cub::DeviceSegmentedScan::InclusiveSegmentedScan(thrust::raw_pointer_cast(temp.data()),
                                                                     temp_bytes,
                                                                     d_in.begin(),
                                                                     d_out.begin(),
                                                                     begin_offsets,
                                                                     end_offsets,
                                                                     num_segments,
                                                                     max_op));
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<int>    h_in(d_in.begin(), d_in.end());
    std::vector<size_t> h_off(d_offsets.begin(), d_offsets.end());
    std::vector<int>    got(d_out.begin(), d_out.end());
    std::vector<int>    expected = host_inclusive_segmented_max(h_in, h_off);

    printf("cub::DeviceSegmentedScan::InclusiveSegmentedScan (running max)\n");
    print_vec("input:", h_in);
    printf("  %-24s{", "offsets:");
    for (auto o : h_off)
        printf(" %zu", o);
    printf(" }\n");
    print_vec("got:", got);
    print_vec("expected:", expected);
    const bool ok = got == expected;
    printf("  %s\n", ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char **argv)
{
    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device: %s (Compute Capability %d.%d)\n\n", props.name, props.major, props.minor);

    bool ok = true;
    ok &= run_exclusive_segmented_sum();
    printf("\n");
    ok &= run_inclusive_segmented_max();

    printf("\n%s\n", ok ? "Done" : "FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
