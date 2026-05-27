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

/* This sample demonstrates two mdspan-centric features from CCCL 3.3:
 *
 *   1. DLPack <-> cuda::std::mdspan bridging through
 *      cuda::to_device_mdspan<T, Rank>(DLTensor)  ->  cuda::device_mdspan
 *      cuda::to_dlpack_tensor(device_mdspan)      ->  DLManagedTensor
 *      The DLPack format is the interchange protocol used by PyTorch,
 *      JAX, CuPy, and other frameworks; cuda::device_mdspan is the
 *      device-side view with rich shape/stride metadata for kernels.
 *
 *   2. cuda::shared_memory_mdspan: a multi-dimensional view over a
 *      shared-memory tile. The accessor guarantees shared-memory
 *      load/store instructions and adds address-space safety checks.
 *
 * A sample matrix is built on the device, wrapped in a DLTensor,
 * converted to a cuda::device_mdspan, and two kernels run against it:
 * scale_rows_kernel multiplies row i by (i + 1), and
 * shared_tile_transpose_kernel uses a cuda::shared_memory_mdspan to
 * transpose a block-sized tile through shared memory. The output
 * mdspan is then converted back to DLPack metadata and printed.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <helper_cuda.h>

/* Includes, cccl */
#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>

#define ROWS 8
#define COLS 8
#define TILE 8 /* matches ROWS / COLS for simplicity */

using extents2d = cuda::std::dextents<cuda::std::size_t, 2>;

/* Kernel A: multiply row i of a 2-D device_mdspan by (i + 1).  Templated
 * on the mdspan type so it accepts the exact type produced by
 * cuda::to_device_mdspan (which uses layout_stride_relaxed and int64_t
 * extents). */
template <typename Tensor>
__global__ void scale_rows_kernel(Tensor tensor)
{
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < static_cast<int>(tensor.extent(0)) && c < static_cast<int>(tensor.extent(1))) {
        tensor(r, c) *= static_cast<float>(r + 1);
    }
}

/* Kernel B: block-tile transpose driven by a shared_memory_mdspan.
 * Each block loads a TILE x TILE tile from the input into shared memory
 * through a cuda::shared_memory_mdspan, transposes in shared, and writes
 * to the output. */
template <typename InTensor, typename OutTensor>
__global__ void shared_tile_transpose_kernel(InTensor in, OutTensor out)
{
    __shared__ float smem_storage[TILE * TILE];
    cuda::shared_memory_mdspan smem(smem_storage, cuda::std::dextents<cuda::std::size_t, 2>{TILE, TILE});

    const int tr = threadIdx.y;
    const int tc = threadIdx.x;
    const int r  = blockIdx.y * TILE + tr;
    const int c  = blockIdx.x * TILE + tc;

    if (r < static_cast<int>(in.extent(0)) && c < static_cast<int>(in.extent(1))) {
        smem(tr, tc) = in(r, c);
    }
    __syncthreads();

    const int r_out = blockIdx.x * TILE + tr;
    const int c_out = blockIdx.y * TILE + tc;
    if (r_out < static_cast<int>(out.extent(0)) && c_out < static_cast<int>(out.extent(1))) {
        out(r_out, c_out) = smem(tc, tr);
    }
}

struct DLTensorStorage
{
    ::DLTensor                              tensor{};
    cuda::std::array<cuda::std::int64_t, 2> shape{};
    cuda::std::array<cuda::std::int64_t, 2> strides{};
};

static DLTensorStorage make_row_major_dltensor(float *device_ptr, int rows, int cols, int device_ordinal)
{
    DLTensorStorage s;
    s.shape              = {rows, cols};
    s.strides            = {cols, 1};
    s.tensor.data        = device_ptr;
    s.tensor.device      = ::DLDevice{::kDLCUDA, device_ordinal};
    s.tensor.ndim        = 2;
    s.tensor.dtype       = ::DLDataType{::DLDataTypeCode::kDLFloat, 32, 1};
    s.tensor.shape       = s.shape.data();
    s.tensor.strides     = s.strides.data();
    s.tensor.byte_offset = 0;
    return s;
}

int main(int argc, char **argv)
{
    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device: %s (Compute Capability %d.%d)\n\n", props.name, props.major, props.minor);

    float       *d_in  = nullptr;
    float       *d_out = nullptr;
    const size_t nelem = static_cast<size_t>(ROWS) * COLS;
    checkCudaErrors(cudaMalloc(&d_in, nelem * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_out, nelem * sizeof(float)));

    std::vector<float> host(nelem);
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            host[r * COLS + c] = static_cast<float>(r * COLS + c);
        }
    }
    checkCudaErrors(cudaMemcpy(d_in, host.data(), nelem * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_out, 0, nelem * sizeof(float)));

    DLTensorStorage in_dl  = make_row_major_dltensor(d_in, ROWS, COLS, devID);
    DLTensorStorage out_dl = make_row_major_dltensor(d_out, ROWS, COLS, devID);

    auto in_md  = cuda::to_device_mdspan<float, 2>(in_dl.tensor);
    auto out_md = cuda::to_device_mdspan<float, 2>(out_dl.tensor);

    printf("cuda::to_device_mdspan produced a 2-D device_mdspan of shape (%zu, %zu)\n\n",
           in_md.extent(0),
           in_md.extent(1));

    dim3 block(8, 8);
    dim3 grid((COLS + block.x - 1) / block.x, (ROWS + block.y - 1) / block.y);
    scale_rows_kernel<<<grid, block>>>(in_md);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<float> scaled(nelem);
    checkCudaErrors(cudaMemcpy(scaled.data(), d_in, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    bool scale_ok = true;
    for (int r = 0; r < ROWS && scale_ok; ++r) {
        for (int c = 0; c < COLS && scale_ok; ++c) {
            const float expect = static_cast<float>((r * COLS + c) * (r + 1));
            if (scaled[r * COLS + c] != expect) {
                printf("scale_rows mismatch at (%d,%d): got %g expected %g\n",
                       r,
                       c,
                       scaled[r * COLS + c],
                       expect);
                scale_ok = false;
            }
        }
    }
    if (scale_ok) {
        printf("scale_rows kernel: OK (row i scaled by i+1 via cuda::device_mdspan)\n");
    }

    cuda::device_mdspan<const float, extents2d> in_md_const(d_in, extents2d{ROWS, COLS});
    cuda::device_mdspan<float, extents2d>       out_md_rw(d_out, extents2d{ROWS, COLS});

    dim3 tile_block(TILE, TILE);
    dim3 tile_grid((COLS + TILE - 1) / TILE, (ROWS + TILE - 1) / TILE);
    shared_tile_transpose_kernel<<<tile_grid, tile_block>>>(in_md_const, out_md_rw);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<float> transposed(nelem);
    checkCudaErrors(cudaMemcpy(transposed.data(), d_out, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    bool tp_ok = true;
    for (int r = 0; r < ROWS && tp_ok; ++r) {
        for (int c = 0; c < COLS && tp_ok; ++c) {
            const float expect = scaled[c * COLS + r];
            if (transposed[r * COLS + c] != expect) {
                printf("transpose mismatch at (%d,%d): got %g expected %g\n",
                       r,
                       c,
                       transposed[r * COLS + c],
                       expect);
                tp_ok = false;
            }
        }
    }
    if (tp_ok) {
        printf("shared_tile_transpose kernel: OK (tile transpose via cuda::shared_memory_mdspan)\n");
    }

    auto        dl_wrapper = cuda::to_dlpack_tensor(out_md);
    const auto &dltensor   = dl_wrapper.get();
    printf("\ncuda::to_dlpack_tensor metadata:\n");
    printf("  device       : kDLCUDA (ordinal %d)\n", dltensor.device.device_id);
    printf("  ndim         : %d\n", dltensor.ndim);
    printf("  dtype        : code=%u bits=%u lanes=%u\n",
           static_cast<unsigned>(dltensor.dtype.code),
           static_cast<unsigned>(dltensor.dtype.bits),
           static_cast<unsigned>(dltensor.dtype.lanes));
    printf("  shape        : [%lld, %lld]\n",
           static_cast<long long>(dltensor.shape[0]),
           static_cast<long long>(dltensor.shape[1]));
    if (dltensor.strides != nullptr) {
        printf("  strides      : [%lld, %lld]\n",
               static_cast<long long>(dltensor.strides[0]),
               static_cast<long long>(dltensor.strides[1]));
    }

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    if (!scale_ok || !tp_ok) {
        return EXIT_FAILURE;
    }
    printf("\nDone\n");
    return EXIT_SUCCESS;
}
