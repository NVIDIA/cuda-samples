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

#pragma once
#include <cuda.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////
//! Allocate virtually contiguous memory backed on separate devices
//! @return CUresult error code on failure.
//! @param[out] dptr            Virtual address reserved for allocation
//! @param[out] allocationSize  Actual amount of virtual address space reserved.
//!                             AllocationSize is needed in the free operation.
//! @param[in] size             The minimum size to allocate (will be rounded up
//! to accomodate
//!                             required granularity).
//! @param[in] residentDevices  Specifies what devices the allocation should be
//! striped across.
//! @param[in] mappingDevices   Specifies what devices need to read/write to the
//! allocation.
//! @align                      Additional allignment requirement if desired.
//! @note       The VA mappings will look like the following:
//!
//!     v-stripeSize-v                v-rounding -v
//!     +-----------------------------------------+
//!     |      D1     |      D2     |      D3     |
//!     +-----------------------------------------+
//!     ^-- dptr                      ^-- dptr + size
//!
//! Each device in the residentDevices list will get an equal sized stripe.
//! Excess memory allocated will be  that meets the minimum
//! granularity requirements of all the devices.
//!
//! @note uses cuMemGetAllocationGranularity cuMemCreate cuMemMap and
//! cuMemSetAccess
//!   function calls to organize the va space
//!
//! @note uses cuMemRelease to release the allocationHandle.  The allocation
//! handle
//!   is not needed after its mappings are set up.
////////////////////////////////////////////////////////////////////////////
CUresult simpleMallocMultiDeviceMmap(
    CUdeviceptr *dptr, size_t *allocationSize, size_t size,
    const std::vector<CUdevice> &residentDevices,
    const std::vector<CUdevice> &mappingDevices, size_t align = 0);

////////////////////////////////////////////////////////////////////////////
//! Frees resources allocated by simpleMallocMultiDeviceMmap
//! @CUresult CUresult error code on failure.
//! @param[in] dptr  Virtual address reserved by simpleMallocMultiDeviceMmap
//! @param[in] size  allocationSize returned by simpleMallocMultiDeviceMmap
////////////////////////////////////////////////////////////////////////////
CUresult simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size);
