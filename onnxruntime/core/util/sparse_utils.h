// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {
class Tensor;
class SparseTensor;
class DataTransferManager;
namespace common {
class Status;
}

namespace sparse_utils {
/// <summary>
/// This function converts dense tensor into Csr format using Eigen sparse matrices
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src">dense tensor</param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator">destination allocator</param>
/// <param name="dst">output</param>
/// <returns>Status</returns>
common::Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                                      const AllocatorPtr& dst_allocator, SparseTensor& dst);

/// <summary>
/// Converts Csr format to Dense matrix
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src">SparseTensor</param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator">destination allocator</param>
/// <param name="dst">out parameter</param>
/// <returns>Status</returns>
common::Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                                      const AllocatorPtr& dst_allocator, Tensor& dst);
}  // namespace sparse_utils
}  // namespace onnxruntime
