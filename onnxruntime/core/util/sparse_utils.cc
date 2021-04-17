// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_csrcformat_rep.h"

#include "math_cpuonly.h"

#include <Eigen/SparseCore>

namespace onnxruntime {
namespace sparse_utils {
// StorageIndex must be int64_t, since our Csr/Csc formats currently require int64_t type
// for indexing, but this may change
template <class T>
using SparseMatrixRowMajor = Eigen::SparseMatrix<T, Eigen::RowMajor, int64_t>;

template <typename In>
struct TypeMap {
  using Out = In;
};

template <>
struct TypeMap<MLFloat16> {
  using Out = Eigen::half;
};

template <typename T>
struct ToCsrSparseConvert {
  Status operator()(const DataTransferManager& data_manager, const Tensor& src_cpu,
                    const AllocatorPtr& allocator, SparseTensor& dst) const {
    const auto* input_data = src_cpu.Data<T>();
    const auto& dense_shape = src_cpu.Shape();
    // We do not support a stack of matrices here
    ORT_RETURN_IF_NOT(dense_shape.NumDimensions() == 2, "Currently support two dim tensors");
    const auto M = dense_shape.GetDims()[0];
    const auto N = dense_shape.GetDims()[1];

    ConstEigenMatrixMapRowMajor<TypeMap<T>::Out> dense_map(reinterpret_cast<const TypeMap<T>::Out*>(input_data), M, N);
    // Quick way to convert.
    SparseMatrixRowMajor<TypeMap<T>::Out> sparse_matrix = dense_map.sparseView();
    sparse_matrix.makeCompressed();
    static_assert(sizeof(T) == sizeof(typename SparseMatrixRowMajor<T>::Scalar), "Expecting data type parity");
    static_assert(sizeof(int64_t) == sizeof(typename SparseMatrixRowMajor<T>::StorageIndex), "Expecting index type parity");
    static_assert(std::is_signed<int64_t>::value == std::is_signed<typename SparseMatrixRowMajor<T>::StorageIndex>::value,
                  "Indices must be both (un)signed");

    const auto nnz = sparse_matrix.nonZeros();

    TensorShape values_shape{nnz};
    TensorShape inner_shape{nnz};
    TensorShape outer_shape{M + 1};
    const OrtMemoryInfo& cpu_info = src_cpu.Location();
    Tensor values(src_cpu.DataType(), values_shape, sparse_matrix.valuePtr(), cpu_info);
    Tensor inner_indices(DataTypeImpl::GetType<int64_t>(), inner_shape, sparse_matrix.innerIndexPtr(), cpu_info);
    Tensor outer_indices(DataTypeImpl::GetType<int64_t>(), outer_shape, sparse_matrix.outerIndexPtr(), cpu_info);

    SparseTensor sparse_tensor(src_cpu.DataType(), dense_shape, nnz, allocator);
    SparseCsrcFormatRep* rep = nullptr;
    auto builder = sparse_tensor.RepBuilder<SparseCsrcBuilder>();
    ORT_RETURN_IF_ERROR(builder.GetOrCreate(SparseCsrcFormatRep::kRowMajor, inner_shape, outer_shape, rep));
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(values, sparse_tensor.MutableValues()));
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(inner_indices, rep->MutableInner()));
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(outer_indices, rep->MutableOuter()));

    dst = std::move(sparse_tensor);
    return Status::OK();
  }
};

common::Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src,
                                      const AllocatorPtr& cpu_allocator, const AllocatorPtr& allocator,
                                      SparseTensor& dst) {
  const auto num_dims = src.Shape().NumDimensions();
  if (num_dims > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Currently do not support dims higher than 2 dimensions");
  }

  // Eigen currently does not have BFloat16 support but it may be coming.
  utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                              int64_t, uint64_t, double, float, MLFloat16>
      t_disp(src.GetElementType());

  Status status;
  if (src.Location().device != cpu_allocator->Info().device) {
    Tensor src_cpu(src.DataType(), src.Shape(), cpu_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, src_cpu));
    status = t_disp.InvokeRet<common::Status, ToCsrSparseConvert>(data_manager, src_cpu, allocator, dst);
  } else {
    status = t_disp.InvokeRet<common::Status, ToCsrSparseConvert>(data_manager, src, allocator, dst);
  }

  return status;
}

template <typename T>
struct ConvertCsrToDense {
  Status operator()(const DataTransferManager& data_manager, const SparseTensor& cpu_tensor,
                    const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator, Tensor& dst) {
    const auto& dense_shape = cpu_tensor.Shape();
    const auto M = dense_shape.GetDims()[0];
    const auto N = dense_shape.GetDims()[1];
    const auto nnz = cpu_tensor.NumValues();

    const SparseCsrcFormatRep* rep = cpu_tensor.GetRep<SparseCsrcFormatRep>();
    ConstSparseMatrixMap<TypeMap<T>::Out> sparse_map(M, N, nnz,
                                                     rep->Outer().Data<int64_t>(),
                                                     rep->Inner().Data<int64_t>(),
                                                     reinterpret_cast<const TypeMap<T>::Out*>(cpu_tensor.Values().Data<T>()));

    // Convert to a dense tensor
    const AllocatorPtr& conversion_allocator = (cpu_tensor.Location().device == dst_allocator->Info().device) ?
                                                dst_allocator : cpu_allocator;
    Tensor cpu_result(cpu_tensor.DataType(), dense_shape, conversion_allocator);
    EigenMatrixMapRowMajor<TypeMap<T>::Out> result_map(reinterpret_cast<TypeMap<T>::Out*>(cpu_result.MutableData<T>()), 
                                                       M, N);
    result_map = sparse_map;

    if (cpu_tensor.Location().device == dst_allocator->Info().device) {
      dst = std::move(cpu_result);
    } else {
      Tensor dst_result(cpu_tensor.DataType(), dense_shape, dst_allocator);
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(cpu_result, dst_result));
      dst = std::move(dst_result);
    }

    return Status::OK();
  }
};

common::Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src,
                                      const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator,
                                      Tensor& dst) {
  if (!IsSet(src.FormatFlags(), SparseFormatFlags::kCsrc) ||
      src.GetRep<SparseCsrcFormatRep>()->Major() != SparseCsrcFormatRep::kRowMajor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input must be of CRS format");
  }

  if (src.Shape().NumDimensions() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Support 2-D matrices only");
  }

  const SparseCsrcFormatRep* rep = src.GetRep<SparseCsrcFormatRep>();
  const auto inner_num = rep->Inner().Shape().Size();
  const auto outer_num = rep->Outer().Shape().Size();
  ORT_ENFORCE(inner_num == src.NumValues(), "Expecting inner indecies to be same as nnz. Got: ", inner_num);
  ORT_ENFORCE(outer_num == (src.Shape().GetDims()[0] + 1), "Outer indecies must be M + 1. Got: ", outer_num);

  utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                              int64_t, uint64_t, double, float, MLFloat16>
      t_disp(src.GetElementType());

  Status status;
  if (src.Location().device != cpu_allocator->Info().device) {
    SparseTensor src_cpu(src.DataType(), src.Shape(), src.NumValues(), cpu_allocator);
    ORT_RETURN_IF_ERROR(src.Copy(data_manager, 0, src_cpu));
    status = t_disp.InvokeRet<Status, ConvertCsrToDense>(data_manager, src_cpu, cpu_allocator, dst_allocator, dst);
  } else {
    status = t_disp.InvokeRet<Status, ConvertCsrToDense>(data_manager, src, cpu_allocator, dst_allocator, dst);
  }

  return status;
}

}  // namespace sparse_utils
}  // namespace onnxruntime