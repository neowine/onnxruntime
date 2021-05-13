// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/math/matmul_integer_base.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {

class MatMulIntegerToFloatBase : public MatMulIntegerBase {
 public:
  MatMulIntegerToFloatBase(const OpKernelInfo& info) : MatMulIntegerBase(info) {
  }

  enum OutputTensors : int { OUT_Y = 0 };

 protected:
  Status ComputeCommon(OpKernelContext* ctx,
                       const uint8_t* a_data,
                       const TensorShape& a_shape,
                       float a_scale,
                       uint8_t a_zp,
                       const Tensor* b_tensor,
                       const Tensor* b_scale,
                       const Tensor* b_zp,
                       const Tensor* bias_tensor) const;
};

Status MatMulIntegerToFloatBase::ComputeCommon(OpKernelContext* ctx,
                                               const uint8_t* A_data,
                                               const TensorShape& A_shape,
                                               float A_scale,
                                               uint8_t A_zp,
                                               const Tensor* B_tensor,
                                               const Tensor* B_scale_tensor,
                                               const Tensor* B_zp_tensor,
                                               const Tensor* bias_tensor) const {
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A_shape, packed_b_ ? b_shape_ : B_tensor->Shape()));
  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->template MutableData<float>();
  const auto* bias_data = bias_tensor != nullptr ? bias_tensor->Data<float>() : nullptr;

  // process zero point of B
  bool is_b_zp_per_column = false;
  uint8_t b_zp_default = 0;
  const uint8_t* b_zp_ptr = &b_zp_default;
  if (nullptr != B_zp_tensor) {
    is_b_zp_per_column = !IsScalarOr1ElementVector(B_zp_tensor);
    b_zp_ptr = static_cast<const uint8_t*>(B_zp_tensor->DataRaw());
  }

  // process scale of B
  bool is_b_scale_per_column = false;
  float multiplier_per_tensor = A_scale;
  const float* b_scale_data = &multiplier_per_tensor;
  std::vector<float> multipliers_per_column;
  if (nullptr != B_scale_tensor) {
    is_b_scale_per_column = !IsScalarOr1ElementVector(B_scale_tensor);
    const float* b_scale_tensor_data = B_scale_tensor->Data<float>();

    if (is_b_scale_per_column) {
      multipliers_per_column.reserve(B_scale_tensor->Shape().Size());
      std::transform(b_scale_tensor_data,
                     b_scale_tensor_data + B_scale_tensor->Shape().Size(),
                     std::back_inserter(multipliers_per_column),
                     [&A_scale](float B_scale) {
                       return A_scale * B_scale;
                     });
      b_scale_data = multipliers_per_column.data();
    } else {
      multiplier_per_tensor *= *b_scale_tensor_data;
    }
  }

  // batch gemm
  MLAS_GEMM_U8X8_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.BIsSigned = packed_b_ ? b_is_signed_ : B_tensor->IsDataType<int8_t>();

  const size_t num_gemms = helper.OutputOffsets().size();
  std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> gemm_scale_procs;
  gemm_scale_procs.reserve(num_gemms);
  std::vector<MLAS_GEMM_U8X8_DATA_PARAMS> gemm_data_vec(num_gemms);

  for (size_t gemm_idx = 0; gemm_idx < num_gemms; gemm_idx++) {
    int64_t scale_zp_offset = helper.RightOffsets()[gemm_idx] / helper.K();
    int64_t scale_offset = is_b_scale_per_column ? scale_zp_offset : 0;
    int64_t zp_offset = is_b_zp_per_column ? scale_zp_offset : 0;
    gemm_scale_procs.emplace_back(y_data + helper.OutputOffsets()[gemm_idx],
                                  gemm_shape.N,
                                  b_scale_data + scale_offset,
                                  bias_data,
                                  MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                  is_b_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
    auto& params = gemm_data_vec[gemm_idx];
    params.OutputProcessor = &(gemm_scale_procs[gemm_idx]);
    params.A = A_data + helper.LeftOffsets()[gemm_idx];
    params.lda = gemm_shape.K;
    params.ZeroPointA = A_zp;
    params.BIsPacked = bool(packed_b_);
    params.B = bool(packed_b_) ? packed_b_.get() : static_cast<const uint8_t*>(B_tensor->DataRaw()) + helper.RightOffsets()[gemm_idx];
    params.ldb = gemm_shape.N;
    params.ZeroPointB = b_zp_ptr + zp_offset;
    params.PerColumnZeroPoints = is_b_zp_per_column;
    params.C = reinterpret_cast<int32_t*>(y_data + helper.OutputOffsets()[gemm_idx]);
    params.ldc = gemm_shape.N;
  }

  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms, ctx->GetOperatorThreadPool());

  return Status::OK();
}

class DynamicQuantizeMatMul final : public MatMulIntegerToFloatBase {
 public:
  DynamicQuantizeMatMul(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_B_SCALE = 2,
    IN_B_ZERO_POINT = 3,
    IN_BIAS = 4
  };

 protected:
  int GetBIdx() override { return IN_B; }
};

class MatMulIntegerToFloat final : public MatMulIntegerToFloatBase {
 public:
  MatMulIntegerToFloat(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_SCALE = 2,
    IN_B_SCALE = 3,
    IN_A_ZERO_POINT = 4,
    IN_B_ZERO_POINT = 5,
    IN_BIAS = 6
  };

 protected:
  int GetBIdx() override { return IN_B; }

 private:
  // A scale and B scale may be switched in fusion stage because of lack of shape information.
  // Fix them up before computation.
  static void FixupScaleTensor(const Tensor*& a_scale_tensor, const Tensor*& b_scale_tensor);
};

Status DynamicQuantizeMatMul::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  const Tensor* b_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);

  // calculate quantization parameter of A
  const float* a_data = a->template Data<float>();
  int64_t num_of_elements = a->Shape().Size();

  float a_scale;
  uint8_t a_zero_point;
  GetQuantizationParameter(a_data, num_of_elements, a_scale, a_zero_point, ctx->GetOperatorThreadPool());

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  uint8_t* a_data_quant = static_cast<uint8_t*>(allocator->Alloc(SafeInt<size_t>(num_of_elements) * sizeof(uint8_t)));
  BufferUniquePtr a_buffer_quant_holder(a_data_quant, BufferDeleter(allocator));

  ParQuantizeLinear(a_data, a_data_quant, num_of_elements, a_scale, a_zero_point, ctx->GetOperatorThreadPool());

  return ComputeCommon(ctx,
                       a_data_quant,
                       a->Shape(),
                       a_scale,
                       a_zero_point,
                       b,
                       b_scale_tensor,
                       b_zp_tensor,
                       ctx->Input<Tensor>(IN_BIAS));
}

void MatMulIntegerToFloat::FixupScaleTensor(const Tensor*& A_scale_tensor, const Tensor*& B_scale_tensor) {
  const TensorShape A_scale_shape = A_scale_tensor->Shape();
  const TensorShape B_scale_shape = B_scale_tensor->Shape();
  if (!IsScalarOr1ElementVector(A_scale_tensor)) {
    size_t A_scale_rank = A_scale_shape.NumDimensions();
    if (A_scale_rank == 1 || A_scale_shape[A_scale_rank - 1] != 1) {
      std::swap(A_scale_tensor, B_scale_tensor);
    }
  } else if (!IsScalarOr1ElementVector(B_scale_tensor)) {
    size_t B_scale_rank = B_scale_shape.NumDimensions();
    if (B_scale_rank > 1 && B_scale_shape[B_scale_rank - 2] != 1) {
      std::swap(A_scale_tensor, B_scale_tensor);
    }
  }
}

template <typename T>
void ScaleOutput(const Tensor& scale, Tensor& output) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.ScalarInput0<T>() * per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().array() * per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().cwiseProduct(per_iter_bh.EigenInput1<T>());
      }};

  InputBroadcaster input_broadcaster(scale, output);
  OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(),
                                       output);
  BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster);

  BroadcastLooper(broadcast_helper, funcs);
}

// A standard Scale of Matrix B is in one of the formats below:
// 1. Scalar
// 2. 1D tensor with size 1
// 3. 1D tensor with size equal to B_Shape last dimension and B_Shape is a 2D tensor
// 4. Equal to B_shape except that second to last is 1
static bool IsBScaleStandard(const TensorShape& B_scale_tensor, const TensorShape& B_Shape) {
  int64_t B_scale_rank = B_scale_tensor.NumDimensions();
  int64_t B_shape_rank = B_Shape.NumDimensions();
  if (B_scale_rank == 0 ||                                //scalar
      B_scale_rank == 1 && B_scale_tensor.Size() == 1) {  // 1D tensor with size 1
    return true;
  }

  if (B_scale_rank == 1 &&
      B_shape_rank == 2 &&
      B_scale_tensor[B_scale_rank - 1] == B_Shape[B_shape_rank - 1]) {
    return true;
  }

  if (B_scale_rank != B_shape_rank ||
      B_scale_rank <= 1 ||
      B_scale_tensor[B_scale_rank - 2] != 1) {
    return false;
  }

  for (int64_t rank = 0; rank < B_scale_rank; rank++) {
    if (rank != B_scale_rank - 2 &&
        B_scale_tensor[rank] != B_Shape[rank]) {
      return false;
    }
  }

  return true;
}

Status MatMulIntegerToFloat::Compute(OpKernelContext* ctx) const {
  const Tensor* A = ctx->Input<Tensor>(IN_A);
  const Tensor* B = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  const Tensor* A_scale_tensor = ctx->Input<Tensor>(IN_A_SCALE);
  const Tensor* B_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  FixupScaleTensor(A_scale_tensor, B_scale_tensor);
  bool is_A_scale_scalar = IsScalarOr1ElementVector(A_scale_tensor);
  bool is_B_scale_standard = IsBScaleStandard(B_scale_tensor->Shape(), nullptr != B ? B->Shape() : b_shape_);

  // validate zero point of A
  uint8_t A_zero_point = 0;
  const Tensor* A_zero_point_tensor = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  if (A_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(A_zero_point_tensor),
                "MatMulIntegerToFloat : input A zero point must be A scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    A_zero_point = *A_zero_point_tensor->Data<uint8_t>();
  }

  const Tensor* B_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  ORT_RETURN_IF_ERROR(ComputeCommon(
      ctx,
      A->Data<uint8_t>(),
      A->Shape(),
      is_A_scale_scalar ? *A_scale_tensor->template Data<float>() : 1.f,
      A_zero_point,
      B,
      is_B_scale_standard ? B_scale_tensor : nullptr,
      B_zp_tensor,
      ctx->Input<Tensor>(IN_BIAS)));

  if (!is_A_scale_scalar) {
    ScaleOutput<float>(*A_scale_tensor, *ctx->Output<Tensor>(0));
  }
  if (!is_B_scale_standard) {
    ScaleOutput<float>(*B_scale_tensor, *ctx->Output<Tensor>(0));
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DynamicQuantizeMatMul,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
    DynamicQuantizeMatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

}  // namespace contrib
}  // namespace onnxruntime
