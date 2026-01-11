#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "device/b2b_gemm.h"
#include "b2b_gemm_run.h"
#include "test_run.h"

////////////////////////////////////////////////////////////////////////////////

cutlass::gemm::GemmCoord gemm_f16_sm90_problem_size_0(128*640, 64, 576);
cutlass::gemm::GemmCoord gemm_f16_sm90_problem_size_1(128*640, 128, 64);

bool run_nonfused_gemm_f16_sm90() {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(1); //beta=1 for bias
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using Gemm0 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;
  using Gemm1 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;

  B2bNonFusedGemmRun<Gemm0, Gemm1> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back FP16 TN GEMMs...\n";
  bool pass = nonFusedGemm.run(gemm_f16_sm90_problem_size_0, gemm_f16_sm90_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

bool run_fused_gemm_f16_sm90_rf_res() {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0); 
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      InstructionShape::kM * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;

  B2bFusedGemmRun<B2bGemm> fusedGemm;

  std::cout << "Running Fused back-to-back FP16 TN GEMMs with RF residency...\n";
  bool passed = fusedGemm.run(gemm_f16_sm90_problem_size_0, gemm_f16_sm90_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;

}

int main() {

  std::vector<bool (*)()>funcs = {
    &run_nonfused_gemm_f16_sm90,
    &run_fused_gemm_f16_sm90_rf_res
  };

  return testRun(90, funcs, "gemm f16 RF residency");


}




////////////////////////////////////////////////////////////////////////////////
