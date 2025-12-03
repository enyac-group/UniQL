// W4A16O16
#include "marlin_gemm_kernel.cuh"
torch::Tensor w4a16o16_gemm(
      const torch::Tensor& A,
      const torch::Tensor& B,
      const torch::Tensor& s,
            torch::Tensor& workspace,
      int64_t size_m,
      int64_t size_n,
      int64_t size_k,
      bool output_transpose,
      int thread_k,
      int thread_n,
      int sms,
      int max_par
);