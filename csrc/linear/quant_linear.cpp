#include <torch/extension.h>



// W4A16O16, adapted from Marlin
torch::Tensor  w4a16o16_gemm(
  const torch::Tensor& A,
  const torch::Tensor& B,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int64_t size_m,
  int64_t size_n,
  int64_t size_k,
  bool output_transpose,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 // adapted from Marlin
  m.def("w4a16o16_gemm", &w4a16o16_gemm);
}
