#include "../kernels_interface.h"
#include "topp_kernel.cuh"
#include "tensor/tensor.h"

#include <cuda_runtime_api.h>
#include <cfloat>
#include <cstdint>
#include <memory>

namespace kernel {
void topp_kernel_cu(const float* input_ptr, float* output_values_ptr, int* output_indices_ptr, size_t size, float p, void* stream)
{
    if (!input_ptr || !output_values_ptr || !output_indices_ptr) return;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

    /*
    计算 w_i = exp(logit-max)（或直接 prob）

    排序得到 sorted_w、sorted_idx

    计算前缀累计概率 cdf（inclusive scan）

    找到最小 k 使 cdf[k] >= p（这一步在 GPU 做掉）

    只把 0..k 的 sorted_w（或 probs）和 sorted_idx 拷回 CPU
    */
}
}  // namespace kernel
