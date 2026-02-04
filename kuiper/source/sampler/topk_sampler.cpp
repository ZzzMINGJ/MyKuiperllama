#include "sampler/topk_sampler.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include "base/base.h"
#include "tensor/tensor.h"
#include "../op/kernels/cuda/topk_kernel.cuh"   // 按你的实际路径 include
// 或者如果你有统一接口头：#include "../op/kernels_interface.h"

namespace sampler {

size_t TopKSampler::sample(const float* logits, size_t size, void* stream) {
  // 兜底
  if (!logits || size == 0) return 0;

  // clamp k
  size_t k = k_;
  if (k == 0) k = 1;
  if (k > size) k = size;

  // CPU 路径：你原来的逻辑
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    if (k == 1 || k == size) {
      return std::distance(logits, std::max_element(logits, logits + size));
    }

    std::vector<std::pair<float, size_t>> v;
    v.reserve(size);
    for (size_t i = 0; i < size; ++i) v.emplace_back(logits[i], i);

    std::partial_sort(v.begin(), v.begin() + k, v.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    return v[static_cast<size_t>(rand()) % k].second;
  }

  // GPU 路径：用 CUDA kernel 做 top-k
  // 1) 在 device 上分配 top-k 输出
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  float* d_topk_vals = static_cast<float*>(alloc_cu->allocate(sizeof(float) * k));
  int*   d_topk_idx  = static_cast<int*>(alloc_cu->allocate(sizeof(int) * k));

  // 2) 调用 kernel，输出仍在 device
  kernel::topk_kernel_cu(logits, d_topk_vals, d_topk_idx, size, static_cast<int>(k), stream);

  // 3) 拷回 host（先跑通版：只拷 indices）
  std::vector<int> h_idx(k);
  if (stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(h_idx.data(), d_topk_idx, sizeof(int) * k, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
  } else {
    cudaMemcpy(h_idx.data(), d_topk_idx, sizeof(int) * k, cudaMemcpyDeviceToHost);
  }

  // 4) 简单随机选一个 top-k index（注意检查有效性）
  int picked = h_idx[static_cast<size_t>(rand()) % k];
  if (picked < 0) {
    // 保险：如果 topk 输出里有 -1（不该发生），退化成 0 或 argmax
    return 0;
  }
  return static_cast<size_t>(picked);

  // 5) 如果你的 allocator 需要手动释放，这里要 deallocate
  // alloc_cu->deallocate(d_topk_vals);
  // alloc_cu->deallocate(d_topk_idx);
}

}  // namespace sampler
