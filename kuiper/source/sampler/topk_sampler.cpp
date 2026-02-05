#include "sampler/topk_sampler.h"

#include <algorithm>
#include <cmath>
#include <random> // 引入现代随机数库
#include <vector>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include "base/base.h"
#include "tensor/tensor.h"
#include "../op/kernels/cuda/topk_kernel.cuh" 

namespace sampler {

// 使用静态变量保持随机引擎状态，避免每次调用都重置
// 也可以作为 TopKSampler 的成员变量
static std::mt19937& get_rng() {
    static std::mt19937 rng(std::random_device{}()); 
    return rng;
}

size_t TopKSampler::sample(const float* logits, size_t size, void* stream) {
  if (!logits || size == 0) return 0;

  size_t k = k_;
  if (k == 0) k = 1;
  if (k > size) k = size;

  // ==================== CPU 路径 (略) ====================
  if (device_type_ == base::DeviceType::kDeviceCPU) {
      // ... (建议这里也改用 softmax 加权采样，暂时略过以聚焦 GPU) ...
      return 0; 
  }

  // ==================== GPU 路径 ====================
  
  // 1. 分配显存
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  float* d_topk_vals = static_cast<float*>(alloc_cu->allocate(sizeof(float) * k));
  int* d_topk_idx  = static_cast<int*>(alloc_cu->allocate(sizeof(int) * k));

  // 2. 调用 Kernel
  kernel::topk_kernel_cu(logits, d_topk_vals, d_topk_idx, size, static_cast<int>(k), stream);

  // 3. 拷回 Host (关键：Values 和 Indices 都要拷回)
  std::vector<float> h_vals(k);
  std::vector<int>   h_idx(k);

  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (s) {
    cudaMemcpyAsync(h_vals.data(), d_topk_vals, sizeof(float) * k, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_idx.data(), d_topk_idx, sizeof(int) * k, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
  } else {
    cudaMemcpy(h_vals.data(), d_topk_vals, sizeof(float) * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_topk_idx, sizeof(int) * k, cudaMemcpyDeviceToHost);
  }

  // 4. 【修复内存泄漏】必须调用 release
  alloc_cu->release(d_topk_vals);
  alloc_cu->release(d_topk_idx);

  // 5. 【实现加权采样】 (Softmax + Sampling)
  // 步骤 A: 找到最大值以保持数值稳定 (Log-Sum-Exp trick)
  float max_logit = h_vals[0]; // 假设 kernel 已经降序排列，第一个最大
  for (size_t i = 1; i < k; ++i) {
      if (h_vals[i] > max_logit) max_logit = h_vals[i];
  }

  // 步骤 B: 计算指数并求和 (应用 Temperature，这里假设 temp=1.0，如果有配置可加入)
  float temperature = 1.0f; // 你可以从类成员读取
  std::vector<float> probs(k);
  float sum_probs = 0.0f;
  
  for (size_t i = 0; i < k; ++i) {
      // 检查 index 是否有效
      if (h_idx[i] < 0) { 
          probs[i] = 0.0f; 
      } else {
          probs[i] = std::exp((h_vals[i] - max_logit) / temperature);
          sum_probs += probs[i];
      }
  }

  // 步骤 C: 随机选择
  // 使用 C++11 随机数引擎，确保随机性
  std::uniform_real_distribution<float> dist(0.0f, sum_probs);
  float random_val = dist(get_rng());

  // 步骤 D: 轮盘赌 (Cumulative Distribution Function)
  float acc = 0.0f;
  size_t selected_index = 0;
  for (size_t i = 0; i < k; ++i) {
      acc += probs[i];
      if (random_val <= acc) {
          selected_index = i;
          break;
      }
  }

  // 返回实际的 Token ID
  return static_cast<size_t>(h_idx[selected_index]);
}

}  // namespace sampler