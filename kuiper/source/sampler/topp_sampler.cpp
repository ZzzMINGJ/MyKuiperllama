#include "sampler/topp_sampler.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include "base/base.h"
#include "tensor/tensor.h"
#include "../op/kernels/cuda/topp_kernel.cuh"

namespace sampler {

static std::mt19937& get_rng() {
  static std::mt19937 rng(std::random_device{}());
  return rng;
}

size_t TopPSampler::sample(const float* logits, size_t size, void* stream) {
  if (!logits || size == 0) return 0;

  // ==================== CPU 路径 ====================
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    float max_exp = logits[0];
    for (size_t i = 1; i < size; ++i)
      if (logits[i] > max_exp) max_exp = logits[i];

    // 计算全局 sum_exp（用于归一化整体概率）
    float sum_exp = 0.0f;
    std::vector<std::pair<float, int>> logit_pairs(size);
    for (size_t i = 0; i < size; ++i) {
      logit_pairs[i] = {logits[i], static_cast<int>(i)};
      sum_exp += std::exp(logits[i] - max_exp);
    }

    // 按 logit 值降序排序
    std::sort(logit_pairs.begin(), logit_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // 逐步累计概率（CDF），找到使 CDF >= p 的最小截止索引
    float cumulative  = 0.0f;
    float sum_top_exp = 0.0f; // 截止范围内的 exp 之和，用于局部归一化
    size_t cutoff     = 0;
    for (size_t i = 0; i < size; ++i) {
      float e    = std::exp(logit_pairs[i].first - max_exp);
      sum_top_exp += e;
      cumulative  += e / sum_exp;
      if (cumulative >= p_) {
        cutoff = i;
        break;
      }
    }

    // 在前 cutoff+1 个元素中按（局部归一化）概率加权采样
    std::vector<float> probs(cutoff + 1);
    float sum_probs = 0.0f;
    for (size_t i = 0; i <= cutoff; ++i) {
      probs[i]    = std::exp(logit_pairs[i].first - max_exp) / sum_top_exp;
      sum_probs  += probs[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum_probs);
    float random_val = dist(get_rng());

    float acc          = 0.0f;
    size_t sampled_idx = 0;
    for (size_t i = 0; i <= cutoff; ++i) {
      acc += probs[i];
      if (random_val <= acc) {
        sampled_idx = i;
        break;
      }
    }
    return static_cast<size_t>(logit_pairs[sampled_idx].second);
  }

  // ==================== GPU 路径 ====================
  //
  // topp_kernel_cu 输出：
  //   output_values_ptr[0..K-1]  — 归一化概率，top-p 范围内有效
  //   output_indices_ptr[0..K-1] — 对应 token 索引，-1 为截止哨兵
  // K = 128（与 topp_kernel.cu 中 K_MAX_TOPP 一致）
  static constexpr int K_MAX_TOPP = 128;

  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  float* d_out_vals = static_cast<float*>(alloc_cu->allocate(sizeof(float) * K_MAX_TOPP));
  int*   d_out_idx  = static_cast<int*>(alloc_cu->allocate(sizeof(int)    * K_MAX_TOPP));

  kernel::topp_kernel_cu(logits, d_out_vals, d_out_idx, size, p_, stream);

  // 将结果拷回 Host
  std::vector<float> h_vals(K_MAX_TOPP);
  std::vector<int>   h_idx(K_MAX_TOPP);

  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (s) {
    cudaMemcpyAsync(h_vals.data(), d_out_vals, sizeof(float) * K_MAX_TOPP,
                    cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_idx.data(),  d_out_idx,  sizeof(int)   * K_MAX_TOPP,
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
  } else {
    cudaMemcpy(h_vals.data(), d_out_vals, sizeof(float) * K_MAX_TOPP, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(),  d_out_idx,  sizeof(int)   * K_MAX_TOPP, cudaMemcpyDeviceToHost);
  }

  alloc_cu->release(d_out_vals);
  alloc_cu->release(d_out_idx);

  // 统计有效候选数量及其概率之和（idx == -1 为哨兵，表示超出 top-p 截止）
  float sum_probs = 0.0f;
  int   valid_k   = 0;
  for (int i = 0; i < K_MAX_TOPP; ++i) {
    if (h_idx[i] < 0) break;
    sum_probs += h_vals[i];
    ++valid_k;
  }

  if (valid_k == 0) return 0;

  // 轮盘赌加权采样
  std::uniform_real_distribution<float> dist(0.0f, sum_probs);
  float random_val = dist(get_rng());

  float acc          = 0.0f;
  size_t sampled_idx = 0;
  for (int i = 0; i < valid_k; ++i) {
    acc += h_vals[i];
    if (random_val <= acc) {
      sampled_idx = i;
      break;
    }
  }

  return static_cast<size_t>(h_idx[sampled_idx]);
}

}  // namespace sampler
