#include "../kernels_interface.h"
#include "topp_kernel.cuh"
#include "topk_kernel.cuh"
#include "tensor/tensor.h"

#include <cuda_runtime_api.h>
#include <cfloat>
#include <cstdint>
#include <memory>

namespace kernel {

// top-p 过滤使用的候选数上限，与 topp_sampler.cpp 中的 K_MAX_TOPP 保持一致
static constexpr int K_MAX_TOPP = 64;

// 对 top-K 候选集应用 top-p（nucleus）过滤。
//
// 输入：
//   topk_vals[0..k-1]  — 已按 logit 值降序排列的 top-K 值
//   topk_idx[0..k-1]   — 对应 token 索引，-1 为无效占位符
//   k                  — 候选数量（≤ K_MAX_TOPP）
//   p                  — nucleus 累计概率阈值
//
// 输出：
//   out_vals[i]  — 在 nucleus 内：exp(logit_i - max_logit)；nucleus 外：0.0f
//   out_idx[i]   — 在 nucleus 内：token 索引；nucleus 外：-1（哨兵）
//
// 由于 k ≤ 128，使用单线程即可，避免不必要的同步开销。
__global__ void topp_filter_kernel(const float* topk_vals, const int* topk_idx,
                                   float* out_vals, int* out_idx,
                                   int k, float p) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  // 统计有效候选数（遇到 idx < 0 的哨兵即停止）
  int valid_k = 0;
  for (int i = 0; i < k; ++i) {
    if (topk_idx[i] < 0) break;
    ++valid_k;
  }

  // 无有效候选时全部置哨兵并返回
  if (valid_k == 0) {
    for (int i = 0; i < k; ++i) {
      out_vals[i] = 0.0f;
      out_idx[i]  = -1;
    }
    return;
  }

  // topk_vals 已降序排列，第一个元素即为最大值，用于数值稳定的 exp 计算
  float max_val = topk_vals[0];

  // 计算 exp(logit_i - max) 及其总和，用于将 exp 值归一化为概率
  float sum_exp = 0.0f;
  for (int i = 0; i < valid_k; ++i) {
    sum_exp += expf(topk_vals[i] - max_val);
  }

  // 前缀累计概率（CDF），找到最小 cutoff 使 CDF >= p
  // cutoff 默认为 valid_k - 1，确保浮点精度问题时也覆盖全部有效候选
  float cumulative = 0.0f;
  int cutoff = valid_k - 1;
  for (int i = 0; i < valid_k; ++i) {
    cumulative += expf(topk_vals[i] - max_val) / sum_exp;
    if (cumulative >= p) {
      cutoff = i;
      break;
    }
  }

  // 写出结果：
  //   nucleus 内（i <= cutoff）：保留 exp 值和 token 索引
  //   nucleus 外               ：置 0 / -1 哨兵
  for (int i = 0; i < k; ++i) {
    if (i <= cutoff) {
      out_vals[i] = expf(topk_vals[i] - max_val);
      out_idx[i]  = topk_idx[i];
    } else {
      out_vals[i] = 0.0f;
      out_idx[i]  = -1;
    }
  }
}

// top-p 采样 GPU 入口：
//
// 输出格式（与 topp_sampler.cpp GPU 路径约定一致）：
//   output_values_ptr[i]  — nucleus 内 token 的 exp(logit - max) 权重，nucleus 外为 0
//   output_indices_ptr[i] — nucleus 内 token 索引，nucleus 外为 -1（哨兵）
//   有效候选连续存放在头部，首个 -1 即为截止位置
void topp_kernel_cu(const float* input_ptr, float* output_values_ptr,
                    int* output_indices_ptr, size_t size, float p, void* stream) {
  if (!input_ptr || !output_values_ptr || !output_indices_ptr) return;
  if (size == 0) return;

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  // 实际候选数不超过词表大小
  int k = (static_cast<int>(size) < K_MAX_TOPP) ? static_cast<int>(size) : K_MAX_TOPP;

  float* tmp_topk_vals = static_cast<float*>(alloc_cu->allocate(sizeof(float) * K_MAX_TOPP));
  int*   tmp_topk_idx  = static_cast<int*>(alloc_cu->allocate(sizeof(int) * K_MAX_TOPP));

  // 将剩余槽位初始化为哨兵，防止 k < K_MAX_TOPP 时 topp_filter_kernel 读到垃圾数据
  // cudaMemset 以字节为单位；0xFF 填充 int 得到 0xFFFFFFFF = -1（补码）
  cudaMemsetAsync(tmp_topk_vals, 0, sizeof(float) * K_MAX_TOPP, cuda_stream);
  cudaMemsetAsync(tmp_topk_idx, 0xFF, sizeof(int) * K_MAX_TOPP, cuda_stream);

  // Stage 1：用现有 top-k kernel 取出降序排列的 top-k 候选
  topk_kernel_cu(input_ptr, tmp_topk_vals, tmp_topk_idx, size, k, stream);

  // Stage 2：对 top-k 候选集应用 top-p 过滤，输出 nucleus 结果
  topp_filter_kernel<<<1, 1, 0, cuda_stream>>>(
      tmp_topk_vals, tmp_topk_idx,
      output_values_ptr, output_indices_ptr,
      K_MAX_TOPP, p);

  alloc_cu->release(tmp_topk_vals);
  alloc_cu->release(tmp_topk_idx);
}

}  // namespace kernel
