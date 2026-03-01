#include "../kernels_interface.h"
#include "topk_kernel.cuh"
#include "tensor/tensor.h"

#include <cuda_runtime_api.h>
#include <cfloat>
#include <cstdint>
#include <memory>

namespace kernel {

// 当前实现使用栈上固定长度数组保存中间 Top-K，因此这里限制最大 K。
// 如需支持更大 K，需要同步评估局部数组、共享内存和寄存器压力。
static constexpr int K_MAX = 32;

// 将一个候选值插入到降序排列的 Top-K 缓冲区中。
// - vals/idxs 长度至少为 k，且初始值通常为 (-FLT_MAX, -1)。
// - 若 v 不大于当前第 k 名，直接丢弃，避免无效移动。
// - 插入后仍保持 vals 从大到小有序。
__device__ __forceinline__ void insert_topk(float v, int idx,
                                            float* vals, int* idxs, int k) {
  if (k <= 0) return;
  if (v <= vals[k - 1]) return;

  int pos = k - 1;
  while (pos > 0 && v > vals[pos - 1]) {
    vals[pos] = vals[pos - 1];
    idxs[pos] = idxs[pos - 1];
    --pos;
  }
  vals[pos] = v;
  idxs[pos] = idx;
}

// 将 src Top-K 合并到 dst Top-K。
// src 中 idx < 0 视为无效项（占位符），会被忽略。
__device__ __forceinline__ void merge_topk(const float* src_vals, const int* src_idxs,
                                           float* dst_vals, int* dst_idxs, int k) {
  for (int j = 0; j < k; ++j) {
    int idx = src_idxs[j];
    if (idx >= 0) {
      insert_topk(src_vals[j], idx, dst_vals, dst_idxs, k);
    }
  }
}

// Stage 1:
// 对原始输入做 block 级 Top-K，输出每个 block 的候选 Top-K（values + indices）。
// out_vals/out_idx 的布局为 [block0_k个][block1_k个]...，总长度 blocks * k。
__global__ void block_topk_fp32(const float* input, int n,
                                float* out_vals, int* out_idx, int k) {
  if (k <= 0 || k > K_MAX) return;

  // 1) 每个线程先在寄存器/本地数组中维护自己的局部 Top-K。
  //    通过 grid-stride 遍历输入，覆盖任意规模 n。
  float local_vals[K_MAX];
  int   local_idx[K_MAX];
  for (int j = 0; j < k; ++j) { local_vals[j] = -FLT_MAX; local_idx[j] = -1; }

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    insert_topk(input[i], i, local_vals, local_idx, k);
  }

  // 2) 先做 warp 级合并：
  //    每个 warp 的 lane 0 负责收集该 warp 内所有线程的局部 Top-K，
  //    再将 warp 结果写入共享内存，供后续 block 级合并使用。
  __shared__ float warp_vals[32][K_MAX];
  __shared__ int   warp_idx[32][K_MAX];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;

  // 关键点：
  // - __shfl_sync 必须由 warp 内所有线程执行，避免行为未定义。
  // - 只有 lane 0 维护累计结果 acc_*，其他线程仅参与数据搬运。
  float acc_vals[K_MAX];
  int   acc_idx[K_MAX];

  // 仅 lane 0 需要初始化累计 Top-K。
  if (lane == 0) {
      for (int j = 0; j < k; ++j) { acc_vals[j] = -FLT_MAX; acc_idx[j] = -1; }
  }

  unsigned mask = 0xffffffff;

  // 依次把 warp 内每个源线程 src 的 local Top-K 广播出来并合并。
  for (int src = 0; src < 32; ++src) {
      for (int j = 0; j < k; ++j) {
          float v = __shfl_sync(mask, local_vals[j], src);
          int   i = __shfl_sync(mask, local_idx[j], src);

          // 只在 lane 0 执行真正的插入，避免重复计算。
          if (lane == 0 && i >= 0) {
              insert_topk(v, i, acc_vals, acc_idx, k);
          }
      }
  }

  // 每个 warp 的 lane 0 将 warp 结果写入共享内存。
  if (lane == 0) {
    for (int j = 0; j < k; ++j) {
      warp_vals[warp][j] = acc_vals[j];
      warp_idx[warp][j]  = acc_idx[j];
    }
  }

  __syncthreads();

  // 3) block 级合并：由 warp 0 的 lane 0 合并所有 warp 的 Top-K，
  //    最终写出当前 block 的 k 个候选结果。
  if (warp == 0 && lane == 0) {
    // 复用 acc_* 作为 block 级累计缓冲，先重置。
    for (int j = 0; j < k; ++j) { acc_vals[j] = -FLT_MAX; acc_idx[j] = -1; }

    for (int w = 0; w < num_warps; ++w) {
      merge_topk(warp_vals[w], warp_idx[w], acc_vals, acc_idx, k);
    }

    int base = blockIdx.x * k;
    for (int j = 0; j < k; ++j) {
      out_vals[base + j] = acc_vals[j];
      out_idx[base + j]  = acc_idx[j];
    }
  }
}

// Stage 2:
// 对 Stage 1 产出的候选对（cand_vals/cand_idx）做最终一次 Top-K 归约。
// 这里默认只启动一个 block，输出全局最终 Top-K。
__global__ void topk_from_pairs_fp32(const float* cand_vals, const int* cand_idx, int n_cand,
                                     float* out_vals, int* out_idx, int k) {
  if (k <= 0 || k > K_MAX) return;

  float local_vals[K_MAX];
  int   local_idx[K_MAX];
  for (int j = 0; j < k; ++j) { local_vals[j] = -FLT_MAX; local_idx[j] = -1; }

  int tid = threadIdx.x;
  // 每个线程处理一部分候选数据，先形成线程局部 Top-K。
  for (int i = tid; i < n_cand; i += blockDim.x) {
    int idx = cand_idx[i];
    if (idx >= 0) insert_topk(cand_vals[i], idx, local_vals, local_idx, k);
  }

  // 与 Stage 1 相同的 warp/block 两级归约流程。
  __shared__ float warp_vals[32][K_MAX];
  __shared__ int   warp_idx[32][K_MAX];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;

  float acc_vals[K_MAX];
  int   acc_idx[K_MAX];
  if (lane == 0) {
      for (int j = 0; j < k; ++j) { acc_vals[j] = -FLT_MAX; acc_idx[j] = -1; }
  }

  unsigned mask = 0xffffffff;
  for (int src = 0; src < 32; ++src) {
      for (int j = 0; j < k; ++j) {
          float v = __shfl_sync(mask, local_vals[j], src);
          int   i = __shfl_sync(mask, local_idx[j], src);
          if (lane == 0 && i >= 0) {
              insert_topk(v, i, acc_vals, acc_idx, k);
          }
      }
  }

  if (lane == 0) {
    for (int j = 0; j < k; ++j) {
      warp_vals[warp][j] = acc_vals[j];
      warp_idx[warp][j]  = acc_idx[j];
    }
  }
  __syncthreads();

  // 最终 block 级归约，写出全局 Top-K。
  if (warp == 0 && lane == 0) {
    for (int j = 0; j < k; ++j) { acc_vals[j] = -FLT_MAX; acc_idx[j] = -1; }

    for (int w = 0; w < num_warps; ++w) {
      merge_topk(warp_vals[w], warp_idx[w], acc_vals, acc_idx, k);
    }
    for (int j = 0; j < k; ++j) {
      out_vals[j] = acc_vals[j];
      out_idx[j]  = acc_idx[j];
    }
  }
}

void topk_kernel_cu(const float* input_ptr,
                    float* output_values_ptr,
                    int* output_indices_ptr,
                    size_t size,
                    int k,
                    void* stream) {
  // 基本参数校验：任一输出或输入指针无效则直接返回。
  if (!input_ptr || !output_values_ptr || !output_indices_ptr) return;
  // 空输入或非法 k 不需要启动 kernel。
  if (size == 0 || k <= 0) return;

  // 限制 k，确保不超过本实现的固定上限。
  if (k > K_MAX) k = K_MAX;

  // 允许外部传入 CUDA stream；为空时使用默认 stream。
  cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

  // Stage 1 启动配置：
  // - threads 固定为 256（8 个 warp）
  // - blocks 根据输入规模计算，并设置上限避免过多小 block 带来的调度开销
  const int threads = 256;
  int blocks = static_cast<int>((size + threads - 1) / threads);
  if (blocks > 1024) blocks = 1024;

  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  // 分配 stage 间临时缓冲：每个 block 产出 k 个候选值和对应索引。
  float* tmp_vals = static_cast<float*>(alloc_cu->allocate(sizeof(float) * blocks * k));
  int* tmp_idx  = static_cast<int*>(alloc_cu->allocate(sizeof(int)    * blocks * k));

  // Stage 1：输入 -> 每个 block 的候选 Top-K。
  block_topk_fp32<<<blocks, threads, 0, s>>>(input_ptr, static_cast<int>(size),
                                             tmp_vals, tmp_idx, k);

  // Stage 2：候选集合 -> 全局最终 Top-K。
  const int cand_n = blocks * k;
  const int threads2 = 256;
  topk_from_pairs_fp32<<<1, threads2, 0, s>>>(tmp_vals, tmp_idx, cand_n,
                                              output_values_ptr, output_indices_ptr, k);

  // 释放临时显存，避免每次调用累积泄漏。
  // 这里 allocate/release 成对出现，符合当前分配器接口约定。
  alloc_cu->release(tmp_vals);
  alloc_cu->release(tmp_idx);
}

}  // namespace kernel
