#include "../kernels_interface.h"
#include "topk_kernel.cuh"
#include "tensor/tensor.h"

#include <cuda_runtime_api.h>
#include <cfloat>
#include <cstdint>
#include <memory>

namespace kernel {

static constexpr int K_MAX = 128;

// 辅助函数保持不变
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

__device__ __forceinline__ void merge_topk(const float* src_vals, const int* src_idxs,
                                           float* dst_vals, int* dst_idxs, int k) {
  for (int j = 0; j < k; ++j) {
    int idx = src_idxs[j];
    if (idx >= 0) {
      insert_topk(src_vals[j], idx, dst_vals, dst_idxs, k);
    }
  }
}

// 修复后的 stage kernel
__global__ void block_topk_fp32(const float* input, int n,
                                float* out_vals, int* out_idx, int k) {
  if (k <= 0 || k > K_MAX) return;

  // 1. 线程局部 Top-K
  float local_vals[K_MAX];
  int   local_idx[K_MAX];
  for (int j = 0; j < k; ++j) { local_vals[j] = -FLT_MAX; local_idx[j] = -1; }

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    insert_topk(input[i], i, local_vals, local_idx, k);
  }

  // 2. Warp 级合并
  // 共享内存用于存储每个 Warp 的最终结果
  __shared__ float warp_vals[32][K_MAX];
  __shared__ int   warp_idx[32][K_MAX];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;

  // --- 修复开始：正确使用 shuffle ---
  // 我们需要一个临时寄存器来累积结果，仅 lane 0 需要维护这个累积列表
  // 但所有线程都必须参与循环以提供数据
  float acc_vals[K_MAX];
  int   acc_idx[K_MAX];
  
  // 只有 lane 0 需要初始化 acc
  if (lane == 0) {
      for (int j = 0; j < k; ++j) { acc_vals[j] = -FLT_MAX; acc_idx[j] = -1; }
  }

  unsigned mask = 0xffffffff;
  
  // 遍历 warp 中的每一个线程作为数据源 (src)
  for (int src = 0; src < 32; ++src) {
      // 遍历 Top-K 的每一个元素
      for (int j = 0; j < k; ++j) {
          // 关键修正：所有线程都执行 __shfl_sync
          float v = __shfl_sync(mask, local_vals[j], src);
          int   i = __shfl_sync(mask, local_idx[j], src);
          
          // 只有 Lane 0 负责收集数据
          if (lane == 0 && i >= 0) {
              insert_topk(v, i, acc_vals, acc_idx, k);
          }
      }
  }
  
  // Lane 0 将 Warp 的结果写入共享内存
  if (lane == 0) {
    for (int j = 0; j < k; ++j) {
      warp_vals[warp][j] = acc_vals[j];
      warp_idx[warp][j]  = acc_idx[j];
    }
  }
  // --- 修复结束 ---

  __syncthreads();

  // 3. Block 级合并：由 Warp 0 的 Lane 0 完成
  if (warp == 0 && lane == 0) {
    // 复用之前的 acc_vals (或者重新初始化，因为上面已经用过了)
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

// 修复后的 stage2 kernel
__global__ void topk_from_pairs_fp32(const float* cand_vals, const int* cand_idx, int n_cand,
                                     float* out_vals, int* out_idx, int k) {
  if (k <= 0 || k > K_MAX) return;

  float local_vals[K_MAX];
  int   local_idx[K_MAX];
  for (int j = 0; j < k; ++j) { local_vals[j] = -FLT_MAX; local_idx[j] = -1; }

  int tid = threadIdx.x;
  // 处理输入数据
  for (int i = tid; i < n_cand; i += blockDim.x) {
    int idx = cand_idx[i];
    if (idx >= 0) insert_topk(cand_vals[i], idx, local_vals, local_idx, k);
  }

  // Warp 级归约 (逻辑同上，修复 shuffle)
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

  // Block 级归约
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
  if (!input_ptr || !output_values_ptr || !output_indices_ptr) return;
  if (size == 0 || k <= 0) return;
  
  // 限制 K，防止溢出
  if (k > K_MAX) k = K_MAX;

  cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

  const int threads = 256;
  int blocks = static_cast<int>((size + threads - 1) / threads);
  if (blocks > 1024) blocks = 1024;

  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  // 分配临时显存
  float* tmp_vals = static_cast<float*>(alloc_cu->allocate(sizeof(float) * blocks * k));
  int* tmp_idx  = static_cast<int*>(alloc_cu->allocate(sizeof(int)    * blocks * k));

  block_topk_fp32<<<blocks, threads, 0, s>>>(input_ptr, static_cast<int>(size),
                                             tmp_vals, tmp_idx, k);

  const int cand_n = blocks * k;
  const int threads2 = 256;
  topk_from_pairs_fp32<<<1, threads2, 0, s>>>(tmp_vals, tmp_idx, cand_n,
                                              output_values_ptr, output_indices_ptr, k);

  // --- 修复：必须释放显存 ---
  // 如果 alloc_cu 是智能指针管理的资源池，可能不需要手动释放，
  // 但通常 allocate/deallocate 是成对出现的。如果这是 raw pointer wrapper，必须释放。
  alloc_cu->release(tmp_vals);
  alloc_cu->release(tmp_idx);
}

}  // namespace kernel
