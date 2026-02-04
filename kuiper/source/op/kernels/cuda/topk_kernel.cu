#include "../kernels_interface.h"
#include "topk_kernel.cuh"
#include "tensor/tensor.h"

#include <cuda_runtime_api.h>
#include <cfloat>
#include <cstdint>
#include <memory>

namespace kernel {

static constexpr int K_MAX = 128;

// 把 (v, idx) 插入到降序 topk 列表 vals/idxs（只用前 k 项）
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

// 合并 src(topk) 到 dst(topk)
__device__ __forceinline__ void merge_topk(const float* src_vals, const int* src_idxs,
                                           float* dst_vals, int* dst_idxs, int k) {
  for (int j = 0; j < k; ++j) {
    int idx = src_idxs[j];
    if (idx >= 0) {
      insert_topk(src_vals[j], idx, dst_vals, dst_idxs, k);
    }
  }
}

// stage kernel：对 input[n] 求 block top-k，输出到 out_vals/out_idx（每个 block 写 k 个）
__global__ void block_topk_fp32(const float* input, int n,
                                float* out_vals, int* out_idx, int k) {
  if (k <= 0) return;
  if (k > K_MAX) return;  // 也可以 assert/写错误码

  // 每线程局部 top-k（寄存器）
  float local_vals[K_MAX];
  int   local_idx[K_MAX];
  for (int j = 0; j < k; ++j) { local_vals[j] = -FLT_MAX; local_idx[j] = -1; }

  // stride 扫全局数组
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    insert_topk(input[i], i, local_vals, local_idx, k);
  }

  // 每个 warp 先合并成 warp top-k（用共享内存存每个 warp 的 top-k）
  __shared__ float warp_vals[32][K_MAX];
  __shared__ int   warp_idx[32][K_MAX];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;

  // 让每个 warp 的 lane0 收集本 warp 的 top-k（用 shfl 拉数据）
  unsigned mask = 0xffffffff;
  if (lane == 0) {
    float acc_vals[K_MAX];
    int   acc_idx[K_MAX];
    for (int j = 0; j < k; ++j) { acc_vals[j] = -FLT_MAX; acc_idx[j] = -1; }

    // 合并 warp 内 32 个线程的 top-k
    for (int src = 0; src < 32; ++src) {
      for (int j = 0; j < k; ++j) {
        float v = __shfl_sync(mask, local_vals[j], src);
        int   i = __shfl_sync(mask, local_idx[j], src);
        if (i >= 0) insert_topk(v, i, acc_vals, acc_idx, k);
      }
    }

    for (int j = 0; j < k; ++j) {
      warp_vals[warp][j] = acc_vals[j];
      warp_idx[warp][j]  = acc_idx[j];
    }
  }
  __syncthreads();

  // block 级合并：warp0 合并所有 warp 的 top-k
  if (warp == 0 && lane == 0) {
    float acc_vals[K_MAX];
    int   acc_idx[K_MAX];
    for (int j = 0; j < k; ++j) { acc_vals[j] = -FLT_MAX; acc_idx[j] = -1; }

    for (int w = 0; w < num_warps; ++w) {
      merge_topk(warp_vals[w], warp_idx[w], acc_vals, acc_idx, k);
    }

    // 写出本 block 的 top-k 到全局（连续 k 个）
    int base = blockIdx.x * k;
    for (int j = 0; j < k; ++j) {
      out_vals[base + j] = acc_vals[j];
      out_idx[base + j]  = acc_idx[j];
    }
  }
}

// stage2：对候选 pairs（vals/idx）求最终 top-k
__global__ void topk_from_pairs_fp32(const float* cand_vals, const int* cand_idx, int n_cand,
                                     float* out_vals, int* out_idx, int k) {
  if (k <= 0) return;
  if (k > K_MAX) return;

  float local_vals[K_MAX];
  int   local_idx[K_MAX];
  for (int j = 0; j < k; ++j) { local_vals[j] = -FLT_MAX; local_idx[j] = -1; }

  int tid = threadIdx.x;
  for (int i = tid; i < n_cand; i += blockDim.x) {
    int idx = cand_idx[i];
    if (idx >= 0) insert_topk(cand_vals[i], idx, local_vals, local_idx, k);
  }

  // 用一个 block 做归约（同样 warp->block）
  __shared__ float warp_vals[32][K_MAX];
  __shared__ int   warp_idx[32][K_MAX];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;

  unsigned mask = 0xffffffff;
  if (lane == 0) {
    float acc_vals[K_MAX];
    int   acc_idx[K_MAX];
    for (int j = 0; j < k; ++j) { acc_vals[j] = -FLT_MAX; acc_idx[j] = -1; }

    for (int src = 0; src < 32; ++src) {
      for (int j = 0; j < k; ++j) {
        float v = __shfl_sync(mask, local_vals[j], src);
        int   i = __shfl_sync(mask, local_idx[j], src);
        if (i >= 0) insert_topk(v, i, acc_vals, acc_idx, k);
      }
    }
    for (int j = 0; j < k; ++j) {
      warp_vals[warp][j] = acc_vals[j];
      warp_idx[warp][j]  = acc_idx[j];
    }
  }
  __syncthreads();

  if (warp == 0 && lane == 0) {
    float acc_vals[K_MAX];
    int   acc_idx[K_MAX];
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

// 你需要的 wrapper：输出在 device 上（out_vals/out_idx 长度为 k）
void topk_kernel_cu(const float* input_ptr,
                    float* output_values_ptr,
                    int* output_indices_ptr,
                    size_t size,
                    int k,
                    void* stream) {
  if (!input_ptr || !output_values_ptr || !output_indices_ptr) return;
  if (size == 0 || k <= 0) return;
  if (k > K_MAX) return;  // 或者你改成 clamp(k, K_MAX)

  cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

  // launch config
  const int threads = 256;
  int blocks = static_cast<int>((size + threads - 1) / threads);
  // 不要太多 block，否则 stage2 候选太多；这里给个上限
  if (blocks > 1024) blocks = 1024;

  // 临时缓冲：num_blocks * k
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  float* tmp_vals = static_cast<float*>(alloc_cu->allocate(sizeof(float) * blocks * k));
  int*   tmp_idx  = static_cast<int*>(alloc_cu->allocate(sizeof(int)   * blocks * k));

  // stage1: block top-k
  block_topk_fp32<<<blocks, threads, 0, s>>>(input_ptr, static_cast<int>(size),
                                             tmp_vals, tmp_idx, k);

  // stage2: merge candidates to final top-k (one block is enough)
  const int cand_n = blocks * k;
  const int threads2 = 256;
  topk_from_pairs_fp32<<<1, threads2, 0, s>>>(tmp_vals, tmp_idx, cand_n,
                                              output_values_ptr, output_indices_ptr, k);

  // 如果你 allocator 需要手动释放，记得在这里释放 tmp_vals/tmp_idx
  // alloc_cu->deallocate(tmp_vals);
  // alloc_cu->deallocate(tmp_idx);
}

}  // namespace kernel
