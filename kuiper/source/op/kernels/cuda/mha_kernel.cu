#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {
constexpr static int thread_num = 256;
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}


__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // 预加载query到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  // head当前的注意力头索引，kv_mul用于gqa，head_size表示一个自注意力头的维度
  // kv_dim = head_size * head_num，多头自注意力情况下的key,value 维度
  // kv_dim = head_size * head_num / kv_num，GQA情况下的key,value 维度
  int head_offset = (head / kv_mul) * head_size;
  // 计算自注意力分数
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  // 使用自注意力分数对value矩阵加权
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}

// Paged attention kernel: K/V layout per layer is [max_blocks_physical, block_size, kv_dim].
// page_table[logical_block] -> physical_block index (or -1 if not allocated).
// Shared memory layout: [head_size floats for query] + [max_logical_blocks ints for page table].
__global__ void multi_head_attention_paged_kernel(int32_t pos, int32_t seq_len, float* query,
                                                  float* score_ptr, float* output,
                                                  float* key_cache_paged, float* val_cache_paged,
                                                  const int32_t* page_table, int32_t kv_dim,
                                                  int32_t kv_mul, int32_t head_num,
                                                  int32_t head_size, int32_t block_size,
                                                  int32_t num_logical_blocks) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  // Shared memory: query[head_size] + page_table_cache[num_logical_blocks]
  extern __shared__ float s_data[];
  float* s_query = s_data;
  int32_t* s_page_table = reinterpret_cast<int32_t*>(s_data + head_size);

  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // Load query head into shared memory
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = query_head[i];
  }

  // Load page table into shared memory (all threads collaborate)
  int num_blocks_used = (pos + block_size) / block_size;  // ceil((pos+1)/block_size)
  for (int i = threadIdx.x; i < num_blocks_used; i += blockDim.x) {
    s_page_table[i] = page_table[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  // GQA: map query head to kv head
  int head_offset = (head / kv_mul) * head_size;

  // Pass 1: compute QK^T scores with paged K lookup
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    int lb = t / block_size;
    int phys_block = s_page_table[lb];
    int off = t % block_size;
    // Layout: key_cache_paged[(phys_block * block_size + off) * kv_dim + head_offset]
    float* key_head = key_cache_paged + ((int64_t)phys_block * block_size + off) * kv_dim + head_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y +
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  // Pass 2: weighted sum over V with paged V lookup
  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float out_val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      int lb = t / block_size;
      int phys_block = s_page_table[lb];
      int off = t % block_size;
      float* val_head = val_cache_paged + ((int64_t)phys_block * block_size + off) * kv_dim + head_offset;
      out_val += score_head[t] * val_head[i];
    }
    output_head[i] = out_val;
  }
}

void mha_kernel_paged_cu(int32_t pos, int32_t head_num, int32_t seq_len, int32_t kv_dim,
                         int32_t kv_mul, int32_t head_size, int32_t block_size,
                         const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                         const tensor::Tensor& score_tensor, float* key_cache_paged,
                         float* val_cache_paged, const int32_t* page_table, CudaConfig* config) {
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  int32_t num_logical_blocks = (pos + block_size) / block_size;  // ceil((pos+1)/block_size)

  // Shared memory: query head (head_size floats) + page table cache (num_logical_blocks int32s)
  // Use max possible logical blocks for static shared mem size calculation
  size_t smem_bytes = (size_t)head_size * sizeof(float) +
                      (size_t)num_logical_blocks * sizeof(int32_t);

  cudaStream_t stream = config->stream;
  multi_head_attention_paged_kernel<<<head_num, thread_num, smem_bytes, stream>>>(
      pos, seq_len, query, score, output, key_cache_paged, val_cache_paged, page_table, kv_dim,
      kv_mul, head_num, head_size, block_size, num_logical_blocks);
}

}  // namespace kernel
