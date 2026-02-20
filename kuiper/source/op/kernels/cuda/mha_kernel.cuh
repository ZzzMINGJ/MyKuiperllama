#ifndef MHA_KERNEL_H
#define MHA_KERNEL_H
namespace kernel {
void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config);

// Paged attention kernel: K/V accessed via page_table[logical_block] -> physical block index
void mha_kernel_paged_cu(int32_t pos, int32_t head_num, int32_t seq_len, int32_t kv_dim,
                         int32_t kv_mul, int32_t head_size, int32_t block_size,
                         const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                         const tensor::Tensor& score_tensor, float* key_cache_paged,
                         float* val_cache_paged, const int32_t* page_table, CudaConfig* config);
}
#endif  // MHA_KERNEL_H
