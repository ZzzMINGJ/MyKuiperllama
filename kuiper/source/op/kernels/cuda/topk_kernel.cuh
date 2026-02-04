#ifndef TOPK_KERNEL_CUH
#define TOPK_KERNEL_CUH
namespace kernel {
void topk_kernel_cu(const float* input_ptr, float* output_values_ptr, int* output_indices_ptr, size_t size, int k, void* stream);
}
#endif  // TOPK_KERNEL_CUH