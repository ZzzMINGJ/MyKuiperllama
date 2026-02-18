#ifndef TOPP_KERNEL_CUH
#define TOPP_KERNEL_CUH
namespace kernel {
void topp_kernel_cu(const float* input_ptr, float* output_values_ptr, int* output_indices_ptr, size_t size, float p, void* stream);
}
#endif  // TOPP_KERNEL_CUH