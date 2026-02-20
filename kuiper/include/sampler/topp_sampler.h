#ifndef LLAMA_INFER_TOPP_SAMPLER_H
#define LLAMA_INFER_TOPP_SAMPLER_H
#include <base/base.h>
#include "sampler.h"
namespace sampler {
class TopPSampler : public Sampler {
 public:
  TopPSampler(base::DeviceType device_type, float p) : Sampler(device_type), p_(p) {}

  size_t sample(const float* logits, size_t size, void* stream) override;

 private:
  float p_;
};
}  // namespace sampler
#endif  // LLAMA_INFER_TOPP_SAMPLER_H