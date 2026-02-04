#ifndef LLAMA_INFER_NON_SAMPLER_H
#define LLAMA_INFER_NON_SAMPLER_H
#include <base/base.h>
#include "sampler.h"
namespace sampler {
class TopKSampler : public Sampler {
 public:
  TopKSampler(base::DeviceType device_type, size_t k) : Sampler(device_type), k_(k) {}

  size_t sample(const float* logits, size_t size, void* stream) override;

 private:
  size_t k_;
};
}  // namespace sampler
#endif  // LLAMA_INFER_NON_SAMPLER_H