#include "sampler/topk_sampler.h"
#include <algorithm>
#include <vector>
namespace sampler {
size_t TopKSampler::sample(const float* logits, size_t size, void* stream) {
  if (k_ == 0 || k_ >= size) {
    return std::distance(logits, std::max_element(logits, logits + size));
  }

  // Create a vector of pairs (logit, index)
  std::vector<std::pair<float, size_t>> logit_index_pairs;
  logit_index_pairs.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    logit_index_pairs.emplace_back(logits[i], i);
  }

  // Partially sort to get the top-k logits
  std::partial_sort(
      logit_index_pairs.begin(),
      logit_index_pairs.begin() + k_,
      logit_index_pairs.end(),
      [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
        return a.first > b.first;
      });

  // Randomly select one index from the top-k
  size_t random_index = rand() % k_;
  return logit_index_pairs[random_index].second;
}
}  // namespace sampler