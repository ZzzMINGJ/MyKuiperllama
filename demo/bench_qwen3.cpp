#ifdef QWEN3_SUPPORT
#include <base/base.h>
#include <glog/logging.h>
#include "model/qwen3.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// Arg helpers
// ---------------------------------------------------------------------------
static bool kv_str(const std::string& a, const std::string& key, std::string& out) {
  const std::string p = "--" + key + "=";
  if (a.rfind(p, 0) == 0) { out = a.substr(p.size()); return true; }
  return false;
}
static bool kv_int(const std::string& a, const std::string& key, int& out) {
  std::string v;
  if (kv_str(a, key, v)) { out = std::stoi(v); return true; }
  return false;
}

// ---------------------------------------------------------------------------
// Build a token sequence of exactly `len` tokens
// ---------------------------------------------------------------------------
static std::vector<int32_t> make_tokens(const model::Qwen3Model& model, int len) {
  const std::string base =
      "The quick brown fox jumps over the lazy dog. "
      "Artificial intelligence is reshaping how we live and work. ";
  std::string text;
  std::vector<int32_t> toks;
  while ((int)toks.size() < len) {
    text += base;
    toks = model.encode(text);
  }
  toks.resize(len);
  return toks;
}

// ---------------------------------------------------------------------------
// Run one full generate pass: prefill (not timed) + gen_len decode steps (timed).
// Returns per-decode-step latencies in ms.
// If out_logits != nullptr, captures full vocabulary logits at each decode step.
// ---------------------------------------------------------------------------
static std::vector<double> run_once(model::Qwen3Model& model,
                                    const std::vector<int32_t>& prompt,
                                    int gen_len,
                                    std::vector<std::vector<float>>* out_logits = nullptr) {
  const int plen = static_cast<int>(prompt.size());

  // Reset paged state (safe no-op in baseline mode)
  model.reset_paged_state();

  const auto& prompt_emb = model.embedding(prompt);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  int next = prompt[0];
  bool is_prompt = true;
  std::vector<double> step_ms;
  step_ms.reserve(gen_len);

  for (int pos = 0; pos < plen - 1 + gen_len; ++pos) {
    pos_tensor.index<int32_t>(0) = pos;

    if (pos < plen - 1) {
      // Prefill — not timed
      tensor::Tensor inp = model.fill_input(pos_tensor, prompt_emb, is_prompt);
      model.predict(inp, pos_tensor, is_prompt, next);
      next = prompt[pos + 1];
    } else {
      // Decode — timed
      is_prompt = false;
      std::vector<int32_t> tok{next};
      const auto& emb = model.embedding(tok);
      tensor::Tensor inp = model.fill_input(pos_tensor, emb, is_prompt);

      auto t0 = Clock::now();
      model.predict(inp, pos_tensor, is_prompt, next);
      cudaDeviceSynchronize();
      auto t1 = Clock::now();

      step_ms.push_back(Ms(t1 - t0).count());

      if (out_logits) {
        const auto& buf = model.get_buffer(model::ModelBufferType::kForwardOutput);
        std::vector<float> lv(buf.size());
        cudaMemcpy(lv.data(), buf.ptr<float>(), lv.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        out_logits->push_back(std::move(lv));
      }
    }
    if (model.is_sentence_ending(next)) break;
  }
  return step_ms;
}

// ---------------------------------------------------------------------------
// Correctness check: compare baseline vs paged logits step by step
// ---------------------------------------------------------------------------
static void check_correctness(model::Qwen3Model& model,
                               const std::vector<int32_t>& tokens,
                               int gen_len) {
  printf("\n[CHECK] prompt_len=%d  gen_len=%d\n", (int)tokens.size(), gen_len);
  fflush(stdout);

  // --- Paged run ---
  printf("[CHECK] Running paged...\n"); fflush(stdout);
  model.set_use_paged_attention(true);
  std::vector<std::vector<float>> paged_logits;
  run_once(model, tokens, gen_len, &paged_logits);

  // --- Baseline run ---
  printf("[CHECK] Running baseline...\n"); fflush(stdout);
  model.set_use_paged_attention(false);
  std::vector<std::vector<float>> base_logits;
  run_once(model, tokens, gen_len, &base_logits);

  // Restore paged mode
  model.set_use_paged_attention(true);

  // --- Compare ---
  int steps = (int)std::min(paged_logits.size(), base_logits.size());
  float global_max = 0.f;
  for (int s = 0; s < steps; ++s) {
    size_t n = std::min(paged_logits[s].size(), base_logits[s].size());
    for (size_t i = 0; i < n; ++i)
      global_max = std::max(global_max, std::abs(paged_logits[s][i] - base_logits[s][i]));
  }

  bool pass = global_max <= 1e-5f;
  printf("[CHECK] steps=%d  max|delta|=%.3e  %s\n\n",
         steps, global_max, pass ? "PASS ✓" : "FAIL ✗");
  fflush(stdout);
}

// ---------------------------------------------------------------------------
// Performance benchmark for one (attn_impl, prompt_len, gen_len) configuration
// ---------------------------------------------------------------------------
struct BenchResult {
  std::string attn_impl;
  int prompt_len, gen_len;
  double tok_s, ms_avg, ms_p50, ms_p95;
  size_t kv_bytes, baseline_kv_bytes;
};

static BenchResult bench_one(model::Qwen3Model& model,
                              const std::string& attn_impl,
                              const std::vector<int32_t>& tokens,
                              int gen_len,
                              int warmup,
                              int iters) {
  bool use_paged = (attn_impl == "paged");
  model.set_use_paged_attention(use_paged);

  printf("[BENCH] attn=%s  prompt=%d  gen=%d  warmup=%d  iters=%d\n",
         attn_impl.c_str(), (int)tokens.size(), gen_len, warmup, iters);
  fflush(stdout);

  for (int i = 0; i < warmup; ++i) run_once(model, tokens, gen_len);

  std::vector<double> all_ms;
  all_ms.reserve((size_t)iters * gen_len);
  for (int i = 0; i < iters; ++i) {
    auto ms = run_once(model, tokens, gen_len);
    for (double t : ms) all_ms.push_back(t);
    printf("  iter %d/%d\n", i + 1, iters); fflush(stdout);
  }

  double avg_ms = 0, p50_ms = 0, p95_ms = 0, tok_s = 0;
  if (!all_ms.empty()) {
    avg_ms = std::accumulate(all_ms.begin(), all_ms.end(), 0.0) / all_ms.size();
    tok_s  = 1000.0 / avg_ms;
    auto sorted = all_ms;
    std::sort(sorted.begin(), sorted.end());
    p50_ms = sorted[sorted.size() / 2];
    p95_ms = sorted[(size_t)(sorted.size() * 0.95)];
  }

  size_t kv_bytes = use_paged ? model.get_paged_kv_bytes_allocated()
                               : model.get_baseline_kv_bytes();

  BenchResult r;
  r.attn_impl       = attn_impl;
  r.prompt_len      = (int)tokens.size();
  r.gen_len         = gen_len;
  r.tok_s           = tok_s;
  r.ms_avg          = avg_ms;
  r.ms_p50          = p50_ms;
  r.ms_p95          = p95_ms;
  r.kv_bytes        = kv_bytes;
  r.baseline_kv_bytes = model.get_baseline_kv_bytes();
  return r;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  if (argc < 3) {
    printf(
        "Usage: bench_qwen3 <checkpoint> <tokenizer> [options]\n"
        "  --attn_impl=baseline|paged|both  (default: both)\n"
        "  --prompt_len=N                   (default: runs 64,128,256,512)\n"
        "  --gen_len=N                      (default: 128)\n"
        "  --warmup=N                       (default: 3)\n"
        "  --iters=N                        (default: 10)\n"
        "  --check                          (correctness check, requires paged init)\n"
        "  --csv=bench.csv\n");
    return 1;
  }

  const char* checkpoint_path = argv[1];
  const char* tokenizer_path  = argv[2];

  std::string attn_impl  = "both";
  int fixed_prompt_len   = -1;   // -1 = sweep
  int gen_len            = 128;
  int warmup             = 3;
  int iters              = 10;
  bool do_check          = false;
  std::string csv_path   = "bench.csv";

  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    std::string v;
    kv_str(a, "attn_impl",   attn_impl);
    kv_int(a, "prompt_len",  fixed_prompt_len);
    kv_int(a, "gen_len",     gen_len);
    kv_int(a, "warmup",      warmup);
    kv_int(a, "iters",       iters);
    kv_str(a, "csv",         csv_path);
    if (a == "--check") do_check = true;
  }

  // Determine prompt lengths to sweep
  std::vector<int> prompt_lens;
  if (fixed_prompt_len > 0) {
    prompt_lens = {fixed_prompt_len};
  } else {
    prompt_lens = {64, 128, 256, 512};
  }

  printf("=== bench_qwen3 ===\n");
  printf("attn_impl  : %s\n", attn_impl.c_str());
  printf("gen_len    : %d\n", gen_len);
  printf("warmup     : %d\n", warmup);
  printf("iters      : %d\n", iters);
  printf("csv        : %s\n\n", csv_path.c_str());
  fflush(stdout);

  // Initialize model (paged=true so paged buffers are always allocated,
  // allowing runtime switching for correctness check)
  bool need_paged = (attn_impl == "paged" || attn_impl == "both" || do_check);
  model::Qwen3Model model(base::TokenizerType::kEncodeBpe,
                          tokenizer_path, checkpoint_path, false);
  model.set_use_paged_attention(need_paged);
  auto status = model.init(base::DeviceType::kDeviceCUDA);
  if (!status) {
    LOG(FATAL) << "Model init failed: " << status.get_err_code();
  }

  const int seq_len = model.get_seq_len();
  printf("model seq_len = %d\n", seq_len);
  printf("baseline KV  = %.2f MB\n\n", model.get_baseline_kv_bytes() / 1e6);
  fflush(stdout);

  // Build CSV header
  bool csv_exists = false;
  { std::ifstream f(csv_path); csv_exists = f.good(); }
  std::ofstream csv(csv_path, std::ios::app);
  if (!csv_exists) {
    csv << "attn_impl,prompt_len,gen_len,tok_s,ms_avg,ms_p50,ms_p95,"
           "kv_bytes_MB,baseline_kv_MB,saved_pct\n";
  }

  // Determine which impls to run
  std::vector<std::string> impls;
  if (attn_impl == "both") {
    impls = {"baseline", "paged"};
  } else {
    impls = {attn_impl};
  }

  std::vector<BenchResult> results;

  for (int plen : prompt_lens) {
    if (plen + gen_len > seq_len) {
      printf("[SKIP] prompt_len=%d + gen_len=%d exceeds seq_len=%d\n",
             plen, gen_len, seq_len);
      continue;
    }

    // Correctness check for this prompt length
    if (do_check && need_paged) {
      auto tokens = make_tokens(model, plen);
      check_correctness(model, tokens, gen_len);
    }

    auto tokens = make_tokens(model, plen);

    for (const auto& impl : impls) {
      // Skip paged if not initialized
      if (impl == "paged" && !need_paged) continue;

      auto r = bench_one(model, impl, tokens, gen_len, warmup, iters);
      results.push_back(r);

      double saved_pct = 100.0 * (1.0 - (double)r.kv_bytes / r.baseline_kv_bytes);
      double kv_mb     = r.kv_bytes / 1e6;
      double base_mb   = r.baseline_kv_bytes / 1e6;

      printf("  → tok/s=%.2f  ms_avg=%.3f  ms_p95=%.3f  KV=%.2fMB",
             r.tok_s, r.ms_avg, r.ms_p95, kv_mb);
      if (impl == "paged")
        printf("  saved=%.1f%%", saved_pct);
      printf("\n\n");
      fflush(stdout);

      csv << impl << "," << plen << "," << gen_len << ","
          << std::fixed << std::setprecision(3)
          << r.tok_s << "," << r.ms_avg << "," << r.ms_p50 << "," << r.ms_p95 << ","
          << std::setprecision(2) << kv_mb << "," << base_mb << ","
          << std::setprecision(1) << saved_pct << "\n";
      csv.flush();
    }
  }

  // Summary table
  printf("=== Summary ===\n");
  printf("%-10s %-10s %-8s %-8s %-8s %-10s %-10s %-8s\n",
         "attn_impl", "prompt", "tok/s", "ms_avg", "ms_p95", "KV_MB", "base_MB", "saved%");
  for (const auto& r : results) {
    double saved = 100.0 * (1.0 - (double)r.kv_bytes / r.baseline_kv_bytes);
    printf("%-10s %-10d %-8.2f %-8.3f %-8.3f %-10.2f %-10.2f %-8.1f\n",
           r.attn_impl.c_str(), r.prompt_len,
           r.tok_s, r.ms_avg, r.ms_p95,
           r.kv_bytes / 1e6, r.baseline_kv_bytes / 1e6, saved);
  }
  printf("\nResults written to: %s\n", csv_path.c_str());
  return 0;
}
#endif  // QWEN3_SUPPORT
