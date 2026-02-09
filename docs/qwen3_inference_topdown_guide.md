# Qwen3 推理代码自顶向下指南（新手进阶版）

这版文档按“控制流 -> 模型层 -> 算子层 -> 内存层”的顺序讲解。  
目标是让你不只知道流程，还知道每一步的代码落点、数据形状、内存归属和释放路径。

---

## 0. 一句话先建立全局图

一次生成的完整链路：

1. `main` 创建模型并 `init`
2. `generate` 控制 prefill/decode 循环
3. `predict -> forward -> post_processing`
4. `forward` 内部每层执行 attention + FFN
5. 算子调用 `kernel::get_xxx_kernel` 下发到 CPU/CUDA
6. 所有中间值都在 `Model::buffers_` 管理的 Tensor/Buffer 中流动

关键入口文件：`demo/main_qwen3.cpp`

---

## 1. 第一步：入口和生成循环（控制流层）

代码位置：`demo/main_qwen3.cpp:60`

### 1.1 初始化入口

```cpp
model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
auto init_status = model.init(base::DeviceType::kDeviceCUDA);
```

分析：

1. `kEncodeBpe` + QWEN3 宏最终会走 `QwenEncodeLayer`。
2. `init(kDeviceCUDA)` 触发：读取模型文件、构建层、分配运行时缓冲区、准备采样器。

### 1.2 生成循环（prefill + decode）

代码位置：`demo/main_qwen3.cpp:18`

```cpp
while (pos < total_steps) {
  pos_tensor.index<int32_t>(0) = pos;
  if (pos < prompt_len - 1) {
    tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
    model.predict(input, pos_tensor, is_prompt, next);
  } else {
    is_prompt = false;
    tokens = std::vector<int32_t>{next};
    const auto& token_embedding = model.embedding(tokens);
    tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
    model.predict(input, pos_tensor, is_prompt, next);
  }
  if (model.is_sentence_ending(next)) {
    break;
  }
  if (is_prompt) {
    next = tokens.at(pos + 1);
  }
  pos += 1;
}
```

分析：

1. prefill 阶段不会使用采样结果推进，而是用“下一个真值 token”。
2. decode 阶段每次只喂 1 个 token，依赖 KV Cache 复用历史。
3. 这层只做流程调度，不做矩阵计算。

---

## 2. 第二步：`init` 里到底做了什么（模型装配层）

代码位置：`kuiper/source/model/qwen3.cpp:108`

```cpp
Status read_status = gen_model_from_file();
init_mem();
kernel::sin_cos_cache_calc_cu(...);
sampler_ = std::make_unique<sampler::TopKSampler>(device_type_,16);
```

分析：

1. `gen_model_from_file()`：分词器 + mmap 权重 + create_layers。
2. `init_mem()`：创建所有运行期 Tensor（包括 KV cache）。
3. `sin/cos cache` 预计算：RoPE 前向时直接查表。
4. `TopKSampler(16)`：decode 采用 top-k=16 采样。

---

## 3. 第三步：模型文件如何映射成可读权重（文件映射层）

代码位置：`kuiper/source/model/model.cpp:41`

### 3.1 mmap 路径

```cpp
int32_t fd = open(model_path_.data(), O_RDONLY);
...
fread(&config, sizeof(ModelConfig), 1, file);
...
raw_model_data_->data = mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);
raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
```

分析：

1. 头部先读取 `ModelConfig`，然后用 `mmap` 挂整文件。
2. FP32 权重起点是 `sizeof(ModelConfig)` 后。
3. 这里没有把权重复制到新内存，后续层权重大多是“外部视图”。

### 3.2 资源释放

代码位置：`kuiper/source/model/raw_model_data.cpp:5`

```cpp
if (data != nullptr && data != MAP_FAILED) {
  munmap(data, file_size);
}
if (fd != -1) {
  close(fd);
}
```

分析：模型销毁时解除映射并关闭 fd，避免泄漏。

---

## 4. 第四步：权重是怎么绑定到每一层的（参数层）

代码位置：`kuiper/source/model/qwen3.cpp:187`

### 4.1 顺序绑定

```cpp
// 1) rmsnorm (2*L+1)
// 2) embedding
// 3) wq, q_norm
// 4) wk, k_norm
// 5) wv
// 6) wo
// 7) w1, w2, w3
// 8) lm_head
```

### 4.2 关键绑定代码

```cpp
float* weight_ptr = (float*)raw_model_data_->weight(pos);
rms_norm_layer->set_weight(0, {hidden_dim}, weight_ptr, cpu_device_type);
...
wq->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
```

分析：

1. `set_weight(..., cpu_device_type)` 把权重包装为 CPU 外部 Buffer 视图。
2. 读取顺序必须和导出脚本一致，否则会“形状对了但语义错位”。

### 4.3 对应导出顺序

代码位置：`tools/export_qwen3/write_bin.py:36`

```python
weights = [
    *[layer.input_layernorm.weight ...],
    *[layer.post_attention_layernorm.weight ...],
    model.model.norm.weight,
    model.model.embed_tokens.weight,
    *[layer.self_attn.q_proj.weight ...],
    *[layer.self_attn.q_norm.weight ...],
    ...
    model.lm_head.weight
]
```

分析：`write_bin.py` 的权重序列要和 `create_param_layers` 完全同序。

---

## 5. 第五步：运行时 Buffer 是怎么建出来的（运行时内存层）

代码位置：`kuiper/source/model/qwen3.cpp:302`

### 5.1 缓冲区创建代码

```cpp
tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, alloc);
CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
```

`init_mem` 中关键 buffer：

1. 输入：`kInputTokens`、`kInputEmbeddings`、`kInputPos`
2. cache：`kKeyCache`、`kValueCache`
3. attention：`kQuery`、`kScoreStorage`、`kOutputMHA`、`kAttnOutput`
4. FFN：`kW1Output`、`kW2Output`、`kW3Output`、`kFFNRMSNorm`
5. 输出：`kForwardOutput`

### 5.2 一个重要实现细节：Buffer 复用（内存复用）

代码位置：`kuiper/source/model/qwen3.cpp:337`

```cpp
tensor::Tensor rms_output(...);
CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
```

分析：

1. 这 3 个键共享同一底层 `Buffer`，是显式内存复用策略。
2. 前提是生命周期不重叠，否则会互相覆盖。

---

## 6. 第六步：单 token 前向执行细节（计算层）

代码位置：`kuiper/source/model/qwen3.cpp:148`

```cpp
for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
  attention_rms(layer_idx, input);
  attention_qkv(layer_idx, pos_tensor);
  attention_mha(layer_idx, pos_tensor);
  feed_forward(layer_idx, input);
}
cls_logits(input);
```

### 6.1 `attention_qkv` 关键点

代码位置：`kuiper/source/model/qwen3.cpp:471`

```cpp
auto [key, val] = slice_kv_cache(layer_idx, pos);
STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
STATUS_CHECK(value_layer->forward(rmsnorm_output, val));
STATUS_CHECK(qwen_layers_->rope_layer_->forward(query, key, pos_tensor, ...));
```

分析：

1. `slice_kv_cache` 返回的是 cache 当前位置视图，不是新分配内存。
2. `wk/wv` 直接把当前 token 的 K/V 写入 cache 对应槽位。

### 6.2 `attention_mha` 关键点

代码位置：`kuiper/source/model/qwen3.cpp:526`

```cpp
std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
```

分析：

1. MHA 内部根据 `pos` 控制有效时间步。
2. `wo` 把注意力输出映射回 hidden 维度。

### 6.3 `feed_forward` 关键点

代码位置：`kuiper/source/model/qwen3.cpp:552`

```cpp
add(input, attn_output, input);
ffn_rmsnorm(input, ffn_norm_output);
w1(ffn_norm_output, w1_output);
w3(ffn_norm_output, w3_output);
swiglu(w1_output, w3_output, w1_output);
w2(w1_output, w2_output);
add(input, w2_output, input);
```

分析：典型 pre-norm + SwiGLU + 双残差结构。

### 6.4 `post_processing` 关键点

代码位置：`kuiper/source/model/qwen3.cpp:629`

```cpp
if (is_prompt) {
  next = -1;
} else {
  next = sampler_->sample(forward_logits, forward_output.size(), cuda_config_ ? cuda_config_->stream : nullptr);
}
```

分析：prefill 阶段不采样，decode 才采样。

---

## 7. 第七步：从算子到 kernel 的下发机制（算子抽象层）

### 7.1 `Layer::forward` 的模板式封装

代码位置：`kuiper/source/op/layer.cpp:237`

```cpp
base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}
```

分析：所有具体层都复用这一套入参装配逻辑。

### 7.2 参数层如何接 mmap 权重

代码位置：`kuiper/source/op/layer.cpp:184`

```cpp
std::shared_ptr<base::Buffer> buffer =
    std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
weight.assign(buffer);
```

分析：

1. `use_external=true` 表示这块内存不归 `Buffer` 释放。
2. 对应 mmap 生命周期由 `RawModelData` 管。

### 7.3 kernel 分发

代码位置：`kuiper/source/op/kernels/kernels_interfaces.cpp`

```cpp
MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) return matmul_kernel_cpu;
  if (device_type == base::DeviceType::kDeviceCUDA) return matmul_kernel_cu;
  LOG(FATAL) << "Unknown device type";
}
```

分析：模型层只管调用接口，不关心后端实现。

---

## 8. 深入底层：Tensor / Buffer / Allocator 数据与所有权

这一节是重点，专门回答“底层到底怎么分配和释放”。

## 8.1 Tensor 的两种构建方式

代码位置：`kuiper/source/tensor/tensor.cpp:35`

方式 A：`need_alloc=true + allocator`（自己分配）

```cpp
if (need_alloc && alloc) {
  allocate(alloc);
}
```

方式 B：`ptr != nullptr`（外部内存视图）

```cpp
std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
```

分析：

1. A 用于运行时中间张量、cache。
2. B 用于 mmap 权重视图、cache 切片视图、embedding 单步视图。

## 8.2 Tensor 分配调用链（运行时）

调用链：

1. `Qwen3Model::init_mem` 创建 `Tensor(..., need_alloc=true, alloc)`
2. `Tensor::allocate` -> `std::make_shared<Buffer>(byte_size, allocator, nullptr)`
3. `Buffer` 构造里调用 `allocator_->allocate(byte_size)`

关键代码：

```cpp
buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
```

代码位置：`kuiper/source/tensor/tensor.cpp:193`

## 8.3 Buffer 的释放规则（非常关键）

代码位置：`kuiper/source/base/buffer.cpp:18`

```cpp
Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->release(ptr_);
    }
  }
}
```

分析：

1. `use_external=true`：不释放底层地址（例如 mmap 权重、切片视图）。
2. `use_external=false`：交给 allocator 回收。

## 8.4 CPU 分配器

代码位置：`kuiper/source/base/alloc_cpu.cpp:13`

```cpp
int status = posix_memalign((void**)&data, alignment, byte_size);
```

分析：大于 1KB 时 32 字节对齐，小于 1KB 时 16 字节对齐。

## 8.5 CUDA 分配器（带简单池化）

代码位置：`kuiper/source/base/alloc_cu.cpp:7`

核心策略：

1. `>1MB` 走 `big_buffers_map_`，优先复用“足够且最小”的空闲块。
2. `<=1MB` 走 `cuda_buffers_map_`，线性找空闲块复用。
3. 找不到才 `cudaMalloc`。
4. `release` 不是立刻 `cudaFree`，通常只标记 `busy=false`。
5. 某设备累计空闲超过 1GB 时做一次批量 `cudaFree` 清理。

关键代码：

```cpp
if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
  cuda_buffers[i].busy = true;
  return cuda_buffers[i].data;
}
```

```cpp
if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
  // 批量释放空闲块
}
```

## 8.6 跨设备拷贝入口

代码位置：`kuiper/source/base/alloc.cpp:4`

```cpp
if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
  cudaMemcpyAsync(...)
}
```

分析：统一通过 `DeviceAllocator::memcpy`，支持 stream 和可选同步。

## 8.7 权重从 mmap 到 CUDA 的真实路径

调用链：

1. `create_param_layers` 用外部 Buffer 引用 mmap 权重（CPU）
2. `init_mem` 里调用 `qwen_layers_->to_cuda`
3. `LayerParam::to_cuda` 遍历每个权重执行 `Tensor::to_cuda`
4. `Tensor::to_cuda` 分配 CUDA Buffer + CPU2CUDA 拷贝

关键代码位置：

1. `kuiper/source/op/layer.cpp:174`
2. `kuiper/source/tensor/tensor.cpp:104`

---

## 9. 关键“视图”实现：不分配也能写数据

## 9.1 KV Cache 切片视图

代码位置：`kuiper/source/model/model.cpp:215`

```cpp
float* key_cache_ptr = const_cast<float*>(get_buffer(...).ptr<float>(cache_offset));
tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, key_cache_ptr);
key.set_device_type(device_type_);
```

分析：`key` 只是指向 `kKeyCache` 某一段的窗口，`wk` 结果会直接写回总 cache。

## 9.2 单步输入视图

代码位置：`kuiper/source/model/model.cpp:245`

```cpp
std::make_shared<base::Buffer>(config_->hidden_dim_ * sizeof(float), nullptr,
                               input_embeddings.ptr<float>(index * config_->hidden_dim_), true);
```

分析：decode 时只取一个 token 的 embedding 片段作为本轮输入，避免复制整段 prompt。

---

## 10. 采样层实现细节（输出策略层）

代码位置：`kuiper/source/sampler/topk_sampler.cpp:23`

GPU 路径流程：

1. 临时申请 `d_topk_vals/d_topk_idx`
2. 调 `topk_kernel_cu`
3. 拷回 host
4. softmax 后轮盘采样
5. `release` 临时显存

关键代码：

```cpp
float* d_topk_vals = static_cast<float*>(alloc_cu->allocate(sizeof(float) * k));
...
alloc_cu->release(d_topk_vals);
```

分析：这里依赖 `CUDADeviceAllocator` 的缓存式释放，不是每次 `cudaFree`。

---

## 11. 两个重要实现注意点（新手高频踩坑）

## 11.1 Embedding CUDA kernel 的 grid 固定 512

代码位置：`kuiper/source/op/kernels/cuda/emb_kernel.cu:35`

```cpp
constexpr int32_t max_seq_len = 512;
emb_kernel_cu_fp32<<<max_seq_len, thread_num, ...>>>(...);
```

影响：当 `input_num > 512` 时，超过 512 的 token 不会被处理。  
如果你要支持更长 prefill，需要把 grid 改成基于 `input_num` 动态计算。

## 11.2 Buffer 共享复用要确认生命周期

`kOutputRMSNorm` / `kW2Output` / `kFFNRMSNorm` 共用同一块 buffer（见第 5.2 节）。  
改算子顺序时必须确认没有并行读写冲突。

---

## 12. 维护清单（改代码时同步更新本文）

只要改了以下任意内容，就要更新本指南：

1. `demo/main_qwen3.cpp` 的 prefill/decode 控制逻辑
2. `qwen3.cpp` 的 `forward` 执行顺序
3. `init_mem` 的 buffer shape 或复用关系
4. `create_param_layers` 的权重读取顺序
5. `tools/export_qwen3/write_bin.py` 的导出顺序
6. `alloc_cu.cpp` 的内存池策略
7. `tensor.cpp` 的 `to_cuda/to_cpu/reshape/assign` 语义

提交前 3 条自检：

1. 文档流程和实际调用链逐行可对上。
2. 导出顺序和加载顺序逐项可对上。
3. 内存所有权（external/internal）在文档中描述正确。

---

## 13. 速查文件索引

1. 入口与循环：`demo/main_qwen3.cpp`
2. 模型主实现：`kuiper/source/model/qwen3.cpp`
3. 模型基类：`kuiper/source/model/model.cpp`
4. 权重映射释放：`kuiper/source/model/raw_model_data.cpp`
5. Tensor：`kuiper/source/tensor/tensor.cpp`
6. Buffer：`kuiper/source/base/buffer.cpp`
7. Allocator：`kuiper/source/base/alloc.cpp`
8. CPU Alloc：`kuiper/source/base/alloc_cpu.cpp`
9. CUDA Alloc：`kuiper/source/base/alloc_cu.cpp`
10. 层抽象：`kuiper/source/op/layer.cpp`
11. 采样：`kuiper/source/sampler/topk_sampler.cpp`
12. 导出：`tools/export_qwen3/write_bin.py`
