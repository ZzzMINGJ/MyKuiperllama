# Qwen3 推理代码自顶向下指南（新手进阶版）

本文档采用"控制流 → 模型层 → 算子层 → 内存层"的自顶向下顺序，深入剖析 Qwen3 推理实现。阅读完本指南后，你将清楚了解每一步的代码位置、数据形状、内存归属和释放路径，能够自信地修改和扩展代码。

---

## 0. 一句话理解整体架构

从用户输入到模型输出的完整链路如下：

1. **初始化阶段**: `main` 创建 `Qwen3Model` 对象并调用 `init()`，完成权重加载、内存分配、Kernel 预计算
2. **生成循环**: `generate()` 函数控制 prefill 和 decode 两个阶段的循环
3. **推理核心**: 每个 token 经过 `predict() → forward() → post_processing()` 流程
4. **层级计算**: `forward()` 内部遍历所有层，每层执行 RMSNorm → Attention → FFN 的标准 Transformer 结构
5. **Kernel 下发**: 算子层通过 `kernel::get_xxx_kernel()` 将计算任务分发到 CPU 或 CUDA
6. **内存管理**: 所有中间结果和 KV Cache 都存储在 `Model::buffers_` 中统一管理

**核心入口**: `demo/main_qwen3.cpp`
**模型主体**: `kuiper/source/model/qwen3.cpp`

**架构分层示意**:
```
┌────────────────────────────────────────────────────────┐
│           控制流层 (main_qwen3.cpp)                    │
│  生成循环控制、Prefill/Decode 调度、采样输出          │
└────────────────────┬───────────────────────────────────┘
                     │ 调用 predict()
┌────────────────────▼───────────────────────────────────┐
│          模型装配层 (qwen3.cpp)                         │
│  forward() 遍历所有层、attention/FFN 调度              │
└────────────────────┬───────────────────────────────────┘
                     │ 调用各算子 forward()
┌────────────────────▼───────────────────────────────────┐
│          算子抽象层 (Layer 子类)                        │
│  MatmulLayer、RmsNormLayer、RoPELayer、MHALayer 等     │
└────────────────────┬───────────────────────────────────┘
                     │ 获取 kernel 函数指针
┌────────────────────▼───────────────────────────────────┐
│       Kernel 分发层 (kernels_interfaces.cpp)           │
│  get_matmul_kernel() 等根据 DeviceType 返回函数指针    │
└─────┬──────────────────────────────────────┬───────────┘
      │ CPU 路径                             │ CUDA 路径
┌─────▼──────────────┐            ┌─────────▼───────────┐
│  CPU Kernel 实现   │            │  CUDA Kernel 实现   │
│  (Armadillo/BLAS)  │            │  (CUDA C++ / CUB)   │
└────────────────────┘            └─────────────────────┘
                     │                       │
┌────────────────────▼───────────────────────▼───────────┐
│              内存管理层 (Tensor/Buffer/Allocator)      │
│  统一管理 CPU/CUDA 内存、零拷贝视图、内存池复用        │
└────────────────────────────────────────────────────────┘
```

---

## 1. 控制流层：入口与生成循环

### 1.1 模型初始化入口

**代码位置**: `demo/main_qwen3.cpp:68-72`

```cpp
model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
auto init_status = model.init(base::DeviceType::kDeviceCUDA);
if (!init_status) {
  LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
}
```

**关键要点**:

1. **Tokenizer 类型**: `kEncodeBpe` 指定使用 BPE 编码器，配合 `QWEN3_SUPPORT` 宏，最终会在 `model.cpp:143` 处创建 `QwenEncodeLayer` 实现
2. **第四个参数 `false`**: 表示不使用量化模型（如果为 `true` 则加载 Int8 量化权重）。注意：CPU 设备不支持量化模型，会在 `qwen3.cpp:113` 返回错误
3. **初始化触发**: `init(kDeviceCUDA)` 会触发以下操作：
   - **CUDA 环境初始化**（`qwen3.cpp:117-126`）：创建 `CudaConfig` 并通过 `cudaStreamCreate()` 创建异步 stream
   - **读取模型文件**（`gen_model_from_file()`）：通过 mmap 方式将整个模型文件映射到虚拟内存
   - **构建所有参数层和非参数层**（`create_param_layers()` 和 `create_nonparam_layers()`）
   - **分配运行时缓冲区**（`init_mem()`）：包括 KV Cache、中间激活值等所有运行时 Tensor
   - **权重传输到 GPU**（如果是 CUDA 模式）：调用 `qwen_layers_->to_cuda(cuda_config_)` 异步复制所有权重
   - **预计算 RoPE 缓存**（`qwen3.cpp:133-142`）：sin/cos 查找表，CPU 用 `sin_cos_cache_calc_cpu()`，CUDA 用 `sin_cos_cache_calc_cu()`
   - **初始化采样器**（`qwen3.cpp:144`）：创建 TopKSampler(k=16)

**调用链**:
```
main() 
  → Qwen3Model::init(DeviceType::kDeviceCUDA)
    → gen_model_from_file()
      → create_encode_layer()  // 创建分词器
      → read_model_file()      // mmap 映射模型文件
      → create_layers()        // 创建所有层
        → create_param_layers()     // 参数层（权重绑定）
        → create_nonparam_layers()  // 非参数层（算子创建）
    → init_mem()               // 分配运行时 Buffer
    → qwen_layers_->to_cuda()  // 权重传输到 GPU
    → sin_cos_cache_calc_cu()  // 预计算 RoPE 缓存
    → TopKSampler(...)         // 创建采样器
```

### 1.2 生成循环：Prefill 与 Decode

**代码位置**: `demo/main_qwen3.cpp:18-46`

```cpp
while (pos < total_steps) {
  pos_tensor.index<int32_t>(0) = pos;
  if (pos < prompt_len - 1) {
    // Prefill 阶段：处理 prompt 中的 token
    tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
    model.predict(input, pos_tensor, is_prompt, next);
  } else {
    // Decode 阶段：自回归生成
    is_prompt = false;
    tokens = std::vector<int32_t>{next};
    const auto& token_embedding = model.embedding(tokens);
    tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
    model.predict(input, pos_tensor, is_prompt, next);
    if (next != 151645 && next != 151644) {  // 过滤特殊 token
      words.push_back(next);
    }
  }
  if (model.is_sentence_ending(next)) {
    break;
  }
  if (is_prompt) {
    next = tokens.at(pos + 1);  // Prefill 阶段使用真实 token
  }
  pos += 1;
}
```

**两阶段详解**:

1. **Prefill 阶段** (`pos < prompt_len - 1`):
   - **目的**：填充 KV Cache，为后续生成做准备
   - **特点**：不使用采样器输出，而是直接用 prompt 中的下一个真实 token（`main_qwen3.cpp:38`）
   - **性能**：理论上可以批量处理多个 token（当前实现是逐个处理，循环 `pos` 从 0 到 `prompt_len-1`）
   - **输入构建**：`model.fill_input(pos_tensor, prompt_embedding, is_prompt)` 从预先计算的 `prompt_embedding` 中取出当前位置的 embedding 作为输入

2. **Decode 阶段** (`pos >= prompt_len - 1`):
   - **目的**：自回归生成新 token
   - **特点**：每次只处理 1 个 token，依赖 KV Cache 复用历史信息
   - **采样**：使用 TopK 采样器从 logits 中选择下一个 token（`post_processing()` 内部调用 `sampler_->sample()`）
   - **输入构建**：每次为新生成的 token 单独计算 embedding（`main_qwen3.cpp:26`）
   - **特殊 token 过滤**：151645 和 151644 是 Qwen3 的特殊控制 token，不加入输出序列（`main_qwen3.cpp:29`）

3. **终止条件**:
   - 达到最大步数 `total_steps`
   - 遇到句子结束标志（`model.is_sentence_ending(next)` 返回 true）

4. **流程控制**: 这层只负责调度，不涉及矩阵计算和 Kernel 调用，所有计算逻辑封装在 `model.predict()` 内部

**Prefill vs Decode 的关键区别**:
| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入 token 数 | 多个（prompt 长度） | 1 个 |
| KV Cache 使用 | 写入（填充） | 读取并追加 |
| 计算依赖 | 可并行（理论上） | 串行（依赖上一步输出） |
| 采样器 | 不使用 | 使用 TopK 采样 |
| 输出使用 | 丢弃（只填充 cache） | 添加到最终结果 |

---

## 2. 模型装配层：`init()` 的内部实现

**代码位置**: `kuiper/source/model/qwen3.cpp:108-146`

### 2.1 初始化完整流程

```cpp
base::Status Qwen3Model::init(base::DeviceType device_type) {
  device_type_ = device_type;

  // 1. CUDA 环境初始化
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
  }

  // 2. 读取模型文件（mmap 映射）
  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }

  // 3. 分配运行时内存
  init_mem();

  // 4. 预计算 RoPE sin/cos 缓存
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(...);
  } else {
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache),
                                   get_buffer(ModelBufferType::kCosCache),
                                   cuda_config_->stream);
  }

  // 5. 创建采样器
  sampler_ = std::make_unique<sampler::TopKSampler>(device_type_, 16);
  return error::Success();
}
```

### 2.2 各步骤详解

#### 步骤 1: CUDA 环境初始化

**代码位置**: `qwen3.cpp:117-126`

```cpp
if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);  // 选择 GPU 0
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);  // 创建异步流
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return error::InternalError("The cuda handle create failed.");
    }
}
```

**关键点**:
- **Stream 的作用**：所有 CUDA kernel 和 memcpy 都使用这个 stream，实现异步执行和 CPU/GPU 并行
- **CudaConfig 结构**：存储在 `cuda_config_->stream` 中，会传递给所有算子层和 kernel 函数
- **错误处理**：创建失败会返回错误状态，避免后续 GPU 操作崩溃

#### 步骤 2: `gen_model_from_file()` 详解

**代码位置**: `qwen3.cpp:128` 调用，实现在 `model.cpp:169-193`

该函数包含三个关键步骤：

**2.1 创建分词器** (`model.cpp:143-166`):
```cpp
if (tokenizer_type_ == TokenizerType::kEncodeBpe) {
#ifdef QWEN3_SUPPORT
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(token_path_, true, false);
#endif
}
```
- Qwen3 使用自定义 BPE 分词器，不同于 SentencePiece
- `true, false` 参数控制是否添加 BOS/EOS token

**2.2 读取并映射模型文件** (`read_model_file()` → `model.cpp:41-111`):
```cpp
// 1. 打开文件描述符
int32_t fd = open(model_path_.data(), O_RDONLY);

// 2. 读取文件头部的 ModelConfig 结构体
FILE* file = fopen(model_path_.data(), "rb");
ModelConfig config;
fread(&config, sizeof(ModelConfig), 1, file);

// 3. 使用 mmap 映射整个文件到虚拟内存
struct stat sb;
fstat(fd, &sb);
raw_model_data_->file_size = sb.st_size;
raw_model_data_->data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// 4. 权重数据起始位置（跳过 ModelConfig 头部）
raw_model_data_->weight_data = static_cast<int8_t*>(data) + sizeof(ModelConfig);
```

**mmap 的三大优势**:
1. **零拷贝**：权重数据不会被复制到用户空间，操作系统按需加载页面（page fault 驱动）
2. **共享内存**：多个进程可以共享同一份物理内存（节省内存）
3. **懒加载**：只有实际访问的权重才会从磁盘加载到内存，启动速度快

**2.3 创建所有层** (`create_layers()`):
- 调用 `create_param_layers()` 创建所有带权重的层（详见第 4 节）
- 调用 `create_nonparam_layers()` 创建计算层（详见 `qwen3.cpp:170-183`）

#### 步骤 3: `init_mem()` 内存分配

在 `qwen3.cpp:132` 调用，详细实现见第 5 节。简要流程：
1. 获取 CPU 或 CUDA 分配器单例
2. 创建所有运行时 Tensor（KV Cache、中间激活值）
3. 如果是 CUDA 模式，调用 `to_cuda()` 传输权重

#### 步骤 4: RoPE 预计算详解

**代码位置**: `qwen3.cpp:133-142`

RoPE（Rotary Position Embedding）的 sin/cos 值只与位置和维度相关，可以预先计算：

**CPU 路径** (`rope_kernel.cpp:44-70`):
```cpp
for (int pos = 0; pos < seq_len; pos++) {
    for (int head_dim = 0; head_dim < head_size/2; head_dim++) {
        float freq = 1.0f / powf(1000000.0f, (float)head_dim / (float)head_size);
        float val = pos * freq;
        sin_cache_ptr[pos * head_size + head_dim*2] = sinf(val);
        cos_cache_ptr[pos * head_size + head_dim*2] = cosf(val);
        // Qwen 使用 stride=2 的交错存储
        sin_cache_ptr[pos * head_size + head_dim*2 + 1] = sinf(val);
        cos_cache_ptr[pos * head_size + head_dim*2 + 1] = cosf(val);
    }
}
```

**CUDA 路径** (`rope_kernel.cu`):
- 每个 thread 计算一个位置的多个维度
- 结果直接写入 GPU 显存中的 sin/cos cache
- 使用 stream 异步执行

**Shape 和内存占用**:
- `[seq_len, head_size]` = `[32768, 128]`
- 单个 cache: `32768 * 128 * 4 bytes = 16 MB`
- Sin + Cos 总共: `32 MB`

**RoPE 频率基数差异**:
- **Llama**: 基数 = 10000
- **Qwen**: 基数 = 1000000（更大的基数 → 更慢的旋转 → 更好的长序列外推能力）

#### 步骤 5: TopK 采样器创建

**代码位置**: `qwen3.cpp:144`

```cpp
sampler_ = std::make_unique<sampler::TopKSampler>(device_type_, 16);
```

**采样器职责**:
- 从 `vocab_size` 维的 logits 中选出概率最高的 k=16 个 token
- 对这 k 个 token 的 logits 做 softmax 归一化
- 基于概率分布随机采样一个 token（温度采样）
- 支持 CPU 和 CUDA 两种实现路径（详见第 10 节）

---

## 3. 文件映射层：权重如何从磁盘到内存

**代码位置**: `kuiper/source/model/model.cpp:41-100`

### 3.1 mmap 映射策略

```cpp
// 1. 打开文件描述符
int32_t fd = open(model_path_.data(), O_RDONLY);
if (fd == -1) {
  return error::PathNotValid("Failed to open the weight file");
}

// 2. 读取文件头部的 ModelConfig
FILE* file = fopen(model_path_.data(), "rb");
auto config = ModelConfig{};
if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
  return error::ModelParseError("Failed to retrieve the configuration");
}

// 3. 获取文件大小
struct stat sb;
if (fstat(fd, &sb) == -1) {
  close(fd);
  return error::ModelParseError("Failed to retrieve the file size");
}
raw_model_data_->file_size = sb.st_size;

// 4. mmap 映射整个文件
raw_model_data_->data = mmap(nullptr, raw_model_data_->file_size,
                              PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

if (raw_model_data_->data == MAP_FAILED) {
  return error::ModelParseError("Failed to map the weight file into memory");
}

// 5. 权重数据起始位置（跳过 ModelConfig）
raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
```

### 3.2 mmap 的工作原理与优势

**虚拟内存映射机制**:
```
┌─────────────────────────────────────────────────────┐
│  模型文件 (磁盘)                                    │
│  ┌─────────┬─────────┬─────────┬─────────┐         │
│  │ Config  │ Weight1 │ Weight2 │  ...    │         │
│  └─────────┴─────────┴─────────┴─────────┘         │
└────────────────────┬────────────────────────────────┘
                     │ mmap()
                     ↓
┌─────────────────────────────────────────────────────┐
│  虚拟内存空间 (raw_model_data_->data)               │
│  进程可以像访问内存一样访问文件                     │
│  ┌─────────┬─────────┬─────────┬─────────┐         │
│  │ Config  │ Weight1 │ Weight2 │  ...    │         │
│  └─────────┴─────────┴─────────┴─────────┘         │
└────────────────────┬────────────────────────────────┘
                     │ Page Fault 触发加载
                     ↓
┌─────────────────────────────────────────────────────┐
│  物理内存 (RAM)                                      │
│  操作系统按需加载真正访问的页面（通常 4KB）         │
│  ┌─────────┐           ┌─────────┐                 │
│  │ Weight1 │  ...      │ Weight10│                 │
│  └─────────┘           └─────────┘                 │
└─────────────────────────────────────────────────────┘
```

**三大优势详解**:

1. **零拷贝**：
   - 传统方式：文件内容 → 内核缓冲区 → 用户空间缓冲区（两次复制）
   - mmap 方式：文件内容 → 页面缓存，进程直接访问页面缓存（零次复制）
   - 对于 7B 模型（~26GB），节省约 26GB 的内存复制开销

2. **共享内存**：
   - 多个进程 mmap 同一个文件时，物理内存只需加载一份
   - 例如：运行 4 个模型实例，传统方式需要 4×26GB = 104GB，mmap 只需 26GB

3. **懒加载**（Demand Paging）：
   - 调用 mmap 后，文件并未立即加载到内存
   - 第一次访问某个地址时触发 page fault，操作系统才加载该页面
   - 模型初始化时间从数十秒降低到毫秒级

**Page Fault 工作流程**:
```
CPU 访问 weight_ptr[1000000]
    ↓
MMU 检查页表，发现该页未加载
    ↓
触发 Page Fault 异常
    ↓
操作系统中断处理
    ↓
从磁盘读取对应 4KB 页面到物理内存
    ↓
更新页表映射
    ↓
恢复进程执行，CPU 重新访问该地址
    ↓
访问成功（后续访问该页面无需再次加载）
```

### 3.3 模型文件布局

```
+---------------------+
| ModelConfig (头部)  |  <- sizeof(ModelConfig) 字节
+---------------------+
| RMSNorm 权重        |
+---------------------+
| Embedding 权重      |
+---------------------+
| Attention 权重      |
| (wq, wk, wv, wo)   |
+---------------------+
| FFN 权重           |
| (w1, w2, w3)       |
+---------------------+
| lm_head 权重        |
+---------------------+
```

**重要**: 权重顺序必须与 `create_param_layers()` 的读取顺序完全一致，否则会导致语义错位。

### 3.4 资源释放机制

**代码位置**: `kuiper/source/model/raw_model_data.cpp:5-10`

```cpp
RawModelData::~RawModelData() {
  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, file_size);  // 解除映射
  }
  if (fd != -1) {
    close(fd);  // 关闭文件描述符
  }
}
```

**释放时机与顺序**:
1. **模型对象销毁时**：`Qwen3Model` 析构 → `raw_model_data_` 智能指针引用计数归零 → 调用 `~RawModelData()`
2. **解除映射**：`munmap()` 通知操作系统释放虚拟内存映射，但物理页面可能暂时保留在 page cache
3. **关闭文件**：`close(fd)` 释放文件描述符，如果没有其他进程打开该文件，kernel 可能释放 page cache

**内存所有权链**:
```
┌──────────────────────────────────────────────────────┐
│  Qwen3Model                                          │
│    raw_model_data_: shared_ptr<RawModelData>         │
│    qwen_layers_->wq_layers_[0]->weights_[0]          │
└───────────┬───────────────────┬──────────────────────┘
            │ 拥有              │ 引用（不拥有）
            ↓                   ↓
┌───────────────────────┐   ┌──────────────────────────┐
│  RawModelData         │   │  Layer::weights_         │
│    data: void*        │   │    Buffer                │
│    fd: int            │   │      ptr_ = data + offset│
│  ~RawModelData():     │   │      use_external_ = true│
│    munmap(data, ...)  │   │  ~Buffer():              │
│    close(fd)          │   │    if (!use_external_)   │
└───────────────────────┘   │      allocator->release()│
                            │    // 不释放外部指针     │
                            └──────────────────────────┘
```

**关键点**:
1. **所有权唯一性**：只有 `RawModelData` 真正拥有 mmap 内存，负责释放
2. **外部视图**：所有通过 `set_weight()` 创建的 Buffer 都标记为 `use_external=true`
3. **析构安全性**：Layer 的 Buffer 先析构（不释放内存），最后 `RawModelData` 析构才解除映射
4. **内存泄漏防护**：即使程序异常退出，操作系统也会自动 munmap 和 close

---

## 4. 参数层：权重如何绑定到每一层

**代码位置**: `kuiper/source/model/qwen3.cpp:187-280`

### 4.1 权重绑定顺序

Qwen3 模型的权重按以下严格顺序从 mmap 数据中读取：

```cpp
void Qwen3Model::create_param_layers() {
  size_t pos = 0;  // 当前读取位置（相对于 weight_data 起点）
  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int hidden_dim = config_->hidden_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // ========== 1. RMSNorm 权重 (2 * layer_num + 1) ==========
  // 每层有 2 个 RMSNorm: attention_rmsnorm + ffn_rmsnorm
  // 最后还有 1 个 final_rmsnorm
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    float* weight_ptr = (float*)raw_model_data_->weight(pos);
    rms_norm_layer->set_weight(0, {hidden_dim}, weight_ptr, cpu_device_type);
    pos += hidden_dim;
  }

  // ========== 2. Embedding 权重 ==========
  float* emb_weight_ptr = (float*)raw_model_data_->weight(pos);
  embedding_layer_->set_weight(0, {vocab_size_, hidden_dim}, emb_weight_ptr, cpu_device_type);
  pos += vocab_size_ * hidden_dim;

  // ========== 3-6. Attention 权重（每层 4 个矩阵）==========
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    // 3. wq (Query projection)
    wq->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim * hidden_dim;

    // 4. q_norm (Query normalization for Qwen3)
    q_norm->set_weight(0, {head_size}, raw_model_data_->weight(pos), cpu_device_type);
    pos += head_size;

    // 5. wk (Key projection)
    wk->set_weight(0, {kv_dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += kv_dim * hidden_dim;

    // 6. k_norm (Key normalization for Qwen3)
    k_norm->set_weight(0, {head_size}, raw_model_data_->weight(pos), cpu_device_type);
    pos += head_size;

    // 7. wv (Value projection)
    wv->set_weight(0, {kv_dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += kv_dim * hidden_dim;

    // 8. wo (Output projection)
    wo->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += hidden_dim * dim;
  }

  // ========== 7. FFN 权重（每层 3 个矩阵）==========
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    // w1 (gate projection)
    w1->set_weight(0, {intermediate_dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += intermediate_dim * hidden_dim;

    // w2 (down projection)
    w2->set_weight(0, {hidden_dim, intermediate_dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += hidden_dim * intermediate_dim;

    // w3 (up projection for SwiGLU)
    w3->set_weight(0, {intermediate_dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += intermediate_dim * hidden_dim;
  }

  // ========== 8. lm_head 权重 ==========
  cls_layer_->set_weight(0, {vocab_size_, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
  pos += vocab_size_ * hidden_dim;
}
```

### 4.2 `set_weight()` 的作用

**代码位置**: `kuiper/source/op/layer.cpp:184-190`

```cpp
std::shared_ptr<base::Buffer> buffer =
    std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
                                                                                   // ^^^^ use_external=true
weight.assign(buffer);
```

**关键点**:
1. `use_external=true` 表示这块内存**不归 Buffer 管理**，Buffer 析构时不会释放
2. 权重只是 mmap 映射区域的一个**视图**，没有发生数据复制
3. 真正的生命周期由 `RawModelData` 的 `munmap` 控制

### 4.3 导出顺序必须匹配

**代码位置**: `tools/export_qwen3/write_bin.py:36-60`

Python 导出脚本必须按照与 C++ 完全相同的顺序写入权重：

```python
weights = [
    # 1. RMSNorm (2L + 1)
    *[layer.input_layernorm.weight for layer in model.model.layers],
    *[layer.post_attention_layernorm.weight for layer in model.model.layers],
    model.model.norm.weight,

    # 2. Embedding
    model.model.embed_tokens.weight,

    # 3-6. Attention (per layer)
    *[layer.self_attn.q_proj.weight for layer in model.model.layers],
    *[layer.self_attn.q_norm.weight for layer in model.model.layers],
    *[layer.self_attn.k_proj.weight for layer in model.model.layers],
    *[layer.self_attn.k_norm.weight for layer in model.model.layers],
    *[layer.self_attn.v_proj.weight for layer in model.model.layers],
    *[layer.self_attn.o_proj.weight for layer in model.model.layers],

    # 7. FFN (per layer)
    *[layer.mlp.gate_proj.weight for layer in model.model.layers],
    *[layer.mlp.down_proj.weight for layer in model.model.layers],
    *[layer.mlp.up_proj.weight for layer in model.model.layers],

    # 8. lm_head
    model.lm_head.weight
]
```

**警告**: 如果顺序不一致，模型虽然能加载成功，但会产生完全错误的输出！

---

## 5. 运行时内存分配：`init_mem()` 的实现细节

**代码位置**：`kuiper/source/model/qwen3.cpp:302-381`

### 5.1 分配器获取与权重迁移

```cpp
void Qwen3Model::init_mem() {
    // 1. 根据设备类型获取对应的分配器单例
    std::shared_ptr<base::DeviceAllocator> alloc;
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        alloc = base::CPUDeviceAllocatorFactory::get_instance();
    } else {
        alloc = base::CUDADeviceAllocatorFactory::get_instance();
    }

    // 2. CUDA 模式下，将所有权重从 CPU 迁移到 GPU
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
        qwen_layers_->to_cuda(cuda_config_);  // 异步传输权重
    }

    // 3. 保留 CPU 和 CUDA 分配器引用（用于特定 Buffer）
    std::shared_ptr<base::DeviceAllocator> alloc_cpu =
        base::CPUDeviceAllocatorFactory::get_instance();
    std::shared_ptr<base::DeviceAllocator> alloc_cu =
        base::CUDADeviceAllocatorFactory::get_instance();
```

**权重迁移详解** (`qwen3.cpp:15-101`):
- `to_cuda()` 遍历所有参数层（wq, wk, wv, wo, w1, w2, w3, rmsnorm）
- 每层调用 `layer->to_cuda()`，内部执行：
  ```cpp
  for (auto& weight : weights_) {
      weight.to_cuda(config->stream);  // 使用 stream 异步拷贝
  }
  ```
- 底层实现 (`tensor.cpp:104-119`):
  ```cpp
  void Tensor::to_cuda(cudaStream_t stream) {
      auto cu_buffer = std::make_shared<Buffer>(byte_size, cu_alloc);
      cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, 
                       MemcpyKind::kMemcpyCPU2CUDA, stream);
      this->buffer_ = cu_buffer;  // 替换为 CUDA buffer
  }
  ```
- **异步传输**：所有拷贝使用同一个 stream，cudaMemcpyAsync 不阻塞 CPU
- **内存影响**：原 CPU mmap 视图保留（由 RawModelData 管理），GPU 新分配一份副本

### 5.2 Buffer 分配详细清单

**持久化 Buffer（整个推理过程保持）**:

| Buffer 类型 | Shape | 用途 | 设备 |
|------------|-------|------|------|
| `kSinCache` | `[head_size * seq_len]` | RoPE 预计算 sin 值 | alloc (CPU/CUDA) |
| `kCosCache` | `[head_size * seq_len]` | RoPE 预计算 cos 值 | alloc (CPU/CUDA) |
| `kKeyCache` | `[layer_num, seq_len, kv_dim]` | 所有层的 Key 缓存 | alloc (CPU/CUDA) |
| `kValueCache` | `[layer_num, seq_len, kv_dim]` | 所有层的 Value 缓存 | alloc (CPU/CUDA) |
| `kInputTokens` | `[1]` int32 | 当前 token ID | alloc_cpu (始终 CPU) |
| `kInputPos` | `[1]` int32 | 当前位置索引 | alloc_cpu (始终 CPU) |
| `kForwardOutput` | `[vocab_size]` | 最终 logits | alloc (CPU/CUDA) |
| `kForwardOutputCPU` | `[vocab_size]` | logits 的 CPU 副本（仅 CUDA 模式） | alloc_cpu |

**代码示例** (`qwen3.cpp:320-380`):
```cpp
// 输入相关
tensor::Tensor input_tokens(DataType::kDataTypeInt32, 1, true, alloc_cpu);
tensor::Tensor input_embeddings(DataType::kDataTypeFp32, 1, config_->hidden_dim_, 
                                true, alloc);

// RoPE 缓存（预计算后不再修改）
tensor::Tensor sin_cache(DataType::kDataTypeFp32, 
                         config_->head_size_ * config_->seq_len_, true, alloc);
tensor::Tensor cos_cache(DataType::kDataTypeFp32, 
                         config_->head_size_ * config_->seq_len_, true, alloc);

// KV Cache（动态填充）
tensor::Tensor key_cache(DataType::kDataTypeFp32, 
                         config_->layer_num_, config_->seq_len_, config_->kv_dim_, 
                         true, alloc);
tensor::Tensor value_cache(DataType::kDataTypeFp32, 
                           config_->layer_num_, config_->seq_len_, config_->kv_dim_, 
                           true, alloc);
```

**中间激活 Buffer（层间复用）**:

| Buffer 类型 | Shape | 用途 | 复用关系 |
|------------|-------|------|---------|
| `kInputEmbeddings` | `[hidden_dim]` | Token embedding 输出 | 独立 |
| `kOutputRMSNorm` | `[hidden_dim]` | Attention RMSNorm 输出 | **复用 1** |
| `kW2Output` | `[hidden_dim]` | FFN down projection 输出 | **复用 1**（同一 Tensor） |
| `kFFNRMSNorm` | `[hidden_dim]` | FFN RMSNorm 输出 | **复用 1**（同一 Tensor） |
| `kQuery` | `[dim]` | Query 向量 | 独立 |
| `kOutputMHA` | `[dim]` | MHA 输出（wo 之前） | 独立 |
| `kAttnOutput` | `[hidden_dim]` | Attention 输出（wo 之后） | 独立 |
| `kScoreStorage` | `[head_num, seq_len]` | 注意力分数矩阵 | 独立 |
| `kW1Output` | `[immediate_dim]` | FFN gate projection 输出 | SwiGLU 原地修改 |
| `kW3Output` | `[immediate_dim]` | FFN up projection 输出 | 独立 |

**关键复用策略** (`qwen3.cpp:334-340`):
```cpp
tensor::Tensor rms_output(DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);

// 三个 Buffer 键共享同一个 Tensor 对象
CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
```

**为什么可以复用？**
- 这三个 Buffer 在执行流中**不会同时被读取**
- 生命周期链：`kOutputRMSNorm` → Attention → `kFFNRMSNorm` → FFN → `kW2Output`
- 每个阶段结束后，Buffer 内容会被下一阶段覆盖
- **节省内存**：3 个 `[hidden_dim]` Buffer 合并为 1 个（例如 hidden_dim=3584 时节省 28KB×2 = 56KB）

**危险操作示例**（会导致 bug）:
```cpp
// 错误：如果在 FFN 中需要同时读取 kOutputRMSNorm 和 kFFNRMSNorm
auto attn_norm = get_buffer(kOutputRMSNorm);  // 期望是 Attention RMSNorm 输出
auto ffn_norm = get_buffer(kFFNRMSNorm);      // 但实际指向同一块内存！
// 此时两者内容相同，会导致计算错误
```

### 5.3 Tensor 构造与分配调用链

**Tensor 构造函数** (`tensor.cpp:35-49`):
```cpp
Tensor::Tensor(DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    size_ = dim0;
    
    if (need_alloc && alloc) {
        allocate(alloc);  // 路径 A：自己分配
    } else {
        if (ptr != nullptr) {
            CHECK(need_alloc == false);  // 不能既分配又传入外部指针
            init_buffer(alloc, data_type_, need_alloc, ptr);  // 路径 B：外部视图
        }
    }
}
```

**分配路径 A** (`tensor.cpp:187-193`):
```cpp
void Tensor::allocate(std::shared_ptr<DeviceAllocator> allocator) {
    CHECK_NE(allocator, nullptr);
    CHECK_NE(size_, 0);
    size_t byte_size = data_type_size(data_type_) * size_;
    buffer_ = std::make_shared<Buffer>(byte_size, allocator, nullptr);
    // Buffer 构造函数会调用 allocator->allocate(byte_size)
}
```

**Buffer 构造与分配** (`buffer.cpp:5-16`):
```cpp
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
               void* ptr, bool use_external)
    : byte_size_(byte_size), allocator_(allocator), 
      ptr_(ptr), use_external_(use_external) {
    
    if (!ptr_ && allocator_) {  // ptr 为 nullptr 且有分配器
        device_type_ = allocator_->device_type();
        use_external_ = false;  // 标记为内部管理
        ptr_ = allocator_->allocate(byte_size);  // 调用分配器分配内存
    }
}
```

**外部视图路径 B** (`tensor.cpp:178-184`):
```cpp
void Tensor::init_buffer(std::shared_ptr<DeviceAllocator> alloc,
                         DataType data_type, bool need_alloc, void* ptr) {
    CHECK_EQ(need_alloc, false);  // 外部视图不能要求分配
    size_t byte_size = data_type_size(data_type) * size_;
    buffer_ = std::make_shared<Buffer>(byte_size, nullptr, ptr, true);
    //                                             ^^^^^^  ^^^  ^^^^
    //                                             无分配器  外部指针  use_external=true
}
```

**完整调用链总结**:
```
init_mem()
  → Tensor(..., need_alloc=true, allocator)
    → Tensor::allocate(allocator)
      → Buffer(byte_size, allocator, nullptr)
        → allocator->allocate(byte_size)  // CPU: posix_memalign / CUDA: cudaMalloc 或池复用
          → 返回 void* ptr
        → Buffer::ptr_ = ptr, use_external_ = false
      → Tensor::buffer_ = Buffer 智能指针
    → insert_buffer(buffer_type, tensor)  // 存入 Model::buffers_ 映射表
```

### 5.4 内存使用量估算（以 Qwen3-0.5B 为例）

假设配置：
- `layer_num = 28`
- `seq_len = 32768`
- `hidden_dim = 3584`
- `dim = 3584` (head_num=28 * head_size=128)
- `kv_dim = 512` (kv_head_num=4 * head_size=128)
- `immediate_dim = 18944`
- `vocab_size = 151936`

**KV Cache**:
- Key: `28 * 32768 * 512 * 4 bytes = 1.8 GB`
- Value: `28 * 32768 * 512 * 4 bytes = 1.8 GB`
- 总计: `3.6 GB`

**RoPE Cache**:
- Sin + Cos: `2 * 128 * 32768 * 4 bytes = 32 MB`

**中间激活**（最大值）:
- `kW1Output/kW3Output`: `18944 * 4 = 74 KB`
- `kForwardOutput`: `151936 * 4 = 592 KB`
- 其他小 Buffer: ~50 KB

**总运行时内存**: ~3.7 GB（不包括权重）

**权重内存**（FP32）:
- 模型文件大小约 ~2 GB（FP32）
- CUDA 模式下需要额外 2 GB 显存存储权重副本
- **总 CUDA 显存需求**: 3.7 GB (运行时) + 2 GB (权重) = **5.7 GB**

---

## 6. 单层前向执行详解：从输入到输出的完整数据流

**代码位置**：`kuiper/source/model/qwen3.cpp:148-168`

### 6.1 Forward 主循环结构

```cpp
base::Status Qwen3Model::forward(const Tensor& input, const Tensor& pos_tensor,
                                 int& next) const {
    // 遍历所有 Transformer 层
    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
        attention_rms(layer_idx, input);       // 步骤 1: Attention 前归一化
        attention_qkv(layer_idx, pos_tensor);  // 步骤 2: QKV 投影 + RoPE
        attention_mha(layer_idx, pos_tensor);  // 步骤 3: 多头注意力
        feed_forward(layer_idx, input);        // 步骤 4: FFN + 第二个残差
    }
    cls_logits(input);  // 步骤 5: Final RMSNorm + LM Head
    return base::error::Success();
}
```

**单层数据流全景图**:
```
输入: input[hidden_dim] (来自上一层或 embedding)
│
├─────────────────── 残差路径 1 ─────────────────┐
│                                                 │
│   [步骤 1] Attention RMSNorm                   │
│      input → rmsnorm → rmsnorm_output           │
│                                                 │
│   [步骤 2] QKV 投影 + RoPE                     │
│      rmsnorm_output → wq → query                │
│      rmsnorm_output → wk → key (写入 cache)     │
│      rmsnorm_output → wv → value (写入 cache)   │
│      query_norm(query) → query                  │
│      key_norm(key) → key                        │
│      RoPE(query, key, pos) → query, key (原地)  │
│                                                 │
│   [步骤 3] 多头注意力                          │
│      query × key_cache → scores                 │
│      softmax(scores) → attention_weights        │
│      attention_weights × value_cache → mha_out  │
│      mha_out → wo → attn_output                 │
│                                                 │
└──> [残差连接 1]                                 │
     input = input + attn_output  ◄───────────────┘
│
├─────────────────── 残差路径 2 ─────────────────┐
│                                                 │
│   [步骤 4a] FFN RMSNorm                        │
│      input → rmsnorm → ffn_norm_output          │
│                                                 │
│   [步骤 4b] SwiGLU FFN                         │
│      ffn_norm_output → w1 → gate                │
│      ffn_norm_output → w3 → up                  │
│      SwiGLU(gate, up) → hidden                  │
│      hidden → w2 → w2_output                    │
│                                                 │
└──> [残差连接 2]                                 │
     input = input + w2_output  ◄─────────────────┘
│
输出: input[hidden_dim] (传递给下一层)
```

### 6.2 步骤 1: Attention RMSNorm

**代码位置**: `qwen3.cpp:460-469`

```cpp
void Qwen3Model::attention_rms(int32_t layer_idx, const Tensor& input) const {
    Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    auto rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
    STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}
```

**RMSNorm 公式**:
```
output[i] = input[i] / RMS(input) * weight[i]
其中 RMS(input) = sqrt(mean(input^2) + epsilon)
```

**实现细节** (`rmsnorm_kernel.cpp`):
1. 计算均方根：`sum = Σ(input[i]²)`，`rms = sqrt(sum / hidden_dim + 1e-6)`
2. 归一化并缩放：`output[i] = input[i] * weight[i] / rms`
3. **注意**：RMSNorm 不减去均值（区别于 LayerNorm），计算更快

**Shape 变换**: `[hidden_dim]` → `[hidden_dim]` (原地或写入新 Buffer)

### 6.3 步骤 2: QKV 投影与 RoPE（关键步骤）

**代码位置**: `qwen3.cpp:471-514`

#### 2.1 KV Cache 切片（零拷贝视图）

```cpp
void Qwen3Model::attention_qkv(int32_t layer_idx, const Tensor& pos_tensor) const {
    Tensor query = get_buffer(ModelBufferType::kQuery);
    int32_t pos = pos_tensor.index<int32_t>(0);  // 读取当前位置
    
    // 获取当前层当前位置的 KV Cache 切片视图
    auto [key, val] = slice_kv_cache(layer_idx, pos);
```

**`slice_kv_cache()` 实现** (`model.cpp:215-232`):
```cpp
std::pair<Tensor, Tensor> Model::slice_kv_cache(int32_t layer_idx, 
                                                 int32_t token_pos) const {
    // 计算偏移量：跳过前面的层 + 跳过当前层前面的位置
    int32_t layer_offset = layer_idx * seq_len * kv_dim;
    int32_t cache_offset = layer_offset + token_pos * kv_dim;
    
    // 获取 KV Cache 总 Buffer 中的子区域指针
    float* key_cache_ptr = const_cast<float*>(
        get_buffer(kKeyCache).ptr<float>(cache_offset));
    float* val_cache_ptr = const_cast<float*>(
        get_buffer(kValueCache).ptr<float>(cache_offset));
    
    // 创建视图 Tensor（use_external=true，不分配新内存）
    Tensor key(DataType::kDataTypeFp32, kv_dim, false, nullptr, key_cache_ptr);
    Tensor val(DataType::kDataTypeFp32, kv_dim, false, nullptr, val_cache_ptr);
    
    key.set_device_type(device_type_);
    val.set_device_type(device_type_);
    
    return {key, val};
}
```

**关键理解**:
- `key` 和 `val` 不是新分配的内存，而是指向 `kKeyCache` 和 `kValueCache` 特定位置的**窗口**
- 任何对 `key` 和 `val` 的写操作都会**直接修改 cache**
- 这实现了自动的 cache 更新机制，无需显式写回

**内存布局示意**:
```
kKeyCache: [layer_num, seq_len, kv_dim]
├── Layer 0: ┌─────────────────────────────────┐
│            │ pos0 │ pos1 │ pos2 │ ... │ posN │
│            └─────────────────────────────────┘
├── Layer 1: ┌─────────────────────────────────┐
│            │ pos0 │ pos1 │ pos2 │ ... │ posN │
│            └──────▲──────────────────────────┘
│                   │
│                   └── slice_kv_cache(1, 1) 返回的 key 视图
└── ...
```

#### 2.2 Query 投影与归一化

```cpp
    const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
    auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
    
    // Qwen3 特有：Query Norm（按 head 归一化）
    auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * layer_num + 1);
    query.reshape({(int32_t)query.size() / head_size, head_size});  // [dim] → [num_heads, head_size]
    query_norm->forward(query, query);  // 原地归一化每个 head
    query.reshape({(int32_t)query.size()});  // 恢复 [dim]
```

**Query Norm 的作用** (Qwen3 创新):
- 标准 Attention: `Attention(Q, K, V) = softmax(QK^T / √d) V`
- Qwen3 Attention: `Attention(Norm(Q), Norm(K), V) = softmax(Norm(Q)·Norm(K)^T / √d) V`
- **好处**: 稳定训练，减少大模型中 Q/K 的数值范围，改善梯度流

**Shape 变换**:
- `rmsnorm_output: [hidden_dim=3584]`
- `wq: [hidden_dim, dim]` = `[3584, 3584]`
- `query: [dim=3584]` = `[28 heads * 128 head_size]`
- Reshape: `[28, 128]` → Norm → `[28, 128]` → Reshape: `[3584]`

#### 2.3 Key/Value 投影与归一化

```cpp
    const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));  // key 是 cache 视图！
    
    // Key Norm
    auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * layer_num + 1);
    key.reshape({(int32_t)key.size() / head_size, head_size});
    key_norm->forward(key, key);  // 原地归一化
    key.reshape({(int32_t)key.size()});
    
    const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));  // val 也是 cache 视图！
```

**重要**:
- `key` 和 `val` 是通过 `slice_kv_cache()` 返回的视图
- `forward()` 的输出写入这些视图时，**自动更新了 KV Cache**
- 这就是为什么不需要显式调用 "写入 cache" 的操作

**Shape 变换**:
- `wk/wv: [hidden_dim, kv_dim]` = `[3584, 512]`
- `key/val: [kv_dim=512]` = `[4 kv_heads * 128 head_size]`
- Key Reshape: `[4, 128]` → Norm → `[4, 128]` → `[512]`

#### 2.4 RoPE 旋转位置编码

```cpp
    STATUS_CHECK(qwen_layers_->rope_layer_->forward(
        query, key, pos_tensor,
        get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache),
        tensor::Tensor{}));  // 空 Tensor 占位符
}
```

**RoPE 原理**（复数旋转视角）:
- 将 head 中的每两个维度 `(x, y)` 视为复数平面上的向量
- 根据位置 `pos` 旋转向量：
  ```
  [x']   [cos(θ)  -sin(θ)]   [x]
  [y'] = [sin(θ)   cos(θ)] * [y]
  
  其中 θ = pos / (base ^ (dim_idx / head_size))
  ```

**Qwen 的 RoPE 实现** (`rope_kernel.cpp:72-126` for Qwen):
```cpp
for (int i = 0; i < dim; i += head_size) {
    for (int head_dim = 0; head_dim < head_size/2; head_dim++) {
        // 从预计算的 cache 中查表
        float fci = sin_cache[pos * head_size + head_dim * 2];
        float fcr = cos_cache[pos * head_size + head_dim * 2];
        
        // 旋转 Query
        float v0 = vec[i + head_dim];
        float v1 = vec[i + head_dim + head_size/2];
        vec[i + head_dim] = v0 * fcr - v1 * fci;
        vec[i + head_dim + head_size/2] = v0 * fci + v1 * fcr;
    }
}
// Key 同样处理
```

**Qwen vs Llama RoPE 区别**:
| 维度 | Llama | Qwen |
|------|-------|------|
| 频率基数 | 10000 | 1000000 |
| 旋转配对方式 | `(vec[i], vec[i+1])` 相邻元素 | `(vec[i], vec[i+head_size/2])` 前后半段 |
| 长序列外推 | 需要调整基数 | 天然支持更长序列 |

**原地修改**:
- RoPE 直接修改 `query` 和 `key` 的内容（in-place）
- 由于 `key` 是 cache 视图，RoPE 后的 Key 已经存储在 cache 中

### 6.4 步骤 3: 多头注意力计算

**代码位置**: `qwen3.cpp:526-550`

```cpp
void Qwen3Model::attention_mha(int32_t layer_idx, const Tensor& pos_tensor) const {
    Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
    Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    Tensor query = get_buffer(ModelBufferType::kQuery);
    
    // 设置当前位置和层索引（MHA 内部需要）
    const auto& mha_layer = qwen_layers_->mha_layer_;
    int pos = pos_tensor.index<int32_t>(0);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    
    // 执行多头注意力
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
    
    // 输出投影（wo）
    Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
    const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
    STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}
```

**MHA Kernel 实现详解** (`mha_kernel.cpp:5-61`):

```cpp
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, 
                int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                const Tensor& mha_out, const Tensor& query_tensor, 
                const Tensor& score_tensor,
                const Tensor& key_cache_tensor, const Tensor& value_cache_tensor,
                DeviceType device_type, CudaConfig* config) {
    
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    float scale = 1.f / std::sqrt(static_cast<float>(head_size));  // √d 缩放
    
    // 遍历每个 Query Head
    for (int32_t h = 0; h < head_num; ++h) {
        float* score_head_addr = score_tensor.ptr<float>() + h * seq_len;
        float* query_head_addr = query_tensor.ptr<float>() + h * head_size;
        
        // 创建当前 head 的 Query 视图
        Tensor query_mat(DataType::kDataTypeFp32, head_size, 
                        false, nullptr, query_head_addr);
        query_mat.set_device_type(device_type);
        
        // 计算与所有历史 Key 的点积（Attention Score）
        for (int32_t t = 0; t <= pos; t++) {  // 只到当前位置（因果注意力）
            // Grouped Query Attention: 多个 query head 共享一个 kv head
            int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
            const float* key_head_addr = key_cache_tensor.ptr<float>() 
                                       + layer_offset + cache_offset;
            
            // 创建 Key 视图
            Tensor key_mat(DataType::kDataTypeFp32, 1, head_size, 
                          false, nullptr, const_cast<float*>(key_head_addr));
            
            // score[t] = query · key / √d
            Tensor score_mat(DataType::kDataTypeFp32, 1, 
                            false, nullptr, score_head_addr + t);
            key_mat.set_device_type(device_type);
            score_mat.set_device_type(device_type);
            get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
        }
        
        // Softmax 归一化 attention weights
        Tensor score_head_tensor(DataType::kDataTypeFp32, pos + 1, 
                                false, nullptr, score_head_addr);
        score_head_tensor.set_device_type(device_type);
        get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);
        
        // 加权求和 Values
        float* output_head_ptr = mha_out.ptr<float>() + h * head_size;
        allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                              config ? config->stream : nullptr, false);
        
        Tensor output_tensor(DataType::kDataTypeFp32, head_size, 
                            false, nullptr, output_head_ptr);
        output_tensor.set_device_type(device_type);
        
        int32_t cache_offset = (h / kv_mul) * head_size;
        float* value_head_addr = value_cache_tensor.ptr<float>() 
                               + layer_offset + cache_offset;
        Tensor value_tensor(DataType::kDataTypeFp32, head_size, 
                           false, nullptr, value_head_addr);
        
        // output = Σ(attention_weights[t] * value[t])
        get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, 
                                          pos, head_size, kv_dim, 
                                          config ? config->stream : nullptr);
    }
}
```

**关键实现细节**:

1. **Grouped Query Attention (GQA)**:
   - `kv_mul = head_num / kv_head_num` (例如 28/4 = 7)
   - Query head `h` 使用 Key/Value head `h / kv_mul`
   - Query heads 0-6 都使用 KV head 0
   - **内存节省**: KV Cache 只需 4 个 head 而不是 28 个（节省 75% KV cache）

2. **因果掩码**（Causal Masking）:
   - 循环 `for (t = 0; t <= pos; t++)` 只计算到当前位置
   - 未来位置的 scores 不计算，自然实现因果约束
   - 比显式掩码（写入 -inf）更高效

3. **数值稳定性**:
   - `scale = 1/√head_size` 防止点积值过大导致 softmax 饱和
   - Softmax 使用 `exp(x - max(x))` 防止溢出

4. **零拷贝优化**:
   - 所有中间 Tensor（query_mat, key_mat, score_mat）都是视图
   - 没有额外的内存分配和拷贝

**Shape 变换**:
- `query: [dim=3584]` → 视为 `[28 heads, 128 head_size]`
- `key_cache: [layer_num, seq_len, kv_dim=512]` → 每个 head: `[seq_len, 128]`
- `scores: [28 heads, seq_len]`
- `value_cache: [layer_num, seq_len, kv_dim=512]` → 每个 head: `[seq_len, 128]`
- `mha_output: [28 heads, 128]` = `[dim=3584]`

**输出投影**:
```cpp
STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
```
- `wo: [hidden_dim, dim]` = `[3584, 3584]`
- `mha_output: [dim=3584]`
- `attn_output: [hidden_dim=3584]`

### 6.5 步骤 4: Feed Forward Network (FFN)

**代码位置**: `qwen3.cpp:552-594`

```cpp
void Qwen3Model::feed_forward(int32_t layer_idx, const Tensor& input) const {
    // 4.1 第一个残差连接：input += attention_output
    STATUS_CHECK(qwen_layers_->add_layer_->forward(
        input, get_buffer(ModelBufferType::kAttnOutput), input));
    
    // 4.2 FFN RMSNorm
    Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
    const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + layer_num);
    STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));
    
    // 4.3 Gate Projection (W1) + 4.4 Up Projection (W3) 并行计算
    Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
    Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));
    
    // 4.5 SwiGLU 激活函数（原地写入 w1_output）
    STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));
    
    // 4.6 Down Projection (W2)
    Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
    STATUS_CHECK(w2_layer->forward(w1_output, w2_output));
    
    // 4.7 第二个残差连接：input += w2_output
    STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}
```

**SwiGLU 激活函数** (`swiglu_kernel.cpp`):
```cpp
// SwiGLU(x, y) = Swish(x) * y
// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
for (int i = 0; i < len; i++) {
    float swish = input1[i] / (1.0f + expf(-input1[i]));
    output[i] = swish * input2[i];
}
```

**为什么用 SwiGLU?**
- 传统 FFN: `FFN(x) = W2·ReLU(W1·x)`
- SwiGLU FFN: `FFN(x) = W2·(Swish(W1·x) ⊙ W3·x)`
- 优势：平滑激活 + 门控机制 → 更好的表达能力

**Shape 变换**:
- `w1/w3: [immediate_dim, hidden_dim] = [18944, 3584]`
- `gate/up: [immediate_dim=18944]`
- `w2: [hidden_dim, immediate_dim] = [3584, 18944]`
- `w2_output: [hidden_dim=3584]`

### 6.6 最终层归一化与 LM Head

**代码位置**: `qwen3.cpp:618-627`

```cpp
void Qwen3Model::cls_logits(const Tensor& input) const {
    // Final RMSNorm
    const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * layer_num);
    STATUS_CHECK(norm->forward(input, input));  // 原地归一化
    
    // LM Head 投影到词表空间
    Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
}
```

**Shape**: `[hidden_dim=3584] → [vocab_size=151936]`

### 6.7 后处理与采样

**代码位置**: `qwen3.cpp:629-641`

```cpp
int32_t Qwen3Model::post_processing(const Tensor& pos, bool is_prompt) const {
    Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    const float* forward_logits = forward_output.ptr<float>();
    
    int32_t next = 0;
    if (is_prompt) {
        next = -1;  // Prefill 阶段不采样
    } else {
        // Decode 阶段使用 TopK 采样
        next = sampler_->sample(forward_logits, forward_output.size(),
                               cuda_config_ ? cuda_config_->stream : nullptr);
    }
    return next;
}
```

**完整 forward 流程总结**:
```
[输入] input[hidden_dim]
  ↓
For each layer (layer_idx = 0 to layer_num-1):
  1. Attention RMSNorm
  2. QKV Projection + Norm + RoPE → 更新 KV Cache
  3. Multi-Head Attention + wo
  4. 残差连接 1: input += attn_output
  5. FFN RMSNorm
  6. W1/W3 + SwiGLU + W2
  7. 残差连接 2: input += w2_output
  ↓
[Final] RMSNorm + LM Head
  ↓
[输出] logits[vocab_size]
  ↓
[采样] Prefill: 丢弃 | Decode: TopK → next_token
```

---

## 7. 从算子到 Kernel 的下发机制（算子抽象层）

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
