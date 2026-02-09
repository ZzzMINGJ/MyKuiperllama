# PagedAttention 简易版设计文档

> 基于 MyKuiperllama 推理框架的 PagedAttention 实现方案

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [核心概念](#2-核心概念)
3. [现有实现分析](#3-现有实现分析)
4. [整体架构设计](#4-整体架构设计)
5. [数据结构设计](#5-数据结构设计)
6. [模块详细设计](#6-模块详细设计)
7. [Kernel 改造方案](#7-kernel-改造方案)
8. [模型层适配方案](#8-模型层适配方案)
9. [内存分析与收益估算](#9-内存分析与收益估算)
10. [分阶段实施计划](#10-分阶段实施计划)
11. [测试策略](#11-测试策略)
12. [附录：关键文件变更清单](#附录关键文件变更清单)

---

## 1. 背景与动机

### 1.1 问题：当前 KV Cache 的内存浪费

当前框架中，KV Cache 在模型初始化阶段被一次性预分配为形状 `[layer_num, seq_len, kv_dim]` 的连续大块内存（见各模型的 `init_mem()` 方法）。以典型配置为例：

| 参数 | Qwen3-1.7B 典型值 |
|------|-------------------|
| layer_num | 28 |
| seq_len | 32768 |
| kv_dim | 512 |
| 单份 Cache 大小 | 28 × 32768 × 512 × 4B = **1.75 GB** |
| K+V 合计 | **3.5 GB** |

这带来两个核心问题：

1. **内存浪费**：即使实际只使用了 100 个 token，也需要预分配全部 32768 个位置的内存。对于多个并发请求，浪费呈倍数增长。
2. **无法灵活管理**：连续内存要求使得无法在多个请求间动态共享和回收 KV Cache 空间。

### 1.2 PagedAttention 的核心思想

PagedAttention（来源：vLLM, 2023）借鉴操作系统虚拟内存分页的思想：

- 将 KV Cache 划分为固定大小的 **物理块（Block）**
- 每个 Block 存储固定数量 token 的 KV 数据
- 通过 **页表（Block Table）** 维护逻辑位置到物理块的映射
- 按需分配、用完释放，消除内存碎片和浪费

### 1.3 本方案的目标

实现一个 **简易版 PagedAttention**，目标如下：

- [x] 支持单请求场景下的分页式 KV Cache 管理
- [x] 消除预分配全量 seq_len 的内存浪费
- [x] 同时支持 CPU 和 CUDA 后端
- [x] 最小化对现有代码的侵入性改动
- [ ] 暂不实现：多请求批处理、前缀共享、Copy-on-Write 等高级特性

---

## 2. 核心概念

### 2.1 术语定义

| 术语 | 定义 |
|------|------|
| **Block（物理块）** | KV Cache 内存的最小分配单元，存储 `block_size` 个 token 的 K 或 V 数据 |
| **Block Size** | 每个 Block 可容纳的 token 数量（推荐值：16） |
| **Block Table（块表）** | 记录每个序列的逻辑 Block 到物理 Block ID 的映射，形状 `[max_num_blocks_per_seq]` |
| **Physical Block ID** | 物理内存池中 Block 的唯一索引 |
| **Logical Block ID** | 序列内的 Block 编号，`logical_block = token_pos / block_size` |
| **Slot（槽位）** | Block 内部的 token 位置，`slot = token_pos % block_size` |
| **Block Pool（块池）** | 管理所有物理 Block 的分配和释放的内存池 |

### 2.2 逻辑→物理地址映射

对于给定的 `layer_idx` 和 `token_pos`，KV 数据的物理地址计算过程如下：

```
logical_block_id = token_pos / block_size
slot_offset      = token_pos % block_size
physical_block_id = block_table[logical_block_id]

// 物理地址 = 物理块基址 + slot 偏移
address = block_pool_base
        + physical_block_id * (layer_num * block_size * kv_dim)
        + layer_idx * (block_size * kv_dim)
        + slot_offset * kv_dim
```

**可视化**：

```
逻辑视图（序列连续）            物理视图（Block 池）
┌───────────────────┐           ┌──────────────────────────────────┐
│ Token 0..15       │ ─── Block Table ──► │ Physical Block 3  │
│ (Logical Block 0) │           │  layer 0: [16 × kv_dim] K data │
├───────────────────┤           │  layer 1: [16 × kv_dim] K data │
│ Token 16..31      │ ──────────►│  ...                           │
│ (Logical Block 1) │           │  layer N: [16 × kv_dim] K data │
├───────────────────┤           ├──────────────────────────────────┤
│ Token 32..47      │ ──────────►│ Physical Block 7               │
│ (Logical Block 2) │           │  ...                            │
└───────────────────┘           ├──────────────────────────────────┤
                                │ Physical Block 0 (free)         │
                                ├──────────────────────────────────┤
                                │ Physical Block 1 (free)         │
                                └──────────────────────────────────┘
```

---

## 3. 现有实现分析

### 3.1 当前 KV Cache 生命周期

```
init_mem()                     → 预分配 [layer_num, seq_len, kv_dim] 连续内存
    ↓
slice_kv_cache(layer, pos)     → 返回零拷贝视图，指向 cache 内特定偏移
    ↓
wk.forward(input, key_view)    → QKV 投影直接写入 cache 视图
    ↓
mha_kernel(key_cache, val_cache) → MHA 从 cache 读取所有历史 KV
```

### 3.2 涉及 KV Cache 的关键代码路径

| 组件 | 文件 | 行号 | 当前逻辑 |
|------|------|------|----------|
| KV Cache 分配 | `qwen3.cpp` | 348-355 | `Tensor(layer_num, seq_len, kv_dim)` 一次性分配 |
| KV Cache 分配 | `llama3.cpp` | 469-475 | 同上 |
| KV Cache 分配 | `qwen2.cpp` | 473-479 | 同上 |
| Cache 切片 | `model.cpp` | 215-232 | `slice_kv_cache()` 线性偏移计算 |
| Score 缓冲区 | 各模型 `init_mem()` | - | `Tensor(head_num, seq_len)` 预分配全量 |
| CPU MHA | `mha_kernel.cpp` | 28-39 | `key_cache + layer_offset + t * kv_dim + head_offset` |
| CUDA MHA | `mha_kernel.cu` | 76-90 | `key_cache + layer_offset + t * kv_dim + head_offset` |
| MHA 调用接口 | `kernels_interface.h` | 22-28 | `MHAKernel` 签名：传入整个 key/value cache tensor |

### 3.3 当前内存布局

```
Key Cache 内存布局 (连续):
┌─────────────────────────────────────────────────────┐
│ Layer 0                                             │
│ ┌───────┬───────┬───────┬─ ─ ─ ─┬───────┐          │
│ │ Pos 0 │ Pos 1 │ Pos 2 │       │Pos N-1│          │
│ │kv_dim │kv_dim │kv_dim │       │kv_dim │          │
│ └───────┴───────┴───────┴─ ─ ─ ─┴───────┘          │
├─────────────────────────────────────────────────────┤
│ Layer 1                                             │
│ ┌───────┬───────┬───────┬─ ─ ─ ─┬───────┐          │
│ │ Pos 0 │ Pos 1 │ Pos 2 │       │Pos N-1│          │
│ └───────┴───────┴───────┴─ ─ ─ ─┴───────┘          │
├─────────────────────────────────────────────────────┤
│ ...                                                 │
└─────────────────────────────────────────────────────┘
偏移量 = layer_idx * seq_len * kv_dim + token_pos * kv_dim
```

---

## 4. 整体架构设计

### 4.1 新增组件

在现有四层架构（控制流 → 模型装配 → 算子抽象 → Kernel）的基础上，新增一个 **KV Cache 管理层**，位于模型装配层和内存层之间：

```
┌─────────────────────────────────────────┐
│         控制流层 (demo/*.cpp)            │
│   generate → predict → forward          │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│       模型装配层 (model/*.cpp)           │
│  attention_qkv() 调用 cache_manager     │
│  attention_mha() 传入 block_table       │
└───────────┬─────────────┬───────────────┘
            │             │
┌───────────▼──────┐ ┌────▼──────────────────────┐
│  算子/Kernel 层  │ │ KV Cache 管理层 [NEW]     │
│  paged_mha_kernel│ │ BlockManager              │
│  (分页感知)      │ │ BlockAllocator            │
└───────────┬──────┘ │ BlockTable                │
            │        └────┬──────────────────────┘
┌───────────▼─────────────▼───────────────┐
│       内存管理层 (Buffer/Allocator)      │
│  物理 Block 内存由现有 Allocator 管理    │
└─────────────────────────────────────────┘
```

### 4.2 设计原则

1. **最小侵入**：现有 `Tensor`、`Buffer`、`DeviceAllocator` 不做修改，新增管理层在其上构建
2. **接口兼容**：`Model::slice_kv_cache()` 改为通过 `BlockManager` 路由，对外语义不变
3. **Kernel 双版本共存**：新增 `paged_mha_kernel` 与原有 `mha_kernel` 并存，通过配置切换
4. **设备无关**：`BlockManager` 逻辑层设备无关，物理内存分配委托给对应设备的 Allocator

---

## 5. 数据结构设计

### 5.1 核心类关系

```
┌─────────────────────┐      ┌─────────────────────┐
│    BlockAllocator    │      │     BlockTable       │
│  ─────────────────── │      │  ─────────────────── │
│  pool_base_ptr_      │◄─────│  block_ids_[]        │
│  free_list_          │      │  num_logical_blocks_  │
│  total_blocks_       │      │  block_size_          │
│  block_size_bytes_   │      └──────────┬────────────┘
│  device_type_        │                 │
│  ─────────────────── │                 │ 1:1 per sequence
│  allocate() → int    │      ┌──────────▼────────────┐
│  free(int block_id)  │      │    BlockManager       │
│  get_ptr(int id)     │      │  ─────────────────────│
└─────────────────────┘      │  k_allocator_         │
                              │  v_allocator_         │
                              │  block_table_         │
                              │  block_size_          │
                              │  ─────────────────────│
                              │  allocate_slot(pos)   │
                              │  get_kv_slot(l, pos)  │
                              │  get_block_table_tensor() │
                              │  reset()              │
                              └───────────────────────┘
```

### 5.2 `BlockAllocator` 类

负责物理 Block 的内存池管理，管理 K 或 V 其中一种 cache。

```cpp
// 文件: kuiper/include/base/block_allocator.h

namespace base {

class BlockAllocator {
 public:
  /// @brief 构造函数
  /// @param device_type CPU 或 CUDA
  /// @param block_size 每个 Block 可容纳的 token 数量
  /// @param num_blocks 物理 Block 总数
  /// @param layer_num Transformer 层数
  /// @param kv_dim KV head 维度（= head_size * kv_head_num）
  BlockAllocator(DeviceType device_type, int32_t block_size,
                 int32_t num_blocks, int32_t layer_num, int32_t kv_dim);

  ~BlockAllocator();

  /// @brief 分配一个空闲物理 Block，返回 Block ID
  /// @return 物理 Block ID，如果无空闲块则返回 -1
  int32_t allocate();

  /// @brief 释放指定物理 Block
  void free(int32_t block_id);

  /// @brief 获取指定物理 Block 内某层某 slot 的数据指针
  /// @param block_id 物理 Block ID
  /// @param layer_idx 层索引
  /// @param slot 块内 token 偏移 (0 ~ block_size-1)
  /// @return 指向 kv_dim 个 float 的指针
  float* get_slot_ptr(int32_t block_id, int32_t layer_idx, int32_t slot) const;

  /// @brief 获取整个 Block Pool 的基地址（用于传给 Kernel）
  float* pool_base_ptr() const;

  /// @brief 剩余空闲 Block 数量
  int32_t num_free_blocks() const;

  /// @brief 每个 Block 的总字节数（跨所有层）
  size_t block_bytes() const;

 private:
  DeviceType device_type_;
  int32_t block_size_;      // token 数 / block
  int32_t num_blocks_;      // 物理块总数
  int32_t layer_num_;
  int32_t kv_dim_;

  void* pool_ptr_ = nullptr;               // 整个池的基地址
  std::shared_ptr<DeviceAllocator> alloc_;  // 复用现有 Allocator
  std::vector<int32_t> free_list_;          // 空闲块 ID 栈
};

}  // namespace base
```

**内存布局**（单个物理 Block 内部）：

```
Physical Block [block_id]:
┌──────────────────────────────────────────────────────────┐
│ Layer 0                                                  │
│ ┌──────────┬──────────┬──────────┬─ ─ ─┬──────────┐      │
│ │  Slot 0  │  Slot 1  │  Slot 2  │     │ Slot B-1 │      │
│ │ [kv_dim] │ [kv_dim] │ [kv_dim] │     │ [kv_dim] │      │
│ └──────────┴──────────┴──────────┴─ ─ ─┴──────────┘      │
├──────────────────────────────────────────────────────────┤
│ Layer 1                                                  │
│ ┌──────────┬──────────┬ ─ ─ ─ ─ ─┬──────────┐            │
│ │  Slot 0  │  Slot 1  │           │ Slot B-1 │            │
│ └──────────┴──────────┴ ─ ─ ─ ─ ─┴──────────┘            │
├──────────────────────────────────────────────────────────┤
│ ...                                                      │
├──────────────────────────────────────────────────────────┤
│ Layer L-1                                                │
│ ┌──────────┬──────────┬ ─ ─ ─ ─ ─┬──────────┐            │
│ │  Slot 0  │  Slot 1  │           │ Slot B-1 │            │
│ └──────────┴──────────┴ ─ ─ ─ ─ ─┴──────────┘            │
└──────────────────────────────────────────────────────────┘

每个 Block 大小 = layer_num * block_size * kv_dim * sizeof(float)
Block 在 Pool 中的偏移 = block_id * block_bytes()
```

### 5.3 `BlockTable` 类

维护单个序列的逻辑 Block → 物理 Block 映射。

```cpp
// 文件: kuiper/include/base/block_table.h

namespace base {

class BlockTable {
 public:
  explicit BlockTable(int32_t block_size, int32_t max_blocks_per_seq);

  /// @brief 追加一个物理 Block ID 到表尾
  void append(int32_t physical_block_id);

  /// @brief 获取第 logical_idx 个逻辑块对应的物理块 ID
  int32_t get_physical_block(int32_t logical_idx) const;

  /// @brief 当前已分配的逻辑块数量
  int32_t num_blocks() const;

  /// @brief 导出为 Tensor，用于传给 Kernel
  ///        shape: [num_blocks], dtype: int32
  tensor::Tensor to_tensor(DeviceType device_type) const;

  /// @brief 重置
  void reset();

 private:
  int32_t block_size_;
  int32_t max_blocks_per_seq_;
  std::vector<int32_t> table_;  // table_[logical_idx] = physical_block_id
};

}  // namespace base
```

### 5.4 `BlockManager` 类

统一管理 K Cache 和 V Cache 的 Block 分配，对上层提供简洁接口。

```cpp
// 文件: kuiper/include/base/block_manager.h

namespace base {

class BlockManager {
 public:
  /// @brief 构造函数
  /// @param device_type 设备类型
  /// @param block_size 每块 token 数
  /// @param max_num_blocks 物理块总数上限
  /// @param layer_num 层数
  /// @param kv_dim KV 维度
  /// @param max_seq_len 最大序列长度（用于计算 max_blocks_per_seq）
  BlockManager(DeviceType device_type, int32_t block_size, int32_t max_num_blocks,
               int32_t layer_num, int32_t kv_dim, int32_t max_seq_len);

  /// @brief 为新 token 分配 KV cache slot
  ///        如果 token_pos 跨入新的逻辑块，自动分配新物理块
  /// @return Status::kSuccess 或 Status::kInternalError（无空闲块）
  Status allocate_slot(int32_t token_pos);

  /// @brief 获取指定层、指定位置的 K/V 写入指针
  ///        供 attention_qkv() 中 wk/wv.forward() 使用
  std::pair<float*, float*> get_kv_slot_ptr(int32_t layer_idx, int32_t token_pos) const;

  /// @brief 获取 K/V block pool 基地址（传给 Kernel）
  float* k_pool_ptr() const;
  float* v_pool_ptr() const;

  /// @brief 获取 block table tensor（传给 Kernel）
  tensor::Tensor get_block_table_tensor() const;

  /// @brief 获取 block_size（传给 Kernel）
  int32_t block_size() const;

  /// @brief 释放所有已分配块，重置状态（新请求开始时调用）
  void reset();

  /// @brief 当前已分配的 token 数量
  int32_t num_allocated_tokens() const;

  /// @brief 剩余可用 Block 数
  int32_t num_free_blocks() const;

 private:
  int32_t block_size_;
  int32_t layer_num_;
  int32_t kv_dim_;
  int32_t max_blocks_per_seq_;
  int32_t num_allocated_tokens_ = 0;

  std::unique_ptr<BlockAllocator> k_allocator_;
  std::unique_ptr<BlockAllocator> v_allocator_;
  BlockTable block_table_;
};

}  // namespace base
```

---

## 6. 模块详细设计

### 6.1 `BlockAllocator` 实现细节

```cpp
// 文件: kuiper/source/base/block_allocator.cpp

BlockAllocator::BlockAllocator(DeviceType device_type, int32_t block_size,
                               int32_t num_blocks, int32_t layer_num, int32_t kv_dim)
    : device_type_(device_type),
      block_size_(block_size),
      num_blocks_(num_blocks),
      layer_num_(layer_num),
      kv_dim_(kv_dim) {
  // 选择对应设备的 Allocator（复用已有工厂）
  if (device_type == DeviceType::kDeviceCPU) {
    alloc_ = CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc_ = CUDADeviceAllocatorFactory::get_instance();
  }

  // 一次性分配整个 Block Pool
  size_t total_bytes = static_cast<size_t>(num_blocks) * block_bytes();
  pool_ptr_ = alloc_->allocate(total_bytes);
  alloc_->memset_zero(pool_ptr_, total_bytes, nullptr, true);

  // 初始化空闲列表（所有 Block 都是空闲的）
  free_list_.reserve(num_blocks);
  for (int32_t i = num_blocks - 1; i >= 0; --i) {
    free_list_.push_back(i);
  }
}

size_t BlockAllocator::block_bytes() const {
  return static_cast<size_t>(layer_num_) * block_size_ * kv_dim_ * sizeof(float);
}

int32_t BlockAllocator::allocate() {
  if (free_list_.empty()) {
    return -1;  // 无空闲块
  }
  int32_t block_id = free_list_.back();
  free_list_.pop_back();
  return block_id;
}

void BlockAllocator::free(int32_t block_id) {
  free_list_.push_back(block_id);
}

float* BlockAllocator::get_slot_ptr(int32_t block_id, int32_t layer_idx,
                                    int32_t slot) const {
  size_t offset = static_cast<size_t>(block_id) * block_bytes() / sizeof(float)
                + static_cast<size_t>(layer_idx) * block_size_ * kv_dim_
                + static_cast<size_t>(slot) * kv_dim_;
  return static_cast<float*>(pool_ptr_) + offset;
}
```

### 6.2 `BlockManager` 核心逻辑

```cpp
// allocate_slot() 的逻辑
Status BlockManager::allocate_slot(int32_t token_pos) {
  int32_t logical_block = token_pos / block_size_;
  int32_t slot = token_pos % block_size_;

  // 需要新 Block？（当进入新的逻辑块的第一个 slot 时）
  if (slot == 0) {
    // 从 K 和 V allocator 各分配一个物理块
    int32_t k_block = k_allocator_->allocate();
    int32_t v_block = v_allocator_->allocate();
    if (k_block < 0 || v_block < 0) {
      return error::InternalError("No free blocks available");
    }
    // K 和 V 使用相同的逻辑→物理映射
    // （简易版使用同一个 block_table，K/V 的物理块 ID 相同）
    block_table_.append(k_block);
    // 注意：简易版要求 K/V allocator 返回相同的 block_id
    // 实现方式：K/V allocator 同步分配
  }

  num_allocated_tokens_ = token_pos + 1;
  return error::Success();
}

std::pair<float*, float*> BlockManager::get_kv_slot_ptr(
    int32_t layer_idx, int32_t token_pos) const {
  int32_t logical_block = token_pos / block_size_;
  int32_t slot = token_pos % block_size_;
  int32_t physical_block = block_table_.get_physical_block(logical_block);

  float* k_ptr = k_allocator_->get_slot_ptr(physical_block, layer_idx, slot);
  float* v_ptr = v_allocator_->get_slot_ptr(physical_block, layer_idx, slot);
  return {k_ptr, v_ptr};
}
```

### 6.3 简化设计：K/V 同步分配

在简易版中，K 和 V 的 BlockAllocator 使用 **相同的分配顺序**，因此对于同一个逻辑块，K 和 V 的物理块 ID 总是相同的。这使得只需要 **一张 Block Table** 就可以同时索引 K 和 V，大幅简化 Kernel 的实现。

---

## 7. Kernel 改造方案

### 7.1 新增 Kernel 签名

```cpp
// 文件: kuiper/source/op/kernels/kernels_interface.h (新增)

typedef void (*PagedMHAKernel)(
    int32_t pos,             // 当前 token 位置
    int32_t head_num,        // 注意力头数
    int32_t kv_dim,          // KV 维度
    int32_t kv_mul,          // GQA 倍数
    int32_t head_size,       // 每头维度
    int32_t block_size,      // 每块 token 数
    int32_t layer_idx,       // 当前层索引
    const tensor::Tensor& mha_out,     // [dim] 输出
    const tensor::Tensor& query,       // [dim] 查询
    const tensor::Tensor& score,       // [head_num, max_num_blocks * block_size] 分数缓冲
    const tensor::Tensor& k_cache,     // K block pool 整体
    const tensor::Tensor& v_cache,     // V block pool 整体
    const tensor::Tensor& block_table, // [num_logical_blocks] int32 页表
    base::DeviceType device_type,
    CudaConfig* config
);

PagedMHAKernel get_paged_mha_kernel(base::DeviceType device_type);
```

### 7.2 CUDA Paged MHA Kernel

核心改动：将线性寻址 `key_cache + layer_offset + t * kv_dim` 改为分页寻址。

```cuda
// 文件: kuiper/source/op/kernels/cuda/paged_mha_kernel.cu

__global__ void paged_mha_kernel_impl(
    int32_t pos,
    int32_t head_num, int32_t kv_dim, int32_t kv_mul,
    int32_t head_size, int32_t block_size, int32_t layer_idx,
    float* __restrict__ query,
    float* __restrict__ score_ptr,
    float* __restrict__ output,
    const float* __restrict__ k_pool,     // K block pool 基址
    const float* __restrict__ v_pool,     // V block pool 基址
    const int32_t* __restrict__ block_table,  // 页表
    int32_t num_layers,       // layer_num (用于计算 block 内层偏移)
    int32_t max_score_len     // score 缓冲区每头的长度
) {
  int head = blockIdx.x;
  if (head >= head_num) return;

  extern __shared__ float s_query[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // 预加载 query 到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * max_score_len;
  int head_offset = (head / kv_mul) * head_size;

  // 单块大小（所有层的数据总量，以 float 计）
  int block_stride = num_layers * block_size * kv_dim;

  // ========= 计算注意力分数（分页寻址）=========
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    int logical_block = t / block_size;
    int slot = t % block_size;
    int physical_block = block_table[logical_block];

    // K 地址 = pool_base + physical_block * block_stride
    //        + layer_idx * (block_size * kv_dim)
    //        + slot * kv_dim + head_offset
    const float* key_head = k_pool
        + physical_block * block_stride
        + layer_idx * (block_size * kv_dim)
        + slot * kv_dim
        + head_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 kv = *reinterpret_cast<const float4*>(key_head + i);
      float4 qv = *reinterpret_cast<const float4*>(s_query + i);
      score += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  // ========= Softmax =========
  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  // ========= 加权求和 V（分页寻址）=========
  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      int logical_block = t / block_size;
      int slot = t % block_size;
      int physical_block = block_table[logical_block];

      const float* value_head = v_pool
          + physical_block * block_stride
          + layer_idx * (block_size * kv_dim)
          + slot * kv_dim
          + head_offset;

      val += score_head[t] * value_head[i];
    }
    output_head[i] = val;
  }
}
```

**与原始 CUDA Kernel 的对比**：

| 方面 | 原始 `mha_kernel.cu` | 新 `paged_mha_kernel.cu` |
|------|---------------------|--------------------------|
| K/V 地址计算 | `cache + layer_offset + t * kv_dim + head_offset` | 通过 `block_table` 查表后计算 |
| 参数 | `layer_offset` (整数偏移) | `block_table` (int32 tensor) + `block_size` |
| 内存访问模式 | 连续线性访问 | 逻辑连续但物理可能不连续 |
| Grid/Block 配置 | `<<<head_num, 256>>>` | `<<<head_num, 256>>>` (不变) |
| 共享内存用法 | query 预加载 | query 预加载 (不变) |

### 7.3 CPU Paged MHA Kernel

CPU 版本改动类似，将线性偏移替换为分页查表：

```cpp
// 文件: kuiper/source/op/kernels/cpu/paged_mha_kernel.cpp

void paged_mha_kernel_cpu(
    int32_t pos, int32_t head_num, int32_t kv_dim, int32_t kv_mul,
    int32_t head_size, int32_t block_size, int32_t layer_idx,
    const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor,
    const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
    const tensor::Tensor& block_table_tensor,
    base::DeviceType device_type, CudaConfig* config)
{
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));
  int32_t num_layers = /* 从 block 内存布局推算，或作为参数传入 */;
  int32_t block_stride = num_layers * block_size * kv_dim;

  const int32_t* block_table = block_table_tensor.ptr<int32_t>();
  const float* k_pool = k_cache.ptr<float>();
  const float* v_pool = v_cache.ptr<float>();

  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head = const_cast<float*>(score_tensor.ptr<float>()) + h * (pos + 1);
    float* query_head = const_cast<float*>(query_tensor.ptr<float>()) + h * head_size;
    int32_t head_offset = (h / kv_mul) * head_size;

    // 计算注意力分数
    for (int32_t t = 0; t <= pos; ++t) {
      int32_t logical_block = t / block_size;
      int32_t slot = t % block_size;
      int32_t physical_block = block_table[logical_block];

      const float* key_head = k_pool
          + physical_block * block_stride
          + layer_idx * (block_size * kv_dim)
          + slot * kv_dim + head_offset;

      float score = 0.f;
      for (int32_t d = 0; d < head_size; ++d) {
        score += query_head[d] * key_head[d];
      }
      score_head[t] = score * scale;
    }

    // softmax
    kernel::softmax_inplace_cpu(score_head, pos + 1);

    // 加权求和 V
    float* output_head = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    std::memset(output_head, 0, head_size * sizeof(float));

    for (int32_t t = 0; t <= pos; ++t) {
      int32_t logical_block = t / block_size;
      int32_t slot = t % block_size;
      int32_t physical_block = block_table[logical_block];

      const float* value_head = v_pool
          + physical_block * block_stride
          + layer_idx * (block_size * kv_dim)
          + slot * kv_dim + head_offset;

      float s = score_head[t];
      for (int32_t d = 0; d < head_size; ++d) {
        output_head[d] += s * value_head[d];
      }
    }
  }
}
```

---

## 8. 模型层适配方案

### 8.1 `Model` 基类变更

```cpp
// 文件: kuiper/include/model/model.h (修改)

class Model {
 // ... 现有成员 ...

 protected:
  // 新增：Block Manager（PagedAttention 模式下非空）
  std::unique_ptr<base::BlockManager> block_manager_;

  // 新增：PagedAttention 配置
  bool use_paged_attention_ = false;
  int32_t block_size_ = 16;  // 默认 block size

 public:
  // 修改：slice_kv_cache 改为虚函数，支持 paged 模式重写
  // 原有签名不变，内部实现分支
  virtual std::pair<tensor::Tensor, tensor::Tensor>
  slice_kv_cache(int32_t layer_idx, int32_t token_pos) const override;

  // 新增：PagedAttention 专用接口
  base::BlockManager* get_block_manager() const;
};
```

### 8.2 `slice_kv_cache()` 适配

```cpp
// 文件: kuiper/source/model/model.cpp (修改)

std::pair<Tensor, Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                int32_t token_pos) const {
  if (use_paged_attention_ && block_manager_) {
    // === PagedAttention 路径 ===
    // 如果进入新的逻辑块，先分配
    const_cast<BlockManager*>(block_manager_.get())->allocate_slot(token_pos);

    // 获取分页后的写入指针
    auto [k_ptr, v_ptr] = block_manager_->get_kv_slot_ptr(layer_idx, token_pos);

    tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_,
                       false, nullptr, k_ptr);
    tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_,
                       false, nullptr, v_ptr);
    key.set_device_type(device_type_);
    val.set_device_type(device_type_);
    return {key, val};
  }

  // === 原有路径（保持不变）===
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;
  float* key_cache_ptr = const_cast<float*>(
      get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr = const_cast<float*>(
      get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));
  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_,
                     false, nullptr, key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_,
                     false, nullptr, val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}
```

### 8.3 模型 `init_mem()` 适配

以 Qwen3 为例（LLama2/Qwen2 同理）：

```cpp
// 文件: kuiper/source/model/qwen3.cpp (修改 init_mem())

void Qwen3Model::init_mem() {
  // ... 其他 buffer 分配不变 ...

  if (use_paged_attention_) {
    // === PagedAttention 路径 ===
    // 计算需要的最大物理块数
    // 这里可以设为按需增长，初始分配一定比例
    int32_t max_blocks_per_seq = (config_->seq_len_ + block_size_ - 1) / block_size_;
    int32_t num_physical_blocks = max_blocks_per_seq;  // 简易版：1:1 映射

    block_manager_ = std::make_unique<base::BlockManager>(
        device_type_, block_size_, num_physical_blocks,
        config_->layer_num_, config_->kv_dim_, config_->seq_len_);

    // 不再需要 kKeyCache/kValueCache 大 buffer
    // 但仍需要 score storage
  } else {
    // === 原有路径 ===
    tensor::Tensor key_cache(DataType::kDataTypeFp32,
        config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, alloc);
    tensor::Tensor value_cache(DataType::kDataTypeFp32,
        config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, alloc);
    insert_buffer(ModelBufferType::kKeyCache, key_cache);
    insert_buffer(ModelBufferType::kValueCache, value_cache);
  }
}
```

### 8.4 `attention_mha()` 适配

```cpp
// 文件: kuiper/source/model/qwen3.cpp (修改 attention_mha())

void Qwen3Model::attention_mha(int32_t layer_idx,
                               const tensor::Tensor& pos_tensor) const {
  // ... 获取 query, score, mha_out 不变 ...

  if (use_paged_attention_) {
    // === PagedAttention 路径 ===
    // 构造 K/V pool tensor（零拷贝视图）
    tensor::Tensor k_pool_tensor(base::DataType::kDataTypeFp32,
        /* total K pool size */, false, nullptr, block_manager_->k_pool_ptr());
    tensor::Tensor v_pool_tensor(base::DataType::kDataTypeFp32,
        /* total V pool size */, false, nullptr, block_manager_->v_pool_ptr());
    k_pool_tensor.set_device_type(device_type_);
    v_pool_tensor.set_device_type(device_type_);

    tensor::Tensor block_table_tensor = block_manager_->get_block_table_tensor();

    // 调用分页 MHA kernel
    kernel::get_paged_mha_kernel(device_type_)(
        pos, head_num, kv_dim, kv_mul, head_size, block_size_, layer_idx,
        mha_out, query, score, k_pool_tensor, v_pool_tensor,
        block_table_tensor, device_type_, cuda_config_.get());
  } else {
    // === 原有路径 ===
    kernel::get_mha_kernel(device_type_)(
        pos, head_num, layer_idx, seq_len, kv_dim, kv_mul, head_size,
        mha_out, query, score, key_cache, value_cache, device_type_,
        cuda_config_.get());
  }
}
```

### 8.5 新增 `MultiHeadAttention` 子类或模式

为了不修改现有 `MultiHeadAttention` 类太多，推荐新增 `PagedMultiHeadAttention` 层：

```cpp
// 文件: kuiper/include/op/paged_mha.h (新增)

namespace op {
class PagedMultiHeadAttention : public Layer {
 public:
  PagedMultiHeadAttention(base::DeviceType device_type,
                          int32_t kv_mul, int32_t kv_dim,
                          int32_t head_num, int32_t head_size,
                          int32_t block_size, int32_t layer_idx);

  void set_pos(int32_t pos);
  base::Status forward() override;
  // inputs: [0]=query, [1]=score, [2]=k_pool, [3]=v_pool, [4]=block_table
  // outputs: [0]=mha_out

 private:
  int32_t pos_ = 0;
  int32_t layer_idx_ = 0;
  int32_t kv_mul_, kv_dim_, head_num_, head_size_, block_size_;
};
}  // namespace op
```

---

## 9. 内存分析与收益估算

### 9.1 内存使用对比

以 Qwen3-1.7B（layer=28, kv_dim=512, head_size=128）为例：

**原始方案**（seq_len=32768）：

| 项目 | 大小 |
|------|------|
| K Cache | 28 × 32768 × 512 × 4B = **1.75 GB** |
| V Cache | 同上 = **1.75 GB** |
| **KV Cache 合计** | **3.5 GB** |

**PagedAttention 方案**（block_size=16，实际使用 1024 tokens）：

| 项目 | 计算 | 大小 |
|------|------|------|
| 实际需要逻辑块数 | 1024 / 16 = 64 | - |
| 每块大小 | 28 × 16 × 512 × 4B | 917 KB |
| K Cache 实际使用 | 64 × 917KB | **57.3 MB** |
| V Cache 实际使用 | 同上 | **57.3 MB** |
| Block Table | 64 × 4B | 256 B (可忽略) |
| **KV Cache 合计** | | **~115 MB** |
| **节省** | | **约 97%** |

### 9.2 Block Size 选择分析

| block_size | 优点 | 缺点 |
|-----------|------|------|
| 1 | 最细粒度，零浪费 | 页表过大，Kernel 频繁查表 |
| 8 | 粒度较细 | 每次寻址需要更多计算 |
| **16** | **平衡粒度和开销** | **推荐值** |
| 32 | 块内连续访问友好 | 最后一块可能浪费较多 |
| 64 | GPU cache line 友好 | 粒度太粗 |

**推荐 block_size=16**，理由：
- 16 × kv_dim × sizeof(float) = 16 × 512 × 4 = 32KB，与 L1 cache 大小匹配
- 块内浪费最多 15 tokens（可接受）
- 页表不会过大：32768 / 16 = 2048 个 int32 = 8KB

### 9.3 性能开销分析

| 方面 | 影响 | 缓解措施 |
|------|------|----------|
| 页表查询 | 每个 token 额外 1 次 int32 全局内存读取 | block_table 小，易于缓存 |
| 非连续内存访问 | 可能降低 GPU 内存合并效率 | 块内连续，跨块时间局部性好 |
| Block 分配开销 | CPU 端 free_list pop/push | O(1)，可忽略 |
| 额外参数传递 | Kernel 参数增加 | 不影响计算 |

**预期性能影响**：在推理（decode）阶段，每步只处理 1 个 token，MHA Kernel 的瓶颈是对历史 KV 的读取（memory-bound），分页寻址增加的计算量相比内存访问延迟可忽略，预计性能降幅 < 5%。

---

## 10. 分阶段实施计划

### Phase 1: 基础设施（Block 管理层）

**目标**：实现 `BlockAllocator`、`BlockTable`、`BlockManager` 三个类并通过单元测试。

**新增文件**：
```
kuiper/include/base/block_allocator.h
kuiper/include/base/block_table.h
kuiper/include/base/block_manager.h
kuiper/source/base/block_allocator.cpp
kuiper/source/base/block_table.cpp
kuiper/source/base/block_manager.cpp
test/test_op/test_block_manager.cpp
```

**验收标准**：
- [x] `BlockAllocator` 能正确分配/释放/获取指针
- [x] `BlockTable` 能正确维护映射关系并导出 tensor
- [x] `BlockManager` 能为连续 token 序列正确分配 slot
- [x] CUDA 设备同样可以正确分配和释放

### Phase 2: Kernel 实现

**目标**：实现 CPU 和 CUDA 的 Paged MHA Kernel，通过数值正确性测试。

**新增文件**：
```
kuiper/source/op/kernels/cpu/paged_mha_kernel.h
kuiper/source/op/kernels/cpu/paged_mha_kernel.cpp
kuiper/source/op/kernels/cuda/paged_mha_kernel.cuh
kuiper/source/op/kernels/cuda/paged_mha_kernel.cu
kuiper/include/op/paged_mha.h
kuiper/source/op/paged_mha.cpp
test/test_op/test_paged_mha.cpp
```

**修改文件**：
```
kuiper/source/op/kernels/kernels_interface.h   (新增 PagedMHAKernel typedef)
kuiper/source/op/kernels/kernels_interfaces.cpp (新增 get_paged_mha_kernel)
```

**验收标准**：
- [x] 对于相同的 QKV 输入，Paged MHA 输出与原始 MHA 的输出在 FP32 精度下 bit-exact 一致
- [x] GQA 场景正确

### Phase 3: 模型层集成

**目标**：将 PagedAttention 集成到模型推理流程，端到端测试。

**修改文件**：
```
kuiper/include/model/model.h   (新增 block_manager_ 等成员)
kuiper/source/model/model.cpp  (修改 slice_kv_cache)
kuiper/include/base/base.h     (可能新增 ModelBufferType)
kuiper/source/model/llama3.cpp (修改 init_mem, attention_mha)
kuiper/source/model/qwen2.cpp  (同上)
kuiper/source/model/qwen3.cpp  (同上)
```

**验收标准**：
- [x] 通过 `use_paged_attention_=true` 开关切换到分页模式
- [x] 生成结果与非分页模式完全一致
- [x] 内存使用量随实际 token 数增长而非 seq_len

### Phase 4: 优化与完善

**可选优化**：
- Block Pool 动态扩容（当前为固定大小）
- Prefill 阶段批量分配 Block（减少分配调用次数）
- Block Table 常驻 GPU（避免每步 H2D 拷贝）
- 更细粒度的 CUDA Kernel 优化（如 Block 内 float4 对齐读取）

---

## 11. 测试策略

### 11.1 单元测试

```cpp
// test/test_op/test_block_manager.cpp

TEST(BlockAllocatorTest, AllocateAndFree) {
  // 测试基本分配和释放
  BlockAllocator alloc(DeviceType::kDeviceCPU, 16, 10, 4, 128);
  auto id1 = alloc.allocate();
  EXPECT_GE(id1, 0);
  EXPECT_EQ(alloc.num_free_blocks(), 9);
  alloc.free(id1);
  EXPECT_EQ(alloc.num_free_blocks(), 10);
}

TEST(BlockAllocatorTest, ExhaustBlocks) {
  // 测试分配耗尽
  BlockAllocator alloc(DeviceType::kDeviceCPU, 16, 2, 4, 128);
  alloc.allocate();
  alloc.allocate();
  EXPECT_EQ(alloc.allocate(), -1);  // 无空闲块
}

TEST(BlockManagerTest, SlotAllocation) {
  // 测试 slot 分配正确性
  BlockManager mgr(DeviceType::kDeviceCPU, 16, 100, 4, 128, 1024);
  for (int i = 0; i < 32; ++i) {
    EXPECT_TRUE(mgr.allocate_slot(i));
  }
  // 32 tokens = 2 logical blocks
  EXPECT_EQ(mgr.num_free_blocks(), 98);
}

TEST(BlockManagerTest, KVPointerConsistency) {
  // 验证同一 token 的 K/V 指针在多次获取时一致
  BlockManager mgr(DeviceType::kDeviceCPU, 16, 100, 4, 128, 1024);
  mgr.allocate_slot(0);
  auto [k1, v1] = mgr.get_kv_slot_ptr(0, 0);
  auto [k2, v2] = mgr.get_kv_slot_ptr(0, 0);
  EXPECT_EQ(k1, k2);
  EXPECT_EQ(v1, v2);
}
```

### 11.2 数值正确性测试

```cpp
// test/test_op/test_paged_mha.cpp

TEST(PagedMHATest, MatchOriginalMHA) {
  // 1. 构造随机 QKV 数据
  // 2. 分别用原始 MHA Kernel 和 Paged MHA Kernel 计算
  // 3. 比较输出，要求 max absolute error < 1e-5
  int head_num = 8, head_size = 128, kv_dim = 512, kv_mul = 1;
  int seq_len = 256, block_size = 16;

  // ... 构造数据 ...

  // 原始 MHA
  kernel::mha_kernel(pos, head_num, layer_index, seq_len, kv_dim, ...);

  // Paged MHA
  kernel::paged_mha_kernel_cpu(pos, head_num, kv_dim, ...);

  // 比较
  for (int i = 0; i < head_num * head_size; ++i) {
    EXPECT_NEAR(original_output[i], paged_output[i], 1e-5f);
  }
}
```

### 11.3 端到端测试

```
# 对比两种模式的生成结果
./llama_infer --model xxx.bin --mode original  --prompt "Hello"
./llama_infer --model xxx.bin --mode paged     --prompt "Hello"
# 两者输出应完全一致
```

---

## 附录：关键文件变更清单

### 新增文件

| 文件路径 | 说明 |
|---------|------|
| `kuiper/include/base/block_allocator.h` | BlockAllocator 声明 |
| `kuiper/include/base/block_table.h` | BlockTable 声明 |
| `kuiper/include/base/block_manager.h` | BlockManager 声明 |
| `kuiper/source/base/block_allocator.cpp` | BlockAllocator 实现 |
| `kuiper/source/base/block_table.cpp` | BlockTable 实现 |
| `kuiper/source/base/block_manager.cpp` | BlockManager 实现 |
| `kuiper/include/op/paged_mha.h` | PagedMultiHeadAttention 层声明 |
| `kuiper/source/op/paged_mha.cpp` | PagedMultiHeadAttention 层实现 |
| `kuiper/source/op/kernels/cpu/paged_mha_kernel.h` | CPU Paged MHA Kernel 声明 |
| `kuiper/source/op/kernels/cpu/paged_mha_kernel.cpp` | CPU Paged MHA Kernel 实现 |
| `kuiper/source/op/kernels/cuda/paged_mha_kernel.cuh` | CUDA Paged MHA Kernel 声明 |
| `kuiper/source/op/kernels/cuda/paged_mha_kernel.cu` | CUDA Paged MHA Kernel 实现 |
| `test/test_op/test_block_manager.cpp` | Block 管理单元测试 |
| `test/test_op/test_paged_mha.cpp` | Paged MHA 正确性测试 |

### 修改文件

| 文件路径 | 修改内容 |
|---------|----------|
| `kuiper/include/model/model.h` | 新增 `block_manager_`、`use_paged_attention_`、`block_size_` 成员 |
| `kuiper/source/model/model.cpp` | `slice_kv_cache()` 新增分页路径分支 |
| `kuiper/source/op/kernels/kernels_interface.h` | 新增 `PagedMHAKernel` typedef 和 `get_paged_mha_kernel()` |
| `kuiper/source/op/kernels/kernels_interfaces.cpp` | 新增 `get_paged_mha_kernel()` 实现 |
| `kuiper/source/model/llama3.cpp` | `init_mem()` 新增分页路径；`attention_mha()` 适配 |
| `kuiper/source/model/qwen2.cpp` | 同上 |
| `kuiper/source/model/qwen3.cpp` | 同上 |
| `CMakeLists.txt` | 新增源文件到构建目标 |
| `test/CMakeLists.txt` | 新增测试文件 |

---

## 附录：Block 内存寻址速查

给定参数：
- `block_size` = B（每块 token 数）
- `layer_num` = L
- `kv_dim` = D
- `token_pos` = t
- `layer_idx` = l

```
逻辑块号:    logical  = t / B
块内偏移:    slot     = t % B
物理块号:    physical = block_table[logical]
块字节数:    blk_sz   = L * B * D * sizeof(float)

K/V 地址 = pool_base
         + physical * (L * B * D)      // 块基址
         + l * (B * D)                  // 层内偏移
         + slot * D                     // slot 内偏移

(单位: float 个数, 实际字节需 × sizeof(float))
```
