# 0. 总目标（Definition of Done）

在现有项目基础上实现 **Paged KV Cache + Page Table + Paged MHA（kernel 内 gather）**，满足：

1. **数值正确性**：与 baseline（连续 KV cache）逐 token logits 对齐

* `max_abs_diff(logits) <= 1e-5`（FP32）
* 覆盖跨页边界（B-1→B）与长上下文（>512）场景

2. **显存收益**（可写简历）：

* 在 `max_seq_len = 8192/16384` 的配置下，实际 context `L_used=2048/4096/8192` 时：

  * **KV cache allocated bytes** 随 `ceil(L_used/B)` 增长，而不是随 `max_seq_len` 固定分配
  * 至少给出：**节省 X GB / 降低 Y%** 或 “同显存预算上下文提升 Z×”

3. **性能（最终目标）**：paged kernel 内 gather 后

* 在 `L_used=8192` decode（batch=1）时 tok/s **不低于 baseline 的 0.95×**（或按你们期望阈值）
* 记录并报告：tok/s、ms/token（avg + P95）、peak allocated bytes

---

# 1) 范围与约束（简化但到最终目标）

* batch = 1（先把单序列闭环做到最强）
* dtype = FP32（与当前 KV cache 一致）
* 固定 block size `B`（建议 16 或 32；默认 16 更细粒度节省）
* **必须做到：按需分配 block**（否则显存收益出不来）
* 不做 eviction / free list（简易：只增长，不回收；够写简历）
* 需要修复/绕开 prefill >512 的 embedding kernel 限制（你文档提到 grid 固定 512 会丢 token；否则长上下文验收会“假失败”）

---

# 2) 代码改造点总览（agent 直接按此改）

你当前关键点（从文档抽象）：

* `Qwen3Model::init_mem`：创建 KV cache buffers（`kKeyCache/kValueCache`）
* `attention_qkv`：`slice_kv_cache(layer_idx, pos)` 得到 key/val 的视图，`wk/wv` 写进去
* `attention_mha`：MHA forward 读整块 cache + `set_pos(pos)` 控制有效长度

实现 paged attention 后：

## 2.1 新增 Buffer（替代原 Key/ValueCache）

在 `init_mem` 中新增：

* `KeyCachePaged`: shape `[n_layers, max_blocks_physical, B, kv_dim]`（max_blocks_physical 可动态增长，初始给一个小值比如 256 blocks）
* `ValueCachePaged`: 同上
* `PageTable`: shape `[n_layers, max_blocks_logical]`，int32，默认 -1

  * `max_blocks_logical = ceil(max_seq_len / B)`（这个可以一次性分配，成本小）
* `NextFreeBlock`: shape `[n_layers]` int32（每层一个增长指针；简易实现）
* 可选：`PageTableHostMirror`（为了 debug/统计方便，可把 page table 拷回 host）

> 注：physical blocks 的容量应支持 grow（例如达到上限时 reallocate 到 2× 并 memcpy 旧数据）

## 2.2 改造 `slice_kv_cache(layer_idx, pos)` 为分页写入

实现逻辑（伪代码）：

```
logical_block = pos / B
offset = pos % B
p = PageTable[layer, logical_block]
if p == -1:
    p = NextFreeBlock[layer]++
    ensure_capacity(KeyCachePaged, ValueCachePaged, p)
    PageTable[layer, logical_block] = p
return view(KeyCachePaged[layer, p, offset, :]),
       view(ValueCachePaged[layer, p, offset, :])
```

这样 `attention_qkv` 几乎不动：仍然 `wk/wv` 写到返回的 view。

## 2.3 新增 Paged MHA forward（kernel 内 gather）

新增一条分支，不破坏原 MHA（便于 A/B 对比）：

* `mha_layer->forward_paged(query, score_storage, key_cache_paged, val_cache_paged, page_table_layer, pos, mha_output)`

实现要求：

* attention 计算中，访问 token `t` 的 K/V 时通过页表定位：

  * `lb = t/B`
  * `p = page_table[lb]`
  * `off = t%B`
  * K 地址 = `KeyCachePaged[layer, p, off, :]`
  * V 地址同理

> 这就是“最终目标”。不要先做 gather-to-contig（那是中间态）。如果你担心一次到位风险：允许 agent 先用 gather-to-contig 打通 correctness（1 天内），再把 kernel 内 gather 做成最终版本并对比性能（第 2 天），但最终交付必须是 kernel 内 gather。

---

# 3) Kernel 设计（paged gather 版 MHA）

这里给 agent 一个“可直接落地”的 kernel 设计要点（不用追求极限性能，但要接近 baseline）。

## 3.1 输入输出

输入（device）：

* `Q`: `[n_heads, head_dim]`（或你现有 query 布局）
* `K_paged`: `[max_blocks_physical, B, n_kv_heads, head_dim]` 或等价 flatten
* `V_paged`: 同上
* `page_table`: `[max_blocks_logical]` int32
* `pos`：当前有效最后位置（含 pos 这一步，一般有效 token 数 = pos+1）
* 其他：rope/scale/causal mask 参数（按你现有实现）

输出：

* `O`: `[n_heads, head_dim]`

注意：你现有结构里 `attention_mha` 会传 `score_storage`，并且 `set_pos(pos)` 控制有效长度。paged forward 里仍保留 `pos` 作为唯一有效长度真值。

## 3.2 核心策略（通用、可实现）

* 对每个 head 做 attention：`softmax(QK^T) V`
* 关键是 K/V 不连续：每次读 K/V 需做 `page_table` 索引
* 推荐实现方式：

  * 每个 block 负责一个 head 的一部分 `t` 范围
  * 逐 `t` 取 K：`lb=t/B`, `p=page_table[lb]`，`off=t%B`
  * dot(Q,K_t) 做 reduce
  * 使用在线 softmax（log-sum-exp）避免存整个 score
  * 再第二遍遍历 t 计算加权 V（或 fuse 版本）

为了实现简单与可控，可以采取 **两遍**（更容易写对）：

**Pass1**：计算 `m = max(score)`，`s = sum(exp(score-m))`
**Pass2**：计算 `O = sum(exp(score-m)/s * V_t)`

这会多遍历一次 t，但正确性好、好验收。

## 3.3 性能优化（必须的最低限）

* 把 `page_table` 先拷到 shared memory（每个 kernel block 一份），减少全局访存
* 按 `t` 连续遍历时，`lb` 变化频率为每 B 次一次，尽量避免每 token 都读 page table

  * 例如外层遍历 logical_block，内层遍历 offset
* 对 K/V 的 layout 要保证 `head_dim` 连续，以便 vectorized load（float4/float2）
* 如果你们是 multi-query attention（n_kv_heads < n_heads），注意 head 到 kv_head 的映射

> Agent 如果不确定你现有 MHA kernel 的布局，优先复用原 kernel 里的 indexing 方式，只把 “K/V 取址” 换成 paged 取址。

---

# 4) 动态分配与 grow（让显存收益真实发生）

必须实现 physical blocks 动态增长，否则“paged”只是逻辑分页、显存仍是一次性打满。

## 4.1 初始容量

* `initial_blocks = 256`（对应 B=16 时能容纳 4096 tokens；足够覆盖很多 prompt）
* 当 `NextFreeBlock[layer] == capacity_blocks` 时：

  * `capacity_blocks *= 2`
  * 重新分配更大 `KeyCachePaged/ValueCachePaged` 并 memcpy old
  * 更新 buffer 指针

## 4.2 为什么每层独立 next_free_block

最省心：每层互不干扰，不需要跨层共享 block id。缺点是每层都要增长一次，但实现简单、稳定。

（如果要进一步节省，可以共享物理 block 池，但复杂度上升，不建议在“简易版到最终目标”阶段做。）

---

# 5) Correctness 验收（必须自动化）

新增一个 `--attn_impl={baseline,paged}` 运行开关，用同一套输入跑两遍。

## 5.1 基本对齐测试（逐 token）

测试集合：

1. 小 prompt（<B）
2. 跨页边界：长度 = `B-1, B, B+1, 2B-1, 2B, 2B+1`
3. 长 prompt：`1024, 2048, 4096, 8192`（至少到你 max_seq_len 的 1/2 或更多）
4. decode：固定生成 `gen_len=256` tokens

验收：

* 每一步 logits 对比：`max_abs_diff <= 1e-5`
* 或者更稳：最终生成序列完全一致（贪心 decode）+ logits diff 抽样检查

## 5.2 page table 自检

每次分配新 block 后，做 debug 检查：

* `PageTable[layer, lb] != -1`
* `NextFreeBlock[layer]` 单调递增
* 当 `pos` 遍历时，`lb` 的映射不变（不允许被覆盖）

可选：在 debug 模式下，把每个 pos 写入的 K 第一维写成 `pos`，再随机抽查读回验证（只在小规模测试打开）。

---

# 6) 性能与显存实验（可写简历的数字怎么跑）

## 6.1 统一测量方法（强制）

* warmup：20 step（不计时）
* measure：200 step
* batch=1
* 记录：

  * `tok/s`
  * `ms/token`（avg、P50、P95）
  * `peak allocated bytes`（你们 allocator/Buffer 统计优先；不行再用 cudaMemGetInfo 差值）
  * `kv_cache_bytes`（由你的 buffer 元信息计算）

输出写到 CSV：`attn_impl, B, L_prompt, L_gen, tok_s, ms_p50, ms_p95, peak_bytes, kv_bytes`

## 6.2 Benchmark 维度（最小但能出图）

* `max_seq_len`：8192（或 16384）
* `L_prompt`：512 / 2048 / 4096 / 8192
* `L_gen`：256
* 比较：

  * baseline（连续 KV）
  * paged（kernel gather）

## 6.3 你要在报告里给的核心结论（agent 必须生成）

1. **KV 显存节省**（强制输出）

* 在 max_seq_len=8192 时：

  * prompt=2048：kv_bytes baseline vs paged（节省 % 和 GB）
  * prompt=4096：同上
  * prompt=8192：paged 与 baseline 接近（分页的 padding 损失 <= (B-1)/B）

2. **吞吐与延迟**

* 给出在 L_prompt=8192 时：

  * paged tok/s / baseline tok/s
  * paged ms_p95 / baseline ms_p95

3. **上下文扩展（如果你能用显存预算推出来）**

* 固定 GPU 显存预算下（例如 peak_bytes 不超过某阈值），baseline 触发 OOM 的长度 vs paged 可运行的长度（至少用“估算 + 实测一个点”也行）

---

# 7) 简历可直接用的产出（agent 生成最终 bullet）

agent 最终需要在 `REPORT.md` 里生成一段可复制进简历的 bullet（中英文都可），格式类似：

* “实现 PagedAttention（分页 KV cache + page table + kernel 内 gather），将 KV 显存从 O(L_max) 降为 O(L_used)；在 max context=8192、实际 context=2048/4096 时 KV 显存分别节省 **X%/Y%（Z GB）**，并在 8k context 下 decode 吞吐达到 baseline 的 **W%**（batch=1，gen=256）；逐 token logits 对齐 **max|Δ|<1e-5**。”

---

# 8) 具体任务拆解（agent 的 Todo 清单）

## 8.1 实现任务（按顺序）

1. **修复 prefill >512 的 embedding kernel 问题**

* 使得输入长度 >512 时不会丢 token（否则长上下文对齐测试无意义）

2. **新增 buffers（init_mem）**

* KeyCachePaged / ValueCachePaged / PageTable / NextFreeBlock
* PageTable 初始化为 -1，NextFreeBlock 初始化为 0

3. **实现动态 grow**

* ensure_capacity(layer, needed_block_id)

4. **改造 slice_kv_cache 为分页写入**

* 返回指向 paged cache 的 view

5. **实现 paged MHA forward（kernel gather）**

* 添加 `forward_paged` 接口 + CUDA kernel
* 在 attention_mha 中通过 flag 切换

6. **测试与回归**

* correctness：logits 对齐 + 跨页边界 + 长上下文
* perf：基准脚本输出 CSV

## 8.2 交付物（必须产出）

* 新增/修改代码（按你 repo 结构）
* `bench.csv`
* `REPORT.md`（包含：实验设置、结果表、结论、简历 bullet）

---

# 9) 关键实现细节（避免 agent 走歪）

* **paged 版本的 page_table 必须是 per-layer 或者明确的共享策略**：这里要求 per-layer（最简单）
* **pos 的定义一致**：你现在 `set_pos(pos)`，通常有效长度 = pos+1；paged forward 必须严格一致
* **跨页边界**是最容易出错点：`pos=B-1` 到 `pos=B` 时，`logical_block` 发生变化，确保 page_table 分配与索引正确
* **grow 的 memcpy** 必须按 byte 复制，且要更新所有 view 指针（最好通过你们 buffer 管理统一更新）