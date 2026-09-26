# SGLang ASD 与 `SPECULATIVE_NUM_STEPS=0` 设计简述

## 1. ASD 的总体思路

ASD（Adaptive Speculative Decoding）根据当前 batch size 和近期草稿接受长度，在若干离散的 `num_steps` 档位间切换，以平衡“多产草稿”的收益和“草稿被拒”的浪费。

核心链路为：

```text
batch size 选档 → draft → target verify → draft extend
                                  ↓
                       accept_lens - 1
                                  ↓
                    EMA 策略更新下一轮档位
```

主要设计如下：

- **按 batch size 分槽**：配置以 batch size 为键，每个区间拥有独立的候选档位、EMA 和迟滞参数。实际 batch 会先向上对齐到 CUDA Graph batch size，再采用不大于它的最近配置槽。默认配置会随 batch 增大逐步缩短草稿，最高负载可选择 `0`。
- **保守的反馈控制**：使用接受草稿数的 EMA，并通过 `warmup_batches`、`update_interval`、上下行迟滞和可选 ceiling 抑制频繁抖动。接受结果在异步拷回 CPU 后更新，避免在 worker 热路径中同步 GPU。
- **离散档位而非任意动态形状**：启动时合并所有槽的候选步数，为每个步数预建完整的 `SpecRuntimeState`，其中包含 draft、target verify、draft extend 三阶段各自的 attention backend 和 CUDA Graph。
- **原子切换运行时状态**：切档时同时替换步数、draft token 数、三阶段 backend/graph、top-k chain buffer 和 `server_args`。切换本质上是引用替换，不在请求中途重新 capture graph。
- **控制图和显存开销**：每个步数只 capture 能路由到该档位的 CUDA Graph batch size；公共 eager buffer、KV/cache 预留和 tokenizer 输出槽按所有候选档位的最大 `steps + 1` 分配。
- **可观测与验证**：`/server_info` 暴露当前步数和平均接受长度，Prometheus 暴露当前 steps/draft tokens、接受长度和接受率；单元测试覆盖策略、batch 路由和 graph 剪枝，端到端测试覆盖升降档及 GSM8K 正确性。

关键实现：`adaptive_spec_params.py`（策略和 BS 路由）、`adaptive_runtime_state.py`（状态池与控制器）、`eagle_worker_v2.py`（状态构建/切换和执行）、`batch_result_processor.py`（CPU 侧反馈）。

## 2. `SPECULATIVE_NUM_STEPS=0` 的语义与配套设计

`0` 表示**本轮不生成草稿 token**，但并不等于启动一个完全没有 draft model 的普通 SGLang 服务。它仍处在 EAGLE/EAGLE3 worker 中，以便之后无重启切回正步数。

- **保持最小合法形状**：top-k=1 时始终维持 `num_draft_tokens = num_steps + 1`，因此零步仍有 1 个 root/bonus token；内存计算中的 `or 1` 也避免零宽 buffer。
- **跳过 draft，复用 verify**：worker 不调用 draft model 的多步生成，而是构造单节点 `EagleVerifyInput`。target verify 必然接受 root，再从 target logits 采样一个 bonus token，功能上等价一次普通 decode。
- **默认保持 draft 状态温热**：verify 后仍执行一次 draft extend，使 draft KV、hidden state 和下一轮 top-k 输入保持同步；batch 变小时可直接恢复正步数，无需重放 prompt。
- **可选省掉 draft extend**：`SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND=1` 会填充形状正确的占位 tensor，进一步减少零步成本；代价是 draft KV 变旧，重新升档后会经历冷恢复和短期低接受率。
- **零步专用资源路径**：`steps <= 1` 不创建多步 draft attention backend，也不 capture draft-decode graph；仍为单 token target verify 和 draft extend 准备合法 backend/graph。部分 attention backend 还对零步 idle/padding metadata 做了专门分支。
- **策略可进入也可离开零步**：候选值允许非负整数；低接受率可降到 `0`。处于 `0` 时没有新的接受率信息，因此策略周期性探测最小正档位；若该 BS 槽只有 `[0]`，则保持关闭草稿。

## 3. 边界与结论

当前 ASD 仅支持 EAGLE/EAGLE3 且 `topk=1`，并显式排除 DP attention、multi-layer EAGLE、two-batch-overlap 和 PDMux。零步模式的价值是让同一 EAGLE 实例按负载快速退化为“近似普通 decode”并可快速恢复；它仍保留 draft model、verify 管线以及默认的 draft-extend 开销，因此性能和显存占用不能视为纯 NONSD 基线。
