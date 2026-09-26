# vLLM iteration 级别 SD 切换改动说明

本文记录 `/home/user/vllm` 中为了实现 iteration 级别 speculative decoding 切换所做的改动，以及每项改动的含义。

## 目标语义

这里的 SD 指 speculative decoding。

目标不是 request 级别开关，也不是立即中断当前执行中的 batch，而是 iteration 边界生效：

- 一个 scheduler iteration 内，整个 batch 使用同一种模式：SD 或 Non-SD。
- 如果某个 iteration 已经开始执行，再收到 SD 开关请求，本轮 iteration 不被打断。
- 新的开关状态从下一轮 `schedule()` 开始生效。
- SD 到 Non-SD、Non-SD 到 SD 两个方向都遵循同一规则。

核心实现方式是：

1. 增加运行时 HTTP 控制入口。
2. 在 scheduler 中维护全局运行时开关 `enable_sd`。
3. 每次 `schedule()` 开始时，把 `enable_sd` 快照成该轮的 `spec_decode_enabled`。
4. 把这个快照写入 `SchedulerOutput`，后续 engine、worker、cudagraph 和 draft-token 逻辑都只看本轮快照。
5. toggle 只修改下一轮会读取到的 `scheduler.enable_sd`，不修改已经生成的 `SchedulerOutput`。

## 当前改动清单

已跟踪源码改动：

- `vllm/entrypoints/openai/completion/api_router.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/output.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/sched/async_scheduler.py`
- `vllm/forward_context.py`
- `vllm/v1/cudagraph_dispatcher.py`
- `vllm/v1/worker/gpu_model_runner.py`

新增但未跟踪的辅助文件和目录：

- `ITERATION_LEVEL_SD_SWITCH_SUMMARY.md`：本文档。
- `scripts/toggle_speed-test.sh`：同一个 SD 引擎内，对 Non-SD 与 SD 两种状态做多轮 batch 性能测试。
- `scripts/verify_toggle_batch_consistency.sh`：验证 batch 执行中途 toggle 不改变当前 in-flight batch 的输出。
- `scripts/verify_batch.sh`：对比纯 Non-SD 引擎与 SD 引擎切到 Non-SD 后的输出一致性。
- `scripts/speed-test.sh`：默认请求 SD 引擎端口的单模式性能测试脚本。
- `scripts/raw_speed-test.sh`：默认请求 Non-SD 引擎端口的单模式性能测试脚本。
- `logs/`：上述验证与性能测试产生的实验结果目录。
- `vllm/v1/worker/gpu_model_runner time.py`：旧调试/计时副本，不属于正式运行路径。
- `vllm/v1/worker/gpu_model_runner(save).py`：旧方案备份副本，不属于正式运行路径。

## 控制链路

运行时开关请求的路径如下：

1. 调用 `POST /v1/engine/speculative_decoding`，请求体为 `{"enabled": true}` 或 `{"enabled": false}`。
2. API router 调用 `engine_client.toggle_speculative_decoding(enabled)`。
3. `AsyncLLM` 通过 utility call 转发到 `EngineCore.toggle_speculative_decoding()`。
4. `EngineCore` 检查引擎是否配置了 `speculative_config`，然后更新 `scheduler.enable_sd`。
5. 下一次 `Scheduler.schedule()` 读取 `scheduler.enable_sd` 并生成本轮 `spec_decode_enabled` 快照。
6. `SchedulerOutput.spec_decode_enabled` 传给 worker。
7. worker 按该快照决定是否构建 speculative metadata、是否运行 drafter、是否保留 aux hidden states、是否使用 SD query length 的 cudagraph。

因此，HTTP toggle 修改的是未来 iteration 的输入状态；已经创建出来的 `SchedulerOutput` 不会被回写修改。

## 源码改动说明

### `vllm/entrypoints/openai/completion/api_router.py`

新增 `ToggleSpeculativeRequest`：

- 字段 `enabled: bool` 表示目标 SD 状态。

新增接口：

- 路径：`POST /v1/engine/speculative_decoding`
- 成功返回：
  - `status`: `success` 或 `ignored`
  - `speculative_decoding_enabled`: 请求中的目标状态

含义：

- 这是运行时控制面的入口。
- 该层只负责接收外部开关请求并转发给 engine client，不参与 scheduler 决策。
- 如果底层引擎没有配置 speculative decoding，`EngineCore` 会返回 `False`，接口状态表现为 `ignored`。

### `vllm/v1/engine/async_llm.py`

新增：

- `async def toggle_speculative_decoding(self, enable: bool) -> bool`

该方法通过：

- `self.engine_core.call_utility_async("toggle_speculative_decoding", enable)`

把 API 层请求转成 EngineCore utility 调用。

含义：

- API server 持有的是异步 engine client，不直接持有 `EngineCore`。
- 这里补齐异步控制面转发层，使 HTTP toggle 能进入 engine core。

### `vllm/v1/engine/core.py`

主要改动：

- 移除单独的 `self.use_spec_decode` 镜像状态，避免 engine core 与 scheduler 各自维护一份 SD 开关。
- 新增 `self._last_scheduler_output: SchedulerOutput | None`。
- 在 `step()` 和 `step_with_batch_queue()` 中保存最近一次实际执行的 `scheduler_output`。
- draft token 回收逻辑不再只看全局 speculative 配置，而是检查本轮 `scheduler_output.spec_decode_enabled`。
- `step_with_batch_queue()` 中 deferred scheduler output 的 draft-token 处理改为看 `scheduled_spec_decode_tokens` 是否存在。
- 新增 `toggle_speculative_decoding(self, enable: bool) -> bool`：
  - 没有 `speculative_config` 时记录 warning 并返回 `False`。
  - 目标状态与当前 `scheduler.enable_sd` 一致时直接返回 `True`。
  - 否则只更新 `self.scheduler.enable_sd`，并记录日志。

含义：

- EngineCore 只作为 toggle 的落点，不直接决定某个 iteration 是否启用 SD。
- `_last_scheduler_output` 让 engine 在模型执行后的 draft-token 处理阶段仍然使用“本轮实际执行时的快照”，避免 toggle 在执行后半段改变判断条件。
- 状态源被收敛到 `scheduler.enable_sd`，降低跨模块状态不一致的风险。

### `vllm/v1/core/sched/output.py`

在 `SchedulerOutput` 中新增：

- `spec_decode_enabled: bool`

并在 `SchedulerOutput.empty()` 中默认设为 `False`。

含义：

- 这是 iteration 级切换的关键数据通道。
- scheduler 在每轮开始时决定本轮模式，然后通过该字段把快照传给 engine/worker。
- worker 不能再只凭 `self.speculative_config is not None` 判断本轮是否走 SD，因为同一个 SD-capable 引擎可能临时运行在 Non-SD 模式。

### `vllm/v1/core/sched/scheduler.py`

新增运行时状态：

- `self.enable_sd: bool = True if speculative_config else False`

在 `schedule()` 开头生成本轮快照：

- `spec_decode_enabled = self.enable_sd`

围绕该快照调整调度逻辑：

- Non-SD iteration 中清空遗留的 `request.spec_token_ids`。
- 计算 `num_new_tokens` 时：
  - SD 使用 `request.num_tokens_with_spec`
  - Non-SD 使用 `request.num_tokens`
- EAGLE 相关的 `shift_computed_tokens` 只在 `self.use_eagle and spec_decode_enabled` 时启用。
- KV cache 分配的 lookahead tokens 只在 SD iteration 中使用；Non-SD iteration 使用 0。
- `scheduled_spec_decode_tokens` 只在 `spec_decode_enabled=True` 时填充。
- Waiting request 的 effective lookahead 也在 Non-SD iteration 中置 0。
- 构造 `SchedulerOutput` 时写入 `spec_decode_enabled`。
- `update_draft_token_ids()` 在 `self.enable_sd=False` 时直接返回。
- `update_draft_token_ids_in_output()` 在 `scheduler_output.spec_decode_enabled=False` 时直接返回。
- update 输出时，只有本轮 SD 快照为真才读取 `scheduled_spec_decode_tokens`。
- KV connector 的 `finished_recving` / `finished_sending` 从强断言 `req_id in self.requests` 改为缺失时跳过。

含义：

- 这个文件是实现 iteration 边界语义的核心。
- `enable_sd` 是未来 iteration 的开关，`spec_decode_enabled` 是当前 iteration 的不可变快照。
- Non-SD iteration 必须同时关闭 speculative token 计数、lookahead block、EAGLE shift 和 draft-token 传播，否则会把上一轮或配置层面的 SD 状态泄漏到当前轮。
- KV connector 的防御性修改用于避免异步回调晚于 request 生命周期时触发崩溃；它不是 SD 切换算法核心，但提升运行时切换场景下的容错性。

### `vllm/v1/core/sched/async_scheduler.py`

改动：

- 异步调度补 speculative placeholders 时检查 `scheduler_output.spec_decode_enabled`。
- 本轮是 SD 时继续写 `_spec_token_placeholders`。
- 本轮是 Non-SD 时把 `request.spec_token_ids` 置空。

含义：

- async scheduler 路径中 placeholder 是 draft/spec token 的跨步协作机制。
- 如果本轮已经切到 Non-SD，还继续保留 placeholder，会让 worker 或后续 scheduler 误判仍有 speculative token 需要处理。
- 因此 placeholder 也必须跟随本轮 iteration 快照。

### `vllm/forward_context.py`

在 `BatchDescriptor` 中新增：

- `uniform_decode_query_len: int = 1`
- `aux_hidden_state_outputs: bool = False`

含义：

- `BatchDescriptor` 是 cudagraph key 的组成部分。
- SD decode 与 Non-SD decode 都可能是 uniform decode，但每个 request 的 query length 不同：
  - Non-SD decode 是 `query_len=1`
  - SD decode 通常是 `query_len=1 + num_spec_tokens`
- 如果 cudagraph key 不区分 query length，SD 引擎切到 Non-SD 后可能仍使用按 SD query length 捕获的图和 padding。
- EAGLE3 / extract hidden states 场景下，目标模型 forward 是否返回 aux hidden states 会改变输出结构，因此也必须进入 cudagraph key。

### `vllm/v1/cudagraph_dispatcher.py`

主要改动：

- 抽出 `_compute_bs_to_padded_graph_size_for_capture_sizes()`，允许对不同 capture size 集合分别生成 padding 表。
- 新增 `_get_non_spec_decode_capture_sizes()`，用于 SD 引擎中额外生成普通 Non-SD decode 的 capture sizes。
- `_create_padded_batch_descriptor()` 增加参数：
  - `uniform_decode_query_len`
  - `exact_num_tokens`
  - `aux_hidden_state_outputs`
  - `use_non_spec_decode_padding`
- `initialize_cudagraph_keys()` 中增加 aux-hidden-state 维度：
  - piecewise graph
  - SD decode graph
  - SD 引擎切到 Non-SD 时使用的 query_len=1 decode graph
- 对带 speculative_config 且 `uniform_decode_query_len > 1` 的引擎，额外注册 Non-SD decode cudagraph key。
- `dispatch()` 增加参数：
  - `uniform_decode_query_len`
  - `exact_uniform_decode`
  - `aux_hidden_state_outputs`
- 当 SD-capable 引擎运行 query_len=1 的 uniform decode 时，使用 Non-SD capture size padding 表。
- cudagraph capture 顺序中，优先捕获 aux-hidden-state / SD 图，再捕获普通 query_len=1 Non-SD 图。

含义：

- 原始 SD 引擎的 cudagraph capture sizes 会按 speculative query length 对齐，例如按 6 token 粒度 padding。
- 运行时切到 Non-SD 后，如果仍使用这套 padding，batch size 或 token 数可能被错误 pad 到 SD 图形态，导致输出结构或采样位置不一致。
- 现在 SD 引擎启动时同时捕获两类 decode 图：
  - SD decode 图：`query_len=1 + num_spec_tokens`
  - Non-SD decode 图：`query_len=1`
- 这样 runtime toggle 到 Non-SD 后仍可以使用 CUDA graph，而不需要退回 eager。
- 对 EAGLE3，aux hidden state 输出结构也被纳入 key，避免 torch.compile/AOT 或 cudagraph 复用到不匹配的输出结构。

### `vllm/v1/worker/gpu_model_runner.py`

主要改动：

- 新增 `import os`，用于可选采样调试开关 `VLLM_TOGGLE_DEBUG_SAMPLE`。
- 新增 aux hidden state 状态：
  - `_aux_hidden_state_layers`
  - `_aux_hidden_state_outputs_enabled`
  - `_set_aux_hidden_state_outputs_enabled()`
- hybrid model 的 accepted-token 状态更新增加 `scheduler_output.spec_decode_enabled` 保护。
- `use_spec_decode` 改为同时检查：
  - `scheduler_output.spec_decode_enabled`
  - `scheduled_spec_decode_tokens` 非空
- speculative common attention metadata 只在本轮 SD 时构造。
- `_get_cudagraph_intermediates()` 增加 query length、exact uniform decode 和 aux hidden state 参数，并传给 dispatcher。
- `execute_model()` 中根据本轮快照设置：
  - `use_spec_decode`
  - `use_aux_hidden_state_outputs`
  - `uniform_decode_query_len`
- Non-SD iteration 使用 `uniform_decode_query_len=1`。
- SD iteration 使用 `self.uniform_decode_query_len`；EAGLE3 首 token 或无 speculative metadata 的特殊情况使用 query_len=1。
- 模型 forward 前调用 `_set_aux_hidden_state_outputs_enabled()`，让目标模型输出结构与当前 cudagraph key 对齐。
- `clear_kv_metadata` 改为 Non-SD iteration 立即清理，SD iteration 延迟到 drafter 后清理。
- postprocess 中如果模型返回 tuple，则按需取 aux hidden states；Non-SD iteration 忽略 aux tensor。
- draft-token propose 逻辑只在 `scheduler_output.spec_decode_enabled=True` 时运行。
- 如果 EAGLE3 当前 SD iteration 缺少 aux hidden states，则记录 warning，并为本轮回退为 zero draft tokens。
- CUDA graph dummy run / capture 路径增加：
  - `uniform_decode_query_len`
  - `aux_hidden_state_outputs`
  - `use_spec_decode_for_dummy`
- cudagraph capture 遍历 `BatchDescriptor` 时读取新增的 query length 与 aux-hidden-state 字段。

含义：

- worker 是实际执行模型、采样和 drafter 的地方，必须严格按本轮 `SchedulerOutput` 快照执行。
- 不能再用“引擎是否配置 speculative_config”代表“本轮是否启用 SD”。
- Non-SD iteration 中仍运行在 SD-capable 引擎上，因此 drafter、spec metadata、lookahead KV metadata、aux hidden states 都需要按本轮快照关闭或忽略。
- EAGLE3 对 aux hidden states 依赖很强，同时 torch.compile/AOT 对 forward 返回结构敏感，所以新增了稳定输出结构和 cudagraph key 区分逻辑。
- 该文件也移除了旧方案中 worker 局部 `enable_sd` 的思路，避免 scheduler 与 worker 维护两份开关状态。

## 辅助脚本和实验产物

### `scripts/toggle_speed-test.sh`

用途：

- 在同一个 SD-capable 服务上反复测试 Non-SD 与 SD 两种状态的 batch 性能。
- 每个 batch 前可通过 `POST /v1/engine/speculative_decoding` 设置状态。
- 记录 per-request JCT、per-run summary 和 aggregate summary。

关键变量：

- `BASE_URL`：默认 `http://172.16.33.142:8002`
- `BATCH_SIZE`：默认 64
- `NUM_RUNS`：默认 5
- `SET_SD_BEFORE_BATCH`：默认 1，控制每个 batch 前是否调用 toggle 接口
- `DATASET_PATH`：默认 MBPP parquet 数据集

含义：

- 用来比较同一个引擎在 SD 与 Non-SD runtime 状态下的吞吐和 JCT。
- 输出位于 `logs/sd-nonsd-jct-throughput-multirun-*` 或用户指定的 `OUT_DIR`。

### `scripts/verify_toggle_batch_consistency.sh`

用途：

- 验证 iteration 级切换不会改变已经在飞的 batch 输出。
- 运行四组：
  - 强制 Non-SD baseline
  - 强制 SD baseline
  - batch 启动后执行 SD -> Non-SD
  - batch 启动后执行 Non-SD -> SD
- 提取响应文本并逐个 request 做 `cmp` / `diff`。

关键变量：

- `TOGGLE_DELAY`：默认 0.5 秒，在 batch 发出后延迟 toggle。
- `USE_CACHE_SALT`：默认 1，用于避免 prefix cache 干扰一致性判断。

含义：

- 这是验证“toggle 只影响下一轮 iteration、不污染当前 batch”的主要脚本。

### `scripts/verify_batch.sh`

用途：

- 对比纯 Non-SD 引擎与 SD-capable 引擎切到 Non-SD 后的输出。
- 默认使用：
  - `NON_SD_BASE_URL=http://<HOST>:8001`
  - `SD_BASE_URL=http://<HOST>:8002`

注意：

- 当前脚本里的 `toggle_sd()` 函数中 curl 调用被注释掉。
- 因此它默认假设 SD 引擎状态已经由外部设置好，或者需要恢复该 curl 调用后再自动切换。

含义：

- 用来验证“SD 引擎运行在 Non-SD 模式时，行为应与纯 Non-SD 引擎一致”。

### `scripts/speed-test.sh`

用途：

- 单模式多轮 batch 性能测试。
- 默认请求 `http://172.16.33.142:8002`。
- 默认 label 为 `dsd-sd`。

含义：

- 用于测 SD-capable 服务某个固定状态下的性能。
- 不主动调用 toggle 接口，依赖服务当前状态。

### `scripts/raw_speed-test.sh`

用途：

- 与 `speed-test.sh` 逻辑基本相同。
- 默认请求 `http://172.16.33.142:8001`。
- 默认 label 为 `specsys-sd`。

含义：

- 用于测另一组服务或 baseline 服务的固定状态性能。
- 和 `speed-test.sh` 的主要差异是默认 `BASE_URL` 与 `LABEL`。

### `logs/`

当前存在：

- `logs/token-level_test/`
- `logs/speed_test/`

含义：

- 这些是验证脚本和性能脚本产生的实验输出，包括 prompt、payload、原始响应、diff、metrics 和 summary。
- 它们不是运行时代码改动，但记录了本次功能验证过程。

### `vllm/v1/worker/gpu_model_runner time.py`

含义：

- 旧调试副本。
- 包含过 GPU event 计时代码、profiler 相关注释和 worker 局部 `enable_sd` 旧方案痕迹。
- 不被 Python import 路径使用，不属于正式实现。

### `vllm/v1/worker/gpu_model_runner(save).py`

含义：

- 旧实现备份副本。
- 保留过 worker 局部 `enable_sd`、强制 eager、直接打印 cudagraph/input 等旧调试逻辑。
- 当前正式实现已经转向 scheduler iteration 快照与额外 Non-SD cudagraph capture，该文件不属于正式运行路径。

## 设计要点

### 为什么开关放在 scheduler

scheduler 是 iteration 的边界。把 `enable_sd` 放在 scheduler，并在 `schedule()` 开头快照，可以自然表达“下一轮生效”：

- toggle 修改 `scheduler.enable_sd`
- 当前已经产出的 `SchedulerOutput` 不变
- 下一轮 `schedule()` 才会读取新值

如果把开关放在 worker，worker 可能在当前 batch 执行过程中读到新状态，破坏 iteration 内一致性。

### 为什么需要 `SchedulerOutput.spec_decode_enabled`

同一个引擎可能同时满足：

- 配置了 `speculative_config`
- 当前 iteration 被切到 Non-SD

因此 `speculative_config is not None` 只能表示“引擎具备 SD 能力”，不能表示“本轮正在使用 SD”。

`SchedulerOutput.spec_decode_enabled` 才是本轮真实执行模式。

### 为什么需要额外 Non-SD cudagraph

SD decode 和 Non-SD decode 的 uniform query length 不同。只捕获 SD 图会导致 runtime 切到 Non-SD 后仍按 SD padding 和图结构执行。

现在在 SD-capable 引擎启动时额外注册 query_len=1 的 Non-SD decode 图，运行时切到 Non-SD 后仍可走 cudagraph。

### 为什么 aux hidden state 要进 cudagraph key

EAGLE3 / extract hidden states 场景下，目标模型 forward 返回结构可能是：

- 普通 hidden states
- `(hidden_states, aux_hidden_states)`

这会影响 torch.compile/AOT 和 cudagraph 捕获复用。把 `aux_hidden_state_outputs` 放入 `BatchDescriptor` 可以避免不同输出结构误用同一个图。

## 当前状态和注意事项

- 正式代码路径已经支持通过 HTTP 接口在 iteration 边界切换 SD。
- 当前接口返回体没有显式 `applies: next_iteration` 字段；语义由实现保证。
- `verify_batch.sh` 中的 toggle curl 目前是注释状态，使用前需要确认是外部手动设置状态，还是恢复脚本内自动 toggle。
- `gpu_model_runner time.py` 和 `gpu_model_runner(save).py` 是临时副本，建议不要纳入正式提交，除非有保留调试历史的明确需求。
- `logs/` 是实验结果目录，通常也不应进入正式源码提交。
