# 配置、执行与 Codex 决策

当前 Codex 提出计划并监督选择；skill 内置运行库负责提示构造、schema 校验、critic、候选筛选、调度、直接启动后端和归档。mailbox 只是远程文件通道，不是另一个 T-REX 服务，也不调用其他模型。

## 初始化与检查

以下所有代码块都是**远程服务器 Bash 命令**，由本地 Codex 通过 SSH 执行，或写入服务器上的 Slurm 脚本。不要在 Mac／Windows 本地直接运行 Python。Bash 变量都指远程实际路径和已确认输入。`SKILL_PYTHON` 是安装了本技能 requirements 的 Python；`SKILL_DIR` 为完整技能目录。

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" init --config "$CONFIG" \
  --target "$TARGET_LABEL" --target-constraint "$CONSTRAINT_JSON" \
  --target-pdb "$TARGET_PDB" --worker-gpus "$WORKER_GPU_IDS" \
  --hours "$CAMPAIGN_HOURS" --output "$ARCHIVE"
```

`init` 不覆盖文件。自定义靶标提供实际 PDB 和约束 JSON；约束数据结构见 `scripts/binder_runtime/schemas.py` 的 `TargetConstraint`，不要把 toy/synthetic 测试输入当作研究数据。已确认的注册靶标可使用 `--asset-root`，资源解析使用包内 registry 与提供的资产。

随后编辑服务器上 YAML 的 `backends`，填实际后端仓库、环境、权重与聚类工具路径。只启用已确认的行动类型；`--families` 接受逗号分隔列表。`init` 的配置不代表这些后端已安装，不能跳过检查。

启用 Complexa 且使用后端未注册的自定义靶标时，每个生成动作的 `config_delta` 都必须显式提供 `target_length_min` 和 `target_length_max`，由当前 Codex 根据任务选择 binder 长度范围。缺少范围的动作会被跳过，不会代填。skill 会将实际 PDB、选定链、残基编号和热点传给后端；插入码等无法准确表达的输入会在检查时拒绝。使用后端已注册名称时，实际输入必须与注册的裁剪范围一致，否则改用准确描述该输入的自定义名称或提供明确裁剪的结构。

`--worker-gpus 0` 表示一个 worker，`0,1` 表示两个，也可使用受支持的 GPU UUID。没有为本地 LLM 预留 GPU。`--agent-model` 仅记录已知 Codex 模型名，不选择模型；未知省略。保持 `policy.evidence_skip: false`，防止新的决策轮次跳过当前 Codex。

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" check --config "$CONFIG"
```

检查包含实际输入、资产、后端版本、包哈希和归档恢复资格。失败时解决报告中的具体问题。成功仍不证明 GPU 推理已经通过。

## 启动与响应

通过 SSH 在服务器已有授权 allocation 内启动，并立即服务决策。用工具的短 yield 保存后台进程句柄，或保存 Slurm job ID，不要阻塞等启动命令完成才读取 mailbox。

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" start --config "$CONFIG" \
  --mailbox "$MAILBOX" --decision-timeout 900
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" requests --mailbox "$MAILBOX"
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" show --mailbox "$MAILBOX" --request-id "$REQUEST_ID"
```

以上表示三个独立工具调用，第一条保留后台会话。`run` 是 `start` 的别名。默认 `requests` 列出等待项及损坏请求提示；`--all` 查看历史。

完整读取请求的 `system`、`messages`、合法角色/字段、证据与截止时间。根据当前证据由 Codex 编写 JSON 对象；只引用本轮实际 evidence/candidate ID，并遵守资源与动作限制。supervisor 可能要求纠错，继续读取新请求而不是重发旧 JSON。

若完整 `show` 输出超过工具显示上限，使用可选分页；默认不加参数时仍返回完整请求：

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" show --mailbox "$MAILBOX" \
  --request-id "$REQUEST_ID" --offset 0 --limit 6000
```

`--offset` 或 `--limit` 任一参数会启用分页，缺省偏移为 0、页长为 6000，上限 8000。偏移、页长和 `total` 均按 Unicode 码点计数，不是字节或 token。按返回的 `next_offset` 继续调用，直到它为 `null`；每页都须读全，若工具仍截断则缩小 `--limit`，从该页偏移重新读取。跨页核对 `request_id`、`session_id`、`role`、`prompt_hash`、`payload_hash` 与 `total` 一致，将 JSON 解码后的 `text` 依序拼接，长度应为 `total`，其 UTF-8 SHA-256 应为 `payload_hash`。

分页文本是稳定排序的完整请求 JSON，仅排除会变动的 `status`、`status_updated_at`、`reason`；当前 `status` 和固定 deadline 另列在每页顶层，每次读取仍验证权威请求及状态绑定。当前 Codex 必须读完全部页中的 `system`、`messages` 和证据后才决策，不能仅保存文件或根据截断片段补造判断。分页不延长请求中的固定 deadline；默认决策窗口为 900 秒，剩余 campaign 时间更短时以实际 deadline 为准。若未读全、跨页不一致或请求已不再等待，不提交决策，说明原因并按停止与恢复流程处理。

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" respond --mailbox "$MAILBOX" \
  --request-id "$REQUEST_ID" --response "$DECISION_JSON"
```

决策 JSON 的持久副本保存在服务器。优先采用 [可靠决策传输](decision-transport.md) 的流程：保存唯一 UTF-8 JSON、SCP 上传、非交互 SSH 调用 `respond --response <远程文件>`，检查退出码和返回状态。macOS 与 Windows PowerShell 模板均无需本地 Python。`--response -` 仍可用于具有明确 EOF 的非 TTY 管道；不要通过交互终端或 ConPTY 输入长 JSON，也不要把终端回显当作提交成功。JSON 作为数据传输，不能拼接为 shell 代码。最大响应为 1 MiB。脚本生成信封并绑定 UUID、session、role、prompt hash 与 deadline，拒绝晚到/重复/取消后的提交。

`accepted` 仅表示传输接受，`consumed` 表示执行器读到，之后还需内容校验和筛选。观察 dispatch/结果，不将回复状态当作 GPU 任务完成。

BindCraft 冷启动会将当前 planner 的合法配置应用到对应 seed 候选，随后仍经过校验和 supervisor 选择。核对候选 `config_delta`、实际 dispatch 和生成的 settings 后再说明试跑规模。

短预算使用以下互相独立的参数：

| 参数 | 含义 |
| --- | --- |
| `max_runtime_seconds` | 300–7200 的整数；从子进程启动计时，由独立 watchdog 执行单批墙钟上限，终止收尾另预留 20 秒。 |
| `max_relaxed_trajectories` | 1–64 的整数；映射上游 advanced settings 的 `max_trajectories`，达到已松弛轨迹数量后自然结束。早停尝试不计入，所以仍需运行时限兜底。 |
| `max_trajectories` | 兼容旧接口，映射上游 `number_of_final_designs`，表示通过筛选的设计配额；不是尝试次数。 |

例如首批可提出 `max_runtime_seconds: 2100`、`max_relaxed_trajectories: 1`，再按靶标和预算选择长度、MPNN 数量等科学参数。这是资源配置示例，不是可直接重放的 planner 答案；不能保证 35 分钟内产出完整结构。

预算检查为有时限的批次预留“单批时限 + 20 秒收尾 + 900 秒 canonical AF2”，无时限仍保留保守估算。到限先停止该批自己的进程组，再解析完整产物，归档中区分预算结束、自然完成和后端失败。当前 Codex 必须读取新的、可比较的复评证据并决定后续动作；验收要核对该动作实际执行、产物回到后续决策，不能以 JSON 被接受代替。后续动作可以是重新生成、序列重设计或改变评估设置后的再复评。

Complexa 内部 AF2 奖励仅用于搜索及诊断，最终结构仍通过独立 AF2 获得统一评分。当前固定后端的内部奖励默认移除 binder 模板序列，并使用 bfloat16；独立复评保留模板序列并关闭 bfloat16。指定相同模型、recycles 和 initial guess 不能消除这些差异。原生分数保存在 `complexa_native_*` 字段，不直接用于严格通过判定或 SU 计数，也不与独立复评分数混排。只有完成独立复评的结构才有对应统一评分，仍待复评的数量须单列。

## 停止、时限与恢复

### 停止生成，继续复评已有候选

用户可以说：“不再生成新候选，把已有候选的独立 AF2 复评做完，然后导出结果。”对活动任务执行：

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" drain --archive "$ARCHIVE"
```

如果任务已停止，在已分配的 GPU 内使用原配置与兼容 bundle：

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" start --config "$CONFIG" \
  --mailbox "$MAILBOX" --evaluation-only
```

此模式冻结已有原生产物和切换时正在运行的生成任务范围，允许这些运行中的任务产生的完整结果继续进入复评。它不再启动生成、序列重设计或 advisory 复评，也不调用 planner/supervisor；等待中的科学请求会被取消，已提交回复保留作审计，不冒充新的决策。系统 canonical AF2 使用原统一评估协议。

复评排队不受生成探索的路线份额限制，仍检查父产物、文件、协议、失败记录和实际剩余预算。每次启动需要满足 AF2 运行与清理预留，不延长原 campaign 预算。缺文件、已失败或预算不足的项目逐项报告，不能承诺所有产物一定评完。`status` 读取独立的 `evaluation_only_state.json` 收尾状态；有未完成项时返回 `76`，显式停止仍返回 `75`，正常排空返回 `0`。这些退出码均不表示实验结合成功。

模式保存在归档中，使用普通 `start` 恢复也继续复评模式；本版不支持在该归档重新开启生成，新增设计需新建 campaign。`stop` 可随时停止此模式的一切新增派发。请求提交后继续核对 worker、控制器与 Slurm 最终状态。

### 停止全部新增派发

停止整个正在运行的 campaign，使用归档路径：

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" stop --archive "$ARCHIVE"
```

停止请求绑定当前 session，控制器在安全检查点停止新增调度并收尾已有 worker，退出 `75`。同步后端准备期间可能需要等待；收尾对每个运行中的 worker 最多等待 10 分钟，再尝试终止。它不直接取消 Slurm allocation；需要立即终止 GPU 作业时，使用已授权的调度器取消操作。重启后的新 session 不消费旧停止请求。

只取消某个正在等待的决策请求时，也可以使用：

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" cancel --mailbox "$MAILBOX" --request-id "$REQUEST_ID"
```

取消、超时、无效最终决策、abstain 或空决策停止新增任务并进入 worker 收尾，返回 `75`。取消请求不等于 GPU 进程已停止；需要立即结束 allocation 时使用已授权的调度器取消操作。

单次等待默认 900 秒，允许设置 `(0, 3600]` 秒，且不能超过剩余 campaign 时长；读取请求不重置倒计时。等待期间，同一控制器线程持续监督已有 worker，处理完成产物、超时和进度保存，不启动新的后端。当前请求使用发布时的证据快照，哈希不变；等待中新增的结果会进入后续科学决策。停止信号在等待和监督前后检查；一次输出解析或进程终止仍可能占用时间，不能保证即时返回。

批次 watchdog 独立计时，等待决策不会延长单批预算。最后每个 worker 可等待最多 600 秒收尾，同时遵守已设置的批次时限；调度器负责整个 allocation 的硬截止。此版本仍需要 Codex 会话持续响应；更长的决策窗口用于容纳完整读取和会话处理延迟，不代表无人值守决策。

同一 archive 和 mailbox 只能有一个活动执行器。恢复前确认旧控制器及 worker 均已结束，累计原 campaign 时间。使用同一配置重启会生成新 session；不消费旧回复，但可继续之前已验证的待调度选择。部分中止 tick 已写入的证据不删除，也不能从头重播。

本版只恢复本独立 bundle 创建、代码哈希与输入哈希匹配的归档。旧桥接版或原 T-REX archive 可以只读分析，但不能伪造 provenance 绕过恢复检查。修改输入或代码后应新建 archive，或另行实现有审核记录的迁移。

在服务器保留配置、日志、mailbox、`codex_provenance_*.json`、`codex_outcome_*.json` 及全部 archive 产物。它们记录包来源/哈希、输入、预算、实际决策和退出原因。不要把未知模型、token 或未运行的后端记录成已验证结果。
