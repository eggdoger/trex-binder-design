# 结果检查与导出

先确定服务器配置里的 archive。以下 Bash 命令全部由本地 Codex 经 SSH 在服务器执行，使用服务器的 Python 环境和 `scripts/trex_skill.py`。分析、聚类、导出均在服务器；本地只读取需要展示的摘要。完整归档和产物默认不下载，因为部分产物引用绝对路径，部分 Complexa 文件位于 archive 之外。

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" status --archive "$ARCHIVE"
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" analyze validate --archive "$ARCHIVE"
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" analyze summary --archive "$ARCHIVE"
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" analyze trace --archive "$ARCHIVE" --limit 10
```

这些命令直接使用 bundled runtime，不调用 T-REX 可执行文件。`status` 不是 Slurm job 状态；`validate` 不能证明真实分子推理已完成。运行中 JSONL 会持续追加，记录观测时间。

解释结果时区分：

- 被选中、排队、worker 启动、完成是不同事件。用 dispatch 和结果记录核实执行。
- 一条记录不一定是一个独立候选；失败任务仍可能产生计算成本。
- `strict_count` 是计算筛选计数；结构独特性 `run_su_count` 只有真实聚类证据才有相应含义。缺失指标、未运行聚类不填零。
- `status.current_results` 和 `summary.current_results` 从当前归档全部靶标的结果记录重新统计 canonical AF2 评估、通过数及对应原生筛选状态（目标见 `target_ids`）；这里只读已保存记录，不重新读取 PDB 核验或开展推理。`summary.endpoint` 与 `status.latest_evidence` 是历史规划/聚类快照；停止收尾后可能较旧。比较 `current_results.latest_evidence_comparison`，不要把旧快照里的零通过或 `no_strict` 当作最终结果，也不要根据新增通过数推定新的结构簇数。`llm_usage.calls` 是记录计数，不是精确 API 调用数；未知 token 不等于零成本。
- `status.runtime_metrics` 与 `summary.budget.source` 标明当前持久化预算来源；较新的 controller checkpoint 包含停止收尾和短恢复会话，不计停止期间的空档。`latest_evidence` 仍是历史规划快照。GPU 小时是配置容量乘会话时长，不是利用率实测，也不等同 Slurm 账单；授权的总分配预算以调度器实际记录核对。v0.3.1 旧归档丢失的尾段不能靠升级自动补回。
- 结合 `codex_provenance_<session>.json`、`codex_outcome_<session>.json` 与 mailbox 检查输入、bundle、决策和退出原因。退出 `75` 表示协作停止或决策中止。
- 模型评分不是实验亲和力、特异性或成功率。保留样本数、缺失项和筛选定义。

## 统一评分的含义

当前 canonical AF2 的严格筛选条件为 `pLDDT >= 90`、归一化 `iPAE <= 7/31`、`binder_scRMSD < 1.5 Å`，三项必须同时满足。这里的 pLDDT 是 binder 的平均置信度（0–100），iPAE 是 binder–target 双向平均 PAE 除以 31 后的数值。不要把后端原生评分或改变评估设置后的 `refold_*` 诊断值混入这一计数；ipTM 只作诊断。

这里的 `binder_scRMSD` 采用 ColabDesign `losses.rmsd`：先将预测结构的靶标 Cα 对齐输入靶标，再用同一变换计算 binder Cα 的偏差。因此它同时反映 binder 相对靶标的位置与自身形状变化。Complexa 离线 `binder_scRMSD_ca` 则单独对齐 binder，去除了它相对靶标的整体平移和旋转；两者即使名称或阈值相似也不能直接替换。较大的当前 RMSD 不等于 binder 自身折叠出现同等大小的误差。

保留原始记录的指标定义、单位和评分来源；历史缺失值仍是缺失值。另行计算的诊断必须使用不同字段并记录来源，不能覆盖历史 canonical 结果。

## 原生筛选与 AF2 复评分开解释

`canonical_af2_pass` 只说明上述三项门槛；`native_filter_status` 说明直接生成或重设计父本的后端原生状态。两者可能是“AF2 通过、BindCraft 原生 rejected”，不能合称为全面质控通过。原生界面能量、未满足氢键等诊断来自原生模型，并非对最后的 canonical 预测结构重新计算的结果。

同时查看 `canonical_assessment_status`：`complete` 表示可确认 canonical 来源；`not_canonical` 表示这条记录不是 canonical 复评，不代表复评失败；`unknown` 或通过值为 `null` 表示来源或资格尚不能确认。

新 BindCraft worker 启动时保存实际使用的过滤配置快照与哈希，并从逐设计 CSV 记录失败条件、数值、比较符和阈值。`failure_csv.csv` 是批次汇总，不能据此给每份候选分配失败原因。缺快照、数据损坏或结论冲突时，报告 `unavailable`/`inconsistent`；旧归档可以保留已记录的 accepted/rejected，但不能拿当前工具安装中的默认配置补称为当时使用的阈值。ProteinMPNN 等产生新序列后，不继承祖先 BindCraft 的筛选结论。

规划器可从 `qualification_summary` 和 exemplar 的 `qualification` 查看这些信息，决定下一步。保留原有 canonical 筛选定义与分数，不把记录原因的修复当成新增通过候选。

## 导出

从已解析的目标记录取得 `TARGET_ID`，在服务器上选新输出目录。`export` 后的 `--` 将参数传给技能内置导出器：

```bash
"$SKILL_PYTHON" "$SKILL_DIR/scripts/trex_skill.py" export -- \
  --archive-root "$ARCHIVE" --target-id "$TARGET_ID" \
  --n "$EXPORT_COUNT" --out-dir "$EXPORT_DIR" \
  --foldseek-binary "$FOLDSEEK_BIN" --mmseqs-binary "$MMSEQS_BIN"
```

输出包含 `manifest.csv`、`manifest.json`、`pdbs/`，不自动生成 FASTA。结构文件保留原始内容，通常是 binder–target 复合物；导出器不会自动裁出 binder 单链。manifest 的 `binder_chain`、`target_chains` 和 `output_chain_identity` 用于解释链身份；旧归档可能显示 `legacy_unverified`，不能把它当作已核验。`structure_sha256` 用于核对每个导出结构的内容。

每行还包含父本、生成来源、canonical 与 native 两套状态、原生失败项及证据/缺失原因；CSV 中复合字段采用 JSON 编码。原生 rejected 的 AF2 通过候选仍可作为待审查候选导出，摘要会显式计数。来源缺失、关联歧义、跨靶标或仅 advisory 的三分数记录不能获得 canonical 导出资格，排除情况会列在 diagnostics。

导出依据归档中保存的评分，不重新推理，也不验证结构是否完整或适合合成；占位结构也可能随达标记录被复制。回答“能否直接合成”时，需要核实实际结构、binder 序列与链身份、来源和缺失项；复制成功或哈希一致只证明文件一致。未经这些核实，交付应称为供审查的候选。

`--n` 为上限，报告实际数量。结果按计算评分排序；聚类只做标注，不改变排名，也不保证排除所有重复结构。`--no-dedup` 跳过标注，不能称为完成去重。仅分析模式跳过工具时明确限制。

导出前会检查归档完整性，损坏 JSONL、重复结果 ID、非法评分或不存在的目标 ID 会中止导出。检查失败时先查看 `analyze validate`，不要删掉报错记录来获得成功状态。导出期间若复制失败或源结构发生变化，不会发布不完整的新目录；已存在的完整导出保留。`--overwrite` 只接受带有效导出 manifest 的旧导出目录，额外放入的结构文件会阻止覆盖。

默认将导出目录保留在服务器，向用户报告远程路径与摘要。只有用户明确要求下载时，才在验证 manifest 与结构文件数后，用可用的 SCP/SFTP 等方式传回用户指定的文件或整个导出目录；不要默认安装本地分析工具或自动下载原始数据。仅复制 JSONL 或 archive 可能漏掉它引用的外部结构。完整备份需要包含全部引用产物、原始配置与 provenance。优先新目录，覆盖仅在用户已授权且导出器支持时使用。
