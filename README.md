# trex-binder-design

在 macOS 或 Windows 的 Codex 中，用自然语言开展远程 binder design。skill 内置从 T-REX 改编的工作流程代码，由当前 Codex 规划与监督，通过 SSH 调用远程 Linux 服务器上的设计工具。

当前发行版：**v0.3.10**。本地无需 Python、WSL、CUDA 或模型权重；输入、计算环境与完整结果默认保留在服务器。

## 安装到 Codex

1. 下载 [v0.3.10 完整安装包](https://github.com/eggdoger/trex-binder-design/releases/download/v0.3.10/trex-binder-design-v0.3.10.zip)。
2. 解压后，将整个 `trex-binder-design` 目录放进准备使用它的项目的 `.agents/skills/`。
3. 检查目录层级如下。不要只复制 `SKILL.md`，也不要多套一层同名目录。
4. 在 Codex 中打开该项目，新建会话，输入下面的首次使用指令。

```text
你的项目/
└── .agents/
    └── skills/
        └── trex-binder-design/
            ├── SKILL.md
            ├── agents/
            ├── assets/
            ├── references/
            └── scripts/
```

也可以直接克隆本仓库并在 Codex 中打开；项目内已包含完整 skill。若希望所有项目共用，可将同一完整目录放到 `$CODEX_HOME/skills/`；未设置 `CODEX_HOME` 时通常为用户目录下的 `.codex/skills/`。

## 从头使用

首次连接：

> 用 trex-binder-design 带我配置远程 GPU 服务器，先检查设计环境。我的 SSH 别名是 `<你的别名>`。先说明哪些已可用、缺什么，以及下一步怎么做。

准备靶标：

> 我要为这个靶标设计 binder。先检查服务器上的结构，再帮我整理链、热点和设计约束，只问还缺的信息。

确认输入、资源与预算后，开始自适应设计：

> 沿用已经确认的靶标和服务器，最多同时用 1 张 GPU、总时限 4 小时。先设计一批并独立复评，再根据结果调整策略继续下一批，直到达到预算或我要求停止。报告每次调整的依据。

结束时：

> 不再生成新候选，把已有候选的独立 AF2 复评做完，再汇总筛选与聚类情况，导出结果到服务器。列出未完成的项目和原因。

日常使用不必指定工具名称或编写命令。Codex 会检查实际可用的后端、生成配置并操作服务器。首次设置需要 SSH 入口，真实计算还需要明确的靶标、输入结构、约束、GPU 资源和预算；安装 skill 本身不会启动 GPU 任务。

## 工作流程与后端

| 部分 | 职责 |
| --- | --- |
| 本地 Codex | 与用户交互，读取证据，选择下一步动作，监督进度 |
| 远程内置执行器 | 校验计划、调度后端、解析结果、记录来源与归档 |
| BindCraft、BoltzGen、Complexa | 已启用且检查通过时，用于生成设计候选 |
| ProteinMPNN | 用于序列重设计，之后仍需重新折叠与复评 |
| 独立 AF2 | 形成可比较的 canonical 复评结果 |
| Foldseek、MMseqs2 | 标注结构和序列聚类 |

核心是“**读取当前结果 → 选择合法动作 → 执行 → 形成新证据 → 再次决策**”的循环。每一轮可以生成、重设计或复评，并非固定重复同一条工具路径。仅启用 BindCraft 时，常见最小流程是 BindCraft → 独立 AF2 → 聚类 → 调整下一批；启用其他后端后，Codex 可在允许的路线内选择。

无需另外安装 T-REX。设计后端、Python 环境与权重仍需在服务器单独配置，本 skill 不含一键安装全部后端和权重的安装器，也不会自动申请 GPU allocation。

## 使用时要了解

- **当前 Codex 会话需要持续参与决策。** 关闭会话后不能保证继续自主运行。决策超时会停止新增调度并收尾已有任务；硬时限由服务器调度器保证。
- **AF2 达标和后端原生筛选是两份结论。** BindCraft 原生 Rejected 的候选可以继续复评或导出，但会保留拒绝状态和已有原因；历史证据未保存的原因显示未知。计算筛选不能替代结合能力、亲和力或特异性的湿实验验证。
- 停止、复评收尾、恢复和导出均可用自然语言提出；恢复要求原输入与运行代码一致，并先确认旧作业已结束。
- 目前支持目标链和热点；禁结合面、实验几何、成药性过滤尚未实现，不能将这些约束当作已执行。

详细说明：[上手指南](.agents/skills/trex-binder-design/references/getting-started.md) · [远程部署](.agents/skills/trex-binder-design/references/remote-environment.md) · [运行、停止与恢复](.agents/skills/trex-binder-design/references/campaign-workflow.md) · [结果与导出](.agents/skills/trex-binder-design/references/results.md) · [验证范围](docs/validation.md)。

上游来源与许可见 [NOTICE](NOTICE.md) 和 [LICENSE](LICENSE)。
