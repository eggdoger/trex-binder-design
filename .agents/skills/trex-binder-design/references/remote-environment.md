# 个人电脑 Codex → 远程 GPU

## 本地客户端

本地是 macOS 或 Windows 的 Codex。用户安装 skill 并使用已配置的 SSH 连接即可；不要求本地 Python、WSL、CUDA、后端环境、权重或实验数据。Windows 可使用已有的原生 SSH 客户端；不要因为远程 Python 使用 `fcntl` 而要求用户改用 WSL。

Windows 本地解压优先使用短项目路径。PowerShell 5.1 的 `Expand-Archive` 在完整成员路径过长时可能只解压部分文件；出现错误应换到短路径重新解压，并核验 manifest 中全部文件后再部署，不能把部分目录当作有效安装。

包内 Python 文件用于分发执行代码，部署副本在服务器运行。Codex 本地只读技能说明、编写决策，通过 SSH 与服务器交互。输入、mailbox、日志、完整结果默认留在服务器。

提交 planner/supervisor 决策时使用 [可靠决策传输](decision-transport.md)：SCP 上传 UTF-8 JSON，再以非 TTY 的 `ssh -T` 调用官方 `respond`。其中提供原生 PowerShell 与 macOS shell 的具体调用及回执核验，避免把长 JSON 写入 Windows 交互控制台。

## 在服务器部署执行代码

本地使用 Codex 已能发现的 skill 即可，例如全局 `$CODEX_HOME/skills/trex-binder-design/`（未设置时为 `~/.codex/skills/`）或项目 `.agents/skills/`；无需重复安装到项目。将完整 `trex-binder-design/` 目录复制到远程可读位置。远程无需下载或安装 T-REX。不能只传 SKILL.md；`scripts/binder_runtime/` 和其 `data/` 是执行所需代码与资源。

**以下安装命令在远程 Linux 服务器上执行**。服务器控制器使用 Python 3.12（CPU 测试版本），依赖 `fcntl`；这是服务器的运行要求，不是个人电脑的要求。默认 `python3` 为 3.10 不代表服务器没有 3.12：先清点已知 conda 安装、环境列表和 module，并直接检查候选环境 `bin/python --version`；非交互 PATH 不能代替这一步。复用已有兼容控制器环境；需新建时，将 `CONTROLLER_PYTHON` 设为已核实的 Python 3.12 解释器绝对路径，在独立的 `ENV_DIR` 创建环境。已有后端环境可提供基础解释器，但不重新初始化它，也不向其中安装控制器依赖。确无可用 3.12 时再按实验室允许的方式准备。

```bash
"$CONTROLLER_PYTHON" --version
"$CONTROLLER_PYTHON" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --log "$ENV_DIR/pip-install.log" --timeout 30 --retries 2 -r "$SKILL_DIR/requirements.txt"
"$ENV_DIR/bin/python" "$SKILL_DIR/scripts/doctor.py" --skill-dir "$SKILL_DIR" --role remote
```

运行依赖是 Biopython 和 PyYAML；NumPy 由 Biopython 引入。不安装 `trex`，不需要 Qwen、vLLM 或 LLM API。`assets/runtime-manifest.json` 记录来源 commit 和文件哈希，启动时校验执行包。部署后不要就地修改已运行 campaign 的 bundle；代码更新应重新验证并建立新任务或显式迁移。

若 Biopython/PyYAML 已命中缓存，安装仍停在 NumPy 下载，保留 pip 日志和进程退出码，检查下载是否有进展及该下载源的连接情况。持续无进展时结束这次安装并定位原因，再作有限重试；同一原因再次失败就报告阻塞与日志位置，不反复启动无日志的安装或把未退出的进程当作安装完成。

## 连接与清点

复用已授权 SSH 别名，保留主机密钥验证；可使用 `BatchMode=yes` 与 `ConnectTimeout=10`。认证失败时报告，不读取或回显私钥。未知的调度方式先只读清点；非交互 SSH 的 module/conda PATH 可能尚未初始化。

从本地 Codex 发出 SSH 命令，远程执行 Python。例如下列单行调用适用于 macOS shell 或 Windows PowerShell；`labgpu` 和路径是占位示例，先替换为已确认的服务器信息：

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 labgpu '/srv/binder-env/bin/python /srv/trex-binder-design/scripts/doctor.py --skill-dir /srv/trex-binder-design --role remote'
```

部署前只需清点时，可先通过 SCP/SFTP 复制 `doctor.py` 到服务器临时位置，再用服务器 `python3` 执行 `--host-only --role remote`。所有参数使用远程路径。不要直接把 Bash 的 `<` 重定向或反斜线续行示例粘到 PowerShell；按当前本地 shell 引用参数，并保持远程命令由服务器 shell 解析。

Windows PowerShell 5 若在 `Get-Content … | & ssh.exe …` 报 `CantActivateDocumentInPipeline`，检查继承的 `$env:PATHEXT`。值仅为 `.CPL` 时，可用 `Start-Process` 指定 `ssh.exe` 完整路径，以 `-RedirectStandardInput`、`-RedirectStandardOutput`、`-RedirectStandardError` 传入文件并保存日志，配合 `-Wait -PassThru` 读取返回进程对象的 `ExitCode`。若用 `WaitForExit(milliseconds)` 限时等待，先读取进程的 `Handle`，等待结束后拒绝空的 `ExitCode` 并传播实际退出码；PowerShell 成功退出不能证明 SSH 命令成功。若执行策略阻止 `-File` 运行 `.ps1`，对这类短调用使用内联 `-Command`；无需为此绕过或修改执行策略。

doctor 不读 `.env`、不导入运行代码、不下载或安装。登录节点没有 GPU 不代表集群没有 GPU；正式检查在已分配 GPU 的节点内完成。

## 外部设计后端

技能直接启动这些后端的程序。它们有各自环境与大型权重，**不包含在 skill ZIP 内**。优先复用实验室现有安装。

| 后端 | skill 调用方式 | 需要配置的内容 |
| --- | --- | --- |
| BindCraft | 后端 Python 执行 `bindcraft.py` | 仓库、独立环境、后端资产/权重 |
| BoltzGen | 独立环境的 `boltzgen run` | 仓库、可执行文件、cache/权重 |
| Complexa | 后端环境执行 `proteinfoundation.generate` | 仓库、Python、其 `env.sh` 和资产 |
| AF2 refilter | 后端 Python 执行技能内置 `binder_runtime.af2_refilter_runner` | community models 根目录、AF2 参数目录 |
| ProteinMPNN | 后端 Python 执行其脚本 | ProteinMPNN 目录、脚本、权重 |
| Foldseek/MMseqs | 服务器上的二进制 | 实际可执行文件，用于结构/序列聚类 |

兼容性检查所用后端 revision 和权重清单在 `scripts/binder_runtime/data/config/reproducibility/`。其中资产清单可能保留上游历史安装文本，只用于核对资产和版本；部署遵循本说明，不执行旧的 T-REX/Qwen 安装或提交流程。新装按各后端自身安装文档与这些版本约束配置；此 skill 不包含一键下载全部权重的安装器。不能拿“目录存在”当作后端可运行，不能跳过 revision/权重检查来声称验收成功。

Complexa 的登记安装流程会修正 `pyproject.toml` 依赖元数据；检查只接受 `complexa_environment.json` 中精确哈希对应的补丁，其他受版本控制的修改仍会失败。安装 CUDA 依赖前核对计算节点的实际驱动；若原锁不兼容，应在隔离环境中解析兼容依赖、保存派生锁与差异，并重新验证推理，不能把它报告成原锁环境。

依赖版本一致和 `pip check` 通过仍不能证明二进制可加载。旧系统可能无法加载要求更高 glibc 的 PyG wheel；先在实际后端环境检查导入，再在 allocation 内验证 PyTorch、JAX 和扩展的 GPU 运算。同版本源码重编译时保留源码版本、编译器、CUDA 架构、wheel 哈希与派生环境记录；只验证过一种 GPU 架构的构建不能视为适用于所有 GPU。

BindCraft 默认保存轨迹动画，需要后端环境中的 Matplotlib 与可用的 H.264 视频编码器。启用动画时，`check` 会在实际后端环境做小型 CPU 编码检查；失败时按报告核对 writer 配置、ffmpeg 路径及共享库，不能凭 `ffmpeg --version` 成功就继续。优先在本次隔离环境修复，不修改实验室共享安装。

在 campaign 的 `backends` 中填写实际路径（完整字段由 `init` 生成）。无需设置源码 `repo_root`；内置代码位置自动确定。路径可为配置文件旁的相对路径，但跨主机不要传本地 Windows/WSL 路径。`legacy_complexa_repo` 是兼容旧 community models 布局的后端路径，不是 T-REX 路径。AF2/MPNN 可用 `community_models_path`、`af2_data_path`、`af2_python`、`proteinmpnn_path`、`proteinmpnn_weights_path`（权重目录）和 `proteinmpnn_python` 分别配置，不要求共享一个环境。

技能不自动 source `.env`。Complexa 启动仍按该后端要求 source 它的 `env.sh`，所以应使用可信的安装目录。仅使用已核实的输入文件、配置和环境。

## GPU 与调度

`start` 在既有 GPU allocation 内执行，不自动申请 GPU。Slurm 按站点 account、partition、GPU、CPU/RAM 和硬时限编写 job 脚本，主体调用 skill 的 `start`。mailbox 放在登录节点与计算节点可访问的共享目录，Codex 经 SSH 响应，无需暴露模型端口。

所有分配的 GPU 可用于设计 worker。将 allocation 中实际 `CUDA_VISIBLE_DEVICES` 标识传给 `--worker-gpus`，不要盲目从 1 开始或自行重编号。后端显存需求需实际验证。禁止在未经分配的共享登录节点启动 GPU 设计。

首个真实验收使用用户确认的目标和短预算，核实后端启动、请求响应、结果解析、归档及停止。这项 GPU 验收不能由本地 CPU 人工数据测试替代。

## 环境检查后如何继续

向用户解释检查结果和下一步时使用 [使用指南](getting-started.md)。下表供 Codex 将问题转为实际配置工作；安装范围以用户当前请求和已有授权为准，单独的“先检查”请求不直接启动安装或 GPU 作业。

| 发现 | 后续动作 | 完成标准 |
| --- | --- | --- |
| 登录和 Slurm 正常，登录节点看不到 GPU | 记录调度入口、可见 GPU 分区和账户；在准备试跑时确定 allocation | 能区分登录节点清点与计算节点上的 GPU 验收 |
| 默认 Python 3.10，尚无控制器环境 | 先检查已知 conda/module 中的解释器，再选用已核实的 Python 3.12 创建独立控制器环境；部署完整 bundle 并安装其 requirements，保留原后端环境 | 解释器和依赖明确，`doctor --skill-dir` 哈希检查通过 |
| 已有 BindCraft，但 revision 不匹配 | 按包内版本清单与补丁准备独立兼容 checkout；逐项核对可复用环境 | 源码、补丁和依赖满足检查，不修改现有共享安装来碰碰运气 |
| 权重目录存在，但未核实内容 | 对照清单检查文件和大小，再验证要求的哈希；只补齐缺失或不匹配的资产 | 必需资产全部匹配；不能用文件夹名代替完整性验证 |
| 部分后端尚未安装 | 依据实际设计需求选择最小流程，只补齐它的依赖 | 已选流程通过配置检查；未启用后端不被报告成可用 |

已有 BindCraft 时可优先准备 **BindCraft → 独立 AF2 复评 → 聚类**。初始化时明确选择 `--families bindcraft`，复评路线会自动纳入有效配置。仍须配置 AF2 的 Python、指定 ColabDesign 源树与参数，以及符合版本要求的 Foldseek/MMseqs2；这些是当前最小流程的必需部分。此时无需启用 BoltzGen、Complexa 生成或外部 ProteinMPNN 重设计，也无需下载它们的生成权重。

没有真实靶标和预算时，可以完成环境部署与资产核验，报告“环境准备到哪一步”；待用户提供输入后再生成 campaign、运行完整 `check` 并安排小规模验收。不要为了让检查通过而填入假靶标、假 GPU 分配或默认预算。用户选择补齐环境后，沿用清点结果继续，不重复要求提供已知路径。
