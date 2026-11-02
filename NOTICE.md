# 来源与许可

`trex-binder-design` 是面向 Codex 和远程 GPU 工作流的衍生实现，内置运行代码改编自 [ml-struct-bio/T-REX](https://github.com/ml-struct-bio/T-REX)，固定来源提交为 [`bbecc1356556293f8a24ada8dc4483def59379e0`](https://github.com/ml-struct-bio/T-REX/tree/bbecc1356556293f8a24ada8dc4483def59379e0)。它不是上游官方发行版。

原项目使用 MIT 许可，版权声明为 `Copyright (c) 2026 Jinyeop Song`。本仓库保留 [原许可全文](LICENSE)，完整 skill 内也包含 [LICENSE-T-REX](.agents/skills/trex-binder-design/LICENSE-T-REX)。分发源码或完整 skill 时请保留这些声明。

skill 的改编包括 Codex 与远程执行器的决策传输、环境检查、任务控制、结果来源核验，以及原生筛选和独立 AF2 结论的分别记录。发行清单记录上游提交、skill 版本及文件哈希，见 [runtime-manifest.json](.agents/skills/trex-binder-design/assets/runtime-manifest.json)。

BindCraft、BoltzGen、Proteina-Complexa、AlphaFold/ColabDesign、ProteinMPNN、Foldseek、MMseqs2 等后端软件需另外安装。其模型权重、第三方可执行文件及结构数据没有随本仓库分发，继续适用各自的许可、条款和引用要求；本项目的 MIT 许可不替代它们。

包内保留的版本锁定文件、兼容性 patch 和资产清单用于核对兼容版本。资产清单中的链接、哈希及历史安装文本来自固定上游，并不表示那些资产已经打包进本仓库，也不应直接替代当前的部署说明。部署请阅读 [远程环境说明](.agents/skills/trex-binder-design/references/remote-environment.md)。

使用本工作流进行研究时，请根据实际使用的软件、模型和数据分别引用上游；T-REX 的引用信息见其 [CITATION.cff](https://github.com/ml-struct-bio/T-REX/blob/bbecc1356556293f8a24ada8dc4483def59379e0/CITATION.cff)。
