# pro6000 释放前归档（2026-10-06）

## 范围与连接

源目录：pro6000:/root/autodl-tmp/CMFO；实际使用 SSH 端口 17974。
本机旧 SSH 配置为 29844，本次通过 `ssh -p 17974 pro6000` 覆盖，未改动共享 SSH 配置。
本地代码：D:/PaperField/CMFO/CMFO-internal。
Git remote：https://github.com/Dramwig/CMFO-internal.git。

## 文件核验与取舍

- 两端原有 79 个非忽略文件逐项 SHA-256 一致，两端原先均无提交和 remote。
- 根目录 idea.md 与仓库镜像 SHA-256 均为 c509a2037057327b030a4e5a8a69b05222e1de4526a1cefc417c951c42b2bd7a。
- datasets/ 与 weights/ 均为空；没有 raw 包、下载脚本或第三方权重需要迁移，不在 hub 创建空数据集条目。
- 两份 checkpoint 已归档至 D:/checkpoints_hub/registry/CMFO/checkpoints/<run_id>/best.pt。
- 原始输出的 9 份 JSON 已归档至同条目 records/；11 个文件的大小和 SHA-256 均与服务器核对一致。
- D:/checkpoints_hub/registry/checkpoints.yaml 已登记两个 run；条目内 manifest.json、checksums.sha256 和 README.md 提供核验与恢复说明。
- Git 内保留轻量运行 JSON 和资产清单：artifacts/reports/pro6000-retirement-2026-10-06/。
- 环境快照：docs/experiment_conditions/2026-10-06_pro6000_requirements.txt。这是共享环境事实清单，含其他项目 editable 路径，不能直接当作可移植 requirements 安装；重建以 pyproject.toml 和既有环境记录为准。
- 不搬运可重建的 .venv、Python/Ruff/pytest cache、egg-info；不删除服务器源数据。
- 本项目本地和远端均仅有主工作树，未发现 .codex-* 工作区或待合并分支。

## 验证

归档前远端 Ruff lint 与 format 检查通过。无卡模式下 nvidia-smi 返回 Permission denied；未运行 GPU 任务。
本次 CPU pytest 单线程复验：35 passed in 154.40s；命令前缀 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1。首次默认线程复验因 167 线程争用被终止，不计为通过；无卡容器 CPU quota 为 0.5 核。
既有实验均为 EXP-000 代码验证，不作为论文性能证据。

## Git 同步与恢复

已在服务器创建初始提交 4fa76555afd1b9541969d88a116e5e4df742f500，再以已验证 bundle 导入本地并推送 origin/main。
服务器直连 GitHub 的 HTTPS 探测遇到 GnuTLS -110，因此允许由本机中转 push。
本地通过系统现有代理 http://127.0.0.1:7897 成功推送。最终验证记录另作提交并同步服务器；核对三端 main 一致后删除两端中转 bundle。

在新服务器 clone 仓库后，将 hub 内 checkpoints/<run_id> 恢复到项目 checkpoints/，
将 records/<run_id> 恢复到仓库 outputs/smoke/，依据新环境调整 resolved_config 中的机器路径。

本地旧 .venv 引用了已不存在的 C:/Users/zixi- Python，不能直接使用；本次测试在服务器执行，后续本地开发需重建虚拟环境。
