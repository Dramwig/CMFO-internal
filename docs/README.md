# CMFO 研究文档

## 当前状态

- 阶段：P0，问题定义与新颖性审计。
- 原始构想：已镜像到 [../idea.md](../idea.md)。
- 实现：P1 完成，Gate A 已通过；包含 CMFO、B0–B4、训练/评估 CLI 与 E0/E1 测试。
- 实验结果：只有代码验证 smoke，不是论文实验结果。
- 投稿目标：NeurIPS / ICLR / ICML；具体会议和届次在核心结果出现后决定。

## 北极星目标

构建并验证一种文本–图像连续场算子：模型参数不依赖固定 token、patch 或持久 latent index，在匹配计算预算下保持有竞争力的任务表现，并对数值离散变化展现可测量的稳定性和收敛性。

首篇论文回答：

> 多模态模型若把内部状态从持久离散单元改为原生域函数场，是否能获得标准离散模型缺少的离散化外推能力，而不牺牲主要任务性能？

## 文档导航

| 文档 | 用途 | 何时更新 |
|---|---|---|
| [research-plan.md](research-plan.md) | 问题、假设、主张、范围、理论目标、反证条件 | 主张或研究范围变化时 |
| [experiment-plan.md](experiment-plan.md) | 数据、基线、指标、实验矩阵、消融与统计协议 | 实验设计变化前 |
| [experiment-log.md](experiment-log.md) | 实验登记、结果索引与决策记录 | 每次正式实验前后 |
| [paper-plan.md](paper-plan.md) | 论文叙事、章节、图表和证据矩阵 | 证据或写作结构变化时 |
| [roadmap.md](roadmap.md) | 阶段、依赖、退出门槛和优先级 | 阶段切换时 |
| [experiment_conditions/2026-08-27_pro6000_inventory.md](experiment_conditions/2026-08-27_pro6000_inventory.md) | pro6000 硬件、存储和路径事实 | 环境或服务器变化时 |
| [design/README.md](design/README.md) | 架构设计记录入口 | 重要设计形成时 |
| [records/README.md](records/README.md) | 正式实验记录入口 | 实验结束后 |

当前实现与验证：

- [P1 实现映射](design/2026-08-27_mvp-implementation.md)
- [EXP-000 代码验证记录](records/2026-08-27_code_validation_v2.md)

## 当前冻结决策

1. 第一篇论文只做文本–图像，不同时扩展音频、视频和 3D。
2. 主线是离散化不变性与算子一致性，不绑定非自回归生成。
3. 第一版使用有限 residual operator blocks；Neural ODE 是后续方向。
4. 先做可控程序化场景，再进入真实多模态基准。
5. 避免绝对化 token-free 表述，优先使用 grid-independent、persistent-unit-free 或 discretization-aware。
6. 本地 CMFO-internal 是独立 Git 根；服务器代码、数据、开源权重和项目 checkpoint 四层分离。

## 待冻结决策

- 最终真实数据集及其许可、规模和评价协议。
- 全局理解任务采用检索还是 VQA；密集任务采用 grounding 还是分割。
- 连续算子的具体参数化。
- 理论结果强度。
- 计算预算、模型规模、Python/torch 环境和正式验收容差。

## 事实优先级

冲突时按以下顺序处理：

1. 已复现的实验事实；
2. research-plan 与 experiment-plan 中冻结的定义和协议；
3. idea.md 的原始愿景；
4. 尚未登记的口头设想。

idea.md 保留完整设计空间；范围收缩和论文决策写入本目录。

## 服务器归档

- [2026-10-06 pro6000 释放前归档](records/2026-10-06_pro6000_retirement.md)：本地资产路径、校验、环境和恢复说明。
