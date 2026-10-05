# CMFO P1 实现映射

日期：2026-08-27。

## 目的

本文把 research-plan 的数学对象映射到可执行代码，避免实现与论文术语漂移。

## 数据与几何

| 研究对象 | 代码 | 约束 |
|---|---|---|
| 输入观测点 | src/cmfo/data/fields.py 的 FieldBatch | 值、坐标、权重、mask 始终对齐 |
| 内部求积点 | GeometryBatch 与 quadrature.py | 无可学习 index；数量与布局可变 |
| 输出查询点 | QueryBatch | 查询坐标与监督分离，不向模型泄漏 target |
| E1 底层场景 | data/synthetic.py | 同一 sample_id 可在任意图像网格重新渲染 |

文本保留全部 UTF-8 byte。改变文本数组存储顺序时必须同步坐标与权重；代码不把删除 byte 当成重采样。

## 模型

- lifting.py：byte/image 原始观测通过带测度权重的积分注意力提升到查询网格。
- common.py：Fourier 坐标编码、measure-aware integral attention、低秩积分算子。
- operators.py：点态、局部、低秩全局和双向跨模态并行更新。
- cmfo.py：残差 operator blocks、完整场泛函分类头和二维坐标查询分割头。
- baselines.py：B0 固定 patch、B1 point Transformer、B2 固定 latent Perceiver、
  B3 固定求积网格 CMFO、B4 同等增强 point Transformer。

模型参数形状只依赖通道、层数、head 和数值近似秩，不依赖输入长度、图像分辨率或求积节点数。byte embedding 是边界值编码器，不是内部求积节点身份。

## 训练与评价

- training/losses.py：全局分类、密集分割和双网格一致性损失。
- training/trainer.py：梯度裁剪、AMP、原子 checkpoint、best-validation 恢复。
- evaluation/evaluator.py：任务指标、ECE、求积 sweep、输入观察网格 sweep、
  输出查询密度 sweep、公共探测网格场误差与延迟。
- cli.py：train、smoke、evaluate 和 inspect 四条独立入口。

报告中的高密度 reference 只是数值参考近似，不是真实连续解。

## 当前边界

- P1 只验证结构与训练机制正确，不证明 H1–H3。
- 程序化 E1 数据已实现；真实 E2 数据集尚未冻结，因此没有伪造通用 adapter。
- B5 必须在新颖性审计后选择。
- FLOPs 的正式统计与参数/FLOPs 匹配实验属于 P2；P1 已记录参数量和实测延迟。

