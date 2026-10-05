# EXP-000：CMFO P1 代码验证

日期：2026-08-27（Asia/Shanghai）。

## 目的

验证本地编写并同步到 pro6000 的 CMFO P1 系统是否满足 Gate A。该运行只验证代码，不用于支持论文性能主张。

## 代码与环境

- 本地代码根：C:/Users/zixi-/Desktop/PaperField/CMFO/CMFO-internal
- 服务器代码根：/root/autodl-tmp/CMFO/CMFO-internal
- Python：/root/autodl-tmp/CMFO/.venv/bin/python
- Python 3.10.20；PyTorch 2.7.1+cu128；pytest 9.0.3；Ruff 0.16.4
- 设备：CPU。检查时 GPU 已由其他任务占用，未干扰或终止其他进程。

## 静态与测试证据

执行：

    python -m ruff check src tests
    python -m ruff format --check src tests
    PYTHONPATH=src python -m pytest -q

结果：

- Ruff：通过；
- 格式检查：通过；
- pytest：35 passed；
- 无 Transformer warning；
- 单批过拟合：loss 2.17398 → 0.000607，accuracy 1.0，mIoU 1.0。

测试覆盖：

- 解析二次函数求积误差随 4、16、64 节点下降；
- 规则、抖动和随机布局；
- 批内可变输入与隐节点数量；
- 输入观测数组和内部求积节点置换不变性；
- 未见节点数直接推理且参数形状不变；
- CPU bfloat16、有限梯度；
- B0–B4 前向与反向；
- checkpoint、报告和端到端训练。

## 独立 smoke 与评估

run_id：

    cmfo_code_validation_20260827_v2

路径：

    /root/autodl-tmp/CMFO/CMFO-internal/outputs/smoke/cmfo_code_validation_20260827_v2
    /root/autodl-tmp/CMFO/checkpoints/cmfo_code_validation_20260827_v2/best.pt

验证：

- 参数量：18,777；
- checkpoint：365,747 bytes；
- quadrature sweep：12 条；
- observation resolution/layout sweep：6 条；
- query resolution sweep：3 条；
- 独立 evaluate 命令成功重新加载 checkpoint；
- inspect 未发现 node_embedding、position_embedding 或 patch_embedding 参数。

一次 epoch、8 个训练样本的 validation loss 为 1.97798，accuracy 与 mIoU 均为
0。该数值符合“管线 smoke 而非性能训练”的定位，不能进入论文结果或被解释为
支持/反驳 H1–H3。

## 结论

P1/Gate A 通过：核心系统可运行、可训练、可换网格、可记录和可复验。下一步是预先冻结 Gate B 容差与预算，然后运行正式 E1 多 seed 基线和消融。

