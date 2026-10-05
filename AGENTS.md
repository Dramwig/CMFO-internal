# CMFO Internal 入口

本目录是 CMFO 的独立代码、实验与研究记录仓库。上层工作区为：

    C:/Users/zixi-/Desktop/PaperField/CMFO

## 仓库职责

本仓库保存：

- src/cmfo 中的模型与研究代码；
- configs、scripts 和 tests；
- docs 中的设计、实验协议、环境与结果记录；
- artifacts 中可入 Git 的轻量图、表和报告。

本仓库不保存完整数据集、开源模型权重、训练 checkpoint、优化器状态、大型 cache、完整外部仓库或长时间原始日志。

## 先读

1. [idea.md](idea.md)：研究构想镜像。
2. [docs/README.md](docs/README.md)：研究状态与文档导航。
3. [docs/research-plan.md](docs/research-plan.md)：主张和范围。
4. [docs/experiment-plan.md](docs/experiment-plan.md)：正式实验协议。
5. [docs/experiment-log.md](docs/experiment-log.md)：运行前登记与运行后结论。

## 本地与远端

本地仓库：

    C:/Users/zixi-/Desktop/PaperField/CMFO/CMFO-internal

正式 GPU 工作区：

    ssh pro6000
    cd /root/autodl-tmp/CMFO/CMFO-internal

远端资产严格分离：

    /root/autodl-tmp/CMFO/datasets      数据集与数据侧 cache
    /root/autodl-tmp/CMFO/weights       第三方开源基座权重，只读使用
    /root/autodl-tmp/CMFO/checkpoints   本项目产生的 checkpoint 权重

开源权重不得写入 checkpoints；CMFO 训练产物不得写回 weights。代码不得依赖 CFI 或其他项目目录中的数据/权重软链接。需要复用资产时，应在 CMFO 自有目录登记来源、版本、许可和校验信息。

## 服务器规则

- 所有大型资产写入持久盘 /root/autodl-tmp，不写入 /root overlay。
- 跑任务前先执行 nvidia-smi，不能终止其他项目进程，除非用户明确授权。
- CMFO 验证环境固定为 /root/autodl-tmp/CMFO/.venv/bin/python；它从已核验的
  pf-vlm Python 3.10 环境继承 PyTorch，并在项目作用域安装 CMFO 与 Ruff。
  环境事实见 docs/experiment_conditions/2026-08-27_pro6000_inventory.md。
- 下载与训练命令必须可重入；大文件先校验空间、来源和校验和。
- outputs、logs、wandb 和临时文件留在被 Git 忽略的仓库工作区；可复现摘要写入 docs/records 或 artifacts/reports。
- 服务器事实、配置、seed、代码 revision、数据 revision 和路径必须写进实验记录。

## 代码结构规则

- src/cmfo 只放可复用逻辑；scripts 只做薄入口，不复制模型或指标实现。
- configs 是正式运行的参数事实来源；命令行覆盖项必须进入日志。
- tests/unit 不访问网络或 GPU；tests/smoke 可做小型端到端验证；tests/integration 明确标注外部依赖。
- 参数形状不得依赖最大 token、patch 或求积节点数。
- 求积节点不能有固定可学习 index embedding；文本顺序必须由坐标而非数组存储顺序表达。
- 输出和 checkpoint 使用稳定 run_id，禁止覆盖已有正式运行。

## 研究边界

当前首篇论文只聚焦文本–图像连续场算子和离散化不变性。连续深度、生成、音频、视频和 3D 均在 Gate C 前延后。

任何结果解释必须区分：

- 输入观测点；
- 内部求积点；
- 输出查询点；
- 数值参考解与真实解析解。

若简单多分辨率增强、普通 attention 或固定 latent 基线解释了收益，应缩小主张，不通过扩大规模掩盖。

## 变更完成标准

- 相关测试通过；
- 新实验可由配置复现；
- 路径未越过资产边界；
- 主张或协议变化已更新 docs；
- 未提交密钥、凭据、大模型文件或机器专有缓存。
