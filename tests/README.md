# Tests

- unit：CPU、无网络、快速确定性测试。
- smoke：最小端到端运行。
- integration：需要 GPU、数据或第三方权重的显式集成测试。

E0 的置换、求积收敛、变节点 batching 和固定 index 排查应在实现时进入测试。

