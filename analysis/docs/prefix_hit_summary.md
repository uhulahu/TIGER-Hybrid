# Prefix-3 vs Full SID 命中分析

> 来源：`model/analyze_prefix_hit.py`，基于 RQ-KMeans + Extra Token (E4)，Beauty test，beam=30。

---

## 问题

Extra token（第四层随机编号）是否显著拖累了 TIGER 的召回？如果前三层前缀已正确、仅 extra token 预测错误，这些样本的 Recall 损失有多大？

---

## 方法

对每个 test 样本的 beam search 结果（beam=30），分别计算：

- **Prefix-3 Recall@K**：前三层 SID 与 target 完全一致的样本比例
- **Full SID Recall@K**：四层 SID 与 target 完全一致的样本比例
- 两者之差即为「前三层正确但第四层错误」导致的 Recall 损失

同时在所有 full-SID miss 样本中统计首次分叉位置，分解 L1/L2/L3/L4 各层的独立贡献。

---

## 结果

| 指标 | 值 |
|---|---|
| Prefix-3 R@20 (R_prefix3@20) | 0.0918 |
| Full SID R@20 (R_full@20) | 0.0875 |
| R@20 差距 | **0.0043** |
| Full-SID miss 总数 | 19,979 |
| 其中：前三层已错 | 19,841 (99.3%) |
| 其中：前三层正确、仅错在 L4 | **138 (0.7%)** |

---

## 结论

Extra token 不是主要瓶颈。

- 即使存在一个完美方法——只要前三层命中就能准确选中桶内正确 item——R@20 的上限也仅从 0.0875 提升至 **0.0918**。
- 在全部 full-SID miss 中，仅 0.7% 属于前三层已经命中、但 suffix 预测错误；99.3% 的失败在前三层就已发生。
- 优化重点应在 L1/L2 的早期路由，而非最后一层 token。
