# TIGER vs Content 重叠与分桶分析汇总

> TIGER + Content last_1 两路 top-50 的命中重叠、item 重叠与分桶表现。
> 来源 notebook: `analysis.ipynb`, `analysis_rqkmeans.ipynb`, `analysis_rqkmeans_L4_lam1.ipynb`

---

## 配置一览

| # | 简称 | TIGER Tokenizer | 来源 notebook |
|---|------|----------------|-------------|
| 1 | **rqvae** | RQ-VAE (TrainSK+InferSK+extra token) | `analysis.ipynb` |
| 2 | **rqkmeans** | RQ-KMeans (3-layer+extra token) | `analysis_rqkmeans.ipynb` |
| 3 | **rqkmeans-L4-lam1** | RQ-KMeans + Structured L4 λ=1 | `analysis_rqkmeans_L4_lam1.ipynb` |

> Content 路完全相同（`content_sim_last1_top50`）。

---

## 表 1：Target 命中重叠（K=20）

对每个用户，哪路命中了 target？

| 模型 | TIGER R@20 | Content R@20 | Both hit | TIGER only | Content only | Union R@20 | 互补率 |
|------|:----------:|:------------:|:--------:|:----------:|:------------:|:----------:|:------:|
| rqvae | 0.0835 | 0.0779 | 330 (1.5%) | 1538 (6.9%) | 1412 (6.3%) | 0.1467 | 89.9% |
| rqkmeans | 0.0877 | 0.0779 | 360 (1.6%) | 1602 (7.2%) | 1382 (6.2%) | 0.1495 | 89.2% |
| rqkmeans-L4-lam1 | 0.0894 | 0.0779 | 368 (1.6%) | 1632 (7.3%) | 1374 (6.1%) | 0.1509 | 89.1% |

> **互补率** = (TIGER only + Content only) / (Union hits)，即命中样本中仅被单路覆盖的比例。近 90% 的命中是单路独占，说明两路高度互补。

---

## 表 2：Item 列表重叠（K=20）

两路 top-20 推荐列表的 item 级别重叠。每用户平均指标。"独有"意识是一个方法推了但另一个方法没推。

| 模型 | 交集 item num/用户 | TIGER 独有/用户 | Content 独有/用户 | Jaccard |
|------|:-------------:|:---------------:|:-----------------:|:-------:|
| rqvae | 0.68 | 19.32 | 19.32 | 0.017 |
| rqkmeans | 0.76 | 19.24 | 19.24 | 0.019 |
| rqkmeans-L4-lam1 | 0.82 | 19.18 | 19.18 | 0.021 |

> Jaccard = `全局总交集 / 全局总并集 = Σ|T_set ∩ C_set| / Σ|T_set ∪ C_set|`。取值范围 [0, 1]，0=完全不重叠，1=完全一致。也可以从每用户均值近似：交集/(40 − 交集) ≈ 0.68/39.32 ≈ 0.017。

> K=20 时每用户平均仅 ~0.7 个 item 同时出现在两路 top-20 中，各自独有 ~19 个。Jaccard 在所有 K 下均 <3%。两路 top-20 候选集合高度异质，但与 target 命中 90% 互补率一致。低列表重叠不一定代表两路学习到的兴趣语义完全独立，也可能受到候选空间、分数尺度等因素影响。

---

## 表 3：RRF Union Recall

使用 RRF（Reciprocal Rank Fusion）对两路 top-50 合并排序。

> **RRF 公式**：`score(item) = 1/(k + rank_T) + 1/(k + rank_C)`，其中 rank 为该 item 在各路内部的排名（1-indexed）。k 控制排名靠后项的衰减速度——k 越大曲线越平，排名差异越不敏感。此处 k=60（标准默认值），两路等权重，按合并 score 降序取 top-K。这里使用 k=60，等权重。

| 模型 | R@5 | R@10 | R@20 | R@50 |
|------|:-----:|:-----:|:-----:|:-----:|
| rqvae | 0.0504 | 0.0764 | 0.1118 | 0.1661 |
| rqkmeans | 0.0507 | 0.0779 | 0.1130 | 0.1676 |
| rqkmeans-L4-lam1 | 0.0518 | 0.0790 | 0.1135 | 0.1706 |

---

## 表 4：内容相似度分桶

按 target 与 last item 的 cosine similarity 三分位分桶。Content **在 Low/Medium 桶中 Recall 接近 0**——Content 路每个用户仍会生成 top-50 候选，但当 target 与 last item 相似度较低时，target 几乎无法通过 last-item content sim 被召回。该分桶依据本身就是 target-last similarity，因此它描述了 Content 路的适用边界，但不能作为完全独立的因果证据。TIGER 在所有桶中均有效。

| 模型 | 桶 | n | TIGER R@20 | Content R@20 |
|------|-----|---|:----------:|:------------:|
| rqvae | Low sim | 7447 | 0.0365 | 0.0000 |
| | Medium sim | 7469 | 0.0665 | 0.0001 |
| | High sim | 7447 | 0.1476 | 0.2338 |
| rqkmeans | Low sim | 7447 | 0.0430 | 0.0000 |
| | Medium sim | 7469 | 0.0640 | 0.0001 |
| | High sim | 7447 | 0.1563 | 0.2338 |
| rqkmeans-L4-lam1 | Low sim | 7447 | 0.0416 | 0.0000 |
| | Medium sim | 7469 | 0.0667 | 0.0001 |
| | High sim | 7447 | 0.1601 | 0.2338 |

> **TIGER 在低/中相似度场景是唯一有效的路**。在 High sim 桶中 Content 更强（R@20: 0.23 vs TIGER 0.15-0.16），但两者融合后进一步提升。

---

## 表 5：Target Popularity 分桶

按 target 在训练集中出现次数三分位分桶。

| 模型 | 桶 | n | TIGER R@20 | Content R@20 | 更强路 |
|------|-----|---|:----------:|:------------:|:------:|
| rqvae | Tail | 8273 | 0.0063 | 0.0898 | Content |
| | Mid | 6466 | 0.0438 | 0.0806 | Content |
| | Head | 7624 | 0.2011 | 0.0627 | TIGER |
| rqkmeans | Tail | 8273 | 0.0098 | 0.0898 | Content |
| | Mid | 6466 | 0.0526 | 0.0806 | Content |
| | Head | 7624 | 0.2021 | 0.0627 | TIGER |
| rqkmeans-L4-lam1 | Tail | 8273 | 0.0088 | 0.0898 | Content |
| | Mid | 6466 | 0.0554 | 0.0806 | Content |
| | Head | 7624 | 0.2058 | 0.0627 | TIGER |

> **Content 擅长长尾，TIGER 擅长头部**。两者的 recall 曲线呈现显著反转——Content 在 Tail 上 >0.08，TIGER <0.01；TIGER 在 Head 上 >0.20，Content 仅 0.06。TIGER 存在明显的热门偏置，而 Content 凭借内容表示能够绕过交互频次不足，在长尾 item 上工作。Popularity 分桶呈现出最显著的性能反转，是观察到的主要互补来源之一。

---

## 表 6：类目跳转分桶

last item 与 target 的 category 前缀匹配。Beauty 类目层级中 Cat-1 全为 "Beauty"，跳过；Cat-2 为二级大类（如 Skincare / Makeup），Cat-3 为三级细分。

| 模型 | 层级 | 桶 | n | TIGER R@20 | Content R@20 |
|------|------|-----|---|:----------:|:------------:|
| rqvae | Cat-2 | Same | 9792 | 0.1147 | 0.1568 |
| | | Diff | 12571 | 0.0593 | 0.0165 |
| | Cat-3 | Same | 5079 | 0.1461 | 0.2264 |
| | | Diff | 16978 | 0.0646 | 0.0339 |
| rqkmeans | Cat-2 | Same | 9792 | 0.1184 | 0.1568 |
| | | Diff | 12571 | 0.0639 | 0.0165 |
| | Cat-3 | Same | 5079 | 0.1488 | 0.2264 |
| | | Diff | 16978 | 0.0694 | 0.0339 |
| rqkmeans-L4-lam1 | Cat-2 | Same | 9792 | 0.1208 | 0.1568 |
| | | Diff | 12571 | 0.0650 | 0.0165 |
| | Cat-3 | Same | 5079 | 0.1490 | 0.2264 |
| | | Diff | 16978 | 0.0717 | 0.0339 |

> Cat-2 和 Cat-3 的规律一致：同类别时 Content 更强，跨类别时 TIGER 更强。Cat-3 的粒度更细（同 Cat-3 样本仅 5079），同 Cat-3 时 Content 优势进一步放大（~0.23 vs TIGER ~0.15）。TIGER 在跨类目场景中补充了 Content 难以覆盖的需求。

---

## 表 7：用户历史长度分桶（R@20）

| 模型 | Short (≤5) | Medium (6-7) | Long (>7) |
|------|:----------:|:------------:|:---------:|
| | n=11384 | n=4490 | n=6489 |
| TIGER (rqvae) | 0.0791 | 0.0826 | 0.0918 |
| TIGER (rqkmeans) | 0.0827 | 0.0831 | 0.0999 |
| TIGER (rqkmeans-L4-lam1) | 0.0842 | 0.0833 | 0.1028 |
| Content | 0.0800 | 0.0742 | 0.0767 |

> Content 不受历史长度影响（R@20 稳定在 0.075-0.080），因其仅依赖 last item，信号量固定。TIGER 在拥有较长历史的用户上表现更优（短→长：+0.013~+0.019），说明 TIGER 能够利用更长的行为序列捕捉用户偏好，与其设计预期一致。短历史上两者接近（~0.08），长历史上 TIGER 明显占优（~0.10 vs ~0.08）。该趋势也可能部分受到用户活跃度、目标流行度等因素影响。

---

## 总结

1. **两路高度互补**：Jaccard ~1-3%，近 90% 命中来自单路独占。融合后 R@20 提升 ~35%。
2. **互补的最大驱动力是 popularity**：Content 占优长尾（Tail R@20 0.09 vs TIGER 0.01），TIGER 占优头部（Head R@20 0.21 vs Content 0.06）——两条 recall 曲线近乎正交。
3. **相似度是第二驱动力**：在低/中相似度桶中 Content 的 top-50 候选几乎为空（candidate generation 阶段未覆盖），TIGER 是唯一有效路。
4. **类目跳转**：同类别时 Content 更强（Cat-2 同: 0.16 vs 0.12; Cat-3 同: 0.23 vs 0.15），跨类别时 TIGER 更强（Cat-2 跨: 0.06 vs 0.02; Cat-3 跨: 0.07 vs 0.03）。类目越细、越同类的场景 Content 优势越大，TIGER 在跨类场景不可替代。
5. **历史长度**：TIGER 随历史增长而提升，Content 稳定。长历史用户 TIGER 占优（0.10 vs 0.08），短历史两者接近。
