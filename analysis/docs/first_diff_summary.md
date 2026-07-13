# First-Diff 位置分布汇总

> 5 个模型配置在 Beauty/test 上（N=22,363, beam=20）的逐层语义 ID 预测分析。
> 来源 notebook: `first_diff_distribution*.ipynb`

---

## 配置一览

| # | 简称 | Checkpoint | Tokenizer | Code path |
|---|------|-----------|----------|-----------|
| 1 | **rqvae-baseline** | `Beauty_baseline_ce/Jul-09-2026_14-20-11` | RQ-VAE (`sk[0-0-0.003]`) | `Beauty_t5_rqvae_260709-sk[0-0-0.003].npy` |
| 2 | **rqvae-collab** | `Beauty_collab_ce/Jul-09-2026_14-20-11` | RQ-VAE + collab loss (`cw[0.001-tau0.2]`) | `Beauty_t5_rqvae_260709-sk[0-0-0.003]-cw[0.001-tau0.2].npy` |
| 3 | **rqkmeans** | `Beauty_baseline_ce_rqkmeans/Jul-09-2026_21-23-03` | RQ-KMeans init codebook | `Beauty_kmeans_code.npy` |
| 4 | **rqkmeans-L4-lam1** | `Beauty_kmeans_L4_lam1/Jul-11-2026_15-59-27` | RQ-KMeans + Structured L4 λ=1 | `Beauty_kmeans_code-L4-lam1.npy` |
| 5 | **rqkmeans-L4-lam10** | `Beauty_kmeans_L4_lam10/Jul-10-2026_23-25-51` | RQ-KMeans + Structured L4 λ=10 | `Beauty_kmeans_code-L4-lam10.npy` |

---

> **关于 `Top-20 Hit`**：此指标通过 **beam search（beam=20）** 生成 20 条 SID 序列，映射到 item 并去重后检查 target item 是否命中。与训练日志 `results/experiment_results.md` 的 **R@20** 评测方式一致（均为 beam search），差异来自 beam size（20 vs 30）和匹配粒度（item 去重 vs SID 精确匹配），训练 R@20 系统性偏高 0.15~0.30pp。详见 [评测方式说明](#评测方式说明)。

---

## 表 1：Top-1 首次分叉分布 & Hit Rate

| Model | EXACT | L1 错 | L2 错 | L3 错 | L4 错 | Top-1 Exact | Top-20 Hit [^2] |
|-------|------:|------:|------:|------:|------:|:-----------:|:----------:|
| rqvae-baseline | 236 (1.1%) | 19046 (85.2%) | 2835 (12.7%) | 246 (1.1%) | 0 (0.0%) | 1.06% | 8.14% |
| rqvae-collab | 223 (1.0%) | 20759 (92.8%) | 1272 (5.7%) | 109 (0.5%) | 0 (0.0%) | 1.00% | 8.03% |
| rqkmeans | 247 (1.1%) | 21098 (94.3%) | 885 (4.0%) | 122 (0.5%) | 11 (0.05%) | 1.10% | **8.45%** |
| rqkmeans-L4-lam1 | **258 (1.2%)** | 21135 (94.5%) | 837 (3.7%) | 116 (0.5%) | 17 (0.1%) | **1.15%** | **8.72%** |
| rqkmeans-L4-lam10 | 247 (1.1%) | 21168 (94.7%) | 820 (3.7%) | 107 (0.5%) | 21 (0.1%) | 1.10% | 8.64% |

---

## 表 2：条件正确率链（Error Propagation）

| Model | P(L1 ok) | P(L2\|L1) | P(L3\|L1+L2) | P(L4\|L1+L2+L3) |
|-------|:--------:|:---------:|:------------:|:----------------:|
| rqvae-baseline | **14.83%** | 14.53% | 48.96% | 100% [^1] |
| rqvae-collab | 7.17% | 20.70% | 67.17% | 100% [^1] |
| rqkmeans | 5.66% | 30.04% | 67.89% | 95.74% |
| rqkmeans-L4-lam1 | 5.49% | **31.84%** | 70.33% | 93.82% |
| rqkmeans-L4-lam10 | 5.34% | 31.38% | **71.47%** | 92.16% |

[^1]: rqvae-baseline / rqvae-collab 的 L4/pad 错误数为 0，但 EXACT > 0，因此 P(L4\|L1+L2+L3) = EXACT / (EXACT + 0) = 100%。这主要因为 post-hoc Sinkhorn 后 L3 已基本唯一化（每层 256 码字已覆盖 item 数），L4 没有实际分类负担。

[^2]: **`Top-20 Hit` vs 训练日志 `R@20`**：两者均使用 `model.generate()` beam search + SID 匹配，差异来自 **(1) beam size**（notebook=20，训练评测=30）和 **(2) 匹配粒度**（notebook 做 SID→item 映射去重，训练直接 SID 精确匹配）。Beam size 是主因：beam 越多搜索空间越大，beam=30 的前 20 条候选优于 beam=20 的前 20 条，因此训练 R@20 系统性偏高 0.15~0.30pp。详见 [评测方式说明](#评测方式说明)。

---

## 表 3：Top-20 Miss 样本的首次分叉位置

| Model | Miss 总数 | L1 错 | L2 错 | L3 错 | L4 错 |
|-------|:---------:|------:|------:|------:|------:|
| rqvae-baseline | 20543 (91.9%) | 18139 (88.3%) | 2280 (11.1%) | 124 (0.6%) | 0 (0.0%) |
| rqvae-collab | 20568 (92.0%) | 19567 (95.1%) | 952 (4.6%) | 49 (0.2%) | 0 (0.0%) |
| rqkmeans | 20474 (91.6%) | 19743 (96.4%) | 657 (3.2%) | 70 (0.3%) | 4 (0.02%) |
| rqkmeans-L4-lam1 | 20413 (91.3%) | 19775 (96.9%) | 562 (2.8%) | 68 (0.3%) | 8 (0.04%) |
| rqkmeans-L4-lam10 | 20431 (91.4%) | 19779 (96.8%) | 581 (2.8%) | 58 (0.3%) | 13 (0.06%) |

---

## 表 4：Beam Search 样本级增益

| Model | Top-1 Exact | Any-Beam Exact [^3] | Beam-path Exact (参考) |
|-------|:-----------:|:-------------------:|:----------------------:|
| rqvae-baseline | 1.06% | 8.14% | 0.41% |
| rqvae-collab | 1.00% | 8.03% | 0.40% |
| rqkmeans | 1.10% | 8.45% | 0.42% |
| rqkmeans-L4-lam1 | 1.15% | 8.72% | 0.44% |
| rqkmeans-L4-lam10 | 1.10% | 8.64% | 0.43% |

[^3]: **Any-Beam Exact** = 任意 beam（1~20）的 SID 四 token 与 target 完全一致（样本级，分母=N），等价于 item 级命中（已验证 SID→item 为单射，两者数值完全一致）。对比 Top-1 的 ~1%，beam search 将 exact match 率提升了 **~7-8×**。**Beam-path Exact** = 路径级 (count / N×20)，分母是样本级的 20 倍，仅保留参考。

---

## 表 5：Beam Survival — 逐层 Recall@K (K=20)

正确前缀出现在任意一条 beam（top-20）中的样本比例，以及逐层条件概率。

| Model | L1@20 | L1+L2@20 | L1+L2+L3@20 | Full@20 | L1→L1+L2 cond. | L1+L2→L1+L2+L3 cond. | L1+L2+L3→Full cond. |
|-------|:-----:|:--------:|:-----------:|:-------:|:--------------:|:--------------------:|:-------------------:|
| rqvae-baseline | **45.25%** | 13.02% | 8.17% | 8.14% | 28.77% | 62.77% | **99.56%** |
| rqvae-collab | 33.76% | 11.03% | 8.03% | 8.03% | 32.67% | 72.79% | 100% |
| rqkmeans | 29.76% | 11.78% | 8.97% | 8.45% | 39.58% | 76.12% | 94.21% |
| rqkmeans-L4-lam1 | 29.33% | **12.14%** | **9.28%** | **8.72%** | **41.40%** | 76.46% | 93.93% |
| rqkmeans-L4-lam10 | 29.24% | 12.18% | 9.24% | 8.64% | 41.66% | **75.88%** | 93.47% |

### 关键发现

1. **L1 不是唯一瓶颈——L1→L1+L2 的条件衰减才是**。以 rqkmeans 为例：K=20 时 29.76% 的样本能在一束 beam 中找到正确 L1，但其中只有 39.58% 同时有正确的 L2，即 L1+L2@20 只剩 11.78%。L1 的缺失（~70%）和 L1→L2 的条件丢失（~60%）共同限制了 Full@20。

2. **RQ-VAE 的 L1@20 更高（45% vs 29%），但 L2 条件概率更差（29% vs 40%）**。RQ-VAE 把更多正确 L1 放进了 beam，但 L2 跟不上；KMeans 的 L1@20 虽然低，但一旦 L1 对了，L2 也对的概率显著更高。

3. **Beam search 对 L1 覆盖改善最大，对 L2 条件存活也有帮助但增幅较小**：对比 top-1 → K=20，L1@K 提升 2~6×（5~15% → 29~45%），L1→L1+L2 条件概率提升 1.3~2×（RQ-VAE: 14.53%→28.77%; RQ-KMeans: 30.04%→39.58%）。Beam 能缓解搜索错误，但无法消除前两层模型预测能力不足的硬约束。

4. **一旦通过 L1+L2，L3 和 Full 的条件概率很高**（76~100%, 93~100%），瓶颈确实集中在 L1 和 L2。

---

## 评测方式说明

本文件 `Top-1 Exact` / `Top-20 Hit` 与从训练日志中汇总得到的`results/experiment_results.md` 中的 R@1 / R@20 **评测方式一致**：均使用 `model.generate()` beam search + SID 匹配。差异仅来自两点：

| | 本文件（first-diff notebook） | 训练日志（experiment_results.md） |
|---|---|---|
| **Beam size** | 20 | 30 |
| **匹配粒度** | SID → item 映射 + item 去重 | SID 精确匹配（`cur_pred == cur_label`） |
| **R@20 典型值** | 比训练日志低 0.15~0.30pp | — |

**Beam size 是差异主因**。beam search 每次只保留 top-k 路径并剪枝，beam 数越多搜索空间越大。即使只看 R@20（前 20 个候选），beam=30 产生的前 20 条序列与 beam=20 不完全相同——更多搜索路径意味着更高概率在 top-20 中命中 target。跨报告比较时需注意 beam size 不可直接对比。

---

## 总结

1. **早期前缀是主要瓶颈**：L1 覆盖与 L1→L2 条件衰减共同决定召回上限；不同 tokenizer 在两者之间呈现不同权衡（RQ-VAE 偏 L1 覆盖，KMeans 偏 L2 条件存活），最终 Full@20 接近（8.0~8.7%）。
2. **RQ-VAE codes (rqvae-baseline) 的 L1 单独最好 (14.83%)**，但 L2 条件正确率极低 (14.53%)，层层叠加后 exact match 反而最差。
3. **KMeans codes 把错误集中到 L1**：L1 正确率 ~5.5%，但后续层条件正确率远高于 RQ-VAE 系列（L2 ~31%, L3 ~70%, L4 ~94%），最终 exact match 略微领先。
4. **Collab loss 反而有负面效果**：rqvae-collab 的 L1 正确率从 14.83% 降至 7.17%，Top-20 hit 最低 (8.03%)。
5. **Structured L4 λ 调参效果极微**：λ=1 vs λ=10 各项指标差异在噪声范围内，L4 条件正确率甚至随 λ 增大而轻微下降。
6. **Beam search 有效，exact match 提升 ~7-8×**（表 4）：Top-1 exact 仅 ~1%，但 20 条 beam 中至少一条 exact 的比例达 ~8%。
7. **瓶颈不在单一层，而在 L1 覆盖率 × L1→L2 条件衰减的叠加**（表 5）：K=20 时 L1@20 = 29~45%，但 L1→L1+L2 条件概率仅 29~42%，两者相乘得到 L1+L2@20 = 11~13%。一旦通过 L1+L2，后续层条件概率 >76%。RQ-VAE L1@20 更高但 L2 条件更差，KMeans 相反，最终 Full@20 接近（8.0~8.7%）。
