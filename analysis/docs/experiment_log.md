# Experiment Registry

> 快速检索所有实验。不承担完整结果分析和研究叙事——分析见 `experiment_results.md` 及各诊断文档，研究思路演进见 `research_journal.md`。

## 图例

**可信度**：
- **H**：规范化单变量对照，日志和 checkpoint 完整（注意：H 不等于多随机种子验证）
- **M**：配置基本明确，但存在少量未控制变量或跨 SID 比较
- **L**：早期探索、未充分收敛、配置不完整或仅凭回忆

**状态**：
- ✅ Complete — 完整运行并评估
- ❌ Failed — 运行完成但结果无效或明确差于基线
- 🐛 Bug — 实现或评测存在已知错误
- ⏸️ Incomplete — 未完成
- ❓ Unknown — 无法确认具体配置

---

## 关键决策时间线

| 时间 | 决策转折 | 依据 | 后续动作 |
|---|---|---|---|
| Jun 29–Jul 3 | 从全层 Sinkhorn 转为仅 L3 Sinkhorn | 浅层强均衡化未稳定改善下游 | 保留 `sk=[0,0,0.003]` 为 RQ-VAE 基线 |
| Jul 6–8 | 放弃 Qwen、listwise、FD、CoST | 训练慢或下游无稳定收益；listwise 有 Bug | 回归 T5 内容嵌入 + 纯 CE |
| Jul 9 | 引入 RQ-KMeans | RQ-KMeans 全面优于 RQ-VAE (R@20 +5.5%) | 以 RQ-KMeans 为主要 tokenizer |
| Jul 9–10 | 区分 Train-SK 与 Infer-SK | post-hoc Sinkhorn 仅搬运身份信息，未减少总不确定性 | 导出时使用自然 L3 + extra token |
| Jul 10–11 | 设计 Structured L4 替代 extra token | Extra token 无跨桶结构，L4 带来边际正收益 | 保留 L4 λ=1 为最优单路配置 |
| Jul 11–13 | 完成逐层瓶颈诊断 | 错误集中于 L1/L2；FD 与 CE 梯度方向一致 | 停止继续修改局部 loss |
| Jul 12–13 | 引入 Content 路和融合 | 两路候选与命中高度互补，融合 R@20 +29~35% | 项目最终方向转向多路召回 |

---

## 实验注册表

### 早期探索 (Run 1–26)

> 设备：3090 (Run 1–23) → 4090 (Run 24–26)。SID 均为 RQ-VAE 生成。日志：`tiger.log` / `tiger-0708.log`，checkpoint 共用 `tiger.pth`（不可复现）。

| ID | 日期 | 目的 | 关键改动 | 当时观察 | 当时决策 | 可信度 |
|---|---|---|---|---|---|---|
| Run 1 | Jul 2 | 全层 Sinkhorn | `sk=[.003,.003,.003]` | L1 util≈100%，早期 test 指标低（未充分收敛） | 怀疑过强均衡化损害浅层语义 | L |
| Run 2 | Jul 2 | 仅 L3 Sinkhorn | `sk=[0,0,.003]` | L1 util≈23%，测试指标好于 Run 1 | 确定 L3-only 为最优 Sinkhorn 配置 | L |
| Run 3 | Jul 2 | 高 lr | lr=0.01 + inv_sqrt | 极差 | 高 lr 不可行 | L |
| Run 4 | Jul 2 | 中 lr | lr=0.001 + inv_sqrt | 极差 | 中 lr 也不可行 | L |
| Run 5 | Jul 2 | 建立 T5 基线 | lr=1e-4, wd=1e-3, inv_sqrt | 比 Run 2 略强 | 作为后续对比基准 | M |
| Run 6 | Jul 3 | user token | `use_user_token=True` | ❓ 结果未记录 | 放弃 | L |
| Run 8 | Jul 3 | L2+L3 Sinkhorn | `sk=[0,.003,.003]` | 20+ epoch 早停，指标很差 | L2 Sinkhorn 连带破坏 L1 | L |
| Run 9 | Jul 6 | Qwen 嵌入（无适配层） | Qwen 1024-dim, 无 dim 适配 | 差于 Run 5 | 需加适配层 | L |
| Run 11 | Jul 6 | Qwen 嵌入 + 适配层 | `layers=[768,512,256,128,64]`, quant_loss=2.0 | 好于 Run 9，差于 Run 5 | Qwen 嵌入路线暂时保留 | L |
| Run 12 | Jul 6 | RQ-VAE 推理用 beam search | beam=10 分配 SID | 差于 Run 11 | 束搜索分配 SID 不带来收益 | L |
| Run 13 | Jul 7 | listwise loss | 4种负采样×4=16负样本 | Val 全 0 | 🐛 有 Bug | L |
| Run 14–15 | Jul 7 | listwise（全部负采样策略） | 同 Run 13 | 训练极慢 | ⏸️ 不可行，放弃 | L |
| Run 16–17 | Jul 7 | listwise（仅 prefix negatives） | 4 个 prefix negatives | 仍太慢，无法与 CE 共享 forward | ⏸️ 架构不可行 | L |
| Run 18 | Jul 7 | first-diff loss | FD + 4 prefix negatives | ⏸️ 未跑完 | 原因已忘记 | L |
| Run 19 | Jul 7 | ❓ | ❓ | ❓ | ❓ | L |
| Run 20 | Jul 8 | FD loss + 混合负采样 | FD w=0.5, 1r+1p+1L1+1L12 | 差于 Run 11 | FD 在 Qwen SID 上无收益 | L |
| Run 21 | Jul 8 | CoST 对比损失 | RQ-VAE 加 InfoNCE, TIGER 用 FD w=0.5 | CoST 碰撞率爆炸(~0.3-0.5), test R@20~0.06 | CoST 破坏码本结构 | L |
| Run 22 | Jul 8 | 回归 T5，测 FD w=0.5 | 旧 SID (Jun 29), FD w=0.5 | R@20 ~0.044 | FD 权重过大有害 | L |
| Run 23 | Jul 8 | ❓ | ❓ | ❓ | 已忘记 | L |
| Run 24 | Jul 8 | 纯 CE 基线 (4090) | 旧 SID (Jun 29), 纯 CE, 上 4090 | R@20=0.0804, 收敛极快 (epoch 63) | 确定为旧 SID 的 CE 基线 | M |
| Run 25 | Jul 8 | FD w=1.0 | 旧 SID, CE+FD w=1.0 | 差于 Run 24 | FD 权重需 <1.0 | M |
| Run 26 | Jul 9 | FD w=0.1 | 旧 SID, CE+FD w=0.1 | R@20=0.0858 (+0.0054 vs Run 24) | ⚠️ 正向，但新 SID 上未复现 | M |

### 规范化对照 (E1–E4, L4 系列)

> 全部使用：beam=30, lr=1e-4, no scheduler, no wd, 4090。RQ-VAE SID 基于 `rqvae/ckpt/Beauty/Jul-09-2026_02-06-03/`。
> 完整指标见 `experiment_results.md`。

| ID | 日期 | Tokenizer | Train SK | Infer SK | Collab | 4th Token | Loss | Test R@20 | Test N@20 | 可信度 |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Jul 9 | RQ-VAE | L3 | L3 | — | Extra | CE | 0.0829 | 0.0365 | H |
| E1+Collab | Jul 9 | RQ-VAE | L3 | L3 | ✓ | Extra | CE | 0.0820 | 0.0356 | H |
| E1+FD | Jul 9 | RQ-VAE | L3 | L3 | — | Extra | CE+FD w=0.1 | 0.0782 | 0.0346 | H |
| E1+Collab+FD | Jul 9 | RQ-VAE | L3 | L3 | ✓ | Extra | CE+FD w=0.1 | 0.0838 | 0.0361 | H |
| E2 | Jul 10 | RQ-VAE | L3 | — | — | Extra | CE | 0.0826 | 0.0366 | H |
| E3 | Jul 11 | RQ-VAE | — | — | — | Extra | CE | 0.0795 | 0.0347 | H |
| E4 | Jul 9 | RQ-KMeans | — | — | — | Extra | CE | 0.0875 | 0.0379 | H |
| E4+FD | Jul 11 | RQ-KMeans | — | — | — | Extra | CE+FD w=0.1 | 0.0811 | — | H |
| — | Jul 11 | RQ-VAE (TrainSK, no InferSK) | L3 | — | — | L4 λ=10 | CE | 0.0806 | 0.0354 | H |
| — | Jul 11 | RQ-VAE (No SK) | — | — | — | L4 λ=10 | CE | 0.0811 | — | H |
| — | Jul 10 | RQ-KMeans | — | — | — | L4 λ=10 | CE | 0.0887 | 0.0386 | H |
| **L4-λ1** | Jul 11 | RQ-KMeans | — | — | — | **L4 λ=1** | CE | **0.0890** | **0.0390** | H |
| — | Jul 11 | RQ-KMeans | — | — | — | L4 λ=10 | CE+FD w=0.1 | 0.0870 | 0.0378 | H |

> **日志路径**：`model/logs/tiger_{variant}.log`（variant = `baseline_ce`, `baseline_fd`, `collab_ce`, `collab_fd`, `baseline_ce_rqkmeans`, `baseline_fd_rqkmeans`, `baseline_ce_inferExtraOnly`, `kmeans_L4_lam1`, `kmeans_L4_lam10`, `kmeans_L4_lam10_fd`, `rqvae_inferExtraOnly_L4_lam10`, `rqvae_noSKinTrain_inferExtraOnly`, `rqvae_noSKinTrain_inferExtraOnly_L4_lam10`）
> **Checkpoint 路径**：`model/ckpt/{variant}/`（variant 同上，用 `_` 替换日志名中的 `tiger_` 前缀和 `.log` 后缀）

### 诊断分析

| 分析 | 来源 | 产出的文档 |
|---|---|---|
| Top-1 首次分叉 + Beam Survival（5 个 tokenizer） | `analysis/first_diff_distribution*.ipynb` | `first_diff_summary.md` |
| Structured L4 深度分析（K0 vs K2 λ=10） | `analysis/l4_analysis.ipynb` | `l4_analysis_summary.md` |
| TIGER vs Content 互补性 + 四维分桶（3 个 TIGER 变体） | `analysis/analysis*.ipynb` | `overlap_analysis_summary.md` |
| 多路融合策略搜索（3 个 TIGER 变体） | `analysis/fusion*.ipynb` | `fusion_summary.md` |
| Extra token 瓶颈（prefix-hit） | `model/analyze_prefix_hit.py` | 结果在 `first_diff_summary.md` 中引用 |
| SID 统计评估（利用率、条件熵） | `eval_sid.py` | 结果在 `experiment_results.md` 中引用 |

---

## 未解问题

| 问题 | 可能影响 | 是否追查 |
|---|---|---|
| 旧 SID (Jun 29) 上 FD w=0.1 正向 (+0.0054)，新 SID (Jul 9) 上反向 (−0.0047) | 影响 FD 结论的稳定性 | 否，项目已收尾；FD 在新 SID 和 RQ-KMeans 上均无效 |
| `Beauty_t5_rqvae_260629.npy` 与 `Beauty_t5_rqvae_260709-sk[0-0-0.003].npy` 统计不同，原因不明 | 说明代码或随机性存在漂移，旧 SID 的实验无法与新 SID 直接对比 | 记录，不追查 |
| Run 19 内容遗忘 | 丢失一条早期实验 | 否 |
| 旧 TIGER checkpoint (Run 1–26) 共用 `tiger.pth` 被覆盖 | 早期实验不可复现 | 后续项目改进日志规范即可 |
| 3090 上训练速度突然从 32s/epoch 变为 50+s/epoch (Run 24 前后) | 原因不明，最终换 4090 绕过 | 否 |
| quant_loss_weight 从 2.0 改回 1.0 的时间未记录 | 影响 Run 11 与后续的精确对比 | 否 |
