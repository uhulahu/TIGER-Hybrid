# Artifact Index

> SID 文件、RQ-VAE checkpoint、训练日志和代码路径速查。

---

## SID 文件

### 当前有效（T5 嵌入，RQ-VAE / RQ-KMeans）

| 文件 | 日期 | Tokenizer | 用途 |
|---|---|---|---|
| `data/Beauty/Beauty_t5_rqvae_260629.npy` | Jun 29 | RQ-VAE, sk=[0,0,0.003] | 旧基准（Run 5, Run 24–26） |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003].npy` | Jul 9 | RQ-VAE, sk=[0,0,0.003] | **新基准**（E1） |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-inferExtraOnly.npy` | Jul 9 | RQ-VAE, TrainSK only | E2 |
| `data/Beauty/Beauty_t5_rqvae_260710-sk[0-0-0]-inferExtraOnly.npy` | Jul 10 | RQ-VAE, No SK | E3 |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-cw[0.001-tau0.2].npy` | Jul 9 | RQ-VAE + Collab | Collab 最优配置 |
| `data/Beauty/Beauty_kmeans_code.npy` | Jul 9 | RQ-KMeans | E4 基准 |
| `data/Beauty/Beauty_kmeans_code-L4-lam1.npy` | Jul 11 | RQ-KMeans + L4 λ=1 | **全局最优单路 SID** |
| `data/Beauty/Beauty_kmeans_code-L4-lam10.npy` | Jul 10 | RQ-KMeans + L4 λ=10 | L4 分析基准（l4_analysis.ipynb 的 K2） |

### RQ-VAE + Collab 超参扫描

| 文件 | cw | τ |
|---|---|---|
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-cw[0.001].npy` | 0.001 | (default) |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-cw[0.001-tau0.2].npy` | 0.001 | 0.2 |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-cw[0.001-tau0.5]-.npy` | 0.001 | 0.5 |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-cw[0.0001].npy` | 0.0001 | (default) |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-cw[0.0001-tau0.2]-.npy` | 0.0001 | 0.2 |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-cw[0.0001-tau0.5]-.npy` | 0.0001 | 0.5 |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-CF[0.001-tau1.0].npy` | 0.001 (CF) | 1.0 |

### RQ-VAE ± Sinkhorn + L4 变体

| 文件 | 日期 | 配置 |
|---|---|---|
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-inferExtraOnly-L4-lam1.npy` | Jul 9 | TrainSK, L4 λ=1 |
| `data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-inferExtraOnly-L4-lam10.npy` | Jul 9 | TrainSK, L4 λ=10 |
| `data/Beauty/Beauty_t5_rqvae_260710-sk[0-0-0]-inferExtraOnly-L4-lam1.npy` | Jul 10 | No SK, L4 λ=1 |
| `data/Beauty/Beauty_t5_rqvae_260710-sk[0-0-0]-inferExtraOnly-L4-lam10.npy` | Jul 10 | No SK, L4 λ=10 |
| `data/Beauty/Beauty_t5_rqvae_260709-L4structured-lambda1.npy` | Jul 9 | 旧格式 L4 λ=1 |
| `data/Beauty/Beauty_t5_rqvae_260709-L4structured-lambda10.npy` | Jul 9 | 旧格式 L4 λ=10 |

### 已废弃（Qwen 嵌入 / 旧 Sinkhorn）

| 文件 | 日期 | 备注 |
|---|---|---|
| `data/Beauty/Beauty_t5_rqvae.npy` | — | 最早版本 |
| `data/Beauty/Beauty_t5_rqvae_260629-[0.003-0.003-0.003].npy` | Jun 29 | Run 1（全层 Sinkhorn） |
| `data/Beauty/Beauty_t5_rqvae_260629-[0.003-0-0.003].npy` | Jun 29 | L1+L3 Sinkhorn |
| `data/Beauty/Beauty_t5_rqvae_260629-sk[0-0-0.003](new-eval).npy` | Jun 29 | 旧 SID 新评估 |
| `data/Beauty/Beauty_t5_rqvae_260708-sk[0-0-0]-*.npy` | Jul 8 | CoST/层内对比损失时期 |
| `data/Beauty/Beauty_qwen_rqvae_260706-*.npy` | Jul 6 | Qwen 嵌入系列（Run 9–12） |
| `data/Beauty/Beauty_t5_rqvae_260710_tmptmptmptmptmp.npy` | Jul 10 | 临时文件 |

### 其他数据文件

| 文件 | 说明 |
|---|---|
| `data/Beauty/item_emb.parquet` | T5 768-dim 内容嵌入 |
| `data/Beauty/item_emb_from_qwen3_0.6B.parquet` | Qwen 1024-dim 内容嵌入 |
| `data/Beauty/item2vec_emb.npy` / `item2vec_emb_gensim.npy` | 协同嵌入 (32-dim) |
| `data/Beauty/train.parquet` / `valid.parquet` / `test.parquet` | 数据集 |
| `data/Beauty/item_mapping.npy` / `user_mapping.npy` | ID 映射 |

---

## RQ-VAE Checkpoint

### 当前有效

| 路径 | 日期 | 配置 | 用途 |
|---|---|---|---|
| `rqvae/ckpt/Beauty/Jun-29-2026_17-38-17/` | Jun 29 | sk=[0,0,0.003], T5 | 旧基准（被 Run 5 和 Jul 9 之前的实验使用） |
| `rqvae/ckpt/Beauty/Jul-09-2026_02-06-03/` | Jul 9 | sk=[0,0,0.003], T5 | **新基准**（E1/E2 tokenizer，含 sid_compare.md） |
| `rqvae/ckpt/Beauty/Jul-09-2026_03-54-29/` | Jul 9 | sk=[0,0,0.003], cw=0.001, τ=0.2 | Collab 最优 |
| `rqvae/ckpt/Beauty/Jul-10-2026_22-18-01/` | Jul 10 | sk=[0,0,0], T5 | No SK RQ-VAE |

### Collab 超参扫描

| 路径 | cw | τ |
|---|---|---|
| `rqvae/ckpt/Beauty/Jul-09-2026_02-55-18/` | 0.001 | (early crash?) |
| `rqvae/ckpt/Beauty/Jul-09-2026_02-59-38/` | 0.001 | 1.0 |
| `rqvae/ckpt/Beauty/Jul-09-2026_03-24-46/` | 0.001 | 0.5 |
| `rqvae/ckpt/Beauty/Jul-09-2026_03-54-38/` | 0.0001 | 1.0 |
| `rqvae/ckpt/Beauty/Jul-09-2026_04-19-49/` | 0.0001 | 0.2 |
| `rqvae/ckpt/Beauty/Jul-09-2026_04-20-09/` | 0.0001 | 0.5 |

### CoST / 层内对比损失时期（已废弃）

| 路径 | 碰撞率 | 备注 |
|---|---|---|
| `rqvae/ckpt/Beauty/Jul-08-2026_01-56-06/` | ~0.11 | sk=[0,0,0], L3 CL w=0.1, τ=0.5 |
| `rqvae/ckpt/Beauty/Jul-08-2026_02-01-40/` | ~0.11 | 同上（续） |
| `rqvae/ckpt/Beauty/Jul-08-2026_02-03-36/` | ~0.09 | sk=[0,0,0.003], L3 CL w=0.01, τ=0.5 |
| `rqvae/ckpt/Beauty/Jul-08-2026_02-07-10/` | ~0.08 | 同上 |
| `rqvae/ckpt/Beauty/Jul-08-2026_02-11-40/` | ~0.07 | sk=[0,0,0.003], L2 CL |
| `rqvae/ckpt/Beauty/Jul-08-2026_02-15-43/` | ~0.06 | sk=[0,0,0.003], L2 CL |
| `rqvae/ckpt/Beauty/Jul-08-2026_22-14-07/` | ~0.39 | CoST 对比损失（极高） |
| `rqvae/ckpt/Beauty/Jul-08-2026_23-37-48/` | ~0.49 | CoST 对比损失（极高） |

### Qwen 嵌入时期（已废弃）

| 路径 | 日期 | 配置 |
|---|---|---|
| `rqvae/ckpt/Beauty/Jul-06-2026_11-52-31/` | Jul 6 | sk=[0,0,0.003], 无适配层 |
| `rqvae/ckpt/Beauty/Jul-06-2026_17-18-08/` | Jul 6 | sk=[0,0,0.003], add768, quant_loss=2.0 |

### 其他

| 路径 | 备注 |
|---|---|
| `rqvae/ckpt/Beauty/Jun-17-2025_15-21-52/` | 最早版本（2025 年） |
| `rqvae/ckpt/Beauty/Jun-29-2026_19-03-07/` | sk=[0.003,0,0.003] |
| `rqvae/ckpt/Beauty/Jun-29-2026_20-09-18/` | sk=[0.003,0.003,0.003] |
| `rqvae/ckpt/Beauty/Jul-08-2026_22-30-28/` | sk=[0,0,0.003], 含 sid_evaluation.txt |
| `rqvae/ckpt/Beauty/Jul-09-2026_18-08-36/` | LETTER 多样性损失 |

---

## 训练日志

| 日志文件 | 对应实验 | 状态 |
|---|---|---|
| `model/logs/tiger.log` | Run 1–20 | 早期探索，多 Run 共用 |
| `model/logs/tiger-0708.log` | Run 21–26 | 过渡期 |
| `model/logs/tiger_baseline_ce.log` | E1 | H |
| `model/logs/tiger_baseline_fd.log` | E1+FD | H |
| `model/logs/tiger_collab_ce.log` | E1+Collab | H |
| `model/logs/tiger_collab_fd.log` | E1+Collab+FD | H |
| `model/logs/tiger_baseline_ce_inferExtraOnly.log` | E2 | H |
| `model/logs/tiger_rqvae_noSKinTrain_inferExtraOnly.log` | E3 | H |
| `model/logs/tiger_baseline_ce_rqkmeans.log` | E4 | H |
| `model/logs/tiger_baseline_fd_rqkmeans.log` | E4+FD | H |
| `model/logs/tiger_kmeans_L4_lam1.log` | L4-λ1 | H |
| `model/logs/tiger_kmeans_L4_lam10.log` | RQ-KMeans + L4 λ=10 | H |
| `model/logs/tiger_kmeans_L4_lam10_fd.log` | RQ-KMeans + L4 λ=10 + FD | H |
| `model/logs/tiger_rqvae_inferExtraOnly_L4_lam10.log` | RQ-VAE + L4 λ=10 | H |
| `model/logs/tiger_rqvae_noSKinTrain_inferExtraOnly_L4_lam10.log` | RQ-VAE NoSK + L4 λ=10 | H |
| `model/logs/_test_l4.log` | Smoke test | 忽略 |

---

## TIGER Checkpoint

| 路径 | 对应实验 |
|---|---|
| `model/ckpt/Beauty_baseline_ce/` | E1 |
| `model/ckpt/Beauty_collab_ce/` | E1+Collab |
| `model/ckpt/Beauty_baseline_fd/` | E1+FD |
| `model/ckpt/Beauty_collab_fd/` | E1+Collab+FD |
| `model/ckpt/Beauty_baseline_ce_rqkmeans/` | E4 |
| `model/ckpt/Beauty_kmeans_L4_lam1/` | L4-λ1 |
| `model/ckpt/Beauty_kmeans_L4_lam10/` | RQ-KMeans + L4 λ=10 |
| `model/ckpt/Beauty_kmeans_L4_lam10_fd/` | RQ-KMeans + L4 λ=10 + FD |
| `model/ckpt/Beauty_rqvae_inferExtraOnly_L4_lam10/` | RQ-VAE + L4 λ=10 |
| `model/ckpt/Beauty_rqvae_noSKinTrain_inferExtraOnly/` | E3 |
| `model/ckpt/Beauty_rqvae_noSKinTrain_inferExtraOnly_L4_lam10/` | RQ-VAE NoSK + L4 λ=10 |
| `model/ckpt/Beauty/` | 早期 Run（共用 tiger.pth） |
| `model/ckpt/_test_l4/` | Smoke test |
| `model/ckpt/test/` | 测试 |

---

## 分析 Notebook

| 文件 | 内容 |
|---|---|
| `analysis/l4_analysis.ipynb` | Structured L4 深度分析（K0 vs K2） |
| `analysis/analysis.ipynb` | TIGER(rqvae) vs Content 互补性 |
| `analysis/analysis_rqkmeans.ipynb` | TIGER(rqkmeans) vs Content 互补性 |
| `analysis/analysis_rqkmeans_L4_lam1.ipynb` | TIGER(rqkmeans-L4-λ1) vs Content 互补性 |
| `analysis/fusion.ipynb` | 融合策略搜索 (rqvae) |
| `analysis/fusion_rqkmeans.ipynb` | 融合策略搜索 (rqkmeans) |
| `analysis/fusion_rqkmeans_L4_lam1.ipynb` | 融合策略搜索 (rqkmeans-L4-λ1) |
| `analysis/first_diff_distribution.ipynb` | 首次分叉分析 (rqvae-baseline) |
| `analysis/first_diff_distribution_collab.ipynb` | 首次分叉分析 (rqvae-collab) |
| `analysis/first_diff_distribution_rqkmeans.ipynb` | 首次分叉分析 (rqkmeans) |
| `analysis/first_diff_distribution_rqkmeans_L4_lam1.ipynb` | 首次分叉分析 (rqkmeans-L4-λ1) |
| `analysis/first_diff_distribution_rqkmeans_L4_lam10.ipynb` | 首次分叉分析 (rqkmeans-L4-λ10) |
| `analysis/inference.py` | 推理脚本 |
| `analysis/l4_analysis.py` | L4 分析辅助脚本 |

## 脚本

| 文件 | 用途 |
|---|---|
| `eval_sid.py` | SID 统计评估（利用率、条件熵） |
| `model/analyze_prefix_hit.py` | Extra token 瓶颈分析 |
| `item2vec/retrieve.py` | Content 相似度检索召回 |
