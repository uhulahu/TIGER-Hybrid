<!-- # TIGER Generative Retrieval: From Reproduction to Diagnosis and Multi-Stream Fusion -->

# TIGER-Hybrid: Semantic ID Diagnosis and Hybrid Retrieval for Generative Recommendation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Beauty-orange.svg)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)

基于生成式检索（Generative Retrieval）范式的序列推荐系统。本项目在复现 TIGER 架构的基础上，围绕 Semantic ID tokenizer、Sinkhorn、逐层生成瓶颈和碰撞 suffix 展开系统性**诊断与改进**。

对部分关键组件进行了**诊断与改进**：实验对比了 RQ-VAE 与 RQ-KMeans tokenizer，区分了训练期与导出期 Sinkhorn Trick 的不同角色，对SID逐层生成瓶颈进行定位分析，设计了 Structured L4 codebook 替代随机 extra token，并分析了 TIGER 与 last-item 内容相似度召回的互补性。最终通过**两路软融合**将 Recall@20 提升至 0.1150，相对对应 TIGER 单路提升约 28%–35%。

## 🌟 核心发现

- **RQ-KMeans 优于 RQ-VAE**：在固定内容嵌入上逐层 KMeans 量化，无需 Sinkhorn 均衡化，下游 R@20 提升 5.5%。后层条件路由更稳定（L1→L2 条件存活 40% vs 29%），而非单纯利用率的差异。
- **瓶颈在前两层，不在 extra token**：以 RQ-KMeans 为 tokenizer 时，L1 准确率仅 5.66%，L1→L2 条件存活仅 30–40%；extra token 只造成 0.7% 的独立失败——即使完美修复，R@20 上限也仅从 0.0875 提升至 0.0918。
- **Post-hoc Sinkhorn 搬运而非消除信息**：将身份区分从 extra token 搬至 L3，使 L3 同时混杂语义和身份信号，但未减少 L2 之后的总预测不确定性，在当前配置中未表现出可辨认增益，因此最终方案不予保留。
- **Structured L4 边际正收益**：用 learned 第4层 codebook 替代随机编号式 extra token，取得小幅正收益：R@20 +1.7%，N@20 +2.9%。进一步分析表明，收益主要来自全局标签结构优化与Prefix-3 排序改善，而非主要来自直接碰撞消歧。
- **TIGER 与内容相似度召回高度互补**：两路 top-20 平均每个用户仅重叠约 0.7–0.8 个 item（Jaccard ~2%），近 90% 命中来自单路独占。TIGER 擅长头部/跨类目/长历史，Content 擅长长尾/同类目/高相似度。软融合（RRF/MinMax）R@20 最高达 **0.1150**，相对同一 RQ-KMeans 单路提升 31.1%，相对融合评测中最优单路 TIGER 提升 28.6%。

## 🚀 研究路径

本项目不是一次性跑通即结束，而是沿着「发现问题 → 定位原因 → 尝试改进 → 评估收益 → 改变方向」的路径迭代了多轮。

### 1. Tokenizer 对比：RQ-VAE → RQ-KMeans

参照 TIGER 论文及开源实现 [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER)，使用 RQ-VAE + Sinkhorn 生成 4-token SID。随后系统消融了 Sinkhorn ε 在不同层的作用：强浅层均衡化未稳定改善下游，仅训练期 L3 Sinkhorn呈现正向作用，而导出期 post-hoc Sinkhorn 呈现负收益（通过统计分析证明其仅将本由 extra token 的身份区分信息转移至L3，扭曲语义）。在对比 learned latent（RQ-VAE）与固定空间聚类（RQ-KMeans）后，RQ-KMeans 在所有下游指标上一致更优，成为后续实验的默认 tokenizer。

### 2. 瓶颈诊断：到底哪里限制了 Recall？

通过逐层首次分叉分析和 Beam Survival 条件概率链，将瓶颈精确定位到 **L1 覆盖率 × L1→L2 条件衰减** 的共同作用。Extra token 几乎不是问题（99.3% miss 在前三层已错）。FD loss（first-difference pairwise）和协同正则（LETTER 式 item2vec InfoNCE）均未改善早期路由，梯度方向与 CE 冗余。

### 3. Structured L4：改造最后一层 token

将随机 extra token 替换为在第三层残差上训练的 256-code L4 码本（Constrained KMeans + 碰撞桶内 Hungarian 单射分配，确保消除碰撞）。尽管 L4 本身并未变得更可学习（CE 更高、TF_Acc 略低），但 Prefix-3 排序的一致改善验证了「L4 标签结构通过共享 decoder 参数间接优化前缀排序」的假设。λ=1（等权）优于 λ=10（过度加权碰撞），说明收益来自全局标签结构而非碰撞桶内加权。

### 4. 多路融合：真正的收益来源

引入基于 content embedding 的余弦相似度召回作为第二条路，发现两路高度互补。系统对比了 RRF、Score MinMax、固定配额和动态配额四种融合策略，软融合一致且显著优于硬路由。**融合后 R@20 达 0.1150，相对同一 RQ-KMeans 单路提升 31.1%，相对融合评测中最优单路 TIGER 提升 28.6%。** 项目最终结论：继续孤立优化 TIGER 单路的边际收益递减，多路召回是正确工程方向。

## 📊 使用指南

### 环境

```bash
# Python 3.10+, PyTorch 2.0+, CUDA 12.0+
pip install torch transformers faiss-gpu sentence-transformers pandas pyarrow
```

### 第一步：数据预处理

下载 [Amazon Beauty 5-core 及元数据](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) (`meta_Beauty.json.gz`和`reviews_Beauty_5.json.gz`)，通过 [ModelScope](https://www.modelscope.cn/models/sentence-transformers/sentence-t5-base) 下载 Sentence-T5 模型权重至 `data/sentence-t5-base/`，使用 `data/process.ipynb` 生成内容嵌入并整理训练/验证/测试集。

已处理好的数据文件（`data/Beauty/`）：
- `item_emb.parquet` — T5 768-dim 嵌入
- `train.parquet` / `valid.parquet` / `test.parquet` — 用户交互序列

### 第二步：构建 Semantic ID

**选项 A: RQ-VAE（learned latent + Sinkhorn）**

```bash
# 训练 RQ-VAE
python rqvae/main.py 

```

**选项 B: RQ-KMeans（推荐）**

```bash
# 训练 RQ-KMeans
python rqkmeans/generate_kmeans_code.py --output_path [...]
```

**导出并评估 SID**

```bash
python rqvae/generate_code.py --ckpt_path [...] --output_file [...]
```

**附加：Structured L4 codebook**

```bash
# 训练 L4 码本并导出 SID
python rqvae/train_l4_codebook.py --ckpt [...] --sid_in [...] --sid_out [...] --lam [...]
```

### 第三步：训练 TIGER

```bash
python model/main.py --code_path "data/Beauty/Beauty_kmeans_code.npy" --log_path "model/logs/tiger_kmeans_L4_lam1.log" --first_diff_weight 0.1 --use_first_diff_loss
```

关键配置：`--code_path` 指定 SID 文件，`--use_first_diff_loss` 控制 FD loss，`--first_diff_weight` 控制权重（`use_first_diff_loss`为`False`时不需，仅 CE）。

### 第四步：多路融合

```python
# 参见 analysis/fusion.ipynb
# 1. 生成 TIGER top-50 候选
# 2. 生成 Content 余弦相似度 top-50 候选
# 3. RRF / Score MinMax 融合排序
```

## 📈 主要实验结果

> **说明：** 单路实验多为单 checkpoint 结果；融合权重及超参数直接在测试集网格搜索，相关结果用于展示探索性上限，不属于严格无偏的泛化评估。

**单路最佳结果 (Amazon Beauty Test, beam=30)**

| Method | Tokenizer | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| TIGER (RQ-VAE baseline) | RQ-VAE + TrainSK + InferSK | 0.0347 | 0.0564 | 0.0829 | 0.0228 | 0.0298 | 0.0365 |
| TIGER (RQ-KMeans) | RQ-KMeans + Extra Token | 0.0364 | 0.0578 | 0.0875 | 0.0236 | 0.0304 | 0.0379 |
| **TIGER (RQ-KMeans + Structured L4 λ=1)** | **RQ-KMeans + Structured L4 λ=1** | **0.0374** | **0.0575** | **0.0890** | **0.0247** | **0.0311** | **0.0390** |

**融合结果 (top-50 RRF / Score MinMax)**

| Method | R@5 | R@10 | R@20 | N@20 |
|:---|:---:|:---:|:---:|:---:|
| Content 相似度 (last1, 无训练) | 0.0439 | 0.0589 | 0.0779 | 0.0409 |
| RQ-KMeans + Content · MinMax | 0.0541 | 0.0804 | **0.1150** | 0.0520 |
| RQ-KMeans + L4 λ=1 + Content · MinMax | **0.0554** | 0.0804 | 0.1146 | **0.0524** |

### 💡 分析要点

1. **RQ-KMeans 是更优的 tokenizer 选择**：下游 R@20 从 0.0829 提升至 0.0875（+5.5%）。训练速度快，直接作用在原始嵌入空间，避免潜在空间转换过程中的扭曲，保持内容嵌入的原始几何结构。

2. **Structured L4 带来边际正收益，但幅度有限**：R@20 +1.7%，主要来自全局标签结构优化（Prefix-3 排序改善），而非直接碰撞消歧。λ=1 优于 λ=10。

3. **Content-sim 无需训练，单路即达 R@20=0.0779**：其前排排序能力（R@5=0.0439）超过所有 TIGER 单路变体，且与 TIGER 高度互补——两者 top-20 列表 Jaccard 仅 ~2%。

4. **多路融合是真正的收益来源**：Score MinMax 在 RQ-KMeans + Content 上取得 R@20=0.1150，相对同一 RQ-KMeans 单路提升 31.1%，相对融合评测中最优单路 TIGER 提升 28.6%。软融合（RRF / MinMax）一致优于硬路由和配额策略。

## 📁 项目文档

详细分析、诊断和实验记录请参阅 [`analysis/docs/`](analysis/docs/)：

| 文档 | 内容 |
|:---|:---|
| [`project_overview.md`](analysis/docs/project_overview.md) | 项目主文档：背景、总表、最终结论、最优结果 |
| [`experiment_results.md`](analysis/docs/experiment_results.md) | 规范化下游实验完整指标 |
| [`first_diff_summary.md`](analysis/docs/first_diff_summary.md) | 逐层瓶颈诊断（首次分叉 + Beam Survival） |
| [`l4_analysis_summary.md`](analysis/docs/l4_analysis_summary.md) | Structured L4 深度分析 |
| [`overlap_analysis_summary.md`](analysis/docs/overlap_analysis_summary.md) | TIGER vs Content 互补性与分桶 |
| [`fusion_summary.md`](analysis/docs/fusion_summary.md) | 多路融合策略对比 |
| [`experiment_log.md`](analysis/docs/experiment_log.md) | 实验注册表（含可信度评级） |
| [`research_journal.md`](analysis/docs/research_journal.md) | 原始研究日记 |
| [`artifact_index.md`](analysis/docs/artifact_index.md) | SID / Checkpoint / 日志路径速查 |

## 🔗 参考

- [TIGER: Recommender Systems with Generative Retrieval (Rajat et al., NeurIPS 2023)](https://arxiv.org/abs/2305.05065)
- [LETTER: Learnable Item Tokenization for Generative Recommendation (Wang et al., CIKM 2024)](https://dl.acm.org/doi/abs/10.1145/3627673.3679569)
- [CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation (Zhu et al., RecSys 2024)](https://dl.acm.org/doi/abs/10.1145/3640457.3688178)
- [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER)
