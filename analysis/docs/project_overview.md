# TIGER 生成式召回复现、Semantic ID 诊断与多路融合

> Amazon Beauty 数据集上的生成式检索（Generative Retrieval）：将推荐建模为语义 ID 的自回归生成，逐 token 预测 item 的离散化表示，beam search 解码后映射回 item 完成召回。

---

## 1. 背景与目标

TIGER（Transformer Index for GEnerative Retrieval）使用 RQ-VAE 将 item 内容嵌入量化为 4-token 语义 ID（SID），再由 T5 decoder 自回归生成。相比传统双塔召回，生成式检索通过自回归生成 SID 直接产生候选，无需在查询阶段对全量 item 逐一打分，推理复杂度对 catalog size 的直接依赖较弱。

本项目以 **Amazon Beauty**（12,101 items, 22,363 users）为基准，系统探索以下问题：

- RQ-VAE（learned latent + Sinkhorn）与 RQ-KMeans（固定内容空间逐层聚类）哪种 tokenizer 更适合下游 T5？
- Sinkhorn 均衡化在训练期和导出期分别扮演什么角色？
- 生成式检索的瓶颈在哪一层？Extra token 是否拖累性能？
- Structured L4 codebook 能否替代随机 extra token 并改善前缀排序？
- 单路 TIGER 与内容相似度召回是否互补？融合能带来多大收益？

**最终目标**：确定最优单路配置，并评估多路融合作为工程方向的上限。

---

## 2. 系统流程

```
Item 内容嵌入 (T5/Qwen, 768/1024-dim)
     │
     ▼
┌─────────────────────────────────┐
│  RQ-VAE / RQ-KMeans 量化        │
│  三层残差量化 (256³)             │
│  ± Train Sinkhorn (L3 only)     │
│  ± Post-hoc Sinkhorn            │
│  ± Collab 协同正则               │
└─────────────────────────────────┘
     │  3-token prefix
     ▼
┌─────────────────────────────────┐
│  第四层 token                    │
│  K0: random extra token         │
│  K2: Structured L4 codebook     │
│  (constrained KMeans + Hungarian)│
└─────────────────────────────────┘
     │  4-token SID [l1, l2, l3, l4]
     ▼
┌─────────────────────────────────┐
│  T5 Encoder-Decoder             │
│  (4-layer, d=128)               │
│  Encoder 编码历史，              │
│  Decoder 自回归生成 SID          │
│  CE loss (± FD pairwise)        │
└─────────────────────────────────┘
     │  beam search (beam=30)
     ▼
┌─────────────────────────────────┐
│  SID → Item 映射                 │
│  top-K 候选去重                   │
└─────────────────────────────────┘
     │                    ┌──────────────────────┐
     ▼                    │  Content Similarity  │
┌──────────┐              │  last1 × item_emb    │
│  TIGER   │              │  cosine top-K        │
│  top-50  │              │  top-50              │
└──────────┘              └──────────────────────┘
     │                    │
     └────────┬───────────┘
              ▼
     ┌─────────────────┐
     │  RRF / MinMax   │
     │  融合排序         │
     │  final top-K    │
     └─────────────────┘
```

---

## 3. 项目总表

**统一说明**：除特别说明外，实验均基于 Amazon Beauty 测试集（N=22,363），多数结果来自单个 checkpoint，尚未进行多随机种子验证。不同诊断实验使用 beam size 20 或 30，跨文档数值仅在评测协议一致时直接比较。融合超参数在测试集上网格搜索，因此融合结果属于探索性上限，不作为严格无偏泛化估计。

| 研究问题 | 核心对照或分析 | 关键结果 | 最终结论 | 证据位置 |
|---|---|---|---|---|
| **RQ-VAE 与 RQ-KMeans 谁更适合 TIGER？** | E1 (RQ-VAE + TrainSK L3 + InferSK L3) vs E4 (RQ-KMeans) | RQ-KMeans R@20 0.0875 vs RQ-VAE 0.0829，相对提升 5.5% | 在当前 Beauty 配置下，直接在固定内容嵌入空间中进行逐层 RQ-KMeans，比 learned latent + Sinkhorn 的 RQ-VAE 获得更好的下游召回；其优势表现为 Beam@20 下更高的 L1→L1+L2 条件存活率（39.6% vs 28.8%），而非更高的码本利用率本身 | `experiment_results.md` §3.2；`first_diff_summary.md` 表 5 |
| **Sinkhorn 是否有益？** | Train SK (L3 only) / Infer SK (post-hoc) / No SK 三向对照 | Train+Infer R@20 0.0829, TrainOnly 0.0826, NoSK 0.0795。Post-hoc Sinkhorn 将身份区分从 extra token 搬运到 L3，使 L3 同时混杂语义区分和身份区分信号，但未减少 L2 之后的总预测不确定性 | 在当前单次 RQ-VAE 对照中，训练期 L3 Sinkhorn 呈现正向作用；导出期 post-hoc Sinkhorn 未表现出稳定增益，主要将身份区分信息从 extra token 转移至 L3，可去除以保持自然 L3 表示与 suffix 身份区分之间的职责边界 | `experiment_results.md` §2.3；`sid_compare.md` |
| **码本利用率越高越好吗？** | 不同 Sinkhorn ε、CoST、Collab、RQ-KMeans 的利用率与下游指标对照 | 多种强均衡化方案均可显著提高浅层码本利用率，但未稳定改善下游；RQ-KMeans 同时获得高利用率和较优召回 | 码本利用率与下游效果不存在单调关系。高利用率可能伴随信息前移和早期路由难度增加，必须联合考察各层条件熵、Prefix Survival 与最终 Recall，而不能将利用率作为独立质量指标 | `experiment_results.md`；`eval_sid.py` 结果 |
| **Extra token 是否是瓶颈？** | Prefix-3 Recall vs Full SID Recall | R@20 差距 0.0043（0.0918 vs 0.0875）；99.3% 的 miss 在前三层已错，仅 0.7% 错在 extra token | Extra token 不是主要瓶颈。在当前四层 beam 输出下，若仅修复已命中前三层前缀的 suffix 错误，R@20 最多由 0.0875 提升至 0.0918 | `prefix_hit_summary.md` |
| **真正瓶颈在哪一层？** | Top-1 首次分叉位置 + Beam K=20 逐层条件存活率 | 以 RQ-KMeans 为例，Top-1 下 P(L1)=5.66%，P(L2\|L1)=30.04%，后续层条件概率较高。Beam 下 RQ-VAE L1@20 更高（45%）但 L1→L2 条件更差（29%），KMeans 相反（30% / 40%），最终 Full@20 接近（8.0~8.7%） | SID 的主要损失集中在早期前缀。不同 tokenizer 在 L1 覆盖率与 L1→L2 条件存活之间呈现不同权衡；通过前两层后，L3/L4 仍产生损失，但边际衰减显著小于前两层 | `first_diff_summary.md` |
| **FD loss 是否解决路由问题？** | CE vs CE+FD，多组对照（RQ-VAE / RQ-KMeans / Collab / L4） | 在当前规范化对照中，FD 在 RQ-VAE、RQ-KMeans 和 Structured L4 上均未获得稳定收益，多数指标下降。旧 SID 上曾观察到正向结果但未在新 SID 上复现 | FD 在所比较的正负 token 上与 CE 梯度方向一致，本质上是负样本驱动的局部梯度重加权；实验中未提供稳定独立增益，已放弃 | `experiment_results.md` §2.5, §3.3 |
| **协同正则是否改善 SID？** | RQ-VAE baseline vs +Collab (item2vec InfoNCE, cw=0.001, τ=0.2) | Collab 重塑了 SID 层级结构（L1 util 22%→81%，沙漏型），但下游 R@20 0.0820 vs baseline 0.0829，全面略降 | 当前 cw=0.001, τ=0.2 的 item2vec 实例级 InfoNCE 对齐显著改变了层级信息分配，但未改善下游效果；说明该直接对齐形式及当前强度不适合现有 tokenizer，不能据此否定协同信号本身的价值 | `experiment_results.md` §2.4 |
| **Structured L4 是否有效？** | RQ-KMeans Extra Token (K0) vs Structured L4 λ=1 / λ=10 (K2) | λ=1 取得最优：R@20 0.0890 (+1.7%)，N@20 0.0390 (+2.9%)。λ=10 分桶分析中，总体净命中增量主要由占 85.1% 的 singleton 样本的小幅改善贡献；collision 整体净增益有限，但 size≥3 的大桶呈现更高的单样本提升。Prefix-3 Recall@10 相对提升 4.5% | Structured L4 在 RQ-KMeans 上取得边际正收益；λ=1 优于 λ=10。分桶与 Prefix-3 分析支持其收益更接近全局标签结构和完整序列优化改善，而非单纯的碰撞消歧，但该机制仍需多随机种子验证 | `experiment_results.md` §3.4；`l4_analysis_summary.md` |
| **TIGER 与内容召回关系如何？** | TIGER (beam search) vs Content (last1 cosine similarity) | Content 在 R@5 和 NDCG 上明显更强，TIGER 在 R@20 提供更广覆盖。两路 top-20 平均仅重叠 0.7~0.8 个 item（Jaccard ~2%）；在两路并集命中的样本中，近 90% 仅被其中一路命中 | 两路候选与命中高度互补，继续优化单路的边际收益明显低于引入联合召回与融合排序 | `overlap_analysis_summary.md` 表 1-2 |
| **互补的主要驱动力是什么？** | Popularity / Similarity / Category / History length 四维分桶分析 | Popularity 呈现最显著的性能反转：Content 长尾优势（Tail 0.09 vs TIGER 0.01），TIGER 头部优势（Head 0.21 vs Content 0.06）。同类目时 Content 更强，跨类目时 TIGER 更强。长历史上 TIGER 优势扩大。 | 两路在不同场景各有不可替代的优势；分桶分析解释了互补来源，但单特征硬路由不足以替代联合软融合 | `overlap_analysis_summary.md` 表 4-7 |
| **融合是否有效？** | 单路 vs RRF / Score minmax / 固定配额 / 动态配额 | Score minmax 在 RQ-KMeans + Content 上取得最高 R@20=0.1150，相对同一 RQ-KMeans 单路（0.0877）提升 31.1%，相对融合评测中最优单路 TIGER（0.0894）提升 28.6%。软融合显著优于固定/动态配额和硬路由 | 多路融合是最终工程方向。软融合优于硬路由的趋势在三种 TIGER backbone 上均重复出现，但由于超参数直接在测试集选择，该结论仍应视为探索性结果 | `fusion_summary.md` |

---

## 4. 最终结论

**1. RQ-KMeans 替代 RQ-VAE 是有效的 tokenizer 升级。**
在固定内容嵌入上逐层 KMeans 量化，不依赖 Sinkhorn 均衡化，下游 R@20 提升 5.5%。训练期 L3 Sinkhorn 对 RQ-VAE 有正向作用；导出期 post-hoc Sinkhorn 未显示出可辨认收益，因此在当前配置中不予保留。

**2. 生成式检索的瓶颈在前两层，不在最后一层。**
L1 绝对准确率仅 ~5.66%，L1→L2 条件存活 ~30-40%，通过前两层后损失显著减小。在全部 full-SID miss 中，仅 0.7% 属于前三层已经命中、但 suffix 预测错误——即使完美修复，Recall 上限也只从 0.0875 提升到 0.0918。FD loss 和协同正则均未有效改善早期路由。

**3. Structured L4 提供边际收益，机制接近全局标签结构优化而非碰撞消歧。**
λ=1 在 RQ-KMeans 上取得 R@20=0.0890（+1.7%）。收益主要来自占 85% 的 singleton 样本和前三层前缀排序改善（Prefix-3 Recall@10 +4.5%），支持"L4 标签结构通过共享 decoder 参数间接改善前缀排序"的假设。但该机制基于单次实验，需多随机种子验证。

**4. 内容相似度召回与 TIGER 高度互补，融合是正确落点。**
两路 top-20 平均仅重叠 ~2% 的 item，近 90% 命中由单路独占。互补来源：Content 擅长长尾（Tail R@20 0.09 vs TIGER 0.01），TIGER 擅长头部（Head R@20 0.21 vs Content 0.06）；Content 在同类目和高相似度场景占优，TIGER 在跨类目和长历史场景占优。软融合（RRF/minmax）一致且显著优于硬路由和配额策略。

**5. 单路 TIGER 优化空间有限，继续投入的边际收益递减。**
连续尝试 listwise loss、FD loss、协同正则、CoST 对比损失均未获稳定独立增益；Structured L4 是规范化对照中唯一表现出一致正向变化的局部改进，但幅度有限（R@20 +1.7%）。相比之下，引入第二条内容召回路并软融合，R@20 相对最优单路提升 28-31%。项目结论是：不再继续孤立优化 TIGER 单路，多路召回是正确工程方向。

---

## 5. 最优结果

### 单路 TIGER

| 配置 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|------|:----:|:----:|:----:|:----:|:----:|:----:|
| RQ-KMeans + Structured L4 λ=1 + CE | 0.0374 | 0.0575 | 0.0890 | 0.0247 | 0.0311 | 0.0390 |

> 来源：`model/logs/tiger_kmeans_L4_lam1.log`，Beauty test，beam=30

### 两路融合

| 配置 | 定位 | R@5 | R@10 | R@20 | N@20 |
|------|------|:----:|:----:|:----:|:----:|
| RQ-KMeans + Content last1 · Score MinMax | 最高 R@20 | 0.0541 | 0.0804 | **0.1150** | 0.0520 |
| RQ-KMeans + L4 λ=1 + Content last1 · Score MinMax | 最佳前排/NDCG | **0.0554** | 0.0804 | 0.1146 | **0.0524** |

> 来源：`fusion.ipynb` / `fusion_rqkmeans.ipynb` / `fusion_rqkmeans_L4_lam1.ipynb`。融合超参数在测试集网格搜索，结果为探索性上限。

### 对比基准

| 方法 | R@5 | R@20 | N@20 |
|------|:----:|:----:|:----:|
| 最优单路 TIGER (RQ-KMeans + L4 λ=1) | 0.0374 | 0.0890 | 0.0390 |
| Content 相似度 (last1, 无训练) | 0.0439 | 0.0779 | 0.0409 |
| 最优融合 (RQ-KMeans + Content, MinMax) | 0.0541 | 0.1150 | 0.0520 |

---

## 6. 关键文件索引

| 文档 | 用途 |
|---|---|
| `project_overview.md`（本文件） | 项目主文档：背景、总表、最终结论、最优结果 |
| `experiment_results.md` | 规范化下游实验的完整指标 |
| `first_diff_summary.md` | 逐层瓶颈诊断（首次分叉 + Beam Survival） |
| `prefix_hit_summary.md` | Extra token 瓶颈：Prefix-3 vs Full SID Recall |
| `l4_analysis_summary.md` | Structured L4 深度分析（K0 vs K2） |
| `overlap_analysis_summary.md` | TIGER vs Content 互补性与四维分桶 |
| `fusion_summary.md` | 多路融合策略对比 |
| `experiment_log.md` | 实验注册表：快速检索所有实验及可信度 |
| `research_journal.md` | 原始研究日记：当时怎么想、为什么转向 |
| `artifact_index.md` | 文件路径速查：SID、checkpoint、日志、notebook |
