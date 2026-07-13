#!/usr/bin/env python3
"""Item embedding similarity retrieval — evaluation.

Builds a per-item top-K neighbour index using faiss (ANN), then retrieves
by aggregating neighbour scores across a user's history with time decay.

Usage:
    python item2vec/retrieve.py \
        --emb_path data/Beauty/item2vec_emb.npy \
        --dataset data/Beauty
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
import os

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)


# ── 1. Index ──────────────────────────────────────────────────────────────────

def build_index(emb_norm, index_k=500):
    """用 faiss 为每个物品构建 top-K 最近邻索引.

    余弦相似度 ← L2 归一化后的内积 (IndexFlatIP).

    Args:
        embeddings: (V, d) float32, row 0 is padding.
        index_k:    每个物品保留的最近邻数量.

    Returns:
        neighbours: (V, index_k) int32.
        scores:     (V, index_k) float32.
    """
    emb = emb_norm.astype(np.float32)
    V, d = emb.shape

    # faiss inner-product index — 一次 search 拿到所有物品的 top-k
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    # emb中的每个向量，从index中取index_k+2个（多取两个给pad和self留余地）
    sims, idxs = index.search(emb, index_k + 2)

    # 过滤 self 和 padding（多取了 2 个，过滤后保证 ≥ index_k）
    neighbours = np.empty((V, index_k), dtype=np.int32)
    neighbour_scores = np.empty((V, index_k), dtype=np.float32)
    for i in range(V):
        mask = (idxs[i] != 0) & (idxs[i] != i)
        neighbours[i] = idxs[i][mask][:index_k]
        neighbour_scores[i] = sims[i][mask][:index_k]

    return neighbours, neighbour_scores


# ── 2. Retrieval ──────────────────────────────────────────────────────────────

def retrieve(neighbours, neighbour_scores, dataset_path,
             last_k=1, decay=0.9, topk=20):
    """用物品最近邻索引做召回.

    取用户最后 last_k 个历史物品，各自查询其最近邻，按位置衰减
    (decay^{距末尾距离}) 加权聚合后取 top-k.

    Args:
        neighbours:       (V, index_k) 每个物品的最近邻 ID.
        neighbour_scores: (V, index_k) 对应分数.
        dataset_path:     数据集目录.
        last_k:           取最后几个历史物品做 query.
        decay:            位置衰减因子 (1.0 = 等权).
        topk:             每个用户召回的物品数.

    Returns:
        preds:        (N, topk) 召回物品 ID.
        pred_scores:  (N, topk) 对应分数.
    """
    test = pd.read_parquet(f'{dataset_path}/test.parquet')
    num_users = len(test)

    preds = np.empty((num_users, topk), dtype=np.int32)
    pred_scores = np.empty((num_users, topk), dtype=np.float32)

    for idx, row in enumerate(tqdm(test.itertuples(index=False),
                                    total=num_users, desc="Retrieving", ncols=90)):
        hist = list(row.history)
        hist_set = set(hist)

        # last_k=0 → hist[-0:] == hist[:]（全序列）
        query_items = hist[-last_k:]
        n = len(query_items)
        candidates = defaultdict(float)
        for i, item_id in enumerate(query_items):
            weight = decay ** (n - 1 - i)          # 最后一个 weight=1
            for nb_item, sim_ij in zip(neighbours[item_id], neighbour_scores[item_id]):
                if nb_item not in hist_set:
                    candidates[nb_item] += sim_ij * weight

        # --- top-k ---
        top_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:topk]
        preds[idx], scores = zip(*top_items) if top_items else ([], [])
        pred_scores[idx] = scores

    return preds, pred_scores


# ── 3. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(preds, dataset_path, topk_list=(5, 10, 20)):
    """计算召回结果的 MRR@k / Recall@k / NDCG@k.

    target 在 preds 中时用其位置作为 rank，否则 rank = max(topk_list)
    （此时所有 @k 指标贡献为 0，无需计算精确排名）。

    Args:
        preds:       (N, topk) retrieve() 返回的召回列表.
        dataset_path: 数据集目录.
        topk_list:   cutoffs.

    Returns:
        dict of metric_name → float.
    """
    test = pd.read_parquet(f'{dataset_path}/test.parquet')
    num_users = len(test)
    topk_max = max(topk_list)
    not_found = topk_max  # sentinel: target not in preds → rank ≥ topk_max

    ranks = np.full(num_users, not_found, dtype=np.int32)

    for idx, row in enumerate(tqdm(test.itertuples(index=False),
                                    total=num_users, desc="Evaluating", ncols=90)):
        target = row.target
        hit = np.where(preds[idx] == target)[0]
        if len(hit) > 0:
            ranks[idx] = hit[0]

    r = ranks.astype(np.float64)
    metrics = {}
    for k in topk_list:
        ok = r < k
        metrics[f'MRR@{k}'] = np.where(ok, 1.0 / (r + 1.0), 0.0).mean()
        metrics[f'Recall@{k}'] = ok.mean()
        metrics[f'NDCG@{k}'] = np.where(ok, 1.0 / np.log2(r + 2.0), 0.0).mean()

    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Item embedding similarity retrieval evaluation")
    parser.add_argument('--emb_path', default='data/Beauty/item_emb.parquet', help='Path to .npy item embedding file.')  # item2vec_emb.npy
    parser.add_argument('--dataset', default='data/Beauty', help='Path prefix for parquet files.')
    parser.add_argument('--last_k', type=int, default=1, help='每个用户根据最后k个交互进行召回')
    parser.add_argument('--decay', type=float, default=0.9, help='位置衰减因子 (1.0 = 等权).')
    parser.add_argument('--index_k', type=int, default=500, help='Number of neighbours per item in the index.')
    parser.add_argument('--topk', nargs='+', type=int, default=[5, 10, 20, 30, 50], help='Cutoffs for Recall/NDCG.')
    parser.add_argument('--save_preds', type=str, default='predictions/content_sim_last1_top50.npy', help='If set, save top-k predictions to this .npy path.')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 加载 embeddings（兼容 .npy 和 .parquet）
    if args.emb_path.endswith('.parquet'):
        df = pd.read_parquet(args.emb_path)
        dim = len(df['embedding'].iloc[0])
        max_id = int(df['ItemID'].max())
        emb = np.zeros((max_id + 1, dim), dtype=np.float32)
        for row in df.itertuples(index=False):
            emb[row.ItemID] = np.array(row.embedding, dtype=np.float32)
    else:
        emb = np.load(args.emb_path).astype(np.float32)
    logger.info(f"Embeddings: {emb.shape}, active: {(emb.sum(1) != 0).sum()}")

    # 归一化embedding，
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    emb_norm = emb / norms

    # 构建索引
    t0 = time.time()
    neighbours, neighbour_scores = build_index(emb_norm, index_k=args.index_k)
    logger.info(f"Index: {neighbours.shape}, {time.time() - t0:.1f}s ({neighbours.nbytes / 1e6:.1f} MB)")

    # 召回
    preds, pred_scores = retrieve(neighbours, neighbour_scores, args.dataset,
                                  last_k=args.last_k, decay=args.decay,
                                  topk=max(args.topk))
    logger.info(f"Retrieved {len(preds)} users, top-k={preds.shape[1]}")

    # 保存预测结果（item IDs + scores）
    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds) or '.', exist_ok=True)
        np.save(args.save_preds, preds)
        score_path = args.save_preds.replace('.npy', '_scores.npy')
        np.save(score_path, pred_scores)
        logger.info(f"Saved predictions to {args.save_preds}")
        logger.info(f"Saved scores to {score_path}")

    # 评估
    metrics = evaluate(preds, args.dataset, args.topk)

    print()
    print("=" * 55)
    print("  Item Embedding Similarity Retrieval")
    print("=" * 55)
    print(f"  Embeddings:   {args.emb_path}")
    print(f"  Index:         {args.index_k} neighbours/item")
    print(f"  Query:         last_{args.last_k}, decay={args.decay}")
    print(f"  Test users:    {len(preds)}")
    print("-" * 55)
    for k in args.topk:
        line = ''
        for metric in [f'MRR@{k}', f'Recall@{k}', f'NDCG@{k}']:
            line += f"  {metric:<10} {metrics[metric]:.4f}"
        print(line)
    print("=" * 55)


if __name__ == '__main__':
    main()

# =======================================================
#   Item Embedding Similarity Retrieval
# =======================================================
#   Embeddings:   data/Beauty/item2vec_emb.npy
#   Index:         500 neighbours/item
#   Query:         last_1
#   Test users:    22363
# -------------------------------------------------------
#   MRR@5      0.0154  Recall@5   0.0256  NDCG@5     0.0179
#   MRR@10     0.0171  Recall@10  0.0382  NDCG@10    0.0220
#   MRR@20     0.0182  Recall@20  0.0552  NDCG@20    0.0263
# =======================================================

# =======================================================
#   Item Embedding Similarity Retrieval
# =======================================================
#   Embeddings:   data/Beauty/item2vec_emb.npy
#   Index:         500 neighbours/item
#   Query:         last_2, decay=0.9
#   Test users:    22363
# -------------------------------------------------------
#   MRR@5      0.0113  Recall@5   0.0209  NDCG@5     0.0136
#   MRR@10     0.0127  Recall@10  0.0318  NDCG@10    0.0171
#   MRR@20     0.0139  Recall@20  0.0491  NDCG@20    0.0215
# =======================================================

# =======================================================
#   Item Embedding Similarity Retrieval
# =======================================================
#   Embeddings:   data/Beauty/item2vec_emb.npy
#   Index:         500 neighbours/item
#   Query:         last_3, decay=0.9
#   Test users:    22363
# -------------------------------------------------------
#   MRR@5      0.0078  Recall@5   0.0158  NDCG@5     0.0097
#   MRR@10     0.0091  Recall@10  0.0257  NDCG@10    0.0129
#   MRR@20     0.0101  Recall@20  0.0412  NDCG@20    0.0168
# =======================================================

# =======================================================
#   Item Embedding Similarity Retrieval
# =======================================================
#   Embeddings:   data/Beauty/item2vec_emb.npy
#   Index:         500 neighbours/item
#   Query:         last_0 (全序列), decay=0.9
#   Test users:    22363
# -------------------------------------------------------
#   MRR@5      0.0045  Recall@5   0.0091  NDCG@5     0.0056
#   MRR@10     0.0053  Recall@10  0.0151  NDCG@10    0.0076
#   MRR@20     0.0060  Recall@20  0.0245  NDCG@20    0.0100
# =======================================================

# 用内容嵌入来做

# =======================================================
#   Item Embedding Similarity Retrieval
# =======================================================
#   Embeddings:   data/Beauty/item_emb.parquet
#   Index:         500 neighbours/item
#   Query:         last_1, decay=0.9
#   Test users:    22363
# -------------------------------------------------------
#   MRR@5      0.0271  Recall@5   0.0439  NDCG@5     0.0313
#   MRR@10     0.0291  Recall@10  0.0589  NDCG@10    0.0361
#   MRR@20     0.0304  Recall@20  0.0779  NDCG@20    0.0409
# =======================================================

# =======================================================
#   Item Embedding Similarity Retrieval
# =======================================================
#   Embeddings:   data/Beauty/item_emb.parquet
#   Index:         500 neighbours/item
#   Query:         last_2, decay=0.9
#   Test users:    22363
# -------------------------------------------------------
#   MRR@5      0.0158  Recall@5   0.0275  NDCG@5     0.0187
#   MRR@10     0.0178  Recall@10  0.0426  NDCG@10    0.0236
#   MRR@20     0.0192  Recall@20  0.0620  NDCG@20    0.0285
# =======================================================

# =======================================================
#   Item Embedding Similarity Retrieval
# =======================================================
#   Embeddings:   data/Beauty/item_emb.parquet
#   Index:         500 neighbours/item
#   Query:         last_0, decay=0.9
#   Test users:    22363
# -------------------------------------------------------
#   MRR@5      0.0060  Recall@5   0.0116  NDCG@5     0.0074
#   MRR@10     0.0070  Recall@10  0.0195  NDCG@10    0.0099
#   MRR@20     0.0079  Recall@20  0.0323  NDCG@20    0.0131
# =======================================================