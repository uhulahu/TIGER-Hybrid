#!/usr/bin/env python3
"""Item2Vec via gensim Word2Vec (skip-gram + negative sampling).

Usage:
    python item2vec/train_item2vec.py \
        --data_path data/Beauty/train.parquet \
        --output_path data/Beauty/item2vec_emb.npy \
        --emb_dim 32 --window 5 --neg 5 --epochs 15
"""

import argparse

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def main():
    parser = argparse.ArgumentParser(description="Item2Vec via gensim Word2Vec")
    parser.add_argument('--data_path', default='data/Beauty/train.parquet')
    parser.add_argument('--output_path', default='data/Beauty/item2vec_emb.npy')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--neg', type=int, default=5, help='negative samples')
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    # ── load sequences ──
    df = pd.read_parquet(args.data_path)
    sequences = []
    for _, row in df.iterrows():
        seq = list(row['history']) + [int(row['target'])]
        sequences.append([str(i) for i in seq])

    max_item_id = max(int(tok) for seq in sequences for tok in seq)
    print(f"Items: {max_item_id}  |  Sequences: {len(sequences)}  "
          f"|  Avg len: {np.mean([len(s) for s in sequences]):.1f}")

    # ── train ──
    model = Word2Vec(
        sentences=sequences,
        vector_size=args.emb_dim,
        window=args.window,
        negative=args.neg,
        sg=1, min_count=1,
        epochs=args.epochs,
        seed=42,
    )
    print(f"Training complete, vocab size: {len(model.wv)}")

    # ── save (1-indexed, row 0 = zero) ──
    embeddings = np.zeros((max_item_id + 1, args.emb_dim), dtype=np.float32)
    for token, idx in model.wv.key_to_index.items():
        embeddings[int(token)] = model.wv.vectors[idx]

    np.save(args.output_path, embeddings)
    print(f"Saved to {args.output_path}  (shape={embeddings.shape})")


if __name__ == '__main__':
    main()
