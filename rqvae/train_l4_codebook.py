#!/usr/bin/env python3
"""Post-hoc structured L4 codebook — constrained KMeans on frozen L1/L2/L3.

Replaces random extra tokens with a learned 4th-layer codebook.  Only L4 is
trained; the first three layers are frozen.

Collision-bucket items (same L1+L2+L3 prefix) are assigned via Hungarian
(injective within bucket); singleton items use plain argmin.

Usage (RQ-VAE):
    python rqvae/train_l4_codebook.py \
        --mode rqvae --ckpt rqvae/ckpt/Beauty/.../best_collision_model.pth \
        --sid_in data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-inferExtraOnly.npy \
        --sid_out data/Beauty/Beauty_t5_rqvae_260709-L4.npy

Usage (RQ-KMeans):
    python rqvae/train_l4_codebook.py \
        --mode rqkmeans \
        --emb_path data/Beauty/item_emb.parquet \
        --sid_in data/Beauty/Beauty_kmeans_code.npy \
        --sid_out data/Beauty/Beauty_kmeans_code-L4.npy
"""

import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from tqdm import tqdm


# ── Hungarian assignment for one collision bucket ─────────────────────────

def hungarian_assign(residuals, centroids):
    """Assign each item in the bucket to a unique centroid via Hungarian.

    Args:
        residuals: (B, d) residual vectors for this bucket.
        centroids: (K, d) all L4 codebook vectors.

    Returns:
        assignments: (B,) centroid indices (all distinct).
        cost: scalar total L2 cost.
    """
    B, d = residuals.shape
    K = centroids.shape[0]
    # Cost matrix: (B, K) L2 distances
    cost = torch.cdist(
        torch.as_tensor(residuals, dtype=torch.float32),
        torch.as_tensor(centroids, dtype=torch.float32),
        p=2,
    ).numpy()  # (B, K) — small, scipy is fine
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind.astype(np.int64), cost[row_ind, col_ind].sum()


# ── Dead-code reinit ──────────────────────────────────────────────────────

def reinit_dead_codes(centroids, residuals, assignments):
    """Re-seed unused centroids from the highest-error residuals."""
    used = set(assignments)
    dead = [k for k in range(len(centroids)) if k not in used]
    if not dead:
        return centroids

    errors = np.array([
        np.linalg.norm(residuals[i] - centroids[assignments[i]])
        for i in range(len(residuals))
    ])
    # Pick items with highest error, sorted descending
    candidates = np.argsort(errors)[::-1]
    for dk in dead:
        if len(candidates) == 0:
            break
        centroids[dk] = residuals[candidates[0]]
        candidates = candidates[1:]
    return centroids


# ── Main constrained KMeans ───────────────────────────────────────────────

def train_l4(residuals, groups, codebook_size=256, lam=10.0,
             max_iters=50, tol=1e-4, verbose=True):
    """Train L4 codebook with Hungarian constraint on collision buckets.

    Args:
        residuals: (N, d) residual vectors r_i^(3).
        groups:    dict mapping prefix tuple → list of item indices.
        codebook_size: number of L4 prototypes (default 256).
        lam:       collision-bucket weight multiplier (λ in the spec).
        max_iters: max EM iterations.
        tol:       convergence threshold (fractional centroid change).

    Returns:
        centroids:   (K, d) L4 codebook.
        assignments: (N,) L4 code index per item.
    """
    N, d = residuals.shape
    K = codebook_size
    residuals = residuals.astype(np.float32)

    # ── separate singletons and collision buckets ──
    singleton_idx = []
    collision_buckets = []  # list of (indices,)
    for prefix, members in groups.items():
        if len(members) == 1:
            singleton_idx.append(members[0])
        else:
            collision_buckets.append(np.array(members))

    singleton_idx = np.array(singleton_idx, dtype=np.int64)
    n_singleton = len(singleton_idx)
    n_collision = sum(len(b) for b in collision_buckets)
    if verbose:
        print(f"  Items: {N} total, {n_singleton} singletons, "
              f"{n_collision} collision ({len(collision_buckets)} buckets)")

    # ── weights ──
    w = np.ones(N, dtype=np.float32)  # default: singleton weight = 1
    for bucket in collision_buckets:
        w[bucket] = lam / len(bucket)  # collision items: λ / |G_b|

    # ── initialise centroids via KMeans on all residuals ──
    km = KMeans(n_clusters=K, n_init=3, max_iter=20, random_state=42)
    km.fit(residuals)
    centroids = km.cluster_centers_.astype(np.float32)

    # ── EM ──
    prev_centroids = centroids.copy()
    for it in range(max_iters):
        assignments = np.zeros(N, dtype=np.int64)

        # E-step: argmin for singletons
        if n_singleton > 0:
            dists = np.linalg.norm(
                residuals[singleton_idx, None, :] - centroids[None, :, :],
                axis=-1,
            )  # (S, K)
            assignments[singleton_idx] = np.argmin(dists, axis=1)

        # E-step: Hungarian for collision buckets
        for bucket in tqdm(collision_buckets, desc=f"  Iter {it+1} Hungarian",
                           disable=not verbose, ncols=80):
            assign, _ = hungarian_assign(residuals[bucket], centroids)
            assignments[bucket] = assign

        # M-step: weighted mean
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = (
                    (residuals[mask] * w[mask, None]).sum(axis=0)
                    / w[mask].sum()
                )
            # else: keep previous (will be reinitialised below)

        # Dead-code reinit
        new_centroids = reinit_dead_codes(
            new_centroids, residuals, assignments)

        # Convergence check
        shift = np.linalg.norm(new_centroids - prev_centroids) / (
            np.linalg.norm(prev_centroids) + 1e-8)
        centroids = new_centroids
        prev_centroids = centroids.copy()

        if verbose:
            # Quick collision-check just for monitoring
            used = len(set(assignments))
            print(f"  Iter {it+1:>3}: shift={shift:.6f}  "
                  f"codes used={used}/{K}")
        if shift < tol:
            if verbose:
                print(f"  Converged at iter {it+1}")
            break

    return centroids, assignments


# ── RQ-VAE: get residuals from model ─────────────────────────────────────

def get_rqvae_residuals(ckpt_path, data_path, device='cuda'):
    """Run RQ-VAE encoder + VQ to get 32-dim residuals per item."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rqvae'))
    from datasets import EmbDataset
    from models.rqvae import RQVAE
    from torch.utils.data import DataLoader

    data = EmbDataset(data_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']

    model = RQVAE(
        in_dim=data.dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
        layers=args.layers, sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters,
        kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
        recon_weight=getattr(args, 'recon_weight', 1.0),
        quant_weight=getattr(args, 'quant_weight', 1.0),
    )
    model_dict = model.state_dict()
    filtered = {k: v for k, v in ckpt['state_dict'].items()
                if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()

    loader = DataLoader(data, batch_size=256, shuffle=False)
    all_z = []
    all_x_q = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='RQ-VAE forward', ncols=80):
            batch = batch.to(device)
            x_e = model.encoder(batch)  # encoder推理得到z
            x_q, _, _, x_q_list, _, _, _ = model.rq(x_e, use_sk=False)  # 从码本中匹配z_q
            all_z.append(x_e.cpu().numpy())
            all_x_q.append(x_q_list[2].cpu().numpy())  # c1+c2+c3 (STE)

    z = np.concatenate(all_z, axis=0).astype(np.float32)
    z_q3 = np.concatenate(all_x_q, axis=0).astype(np.float32)
    residuals = z - z_q3  # (N, 32)
    return residuals


# ── RQ-KMeans: reconstruct centroids & compute residuals ──────────────────

def get_rqkmeans_residuals(emb_path, sid_path):
    """Reconstruct RQ-KMeans centroids from codes + embeddings, compute residuals."""
    df = pd.read_parquet(emb_path)
    embeddings = np.stack(df['embedding'].values).astype(np.float32)
    codes = np.load(sid_path)[:, :3].astype(np.int64)  # (N, 3)
    N, d = embeddings.shape
    K = 256

    # Reconstruct per-layer centroids
    centroids = []
    residual = embeddings.copy()
    for lvl in range(3):
        c = np.zeros((K, d), dtype=np.float32)
        for k in range(K):
            mask = codes[:, lvl] == k
            if mask.any():
                c[k] = residual[mask].mean(axis=0)
        centroids.append(c)
        # Subtract this layer's contribution
        residual = residual - c[codes[:, lvl]]

    # residual now is r_i^(3) = emb - (c1 + c2 + c3)
    return residual.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Post-hoc structured L4 codebook training')
    parser.add_argument('--mode', choices=['rqvae', 'rqkmeans'], required=True)
    # RQ-VAE inputs
    parser.add_argument('--ckpt', type=str, default=None,
                       help='RQ-VAE checkpoint path (mode=rqvae)')
    # Shared
    parser.add_argument('--emb_path', type=str,
                       default='data/Beauty/item_emb.parquet',
                       help='Path to item embeddings parquet')
    # Shared
    parser.add_argument('--sid_in', type=str, required=True,
                       help='Path to existing 3-token SID .npy file')
    parser.add_argument('--sid_out', type=str, required=True,
                       help='Output path for 4-token SID .npy')
    parser.add_argument('--codebook_size', type=int, default=256)
    parser.add_argument('--lam', type=float, default=10.0,
                       help='Collision weight multiplier λ')
    parser.add_argument('--max_iters', type=int, default=50)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # ── Load SID and build groups ──
    codes = np.load(args.sid_in)
    codes_3 = codes[:, :3].astype(np.int64)  # (N, 3)
    N = len(codes_3)

    # 计算L1L2L3前缀桶
    groups = defaultdict(list)
    for i in range(N):
        groups[tuple(codes_3[i])].append(i)

    print(f"Loaded {args.sid_in}: {N} items, {len(groups)} unique prefixes")

    # ── 获取前三层已量化的残差 Get residuals ──
    if args.mode == 'rqvae':
        if args.ckpt is None:
            raise ValueError('--ckpt required for mode=rqvae')
        residuals = get_rqvae_residuals(args.ckpt, args.emb_path, args.device)
    else:
        residuals = get_rqkmeans_residuals(args.emb_path, args.sid_in)

    print(f"Residuals: shape={residuals.shape}, "
          f"mean norm={np.linalg.norm(residuals, axis=-1).mean():.4f}")

    # ── Train L4 ──
    centroids, assignments = train_l4(
        residuals, groups,
        codebook_size=args.codebook_size,
        lam=args.lam,
        max_iters=args.max_iters,
        tol=args.tol,
        verbose=True,
    )

    # ── Generate new SID ──
    codes_new = np.zeros((N, 4), dtype=np.int32)
    codes_new[:, :3] = codes_3  # keep L1/L2/L3
    codes_new[:, 3] = assignments.astype(np.int32)

    # Save RAW codes (0-255); dataset.item2code() applies offset encoding
    np.save(args.sid_out, codes_new)

    unique_full = len(set(tuple(row) for row in codes_new))
    collision_full = N - unique_full
    print(f"\nSaved to {args.sid_out}")
    print(f"  Full 4-token collisions: {collision_full}/{N} "
          f"({100*collision_full/N:.2f}%)")
    print("Done.")


if __name__ == '__main__':
    main()
