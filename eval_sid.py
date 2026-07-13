#!/usr/bin/env python3
"""Evaluate SID quality: codebook utilization, purity, NMI/ARI, entropy.

Can be run standalone (reads Config section) or imported as:
    from eval_sid import evaluate_sid
    report = evaluate_sid(sid_path, meta_path, mapping_path, n_levels=3)
"""

import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def entropy(counts):
    """Shannon entropy H = -Σ p·log2(p)."""
    _, cnt = np.unique(counts, return_counts=True)
    p = cnt / cnt.sum()
    return -np.sum(p * np.log2(p))


def evaluate_sid(sid_path, meta_path, mapping_path, n_levels=3):
    """Run full SID quality evaluation and return the report as a string."""
    lines = []

    codes = np.load(sid_path)[:, :n_levels].astype(int)
    N = len(codes)

    item_id_map = np.load(mapping_path, allow_pickle=True).item()

    with open(meta_path) as f:
        meta = [json.loads(l) for l in f]
    cat_of = {}
    for m in meta:
        asin = m.get('asin')
        if asin in item_id_map:
            cats = m.get('categories', [])
            path = cats[0] if cats and len(cats) > 0 else ['Unknown']
            cat_of[item_id_map[asin]] = path[1] if len(path) > 1 else path[0]

    has_cat = np.array([(i + 1) in cat_of for i in range(N)])
    cats = np.array([cat_of.get(i + 1, 'Unknown') for i in range(N)])

    brand_of = {}
    for m in meta:
        asin = m.get('asin')
        if asin in item_id_map:
            brand_of[item_id_map[asin]] = m.get('brand', 'Unknown')
    brand_arr = np.array([brand_of.get(i + 1, 'Unknown') for i in range(N)])

    # ── 1. Codebook utilization ──
    lines.append("=" * 55)
    lines.append("  SID Quality Evaluation")
    lines.append("=" * 55)
    lines.append("")
    lines.append("─ 1. Codebook utilization ─")
    for level in range(n_levels):
        used = len(set(codes[:, level]))
        total = 256
        lines.append(f"  L{level+1}: {used}/{total} entries used ({100*used/total:.1f}%)")

    # ── 2. Prefix Purity ──
    lines.append("")
    lines.append("─ 2. Prefix Purity ─")
    lines.append(f"  {'L':<3} {'Groups':<8} {'Purity':<10} {'Cat.NMI':<10} {'Cat.ARI':<10} {'BrandPurity':<12}")
    lines.append(f"  {'-'*3} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for L in range(1, n_levels + 1):
        prefix = [tuple(codes[i, :L]) for i in range(N)]
        groups = defaultdict(list)
        for i, p in enumerate(prefix):
            groups[p].append(i)

        total_cat, correct_cat = 0, 0
        total_brand, correct_brand = 0, 0
        prefix_ids = np.empty(N, dtype=int)
        for gid, (p, members) in enumerate(groups.items()):
            for m in members:
                prefix_ids[m] = gid
            m_cats = [cats[m] for m in members if has_cat[m]]
            m_brands = [brand_arr[m] for m in members if brand_arr[m] != 'Unknown']
            if m_cats:
                correct_cat += Counter(m_cats).most_common(1)[0][1]
                total_cat += len(m_cats)
            if m_brands:
                correct_brand += Counter(m_brands).most_common(1)[0][1]
                total_brand += len(m_brands)

        purity = correct_cat / total_cat if total_cat else 0
        brand_purity = correct_brand / total_brand if total_brand else 0
        nmi = normalized_mutual_info_score(cats[has_cat], prefix_ids[has_cat])
        ari = adjusted_rand_score(cats[has_cat], prefix_ids[has_cat])

        lines.append(f"  L{L:<3} {len(groups):<8} {purity:<10.4f} {nmi:<10.4f} {ari:<10.4f} {brand_purity:<12.4f}")

    # ── L1 per-category ──
    lines.append("")
    lines.append("─ L1 assignment by coarse category (top-10 L1 codes) ─")
    l1_counter = Counter(int(codes[i, 0]) for i in range(N) if has_cat[i])
    for l1code, _ in l1_counter.most_common(10):
        members = [i for i in range(N) if has_cat[i] and codes[i, 0] == l1code]
        dist = Counter(cats[m] for m in members)
        majority = dist.most_common(1)[0]
        lines.append(f"  Code {l1code:>3} ({len(members):>4} items): "
                     f"majority={majority[0]} ({majority[1]/len(members):.0%}), "
                     f"dist={{{', '.join(f'{k}:{v}' for k,v in dist.most_common(5))}}}")
    lines.append(f"\n  Total items with category: {has_cat.sum()} / {N}")

    # ── 3. Bucket-size ──
    lines.append("")
    lines.append("─ 3. Prefix bucket-size distribution ─")
    lines.append(f"  {'L':<3} {'Groups':<8} {'Min':<6} {'p25':<6} {'Median':<8} "
                 f"{'Mean':<8} {'p75':<6} {'p95':<6} {'Max':<6} {'Std':<8}")
    lines.append(f"  {'-'*3} {'-'*8} {'-'*6} {'-'*6} {'-'*8} "
                 f"{'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for L in range(1, n_levels + 1):
        prefix = [tuple(codes[i, :L]) for i in range(N)]
        groups = defaultdict(list)
        for i, p in enumerate(prefix):
            groups[p].append(i)
        sizes = np.array([len(v) for v in groups.values()])
        lines.append(f"  L{L:<3} {len(groups):<8} {sizes.min():<6} "
                     f"{np.percentile(sizes, 25):<6.0f} "
                     f"{np.median(sizes):<8.0f} {sizes.mean():<8.1f} "
                     f"{np.percentile(sizes, 75):<6.0f} "
                     f"{np.percentile(sizes, 95):<6.0f} {sizes.max():<6} {sizes.std():<8.1f}")

    # ── 4. Conditional code usage ──
    lines.append("")
    lines.append("─ 4. Conditional code usage (branching factor) ─")
    l1_to_l2 = defaultdict(set)
    l2_to_l3 = defaultdict(lambda: defaultdict(set))
    for i in range(N):
        c0, c1, c2 = int(codes[i, 0]), int(codes[i, 1]), int(codes[i, 2])
        l1_to_l2[c0].add(c1)
        l2_to_l3[(c0, c1)][c0].add(c2)
    l1_branch = np.array([len(v) for v in l1_to_l2.values()])
    lines.append(f"  L1→L2: per L1 code, {len(l1_to_l2)} L1 codes used → "
                 f"mean branching: {l1_branch.mean():.1f}, "
                 f"median: {np.median(l1_branch):.0f}, "
                 f"min={l1_branch.min()}, max={l1_branch.max()}, "
                 f"fraction of 256: {l1_branch.mean()/256:.3f}")
    l1l2_branch = []
    for (c0, c1), inner in l2_to_l3.items():
        l1l2_branch.append(len(inner.get(c0, set())))
    l1l2_branch = np.array(l1l2_branch)
    if len(l1l2_branch) > 0:
        lines.append(f"  L1L2→L3: per prefix, {len(l1l2_branch)} L1L2 pairs → "
                     f"mean branching: {l1l2_branch.mean():.1f}, "
                     f"median: {np.median(l1l2_branch):.0f}, "
                     f"min={l1l2_branch.min()}, max={l1l2_branch.max()}, "
                     f"fraction of 256: {l1l2_branch.mean()/256:.3f}")

    # ── 5. Entropy ──
    lines.append("")
    lines.append("─ 5. Entropy (bits) ─")
    for L in range(n_levels):
        h = entropy(codes[:, L])
        max_h = np.log2(256)
        lines.append(f"  H(L{L+1})        = {h:.4f}  (max {max_h:.4f}, {100*h/max_h:.1f}% used)")
    for L in range(1, n_levels):
        prefix_int = codes[:, 0].copy()
        for lvl in range(1, L + 1):
            prefix_int = prefix_int * 256 + codes[:, lvl]
        h_joint = entropy(prefix_int)
        lines.append(f"  H(L1..L{L+1})     = {h_joint:.4f}")
    h_l1 = entropy(codes[:, 0])
    prefix_l1l2 = codes[:, 0] * 256 + codes[:, 1]
    h_l1l2 = entropy(prefix_l1l2)
    lines.append(f"  H(L2|L1)      = {h_l1l2 - h_l1:.4f}")
    if n_levels >= 3:
        prefix_l1l2l3 = codes[:, 0] * 256 * 256 + codes[:, 1] * 256 + codes[:, 2]
        h_l1l2l3 = entropy(prefix_l1l2l3)
        lines.append(f"  H(L3|L1,L2)   = {h_l1l2l3 - h_l1l2:.4f}")
    lines.append("")
    lines.append("  Information breakdown (% of total):")
    total_h = entropy(codes.ravel())
    for L in range(n_levels):
        h = entropy(codes[:, L])
        lines.append(f"  L{L+1}: {h:.2f} bits ({100*h/total_h:.1f}% of token-level entropy)")
    joint = entropy(prefix_l1l2l3 if n_levels >= 3 else prefix_l1l2)
    lines.append(f"  Joint entropy of full SID: {joint:.4f}")

    return "\n".join(lines)


# ── Standalone entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    SID_PATH     = "data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003].npy" # "data/Beauty/Beauty_kmeans_code.npy"
    META_PATH    = "data/Beauty/Beauty_metadata.json"
    MAPPING_PATH = "data/Beauty/item_mapping.npy"
    N_LEVELS     = 3
    print(evaluate_sid(SID_PATH, META_PATH, MAPPING_PATH, N_LEVELS))
