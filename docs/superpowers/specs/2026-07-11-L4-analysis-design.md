# L4 Structured Codebook Analysis Design

## Background

Standard TIGER uses a 4-token SID: 3 RQ-KMeans tokens + 1 random extra token.
We replace the random extra token with a learned L4 codebook (256 codes, trained
via constrained KMeans with Hungarian assignment for collision buckets).

K0 = RQ-KMeans + random extra (baseline), K2 = RQ-KMeans + structured L4 (λ=10).
Both share identical L1/L2/L3 catalogs; only L4 assignment differs.

The design answers: **why does K2 outperform K0, and through what mechanisms?**

## Dataset

| Variable | Shape | Description |
|----------|-------|-------------|
| `test_target_p3` | (N_test, 3) | Each test user's target item L1+L2+L3 codes |
| `test_bucket_size` | (N_test,) | How many items share that prefix-3 |
| `beam4_codes` | (N_test, B, 4) | Beam search output, full 4-token SIDs |
| `beam3_prefixes` | (N_test, B, 3) | Direct 3-token beam search output |
| `tf_logits_l4` | (N_test, vocab) | Teacher-forcing logits at position 4 |
| `catalog_l4` | (N_items,) | Assigned L4 code for every item |

Bucket size is a property of the item, not the user. Multiple test users can
have the same target item → same bucket size. Grouping is done per test user
by looking up the target item's bucket size.

## Analysis Dimensions

### 1. Bucket-level Recall / NDCG

**Hypothesis**: Structured L4 helps through two mechanisms:
- *Disambiguation*: assigns distinct codes to items sharing the same L1+L2+L3
- *Semantic sharing*: same L4 code reused across collision buckets carries
  consistent residual-semantic meaning, which T5 can learn as a predictive signal

If only disambiguation matters → improvement isolated to collision targets.
If semantic sharing matters → singleton targets also improve.

**Grouping**: singleton (size=1), collision (size>1), size=2, size=3-4, size≥5.

**Metric**: Full Recall@K and NDCG@K computed from beam4 for each group, K0 vs K2.

**Delta table**: K2 - K0 per group.

### 2. L4 Learnability

**Hypothesis**: Structured L4 should be *more predictable* than random numbering
in hard collision scenarios, despite having higher entropy (256-way vs ~10-way).

**Metrics**:
- L4-masked CE: teacher-forcing CE restricted to the 256 valid L4 tokens.
  Isolates L4 prediction difficulty from noise in L1/L2/L3 token ranges.
- TF Accuracy: argmax L4 prediction given correct L1+L2+L3 prefix.
- Beam conditional accuracy: P(L4 correct | L1+L2+L3 correct in beam).

**Grouping**: Same as dimension 1.

### 3. L4 Token Distribution

**Hypothesis**: K0's random extra token produces a near-degenerate distribution
(most items use code 0, only collision items get codes 1,2,...). K2 should use
all 256 codes with high entropy.

**Metrics**:
- Utilization: |{k: n_k > 0}| / 256
- Shannon entropy: H = -Σ p_k log₂ p_k
- Normalized entropy: H / log₂(256)
- Effective codes: 2^H

**Sources**: catalog assignment (ground-truth) and model prediction (TF argmax),
both split by all vs collision-only.

**Visual**: frequency histogram of L4 codes for K0 vs K2.

### 4. Prefix-3 Recall Change

**Hypothesis**: If L4 token quality affects the entire autoregressive chain (via
shared T5 parameters), then K2 should show higher prefix-3 recall than K0,
despite identical L1/L2/L3 catalogs.

**Metric**: Direct prefix-3 Recall@K from beam3 (generate only 3 tokens).
Full-beam P3R from beam4 as auxiliary.

**Rationale for direct beam3**: Stripping L4 from full beam4 undercounts prefix
coverage because multiple beam slots can point to the same prefix-3 with
different L4 codes. Separate 3-token generation avoids this artifact.

## Implementation

Two-phase approach:

### Phase 1: Inference (one-time, ~30 min)

Run `analysis/inference.py`. For both K0 and K2 models:
1. Build catalog (item-level SID → codes, bucket_size)
2. Teacher-forcing: collect logits at all 4 positions (save as float16)
3. Beam-4: generate full SIDs, save (N_test, B, 4) codes
4. Beam-3: generate prefix-3 only, save (N_test, B, 3) codes

Output: 8 .npz files in `analysis/raw/`. All files share `sample_id` for alignment.

### Phase 2: Analysis (repeatable)

`analysis/l4_analysis.ipynb` reads the raw .npz files and produces:
- Table 1: bucket-level Recall/NDCG + Δ
- Table 2: L4 CE / TF Acc / Beam Cond Acc
- Table 3: L4 token distribution + histogram
- Table 4: Direct prefix-3 Recall + curve

## Interpretation Guide

| Observation | Implication |
|---|---|
| Δ_singleton > 0 in Table 1 | Semantic sharing, not just disambiguation |
| Δ grows with bucket size | Disambiguation value scales with collision complexity |
| K2 TF Acc > K0 TF Acc on size≥3 | Structured L4 is genuinely more learnable, not just higher-entropy |
| K2 P3R > K0 P3R in Table 4 | L4 improvement propagates to L1-L3 via shared parameters |
| K2 P3R ≈ K0 P3R | Improvement is local to L4 layer only |
