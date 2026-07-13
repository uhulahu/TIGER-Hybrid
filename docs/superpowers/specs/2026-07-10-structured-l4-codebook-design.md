# Structured L4 Codebook Design

## Motivation

TIGER generates 4-token SIDs. The first 3 tokens come from RQ-VAE or RQ-KMeans
quantization of content embeddings. The 4th token (extra token) resolves L3
collisions — multiple items that share the same (L1,L2,L3) prefix get different
extra tokens (0, 1, 2, ...).

Currently the extra token is assigned randomly within each collision bucket.
It has **no cross-bucket meaning** — `extra=1` in bucket A has no relationship
to `extra=1` in bucket B. Prior analysis confirmed that extra token is NOT the
prediction bottleneck (L1+L2+L3 errors dominate), but a structured 4th codebook
is still worth exploring as a fine-grained improvement.

## Goal

Replace random extra tokens with a **learned 4th-layer codebook** — a
post-hoc, frozen-L1/L2/L3 constrained KMeans that gives L4 codes semantic
meaning while respecting bucket-level injectivity for collision items.

## Algorithm: Post-hoc Structured L4 Codebook

### Inputs

- Frozen L1/L2/L3 codebooks and encoder from trained RQ-KMeans or RQ-VAE
- Item embeddings `z_i` (encoder output, 32-dim)
- Collision-group info from L1/L2/L3 codes

### Step 1: Compute L3 residuals

```
r_i = z_i - (c_i1 + c_i2 + c_i3)    for all items i
```

### Step 2: Group items by L1+L2+L3 prefix

```
G_b = { i : (c_i1, c_i2, c_i3) = b }
  singleton: |G_b| = 1       (~93% of items)
  collision: |G_b| > 1       (~7% of items)
```

### Step 3: Initialize L4 codebook

Run KMeans on all `{r_i}` with K = 256 → initial L4 prototypes `E^(4)`.

### Step 4: Iterative EM

Repeat until convergence or max_iter:

**E-step (assignment):**

- Singleton item: `a(i) = argmin_k ||r_i - e_k^(4)||²`
- Collision bucket `G_b`: Hungarian algorithm solving
  ```
  min_{a: G_b → [0..255] injective}  Σ_{i∈G_b} ||r_i - e_{a(i)}^{(4)}||²
  ```
  (cost matrix: |G_b| × 256 L2 distances; pad with large cost to make square if needed)

**M-step (update):**

```
e_k^(4) = Σ_i w_i · r_i · I(a(i)=k) / Σ_i w_i · I(a(i)=k)
```

Weight per item:
```
w_i = (λ / |G_b|) · I(|G_b| > 1)  +  1 · I(|G_b| = 1)
```
where λ controls how much extra weight collision items receive relative to
singletons. Recommended starting value: λ = 10.

Dead codes (prototypes with zero assignments) are reinitialised from the
residual `r_i` with the highest reconstruction error, or kept unchanged
for one iteration to see if they get picked up in the next E-step.

### Step 5: Generate new SIDs

```
codes_new[i] = [c_i1, c_i2, c_i3, a(i)]     (4-token, raw codes)
```

Apply offset encoding (dataset.item2code convention) for TIGER vocab.

## Boundary Conditions

- Max collision bucket size ≈ 53 (well under 256) — injectivity always feasible
- Different collision buckets may reuse the same L4 code (constraint is per-bucket)
- All items receive an L4 code: collision items via Hungarian, singletons via argmin

## Experiment Plan

Generate SIDs on RQ-KMeans (best downstream so far) and RQ-VAE baseline, then run
TIGER CE-only on each:

| # | L4 source | λ | Hungarian | Argmin for singletons |
|---|-----------|---|-----------|----------------------|
| 0 | Random extra token (baseline) | — | No | — |
| 1 | Structured L4, equal weight | 1 | Yes | Yes |
| 2 | Structured L4, collision-weighted | 10 | Yes | Yes |

All experiments use frozen L1/L2/L3, same TIGER config (CE only, beam=30).

## Implementation

New standalone script: `rqvae/train_l4_codebook.py`

- Reads: trained RQ-KMeans/RQ-VAE checkpoint (or exported L1/L2/L3 codebooks), encoder outputs (32-d latent z_i), and existing L1/L2/L3 assignments
- Outputs: new `.npy` SID file with 4-token codes
- CLI flags for λ, L4 codebook size, max EM iterations
- Does NOT modify existing training code
