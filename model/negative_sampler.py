"""Negative sampler for list-wise / first-diff ranking loss.

Four strategies covering the full difficulty spectrum:
  1. Random    — uniform sample from all items (baseline)
  2. Popular   — sample from top-20% most frequent items (popularity debias)
  3. Same-L1   — share L1 prefix (medium-hard)
  4. Same-L1L2 — share L1+L2 prefix (hardest)

All sampling is fully vectorised (torch ops, no Python loops over batch dim).
"""

from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch


class NegativeSampler:
    """Pre-computes item statistics and samples negative SIDs per batch."""

    def __init__(self, code_path: str, train_path: str,
                 num_per_strategy: int = 4, codebook_size: int = 256):
        self.num_per_strategy = num_per_strategy
        self.codebook_size = codebook_size

        # ── load all SIDs ──
        raw = np.load(code_path)                                      # (N_items, 4)
        offsets = []
        for i in range(raw.shape[1]):
            offsets.append(raw[:, i] + i * codebook_size + 1)
        self.all_sids = torch.tensor(np.stack(offsets, axis=1))       # (N, 4)
        self.N = len(self.all_sids)

        # sid_tuple → item_index (0-based)
        self.sid_to_idx = {}
        for idx in range(self.N):
            tup = tuple(self.all_sids[idx].tolist())
            if tup not in self.sid_to_idx:
                self.sid_to_idx[tup] = idx

        # ── popular items (top 20% by training frequency) ──
        df = pd.read_parquet(train_path)
        counter = Counter()
        for hist in df['history']:
            counter.update(hist)
        for tgt in df['target']:
            counter[tgt] += 1
        n_pop = max(len(counter) // 5, 100)
        pop_ids = [i for i, _ in counter.most_common(n_pop)]
        # popular items are 1-indexed; raw codes use 0-indexed rows
        popular_list = [pid - 1 for pid in pop_ids if 1 <= pid <= len(raw)]
        self.popular_tensor = torch.tensor(popular_list, dtype=torch.long)
        self.P = len(self.popular_tensor)

        # ── prefix-group tensors (padded for vectorised gather) ──
        # L1 groups
        l1_groups = defaultdict(list)
        l1l2_groups = defaultdict(list)
        for idx in range(self.N):
            l1_groups[int(raw[idx, 0])].append(idx)
            l1l2_groups[(int(raw[idx, 0]), int(raw[idx, 1]))].append(idx)

        # Pad L1 groups → (256, max_size)
        max_l1 = max(len(v) for v in l1_groups.values())
        self.l1_group = torch.full((codebook_size, max_l1), -1, dtype=torch.long)
        self.l1_size = torch.zeros(codebook_size, dtype=torch.long)
        for l1, members in l1_groups.items():
            self.l1_group[l1, :len(members)] = torch.tensor(members)
            self.l1_size[l1] = len(members)

        # Pad L1L2 groups → flattened (256*256, max_size)
        max_l1l2 = max(len(v) for v in l1l2_groups.values())
        self.l1l2_group = torch.full(
            (codebook_size * codebook_size, max_l1l2), -1, dtype=torch.long)
        self.l1l2_size = torch.zeros(
            codebook_size * codebook_size, dtype=torch.long)
        for (l1, l2), members in l1l2_groups.items():
            flat = l1 * codebook_size + l2
            self.l1l2_group[flat, :len(members)] = torch.tensor(members)
            self.l1l2_size[flat] = len(members)

    # ── vectorised strategy methods ────────────────────────────────────────

    def _sample_random(self, pos_indices, num=None):
        """Random negatives — fully vectorised."""
        n = num if num is not None else self.num_per_strategy
        B = len(pos_indices)
        pos_t = torch.as_tensor(pos_indices, dtype=torch.long)

        samples = torch.randint(0, self.N, (B, n))
        # Rare collision fix: shift by 1 (wrap) for entries that hit the positive
        collision = samples == pos_t.unsqueeze(1)
        samples[collision] = (samples[collision] + 1) % self.N

        return self.all_sids[samples]                                  # (B, n, 4)

    def _sample_popular(self, pos_indices, num=None):
        """Popular negatives — vectorised via pre-computed popular tensor."""
        n = num if num is not None else self.num_per_strategy
        B = len(pos_indices)
        pos_t = torch.as_tensor(pos_indices, dtype=torch.long)

        # Random positions into the popular list
        idx = torch.randint(0, self.P, (B, n))
        samples = self.popular_tensor[idx]                             # (B, n)

        # Collision fix
        collision = samples == pos_t.unsqueeze(1)
        if collision.any():
            shifted = (idx + 1) % self.P
            samples[collision] = self.popular_tensor[shifted[collision]]

        return self.all_sids[samples]                                  # (B, n, 4)

    def _sample_same_l1(self, pos_indices, raw_l1, num=None):
        """Same-L1 negatives — vectorised via padded group tensor."""
        n = num if num is not None else self.num_per_strategy
        B = len(pos_indices)
        raw_l1_t = torch.as_tensor(raw_l1, dtype=torch.long)
        pos_t = torch.as_tensor(pos_indices, dtype=torch.long)

        sizes = self.l1_size[raw_l1_t]                                 # (B,)
        # Clamp size ≥ 1 to avoid div-by-zero on empty groups
        safe_size = sizes.clamp(min=1)

        rand = torch.rand(B, n)
        idx = (rand * safe_size.unsqueeze(1)).long().clamp(
            max=safe_size.unsqueeze(1) - 1)

        samples = self.l1_group[raw_l1_t.unsqueeze(1), idx]            # (B, n)

        # Collision fix
        collision = samples == pos_t.unsqueeze(1)
        if collision.any():
            shifted = (idx + 1) % safe_size.unsqueeze(1)
            samples[collision] = self.l1_group[
                raw_l1_t.unsqueeze(1).expand(-1, n)[collision],
                shifted[collision]]

        return self.all_sids[samples]                                  # (B, n, 4)

    def _sample_same_l1l2(self, pos_indices, raw_l1l2, num=None):
        """Same-L1L2 negatives — vectorised via padded group tensor."""
        n = num if num is not None else self.num_per_strategy
        B = len(pos_indices)
        raw_l1l2_t = torch.as_tensor(raw_l1l2, dtype=torch.long)
        pos_t = torch.as_tensor(pos_indices, dtype=torch.long)

        flat_keys = (raw_l1l2_t[:, 0] * self.codebook_size
                     + raw_l1l2_t[:, 1])                               # (B,)
        sizes = self.l1l2_size[flat_keys]                              # (B,)
        safe_size = sizes.clamp(min=1)

        rand = torch.rand(B, n)
        idx = (rand * safe_size.unsqueeze(1)).long().clamp(
            max=safe_size.unsqueeze(1) - 1)

        samples = self.l1l2_group[flat_keys.unsqueeze(1), idx]         # (B, n)

        # Collision fix
        collision = samples == pos_t.unsqueeze(1)
        if collision.any():
            shifted = (idx + 1) % safe_size.unsqueeze(1)
            samples[collision] = self.l1l2_group[
                flat_keys.unsqueeze(1).expand(-1, n)[collision],
                shifted[collision]]

        return self.all_sids[samples]                                  # (B, n, 4)

    # ── public API ─────────────────────────────────────────────────────────

    def sample(self, pos_labels):
        """每个正样本采 4 个负样本：random / popular / same-L1 / same-L1L2.

        Args:
            pos_labels: (B, 4)  正样本 SID tokens（已做 offset 编码）

        Returns:
            neg_labels: (B, 4, 4)  [rand, pop, L1, L1L2]
        """
        # ── vectorised pos_indices + raw_codes ──
        pos_tuples = [tuple(row.tolist()) for row in pos_labels]
        pos_indices = np.array([
            self.sid_to_idx.get(t, idx % self.N)
            for idx, t in enumerate(pos_tuples)
        ])

        # raw_codes: strip offset to recover codebook indices
        raw_codes = pos_labels.cpu().numpy() - 1                     # (B, 4)
        for lvl in range(4):
            raw_codes[:, lvl] -= lvl * self.codebook_size

        # 每种策略 1 个，共 4 个负样本
        rand = self._sample_random(pos_indices, num=1)                 # (B, 1, 4)
        pop  = self._sample_popular(pos_indices, num=1)                # (B, 1, 4)
        l1   = self._sample_same_l1(pos_indices,
                                     raw_codes[:, 0], num=1)          # (B, 1, 4)
        l1l2 = self._sample_same_l1l2(pos_indices,
                                       raw_codes[:, :2], num=1)       # (B, 1, 4)

        return torch.cat([rand, pop, l1, l1l2], dim=1)                 # (B, 4, 4)

    def refresh_model_negs(self, model, input_ids, attention_mask,
                           pos_labels, beam_size=30):
        """Generate model-based negatives from the current batch.

        Runs beam search and returns non-positive SIDs as negatives.
        Call this every K batches during training.

        Returns:
            model_negs: (B, num_per_strategy, 4)
        """
        B = pos_labels.shape[0]
        device = pos_labels.device

        with torch.no_grad():
            preds = model.generate(input_ids, attention_mask,
                                   num_beams=beam_size)
            preds = preds[:, 1:]                            # strip start token
            preds = preds.reshape(B, beam_size, -1)         # (B, beam, 4)

        negs = []
        for b in range(B):
            pos = pos_labels[b]
            batch_negs = []
            for k in range(beam_size):
                if not torch.equal(preds[b, k], pos):
                    batch_negs.append(preds[b, k])
                if len(batch_negs) >= self.num_per_strategy:
                    break
            while len(batch_negs) < self.num_per_strategy:
                batch_negs.append(batch_negs[-1] if batch_negs else pos)
            negs.append(torch.stack(batch_negs))

        return torch.stack(negs).to(device)                 # (B, num, 4)
