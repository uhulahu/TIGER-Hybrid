#!/usr/bin/env python3
"""Analyse prefix-3 hit vs full hit to diagnose extra-token bottleneck.

If the model frequently generates the correct L1+L2+L3 but misses the
extra token, then the extra token is the weak link — and bucket-level
generation + content-similarity re-ranking would be a free improvement.

Usage:
    python model/analyze_prefix_hit.py \
        --ckpt model/ckpt/Beauty_baseline_ce/best_model.pth \
        --dataset data/Beauty \
        --code_path data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003].npy
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from main import TIGER, set_seed
from dataset import GenRecDataset
from dataloader import GenRecDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='model/ckpt/Beauty/Jul-09-2026_21-23-03/best_model.pth', help='Path to TIGER model checkpoint.')
parser.add_argument('--dataset', type=str, default='data/Beauty')
parser.add_argument('--code_path', type=str,
                    default='data/Beauty/Beauty_kmeans_code.npy')
parser.add_argument('--beam_size', type=int, default=30)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# ── Load model ──
config = dict(
    num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, feed_forward_proj='relu',
)
model = TIGER(config).to(device)
ckpt = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt)
model.eval()
print(f"Model loaded from {args.ckpt}")

# ── Load test data ──
test_dataset = GenRecDataset(
    f'{args.dataset}/test.parquet', args.code_path,
    mode='evaluation', max_len=20)
test_loader = GenRecDataLoader(test_dataset, batch_size=96, shuffle=False)
print(f"Test items: {len(test_dataset)}")

# ── Evaluate ──
total = 0
prefix3_hit = 0   # any beam has correct L1+L2+L3
full_hit = 0      # any beam has correct L1+L2+L3+extra
extra_correct_given_prefix3 = 0  # given prefix3 is correct in some beam, extra also correct
prefix3_correct_beams = 0  # number of cases where at least one beam has correct prefix3

ranks_prefix3 = []  # at which beam position (0-indexed) is the correct prefix3 found?
ranks_full = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Analyzing'):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)  # (B, 4) full SID tokens

        B = input_ids.shape[0]

        # Generate
        preds = model.generate(input_ids, attention_mask, num_beams=args.beam_size)
        preds = preds[:, 1:]  # strip start token
        preds = preds.reshape(B, args.beam_size, -1)  # (B, beam, 4)

        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        for i in range(B):
            total += 1
            label = labels_np[i]           # (4,) full SID
            label_prefix3 = label[:3]       # (3,) L1+L2+L3

            found_prefix3 = False
            found_full = False
            rank_p3 = args.beam_size
            rank_f = args.beam_size

            for j in range(args.beam_size):
                pred = preds_np[i, j]       # (4,)

                if not found_prefix3 and np.array_equal(pred[:3], label_prefix3):
                    found_prefix3 = True
                    rank_p3 = j

                if not found_full and np.array_equal(pred, label):
                    found_full = True
                    rank_f = j

                if found_prefix3 and found_full:
                    break

            if found_prefix3:
                prefix3_hit += 1

            if found_full:
                full_hit += 1
                extra_correct_given_prefix3 += 1
            elif found_prefix3:
                # prefix correct but extra wrong — the gap we care about
                pass

            ranks_prefix3.append(rank_p3 if found_prefix3 else args.beam_size)
            ranks_full.append(rank_f if found_full else args.beam_size)

# ── Stats ──
print()
print("=" * 65)
print("  Prefix-3 vs Full Hit Analysis")
print("=" * 65)
print(f"  Test items:               {total}")
print(f"  Prefix-3 hit (Recall@{args.beam_size}):  {prefix3_hit/total:.4f}  ({prefix3_hit}/{total})")
print(f"  Full hit    (Recall@{args.beam_size}):  {full_hit/total:.4f}  ({full_hit}/{total})")
print(f"  Gap (prefix3 hit - full hit):          {(prefix3_hit - full_hit)/total:.4f}  ({prefix3_hit - full_hit}/{total})")
print(f"  Extra correct given prefix3 correct:    {full_hit/prefix3_hit:.4f}  ({full_hit}/{prefix3_hit})")

# How many cases would be "rescued" by dropping extra token + bucket re-rank?
print()
print(f"  Items where prefix3 was found but extra token missed:")
gap = prefix3_hit - full_hit
print(f"    Count: {gap}  ({100*gap/total:.2f}% of test set)")

# Rank distribution
r3 = np.array(ranks_prefix3)
rf = np.array(ranks_full)
for k in [1, 5, 10, 20]:
    r3_k = (r3 < k).mean()
    rf_k = (rf < k).mean()
    print(f"  Prefix-3 Recall@{k:>2}: {r3_k:.4f}    Full Recall@{k:>2}: {rf_k:.4f}    gap: {r3_k - rf_k:.4f}")

# Breakdown: among full-miss items, how many have correct prefix3?
full_miss = total - full_hit
prefix3_but_not_full = prefix3_hit - full_hit
print(f"\n  Full-hit misses: {full_miss}")
print(f"    Of these, prefix3 was correct: {prefix3_but_not_full} ({100*prefix3_but_not_full/full_miss:.1f}%)")
print(f"    Of these, prefix3 was also wrong: {full_miss - prefix3_but_not_full} ({100*(full_miss-prefix3_but_not_full)/full_miss:.1f}%)")
print()
print("=" * 65)
if prefix3_but_not_full / max(full_miss, 1) > 0.05:
    print("  CONCLUSION: Extra token is a meaningful bottleneck.")
    print("  Bucket-level generation + content re-ranking is worth trying.")
else:
    print("  CONCLUSION: Extra token is NOT the main bottleneck.")
    print("  Prefix-3 errors dominate — focus on improving L1/L2/L3 prediction.")
print("=" * 65)

# 在tiger_baseline_ce_rqkmeans上做了一个前三层正确率与完整 SID 正确率的分析，然后发现好像并不是我们想象的那样：

# =================================================================
#   Prefix-3 vs Full Hit Analysis
# =================================================================

#   Test items:               22363
#   Prefix-3 hit (Recall@30):  0.1128  (2522/22363)
#   Full hit    (Recall@30):  0.1066  (2384/22363)
#   Gap (prefix3 hit - full hit):          0.0062  (138/22363)
#   Extra correct given prefix3 correct:    0.9453  (2384/2522)

#   Items where prefix3 was found but extra token missed:
#     Count: 138  (0.62% of test set)
#   Prefix-3 Recall@ 1: 0.0116    Full Recall@ 1: 0.0111    gap: 0.0005
#   Prefix-3 Recall@ 5: 0.0388    Full Recall@ 5: 0.0364    gap: 0.0024
#   Prefix-3 Recall@10: 0.0609    Full Recall@10: 0.0577    gap: 0.0032
#   Prefix-3 Recall@20: 0.0918    Full Recall@20: 0.0875    gap: 0.0043

#   Full-hit misses: 19979
#     Of these, prefix3 was correct: 138 (0.7%)
#     Of these, prefix3 was also wrong: 19841 (99.3%)