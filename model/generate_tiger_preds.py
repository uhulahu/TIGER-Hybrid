#!/usr/bin/env python3
"""Generate TIGER top-K item predictions for overlap analysis.

Loads a trained TIGER checkpoint, runs beam-search generation on the test set,
converts generated SIDs back to item IDs, and saves the top-K unique items
per user (with rank-based scores) as .npy files.

Usage:
    python model/generate_tiger_preds.py \
        --ckpt_path model/ckpt/Beauty_baseline_ce/Jul-09-2026_14-20-11/best_model.pth \
        --code_path data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003].npy \
        --output predictions/tiger_baseline_ce_top50.npy
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from main import TIGER


def build_sid_to_item(code_path):
    """Build reverse mapping: raw SID tuple → item ID (1-indexed)."""
    raw_codes = np.load(code_path)  # (V, 4), raw values in [0, 255]
    sid_to_item = {}
    for item_id, sid in enumerate(raw_codes):
        sid_tuple = tuple(int(s) for s in sid)
        sid_to_item[sid_tuple] = item_id + 1  # item IDs start at 1; 0 is padding
    return sid_to_item


def offset_to_raw_sid(offset_tokens, codebook_size=256):
    """Convert offset token values back to raw SID values.

    In dataset.item2code(), raw value c at layer i becomes:
        offset = c + i * codebook_size + 1

    We reverse this:  c = offset - 1 - i * codebook_size
    """
    return tuple(
        int(t) - 1 - layer * codebook_size
        for layer, t in enumerate(offset_tokens)
    )


def main():
    parser = argparse.ArgumentParser(description="Generate TIGER top-50 predictions")
    parser.add_argument('--ckpt_path', type=str, default='model/ckpt/Beauty_kmeans_L4_lam1/Jul-11-2026_15-59-27/best_model.pth',
                        help='Path to TIGER model checkpoint (.pth)')
    parser.add_argument('--code_path', type=str, default='data/Beauty/Beauty_kmeans_code-L4-lam1.npy',
                        help='Path to SID code file (.npy), same as used for training')
    parser.add_argument('--dataset_path', type=str, default='data/Beauty/test.parquet',
                        help='Path to test parquet file')
    parser.add_argument('--output', type=str, default='predictions/tiger_rqkmeans_ce_L4_lam1_top50.npy',
                        help='Output path for predictions .npy')
    parser.add_argument('--beam_size', type=int, default=100,
                        help='Beam size for generation (default: 50)')
    parser.add_argument('--topk', type=int, default=50,
                        help='Number of top unique items to save (default: 50)')
    parser.add_argument('--infer_size', type=int, default=96,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Model config (must match training) ──
    config = {
        'num_layers': 4,
        'num_decoder_layers': 4,
        'd_model': 128,
        'd_ff': 1024,
        'num_heads': 6,
        'd_kv': 64,
        'dropout_rate': 0.1,
        'vocab_size': 1025,
        'pad_token_id': 0,
        'eos_token_id': 0,
        'feed_forward_proj': 'relu',
    }

    # ── Load model ──
    print(f"Loading checkpoint: {args.ckpt_path}")
    model = TIGER(config)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"Model loaded ({model.n_parameters.split(chr(10))[2]})")

    # ── SID → item reverse mapping ──
    print(f"Building SID→item mapping from: {args.code_path}")
    sid_to_item = build_sid_to_item(args.code_path)
    print(f"  {len(sid_to_item)} unique SID→item mappings")

    # ── Test dataset ──
    test_ds = GenRecDataset(args.dataset_path, args.code_path, mode='evaluation', max_len=20)
    test_loader = GenRecDataLoader(test_ds, batch_size=args.infer_size, shuffle=False)
    print(f"Test samples: {len(test_ds)}")

    # ── Generate ──
    all_preds = []   # (N, topk) item IDs
    all_scores = []  # (N, topk) beam-confidence scores (log-prob)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating", ncols=90):
            input_ids = batch['history'].to(device)       # (B, pad_len)
            attention_mask = batch['attention_mask'].to(device)

            result = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=args.beam_size,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # sequences:  (B * beam, max_length)  — token IDs
            # sequences_scores: (B * beam,)  — cumulative log-prob
            sequences = result.sequences[:, 1:].cpu().numpy()          # strip start token → (B*beam, 4)
            seq_scores = result.sequences_scores.cpu().numpy()         # (B*beam,) log-prob

            sequences = sequences.reshape(input_ids.shape[0], args.beam_size, -1)  # (B, beam, 4)
            seq_scores = seq_scores.reshape(input_ids.shape[0], args.beam_size)     # (B, beam)

            for beam_results, beam_scores in zip(sequences, seq_scores):
                seen = set()
                item_preds = []
                item_scores = []
                for sid_tokens, score in zip(beam_results, beam_scores):
                    raw_sid = offset_to_raw_sid(sid_tokens)
                    item_id = sid_to_item.get(raw_sid, None)
                    if item_id is not None and item_id not in seen:
                        seen.add(item_id)
                        item_preds.append(item_id)
                        item_scores.append(float(score))
                        if len(item_preds) >= args.topk:
                            break

                # Pad short lists
                item_preds = item_preds[:args.topk]
                item_scores = item_scores[:args.topk]
                while len(item_preds) < args.topk:
                    item_preds.append(0)
                    item_scores.append(float('-inf'))
                all_preds.append(item_preds)
                all_scores.append(item_scores)

    all_preds = np.array(all_preds, dtype=np.int32)
    all_scores = np.array(all_scores, dtype=np.float32)
    # Replace -inf padding with a very small value for safe downstream use
    all_scores = np.where(np.isneginf(all_scores), -1e10, all_scores)
    print(f"Predictions shape: {all_preds.shape}")
    print(f"Unique items used: {np.setdiff1d(np.unique(all_preds), [0]).size}")
    print(f"Rows with <{args.topk} predictions: {(all_preds == 0).any(axis=1).sum()}")

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.save(args.output, all_preds)
    score_path = args.output.replace('.npy', '_scores.npy')
    np.save(score_path, all_scores)
    print(f"Saved preds to {args.output}")
    print(f"Saved scores to {score_path}")


if __name__ == '__main__':
    main()
