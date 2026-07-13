"""Stage 1 inference: collect raw data for L4 analysis (K0 vs K2).

Saves catalog metadata, teacher-forcing logits, beam4 and beam3 outputs
for both models.  All files share `sample_id` for alignment.

Usage:
    python analysis/inference.py
"""

import os, sys, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from main import TIGER
from dataset import GenRecDataset
from dataloader import GenRecDataLoader

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda'
BEAM_SIZE = 30
BATCH_SIZE = 96
OUT_DIR = 'analysis/raw'

MODELS = {
    'k0': {
        'ckpt': 'model/ckpt/Beauty/Jul-09-2026_21-23-03/best_model.pth',
        'code_path': 'data/Beauty/Beauty_kmeans_code.npy',
    },
    'k2': {
        'ckpt': 'model/ckpt/Beauty_kmeans_L4_lam10/Jul-10-2026_23-25-51/best_model.pth',
        'code_path': 'data/Beauty/Beauty_kmeans_code-L4-lam10.npy',
    },
}

TIGER_CONFIG = dict(
    num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, feed_forward_proj='relu',
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    model = TIGER(TIGER_CONFIG).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"  Saved {path}  ({len(arrays)} arrays)")


# ── Catalog ───────────────────────────────────────────────────────────────────

def build_catalog(code_path):
    codes = np.load(code_path).astype(np.int64)  # (N, 4), raw 0-255
    N = len(codes)
    # prefix_bucket_size
    from collections import Counter
    prefix3 = codes[:, :3]
    prefix_tuples = [tuple(row) for row in prefix3]
    bucket_cnt = Counter(prefix_tuples)
    bucket_size = np.array([bucket_cnt[t] for t in prefix_tuples], dtype=np.int32)

    return {
        'item_id': np.arange(1, N + 1, dtype=np.int32),
        'codes': codes,
        'prefix3': prefix3,
        'l4_code': codes[:, 3],
        'prefix_bucket_size': bucket_size,
    }


# ── Teacher-forcing ───────────────────────────────────────────────────────────

@torch.no_grad()
def collect_teacher_forcing(model, loader, device):
    all_ids = []
    all_logits_l1, all_logits_l2, all_logits_l3, all_logits_l4 = [], [], [], []
    all_target_codes = []

    for batch in tqdm(loader, desc='  Teacher-forcing', ncols=80):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)  # (B, 4) offset-encoded

        _, logits = model(input_ids, attention_mask, labels=labels)
        # logits: (B, 4, vocab)
        logits = logits.cpu().half()  # float16 to save space

        all_ids.append(batch.get('sample_id', torch.arange(len(labels))))
        all_target_codes.append(labels.cpu())
        all_logits_l1.append(logits[:, 0, :])
        all_logits_l2.append(logits[:, 1, :])
        all_logits_l3.append(logits[:, 2, :])
        all_logits_l4.append(logits[:, 3, :])

    # Convert target codes from offset to raw (0-255)
    target_raw = torch.cat(all_target_codes, dim=0).numpy().astype(np.int64)
    for lvl in range(4):
        target_raw[:, lvl] = target_raw[:, lvl] - lvl * 256 - 1

    return {
        'sample_id': np.arange(len(target_raw), dtype=np.int32),
        'target_codes': target_raw,
        'logits_l1': torch.cat(all_logits_l1, dim=0).numpy(),
        'logits_l2': torch.cat(all_logits_l2, dim=0).numpy(),
        'logits_l3': torch.cat(all_logits_l3, dim=0).numpy(),
        'logits_l4': torch.cat(all_logits_l4, dim=0).numpy(),
    }


# ── Beam search ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_beam4(model, loader, device, beam_size):
    all_beam_codes = []
    all_beam_scores = []
    all_target_codes = []

    for batch in tqdm(loader, desc='  Beam-4', ncols=80):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        B = input_ids.shape[0]

        # Use HF generate with scores
        outputs = model.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=5, num_beams=beam_size,
            num_return_sequences=beam_size,
            output_scores=True, return_dict_in_generate=True,
        )
        # sequences: (B*beam, seq_len), strip start token
        seqs = outputs.sequences[:, 1:].reshape(B, beam_size, -1)  # (B, beam, 4)
        # Convert offset → raw
        seqs_raw = seqs.cpu().numpy().astype(np.int64)
        for lvl in range(4):
            seqs_raw[:, :, lvl] = seqs_raw[:, :, lvl] - lvl * 256 - 1

        # Beam scores: use sequences_scores or compute from logprobs
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            scores = outputs.sequences_scores.reshape(B, beam_size).cpu().numpy()
        else:
            scores = np.zeros((B, beam_size), dtype=np.float32)

        labels_raw = labels.cpu().numpy().astype(np.int64)
        for lvl in range(4):
            labels_raw[:, lvl] = labels_raw[:, lvl] - lvl * 256 - 1

        all_beam_codes.append(seqs_raw)
        all_beam_scores.append(scores)
        all_target_codes.append(labels_raw)

    return {
        'sample_id': np.arange(len(np.concatenate(all_target_codes)), dtype=np.int32),
        'target_codes': np.concatenate(all_target_codes, axis=0),
        'beam_codes': np.concatenate(all_beam_codes, axis=0),
        'beam_scores': np.concatenate(all_beam_scores, axis=0),
    }


@torch.no_grad()
def collect_beam3(model, loader, device, beam_size):
    all_beam_prefixes = []
    all_beam_scores = []
    all_target_prefix3 = []

    for batch in tqdm(loader, desc='  Beam-3', ncols=80):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        B = input_ids.shape[0]

        outputs = model.generate_prefix3(
            input_ids, attention_mask, num_beams=beam_size)
        # outputs: (B*beam, 4) — [BOS, L1, L2, L3] or [BOS, token1, token2, token3]
        # Strip start token → (B*beam, 3)
        seqs = outputs[:, 1:].reshape(B, beam_size, -1)
        seqs_raw = seqs.cpu().numpy().astype(np.int64)
        for lvl in range(3):
            seqs_raw[:, :, lvl] = seqs_raw[:, :, lvl] - lvl * 256 - 1

        labels_raw = labels.cpu().numpy().astype(np.int64)
        target_p3 = np.zeros((B, 3), dtype=np.int64)
        for lvl in range(3):
            target_p3[:, lvl] = labels_raw[:, lvl] - lvl * 256 - 1

        # No beam scores available from generate_prefix3 (simple generate)
        all_beam_prefixes.append(seqs_raw)
        all_beam_scores.append(np.zeros((B, beam_size), dtype=np.float32))
        all_target_prefix3.append(target_p3)

    return {
        'sample_id': np.arange(len(np.concatenate(all_target_prefix3)), dtype=np.int32),
        'target_prefix3': np.concatenate(all_target_prefix3, axis=0),
        'beam_prefixes': np.concatenate(all_beam_prefixes, axis=0),
        'beam_scores': np.concatenate(all_beam_scores, axis=0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    for name, cfg in MODELS.items():
        print(f"\n{'='*60}\n  Model: {name}\n{'='*60}")

        model = load_model(cfg['ckpt'], device)

        # ── catalog ──
        print("Catalog...")
        catalog = build_catalog(cfg['code_path'])
        save_npz(f'{OUT_DIR}/catalog_{name}.npz', **catalog)

        # Load test data — aligned by sample order
        test_dataset = GenRecDataset(
            'data/Beauty/test.parquet', cfg['code_path'],
            mode='evaluation', max_len=20)
        test_loader = GenRecDataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"  Test items: {len(test_dataset)}")

        # ── teacher-forcing ──
        print("Teacher-forcing...")
        tf = collect_teacher_forcing(model, test_loader, device)
        save_npz(f'{OUT_DIR}/{name}_teacher_forcing.npz', **tf)

        # ── beam4 ──
        print(f"Beam-4 (size={BEAM_SIZE})...")
        b4 = collect_beam4(model, test_loader, device, BEAM_SIZE)
        save_npz(f'{OUT_DIR}/{name}_beam4.npz', **b4)

        # ── beam3 ──
        print(f"Beam-3 (size={BEAM_SIZE})...")
        b3 = collect_beam3(model, test_loader, device, BEAM_SIZE)
        save_npz(f'{OUT_DIR}/{name}_beam3.npz', **b3)

    # ── Sanity check ──
    print("\nSanity check...")
    k0_tf = np.load(f'{OUT_DIR}/k0_teacher_forcing.npz')
    k2_tf = np.load(f'{OUT_DIR}/k2_teacher_forcing.npz')
    assert np.array_equal(k0_tf['target_codes'][:, :3],
                          k2_tf['target_codes'][:, :3]), \
        "K0 and K2 prefix-3 mismatch!"
    print("  K0/K2 target_codes[:,:3] match ✓")
    print("Done.")


if __name__ == '__main__':
    main()
