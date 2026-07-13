import datetime

import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import random
from tqdm import tqdm
import logging
from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from negative_sampler import NegativeSampler

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
            num_layers=config['num_layers'],  # encoder 层数 (T5Block数)
            num_decoder_layers=config['num_decoder_layers'], # decoder 层数 (T5Block数)
            d_model=config['d_model'],  # transformer隐藏层维度，也决定Q的维度
            d_ff=config['d_ff'],  # FFN 中间层维度 d_model (128) → d_ff (1024) → ReLU → d_model (128)
            num_heads=config['num_heads'],  # 注意力头数
            d_kv=config['d_kv'], # Key/Value 向量的维度
            dropout_rate=config['dropout_rate'],  # 0.1
            vocab_size=config['vocab_size'], # 词表大小：1 + 256*4 = 1025 (用于PAD/EOS的Token0 + SID中四个token位，每位256维)
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'], # 自回归生成的停止token，这里是0，但SID不会出现0，所以EOS不会被触发
            decoder_start_token_id=config['pad_token_id'], # 自回归生成的起始token
            feed_forward_proj=config['feed_forward_proj'], # relu
        )
        # Initialize T5 model with the specified configuration
        self.model = T5ForConditionalGeneration(t5config)
    
    @property
    def n_parameters(self):
      """Calculates the number of trainable parameters in the model.

      Returns:
          str: A string containing the number of embedding parameters,
          non-embedding parameters, and total trainable parameters.
      """
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())  # 4590848, 4.59 M
      emb_params = num_params(self.model.get_input_embeddings().parameters()) # 131200, 0.13B
      return (
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}'
      )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
      """Forward pass of the model. Returns the output logits and the loss value.

      Args:
          batch (dict): A dictionary containing the input data for the model.

      Returns:
          outputs (ModelOutput):
              The output of the model, which includes:
              - loss (torch.Tensor)
              - logits (torch.Tensor)
      """
      outputs = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
      )
      return outputs.loss, outputs.logits
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,  num_beams: int = 20, **kwargs):
        """Generate recommendations using the model.

        Args:
            input_ids (torch.Tensor): Input tensor for the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.
            max_length (int): Maximum length of the generated sequence.
            num_beams (int): Number of beams for beam search.

        Returns:
            torch.Tensor: Generated output tensor.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,  # 指定decoder输入侧序列长度 (decoder_start_token, SID) 
            # 需要注意的是，decoder_start_token只出现在decoder输入侧作为起始条件
            # 输入的时候我们是吧(batch_size, sid_len)展平为(batchsize, n*sid_len)的，
            # 就是说模型不需要知道SID长度为4，它看到的就是SIDs是连在一起形成的一维token流，
            # 训练时学习的是"给定一维 token 历史，预测后续 token"。
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs
        )
    

    def generate_prefix3(self, input_ids, attention_mask, num_beams=30):
        """Generate only the first 3 tokens (L1, L2, L3) — no extra token.

        Used for bucket-level analysis: given the 3-token prefix from beam
        search, expand to all items in that bucket for content re-ranking.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=4,              # decoder_start + L1 + L2 + L3 = 4
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )

    def compute_combined_loss(self, input_ids, attention_mask,
                               pos_labels, neg_labels, temperature=1.0):
        """CE + listwise loss from a single forward pass over all candidates.

        The encoder input is repeated C times so that positives (col 0)
        and negatives (cols 1..C-1) share one decoder call.  CE is
        extracted from the positive slice of the resulting logits.

        Returns:
            ce_loss:   scalar
            list_loss: scalar
        """
        B = input_ids.shape[0]
        N = neg_labels.shape[1]
        C = 1 + N

        # ── pack all candidates: pos at col 0 ──
        all_labels = torch.cat([pos_labels.unsqueeze(1), neg_labels], dim=1)
        all_labels_flat = all_labels.reshape(B * C, 4)

        # ── single forward pass ──
        input_rep = input_ids.repeat_interleave(C, dim=0)       # (B*C, S)
        mask_rep  = attention_mask.repeat_interleave(C, dim=0)   # (B*C, S)

        _, logits_all = self(input_rep, mask_rep, labels=all_labels_flat)
        # logits_all: (B*C, 4, vocab)

        # ── CE loss: positive slice (every C-th item) ──
        pos_logits = logits_all.view(B, C, 4, -1)[:, 0]         # (B, 4, vocab)
        ce_loss = F.cross_entropy(
            pos_logits.reshape(B * 4, -1),
            pos_labels.reshape(B * 4),
            ignore_index=-100,
        )

        # ── listwise loss ──
        lp = F.log_softmax(logits_all, dim=-1)    # (B*C, 4, vocab)
        token_lp = lp.gather(
            -1, all_labels_flat.unsqueeze(-1)).squeeze(-1)      # (B*C, 4)
        seq_scores = token_lp.sum(dim=-1).view(B, C)            # (B, C)

        list_loss = -F.log_softmax(seq_scores / temperature, dim=1)[:, 0].mean()

        return ce_loss, list_loss

    def compute_first_diff_loss(self, logits, pos_labels, neg_labels):
        """First-difference pairwise loss — efficient approximation of list-wise.

        For each (positive, negative) pair, finds the first token position where
        they differ, then applies a pairwise logistic loss at that position::

            ℓ = -log σ( logit(c⁺_t) - logit(c⁻_t) )

        where *t* is the first position where the two SIDs diverge.  This reuses
        ``logits`` from the standard CE teacher-forcing forward pass — **zero
        extra decoder calls**.

        Intuition: only the first-diverging decision matters; once the model
        picks the wrong token at position *t*, the remaining positions are on a
        different beam and their token-level losses are not directly comparable.

        Args:
            logits:     (B, 4, vocab) from standard CE forward pass.
            pos_labels: (B, 4)   positive SID tokens.
            neg_labels: (B, N, 4) negative SID tokens.

        Returns:
            scalar loss (0.0 when no valid pairs exist, e.g. pos == neg).
        """
        B, N = neg_labels.shape[0], neg_labels.shape[1]

        # ── first-differing position per (positive, negative) pair ──
        diff_mask = (neg_labels != pos_labels.unsqueeze(1))        # (B, N, 4)
        first_diff_pos = diff_mask.float().argmax(dim=2)            # (B, N)
        any_diff = diff_mask.any(dim=2)                              # (B, N)

        if not any_diff.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # ── logits at the first-differing position ──
        b_idx = torch.arange(B, device=logits.device).view(B, 1).expand(-1, N)
        logits_at_pos = logits[b_idx, first_diff_pos]               # (B, N, vocab)

        # Positive / negative token id at that position
        pos_token = pos_labels.gather(1, first_diff_pos).long()     # (B, N)
        neg_token = neg_labels.gather(                              # (B, N)
            2, first_diff_pos.unsqueeze(-1)).squeeze(-1).long()

        # Logit of the positive vs negative token
        pos_logit = logits_at_pos.gather(
            -1, pos_token.unsqueeze(-1)).squeeze(-1)                # (B, N)
        neg_logit = logits_at_pos.gather(
            -1, neg_token.unsqueeze(-1)).squeeze(-1)                # (B, N)

        # Pairwise logistic loss
        loss = -F.logsigmoid(pos_logit - neg_logit)                 # (B, N)
        loss = (loss * any_diff.float()).sum() / any_diff.sum().clamp(min=1)

        return loss

    def compute_listwise_loss(self, input_ids, attention_mask,
                               pos_labels, neg_labels, temperature=1.0):
        """
        计算list-wise损失。
        List-wise softmax cross-entropy over positive + negative SID candidates.

        For each item in the batch:
        1. Encodes the interaction history once.
        2. Computes the sequence-level log-probability of every candidate SID
           (positive + negatives) via teacher forcing through the decoder.
        3. Applies softmax cross-entropy so the positive item is pushed above
           all negatives in the ranked list.

        Args:
            input_ids:      (B, S)      history token sequences
            attention_mask: (B, S)
            pos_labels:     (B, 4)      ground-truth next-item SID tokens
            neg_labels:     (B, N, 4)   negative SID tokens
            temperature:    softmax temperature (1.0 = no scaling)

        Returns:
            scalar loss
        """
        B = input_ids.shape[0]
        N = neg_labels.shape[1]                     # negatives per item
        C = 1 + N                                   # total candidates per item (1个正样本+N个负样本)

        # ── pack all candidates: positive at column 0 ── 正样本标签放在第0列，负样本标签拼接在后面
        all_labels = torch.cat([pos_labels.unsqueeze(1), neg_labels], dim=1)
        all_labels_flat = all_labels.view(B * C, 4)  # (B*C, 4)

        # ── encoder (once per item) ──
        enc = self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        enc_hidden = enc.last_hidden_state                         # (B, S, d_model)

        # ── repeat for each candidate ──
        enc_hidden = enc_hidden.repeat_interleave(C, dim=0)        # (B*C, S, d_model)
        enc_mask   = attention_mask.repeat_interleave(C, dim=0)    # (B*C, S)
        # ⭐复用encoder，decoder无法复用

        # ── decoder inputs: teacher forcing on each candidate SID ──
        dec_ids = self.model._shift_right(all_labels_flat)         # (B*C, 4)

        dec = self.model.decoder(
            input_ids=dec_ids,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
        )
        dec_hidden = dec.last_hidden_state                         # (B*C, 4, d_model)

        # ── per-sequence log-probability (no temperature here) ──
        logits = self.model.lm_head(dec_hidden)                    # (B*C, 4, vocab)
        log_probs = F.log_softmax(logits, dim=-1)                  # (B*C, 4, vocab)
        token_lp = log_probs.gather(
            -1, all_labels_flat.unsqueeze(-1)).squeeze(-1)         # (B*C, 4)
        seq_scores = token_lp.sum(dim=-1)                          # (B*C,)
        seq_scores = seq_scores.view(B, C)                         # (B, C)

        # ── list-wise softmax CE with temperature ──
        # Temperature on the *list* softmax controls how sharply the loss
        # penalises hard negatives vs spreads across all negatives:
        #   T < 1 → dominated by the hardest (closest-scoring) negatives
        #   T > 1 → softer, more uniform penalty across negatives
        loss = -F.log_softmax(seq_scores / temperature, dim=1)[:, 0].mean()
        return loss


class InverseSquareRootScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super(InverseSquareRootScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return self.base_lrs
        scale_factor = (self.warmup_steps**0.5) / (step**0.5)
        return [base_lr * scale_factor for base_lr in self.base_lrs]
    

def calculate_pos_index(preds, labels, maxk=20):
    """Calculate the position index of the ground truth items.

    Args:
      preds: The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels: The ground truth token sequences, of shape (batch_size, seq_len).

    Returns:
      A boolean tensor of shape (batch_size, maxk) indicating whether the
      prediction at each position is correct.
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      for j in range(maxk):
        cur_pred = preds[i, j].tolist()
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

def recall_at_k(pos_index, k):
  """Recall@k (single-target → equivalent to HitRate@k).

  Each row has exactly one True; returns 1.0 if it falls in the top-k, else 0.
  """
  return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
  """NDCG@k — normalised discounted cumulative gain with a single relevant item
  per row (so IDCG = 1 / log₂(2) = 1 and NDCG reduces to the DCG value)."""
  ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
  dcg = 1.0 / torch.log2(ranks + 1)
  dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
  return dcg[:, :k].sum(dim=1).cpu().float()

def mrr_at_k(pos_index, k):
  """MRR@k — Mean Reciprocal Rank.

  For each row, if the ground-truth item is found at 1-indexed rank *r* ≤ k,
  contribute 1/r; otherwise contribute 0.
  """
  maxk = pos_index.shape[1]
  ranks = torch.arange(1, maxk + 1, device=pos_index.device).float()
  # rank of the True per row, inf if no True anywhere
  rank_per_row = torch.where(
      pos_index.any(dim=1),
      ranks[pos_index.float().argmax(dim=1)],
      torch.tensor(float('inf'), device=pos_index.device),
  )
  mrr = torch.where(
      rank_per_row <= k,
      1.0 / rank_per_row,
      torch.tensor(0.0, device=pos_index.device),
  )
  return mrr.cpu().float()

def train(model, train_loader, optimizer, device, scheduler=None,
          neg_sampler=None, config=None):
    """Training loop with optional ranking loss (list-wise or first-difference)."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0          # auxiliary loss (listwise or first-diff)

    use_listwise = config.get('use_listwise_loss', False) and neg_sampler is not None
    use_first_diff = config.get('use_first_diff_loss', False) and neg_sampler is not None

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['history'].to(device)     # (B, S)
        attention_mask = batch['attention_mask'].to(device)
        pos_labels = batch['target'].to(device)     # (B, 4)

        optimizer.zero_grad()

        if use_listwise and config['listwise_weight'] > 0:
            # ── CE + listwise: single encoder, C decodings bundled in one pass ──
            neg_labels = neg_sampler.sample(pos_labels).to(device)

            ce_loss, aux_loss = model.compute_combined_loss(
                input_ids, attention_mask, pos_labels, neg_labels,
                temperature=config.get('listwise_temperature', 1.0))

            w = config['listwise_weight']
            loss = ce_loss + w * aux_loss

        elif use_first_diff and config['first_diff_weight'] > 0:
            # ── CE + first-difference pairwise: reuses CE logits, zero extra cost ──
            neg_labels = neg_sampler.sample(pos_labels).to(device)

            ce_loss, logits = model(input_ids, attention_mask, labels=pos_labels)
            aux_loss = model.compute_first_diff_loss(logits, pos_labels, neg_labels)

            w = config['first_diff_weight']
            loss = ce_loss + w * aux_loss

        else:
            # ── CE only ──
            ce_loss, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                               labels=pos_labels)
            aux_loss = torch.tensor(0.0)
            loss = ce_loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_aux += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

    n = len(train_loader)
    return total_loss / n, total_ce / n, total_aux / n  

def evaluate(model, eval_loader, topk_list, beam_size, device, mode):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs   = {'NDCG@'   + str(k): [] for k in topk_list}
    mrrs    = {'MRR@'    + str(k): [] for k in topk_list}
    total_ce = 0.0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating for {mode}"):
            input_ids = batch['history'].to(device) # (96, 80)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            # Validation CE
            ce_loss, _ = model(input_ids, attention_mask, labels=labels)
            total_ce += ce_loss.item()

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]  # Exclude the start token
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # Reshape to (batch_size, beam_size, seq_len) (96, 30, 4)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            for k in topk_list:
                recalls['Recall@' + str(k)].append(recall_at_k(pos_index, k).mean().item())
                ndcgs  ['NDCG@'   + str(k)].append(ndcg_at_k  (pos_index, k).mean().item())
                mrrs   ['MRR@'    + str(k)].append(mrr_at_k   (pos_index, k).mean().item())
    # Calculate average metrics
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs   = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    avg_mrrs    = {k: sum(v) / len(v) for k, v in mrrs.items()}
    avg_ce = total_ce / len(eval_loader)
    return avg_recalls, avg_ndcgs, avg_mrrs, avg_ce

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='Decay weight for the optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default='none', choices=['none', 'inv_sqrt'], help='LR scheduler: none (fixed), inv_sqrt (paper-style: constant + 1/sqrt)')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='Number of linear warmup steps (paper uses 10000)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    parser.add_argument('--dataset_path', type=str, default='data/Beauty', help='Path to the dataset')
    parser.add_argument('--code_path', type=str, default='data/Beauty/Beauty_t5_rqvae_260709-sk[0-0-0.003]-inferExtraOnly.npy', help='Path to the item-to-code mapping file') # ddata/Beauty/Beauty_t5_rqvae_260629.npy
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation')
    parser.add_argument('--log_path', type=str, default='./model/logs/tiger-0709.log', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./model/ckpt/tiger.pth', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', type=list, default=[5,10,20], help='List of top-k values for evaluation metrics')
    parser.add_argument('--beam_size', type=int, default=30, help='Beam size for generation')
    # User token (paper Section 3.2 + Table 8 ablation)
    parser.add_argument('--use_user_token', action='store_true', default=False,
                       help='Whether to prepend a hashed user ID token to the input sequence.')
    parser.add_argument('--num_user_tokens', type=int, default=2000,
                       help='Number of user token hash buckets (paper: 2000).')
    # List-wise ranking loss
    parser.add_argument('--use_listwise_loss', action='store_true', default=False,
                       help='Add list-wise softmax CE loss over positive + negative SID candidates.')
    parser.add_argument('--listwise_weight', type=float, default=0.1,
                       help='Weight for list-wise loss. Lower at init to let CE stabilise.')
    parser.add_argument('--neg_per_strategy', type=int, default=4,
                       help='Number of negative samples per strategy (4 strategies → ×4).')
    parser.add_argument('--listwise_temperature', type=float, default=1.0,
                       help='Softmax temperature for list-wise loss (<1 sharpens, >1 flattens).')
    parser.add_argument('--model_neg_refresh_interval', type=int, default=0,
                       help='Refresh model-generated negatives every K batches (0 = disabled).')
    # First-difference pairwise loss (lightweight alternative to list-wise)
    parser.add_argument('--use_first_diff_loss', action='store_true', default=False,
                       help='Use first-difference pairwise loss (reuses CE logits, no extra decoder calls).')
    parser.add_argument('--first_diff_weight', type=float, default=0.1,
                       help='Weight for first-diff pairwise loss .')

    parser.add_argument("--ckpt_dir", type=str, default="./model/ckpt/Beauty", help="please specify output directory for model")

    config = vars(parser.parse_args())
    # ── vocab_size: auto-compute when user tokens are enabled ──
    # Base vocab: 0=PAD, 1-256=L1, 257-512=L2, 513-768=L3, 769-1024=extra = 1025
    # User tokens:  1025–(1024+num_user_tokens) = up to 3024
    if config['use_user_token']:
        config['user_token_offset'] = 1025
        config['vocab_size'] = 1025 + config['num_user_tokens']
    # Set up logging
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")
    
    saved_model_dir = "{}".format(datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S"))
    ckpt_dir = os.path.join(config['ckpt_dir'], saved_model_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Initialize model
    model = TIGER(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    # Set random seed for reproducibility
    set_seed(config['seed'])
    # Check if the device is available
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path']+ '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len'],
        use_user_token=config['use_user_token'],
        num_user_tokens=config.get('num_user_tokens', 2000),
        user_token_offset=config.get('user_token_offset', 1025),
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/valid.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len'],
        use_user_token=config['use_user_token'],
        num_user_tokens=config.get('num_user_tokens', 2000),
        user_token_offset=config.get('user_token_offset', 1025),
    )
    test_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/test.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len'],
        use_user_token=config['use_user_token'],
        num_user_tokens=config.get('num_user_tokens', 2000),
        user_token_offset=config.get('user_token_offset', 1025),
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                        use_user_token=config['use_user_token'])
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False,
                                             use_user_token=config['use_user_token'])
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False,
                                        use_user_token=config['use_user_token'])
    
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(validation_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")
    # for batch in train_dataloader:
    #     print(f"Batch size: {len(batch['history'])}")
    #     print(f"the first batch history:{batch['history'][0]}")
    #     print(f"the first batch target:{batch['target'][0]}")
    #     print(f"the first batch attention mask:{batch['attention_mask'][0]}")
    #     break

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # learning rate scheduler (optional, paper-style: inv_sqrt with 10k warmup steps)
    total_steps = len(train_dataloader) * config['num_epochs']
    scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=config['warmup_steps'])
    if scheduler is not None:
        logging.info(f"LR scheduler: {config['lr_scheduler_type']} "
                     f"(warmup_steps={config['warmup_steps']}, total_steps={total_steps})")

    # ── negative sampler (shared by list-wise and first-diff losses) ──
    neg_sampler = None
    if config.get('use_listwise_loss') or config.get('use_first_diff_loss'):
        neg_sampler = NegativeSampler(
            code_path=config['code_path'],
            train_path=config['dataset_path'] + '/train.parquet',
            num_per_strategy=config['neg_per_strategy'],
        )
        if config.get('use_listwise_loss'):
            logging.info(
                f"List-wise loss enabled (weight={config['listwise_weight']}, "
                f"neg/strategy={config['neg_per_strategy']}, "
                f"model-neg refresh every {config['model_neg_refresh_interval']} batches)")
        if config.get('use_first_diff_loss'):
            logging.info(
                f"First-diff pairwise loss enabled (weight={config['first_diff_weight']}, "
                f"neg/strategy={config['neg_per_strategy']})")

    # Train the model
    model.to(device)
    best_ndcg = 0.0
    early_stop_counter = 0

    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        # Train the model
        train_loss, train_ce, train_aux = train(
            model, train_dataloader, optimizer, device,
            scheduler=scheduler, neg_sampler=neg_sampler, config=config)

        if neg_sampler is not None:
            aux_name = 'listwise' if config.get('use_listwise_loss') else 'first_diff'
            logging.info(f"Train: total={train_loss:.4f}  CE={train_ce:.4f}  {aux_name}={train_aux:.4f}")
        else:
            logging.info(f"Train: total={train_loss:.4f}")

        # Evaluate the model
        avg_recalls, avg_ndcgs, avg_mrrs, val_ce = evaluate(
            model, validation_dataloader, config['topk_list'],
            config['beam_size'], device, mode='eval')
        logging.info(f"Val: CE={val_ce:.4f}  recalls={avg_recalls}")
        logging.info(f"Val: ndcgs={avg_ndcgs}")
        logging.info(f"Val: mrrs={avg_mrrs}")

        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0  # Reset early stop counter
            test_recalls, test_ndcgs, test_mrrs, test_ce = evaluate(
                model, test_dataloader, config['topk_list'],
                config['beam_size'], device, mode='test')
            logging.info(f"Best NDCG@20: {best_ndcg}")
            logging.info(f"Test: CE={test_ce:.4f}  recalls={test_recalls}")
            logging.info(f"Test: ndcgs={test_ndcgs}")
            logging.info(f"Test: mrrs={test_mrrs}")
            # Save the best model
            save_file = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_file)
            logging.info(f"Best model saved to {save_file}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.\n\n")
                break
        
