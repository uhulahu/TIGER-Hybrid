import collections
import json
import logging

import numpy as np
import torch
import argparse
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


parser = argparse.ArgumentParser(description="Generate SID codes from RQ-VAE checkpoint")
parser.add_argument('--ckpt_path', type=str, default='rqvae/ckpt/Beauty/Jul-08-2026_22-14-07/epoch_349_collision_0.3868_model.pth',
# parser.add_argument('--ckpt_path', type=str, default='rqvae/ckpt/Beauty/Jul-10-2026_22-18-01/last_epoch_model.pth',
                   help='Path to the RQ-VAE checkpoint (.pth).')
parser.add_argument('--dataset', type=str, default='Beauty',
                   help='Dataset name (used for data path prefix).')
parser.add_argument('--output_file', type=str, default='data/Beauty/Beauty_t5_rqvae_260710_tmptmptmptmptmp.npy',
                   help='Output .npy path (default: auto-named from ckpt + dataset).')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--use_sk_resolve', type=bool, default=False,
                   help='Use Sinkhorn to re-assign L3 codes for collision groups (Step 2).')
args_cli = parser.parse_args()

dataset = args_cli.dataset
ckpt_path = args_cli.ckpt_path
output_file = args_cli.output_file
if output_file is None:
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    output_file = f"data/{dataset}/{dataset}_{ckpt_name}.npy"

device = torch.device(args_cli.device)

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(args.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  recon_weight=args.recon_weight,
                  quant_weight=args.quant_weight,
                  cl_weights=args.cl_weights,
                  cl_temperature=args.cl_temperature,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

# Filter out keys whose shape doesn't match (e.g. collab_emb from training)
model_dict = model.state_dict()
filtered = {k: v for k, v in state_dict.items()
            if k not in model_dict or v.shape == model_dict[k].shape}
skipped = [k for k in state_dict
           if k in model_dict and state_dict[k].shape != model_dict[k].shape]
if skipped:
    print(f"  Skipped (inference-only, not needed): {skipped}")
model.load_state_dict(filtered, strict=False)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

## Step1 全量编码
for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d,use_sk=False)
    # indices = model.get_indices_beam(d, beam_size=1)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

## Step2 碰撞处理，Collision Resolution

# 需要注意的是：训练时的sinkhorn是在整个码本上做global code balancing；
# 推理时的sinkhorn是用来在每个局部碰撞组内做唯一化，只是把原本应该由 extra token 位承担的身份区分信息搬到了L3.
# 🎃--> 这反而可能导致L3变得不纯净，并打乱/弱化L3与L1L2的层次依赖；而extra-only情况下，L1-L3在语义上是纯净的，额外的身份区分完全交给extra token。

if args_cli.use_sk_resolve:
    # Sinkhorn 重分配: L1/L2 保持不动, 仅 L3 用 Sinkhorn 重新分配碰撞组
    print("Collision resolution: Sinkhorn re-assignment on L3")
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(f"  Iter {tt}: {len(collision_item_groups)} collision groups")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)  # 取出碰撞组
            indices = model.get_indices(d, use_sk=True) # 用sinkhorn对碰撞组重新分配
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1
else:
    print("Collision resolution: extra token only (no Sinkhorn re-assignment)")


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)


all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

## Step3 添加 extra token

# initialize a list to store the converted codes
codes = []

# iterate through the dictionary and convert each list of indices to a code
for key, value in all_indices_dict.items():
    code = [int(item.split('_')[1].strip('>')) for item in value]
    codes.append(code)

# convert the list of codes to a numpy array
codes_array = np.array(codes)

# Add an extra dimension to all codes
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

# Resolve duplicates by incrementing the last dimension
unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]

if len(duplicates) > 0:
    print("Resolving duplicates in codes...")
    for duplicate in duplicates:
        duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
        for i, idx in enumerate(duplicate_indices):
            codes_array[idx, -1] = i  # Increment the last digit for resolving duplicates



## Step 4 桶内重排 (替代哈希式 extra token)

# 🎃extra token相当于纯随机/哈希的分配，没有语义，没有可进一步区分的规律。-->这对于预测模型来说是是难以学习的。
# 🤞有一种做法是允许SID一对多，把压力转移给召回后的桶内排序/后续的排序阶段。
# SID一对多+TIGER推理阶段桶内重排：用原始 content embedding 上与 last item 的相似度做桶内排序！充分利用已有内容信息：
# - RQ tokenizer 负责粗粒度压缩和生成式检索.
# - 原始 embedding 负责恢复量化过程中丢失的桶内细粒度差异：slocal​(u,i)=cos(e_last​,e_i​)；
#   或者使用最近若干 items 的位置加权兴趣向量。
# 流程：
# 用户历史
#    ↓
# TIGER 生成前三层 SID bucket
#    ↓
# 展开 bucket 内全部 item
#    ↓
# 使用原始 768 维 content embedding 做细粒度桶内重排

# 🤞但是我想先试试改造extra token：
# 仍然保留唯一的四层 SID，但让extra token不再是随意编号，而成为具有跨桶一致含义的“结构化消歧 token
# 具体来说，增4-token SID，但是学习4个码本，这4个码本不是端到端学习:
# - 前三层码本由原来的三层 RQ-KMeans/RQ-VAE 产生；
# - 然后冻结前三层码本，增加第4层码本
# - 在4层训练过程中，同一 L1L2L3 碰撞桶内，利用 Hungarian 强制单射分配，不同碰撞桶可以复用同一个值
# - 使用masked量化损失训练：
#     前3层量化后的剩余残差：ri(3)​=zi​−(qi1​+qi2​+qi3​)
#     收集前三层 SID 发生碰撞的 item：Gb​={i:(ci1​,ci2​,ci3​)=b},∣Gb​∣>1
#     对每个碰撞桶做匈牙利算法分配，并计算masked量化损失：min_{a_b}​∑_{i∈Gb}​​|| ​ri(3)​−e_{a_b​(i)}^(4)||^​2 
#     满足：i != j ⇒ a_b​(i) != a_b​(j).
#     然后根据全体碰撞桶的 assignment 更新 L4 centroid
#     反复执行，本质上就是一种带有“桶内单射约束”的 constrained KMeans。

## 
# 但是上面只是推测，需要统计验证，比如计算前三层正确率与完整 SID 正确率：
# - Prefix-3 hit：前三层 SID 是否生成正确
# - Full hit：前三层 + extra token 是否全部正确
# 如果出现：前三层经常正确、extra token 经常错误，就能直接证明哈希式extra token是瓶颈。

# 然后可以立即做一个不需要重训的实验，我们已有训练好的四 token TIGER，可以先不重新训练模型：
# - beam search 只生成到 L3；
# - 得到前三层 SID 的 top-B；
# - 将每个 SID 映射到全部 item；
# - 用 last-item content cosine 在桶内排序；
# - 保持最终 item 候选数严格为 K；
# - 与 extra-token 完整生成比较。
# 这能快速回答：忽略随机 extra token，让 TIGER 只预测语义桶，再用连续 embedding 消歧，是否更好？
# 不过训练时仍然是基于四token SID训练的，如果这个无重训版本已经更好，可以训练一个纯 3-token 的 bucket-level TIGER进一步验证。

# 但是我们统计分析发现：大部分测试样本连正确前三层路径都没生成出来 
# → extra token 根本还没有机会成为瓶颈，前三种中的某一层才是真正瓶颈。
# 因此当前 TIGER 的问题仍然主要在：
# - 是否选对 L1 粗簇；
# - 是否沿正确 L1 路由到 L2；
# - 是否在正确 L1L2 下找到 L3；
# - 正确 prefix 能否在 beam 中存活。
# 而不是最后一步的身份消歧。




## Final

new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = new_unique_codes[new_counts > 1]

if len(duplicates) > 0:
    print("There still have duplicates:", duplicates)
else:
    print("There are no duplicates in the codes after resolution.")

# save the codes to a numpy file
print(f"Saving codes to {output_file}")
print(f"the first 5 codes: {codes_array[:5]}")
np.save(output_file, codes_array)

# -- run SID evaluation --
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_sid import evaluate_sid
meta_path = f"data/{dataset}/{dataset}_metadata.json"
mapping_path = f"data/{dataset}/item_mapping.npy"
report = evaluate_sid(output_file, meta_path, mapping_path)
eval_file = os.path.join(os.path.dirname(ckpt_path), "sid_evaluation.txt")
with open(eval_file, 'w') as f:
    f.write(report)
print(f"Evaluation saved to {eval_file}")

