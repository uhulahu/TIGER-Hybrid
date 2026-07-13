'''
目标：为所有item，基于rqvae将embedding转换为quantized SID；
codebook匹配：使用sinkhorn algorithm以充分利用codebook，argmax(Q);
损失：重建损失+量化损失；
RQVAE：encoder(768*512*256*128*64*32) + rq + decoder.
'''

import argparse
import os
import random
import torch
import numpy as np
from time import time
import logging

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str, default="data/Beauty/item_emb.parquet", help="Input data path.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0, 0, 0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight (0 = skip).')
    parser.add_argument('--quant_weight', type=float, default=1.0, help='Quantization loss weight (0 = skip).')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128,64], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5, help='save limit for ckpt')
    # Per-layer contrastive loss (latent-space InfoNCE at each VQ depth)
    parser.add_argument('--cl_weights', type=float, nargs=3, default=[0, 0, 0],
                       help='Per-layer InfoNCE weights [L1 L2 L3] (0 = skip).')
    parser.add_argument('--cl_temperature', type=float, default=0.5,
                       help='Temperature for contrastive loss.')
    # Diversity loss (LETTER paper)
    parser.add_argument('--div_weights', type=float, nargs=3, default=[0, 0, 0],
                       help='Per-layer diversity weights [L1 L2 L3] (0 = skip).')
    parser.add_argument('--div_temperature', type=float, default=1,
                       help='Temperature for diversity loss.')
    parser.add_argument('--div_n_clusters', type=int, default=10,
                       help='Number of K-means clusters for diversity loss.')
    parser.add_argument('--div_cluster_interval', type=int, default=1,
                       help='Re-cluster codebook every N training steps.')
    # Collaborative regularisation
    parser.add_argument('--collab_path', type=str, default='data/Beauty/item2vec_emb.npy',
                       help='Path to item2vec collaborative embeddings (.npy).')
    parser.add_argument('--collab_weight', type=float, default=0,
                       help='Weight for collaborative InfoNCE loss (0 = disabled).')
    parser.add_argument('--collab_temperature', type=float, default=1.0,
                       help='Temperature for collaborative loss.')
    parser.add_argument('--collab_debias', action='store_true', default=True,
                       help='Debias negatives by collaborative similarity (default: on).')
    parser.add_argument('--no_collab_debias', action='store_false', dest='collab_debias',
                       help='Disable debiasing (treat all in-batch items as equal negatives).')

    parser.add_argument('--resume', type=str, default=None,  # rqvae/ckpt/Beauty/Jul-09-2026_02-06-03/best_collision_model.pth
                       help='Path to a saved checkpoint (.pth) to resume training from.')
    parser.add_argument('--resume_optimizer', action='store_true', default=False,
                       help='Also restore optimizer/scheduler state from checkpoint.')

    parser.add_argument("--ckpt_dir", type=str, default="./rqvae/ckpt/Beauty", help="please specify output directory for model")

    # parser.add_argument("--log_path", type=str, default="./rqvae/logs/rqvae_training.log")

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    config = vars(args)
    
    # Set up logging
    # logging.basicConfig(
    #     filename=config['log_path'],
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )
    # logging.info(f"Configuration: {config}")

    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    use_collab = args.collab_weight > 0 and args.collab_path is not None
    data = EmbDataset(args.data_path, return_item_id=use_collab)
    model = RQVAE(in_dim=data.dim,  # embedding size 768
                  num_emb_list=args.num_emb_list,  # codebook sizes [256,256,256]
                  e_dim=args.e_dim,  # codebook中的embedding size (相当于hidden size)
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  recon_weight=args.recon_weight,
                  quant_weight=args.quant_weight,
                  beta=args.beta,  # L_rqvae中第二项的权重（β∥r_i−sg[ec_i]∥^2）
                  kmeans_init=args.kmeans_init, # 使用kmeans初始化codebook center
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  cl_weights=args.cl_weights,
                  cl_temperature=args.cl_temperature,
                  div_weights=args.div_weights,
                  div_temperature=args.div_temperature,
                  div_n_clusters=args.div_n_clusters,
                  div_cluster_interval=args.div_cluster_interval,
                  collab_path=args.collab_path,
                  collab_weight=args.collab_weight,
                  collab_temperature=args.collab_temperature,
                  collab_debias=args.collab_debias,
                  )
    print(model)

    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    start_epoch = 1
    trainer = Trainer(args, model, len(data_loader), start_epoch=start_epoch)

    # ── resume from checkpoint ──
    if args.resume is not None:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')

        # Filter out keys with shape mismatches (e.g. collab_emb from [1,d]→[N,d])
        model_dict = model.state_dict()
        filtered_ckpt = {}
        skipped = []
        for k, v in ckpt['state_dict'].items():
            if k in model_dict and v.shape != model_dict[k].shape:
                skipped.append(k)
                continue
            filtered_ckpt[k] = v

        missing, unexpected = model.load_state_dict(filtered_ckpt, strict=False)
        if skipped:
            print(f"  Skipped (shape mismatch, freshly init): {skipped}")
        if missing:
            print(f"  New layers (freshly initialised): {missing}")
        if unexpected:
            print(f"  Unused keys (from old ckpt): {unexpected}")

        # Restore best metrics
        if 'best_loss' in ckpt:
            trainer.best_loss = ckpt['best_loss']
        if 'best_collision_rate' in ckpt:
            trainer.best_collision_rate = ckpt['best_collision_rate']

        # Optionally restore optimizer + scheduler state
        if args.resume_optimizer:
            trainer.optimizer.load_state_dict(ckpt['optimizer'])
            trainer.start_epoch = ckpt.get('epoch', 0) + 1
            print(f"  Resumed optimizer + scheduler from epoch {trainer.start_epoch}")
        else:
            print(f"  Loaded weights only (optimizer fresh)")

    with open(os.path.join(trainer.ckpt_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ': ' + str(v) for k, v in sorted(config.items(), key=lambda x: x[0])]))
    f.close()

    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)



## 调试一下sk_epsilons

# ########## sk_epsilons = [0, 0, 0.003] ##########

# ─ 1. Codebook utilization ─
#   L1: 60/256 entries used (23.4%)
#   L2: 256/256 entries used (100.0%)
#   L3: 256/256 entries used (100.0%)

# ─ 2. Prefix Purity ─
#   L   Groups   Purity     Cat.NMI    Cat.ARI    BrandPurity 
#   --- -------- ---------- ---------- ---------- ------------
#   L1   60       0.7883     0.3166     0.0880     0.1247      
#   L2   5577     0.8795     0.2698     0.0015     0.6768      
#   L3   12075    0.9995     0.2866     0.0000     0.9992      

# ─ L1 assignment by coarse category (top-10 L1 codes) ─
#   Code 139 ( 830 items): majority=Makeup (88%), dist={Makeup:731, Skin Care:52, Tools & Accessories:23, Hair Care:17, Bath & Body:6}
#   Code 128 ( 576 items): majority=Fragrance (74%), dist={Fragrance:426, Skin Care:114, Makeup:19, Hair Care:7, Bath & Body:6}
#   Code   4 ( 526 items): majority=Makeup (61%), dist={Makeup:323, Tools & Accessories:130, Skin Care:29, Hair Care:27, Bath & Body:13}
#   Code 165 ( 480 items): majority=Tools & Accessories (65%), dist={Tools & Accessories:310, Makeup:131, Skin Care:24, Hair Care:12, Fragrance:2}
#   Code  89 ( 450 items): majority=Hair Care (86%), dist={Hair Care:386, Skin Care:29, Bath & Body:20, Makeup:13, Fragrance:1}
#   Code 194 ( 383 items): majority=Makeup (82%), dist={Makeup:315, Skin Care:28, Hair Care:23, Tools & Accessories:10, Bath & Body:4}
#   Code   3 ( 360 items): majority=Hair Care (85%), dist={Hair Care:305, Skin Care:37, Bath & Body:10, Makeup:4, Tools & Accessories:3}
#   Code 123 ( 347 items): majority=Makeup (86%), dist={Makeup:299, Skin Care:18, Hair Care:12, Tools & Accessories:7, Fragrance:6}
#   Code 160 ( 320 items): majority=Makeup (86%), dist={Makeup:275, Tools & Accessories:16, Skin Care:14, Hair Care:10, Bath & Body:4}
#   Code 248 ( 312 items): majority=Tools & Accessories (48%), dist={Tools & Accessories:150, Skin Care:83, Makeup:46, Hair Care:22, Bath & Body:8}

#   Total items with category: 12101 / 12101

# ########## sk_epsilons = [0.003, 0, 0.003] ##########

# Best Loss 0.0016568665305385366 
# Best Collision Rate 0.012808858772002314 

# ─ 1. Codebook utilization ─
#   L1: 256/256 entries used (100.0%)
#   L2: 146/256 entries used (57.0%)
#   L3: 256/256 entries used (100.0%)

# ─ 2. Prefix Purity ─
#   L   Groups   Purity     Cat.NMI    Cat.ARI    BrandPurity 
#   --- -------- ---------- ---------- ---------- ------------
#   L1   256      0.7914     0.2615     0.0153     0.1823      
#   L2   8560     0.9277     0.2766     0.0003     0.7783      
#   L3   12100    1.0000     0.2867     0.0000     1.0000      

# ─ L1 assignment by coarse category (top-10 L1 codes) ─
#   Code 132 (  93 items): majority=Fragrance (85%), dist={Fragrance:79, Skin Care:11, Makeup:1, Bath & Body:1, Hair Care:1}
#   Code 167 (  89 items): majority=Makeup (81%), dist={Makeup:72, Hair Care:7, Fragrance:3, Tools & Accessories:3, Skin Care:3}
#   Code 115 (  81 items): majority=Tools & Accessories (79%), dist={Tools & Accessories:64, Makeup:12, Skin Care:3, Hair Care:2}
#   Code 226 (  79 items): majority=Hair Care (86%), dist={Hair Care:68, Skin Care:8, Makeup:3}
#   Code  47 (  78 items): majority=Skin Care (85%), dist={Skin Care:66, Hair Care:6, Bath & Body:3, Makeup:2, Fragrance:1}
#   Code 101 (  78 items): majority=Makeup (90%), dist={Makeup:70, Hair Care:4, Tools & Accessories:3, Skin Care:1}
#   Code 103 (  75 items): majority=Skin Care (87%), dist={Skin Care:65, Hair Care:4, Makeup:3, Bath & Body:2, Fragrance:1}
#   Code  13 (  74 items): majority=Fragrance (77%), dist={Fragrance:57, Skin Care:13, Makeup:2, Hair Care:1, Bath & Body:1}
#   Code 135 (  74 items): majority=Hair Care (88%), dist={Hair Care:65, Tools & Accessories:4, Makeup:3, Skin Care:1, Bath & Body:1}
#   Code 217 (  73 items): majority=Makeup (89%), dist={Makeup:65, Skin Care:3, Hair Care:2, Tools & Accessories:2, Fragrance:1}

#   Total items with category: 12101 / 12101

# ########## sk_epsilons = [0.003, 0.003, 0.003] ##########
# ─ 1. Codebook utilization ─
#   L1: 256/256 entries used (100.0%)
#   L2: 256/256 entries used (100.0%)
#   L3: 256/256 entries used (100.0%)

# ─ 2. Prefix Purity ─
#   L   Groups   Purity     Cat.NMI    Cat.ARI    BrandPurity 
#   --- -------- ---------- ---------- ---------- ------------
#   L1   256      0.7808     0.2567     0.0162     0.1820      
#   L2   9439     0.9423     0.2786     0.0002     0.8347      
#   L3   12101    1.0000     0.2867     0.0000     1.0000      

# ─ L1 assignment by coarse category (top-10 L1 codes) ─
#   Code  98 ( 122 items): majority=Hair Care (82%), dist={Hair Care:100, Skin Care:13, Bath & Body:3, Tools & Accessories:3, Makeup:2}
#   Code 116 ( 108 items): majority=Fragrance (75%), dist={Fragrance:81, Skin Care:24, Bath & Body:1, Makeup:1, Tools & Accessories:1}
#   Code 206 ( 105 items): majority=Hair Care (90%), dist={Hair Care:95, Skin Care:4, Makeup:3, Bath & Body:3}
#   Code 217 ( 104 items): majority=Skin Care (76%), dist={Skin Care:79, Hair Care:12, Bath & Body:6, Fragrance:4, Makeup:3}
#   Code 213 ( 102 items): majority=Hair Care (87%), dist={Hair Care:89, Skin Care:5, Tools & Accessories:5, Makeup:2, Bath & Body:1}
#   Code 112 ( 101 items): majority=Hair Care (80%), dist={Hair Care:81, Skin Care:8, Makeup:7, Bath & Body:3, Tools & Accessories:2}
#   Code   6 (  98 items): majority=Hair Care (87%), dist={Hair Care:85, Skin Care:8, Makeup:3, Fragrance:1, Bath & Body:1}
#   Code 248 (  91 items): majority=Skin Care (89%), dist={Skin Care:81, Hair Care:3, Tools & Accessories:3, Makeup:2, Bath & Body:2}
#   Code 169 (  89 items): majority=Makeup (87%), dist={Makeup:77, Skin Care:7, Hair Care:2, Tools & Accessories:2, Fragrance:1}
#   Code 158 (  88 items): majority=Skin Care (84%), dist={Skin Care:74, Hair Care:7, Makeup:4, Bath & Body:3}

#   Total items with category: 12101 / 12101


## 使用 Qwen/Qwen3-Embedding-0.6B 生成嵌入(1024维)

# ########## sk_epsilons = [0, 0, 0.003], quant_loss_weight = 1.0 ##########

# reconstruction逐渐下降，train loss逐渐上升，碰撞率很高
# Best Loss 0.002902418593293987
# Best Collision Rate 0.06511858524088918

# ########## sk_epsilons = [0, 0, 0.003], quant_loss_weight = 2.0 ##########
# 也是reconstruction逐渐下降，train loss逐渐上升
# ─ 1. Codebook utilization ─
#   L1: 81/256 entries used (31.6%)
#   L2: 248/256 entries used (96.9%)
#   L3: 256/256 entries used (100.0%)

# ─ 2. Prefix Purity ─
#   L   Groups   Purity     Cat.NMI    Cat.ARI    BrandPurity 
#   --- -------- ---------- ---------- ---------- ------------
#   L1   81       0.8357     0.3377     0.0732     0.1198      
#   L2   5826     0.9038     0.2769     0.0013     0.7044      
#   L3   12083    0.9998     0.2867     0.0000     0.9993      

# ─ L1 assignment by coarse category (top-10 L1 codes) ─
#   Code  96 ( 666 items): majority=Makeup (93%), dist={Makeup:622, Skin Care:16, Tools & Accessories:13, Hair Care:12, Bath & Body:3}
#   Code  75 ( 363 items): majority=Fragrance (85%), dist={Fragrance:307, Skin Care:41, Makeup:7, Bath & Body:4, Hair Care:2}
#   Code 183 ( 351 items): majority=Makeup (84%), dist={Makeup:296, Skin Care:24, Hair Care:23, Tools & Accessories:4, Bath & Body:2}
#   Code   3 ( 321 items): majority=Skin Care (83%), dist={Skin Care:268, Hair Care:34, Bath & Body:10, Makeup:5, Tools & Accessories:4}
#   Code   4 ( 317 items): majority=Hair Care (94%), dist={Hair Care:297, Skin Care:8, Makeup:7, Tools & Accessories:4, Bath & Body:1}
#   Code  59 ( 308 items): majority=Makeup (91%), dist={Makeup:280, Hair Care:10, Skin Care:9, Tools & Accessories:4, Bath & Body:3}
#   Code  82 ( 306 items): majority=Bath & Body (70%), dist={Bath & Body:214, Skin Care:62, Hair Care:22, Makeup:5, Fragrance:2}
#   Code  21 ( 280 items): majority=Tools & Accessories (59%), dist={Tools & Accessories:166, Makeup:90, Skin Care:16, Hair Care:6, Fragrance:2}
#   Code 156 ( 258 items): majority=Makeup (90%), dist={Makeup:231, Skin Care:11, Hair Care:6, Tools & Accessories:6, Bath & Body:3}
#   Code 132 ( 249 items): majority=Tools & Accessories (80%), dist={Tools & Accessories:198, Skin Care:24, Makeup:14, Hair Care:8, Bath & Body:3}

#   Total items with category: 12101 / 12101

# ########## sk_epsilons = [0, 0.003, 0.003], quant_loss_weight = 2.0 ##########
# ─ 1. Codebook utilization ─
#   L1: 249/256 entries used (97.3%)
#   L2: 256/256 entries used (100.0%)
#   L3: 256/256 entries used (100.0%)

# ─ 2. Prefix Purity ─
#   L   Groups   Purity     Cat.NMI    Cat.ARI    BrandPurity 
#   --- -------- ---------- ---------- ---------- ------------
#   L1   249      0.8407     0.2989     0.0286     0.2290      
#   L2   8224     0.9389     0.2812     0.0006     0.8481      
#   L3   12089    1.0000     0.2867     0.0000     0.9998      

# ─ L1 assignment by coarse category (top-10 L1 codes) ─
#   Code 232 ( 269 items): majority=Makeup (92%), dist={Makeup:248, Skin Care:7, Hair Care:6, Bath & Body:3, Tools & Accessories:3}
#   Code 221 ( 235 items): majority=Makeup (86%), dist={Makeup:203, Skin Care:12, Hair Care:9, Tools & Accessories:5, Bath & Body:4}
#   Code 120 ( 235 items): majority=Makeup (96%), dist={Makeup:226, Skin Care:7, Tools & Accessories:1, Bath & Body:1}
#   Code  64 ( 226 items): majority=Fragrance (87%), dist={Fragrance:196, Skin Care:24, Makeup:4, Bath & Body:1, Tools & Accessories:1}
#   Code  85 ( 178 items): majority=Tools & Accessories (51%), dist={Tools & Accessories:90, Makeup:70, Skin Care:13, Hair Care:3, Fragrance:1}
#   Code 122 ( 175 items): majority=Makeup (90%), dist={Makeup:158, Tools & Accessories:5, Skin Care:5, Hair Care:5, Bath & Body:2}
#   Code 193 ( 161 items): majority=Makeup (91%), dist={Makeup:147, Skin Care:5, Tools & Accessories:5, Hair Care:4}
#   Code  92 ( 159 items): majority=Makeup (86%), dist={Makeup:137, Hair Care:11, Skin Care:10, Bath & Body:1}
#   Code  83 ( 158 items): majority=Makeup (82%), dist={Makeup:130, Hair Care:11, Skin Care:10, Tools & Accessories:4, Bath & Body:2}
#   Code  62 ( 154 items): majority=Makeup (92%), dist={Makeup:141, Skin Care:6, Hair Care:3, Bath & Body:2, Tools & Accessories:1}

#   Total items with category: 12101 / 12101

# ########## sk_epsilons = [0, 0, 0.003], quant_loss_weight = 2.0, layers = [768,512,256,128,64] ##########
# ─ 1. Codebook utilization ─
#   L1: 130/256 entries used (50.8%)
#   L2: 252/256 entries used (98.4%)
#   L3: 256/256 entries used (100.0%)

# ─ 2. Prefix Purity ─
#   L   Groups   Purity     Cat.NMI    Cat.ARI    BrandPurity 
#   --- -------- ---------- ---------- ---------- ------------
#   L1   130      0.8379     0.3174     0.0474     0.1608      
#   L2   7646     0.9250     0.2790     0.0007     0.8249      
#   L3   12088    0.9998     0.2867     0.0000     0.9997      

# ─ L1 assignment by coarse category (top-10 L1 codes) ─
#   Code  82 ( 532 items): majority=Makeup (91%), dist={Makeup:485, Skin Care:20, Tools & Accessories:13, Hair Care:12, Bath & Body:2}
#   Code  80 ( 286 items): majority=Makeup (92%), dist={Makeup:264, Skin Care:8, Hair Care:6, Tools & Accessories:4, Fragrance:2}
#   Code  49 ( 238 items): majority=Hair Care (86%), dist={Hair Care:204, Skin Care:23, Makeup:5, Bath & Body:4, Tools & Accessories:2}
#   Code  43 ( 234 items): majority=Makeup (86%), dist={Makeup:202, Skin Care:12, Hair Care:9, Tools & Accessories:5, Bath & Body:4}
#   Code  39 ( 219 items): majority=Fragrance (86%), dist={Fragrance:188, Skin Care:25, Makeup:3, Hair Care:1, Bath & Body:1}
#   Code   1 ( 197 items): majority=Makeup (91%), dist={Makeup:179, Skin Care:8, Hair Care:5, Tools & Accessories:2, Bath & Body:2}
#   Code  88 ( 194 items): majority=Hair Care (95%), dist={Hair Care:184, Skin Care:6, Tools & Accessories:2, Makeup:2}
#   Code  56 ( 190 items): majority=Tools & Accessories (87%), dist={Tools & Accessories:166, Makeup:11, Hair Care:8, Skin Care:5}
#   Code  57 ( 177 items): majority=Makeup (82%), dist={Makeup:146, Hair Care:14, Skin Care:10, Tools & Accessories:4, Bath & Body:2}
#   Code 133 ( 175 items): majority=Hair Care (79%), dist={Hair Care:138, Skin Care:19, Makeup:9, Tools & Accessories:8, Bath & Body:1}

#   Total items with category: 12101 / 12101

# ########## sk_epsilons = [0, 0, 0.003], quant_loss_weight = 2.0, layers = [768,512,256,128,64], 束搜索bw10 ##########
# 加了束搜索后 SID 改变的 item: 4494 / 12101 (37.1%)，变化集中在L2、L3：
#   L1  changed:    161 items (3.6%)     ← 几乎不动
#   L2  changed:  3,885 items (86.5%)    ← 主战场
#   L3  changed:  4,420 items (98.4%)    ← 主战场
#   Extra changed:   11 items (0.2%)     ← 几乎不动

# ─ 1. Codebook utilization ─
#   L1: 130/256 entries used (50.8%)
#   L2: 252/256 entries used (98.4%)
#   L3: 256/256 entries used (100.0%)

# ─ 2. Prefix Purity ─
#   L   Groups   Purity     Cat.NMI    Cat.ARI    BrandPurity 
#   --- -------- ---------- ---------- ---------- ------------
#   L1   130      0.8379     0.3178     0.0473     0.1605      
#   L2   8041     0.9291     0.2790     0.0005     0.8324      
#   L3   12095    0.9998     0.2867     0.0000     0.9999      

# ─ L1 assignment by coarse category (top-10 L1 codes) ─
#   Code  82 ( 531 items): majority=Makeup (91%), dist={Makeup:484, Skin Care:20, Tools & Accessories:13, Hair Care:12, Bath & Body:2}
#   Code  80 ( 285 items): majority=Makeup (92%), dist={Makeup:263, Skin Care:8, Hair Care:6, Tools & Accessories:4, Fragrance:2}
#   Code  49 ( 238 items): majority=Hair Care (86%), dist={Hair Care:204, Skin Care:22, Makeup:5, Bath & Body:5, Tools & Accessories:2}
#   Code  43 ( 233 items): majority=Makeup (86%), dist={Makeup:201, Skin Care:12, Hair Care:9, Tools & Accessories:5, Bath & Body:4}
#   Code  39 ( 217 items): majority=Fragrance (85%), dist={Fragrance:185, Skin Care:26, Makeup:3, Hair Care:1, Bath & Body:1}
#   Code   1 ( 197 items): majority=Makeup (91%), dist={Makeup:180, Skin Care:8, Hair Care:5, Tools & Accessories:2, Bath & Body:2}
#   Code  88 ( 194 items): majority=Hair Care (95%), dist={Hair Care:184, Skin Care:6, Tools & Accessories:2, Makeup:2}
#   Code  56 ( 191 items): majority=Tools & Accessories (87%), dist={Tools & Accessories:167, Makeup:11, Hair Care:7, Skin Care:6}
#   Code  57 ( 176 items): majority=Makeup (82%), dist={Makeup:145, Hair Care:14, Skin Care:10, Tools & Accessories:4, Bath & Body:2}
#   Code 133 ( 174 items): majority=Hair Care (79%), dist={Hair Care:137, Skin Care:19, Makeup:9, Tools & Accessories:8, Bath & Body:1}

#   Total items with category: 12101 / 12101


# === cl loss 把信息从 L2/L3 挤到了 L1，导致 SID 层级结构变「浅」
#   ┌──────────────────┬─────────────────┬─────────────────┬─────────────┐
#   │       指标       │      No CL      │     With CL     │    方向     │
#   ├──────────────────┼─────────────────┼─────────────────┼─────────────┤
#   │ L1 利用率        │ 130/256 (50.8%) │ 254/256 (99.2%) │ ↑           │
#   ├──────────────────┼─────────────────┼─────────────────┼─────────────┤
#   │ L1 bucket median │ 82 items        │ 40 items        │ ↓ 一半      │
#   ├──────────────────┼─────────────────┼─────────────────┼─────────────┤
#   │ H(L1)            │ 6.78 bits       │ 7.74 bits       │ ↑ 1 bit     │
#   ├──────────────────┼─────────────────┼─────────────────┼─────────────┤
#   │ H(L2|L1)         │ 5.83 bits       │ 5.06 bits       │ ↓ 0.77 bits │
#   ├──────────────────┼─────────────────┼─────────────────┼─────────────┤
#   │ H(L3|L1,L2)      │ 0.95 bits       │ 0.76 bits       │ ↓           │
#   ├──────────────────┼─────────────────┼─────────────────┼─────────────┤
#   │ Joint H(L1..L3)  │ 13.56 bits      │ 13.56 bits      │ ≈           │
#   └──────────────────┴─────────────────┴─────────────────┴─────────────┘