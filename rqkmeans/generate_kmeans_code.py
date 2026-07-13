# 读取向量 -> 训练 K-means -> 生成 Code -> 完整的冲突解决（后缀位修正） -> 保存文件

import numpy as np
import pandas as pd
import faiss
import os
import collections

from tqdm import tqdm

# ================= 配置区域 =================
dataset_name = "Beauty"
input_path = f"./data/{dataset_name}/item_emb.parquet"
output_path = f"./data/{dataset_name}/{dataset_name}_kmeans_code.npy"

# K-means 参数配置 (与 RQ-VAE 保持一致以进行公平对比)
num_layers = 3        # 对应 RQ-VAE 的层数
codebook_size = 256   # 对应 RQ-VAE 的 codebook大小
n_bits = 8            # faiss 参数: 2^8 = 256
# ===========================================

def generate_kmeans_codes():

    # 加载embedding
    df = pd.read_parquet(input_path)
    embeddings = np.stack(df['embedding'].values).astype('float32')  # Faiss 强制要求 float32 格式
    num_item, dim = embeddings.shape
    print(f"   Data shape: {embeddings.shape}, Dimension: {dim}")

    # 初始化 Faiss 的残差量化器
    rq = faiss.IndexResidualQuantizer(dim, num_layers, n_bits)
    
    # 开启 Verbose 可以看到训练里的迭代 loss
    rq.train(embeddings)
    
    # 生成 Code，格式是 uint8
    # codes = rq.compute_codes(embeddings)
    # 新方法：使用sa_encode
    codes = np.empty((num_item, num_layers), dtype='uint8')
    rq.sa_encode(embeddings, codes)
    codes = codes.astype(np.int32)  # 转为 int32 方便后续处理

    print(f"   First 5 raw codes:\n{codes[:5]}")

    print(" Handling Collisions (Suffix Correction)...")
    # --- 冲突处理逻辑 ---
    # 加extra token
    
    # 1. 先给所有 code 增加一列，初始化为 0，Shape 变为 (N, 4)
    codes_with_suffix = np.hstack((codes, np.zeros((codes.shape[0], 1), dtype=np.int32)))
    
    # 使用字典统计每个 Tuple 出现的次数
    tuple_counts = collections.defaultdict(int)  # Key: (c1, c2, c3), Value: current_count
    
    conflict_count = 0
    
    for i in tqdm(range(len(codes_with_suffix))):
        current_tuple = tuple(codes_with_suffix[i, :-1])
        
        # 获取当前这个 tuple 之前出现过几次
        count = tuple_counts.get(current_tuple, 0)
        
        if count > 0:
            conflict_count += 1
            
        codes_with_suffix[i, -1] = count

        tuple_counts[current_tuple] += 1

    print(f"   Total items: {len(codes)}")
    print(f"   Collided items count: {conflict_count}")
    print(f"   Collision Rate: {conflict_count / len(codes):.4f}")
    
    print(f"   Final codes shape (with suffix): {codes_with_suffix.shape}")
    print(f"   First 5 final codes:\n{codes_with_suffix[:5]}")

    print(f"5. Saving to {output_path}...")
    np.save(output_path, codes_with_suffix)
    print("Done.")

if __name__ == "__main__":
    generate_kmeans_codes()