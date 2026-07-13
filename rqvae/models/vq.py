import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,
                 cl_weight=0.0, cl_temperature=0.1,
                 div_weight=0.0, div_temperature=0.5,
                 div_n_clusters=16, div_cluster_interval=100):
        super().__init__()
        self.n_e = n_e # codebook size
        self.e_dim = e_dim # latent dimension
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.cl_weight = cl_weight
        self.cl_temperature = cl_temperature

        # Diversity loss (LETTER paper)
        self.div_weight = div_weight
        self.div_temperature = div_temperature
        self.div_n_clusters = div_n_clusters
        self.div_cluster_interval = div_cluster_interval
        self.register_buffer('_train_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_cluster_ids', torch.zeros(n_e, dtype=torch.long))
        self.register_buffer('_buddies', torch.arange(n_e, dtype=torch.long))

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:  # 均匀分布初始化
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else: # kmeans初始化
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):
        
        # 将 batch_size (1024) 个 e_dim (32) 维的 vector 聚成 n_e (256) 个 cluster
        centers = kmeans(
            data,
            self.n_e, 
            self.kmeans_iters,
        )
        # 每个聚类中心对应初始化 codebook 中的一个嵌入向量
        # 一个训练理想的rqvae，应当使得同前缀的SID对应的item大致上是同一个类别/种类的，用聚类寻找自然簇用于初始化就很自然了

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    @torch.no_grad()
    def _update_clusters(self):
        """K-means on codebook embeddings → cluster_ids + per-code buddy."""
        codes = self.embedding.weight.data  # (N, e_dim)
        centers = kmeans(codes, self.div_n_clusters, 50)
        dists = torch.cdist(codes, centers)
        cluster_ids = torch.argmin(dists, dim=-1)           # (N,)
        self._cluster_ids.copy_(cluster_ids)

        # Pre-compute a random same-cluster "buddy" for each code
        buddies = torch.zeros(self.n_e, dtype=torch.long)
        for c in range(self.n_e):
            same = (cluster_ids == cluster_ids[c]).nonzero(as_tuple=True)[0]
            others = same[same != c]
            if len(others) > 0:
                buddies[c] = others[torch.randint(0, len(others), (1,))]
            else:
                buddies[c] = c
        self._buddies.copy_(buddies)

    def _diversity_loss(self, indices):
        """LETTER diversity loss — pull same-cluster codes together,
        push different-cluster codes apart.

        Anchor:         assigned code embedding  e_cl
        Positive:       random code from same K-means cluster
        Negatives:      ALL other codes (N−1) — not batch-limited

        Operates purely in codebook space — no encoder involvement.
        """
        N = self.n_e
        codes = F.normalize(self.embedding.weight, dim=-1)           # (N, e_dim)
        e_cl = codes[indices]                                        # (B, e_dim)

        sim_all = torch.matmul(e_cl, codes.T) / self.div_temperature # (B, N)
        pos_idx = self._buddies[indices]                             # (B,)
        sim_pos = sim_all[torch.arange(len(indices)), pos_idx]       # (B,)

        # exclude self (assigned code) from denominator
        sim_masked = sim_all.clone()
        sim_masked[torch.arange(len(indices)), indices] = float('-inf')

        loss = (-sim_pos + torch.logsumexp(sim_masked, dim=-1)).mean()
        return loss

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)  # (batch_size, e_dim)

        ## 初始化
        if not self.initted and self.training:
            self.init_emb(latent)  # kmeans初始化codebook中的嵌入向量

        ## codebook匹配，为每个input分配entry index --> 梯度会断开
        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not use_sk or self.sk_epsilon <= 0: # 贪心，直接取 closest vector 基于L2距离
            # 贪心是逐层vq做 argmin，而 argmin 序列的累积和不等价于联合 argmin
            indices = torch.argmin(d, dim=-1) 
        else: # 但 argmin L2 问题在于 codebook 利用不均：热门entry被大量点选/冷门entry被忽略(无法更新，成为dead code) --> 使用sinkhorn algorithm 来稳定训练
            # sinkhorn有两个作用：一是平衡码本分配，二是缓解冲突
            d = self.center_distance_for_constraint(d)
            d = d.double()
            # Sinkhorn Algorithm 找到一个"最优运输"方案：让分配尽可能均匀，同时保持距离近的倾向
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)  # 按索引取codebook中的嵌入向量

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)  # 只为encoder提供梯度
        codebook_loss = F.mse_loss(x_q, x.detach())    # 只为codebook提供梯度
        # 为什么不mse(x_q, x)：两个方向都在拉近，两个移动靶相互瞄准，两边都在动，收敛慢甚至振荡
        # --> 拆开：一方固定，另一方移动
        # --> β：控制谁更积极（β = 0.25 意味着 codebook主动适应encoder的输出分布，encoder小幅调整来对齐codebook；
        # 因为codebook更重要，它一旦学好了就是整个语义 ID 体系的基础）
        loss = codebook_loss + self.beta * commitment_loss

        # ── per-layer contrastive loss (InfoNCE) ──
        cl_loss = torch.tensor(0.0, device=x.device)
        if self.cl_weight > 0:
            anchor = F.normalize(x.detach(), dim=-1)
            pos    = F.normalize(x_q, dim=-1)
            sim    = torch.matmul(anchor, pos.T) / self.cl_temperature
            labels = torch.arange(x.shape[0], device=x.device)
            cl_loss = F.cross_entropy(sim, labels)
            cl_loss = self.cl_weight * cl_loss

        # ── diversity loss (LETTER) — pure codebook-space regularisation ──
        div_loss = torch.tensor(0.0, device=x.device)
        if self.div_weight > 0 and self.training:
            self._train_step += 1
            if self._train_step % self.div_cluster_interval == 0:
                self._update_clusters()
            div_loss = self._diversity_loss(indices)
            div_loss = self.div_weight * div_loss

        x_q_raw = x_q  # raw codebook lookup — gradient to embedding.weight

        # preserve gradients （STE trick for downstream）
        x_q = x + (x_q_raw - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, cl_loss, div_loss, x_q_raw


