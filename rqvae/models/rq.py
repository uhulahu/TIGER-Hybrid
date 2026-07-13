import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,
                 cl_weights=None, cl_temperature=0.1,
                 div_weights=None, div_temperature=0.5,
                 div_n_clusters=16, div_cluster_interval=100):
        super().__init__()
        self.n_e_list = n_e_list  # codebook sizes [256,256,256]
        self.e_dim = e_dim  # codebook中的embedding size (相当于hidden size)
        self.num_quantizers = len(n_e_list)  # codebook数量
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cl_weights = cl_weights if cl_weights is not None else [0.] * self.num_quantizers
        self.cl_temperature = cl_temperature
        div_w = div_weights if div_weights is not None else [0.] * self.num_quantizers
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters,
                                                        cl_weight=cl_w,
                                                        cl_temperature=cl_temperature,
                                                        div_weight=dw,
                                                        div_temperature=div_temperature,
                                                        div_n_clusters=div_n_clusters,
                                                        div_cluster_interval=div_cluster_interval)
                                        for n_e, sk_epsilon, cl_w, dw in
                                        zip(n_e_list, sk_epsilons, self.cl_weights, div_w) ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        all_cl = []           # per-layer CL losses
        all_div = []          # per-layer diversity losses
        x_q_list = []         # STE cumsum [c₁, c₁+c₂, c₁+c₂+c₃] → decoder
        x_q_raw_list = []     # raw cumsum → collab loss (gradient to codebook)

        x_q = 0
        x_q_raw = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, cl_l, div_l, x_res_raw = \
                quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            x_q_raw = x_q_raw + x_res_raw          # raw — gradient to embedding.weight
            x_q_list.append(x_q)
            x_q_raw_list.append(x_q_raw)

            all_losses.append(loss)
            all_indices.append(indices)
            all_cl.append(cl_l)
            all_div.append(div_l)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices, x_q_list, x_q_raw_list, all_cl, all_div

    @torch.no_grad()
    def get_indices_beam(self, x, beam_size=10):
        """通过码本层次结构的束搜索实现联合最优码字分配。
        
        贪心法逐层独立选取最近码字，一旦靠前层选到次优解就无法纠正，
        误差会向深层累积。束搜索在整个层次上同时维护 ``beam_size`` 条候选
        路径，按累积重建误差 ``‖x - Σc_i‖²``（等价于每步残差与码字的 L²
        距离）排序，在前层局部最优导致全局残差次优时能有效规避。

        Args:
            x:          (B, e_dim) 编码器输出的隐向量。
            beam_size:  束宽，每层搜索保留的数量。若为 int 则每层使用相同束宽；
                        若为 list[int] 则按层指定，长度需等于码本数。
                        每层实际束宽 = min(beam_size, 该层码本大小)。
                        默认 10。

        Returns:
            indices:    (B, num_quantizers) 各输入对应的码本索引。
        """
        # ── 解析 beam_size ──
        if isinstance(beam_size, int):
            beam_per_layer = [beam_size] * self.num_quantizers
        elif isinstance(beam_size, (list, tuple)):
            if len(beam_size) != self.num_quantizers:
                raise ValueError(
                    f"beam_size 列表长度 ({len(beam_size)}) 与码本数 "
                    f"({self.num_quantizers}) 不一致"
                )
            beam_per_layer = list(beam_size)
        else:
            raise TypeError("beam_size 需为 int 或 list[int]")

        B, e_dim = x.shape
        codebooks = self.get_codebook()  # (num_quantizers, K, e_dim)

        # ── 初始化候选 ──
        cb0 = codebooks[0]  # 第一级codebook (K0, e_dim)
        bw0 = min(beam_per_layer[0], cb0.shape[0])
        d0 = torch.cdist(x, cb0, p=2)  # (B, K0) 欧氏距离
        _, idx0 = torch.topk(d0, k=bw0, dim=1, largest=False)  # 取最近的bw0个（已排序），(B, beam0)

        paths = idx0.unsqueeze(-1)                     # (B, beam0, 1) 累积的码本索引路径，最终的数量等于最后一层的束宽
        residuals = x.unsqueeze(1) - cb0[idx0]         # (B, beam0, e_dim)

        # ── 扩展 + 剪枝 ──
        cur_beam = bw0 # 当前束宽
        for layer in range(1, self.num_quantizers):
            cb_l = codebooks[layer]                           # 当前层 codebook (K, e_dim)
            K = cb_l.shape[0]

            # 计算前一层cur_beam个残差与当前码本中所有向量的欧氏距离矩阵：‖残差 − c‖
            # 所有可能的 "父路径 + 当前层码字" 组合，cur_beam × K 个
            d = torch.cdist(
                residuals.view(B * cur_beam, e_dim), cb_l, p=2
            ).view(B, cur_beam, K)  # (B, cur_beam, K)
            # 不需要显式累加距离，因为残差本身就是前序所有层选择的累积结果

            # ⭐beam search：
            # topk 在这一层全部候选里挑了 bw 个最优的留下，
            # 剩下的 cur_beam × K - bw 条路径直接被丢弃，不会传递到后续层
            d_flat = d.view(B, cur_beam * K)
            bw = min(beam_per_layer[layer], d_flat.shape[1])
            _, top_flat = torch.topk(d_flat, k=bw, dim=1, largest=False)  # torch.topk()返回结果已排序

            # 展平索引 (row × K + col) 解码 → (父候选, 新码字)
            parent = top_flat // K      # (B, new_beam) 来自哪个父候选（beam中的位置索引）
            new_code = top_flat % K     # (B, new_beam) 该层码本中的码字索引

            # 继承父候选的路径并拼接当前层码字
            paths = paths.gather(  # 取出父候选累积token
                1, parent.unsqueeze(-1).expand(-1, -1, paths.shape[-1])
            )                            # (B, new_beam, layer_count)
            paths = torch.cat([paths, new_code.unsqueeze(-1)], dim=-1)

            # 继承父候选的残差并更新
            residuals = residuals.gather(
                1, parent.unsqueeze(-1).expand(-1, -1, e_dim)
            )                            # (B, new_beam, e_dim)
            residuals = residuals - cb_l[new_code]

            cur_beam = bw

        # ── 每个输入取最优路径 ──
        return paths[:, 0, :]  # (B, num_quantizers) 