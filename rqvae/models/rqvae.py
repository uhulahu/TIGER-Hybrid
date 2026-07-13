import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 recon_weight=1.0,
                 quant_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 cl_weights=None,
                 cl_temperature=0.1,
                 div_weights=None,
                 div_temperature=0.5,
                 div_n_clusters=16,
                 div_cluster_interval=100,
                 collab_path=None,
                 collab_weight=0.0,
                 collab_temperature=0.5,
                 collab_debias=True,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.recon_weight = recon_weight
        self.quant_weight = quant_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cl_weights = cl_weights if cl_weights is not None else [0., 0., 0.]
        self.cl_temperature = cl_temperature
        self.div_weights = div_weights if div_weights is not None else [0., 0., 0.]
        self.collab_weight = collab_weight
        self.collab_temperature = collab_temperature
        self.collab_debias = collab_debias

        # -- collaborative embeddings (external, frozen) --
        if collab_path is not None:
            collab = torch.from_numpy(
                np.load(collab_path).astype(np.float32))
            self.register_buffer('collab_emb', collab)
        else:
            self.register_buffer('collab_emb', torch.zeros(1, e_dim))

        # encoder: (in_dim, ..., e_dim)
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

        self.rq = ResidualVectorQuantizer(
            num_emb_list, e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            cl_weights=self.cl_weights,
            cl_temperature=self.cl_temperature,
            div_weights=self.div_weights,
            div_temperature=div_temperature,
            div_n_clusters=div_n_clusters,
            div_cluster_interval=div_cluster_interval)

        # decoder: mirror of encoder
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

    def forward(self, x, use_sk=True):
        x_e = self.encoder(x)
        x_q, rq_loss, indices, x_q_list, x_q_raw_list, cl_list, div_list = \
            self.rq(x_e, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices, x_e, x_q_list, x_q_raw_list, cl_list, div_list

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, _, _, _, _ = self.rq(x_e, use_sk=use_sk)
        return indices

    @torch.no_grad()
    def get_indices_beam(self, xs, beam_size=10):
        if beam_size == 1:
            return self.get_indices(xs, use_sk=False)
        x_e = self.encoder(xs)
        return self.rq.get_indices_beam(x_e, beam_size=beam_size)

    # -- loss ----------------------------------------------------------------

    def compute_loss(self, out, quant_loss, xs, cl_list, div_list,
                     x_q_raw_list=None, item_ids=None):
        """Returns 12 scalars:
            (total, recon, quant, cl_total, cl_l1, cl_l2, cl_l3,
             div_total, div_l1, div_l2, div_l3, collab_loss)
        """
        device = out.device
        loss_total = torch.tensor(0.0, device=device)
        loss_recon = torch.tensor(0.0, device=device)
        loss_quant = torch.tensor(0.0, device=device)

        if self.recon_weight > 0:
            if self.loss_type == 'mse':
                loss_recon = F.mse_loss(out, xs, reduction='mean')
            elif self.loss_type == 'l1':
                loss_recon = F.l1_loss(out, xs, reduction='mean')
            else:
                raise ValueError('incompatible loss type')
            loss_total = loss_total + self.recon_weight * loss_recon

        if self.quant_weight > 0:
            loss_quant = quant_loss
            loss_total = loss_total + self.quant_weight * loss_quant

        cl_l1, cl_l2, cl_l3 = cl_list
        cl_total = cl_l1 + cl_l2 + cl_l3
        loss_total = loss_total + cl_total

        div_l1, div_l2, div_l3 = div_list
        div_total = div_l1 + div_l2 + div_l3
        loss_total = loss_total + div_total

        # -- collaborative regularisation --
        loss_collab = torch.tensor(0.0, device=device)
        if (self.collab_weight > 0 and item_ids is not None
                and x_q_raw_list is not None):
            z_q = x_q_raw_list[-1]                # (B, e_dim) raw, grad→codebook
            cf_emb_in_batch = self.collab_emb[item_ids]     # (B, e_dim) frozen
            loss_collab = self._CF_loss(cf_emb_in_batch, z_q)
            loss_total = loss_total + self.collab_weight * loss_collab

        return (loss_total, loss_recon, loss_quant,
                cl_total, cl_l1, cl_l2, cl_l3,
                div_total, div_l1, div_l2, div_l3,
                loss_collab)

    def _collab_loss(self, collab_emb, z_q):
        """InfoNCE aligning quantised representation with collaborative signal.

        When ``collab_debias=True`` (default), negatives that are themselves
        collab-similar to the anchor are down-weighted — preventing the loss
        from pushing apart items that users frequently co-interact with.
        """
        anchor = F.normalize(collab_emb.detach(), dim=-1)    # (B, d) frozen
        pos = F.normalize(z_q, dim=-1)                        # (B, d) grad→cb
        B = anchor.shape[0]

        sim = torch.matmul(anchor, pos.T) / self.collab_temperature  # (B, B)

        if self.collab_debias:
            with torch.no_grad():
                c_sim = torch.matmul(anchor, anchor.T)       # (B, B)
                neg_weight = (1.0 - c_sim).clamp(min=0.05)   # ↓ for friends
        else:
            neg_weight = torch.ones(B, B, device=sim.device)

        labels = torch.arange(B, device=sim.device)
        pos_score = sim[labels, labels]                       # (B,)

        mask = 1.0 - torch.eye(B, device=sim.device)
        denom = (torch.exp(sim) * neg_weight * mask).sum(dim=-1)

        loss = -torch.log(
            torch.exp(pos_score) / (torch.exp(pos_score) + denom)
        ).mean()
        return loss
    
    def _CF_loss(self, cf_emb, quantized_rep):
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        similarities = torch.matmul(quantized_rep, cf_emb.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss
