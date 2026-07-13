import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans


class MLPLayers(nn.Module):

    def __init__(
        self, layers, dropout=0.0, activation="relu", bn=False
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers)-2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

def activation_layer(activation_name="relu", emb_dim=None):

    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(- distances / epsilon)  # 距离越近Q值越大，距离越远Q值越小

    B = Q.shape[0] # number of samples to assign 行数
    K = Q.shape[1] # how many centroids per block (usually set to 256) 列数

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        # 按行归一化，（列约束）: 每个 input 的分配权重之和 = 1/B  → 所有 input 的"总分配量" = 1
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        # 按列归一化（行约束）: 每个 entry 被分配的总权重 = 1/K  → 所有 entry "被选中的总量"均匀
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B # the colomns must sum to 1 so that Q is an assignment\
    # 迭代收敛后，Q[i][j] 表示 input i 分配给 entry j 的权重。最终用 argmax(Q) 而不是 argmin(d) 来决定分配
    # Q 是由 exp(-d/ε) 算出来的——距离近的点 Q 值大，距离远的 Q 值小;
    # 所以 Sinkhorn 是在距离近和分配均匀之间找一个平衡，ε 控制这个平衡的倾向

    return Q