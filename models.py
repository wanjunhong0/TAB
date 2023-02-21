import torch
import torch.nn.functional as F
from layers import Propagation, MLP, ClusteringLayer
from utils import normalize_adj, sparse_diag


class GraphCAD(torch.nn.Module):
    def __init__(self, args, n_sample, n_feature, n_class, feature_corr):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(GraphCAD, self).__init__()

        self.feature_corr = feature_corr
        self.n_feature = n_feature
        self.k = args.k
        self.n_pool = args.n_pool
        node_pool = round(pow(n_sample, 1 / (self.n_pool)))   # + 1 avoid calculating last layer
        self.dropout = args.dropout
        self.prop = Propagation(alpha=0.)

        self.bn0 = torch.nn.BatchNorm1d(n_feature)
        self.pools = torch.nn.ModuleList()
        n_centroid = n_sample
        for _ in range(args.n_pool):
            n_centroid = round(n_centroid / node_pool)
            self.pools.append(ClusteringLayer(n_feature, args.n_hidden, n_centroid))

        self.mlp = MLP(3, n_feature, args.n_hidden, n_class, dropout=[0., 0., 0.], activation=torch.nn.PReLU())



    def forward(self, x, x_cov, adj, norm_adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label

        """
        x = x0 = self.bn0(x)

        # Pooling
        masks = []
        corrs = []
        mask = adj
        for i in range(self.n_pool):
            x_cov, corr, mask = self.pools[i](x_cov, mask)
            corrs.append(corr)
            masks.append(mask)

        # masks.append((1, 1).to_sparse())
        corrs.append(self.feature_corr)
        
        # Allocation
        gains = []
        for i in range(1, self.n_pool):
            gain = torch.sparse.mm(masks[i], corrs[i+1]) - corrs[i]
            gain = 2 * torch.sigmoid(gain)
            gains.append(gain)
        for i in range(1, self.n_pool):
            masks[i] = torch.sparse.mm(masks[i - 1], masks[i])

        adjs = []
        for j in range(self.n_feature):
            temp = []
            for i in range(self.n_pool - 1):
                mask = torch.sparse.mm(masks[i], sparse_diag(gains[i][:, j]))
                mask = torch.sparse.mm(mask, masks[i].transpose(0, 1))

                adj = torch.mul(adj, mask)
                temp.append(adj)
            temp = torch.stack(temp)
            adjs.append(torch.sparse.sum(temp, dim=0))






        # GNN 
        for i in range(self.k):
            x = self.prop(x, norm_adj, x0)

        x = self.mlp(x)

        return F.log_softmax(x, dim=1)
