import torch
import torch.nn.functional as F
from layers import Propagation, MLP, ClusteringLayer
from utils import normalize_adj, sparse_diag, mask_select


class GraphFADE(torch.nn.Module):
    def __init__(self, args, n_sample, n_feature, n_class, feature_corr):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(GraphFADE, self).__init__()

        self.feature_corr = feature_corr
        self.n_feature = n_feature
        self.k = args.k
        self.n_pool = args.n_pool    # add initial 0th pooling 
        node_pool = round(pow(n_sample, 1 / (args.n_pool)))   
        self.dropout = args.dropout
        self.prop = Propagation(alpha=1.)

        self.bn0 = torch.nn.BatchNorm1d(n_feature)
        self.pools = torch.nn.ModuleList()
        n_centroid = n_sample
        for _ in range(self.n_pool):
            n_centroid = round(n_centroid / node_pool)
            self.pools.append(ClusteringLayer(n_feature, args.n_hidden, n_centroid))

        self.mlp = MLP(3, 2 * n_feature, args.n_hidden, n_class, dropout=[0.2, 0.2, 0.2], activation=torch.nn.PReLU())



    def forward(self, x, x_cov, adj):
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
        loss = 0.
        for i in range(self.n_pool):
            x_cov, corr, mask = self.pools[i](x_cov, mask)
            corrs.append(corr)
            masks.append(mask)
            # clustering loss
            if i > 0:
                loss += corr.mean(0)
        loss = loss.mean() / (self.n_pool - 1)
        corrs.append(self.feature_corr)


        
        # Allocation
        gains = []
        for i in range(1, self.n_pool):
            gain = torch.sparse.mm(masks[i], corrs[i+1]) - corrs[i]
            gain = 2 * torch.sigmoid(gain)
            gains.append(gain)

    
        for i in range(1, self.n_pool-1):
            masks[i] = torch.sparse.mm(masks[i - 1], masks[i])

        adjs = []
        # for _ in range(self.n_feature):
        #     adjs.append(normalize_adj(adj))


        for j in range(self.n_feature):
            temp = [adj]
            for i in range(len(gains)):
                mask = torch.sparse.mm(masks[i], sparse_diag(gains[i][:,j]))
                mask = torch.sparse.mm(mask, mask.transpose(0, 1)).coalesce().to_dense()
                temp.append(mask.sparse_mask(adj))
                # temp.append(mask_select(masks[i], gains[i][:,j], adj))
            temp = torch.sparse.sum(torch.stack(temp, dim=0),dim=0)
            temp = torch.sparse.softmax(temp, dim=1)
            adjs.append(temp)


        # GNN 
        for i in range(self.k):
            x = self.prop(x, adjs, x0)

        x = self.mlp(torch.cat([x, x0], dim=1))

        # print(x.isnan().any())
        # print(x.isinf().any())

        return x, loss
