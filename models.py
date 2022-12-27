import torch
import torch.nn.functional as F
from layers import Propagation, MLP, ClusteringLayer


class GraphCAD(torch.nn.Module):
    def __init__(self, args, n_sample, n_feature, n_class):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(GraphCAD, self).__init__()

        self.k = args.k
        self.n_pool = args.n_pool
        node_pool = round(pow(n_sample, 1 / (self.n_pool)))
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

        corrs.append(torch.corrcoef(x.T).sum(1) / 6)
        
        # Allocation
        # gains = []
        # for i in reversed(range(self.n_pool)):
        #     gain = 






        # GNN 
        for i in range(self.k):
            x = self.prop(x, norm_adj, x0)

        x = self.mlp(x)

        return F.log_softmax(x, dim=1)
