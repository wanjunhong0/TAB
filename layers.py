import torch
import torch.nn.functional as F
from utils import correlation_readout


class Propagation(torch.nn.Module):
    def __init__(self, alpha):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
            alpha (float): hyperparameter
        """
        super(Propagation, self).__init__()

        self.alpha = alpha

    def forward(self, x, adjs, h):
        """
        Args:
            x (torch tensor): H Hiddens
            adj (torch tensor): L Laplacian matrix
            h (torch tensor): H0
        Returns:
            (torch tensor): (1 - alpha) * A * H + alpha * H0
        """

        # adj = torch.stack([adj for _ in  range(input.shape[1])], dim=0)
        # x = input.T.unsqueeze(2)
        # output = torch.bmm(adj, x)[:,:,0].T

        xs = []
        for i in range(x.shape[1]):
            xs.append(torch.sparse.mm(adjs[i], x[:, i].reshape(-1, 1)))
        output = torch.cat(xs, dim=1)

        return (1 - self.alpha) * output + self.alpha * h


class ClusteringLayer(torch.nn.Module):
    def __init__(self, mlp_layer, n_feature, n_hidden, n_cluster):
        super(ClusteringLayer, self).__init__()

        self.n_feature = n_feature
        self.n_cluster = n_cluster
        self.mlp = MLP(mlp_layer, n_feature, n_hidden, n_cluster)

    def forward(self, x_cov, mask):
        x_cov = torch.sparse.mm(mask.transpose(0, 1), x_cov.reshape(-1, self.n_feature * self.n_feature)).reshape(-1, self.n_feature, self.n_feature)
        x_corr = correlation_readout(x_cov)
        mask = F.gumbel_softmax(self.mlp(x_corr), hard=True, dim=1) 

        return x_cov, x_corr, mask.to_sparse()


class MLP(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, bn=True, activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(n_layer):
            dim_in = n_feature if i == 0 else n_hidden
            dim_out = n_class if i == n_layer - 1 else n_hidden
            self.bns.append(torch.nn.BatchNorm1d(dim_in))
            self.lins.append(torch.nn.Linear(dim_in, dim_out))

        self.n_layer = n_layer
        self.act = activation
        self.bn = bn

    def forward(self, x):
        for i in range(self.n_layer):
            if self.bn: 
                x = self.bns[i](x)
            x = self.lins[i](x)
            if i < self.n_layer - 1:
                x = self.act(x)

        return x

