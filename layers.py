import torch
import torch.nn.functional as F
from sparsemax import Sparsemax


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


    def forward(self, x, adj, h):
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
            xs.append(torch.sparse.mm(adj, x[:, i].reshape(-1, 1)))
        output = torch.cat(xs, dim=1)

        return (1 - self.alpha) * output + self.alpha * h


class ClusteringLayer(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_cluster):
        super(ClusteringLayer, self).__init__()

        self.n_feature = n_feature
        self.n_cluster = n_cluster
        self.mlp = MLP(2, n_feature, n_hidden, n_cluster, bn=True, activation=torch.nn.ReLU(), dropout=[0., 0., 0.])
        self.sparsemax = Sparsemax(dim=0)

    def correlation_readout(self, x_cov):
        n = x_cov.shape[0]
        diag_idx = torch.arange(self.n_feature * self.n_feature).reshape(self.n_feature, -1).diag()
        var = x_cov[:, diag_idx].pow(-0.5)
        # var = torch.bmm(var.reshape(n, self.n_feature, 1), var.reshape(n, 1, self.n_feature)).reshape(n, -1)
        var = torch.mul(torch.cat([var for _ in range(self.n_feature)], dim=1), var.repeat_interleave(self.n_feature, 1))
        corr = torch.mul(var, x_cov)
        x_corr = corr.reshape(n, self.n_feature, -1).sum(1)

        ##### too slow  ######
        # x_cov = x_cov.reshape(n, self.n_feature, self.n_feature)
        # corrs = []
        # for i in range(n):
        #     cov = x_cov[i]
        #     var = torch.diag(cov.diag().pow(-0.5))
        #     corr = torch.mm(torch.mm(var, cov), var)
        #     corrs.append(corr.sum(0))
        # x_corr = torch.stack(corrs, dim=0)

        return x_corr

    def forward(self, x_cov, mask):
        x_cov = torch.sparse.mm(mask.transpose(0, 1), x_cov)
        x_corr = self.correlation_readout(x_cov)
        # mask = self.sparsemax(self.mlp(x_corr))
        mask = F.gumbel_softmax(self.mlp(x_corr), hard=True, dim=1) 

        return x_cov, x_corr, mask.to_sparse()









class MLP(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, dropout, bn=True, activation=torch.nn.ReLU()):
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
        self.dropout = dropout

    def forward(self, x):
        for i in range(self.n_layer):
            if self.bn: 
                x = self.bns[i](x)
            x = F.dropout(x, self.dropout[i], training=self.training)
            x = self.lins[i](x)
            if i < self.n_layer - 1:
                x = self.act(x)

        return x

