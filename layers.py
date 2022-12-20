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

        self.mlp = MLP(2, n_feature, n_hidden, n_cluster, bn=True, activation=torch.nn.ReLU(), dropout=[0., 0., 0.])

    def forward(self, x):

        mask = Sparsemax(self.mlp(x), dim=1)


        return mask









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

