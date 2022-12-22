import torch
import torch.nn.functional as F
from layers import Propagation, MLP, ClusteringLayer
from utils import covariance_transform



class GraphCAD(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, dropout):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(GraphCAD, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.prop = Propagation(alpha=0.)

        self.bn0 = torch.nn.BatchNorm1d(n_feature)
        self.cluster = ClusteringLayer(n_feature, n_hidden, n_class)
        self.mlp = MLP(3, n_feature, n_hidden, n_class, dropout=[0., 0., 0.], activation=torch.nn.PReLU())


    def forward(self, x, x_cov, adj, norm_adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label

        """
        x = x0 = self.bn0(x)

        _, _, mask = self.cluster(x_cov, adj)

        for i in range(self.n_layer):
            x = self.prop(x, norm_adj, x0)

        x = self.mlp(x)

        return F.log_softmax(x, dim=1)
