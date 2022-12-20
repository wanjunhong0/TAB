import torch
import torch.nn.functional as F
from layers import Propagation, MLP



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

        self.mlp = MLP(3, n_feature, n_hidden, n_class, dropout=[0., 0., 0.], activation=torch.nn.PReLU())
        self.bn0 = torch.nn.BatchNorm1d(n_feature)


    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label

        """
        x = feature = self.bn0(feature)
        for i in range(self.n_layer):
            x = self.prop(x, adj, feature)

        x = self.mlp(x)

        return F.log_softmax(x, dim=1)
