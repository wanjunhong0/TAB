import torch
from torch_geometric.utils import subgraph


def normalize_adj(adj, symmetric=True):
    """Convert adjacency matrix into normalized laplacian matrix
    Args:
        adj (torch sparse tensor): adjacency matrix
        symmetric (boolean) True: D^{-1/2}AD^{-1/2}; False: D^{-1}A
    Returns:
        (torch sparse tensor): Normalized laplacian matrix
    """
    degree = torch.sparse.sum(adj, dim=0)
    if symmetric:
        degree_ = sparse_diag(degree.pow(-0.5))
        norm_adj = torch.sparse.mm(torch.sparse.mm(degree_, adj), degree_)
    else:
        degree_ = sparse_diag(degree.pow(-1))
        norm_adj = torch.sparse.mm(adj, degree_)

    return norm_adj


def sparse_diag(vector):
    """Convert vector into diagonal matrix
    Args:
        vector (torch tensor): diagonal values of the matrix
    Returns:
        (torch sparse tensor): sparse matrix with only diagonal values
    """
    if not vector.is_sparse:
        vector = vector.to_sparse()
    n = len(vector)
    index = torch.stack([vector._indices()[0], vector._indices()[0]])

    return torch.sparse_coo_tensor(index, vector._values(), [n, n])


def covariance_transform(x):
    n = x.shape[0]

    # x_bar = x.mean(dim=0)
    # covs = []
    # for i in range(n):
    #     x_ = (x[i] - x_bar).reshape(-1, 1)
    #     covs.append(torch.mm(x_, x_.T) * (1 / (n - 1)))
    # covs = torch.stack(covs, dim=0).reshape(n, -1)     # n * d * d --> n * d^2
    
    covs = torch.bmm(x.reshape(n, -1, 1), x.reshape(n, 1, -1))

    return covs

def correlation_readout(cov):
    n = cov.shape[0]
    var = (torch.diagonal(cov, dim1=1, dim2=2) + 1e-7).pow(-0.5)
    corr = cov * var.reshape(n, 1, -1) * var.reshape(n, -1, 1)
    corr = corr - torch.diagonal(corr, dim1=1, dim2=2).diag_embed()  # remove diag values

    return corr.abs().mean(2)


def mask_select(mask, value, x):
    x_indices = []
    x_values = []
    for i in range(mask.size()[1]):
        index = mask.transpose(0, 1)[i].coalesce().indices()[0]
        x_index, x_value = subgraph(index, x.indices(), x.values())
        x_indices.append(x_index)
        x_values.append(x_value * value[i])

    x_indices = torch.cat(x_indices, dim=1)
    x_values = torch.cat(x_values)
    x_select = torch.sparse_coo_tensor(x_indices, x_values, x.size())

    return x_select


class EarlyStopping:
    """Early stops the training if validation metrics doesn't improve after a given patience."""
    def __init__(self, patience=10, mode='min', delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation improved.
            mode (str): Max or min is prefered improvement
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val, model):
        if self.mode == 'min':
            score = -val
        if self.mode == 'max':
            score = val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        self.val_min = val_loss