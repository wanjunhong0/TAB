import torch


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
    x_bar = x.mean(dim=0)
    covs = []
    for i in range(n):
        x_ = (x[i] - x_bar).reshape(-1, 1)
        covs.append(torch.mm(x_, x_.T) * (1 / (n - 1)))
    covs = torch.stack(covs, dim=0).reshape(n, -1)     # n * d * d --> n * d^2

    return covs