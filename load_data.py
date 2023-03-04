import pandas as pd
import networkx as nx
import json
import torch
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from utils import normalize_adj, sparse_diag, covariance_transform, correlation_readout


class Data():
    def __init__(self, path, dataset):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index

        Args:
            path (str): file path
            dataset (str): dataset name
        """
        feature = pd.read_csv(path + dataset + '/X.csv').values
        self.feature = torch.tensor(feature, dtype=torch.float).nan_to_num()
        # normalization
        # self.feature = F.normalize(self.feature, p=1, dim=1)
        self.feature = (self.feature - self.feature.mean(dim=1, keepdim=True)) / self.feature.std(dim=1, keepdim=True)
        self.n_feature = self.feature.shape[1]
        self.feature_cov = covariance_transform(self.feature)  # n * d * d
        self.feature_corr = correlation_readout(self.feature_cov.sum(0, keepdim=True))
        # print('f_corr: [{}, {}]'.format(self.feature_corr.abs().min(), self.feature_corr.max()))
        # corr = (torch.corrcoef(self.feature.T) - torch.eye(self.n_feature)).abs().mean(0)
        # print('f_corr: [{}, {}]'.format(corr.abs().min(), corr.max()))

        label = pd.read_csv(path + dataset + '/y.csv').values
        
        if dataset in ['house_class', 'vk_class']:
            self.task = 'classification'
            self.label = torch.tensor(label, dtype=torch.long).squeeze(1)
            self.n_class = len(self.label.unique())
        if dataset in ['avazu', 'house', 'county', 'vk']:
            self.task = 'regression'
            self.label = torch.tensor(label, dtype=torch.float)
            self.n_class = 1


        with open(path + dataset + '/masks.json') as f:
            masks = json.load(f)
        self.idx_train = masks['0']['train']
        self.idx_val = masks['0']['test']
        self.idx_test = masks['0']['test']

        G = nx.read_graphml(path + dataset + '/graph.graphml', node_type=int)
        G.graph = {}    # prevent the error of loading graph using from_networkx from pyg
        graph = from_networkx(G)
        self.edge_index = graph.edge_index
        self.n_node = graph.num_nodes
        self.n_edge = graph.num_edges
        self.adj = torch.sparse_coo_tensor(self.edge_index, torch.ones(self.n_edge), [self.n_node, self.n_node])
        self.adj = torch.add(self.adj, sparse_diag(torch.ones(self.n_node))).coalesce()
        # self.norm_adj = normalize_adj(self.adj, symmetric=True)
