import pandas as pd
import networkx as nx
import json
import torch
from torch_geometric.utils import from_networkx
from utils import normalize_adj, sparse_diag, covariance_transform


class Data():
    def __init__(self, path, dataset):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index

        Args:
            path (str): file path
            dataset (str): dataset name
        """
        
        self.feature = torch.tensor(pd.read_csv(path + dataset + '/X.csv').values, dtype=torch.float).nan_to_num()
        self.n_feature = self.feature.shape[1]
        self.feature_cov = covariance_transform(self.feature)
        self.feature_corr = torch.corrcoef(self.feature.T).abs().sum(1).reshape(1, -1)
        
        self.label = torch.tensor(pd.read_csv(path + dataset + '/y.csv').values, dtype=torch.int64).squeeze(1)
        self.n_class = len(self.label.unique())

        with open('datasets/house_class/masks.json') as f:
            masks = json.load(f)
        self.idx_train = masks['0']['train']
        self.idx_val = masks['0']['val']
        self.idx_test = masks['0']['test']

        G = nx.read_graphml(path + dataset + '/graph.graphml', node_type=int)
        G.graph = {}    # prevent the error of loading graph using from_networkx from pyg
        graph = from_networkx(G)
        self.edge_index = graph.edge_index
        self.n_node = graph.num_nodes
        self.n_edge = graph.num_edges
        self.adj = torch.sparse_coo_tensor(self.edge_index, torch.ones(self.n_edge), [self.n_node, self.n_node])
        self.adj = torch.add(self.adj, sparse_diag(torch.ones(self.n_node))).coalesce()
        self.norm_adj = normalize_adj(self.adj, symmetric=True)
