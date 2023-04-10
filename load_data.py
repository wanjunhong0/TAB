import pandas as pd
import networkx as nx
import json
import torch
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from utils import normalize_adj, sparse_diag, covariance_transform, correlation_readout
from category_encoders import CatBoostEncoder
import numpy as np
from sklearn import preprocessing


class Data():
    def __init__(self, path, dataset):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index

        Args:
            path (str): file path
            dataset (str): dataset name
        """
        feature = pd.read_csv(path + dataset + '/X.csv').fillna(0).values


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


        if dataset == 'avazu':
            # with open(path + dataset +  '/cat_features.txt') as f:
            #     cat_features = f.read().splitlines() 
            # cat_features = np.where(columns.isin(categorical_columns))[0]
            feature = feature.astype(str)
            cat_features = np.arange(feature.shape[1])
            enc = CatBoostEncoder()
            feature[np.ix_(self.idx_train, cat_features)] = enc.fit_transform(feature[np.ix_(self.idx_train, cat_features)], label[self.idx_train])
            feature[np.ix_(self.idx_val + self.idx_test, cat_features)] = enc.transform(feature[np.ix_(self.idx_val + self.idx_test, cat_features)])
            feature = feature.astype(float)

        # normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        feature[self.idx_train] = min_max_scaler.fit_transform(feature[self.idx_train])
        feature[self.idx_val + self.idx_test] = min_max_scaler.transform(feature[self.idx_val + self.idx_test])


        self.feature = torch.tensor(feature, dtype=torch.float)
        self.n_feature = self.feature.shape[1]
        self.feature_cov = covariance_transform(self.feature)  # n * d * d
        self.feature_corr = correlation_readout(self.feature_cov.sum(0, keepdim=True))



        G = nx.read_graphml(path + dataset + '/graph.graphml', node_type=int)
        G.graph = {}    # prevent the error of loading graph using from_networkx from pyg
        graph = from_networkx(G)
        self.edge_index = graph.edge_index
        self.n_node = graph.num_nodes
        self.n_edge = graph.num_edges
        self.adj = torch.sparse_coo_tensor(self.edge_index, torch.ones(self.n_edge), [self.n_node, self.n_node])
        self.adj = torch.add(self.adj, sparse_diag(torch.ones(self.n_node))).coalesce()
        # self.norm_adj = normalize_adj(self.adj, symmetric=True)
