import argparse
import time
import torch
import torchmetrics

from models import GraphFADE
from load_data import Data
from utils import EarlyStopping


"""
===========================================================================
Configuation
===========================================================================
"""
parser = argparse.ArgumentParser(description="Run GraphFADE.")
parser.add_argument('--dataset_path', type=str, default='./datasets/', help='Input data path')
parser.add_argument('--model_path', type=str, default='checkpoint.pt', help='Saved model path.')
parser.add_argument('--dataset', type=str, default='avazu', help='Choose a dataset from {house_class, vk_class, avazu, county}')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--epoch', type=int, default=5000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay (L2 norm on parameters)')
parser.add_argument('--k', type=int, default=4, help='Number of Propagation')
parser.add_argument('--n_hidden', type=int, default=64, help='Number of hidden units')
parser.add_argument('--n_pool', type=int, default=4, help='Number of Hierarchical Clustering layers')
parser.add_argument('--mlp_cluster', type=int, default=2, help='Number of MLP layer in clustering')
parser.add_argument('--mlp_out', type=int, default=3, help='Number of MLP layer')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.8, help='Porpagation alpha')
parser.add_argument('--beta', type=float, default=0.1, help='Clustering loss beta')
parser.add_argument('--sparse', action='store_true', help='Sparse mask select')
parser.add_argument('--device', type=str, default="cuda:0", help='Device to run on')
parser.add_argument('--patience', type=int, default=200, help='How long to wait after last time validation improved')
args = parser.parse_args()
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)
device = torch.device(args.device)

"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.dataset_path, dataset=args.dataset)
print('Loaded {0} dataset with {1} nodes and {2} edges'.format(args.dataset, data.n_node, data.n_edge))
feature = data.feature.to(device)
feature_cov = data.feature_cov.to(device)
feature_corr = data.feature_corr.to(device)
adj = data.adj.to(device)
label = data.label.to(device)
"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = GraphFADE(args=args, n_sample=data.n_node, n_feature=data.n_feature, n_class=data.n_class, feature_corr=feature_corr).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if data.task == 'classification':
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = 'ACC'
    metric_fn = torchmetrics.Accuracy(task='multiclass', num_classes=data.n_class).to(device)
    early_stop = EarlyStopping(patience=args.patience, mode='max', path=args.model_path)
if data.task == 'regression':
    loss_fn  = torch.nn.MSELoss()
    metric = 'RMSE'
    metric_fn  = torch.nn.MSELoss()
    early_stop = EarlyStopping(patience=args.patience, mode='min', path=args.model_path)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output, loss = model(feature, feature_cov, adj)
    loss_train = loss_fn(output[data.idx_train], label[data.idx_train]) + loss
    if data.task == 'classification':
        metric_train = metric_fn(output[data.idx_train].max(1)[1], label[data.idx_train])
    if data.task == 'regression':
        metric_train = metric_fn(output[data.idx_train], label[data.idx_train]).sqrt()
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output, loss = model(feature, feature_cov, adj)
    loss_val = loss_fn(output[data.idx_val], label[data.idx_val]) + loss
    if data.task == 'classification':
        metric_val = metric_fn(output[data.idx_val].max(1)[1], label[data.idx_val])
    if data.task == 'regression':
        metric_val = metric_fn(output[data.idx_val], label[data.idx_val]).sqrt()

    print('Epoch {0:04d} | Time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | {6} = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val, metric_train, metric_val, metric))

    # Early stop
    early_stop(metric_val, model)
    if early_stop.early_stop:
        print('Early stop triggered at epoch {0}!'.format(epoch - args.patience))
        model.load_state_dict(torch.load(args.model_path))
        break

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output, loss = model(feature, feature_cov, adj)
loss_test = loss_fn(output[data.idx_test], label[data.idx_test]) + loss
if data.task == 'classification':
    metric_test = metric_fn(output[data.idx_test].max(1)[1], label[data.idx_test])
if data.task == 'regression':
    metric_test = metric_fn(output[data.idx_test], label[data.idx_test]).sqrt()
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | {2} = [test: {1:.4f}]'.format(loss_test, metric_test, metric))
