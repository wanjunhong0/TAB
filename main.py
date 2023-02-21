import argparse
import time
import torch
import torch.nn.functional as F
import torchmetrics

from models import GraphCAD
from load_data import Data


"""
===========================================================================
Configuation
===========================================================================
"""
parser = argparse.ArgumentParser(description="Run GraphCAD.")
parser.add_argument('--dataset_path', nargs='?', default='./datasets/', help='Input data path')
parser.add_argument('--dataset', nargs='?', default='house_class', help='Choose a dataset from {house_class, vk_class}')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 norm on parameters)')
parser.add_argument('--k', type=int, default=2, help='Number of Propagation')
parser.add_argument('--n_pool', type=int, default=3, help='Number of Hierarchical Clustering layers')
parser.add_argument('--mlp_cluster', type=int, default=2, help='Number of MLP layer in clustering')
parser.add_argument('--mlp_out', type=int, default=3, help='Number of MLP layer')
parser.add_argument('--n_hidden', type=int, default=64, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
args = parser.parse_args()
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)
# training on the first GPU if not available on CPU
device = torch.device("cpu")
print('Training on device = {}'.format(device))

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
# norm_adj = data.norm_adj.to(device)
label = data.label.to(device)
"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = GraphCAD(args=args, n_sample=data.n_node, n_feature=data.n_feature, n_class=data.n_class, feature_corr=feature_corr).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
metric = torchmetrics.Accuracy(task='multiclass', num_classes=data.n_class).to(device)


torch.autograd.set_detect_anomaly(True)


for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output, loss = model(feature, feature_cov, adj)
    loss_train = F.nll_loss(output[data.idx_train], label[data.idx_train]) + loss
    acc_train = metric(output[data.idx_train].max(1)[1], label[data.idx_train])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output, loss = model(feature, feature_cov, adj)
    loss_val = F.nll_loss(output[data.idx_val], label[data.idx_val]) + loss
    acc_val = metric(output[data.idx_val].max(1)[1], label[data.idx_val])

    print('Epoch {0:04d} | Time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val, acc_train, acc_val))

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output, loss = model(feature, feature_cov, adj)
loss_test = F.nll_loss(output[data.idx_test], label[data.idx_test]) + loss
acc_test = metric(output[data.idx_test].max(1)[1], label[data.idx_test])
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test, acc_test))
