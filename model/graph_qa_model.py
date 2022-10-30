import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Pytorch Geometric provides three ways for the user to create models on heterogeneous graph data:

# 1) Automatically convert a homogenous GNN model to a heterogeneous GNN model by making use of torch_geometric.nn.to_hetero() or torch_geometric.nn.to_hetero_with_bases().

# 2) Define inidividual functions for different types using PyGs wrapper torch_geometric.nn.conv.HeteroConv for heterogeneous convolution.

# 3) Deploy existing (or write your own) heterogeneous GNN operators.

# More on https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html

class GraphQAModel(torch.nn.Module):
    def __init__(self, num_features: int = 768, num_classes: int = 2):
        super().__init__()
        self.conv1 = GATConv(num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x
