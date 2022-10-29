import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

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
