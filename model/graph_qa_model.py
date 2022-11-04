# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GATConv, Linear, HGTConv

# Pytorch Geometric provides three ways for the user to create models on heterogeneous graph data:

# 1) Automatically convert a homogenous GNN model to a heterogeneous GNN model by making use of torch_geometric.nn.to_hetero() or torch_geometric.nn.to_hetero_with_bases().

# 2) Define inidividual functions for different types using PyGs wrapper torch_geometric.nn.conv.HeteroConv for heterogeneous convolution.

# 3) Deploy existing (or write your own) heterogeneous GNN operators.

# More on https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['context'])

class GraphQA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.linear = Linear(in_channels=768, out_channels=2)

        self.linear1 = torch.nn.Linear(768, 1) 
        self.relu_1 = torch.nn.ReLU()
        self.dropout = torch.nn.dropout(p=0.1)

    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        output = output["pooler_output"]
        output = self.linear1(output)
        output = self.relu_1(output)
        output = self.dropout(output)
        return torch.sigmoid(output)