# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
import torch_geometric.transforms as T
from graph_constructor import ConstituencyGraphConstructor


class GraphQADataset():
    def __init__(self, split: str = 'train', data_size: int = -1):
        assert split in ['train', 'validation']
        self.data_size = data_size
        self.cgc = ConstituencyGraphConstructor("squad", split, "bert-base-uncased")
        self.process()
    
    def process(self):
        self.graph_data = self.cgc.pipeline(self.data_size)
    
    def transform(self):
        # TODO(mingzhe): Control Variables
        # self.graph_data = T.ToUndirected()(self.graph_data)
        # self.graph_data = T.AddSelfLoops()(self.graph_data)
        # self.graph_data = T.NormalizeFeatures()(self.graph_data)
        pass

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx]