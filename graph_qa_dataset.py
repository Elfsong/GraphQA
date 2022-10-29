# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
from graph_constructor import ConstituencyGraphConstructor


class GraphQADataset():
    def __init__(self, split: str = 'train', size: int = -1):
        assert split in ['train', 'val', 'test']

        self.data_size = size
        self.cgc = ConstituencyGraphConstructor("squad", split, "bert-base-uncased")
        
    
    def process(self):
        self.grapg_data = self.cgc.pipeline(self.size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.grapg_data[idx]