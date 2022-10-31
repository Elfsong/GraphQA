# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import pickle
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
    
    def dump(self, filename):
        with open(f'./data/{filename}', 'wb') as dump_file:
            pickle.dump(self.graph_data, dump_file)
        print(f"Dumped {len(self.graph_data)} Graphs Successfully!")

    def load(self, filename):
        with open(f'./data/{filename}', 'rb') as dump_file:
            pickle.load(self.graph_data, dump_file)
        print(f"Loaded {len(self.graph_data)} Graphs Successfully!")
    
    def transform(self):
        # TODO(mingzhe): Control Variables
        # self.graph_data = T.ToUndirected()(self.graph_data)
        # self.graph_data = T.AddSelfLoops()(self.graph_data)
        # self.graph_data = T.NormalizeFeatures()(self.graph_data)
        pass
    
    @property
    def metadata(self):
        return self.cgc.metadata

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx][0]