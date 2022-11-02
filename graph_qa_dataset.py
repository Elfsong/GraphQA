# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import pickle
import torch_geometric.transforms as T

from tqdm import tqdm
from typing import List
from pathlib import Path
from graph_constructor import ConstituencyGraphConstructor

class GraphQADataset():
    def __init__(self, split: str = 'train', data_range: List[int] = [0, -1]):
        assert split in ['train', 'validation']
        self.data_range = data_range
        self.split = split
        self.cgc = ConstituencyGraphConstructor("squad", split, "bert-base-uncased")
        self.graph_data = list()
    
    def process(self):
        print(f"[+] Processing {self.split} dataset...")
        self.graph_data = self.cgc.pipeline(self.data_range)

    def load(self):
        print(f"[+] Loading {self.split} dataset...")
        for path in tqdm(Path(f"./data/{self.split}").glob("*.pkl")):
            with open(path, 'rb') as dump_file:
                self.graph_data += [pickle.load(dump_file)]
    
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