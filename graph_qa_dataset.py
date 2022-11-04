# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 
import torch
import pickle
import torch_geometric.transforms as T

from tqdm import tqdm
from typing import List
from pathlib import Path
from random import sample
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer
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


class SquadDataset(Dataset):
    def __init__(self, split, size, load=False):
        self.split = split
        self.size = size
        self.collection = list()

        if load:
            self.load()
        else:
            self.dataset = load_dataset("squad")[self.split]
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.cgc = ConstituencyGraphConstructor()
            self.build()        

    def build(self):
        print(f"[+] Building {self.split} dataset...")
        for instance in tqdm(list(self.dataset)[:self.size]):
            qid = instance["id"]
            context = instance["context"]
            question = instance["question"]
            answers = instance["answers"]

            graph_instance = self.cgc.construct(instance)

            positive_candidates = graph_instance[1]["positive_candidates"]
            negative_candidates = graph_instance[1]["negative_candidates"]
            candidates = [[" ".join(pc[0]), pc[1], 1] for pc in positive_candidates] + [[" ".join(nc[0]), nc[1], 0] for nc in sample(negative_candidates, len(positive_candidates))]

            for answer in answers["text"]:
                for c in candidates:
                    inputs = self.tokenizer(f"[CLS]{question}[SEP]{context}[SEP]{c[0]}", padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                    
                    instance = {
                        "qid": qid,
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "input_ids": inputs["input_ids"].squeeze(),
                        "attention_mask": inputs["attention_mask"].squeeze(),
                        "answer_embedding": c[1],
                        "label": torch.tensor(c[2])
                    }
                    self.collection += [instance]
                    self.dump(instance)

    def dump(self, data):
        with open(f'./data/{self.split}/{data["qid"]}.pkl', 'wb') as dump_file:
            pickle.dump(data, dump_file)
    
    def load(self):
        print(f"[+] Loading {self.split} dataset...")
        for path in tqdm(Path(f"./data/{self.split}").glob("*.pkl")):
            with open(path, 'rb') as dump_file:
                self.collection += [pickle.load(dump_file)] 

    def __getitem__(self, idx):
        return self.collection[idx]

    def __len__(self):
        return len(self.collection)