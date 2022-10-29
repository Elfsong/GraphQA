# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ----------------------------------------------------------------

import torch
import stanza
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric

from tqdm import tqdm
from typing import List
from datasets import load_dataset
from collections import defaultdict
from torch_geometric.data import HeteroData
from representation_retriever import RepresentationRetriever


class ConstituencyParser(object):
    def __init__(self):
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        self.pos_tags = self.pipeline.processors['pos'].get_known_xpos()

    def get_sentences(self, doc: str) -> List:
        return self.pipeline(doc).sentences
    
    def get_answer(self, answer: str) -> List:
        answer_sentences = self.get_sentences(answer)
        return answer_sentences[0].constituency.leaf_labels()


class GraphConstructor(object):
    def __init__(self, dataset_name: str, dataset_split: str) -> None:
        #TODO(mingzhe): support other datasets
        self.dataset_name = dataset_name
        self.dataset_split_name = dataset_split
        assert self.dataset_name == "squad" and self.dataset_split_name in ["train", "validation"]

        # Load dataset
        self.dataset = load_dataset(self.dataset_name)
        self.dataset_split = self.dataset[self.dataset_split_name] 

    def pipeline(self):
        raise NotImplementedError


class ConstituencyGraphConstructor(GraphConstructor):
    def __init__(self, dataset_name: str, dataset_split: str, representation_model: str) -> None:
        super().__init__(dataset_name, dataset_split)

        # Representation Retriever
        self.r_retriever = RepresentationRetriever(representation_model)

        # Constituency Parser
        self.c_parser = ConstituencyParser()
    
    def set_temporary_variables(self):
        self.X = defaultdict(list)
        self.Y = defaultdict(list)
        self.R = defaultdict(list)
    
    def virtualize(self, graph_data):
        G = torch_geometric.utils.to_networkx(graph_data.to_homogeneous(), to_undirected=False )
        nx.draw_networkx(G)
        plt.savefig("output.jpg")

    def construct(self, current_node, parent_node, answers):
        if not current_node.is_leaf():
            current_label = current_node.label
            current_representation = current_node.leaf_labels()        
            current_index = len(self.X[current_label])
            
            self.X[current_label] += [self.r_retriever.get_pooled_representation(current_representation)]
            self.Y[current_label] += [current_representation in answers]
            self.R[f"{current_label}={parent_node[0]}"] += [[current_index, parent_node[1]]]
            
            for child_node in current_node.children:
                self.construct(child_node, [current_label, current_index], answers)

    def pipeline(self, dataset_size: int) -> list:
        graph_dict = dict()
        for data in tqdm(list(self.dataset_split)[:dataset_size]):
            context = data["context"]
            answers = data["answers"]["text"]
            qid = data["id"]

            self.set_temporary_variables()
            
            for sentence_index, sentence in enumerate(self.c_parser.get_sentences(context)):
                answer_candidates = [self.c_parser.get_answer(answer) for answer in answers]
                self.X["sentence"] += [self.r_retriever.get_pooled_representation(sentence.constituency.leaf_labels())]
                self.Y["sentence"] += [sentence.constituency.leaf_labels() in answer_candidates]
                self.construct(sentence.constituency, ("sentence", sentence_index), answer_candidates)

            graph_data = HeteroData()

            for label in self.X:
                graph_data[label].x = torch.stack(self.X[label])
            
            for label in self.Y:
                graph_data[label].y = torch.tensor(list(map(float, self.Y[label])))
            
            for relation in self.R:
                head, tail = relation.split("=")
                graph_data[head, 'connect', tail].edge_index = torch.tensor(self.R[relation]).t().contiguous()
            
            graph_dict[qid] = graph_data

# Unit test
if __name__ == "__main__":
    cgc = ConstituencyGraphConstructor("squad", "train", "bert-base-uncased")
    cgc.pipeline(10)

