# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ----------------------------------------------------------------

import torch
import stanza
import pickle
import numpy as np
import editdistance
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
import torch_geometric.transforms as T

from tqdm import tqdm
from typing import List
from functools import lru_cache
from datasets import load_dataset
from collections import defaultdict
from torch_geometric.data import HeteroData
from achieve.representation_retriever import RepresentationRetriever


class ConstituencyParser(object):
    def __init__(self):
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        self.pos_tags = self.pipeline.processors['pos'].get_known_xpos()

    @lru_cache(maxsize=64, typed=False)
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
    
    def virtualize(self, index, graph_data):
        G = torch_geometric.utils.to_networkx(graph_data.to_homogeneous(), to_undirected=False )
        plt.figure(figsize=(50, 50))
        nx.draw_networkx(G)
        plt.savefig(f"output_{index}.jpg")
    
    def dump_instance(self, graph_data):
        with open(f'./data/{self.dataset_split_name}/{graph_data[1]["qid"]}.pkl', 'wb') as dump_file:
            pickle.dump(graph_data, dump_file)

    def pipeline(self):
        raise NotImplementedError

class ConstituencyGraphConstructor(GraphConstructor):
    def __init__(self, representation_model: str = "bert-base-uncased") -> None:
        # Representation Retriever
        self.r_retriever = RepresentationRetriever(representation_model)

        # Constituency Parser
        self.c_parser = ConstituencyParser()

        # MetaData
        self.metadata = (
            ['context'], [('context', 'connect', 'context')]
        )
    
    def set_temporary_variables(self):
        self.X = defaultdict(list)
        self.Y = defaultdict(list)
        self.T = defaultdict(list)
        self.R = defaultdict(list)

    def construct_context(self, current_node, parent_node, answer_candidates):
        if not current_node.is_leaf():
            current_label = "context"
            current_span = current_node.leaf_labels()
            current_representation = self.r_retriever.get_pooled_representation(current_span)
            current_index = len(self.X[current_label])
            
            self.X[current_label] += [current_representation]
            self.Y[current_label] += [max([self.distance(current_span, answer) for answer in answer_candidates])]
            self.T[current_label] += [(current_span, current_representation)]
            self.R[f"{current_label}={parent_node[0]}"] += [[current_index, parent_node[1]]]
            
            for child_node in current_node.children:
                self.construct_context(child_node, [current_label, current_index], answer_candidates)

    def distance(self, source, target):
        # distance = editdistance.eval(source, target)
        d = 1 if source == target else 0
        return d
    
    def distance_normalize(self, data):
        data = np.array(data)
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def construct(self, data):
        context = data["context"]
        question = data["question"]
        answers = data["answers"]["text"]
        qid = data["id"]

        self.set_temporary_variables()
        
        # Context Graph
        current_reresentation = self.r_retriever.get_pooled_representation_str(context)
        self.X["context"] += [current_reresentation]
        self.Y["context"] += [max(self.distance(context, answer) for answer in answers)]
        self.T["context"] += [(context, current_reresentation)]

        answer_candidates = [self.c_parser.get_answer(answer) for answer in answers]

        for sentence_index, sentence in enumerate(self.c_parser.get_sentences(context)):       
            current_index = len(self.X["context"])
            current_span = sentence.constituency.leaf_labels()
            current_reresentation = self.r_retriever.get_pooled_representation(current_span)
            self.X["context"] += [current_reresentation]
            self.Y["context"] += [max([self.distance(context, answer) for answer in answer_candidates])]
            self.T["context"] += [(current_span, current_reresentation)]
            self.R["context=context"] += [[current_index, 0]]
            self.construct_context(sentence.constituency, ("context", current_index), answer_candidates)

        graph_data = HeteroData()

        for label in self.X:
            graph_data[label].x = torch.stack(self.X[label])

        for label in self.Y:
            graph_data[label].y = torch.tensor(self.distance_normalize(self.Y[label]))
        
        for relation in self.R:
            head, tail = relation.split("=")
            graph_data[head, 'connect', tail].edge_index = torch.tensor(self.R[relation]).t().contiguous()

        positive_candidates, negative_candidates = list(), list()
        for spans in self.T["context"]:
            for answer in answer_candidates:
                if answer == spans[0]:
                    positive_candidates += [spans]
                else:
                    negative_candidates += [spans]

        # Augmentation
        graph_data = T.ToUndirected()(graph_data)
        
        instance = [graph_data, {"qid": qid, "context": context, "question": question, "answers": answers, "positive_candidates": positive_candidates, "negative_candidates": negative_candidates}]
        return instance

    def pipeline(self, range: List[int]) -> list:
        graph_list = list()
        for data in tqdm(list(self.dataset_split)[range[0]:range[1]]):
            instance = self.construct(data)
            self.dump_instance(instance)
            graph_list += [instance]
        return graph_list

# Unit test
if __name__ == "__main__":
    cgc = ConstituencyGraphConstructor("squad", "train", "bert-base-uncased")

    # gd = cgc.construct({
    #     "context": "Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers.",
    #     "question": "In what year did the student paper Common Sense begin publication at Notre Dame?",
    #     "answers": {
    #         "text": [
    #             "1987"
    #         ]
    #     },
    #     "id": '5733bf84d058e614000b61c1'
    # })

    # cgc.virtualize(0, gd[0])
    # print(gd)

    gds = cgc.pipeline([0, 10])
    for index, gd in enumerate(gds):
        print(gd)
        cgc.virtualize(index, gd[0])


