# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ----------------------------------------------------------------

import torch
import stanza
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
from representation_retriever import RepresentationRetriever


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

    def pipeline(self):
        raise NotImplementedError

class ConstituencyGraphConstructor(GraphConstructor):
    def __init__(self, dataset_name: str, dataset_split: str, representation_model: str) -> None:
        super().__init__(dataset_name, dataset_split)

        # Representation Retriever
        self.r_retriever = RepresentationRetriever(representation_model)

        # Constituency Parser
        self.c_parser = ConstituencyParser()

        # MetaData
        self.metadata = (
            ['question', 'context'], 
            [
                ('question', 'connect', 'question'), 
                ('context', 'connect', 'question'), 
                ('context', 'connect', 'context'), 
                ('question', 'rev_connect', 'context')
            ]
        )
    
    def set_temporary_variables(self):
        self.X = defaultdict(list)
        self.Y = defaultdict(list)
        self.R = defaultdict(list)
    
    def virtualize(self, graph_data):
        G = torch_geometric.utils.to_networkx(graph_data.to_homogeneous(), to_undirected=False )
        nx.draw_networkx(G)
        plt.savefig("output.jpg")

    def construct_context(self, current_node, parent_node, answers):
        if not current_node.is_leaf():
            current_label = "context"
            current_representation = current_node.leaf_labels()        
            current_index = len(self.X[current_label])
            
            self.X[current_label] += [self.r_retriever.get_pooled_representation(current_representation)]
            self.Y[current_label] += [[1., 0.] if current_representation in answers else [0., 1.]]
            self.R[f"{current_label}={parent_node[0]}"] += [[current_index, parent_node[1]]]
            
            for child_node in current_node.children:
                self.construct_context(child_node, [current_label, current_index], answers)
    
    def construct_question(self, current_node, parent_node):
        if not current_node.is_leaf():
            current_label = "question"
            current_representation = current_node.leaf_labels()        
            current_index = len(self.X[current_label])
            
            self.X[current_label] += [self.r_retriever.get_pooled_representation(current_representation)]
            self.Y[current_label] += [[0., 1.]]
            self.R[f"{current_label}={parent_node[0]}"] += [[current_index, parent_node[1]]]
            
            for child_node in current_node.children:
                self.construct_question(child_node, [current_label, current_index])


    def pipeline(self, dataset_size: int) -> list:
        graph_list = list()
        for data in tqdm(list(self.dataset_split)[:dataset_size]):
            context = data["context"]
            question = data["question"]
            answers = data["answers"]["text"]
            qid = data["id"]

            self.set_temporary_variables()

            # Question Graph
            sentence = self.c_parser.get_sentences(question)[0]
            self.X["question"] += [self.r_retriever.get_pooled_representation(sentence.constituency.leaf_labels())]
            self.Y["question"] += [[0., 1.]]
            self.construct_question(sentence.constituency, ("question", 0))
            
            # Context Graph
            for sentence_index, sentence in enumerate(self.c_parser.get_sentences(context)):
                answer_candidates = [self.c_parser.get_answer(answer) for answer in answers]
                self.X["context"] += [self.r_retriever.get_pooled_representation(sentence.constituency.leaf_labels())]
                self.Y["context"] += [[1., 0.] if sentence.constituency.leaf_labels() in answer_candidates else [0., 1.]]

                self.R["context=question"] += [[sentence_index, 0]]

                self.construct_context(sentence.constituency, ("context", sentence_index), answer_candidates)

            graph_data = HeteroData()

            for label in self.X:
                graph_data[label].x = torch.stack(self.X[label])
            
            for label in self.Y:
                graph_data[label].y = torch.tensor(self.Y[label])
            
            for relation in self.R:
                head, tail = relation.split("=")
                graph_data[head, 'connect', tail].edge_index = torch.tensor(self.R[relation]).t().contiguous()

            # Convert the graph to undirected graph
            graph_data = T.ToUndirected()(graph_data)
            
            graph_list += [[graph_data, {"qid": qid, "context": context, "question": question, "answers": answers}]]
        return graph_list

# Unit test
if __name__ == "__main__":
    cgc = ConstituencyGraphConstructor("squad", "train", "bert-base-uncased")

    gd = cgc.pipeline(3)
    for qid in gd:
        print(gd)

