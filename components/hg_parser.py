# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     07/11/2022
# ---------------------------------------------------------------- 

import time
import stanza
from tqdm import tqdm
from typing import List
from functools import lru_cache
from multiprocessing import Pool

class ConstituencyNode(object):
    def __init__(self, cid, label, text, lids, children=[], is_answer=False):
        self.cid = cid
        self.label = label
        self.text = text
        self.lids = lids
        self.children = children
        self.is_answer = is_answer
    
    def __str__(self):
        return f"cid: {self.cid} | label: {self.label} | text: {self.text} | lids: {self.lids} | children: {[child.cid for child in self.children]} | answer: {self.is_answer}"

class ConstituencyParser(object):
    def __init__(self, use_gpu: bool = True):
        self.pipelines = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', use_gpu=use_gpu)

    @lru_cache(maxsize=64, typed=False)
    def get_sentences(self, doc: str) -> List:
        sentences = self.pipelines(doc).sentences
        return sentences
    
    def get_labels(self, answer: str) -> List:
        answer_sentences = self.get_sentences(answer)
        return answer_sentences[0].constituency.leaf_labels()

# Unit test
if __name__ == "__main__":
    # 1. Constituency Parser
    cp = ConstituencyParser(use_gpu=True)


    def process(text, p_index):
        result = cp.get_sentences(text)
        return result[0].constituency

    for i in tqdm(range(100)):
        text = "The Norman's dynasty had a major political, cultural and military impact on medieval Europe and even the Near East."
        results = process(text, 0)
    

    