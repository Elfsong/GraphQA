# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     07/11/2022
# ---------------------------------------------------------------- 

import stanza
from typing import List
from functools import lru_cache

class ConstituencyParser(object):
    def __init__(self):
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

    @lru_cache(maxsize=64, typed=False)
    def get_sentences(self, doc: str) -> List:
        return self.pipeline(doc).sentences
    
    def get_labels(self, answer: str) -> List:
        answer_sentences = self.get_sentences(answer)
        return answer_sentences[0].constituency.leaf_labels()

# Unit test
if __name__ == "__main__":
    # 1. Constituency Parser
    cp = ConstituencyParser()

    # 1.1 Parsing text and lru_cache test
    for i in range(10):
        text = "The Norman's dynasty had a major political, cultural and military impact on medieval Europe and even the Near East."
        result = cp.get_sentences(text)
        print(result)

    

    