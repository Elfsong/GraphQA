# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

from typing import Any
from torch import tensor
from transformers import AutoTokenizer, AutoModel

class RepresentationRetriever(object):
    def __init__(self, model_name: str = 'bert-base-uncased') -> None:
        # TODO(mingzhe): support multiple models
        assert model_name == 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_model_output(self, text: str) -> Any:
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        return output

    def get_pooled_representation(self, text: str) -> tensor:
        model_output = self.get_model_output(text)
        pooled_representation = model_output.pooler_output
        return pooled_representation

    def get_pooled_representation(self, token_list: list) -> tensor:
        model_output = self.get_model_output(" ".join(token_list))
        pooled_representation = model_output.pooler_output
        return pooled_representation


# Unit tests
if __name__ == "__main__":
    rr = RepresentationRetriever("bert-base-uncased")

    text = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season."""
    text_r = rr.get_pooled_representation(text)

    print(text_r.shape)