# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
from typing import Any
from torch import tensor
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel

class RepresentationRetriever(object):
    def __init__(self, model_name: str = 'bert-base-uncased') -> None:
        # TODO(mingzhe): support multiple models
        assert model_name == 'bert-base-uncased'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @lru_cache(maxsize=1024, typed=False)
    def get_model_output(self, text: str) -> Any:
        encoded_input = self.tokenizer(text, max_length=64, padding="max_length", truncation=True,return_tensors='pt').to(self.device)
        model_output = self.model(**encoded_input)
        pooled_representation = model_output.pooler_output
        result = pooled_representation[0].detach()
        return result

    def get_pooled_representation_str(self, text: str) -> tensor:
        model_output = self.get_model_output(text)
        return model_output

    def get_pooled_representation(self, token_list: list) -> tensor:
        model_output = self.get_model_output(" ".join(token_list))
        return model_output


# Unit tests
if __name__ == "__main__":
    rr = RepresentationRetriever("bert-base-uncased")

    text = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season."""
    text_r = rr.get_pooled_representation(text)

    print(text_r.shape)