# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     07/11/2022
# ---------------------------------------------------------------- 

import json
from . import hg_utils
from typing import Any
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HGDataset(Dataset):
    def __init__(self, source_path: Path = None, target_path: Path = None, start_index: int = 0, end_index: int = -1, using_cache: bool = False, tokenizer_name: str = "bert-base-uncased") -> None:
        # Raw data source (Pathlib Path)
        self.source_path = source_path
        self.target_path = target_path

        # Data read indexes
        self.start_index = start_index
        self.end_index = end_index

        # Whether using cache or processing from scratch
        self.using_cache = using_cache

        # Tokenizer config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Processed Data Collection
        self.data_collection = list()

        # Processing from scratch / Loading from cache
        if self.using_cache:
            hg_utils.logger.info(f"[+] Loading cache from {self.target_path}")
            self.load()
        else:
            hg_utils.logger.info(f"[+] processing instances from {self.source_path} in range of [{self.start_index}:{self.end_index}]")
            self.process()
    
    def dump(self) -> None:
        #TODO(mingzhe): implement it
        raise NotImplementedError
    
    def load(self) -> None:
        #TODO(mingzhe): implement it
        raise NotImplementedError

    def process(self) -> None:
        # Load raw data from the given path
        with open(self.source_path, "r") as source_file:
            raw_data = json.load(source_file)["data"]
        
        # Converting the raw data to features
        hg_utils.logger.info(f"[-] Converting the raw data to features...")
        self.data_collection = hg_utils.raw_data_process(raw_data)
        hg_utils.logger.info(f"[-] Convert the raw data to features successfully!")

        # Converting features to squad data format
        hg_utils.logger.info(f"[-] Converting features to squad data format...")
        self.data_collection = hg_utils.calculate_token_position(self.tokenizer, self.data_collection)
        hg_utils.logger.info(f"[-] Convert the features to squad data format successfully!")
    
    def __getitem__(self, idx: int) -> Any:
        return self.data_collection[idx]

    def __len__(self) -> int:
        return len(self.data_collection)


# Unit test
if __name__ == "__main__":
    # 1. Create a new dataset
    source_path = hg_utils.get_path("./data/squad_v2/raw/dev-v2.0.json")
    target_path = hg_utils.get_path("./data/squad_v2/processed/dev/")
    train_dataset = HGDataset(source_path=source_path, target_path=target_path, start_index=0, end_index=-1, using_cache=False)



