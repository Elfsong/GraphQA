# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     07/11/2022
# ---------------------------------------------------------------- 

import components.hg_utils as hg_utils
from components.hg_dataset import HGDataset

# Dataset Loading
dev_dataset = HGDataset(
    source_path=hg_utils.get_path("./data/squad_v2/raw/dev-v2.0.json"), 
    target_path=hg_utils.get_path("./data/squad_v2/processed/dev/"), 
    start_index=0, 
    end_index=-1, 
    using_cache=False
)




if __name__ == "__main__":
    print("Done")