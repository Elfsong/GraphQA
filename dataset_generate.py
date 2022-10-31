# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     31/10/2022
# ---------------------------------------------------------------- 

from multiprocessing import Pool
from graph_qa_dataset import GraphQADataset

# Train Dataset Process
train_dataset = GraphQADataset(split="train", data_range=[0, 5000])
train_dataset.process()

# Val Dataset Process
val_dataset = GraphQADataset(split="validation", data_range=[0, 500])
val_dataset.process()


# # Train Dataset Load
# train_dataset.load()

# # Val Dataset Load
# val_dataset.load()

