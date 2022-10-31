# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     31/10/2022
# ---------------------------------------------------------------- 

from graph_qa_dataset import GraphQADataset

train_dataset = GraphQADataset(split="train", data_size=50)
val_dataset = GraphQADataset(split="validation", data_size=10)

train_dataset.dump("train.pkl")
val_dataset.dump("val.pkl")