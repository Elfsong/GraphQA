# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     31/10/2022
# ---------------------------------------------------------------- 

import time
import argparse
import numpy as np
from multiprocessing import Pool
from achieve.graph_qa_dataset import GraphQADataset

parser = argparse.ArgumentParser()
parser.add_argument('--worker_num', type=int, required=True, help='worker num [CHANGE IT] with the trade off of speed and GPU capacity')
parser.add_argument('--split', type=str, required=True, help='the name of dataset split [train/validation]')
parser.add_argument('--datasize', type=int, required=True, help='How many data you want')
args = parser.parse_args()

# Singleton (Too slow)
# ================================================================
# # Train Dataset Process
# train_dataset = GraphQADataset(split="train", data_range=[0, 5000])
# train_dataset.process()

# # Val Dataset Process
# val_dataset = GraphQADataset(split="validation", data_range=[0, 500])
# val_dataset.process()

# Multiprocessing Parallelism
# ================================================================
# TODO(mingzhe): GPU Distribution
# [CHANGE IT] with the trade off of speed and GPU capacity
worker_num = args.worker_num
split = args.split
datasize = args.datasize

def func(args):
    dataset = GraphQADataset(split=args[0], data_range=args[1])
    dataset.process()

# Train Dataset Process
print("Assemble!")
t1 = time.time()
tasks = [[split, [i[0], i[-1]]] for i in np.array_split(range(datasize), worker_num)]
with Pool(worker_num) as p:
    outputs = p.map(func, tasks)
t2 = time.time()
print(f"Dataset takes {t2 - t1} sec")

