# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
from sklearn.metrics import f1_score
from graph_qa_dataset import GraphQADataset
from torch_geometric.loader import DataListLoader
from model.graph_qa_model import GAT, HGT


train_dataset = GraphQADataset(split="train", data_size=2)
val_dataset = GraphQADataset(split="validation", data_size=1)


train_dataloader = DataListLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataListLoader(val_dataset, batch_size=1, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HGT(hidden_channels=64, out_channels=2, num_heads=4, num_layers=2, metadata=train_dataset.metadata)
model = model.to(device)

loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train(dataloader):
    model.train()
    pass


@torch.no_grad()
def eval(val_loader):
    model.eval()
    pass


for epoch in range(10):
    loss = train(train_dataloader)
    # val_f1 = eval(val_dataloader)
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}')