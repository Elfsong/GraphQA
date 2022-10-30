# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
from sklearn.metrics import f1_score
from graph_qa_dataset import GraphQADataset
from torch_geometric.loader import DataLoader
from model.graph_qa_model import GraphQAModel


train_dataset = GraphQADataset(split="train", data_size=3)
val_dataset = GraphQADataset(split="validation", data_size=3)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphQAModel().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train(train_loader):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def eval(val_loader):
    model.eval()

    ys, preds = [], []
    for data in val_loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


# for epoch in range(100):
#     loss = train(train_loader)
#     val_f1 = eval(val_loader)
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}')