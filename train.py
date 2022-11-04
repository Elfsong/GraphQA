# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from graph_qa_dataset import GraphQADataset, SquadDataset
from torch_geometric.loader import DataListLoader
from model.graph_qa_model import GAT, HGT, GraphQA
from transformers import BertTokenizer

# # Dataset Instance
# train_dataset = GraphQADataset(split="train")
# val_dataset = GraphQADataset(split="validation")

# # Load Dataset from files
# train_dataset.load()
# val_dataset.load()

# # Construct Dataloader
# train_dataloader = DataListLoader(train_dataset, batch_size=2, shuffle=True)
# val_dataloader = DataListLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphQA().to(device)
loss_op = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

train_dataset = SquadDataset(split="train", size=1000, load=True)
val_dataset = SquadDataset(split="validation", size=10, load=True)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def train(train_dataloader):
    model.train()
    total_loss = 0
    for index, batch in tqdm(enumerate(train_dataloader)):
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
        label = batch["label"].to(device, dtype=torch.long)
        pred = model(input_ids, attention_mask)
        loss = loss_op(pred, label.reshape(-1,1).float())
        print(loss)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(total_loss / len(train_dataloader))

@torch.no_grad()
def eval(val_loader):
    model.eval()

    total_count = 0
    total_em = 0
    
    with torch.no_grad():
        for index, batch in tqdm(enumerate(val_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            preds = model(input_ids, attention_mask)
            print(preds)

            for label, pred in zip(labels, preds):
                print(label)
                print(pred)

    em_score = total_em / total_count
    print(f"EM: {em_score}")


for epoch in range(5):
    print("[+] Training...")
    train(train_dataloader)

    # print("[+] Evaluating...")
    # eval(val_dataloader)
