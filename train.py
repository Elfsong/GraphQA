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

# Model Loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphQA(
    metadata=(['context'], [('context', 'connect', 'context')]),
    num_layers=2,
    num_heads=2
).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_op = torch.nn.MSELoss()

# Data loading
train_dataset = SquadDataset(split="train", size=4000, load=False, size_from=0)
val_dataset = SquadDataset(split="validation", size=1000, load=False)
train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

def train(train_dataloader):
    model.train()
    total_loss = 0
    for index, batch in tqdm(enumerate(train_dataloader)):
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        attention_mask = batch["input_attention_mask"].to(device, dtype=torch.long)
        label = batch["label"].to(device, dtype=torch.long)

        answer_embedding = batch["answer_embedding"].to(device, dtype=torch.long)
        pred = model(input_ids, attention_mask, answer_embedding)
        loss = loss_op(pred, label.reshape(-1,1).float())
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index % 500 == 0:
            print(f"Current loss: {loss}")
            print(f"[+] Evaluating at step {index}...")
            eval(val_dataloader)
        
    print(f"[-] Loss: {total_loss / len(train_dataloader)}")

@torch.no_grad()
def eval(val_dataloader):
    model.eval()

    total_count = 0
    total_em = 0
    
    with torch.no_grad():
        for index, batch in enumerate(val_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["input_attention_mask"].to(device)
            labels = batch["label"].to(device)
            answer_embedding = batch["answer_embedding"].to(device, dtype=torch.long)
            preds = model(input_ids, attention_mask, answer_embedding)

            loss = loss_op(labels, preds)
            # print(labels, preds, loss)

            for gt_label, pred in zip(labels, preds):
                pred_label = 1 if pred > 0.5 else 0
                if pred_label == gt_label:
                    total_em += 1
                total_count += 1

    em_score = total_em / (total_count + 1)
    print(f"[-] EM: {em_score}")


for epoch in range(5):
    print("[+] Training...")
    train(train_dataloader)

    print("[+] Evaluating...")
    eval(val_dataloader)
    
    print(f"============== Epoch {epoch} finished ==============")
