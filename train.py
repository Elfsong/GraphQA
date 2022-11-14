# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     07/11/2022
# ---------------------------------------------------------------- 

import os
import torch
import evaluate
import components.hg_utils as hg_utils

from tqdm import tqdm
from collections import defaultdict
from transformers import get_scheduler
from torch.utils.data import DataLoader
from components.hg_dataset import HGDataset
from transformers import AutoTokenizer, BertForQuestionAnswering


# Device Selection
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Loading
hg_utils.logger.info(f"Loading Model...")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)

# Tokenizer loading
hg_utils.logger.info(f"Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Optimizer Loading
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Squad metric
squad_metric = evaluate.load("squad")

def train(dataloader):
    total_loss = 0
    for index, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
        start_positions = batch["start_position"].to(device, dtype=torch.long)
        end_positions = batch["end_position"].to(device, dtype=torch.long)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        current_loss = outputs.loss.detach()
        total_loss += current_loss

        if index % 100 == 0:
            print(f"Loss: {current_loss}")

    print(total_loss / len(dataloader))


if __name__ == "__main__":
    # Dataset Loading
    train_dataset = HGDataset(
        source_path=hg_utils.get_path("./data/squad_v2/raw/train-v2.0.json"), 
        target_path=hg_utils.get_path("./data/squad_v2/processed/train/"),
        tokenizer=tokenizer,
        using_cache=False
    )

    dev_dataset = HGDataset(
        source_path=hg_utils.get_path("./data/squad_v2/raw/dev-v2.0.json"), 
        target_path=hg_utils.get_path("./data/squad_v2/processed/dev/"),
        tokenizer=tokenizer,
        using_cache=False
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(3): 
        train(train_dataloader)
        model.save_pretrained(f"./exp/test_{epoch}")
        
