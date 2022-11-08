# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     07/11/2022
# ---------------------------------------------------------------- 

import os
import torch
import evaluate
from tqdm import tqdm
from collections import defaultdict
import components.hg_utils as hg_utils
from torch.utils.data import DataLoader
from components.hg_dataset import HGDataset
from transformers import AutoTokenizer, BertForQuestionAnswering


# Device Selection
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Loading
hg_utils.logger.info(f"Loading Model...")
model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2").to(device)

# Tokenizer loading
hg_utils.logger.info(f"Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

# Optimizer Loading
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Squad metric
squad_metric = evaluate.load("squad")

def train(dataloader):
    total_loss = 0
    for index, batch in tqdm(enumerate(dataloader)):
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
        start_positions = batch["start_position"].to(device, dtype=torch.long)
        end_positions = batch["end_position"].to(device, dtype=torch.long)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)

        total_loss += outputs.loss.detach()

    print(total_loss / len(dataloader))

def eval(dataloader):
    predicted_answers = defaultdict(set)
    theoretical_answers = defaultdict(list)

    count = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
        start_positions = batch["start_position"].to(device, dtype=torch.long)
        end_positions = batch["end_position"].to(device, dtype=torch.long)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)

        for qid, gt_answer, start_position in zip(batch["qid"], batch["answer_text"], batch["start_position"]):
            theoretical_answers[qid] += [(gt_answer, start_position)]

        for index, input_id in enumerate(input_ids):
            qid = batch["qid"][index]
            pred_start_position = outputs.start_logits[index].argmax()
            pred_end_position = outputs.end_logits[index].argmax()
            predict_answer_tokens = input_id[pred_start_position : pred_end_position + 1]
            predict_answer = tokenizer.decode(predict_answer_tokens)
            predicted_answers[qid].add(predict_answer)
            count += 1
    print(count)
        
    gt, pd = list(), list()

    for qid in predicted_answers:
        answer = list(predicted_answers[qid])[0]
        if "[CLS]" not in answer:
            pd += [{"id": qid, "prediction_text": answer}]
        else:
            pd += [{"id": qid, "prediction_text": ""}]
    
    for qid in theoretical_answers:
        pairs = theoretical_answers[qid]
        instance = {"id": qid, "answers":{"text": list(), "answer_start": list()}}
        for pair in pairs:
            instance["answers"]["text"] += [pair[0]]
            instance["answers"]["answer_start"] += [pair[1]]
        gt += [instance]

    output = squad_metric.compute(predictions=pd, references=gt)

    print(output)



if __name__ == "__main__":
    # Dataset Loading
    dev_dataset = HGDataset(
        source_path=hg_utils.get_path("./data/squad_v2/raw/dev-v2.0.json"), 
        target_path=hg_utils.get_path("./data/squad_v2/processed/dev/"),
        tokenizer=tokenizer,
        using_cache=False
    )

    dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=True)

    # train(dev_dataloader)

    eval(dev_dataloader)