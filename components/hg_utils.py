# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     07/11/2022
# ---------------------------------------------------------------- 

import logging
from tqdm import tqdm

# Logging Configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(module)s] - [%(funcName)s] : %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger()

# Squad Raw Dara Process
def raw_data_process(raw_data):
    data_collection = list()
    for article in tqdm(raw_data):
        title = article["title"]
        paragraphs = article["paragraphs"]
        for paragraph in paragraphs:
            context = paragraph["context"]
            qas = paragraph["qas"]
            for qa in qas:
                qid = qa["id"]
                question = qa["question"]
                is_impossible = qa["is_impossible"]
                answers = qa["answers"]
                for answer in answers:
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"]
                    # The dataset provides us with the start character of the answer in the context, and by adding the length of the answer, we can find the end character in the context.
                    instance = {
                        "title": title,
                        "qid": qid,
                        "question": question,
                        "context": context,
                        "is_improssible": is_impossible,
                        "answer_text": answer_text,
                        "answer_start": answer_start,
                        "answer_end": answer_start + len(answer_text),
                    }
                    data_collection += [instance]
    return data_collection

# Calculate answer token position
def calculate_token_position(tokenizer, data_collection):
    new_data_collection = list()
    for data in tqdm(data_collection):
        question = data["question"]
        context = data["context"]
        start_char = data["answer_start"]
        end_char = data["answer_end"]

        # We will deal with too long contexts by creating several training features from one sample of our dataset, with a sliding window between them.
        # 'truncation="only_second"' to truncate the context (which is in the second position) when the question with its context is too long.
        # 'stride=128' to set the number of overlapping tokens between two successive chunks.
        # 'max_length=384' to set the maximum length
        inputs = tokenizer(
            question,
            context,
            stride=128,
            max_length=384,
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        for i, offset in enumerate(inputs["offset_mapping"]):
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_position, end_position = 0, 0
            # Otherwise it's the start and end token positions
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_position = idx - 1

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_position = idx + 1
            
            # Get start and end positions for each sample
            data["token_start_position"] = start_position
            data["end_position"] = end_position

            # Generate truncated context for the further constituency grpah construction
            # Sequence obey the template "[CLS] {question} [SEP] {part of context} [SEP]", so 'sequence.split("[SEP]")[1]' can capture the current part of context.
            sequence = tokenizer.decode(inputs["input_ids"][i])
            data["context"] = sequence.split("[SEP]")[1]

            # Add tokenization info to the data instance
            data["input_ids"] = inputs["input_ids"][i]
            data["token_type_ids"] = inputs["token_type_ids"][i]
            data["attention_mask"] = inputs["attention_mask"][i]

            new_data_collection += [data]
    return new_data_collection