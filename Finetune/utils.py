from torch.utils.data import Dataset
import numpy as np
import logging
import torch
import json
import linecache
import os
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List
from transformers import BertTokenizer, BartTokenizer



def getLogger(name, file_path):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(file_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def readDataFromFile(data_file):
    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            pair = line.strip().split('\t')
            datas.append(pair)
    return datas


class TrainData(Dataset):
    def __init__(self, data_file, max_length, tokenizer, model_type='bert'):
        self.datas = readDataFromFile(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __getitem__(self, item):
        data = self.datas[item]
        query1, query2, label = data[0], data[1], int(data[2])
        
        tokenzied_dict = self.tokenizer.encode_plus(text=query1,
                                                    text_pair=query2,
                                                    max_length=self.max_length,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids  = np.array(tokenzied_dict['input_ids'])
        attention_mask = np.array(tokenzied_dict['attention_mask'])
        
        if 'roberta' in self.model_type:
            return input_ids, attention_mask, np.array([label]), query1, query2
        else:
            token_type_ids = np.array(tokenzied_dict['token_type_ids'])
            return input_ids, token_type_ids, attention_mask, np.array([label]), query1, query2
        
        

    def __len__(self):
        return len(self.datas)


# 输入是模型输出的预测分数
def accuracy(all_logits, all_labels):
    all_predict = (all_logits > 0) + 0
    results = (all_predict == all_labels)
    acc = results.sum() / len(all_predict)
    return acc


def precision(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FP = ((all_pred == 1) & (all_labels == 0)).sum()
    precision = TP / (TP + FP)
    return precision


def recall(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FN = ((all_pred == 0) & (all_labels == 1)).sum()
    recall = TP / (TP + FN)
    return recall


def f1_score(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FP = ((all_pred == 1) & (all_labels == 0)).sum()
    FN = ((all_pred == 0) & (all_labels == 1)).sum()
    # TN = ((all_pred == 0) & (all_labels == 0)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    return F1


class GenDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length=512,
        prefix="",
    ):
        super().__init__()
        self.src_file = data_dir
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"


        if isinstance(self.tokenizer, BartTokenizer):
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        elif isinstance(self.tokenizer, BertTokenizer):
            source_inputs = self.tokenizer(source_line)

        
        source_ids = source_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = 'dev' + str(index)
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])

        input_ids = []
        masks = []

        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))

        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        qids = [x['qid'] for x in batch]
        if self.prefix == '':
            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
            }
            return batch
        else:
            return None