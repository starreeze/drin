# -*- coding: utf-8 -*-
# @Date    : 2023-01-03 09:43:13
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import sys, json
from pathlib import Path
from typing import Tuple

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent))
sys.path.append(str(directory))
from args import *
import torch, os
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import numpy as np
import lightning as pl
from model import MainModel


class MELDataset(Dataset):
    def __init__(self, type, qid2name, lookup, tokenizer) -> None:
        super().__init__()
        self.qid2name = qid2name
        self.lookup = lookup
        self.tokenizer = tokenizer
        self.mention = np.load(os.path.join(preprocess_dir, "mention-text-raw_%s.npy" % type))
        self.entity_qid = np.load(os.path.join(preprocess_dir, "entity-name-raw_%s.npy" % type)).reshape(
            (-1, num_candidates)
        )
        self.answer = np.load(os.path.join(preprocess_dir, "answer_%s.npy" % type))

    def __len__(self):
        return len(self.mention)

    def __getitem__(self, idx):
        mention_text = self.mention[idx]
        entity_text = list(map(self.qid2name.get, self.entity_qid[idx]))
        mention_token: dict[str, torch.Tensor] = self.tokenizer(
            mention_text, return_tensors="pt", padding=True, truncation=True
        ).data
        mention_token = {
            k: torch.constant_pad_nd(v.squeeze(0), [0, max_bert_len - v.shape[1]]) for k, v in mention_token.items()
        }
        entity_token: dict[str, list] = self.tokenizer(entity_text).data
        answer = self.lookup[self.answer[idx]]
        entities_zipped = self.zip_entities(entity_token["input_ids"])
        return mention_token, entities_zipped[0], entities_zipped[1], answer

    @staticmethod
    def zip_entities(tokens: list[list[int]]) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        # batch tokens with batch_size = num_entity_sentence
        total = len(tokens)
        num_entity_per_sentence = (total + num_entity_sentence - 1) // num_entity_sentence
        tokens_batched = [  # [num_entity_sentence, num_entity_per_sentence*, text_len*]
            tokens[i * num_entity_per_sentence : (i + 1) * num_entity_per_sentence] for i in range(num_entity_sentence)
        ]
        # [[CLS, entity_1, SEP, entity_2, SEP,...]] * num_entity_sentence
        input_ids_zipped = torch.zeros([num_entity_sentence, max_bert_len], dtype=torch.int64)
        input_ids_zipped[:, 0] = CLS
        # the index of all [sep] tokens, 0 as paddings (won't affect further calculation)
        entity_token_sep_idx = torch.zeros([num_entity_sentence, num_entity_per_sentence], dtype=torch.int64)
        token_type_ids = torch.zeros([num_entity_sentence, max_bert_len], dtype=torch.int64)
        pad_masks_zipped = torch.zeros([num_entity_sentence, max_bert_len], dtype=torch.int64)
        for i, sentence_entities in enumerate(tokens_batched):
            current_len = 0
            for j, sample in enumerate(sentence_entities):
                input_ids_zipped[i, current_len + 1 : current_len + len(sample)] = torch.tensor(sample[1:])
                token_type_ids[i, current_len + 1 : current_len + len(sample)] = j
                current_len += len(sample) - 1
                entity_token_sep_idx[i, j] = current_len
            pad_masks_zipped[i, : current_len + 1] = 1
        result_dict = {
            "input_ids": input_ids_zipped,
            "token_type_ids": token_type_ids,
            "attention_mask": pad_masks_zipped,
        }
        return result_dict, entity_token_sep_idx


def create_datasets():
    with open(qid2entity_answer_path, "r") as f:
        qid2name: dict[str, str] = json.load(f)
    lookup = torch.eye(num_candidates - 1, dtype=torch.int8)
    all_zero_line = torch.zeros((1, num_candidates - 1), dtype=torch.int8)
    lookup = torch.concatenate([lookup, all_zero_line], dim=0)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    return [
        DataLoader(MELDataset("train", qid2name, lookup, tokenizer), batch_size, True),
        DataLoader(MELDataset("valid", qid2name, lookup, tokenizer), batch_size, False),
        DataLoader(MELDataset("test", qid2name, lookup, tokenizer), batch_size, False),
    ]


def main():
    pl.seed_everything(seed)
    datasets = create_datasets()
    model = MainModel()
    trainer = pl.Trainer(enable_checkpointing=False, max_epochs=10, accelerator="gpu", devices=1)
    trainer.fit(model, datasets[0], datasets[1])
    trainer.test(model, datasets[2])


if __name__ == "__main__":
    main()
    # iter = create_datasets()[0]._get_iterator()
    # x = next(iter)
    # print(x)
