# -*- coding: utf-8 -*-
# @Date    : 2023-01-03 09:43:13
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from common.args import *
from common.utils import *
import torch, os, json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import numpy as np


def extract_mention(tokens: torch.Tensor, start, end) -> tuple[dict[str, torch.Tensor], int, int]:
    """
    extract mention name tokens into a new sentence and return with start/end pos
    start/end_pos: CLS considered (not included in range) but not SEP
    """
    input_ids = torch.zeros([max_bert_len], dtype=torch.int64)
    input_ids[0] = CLS
    input_ids[1 : end - start + 1] = tokens[start + 1 : end + 1]
    input_ids[end - start + 1] = SEP
    pad_masks = torch.zeros([max_bert_len], dtype=torch.int64)
    pad_masks[: end - start + 2] = 1
    token_type_ids = torch.zeros([max_bert_len], dtype=torch.int64)
    result_dict = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": pad_masks,
    }
    return result_dict, 1, end - start + 1


def zip_entities(tokens: list[list[int]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """zip all entities into a few sentences for quicker inference on bert"""
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
            current_len += len(sample) - 1
            entity_token_sep_idx[i, j] = current_len
        pad_masks_zipped[i, : current_len + 1] = 1
    result_dict = {
        "input_ids": input_ids_zipped,
        "token_type_ids": token_type_ids,
        "attention_mask": pad_masks_zipped,
    }
    return result_dict, entity_token_sep_idx


class MELDataset(Dataset):
    def __init__(self, type, lookup, tokenizer=None, qid2name=None, qid2attr=None, qid2idx=None) -> None:
        super().__init__()
        self.qid2name: dict[str, str] = qid2name  # type: ignore
        self.qid2attr: dict[str, str] = qid2attr  # type: ignore
        self.qid2idx: dict[str, int] = qid2idx  # type: ignore
        self.lookup = lookup
        if online_bert:
            self.tokenizer = tokenizer
            self.mention_text_raw = np.load(os.path.join(preprocess_dir, "mention-text-raw_%s.npy" % type))
            print('#', end='')
            if entity_text_type == "name" or entity_text_type == "attr":
                self.entity_text_raw = np.load(os.path.join(preprocess_dir, "entity-name-raw_%s.npy" % type))
            elif entity_text_type == "brief":
                self.entity_text_raw = np.load(
                    os.path.join(preprocess_dir, "entity-brief-raw_%s.npy" % type), mmap_mode="r"
                )
            else:
                raise ValueError("entity_text_type must be either 'name', 'brief' or 'attr'")
            self.entity_text_raw = self.entity_text_raw.reshape((-1, num_candidates_model))
            print('#', end='')
        else:  # mention feature is aligned with model data but entity is not
            self.mention_text_feature = np.load(
                os.path.join(preprocess_dir, "mention-text-feature_%s.npy" % type), mmap_mode="r"
            )
            print('#', end='')
            self.mention_text_mask = np.load(os.path.join(preprocess_dir, "mention-text-mask_%s.npy" % type))
            print('#', end='')
            if dataset_name == "wikimel":
                self.entity_qid = np.load(os.path.join(preprocess_dir, "entity-name-raw_%s.npy" % type))
                self.entity_qid = self.entity_qid.reshape((-1, num_candidates_model))
                print('#', end='')
                self.entity_text_feature = np.load(
                    os.path.join(preprocess_dir, f"entity-{entity_text_type}-feature.npy")
                )
                print('#', end='')
                self.entity_text_mask = np.load(os.path.join(preprocess_dir, f"entity-{entity_text_type}-mask.npy"))
                print('#', end='')
            elif dataset_name == "wikidiverse":
                self.entity_text_feature = np.load(
                    os.path.join(preprocess_dir, f"entity-{entity_text_type}-feature_{type}.npy")
                ).reshape((-1, num_candidates_model, bert_embed_dim))
                print('#', end='')
        self.start_position = np.load(os.path.join(preprocess_dir, "start-pos_%s.npy" % type))
        print('#', end='')
        self.end_position = np.load(os.path.join(preprocess_dir, "end-pos_%s.npy" % type))
        print('#', end='')
        self.answer = np.load(os.path.join(preprocess_dir, "answer_%s.npy" % type))
        print('#', end='')
        if mention_final_layer_name == "multimodal":
            self.mention_image: np.memmap = np.load(
                os.path.join(preprocess_dir, "mention-image-feature_%s.npy" % type), mmap_mode="r"
            )
            print('#', end='')
        if entity_final_layer_name == "multimodal":
            self.entity_image: np.memmap = np.load(
                os.path.join(preprocess_dir, "entity-image-feature_%s.npy" % type), mmap_mode="r"
            )
            print('#', end='')
        print('\ndata loading completed')

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, idx):
        start, end = self.start_position[idx], self.end_position[idx]
        mention_image = self.mention_image[idx].copy() if mention_final_layer_name == "multimodal" else 0
        entity_image = self.entity_image[idx].copy() if entity_final_layer_name == "multimodal" else 0
        answer = self.lookup[self.answer[idx]]
        if online_bert:
            mention_text = self.mention_text_raw[idx]
            if entity_text_type == "name":
                entity_text = list(map(self.qid2name.get, self.entity_text_raw[idx]))
            elif entity_text_type == "brief":
                entity_text = list(self.entity_text_raw[idx])
            elif entity_text_type == "attr":
                entity_text = [
                    (self.qid2name[qid] + ". " + self.qid2attr[qid].replace(".", ";"))[:max_entity_attr_char_len]
                    for qid in self.entity_text_raw[idx]
                ]
            mention_token: dict[str, torch.Tensor] = self.tokenizer(
                mention_text, return_tensors="pt", padding=True, truncation=True
            )  # type: ignore
            mention_token = {k: v.squeeze(0) for k, v in mention_token.items()}
            if num_entity_sentence:
                entity_token = self.tokenizer(entity_text)  # type: ignore
                entities_processed = zip_entities(entity_token["input_ids"])
            else:
                entity_token = self.tokenizer(entity_text, return_tensors="pt", padding=True, truncation=True)  # type: ignore
                entities_processed = (pad_tokens(entity_token, max_bert_len), 0)
            if pre_extract_mention:
                mention_extracted = extract_mention(mention_token["input_ids"], start, end)
                return mention_extracted + (mention_image,) + entities_processed + (entity_image, answer)
            else:
                mention_token = pad_tokens(mention_token, max_bert_len)
                return (
                    (
                        mention_token,
                        start + 1,
                        end + 1,
                        mention_image,
                    )
                    + entities_processed
                    + (entity_image, answer)
                )
        else:
            mention_feature = torch.from_numpy(self.mention_text_feature[idx].copy())
            mention_mask = torch.from_numpy(self.mention_text_mask[idx])
            if dataset_name == "wikimel":
                entity_feature = torch.empty([num_candidates_model, max_entity_attr_token_len, bert_embed_dim])
                entity_mask = torch.empty([num_candidates_model, max_entity_attr_token_len], dtype=torch.int64)
                for i in range(num_candidates_model):
                    feature_idx = self.qid2idx[self.entity_qid[idx, i]]
                    entity_feature[i] = torch.from_numpy(self.entity_text_feature[feature_idx])
                    entity_mask[i] = torch.from_numpy(self.entity_text_mask[feature_idx])
            elif dataset_name == "wikidiverse":
                entity_feature = torch.from_numpy(self.entity_text_feature[idx].copy())
                entity_mask = 0
            return (
                mention_feature,
                mention_mask,
                start + 1,
                end + 1,
                mention_image,
                entity_feature,  # type: ignore
                entity_mask,  # type: ignore
                entity_image,
                answer,
            )


def create_datasets():
    qid2name, qid2attr, qid2idx = None, None, None
    if online_bert:
        if entity_text_type == "name":
            with open(qid2entity_path, "r") as f:
                qid2name = json.load(f)
        elif entity_text_type == "attr":
            with open(qid2entity_path, "r") as f:
                qid2name = json.load(f)
            with open(qid2attr_path, "r") as f:
                qid2attr = json.load(f)
    elif dataset_name == "wikimel":
        with open(os.path.join(preprocess_dir, "qid2idx.json"), "r") as f:
            qid2idx = json.load(f)
    lookup = torch.eye(num_candidates_data, dtype=torch.int8)
    all_zero_line = torch.zeros((1, num_candidates_data), dtype=torch.int8)
    lookup = torch.concatenate([lookup, all_zero_line], dim=0)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    return [
        DataLoader(
            MELDataset("train", lookup, tokenizer, qid2name, qid2attr, qid2idx),
            batch_size,
            shuffle_train_data,
            num_workers=dataloader_workers,
        ),
        DataLoader(
            MELDataset("valid", lookup, tokenizer, qid2name, qid2attr, qid2idx),
            batch_size,
            False,
            num_workers=dataloader_workers,
        ),
        DataLoader(
            MELDataset("test", lookup, tokenizer, qid2name, qid2attr, qid2idx),
            batch_size,
            False,
            num_workers=dataloader_workers,
        ),
    ]


if __name__ == "__main__":
    pass
    # iter = create_datasets()[0]._get_iterator()
    # x = next(iter)
    # print(x)
