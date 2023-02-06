# -*- coding: utf-8 -*-
# @Date    : 2023-01-30 09:43:13
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from common.args import *
import torch, os, json
from torch.utils.data import DataLoader, Dataset
import numpy as np


class MELData(Dataset):
    """
    mention_text_feature, mention_text_mask, mention_start_pos, mention_end_pos, mention_image_feature,
    mention_object_feature, mention_object_score, entity_text_feature, entity_text_mask, entity_image_feature,
    entity_object_feature, entity_object_score, miet_similarity, mtei_similarity, answer
    """

    def __init__(self, type, onehot, entity_text_feature, entity_text_mask,
                 entity_image_feature, entity_object_feature, entity_object_score):
        super().__init__()
        self.onehot = onehot
        self.entity_text_feature = entity_text_feature
        self.entity_text_mask = entity_text_mask
        self.entity_image_feature = entity_image_feature
        self.entity_object_feature = entity_object_feature
        self.entity_object_score = entity_object_score
        for var_name in 'mention_text_feature, mention_text_mask, mention_start_pos, mention_end_pos, mention_image_feature, mention_object_feature, mention_object_score, miet_similarity, mtei_similarity, answer'.split(', '):
            setattr(self, var_name, np.load(os.path.join(preprocess_dir, f"{var_name}_{type}.npy")))


    def __len__(self):
        return len(self.answer)

    def __getitem__(self, idx):
        start, end = self.start_position[idx], self.end_position[idx]
        answer = self.onehot[self.answer[idx]]
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
        mention_image_raw = Image.open(os.path.join(mention_raw_image_dir, str(self.mention_id[idx])))
        entity_image_raw = [Image.open(os.path.join(entity_raw_image_dir, qid)) for qid in self.entity_text_raw[idx]]

        mention_token: dict[str, torch.Tensor] = self.bert_tokenizer(
            mention_text, return_tensors="pt", padding=True, truncation=True
        )  # type: ignore
        mention_token = {k: v.squeeze(0) for k, v in mention_token.items()}
        if num_entity_sentence:
            entity_token = self.bert_tokenizer(entity_text)  # type: ignore
            entities_processed = zip_entities(entity_token["input_ids"])
        else:
            entity_token = self.bert_tokenizer(entity_text, return_tensors="pt", padding=True, truncation=True)  # type: ignore
            entities_processed = (pad_tokens(entity_token, max_bert_len), 0)
        mention_token = pad_tokens(mention_token, max_bert_len)

        mention_image_processed = self.resnet_processor(mention_image_raw, return_tensors="pt") # TODO
        entity_image_processed = self.resnet_processor(entity_image_raw, return_tensors="pt")

        # mention image to all entity text, [n] stacked to [n,n]
        miet_similarity = self.clip_processor(, return_tensors="pt", padding=True, truncation=True)
        miet_similarity = pad_tokens(miet_similarity, max_entity_attr_token_len)
        mtei_similarity = self.clip_processor(, return_tensors="pt", padding=True, truncation=True)
        mtei_similarity = pad_tokens(mtei_similarity, max_mention_sentence_len)

        return (
            (
                mention_token,
                start + 1,
                end + 1,
                mention_image_processed,
            )
            + entities_processed
            + (entity_image_processed, miet_similarity, mtei_similarity, answer)
        )


def create_datasets():
    qid2name, qid2attr, qid2idx = None, None, None
    with open(qid2entity_answer_path, "r") as f:
        qid2name = json.load(f)
    if entity_text_type == "attr":
        with open(qid2attr_path, "r") as f:
            qid2attr = json.load(f)
    lookup = torch.eye(num_candidates - 1, dtype=torch.int8)
    all_zero_line = torch.zeros((1, num_candidates - 1), dtype=torch.int8)
    lookup = torch.concatenate([lookup, all_zero_line], dim=0)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    resnet_processor = AutoImageProcessor.from_pretrained(resnet_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    return [
        DataLoader(
            MELData("train", lookup, tokenizer, resnet_processor, clip_processor, qid2name, qid2attr, qid2idx),
            batch_size,
            shuffle_train_data,
            num_workers=dataloader_workers,
        ),
        DataLoader(
            MELData("valid", lookup, tokenizer, resnet_processor, clip_processor, qid2name, qid2attr, qid2idx),
            batch_size,
            False,
            num_workers=dataloader_workers,
        ),
        DataLoader(
            MELData("test", lookup, tokenizer, resnet_processor, clip_processor, qid2name, qid2attr, qid2idx),
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
