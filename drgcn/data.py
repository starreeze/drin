# -*- coding: utf-8 -*-
# @Date    : 2023-01-30 09:43:13
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import *
from torch.utils.data import DataLoader, Dataset
import numpy as np


class MELData(Dataset):
    """
    mention_text_feature, mention_text_mask, mention_start_pos, mention_end_pos, mention_image_feature,
    mention_object_feature, mention_object_score, entity_text_feature, entity_text_mask, entity_image_feature,
    entity_object_feature, entity_object_score, miet_similarity, mtei_similarity, answer
    """

    def __init__(self, inputs):
        super().__init__()
        self.onehot = inputs[0]
        self.entity_text_feature = inputs[1]
        self.entity_image_feature = inputs[3]
        self.entity_object_feature = inputs[4]
        self.entity_object_score = inputs[5]
        type = inputs[6]
        if dataset_name == "wikidiverse":
            self.entity_text_feature = self.entity_text_feature.reshape((-1, num_candidates_model, bert_embed_dim))
            self.entity_image_feature = self.entity_image_feature.reshape((-1, num_candidates_model, resnet_embed_dim))
            self.entity_object_feature = self.entity_object_feature.reshape(
                (-1, num_candidates_model, object_topk["entity"], resnet_embed_dim)
            )
            self.entity_object_score = self.entity_object_score.reshape(
                (-1, num_candidates_model, object_topk["entity"])
            )
        elif dataset_name == "wikimel":
            self.entity_text_mask = inputs[2]
            with open(os.path.join(preprocess_dir, "qid2idx.json"), "r") as f:
                self.qid2idx = json.load(f)
            self.entity_qid = np.load(os.path.join(preprocess_dir, f"entity-name-raw_{type}.npy"))
            self.entity_qid = self.entity_qid.reshape((-1, num_candidates_model))
        self.mention_text_feature = np.load(
            os.path.join(preprocess_dir, f"mention-text-feature_{type}.npy"), mmap_mode=mention_mmap
        )
        self.mention_text_mask = np.load(os.path.join(preprocess_dir, f"mention-text-mask_{type}.npy"))
        self.mention_start_pos = np.load(os.path.join(preprocess_dir, f"start-pos_{type}.npy"))
        self.mention_end_pos = np.load(os.path.join(preprocess_dir, f"end-pos_{type}.npy"))
        self.mention_image_feature = np.load(
            os.path.join(preprocess_dir, f"mention-image-feature_{type}.npy"), mmap_mode=mention_mmap
        )
        self.mention_object_feature = np.load(
            os.path.join(preprocess_dir, f"mention-object-feature_{type}.npy"), mmap_mode=mention_mmap
        )
        self.mention_object_score = np.load(os.path.join(preprocess_dir, f"mention-object-score_{type}.npy"))
        self.miet_similarity = np.load(os.path.join(preprocess_dir, f"similarity-miet_{type}.npy"))
        self.mtei_similarity = np.load(os.path.join(preprocess_dir, f"similarity-eimt_{type}.npy"))
        self.answer = np.load(os.path.join(preprocess_dir, f"answer_{type}.npy"))
        assert (
            len(self.mention_text_feature)
            == len(self.mention_start_pos)
            == len(self.mention_image_feature)
            == len(self.mention_object_feature)
            == len(self.miet_similarity)
            == len(self.answer)
        )

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, idx):
        entity_text_mask = 0
        if dataset_name == "wikimel":
            entity_idx = list(map(self.qid2idx.get, self.entity_qid[idx]))
            entity_text_feature = self._torch_tensor(self.entity_text_feature[entity_idx], entity_mmap)
            entity_text_mask = self._torch_tensor(self.entity_text_mask[entity_idx])
            entity_image_feature = self._torch_tensor(self.entity_image_feature[entity_idx], entity_mmap)
            entity_object_feature = self._torch_tensor(self.entity_object_feature[entity_idx], entity_mmap)
            entity_object_score = self._torch_tensor(self.entity_object_score[entity_idx])
        elif dataset_name == "wikidiverse":
            entity_text_feature = self._torch_tensor(self.entity_text_feature[idx], entity_mmap)
            entity_image_feature = self._torch_tensor(self.entity_image_feature[idx], entity_mmap)
            entity_object_feature = self._torch_tensor(self.entity_object_feature[idx], entity_mmap)
            entity_object_score = self._torch_tensor(self.entity_object_score[idx])
        mention_text_feature = self._torch_tensor(self.mention_text_feature[idx], mention_mmap)
        mention_text_mask = self._torch_tensor(self.mention_text_mask[idx])
        mention_start_pos = self._torch_tensor(self.mention_start_pos[idx])
        mention_end_pos = self._torch_tensor(self.mention_end_pos[idx])
        mention_image_feature = self._torch_tensor(self.mention_image_feature[idx], mention_mmap)
        mention_object_feature = self._torch_tensor(self.mention_object_feature[idx], mention_mmap)
        mention_object_score = self._torch_tensor(self.mention_object_score[idx])
        miet_similarity = self._torch_tensor(self.miet_similarity[idx])
        mtei_similarity = self._torch_tensor(self.mtei_similarity[idx])

        answer = self._torch_tensor(self.onehot[self.answer[idx]])
        return (
            mention_text_feature,
            mention_text_mask,
            mention_start_pos + 1,
            mention_end_pos + 1,
            mention_image_feature,
            mention_object_feature,
            mention_object_score,
            entity_text_feature,  # type: ignore
            entity_text_mask,
            entity_image_feature,  # type: ignore
            entity_object_feature,  # type: ignore
            entity_object_score,  # type: ignore
            miet_similarity,
            mtei_similarity,
            answer,
        )

    @staticmethod
    def _torch_tensor(x, mmap=None):
        if mmap == "r":
            x = x.copy()
        return torch.as_tensor(x)

    @staticmethod
    def get_loader(
        type,
        onehot,
        entity_text_feature,
        entity_text_mask,
        entity_image_feature,
        entity_object_feature,
        entity_object_score,
    ):
        dataset = MELData(
            (
                onehot,
                entity_text_feature,
                entity_text_mask,
                entity_image_feature,
                entity_object_feature,
                entity_object_score,
                type,
            )
        )
        return DataLoader(dataset, batch_size, type == "train" and shuffle_train_data, num_workers=dataloader_workers)


def create_datasets():
    onehot = np.eye(num_candidates_model - 1, dtype=np.uint8)
    all_zero_line = np.zeros((1, num_candidates_model - 1), dtype=np.uint8)
    onehot = np.concatenate([onehot, all_zero_line], 0)
    entity_text_feature = np.load(os.path.join(preprocess_dir, "entity-attr-feature.npy"), mmap_mode=entity_mmap)
    entity_text_mask = np.load(os.path.join(preprocess_dir, "entity-attr-mask.npy"))
    entity_image_feature = np.load(os.path.join(preprocess_dir, "entity-image-feature_all.npy"), mmap_mode=entity_mmap)
    entity_object_feature = np.load(
        os.path.join(preprocess_dir, "entity-object-feature_all.npy"), mmap_mode=entity_mmap
    )
    entity_object_score = np.load(os.path.join(preprocess_dir, "entity-object-score_all.npy"))
    return [
        MELData.get_loader(
            type,
            onehot,
            entity_text_feature,
            entity_text_mask,
            entity_image_feature,
            entity_object_feature,
            entity_object_score,
        )
        for type in ["train", "valid", "test"]
    ]


if __name__ == "__main__":
    # pass
    iter = create_datasets()[0]._get_iterator()
    x = next(iter)
    print(x)
