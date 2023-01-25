# -*- coding: utf-8 -*-
# @Date    : 2023-01-03 09:43:13
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from args import *
import torch, os, json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import numpy as np
import lightning as pl
from model import MainModel


class MELDataset(Dataset):
    def __init__(self, type, lookup, tokenizer, *, qid2name=None, qid2attr=None) -> None:
        super().__init__()
        self.qid2name: dict[str, str] = qid2name  # type: ignore
        self.qid2attr: dict[str, str] = qid2attr  # type: ignore
        self.lookup = lookup
        self.tokenizer = tokenizer
        self.mention_text = np.load(os.path.join(text_preprocess_dir, "mention-text-raw_%s.npy" % type))
        if entity_text_type == "name" or entity_text_type == "attr":
            self.entity_text = np.load(os.path.join(text_preprocess_dir, "entity-name-raw_%s.npy" % type))
        elif entity_text_type == "brief":
            self.entity_text = np.load(
                os.path.join(text_preprocess_dir, "entity-brief-raw_%s.npy" % type), mmap_mode="r"
            )
        else:
            raise ValueError("entity_text_type must be either 'name', 'brief' or 'attr'")
        self.entity_text = self.entity_text.reshape((-1, num_candidates))
        self.start_position = np.load(os.path.join(text_preprocess_dir, "start-pos_%s.npy" % type))
        self.end_position = np.load(os.path.join(text_preprocess_dir, "end-pos_%s.npy" % type))
        self.answer = np.load(os.path.join(text_preprocess_dir, "answer_%s.npy" % type))
        if mention_final_layer_name == "multimodal":
            self.mention_image: np.memmap = np.load(
                os.path.join(image_preprocess_dir, "mention-image_%s.npy" % type), mmap_mode="r"
            )
        if entity_final_layer_name == "multimodal":
            self.entity_image: np.memmap = np.load(
                os.path.join(image_preprocess_dir, "entity-image_%s.npy" % type), mmap_mode="r"
            )

    def __len__(self):
        return len(self.mention_text)

    def __getitem__(self, idx):
        mention_text = self.mention_text[idx]
        start, end = self.start_position[idx], self.end_position[idx]
        if entity_text_type == "name":
            entity_text = list(map(self.qid2name.get, self.entity_text[idx]))
        elif entity_text_type == "brief":
            entity_text = list(self.entity_text[idx])
        elif entity_text_type == "attr":
            entity_text = [
                (self.qid2name[qid] + ". " + self.qid2attr[qid].replace(".", ";"))[:max_entity_attr_len]
                for qid in self.entity_text[idx]
            ]
        mention_token: dict[str, torch.Tensor] = self.tokenizer(
            mention_text, return_tensors="pt", padding=True, truncation=True
        )
        mention_token = {k: v.squeeze(0) for k, v in mention_token.items()}
        answer = self.lookup[self.answer[idx]]
        if num_entity_sentence:
            entity_token = self.tokenizer(entity_text)  # type: ignore
            entities_processed = self.zip_entities(entity_token["input_ids"])
        else:
            entity_token = self.tokenizer(entity_text, return_tensors="pt", padding=True, truncation=True)  # type: ignore
            entity_token = {
                k: torch.constant_pad_nd(v, [0, max_bert_len - v.shape[-1]]) for k, v in entity_token.items()
            }
            entities_processed = (entity_token, 0)
        mention_image = self.mention_image[idx].copy() if mention_final_layer_name == "multimodal" else 0
        entity_image = self.entity_image[idx].copy() if entity_final_layer_name == "multimodal" else 0
        if pre_extract_mention:
            mention_extracted = self.extract_mention(mention_token["input_ids"], start, end)
            return mention_extracted + (mention_image,) + entities_processed + (entity_image, answer)
        else:
            mention_token = {
                k: torch.constant_pad_nd(v, [0, max_bert_len - v.shape[-1]]) for k, v in mention_token.items()
            }
            return (mention_token, start + 1, end + 1) + (mention_image,) + entities_processed + (entity_image, answer)

    @staticmethod
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

    @staticmethod
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


def create_datasets():
    qid2name, qid2attr = None, None
    if entity_text_type == "name":
        with open(qid2entity_answer_path, "r") as f:
            qid2name = json.load(f)
    elif entity_text_type == "attr":
        with open(qid2entity_answer_path, "r") as f:
            qid2name = json.load(f)
        with open(qid2attr_path, "r") as f:
            qid2attr = json.load(f)
    lookup = torch.eye(num_candidates - 1, dtype=torch.int8)
    all_zero_line = torch.zeros((1, num_candidates - 1), dtype=torch.int8)
    lookup = torch.concatenate([lookup, all_zero_line], dim=0)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    return [
        DataLoader(
            MELDataset("train", lookup, tokenizer, qid2name=qid2name, qid2attr=qid2attr),
            batch_size,
            True,
            num_workers=dataloader_workers,
        ),
        DataLoader(
            MELDataset("valid", lookup, tokenizer, qid2name=qid2name, qid2attr=qid2attr),
            batch_size,
            False,
            num_workers=dataloader_workers,
        ),
        DataLoader(
            MELDataset("test", lookup, tokenizer, qid2name=qid2name, qid2attr=qid2attr),
            batch_size,
            False,
            num_workers=dataloader_workers,
        ),
    ]


class EpochCallback(pl.Callback):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        status_str = f"\n\n========== Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        status_str += f" - training with {trainer.num_training_batches} steps =========="
        print(status_str)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print("")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        status_str = f"\n\n========== Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        status_str += f" - validating with {trainer.num_val_batches} steps =========="
        print(status_str)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print("")


def main():
    pl.seed_everything(seed)
    datasets = create_datasets()
    model = MainModel()
    trainer = pl.Trainer(
        enable_checkpointing=False,
        max_epochs=epoch,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        callbacks=EpochCallback(),
    )
    trainer.fit(model, datasets[0], datasets[1])
    trainer.test(model, datasets[2])


if __name__ == "__main__":
    main()
    # iter = create_datasets()[0]._get_iterator()
    # x = next(iter)
    # print(x)
