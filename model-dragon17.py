# -*- coding: utf-8 -*-
# @Date    : 2023-01-03 10:57:44
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent))
sys.path.append(str(directory))
import torch
from torch import nn, Tensor
from transformers import BertModel, BertTokenizer
from torchmetrics import Metric
import lightning as pl
from args import *


def binary_loss(y_true: Tensor, y_pred: Tensor):
    if y_pred.shape[1] != y_true.shape[1]:
        y_pred = y_pred[:, :-1]
    y_pred = (1.0 - y_pred) * 0.5  # map [1, -1] -> [0, 1]
    limit = torch.ones_like(y_pred) * 1e-12
    positive = torch.log(torch.maximum(y_pred, limit))
    negative = torch.log(torch.maximum(1.0 - y_pred, limit))
    loss = y_true * positive + (1 - y_true) * negative
    return -torch.sum(loss) / y_true.shape[0]


class TripletLoss:
    """
    y_true: one-hot labels [batch_size, num_candidate_pre_mention]
    y_pred: similarity scores [batch_size, num_candidate_pre_mention + 1]
    """

    def __init__(self, margin):
        self.margin = margin

    def __call__(self, y_true, y_pred):
        if y_pred.shape[1] != y_true.shape[1]:
            y_pred = y_pred[:, :-1]
        y_pred = -y_pred
        positive_val = torch.sum(y_pred * y_true, dim=-1)
        loss = 0.0
        for i in range(y_true.shape[0]):
            loss += torch.mean(torch.maximum(positive_val[i] - y_pred + self.margin, torch.tensor(0)))
        return loss / y_true.shape[0]


class TopkAccuracy(Metric):
    """
    y_true: [batch_size, num_candidate_pre_mention]
    y_pred: [batch_size, num_candidate_pre_mention + 1], last one is the answer
    """

    is_differentiable = False

    def __init__(self, top_k) -> None:
        super().__init__()
        self.top_k = top_k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape[1] != y_true.shape[1]:
            y_pred = y_pred[:, :-1]
        top_k_lowerbound = torch.tile(torch.topk(y_pred, self.top_k)[0][:, -1:], (1, y_true.shape[1]))
        top_k_mask = y_pred >= top_k_lowerbound
        self.correct += torch.sum(y_true * top_k_mask)
        self.total += y_true.shape[0]  # type: ignore

    def compute(self):
        return self.correct / self.total  # type: ignore


def bert_model():
    model = BertModel.from_pretrained("bert-base-cased")
    if not finetune_bert:
        for param in model.parameters():
            param.requires_grad = False
    return model


class MainModel(pl.LightningModule):
    metrics_topk = [1, 5, 10, 20, 50]

    def __init__(self) -> None:
        super().__init__()
        self.mention_encoder = bert_model()
        self.entity_encoder = bert_model()
        if mention_linear_after_avg:
            self.mention_linear = nn.Linear(bert_embed_dim, linear_output_dim)
        if entity_linears_after_avg:
            self.entity_linear = nn.Linear(bert_embed_dim, linear_output_dim)
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        self.metrics = nn.ModuleList([TopkAccuracy(k) for k in self.metrics_topk])
        self.loss = TripletLoss(triplet_margin)

    def forward(self, batch):
        """
        mention_dict: dict of [batch_size, max_bert_len]
        mention_begin/end: begin/end pos of mention name tokens insentence (including CLS) [batch_size]
        entity_dict: dict of [batch_size, num_entity_sentence, max_bert_len]
        entity_token_sep_idx: index of all SEP tokens [batch_size, num_entity_sentence, num_entity_pre_sentense]
        """
        mention_dict, mention_begin, mention_end, entity_dict, entity_token_sep_idx = batch
        bs = entity_token_sep_idx.shape[0]
        # [batch_size, max_bert_len, bert_embed_dim]
        mention_sentence = self.mention_encoder(**mention_dict)["last_hidden_state"]
        encoded_mention = torch.empty([bs, bert_embed_dim], device="cuda")
        for i in range(bs):
            encoded_mention[i, :] = torch.mean(mention_sentence[i, mention_begin[i] : mention_end[i]], dim=0)
        if mention_linear_after_avg:
            # [batch_size, bert_embed_dim] -> [batch_size, linear_output_dim]
            encoded_mention = self.mention_linear(encoded_mention)
        encoded_mention = torch.tile(torch.unsqueeze(encoded_mention, 1), [1, num_candidates, 1])

        entity_dict = {k: v.reshape((bs * num_entity_sentence, max_bert_len)) for k, v in entity_dict.items()}
        zipped_entity = self.entity_encoder(**entity_dict)["last_hidden_state"]
        zipped_entity = zipped_entity.reshape([bs, num_entity_sentence, max_bert_len, bert_embed_dim])
        # zipped_entity = torch.empty([batch_size, num_entity_sentence, max_bert_len, bert_embed_dim], device="cuda")
        # for i in range(num_entity_sentence):
        #     entity_i = {k: v[:, i, :] for k, v in entity_dict.items()}
        #     zipped_entity[:, i, :, :] = self.entity_encoder(entity_i)["last_hidden_state"]

        encoded_entity = torch.empty([bs, num_candidates, bert_embed_dim], device="cuda")
        num_entity_per_sentence = entity_token_sep_idx.shape[-1]
        for i in range(bs):
            for j in range(num_entity_sentence):
                last_idx = 1
                for k in range(num_entity_per_sentence):
                    entity_idx = k + j * num_entity_per_sentence
                    current_idx = entity_token_sep_idx[i, j, k]
                    if entity_idx < num_candidates:
                        entity_feature = torch.mean(zipped_entity[i, j, last_idx:current_idx, :], dim=0)
                        encoded_entity[i, entity_idx, :] = entity_feature
                    last_idx = current_idx + 1
        if entity_linears_after_avg:
            encoded_entity = self.entity_linear(encoded_entity)
        return self.similarity_function(encoded_mention, encoded_entity)

    def _forward_step(self, batch, batch_idx):
        print(f"{batch_idx}", end="\t")
        y = batch[-1]
        y_hat = self(batch[:-1])
        loss = self.loss(y, y_hat)
        print("loss", "%.5f" % float(loss), sep=":", end="\t")
        for k, metric in zip(self.metrics_topk, self.metrics):
            metric.update(y_hat, y)
            self.log(f"top-{k}", metric, on_step=True, on_epoch=True)
            print(f"top-{k}", "%.5f" % float(metric.compute()), sep=":", end="\t")
        print("")
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


def main():
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # bert = BertEncoder(False)
    # tokens = tokenizer("It was the best of times, it was the worst of times.", return_tensors="pt")
    # output = bert(tokens)
    # print(output)
    pass


if __name__ == "__main__":
    main()
