# -*- coding: utf-8 -*-
# @Date    : 2023-01-03 10:57:44
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import sys
from pathlib import Path

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent))
sys.path.append(str(directory))
import torch
from torch import nn, Tensor
from transformers import BertModel, BertTokenizer
import lightning as pl
from args import *
from loss_metric import *


def bert_model():
    model = BertModel.from_pretrained("bert-base-cased")
    if not finetune_bert:
        for param in model.parameters():  # type: ignore
            param.requires_grad = False
    return model


class TransformerAvg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                bert_embed_dim,
                transformer_num_heads,
                transformer_ffn_hidden_size,
                transformer_dropout,
                transformer_ffn_activation,
                batch_first=True,
            ),
            transformer_num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, seq, mask):
        encoded = self.transformer(seq, src_key_padding_mask=~mask)
        # mean value for each sample
        mask_expanded = torch.tile(mask.unsqueeze(-1), [1, 1, bert_embed_dim])
        num_valid_tokens = torch.tile(torch.sum(mask, dim=1).unsqueeze(-1), [1, bert_embed_dim])
        return torch.sum(encoded * mask_expanded, dim=1) / num_valid_tokens


class AvgLinear(nn.Module):
    def __init__(self) -> None:
        # super().__init__()
        # self.linear = nn.Linear(bert_embed_dim, linear_output_dim)
        raise NotImplementedError()


class AvgNone(nn.Module):
    def __init__(self) -> None:
        # super().__init__()
        raise NotImplementedError()


class MainModel(pl.LightningModule):
    metrics_topk = [1, 5, 10, 20, 50]

    def __init__(self) -> None:
        super().__init__()
        self.mention_base_encoder = bert_model()
        self.entity_base_encoder = bert_model()

        # for mentions, we feed token representations and padding mask into final encoder
        if mention_final_layer_name == "linear":
            self.mention_final_encoder = AvgLinear()
        elif mention_final_layer_name == "transformer":
            self.mention_final_encoder = TransformerAvg()
        elif mention_final_layer_name == "none":
            self.mention_final_encoder = AvgNone()

        # for entities, we 'hardly' calculate mean on their tokens to obtain a vector, and then feed this into encoder
        if entity_final_layer_name == "linear":
            self.entity_final_encoder = nn.Linear(bert_embed_dim, linear_output_dim)
        elif entity_final_layer_name == "none":
            self.entity_final_encoder = lambda x: x

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
        mention_sentence = self.mention_base_encoder(**mention_dict)["last_hidden_state"]  # type: ignore
        encoded_mention = torch.zeros([bs, max_token_len, bert_embed_dim], device="cuda")
        mention_pad_mask = torch.zeros([bs, max_token_len], dtype=torch.bool, device="cuda")
        for i in range(bs):
            b, e = mention_begin[i], mention_end[i]
            mention_pad_mask[i, : e - b] = 1
            encoded_mention[i, : e - b, :] = mention_sentence[i, b:e]
        # [batch_size, max_token_len, bert_embed_dim] -> [batch_size, encoder_output_dim]
        encoded_mention = self.mention_final_encoder(encoded_mention, mention_pad_mask)
        encoded_mention = torch.tile(torch.unsqueeze(encoded_mention, 1), [1, num_candidates, 1])

        entity_dict = {k: v.reshape((bs * num_entity_sentence, max_bert_len)) for k, v in entity_dict.items()}
        zipped_entity = self.entity_base_encoder(**entity_dict)["last_hidden_state"]  # type: ignore
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
        encoded_entity = self.entity_final_encoder(encoded_entity)
        return self.similarity_function(encoded_mention, encoded_entity)

    def _forward_step(self, batch, batch_idx):
        print(f"{batch_idx}", end="\t")
        y = batch[-1]
        y_hat = self(batch[:-1])
        loss = self.loss(y, y_hat)
        print("loss", "%.5f" % float(loss), sep=":", end="\t")
        for k, metric in zip(self.metrics_topk, self.metrics):
            metric.update(y_hat, y)  # type: ignore
            self.log(f"top-{k}", metric, on_step=True, on_epoch=True)  # type: ignore
            print(f"top-{k}", "%.5f" % float(metric.compute()), sep=":", end="\t")  # type: ignore
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
