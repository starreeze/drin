# -*- coding: utf-8 -*-
# @Date    : 2023-03-29 9:08:47
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""re-implementation of the melhi model (https://link.springer.com/chapter/10.1007/978-3-030-73197-7_35)"""

from __future__ import annotations
import torch
from torch import nn
from common.args import *
from baselines.ghmfc import Avg

if dataset_name != "wikidiverse":
    raise NotImplementedError(
        "melhi is only implemented for wikidiverse; the result of wikimel can be found in its paper"
    )


def lstm_extract_last(packed: nn.utils.rnn.PackedSequence):
    bs = len(packed.unsorted_indices)
    res = torch.empty(bs, *packed.data.shape[1:], device=use_device)
    for i in range(bs):
        res[i] = packed.data[packed.unsorted_indices[i] - 1]
    return res


class MentionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mention_lstm = nn.LSTM(bert_embed_dim * 3, bert_embed_dim * 3)
        self.mention_final_map = nn.Linear(bert_embed_dim * 6, bert_embed_dim)

    def forward(self, mention_feature, mention_mask, start, end):
        bs = len(mention_feature)
        mention_left = [mention_feature[i, 1 : start[i]] if start[i] > 1 else torch.zeros([1, bert_embed_dim * 3], device=use_device) for i in range(bs)]
        mention_left = nn.utils.rnn.pack_sequence(mention_left, enforce_sorted=False)
        mention_left_feature = lstm_extract_last(self.mention_lstm(mention_left)[0])
        mention_len = torch.sum(mention_mask, dim=-1)
        mention_right = [mention_feature[i, end[i] : mention_len[i]] if mention_len[i] > end[i] else torch.zeros([1, bert_embed_dim * 3], device=use_device) for i in range(bs)]
        mention_right = nn.utils.rnn.pack_sequence(mention_right, enforce_sorted=False)
        mention_right_feature = lstm_extract_last(self.mention_lstm(mention_right)[0])
        return self.mention_final_map(torch.cat([mention_left_feature, mention_right_feature], dim=-1))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sim = nn.CosineSimilarity(-1)
        self.image_map_text = nn.Linear(resnet_embed_dim, bert_embed_dim)
        self.mention_encoder = MentionEncoder()
        self.entity_final_map = nn.Linear(bert_embed_dim * 2, bert_embed_dim)

    def forward(self, batch):
        (
            mention_feature,  # [, mention text len, bert embed dim]
            mention_mask,  # [, mention text len]
            start,
            end,
            mention_image,  # [, resnet regions, resnet embed dim]
            entity_feature,  # [, num candidates, bert embed dim]
            _,
            entity_image,  # [, num candidates, resnet embed dim]
        ) = batch
        mention_image = torch.mean(mention_image, dim=-2)
        mention_image_mapped = self.image_map_text(mention_image)
        entity_image_mapped = self.image_map_text(entity_image)
        sim_tmim = self.sim(mention_feature[:, 0], mention_image_mapped)
        sim_imie = self.sim(mention_image.unsqueeze(1).expand(-1, num_candidates_model, -1), entity_image)
        image_mask = (sim_tmim > thres_tmim) * (torch.sum(sim_imie > thres_imie, dim=-1) > 0)
        image_mask = image_mask.unsqueeze(-1).expand(-1, bert_embed_dim)
        mention_image_mapped = mention_image_mapped * image_mask
        entity_image_mapped = entity_image_mapped * image_mask.unsqueeze(1).expand(-1, num_candidates_model, -1)
        mention_word = Avg.avg(mention_feature, start, end)
        mention_feature = torch.cat(
            [
                mention_feature,
                mention_word.unsqueeze(1).expand(-1, max_mention_sentence_len, -1),
                mention_image_mapped.unsqueeze(1).expand(-1, max_mention_sentence_len, -1),
            ],
            dim=-1,
        )
        entity_feature = torch.cat([entity_feature, entity_image_mapped], dim=-1)
        mention_feature = self.mention_encoder(mention_feature, mention_mask, start, end)
        entity_feature = self.entity_final_map(entity_feature)
        return self.sim(mention_feature.unsqueeze(1).expand(-1, num_candidates_model, -1), entity_feature)


def main():
    pass


if __name__ == "__main__":
    main()
