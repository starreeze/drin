# -*- coding: utf-8 -*-
# @Date    : 2023-01-28 19:39:31
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""implementation of the DRGCN model (ours)"""

from __future__ import annotations
import torch
from torch import nn
from common.args import *
from baseline.model import MentionEncoder, EntityEncoder, bert_model


class VertexEncoder(nn.Module):
    '''
    Encode the 4 types of vertex, along with text features to be used in EdgeEncoder
    '''
    def __init__(self):
        super().__init__()
        self.bert = bert_model()
        self.mention_text_encoder = MentionEncoder(inline_bert=False)
        self.mention_final_repr = MentionEncoder.get_mention_final_repr_fn()
        # self.entity_text_encoder = EntityEncoder(inline_bert=False)
        self.entity_final_pooling = EntityEncoder.get_entity_final_pooling_fn()
        self.mention_image_encoder = Resnet()
        self.entity_image_encoder = Resnet()

    def forward(self, mention_token_dict, mention_start_pos, mention_end_pos, mention_image, entity_token_dict, entity_sep_idx, entity_image):
        bs = mention_start_pos.shape[0]

        mention_sentence = self.bert(**mention_token_dict)["last_hidden_state"]  # type: ignore
        mention_mask = mention_token_dict["attention_mask"]
        mention_sentence = mention_sentence[:, :max_mention_sentence_len, :]
        mention_mask = mention_mask[:, :max_mention_sentence_len]
        mention_text = self.mention_text_encoder((mention_sentence, mention_mask, mention_start_pos, mention_end_pos, None))
        mention_text_feature = self.mention_final_repr()

        zipped_entity = torch.empty([bs, num_entity_sentence, max_bert_len, bert_embed_dim], device='cuda')
        for i in range(num_entity_sentence):
            entity_dict_i = {k: v[:, i, :] for k, v in entity_token_dict.items()}
            zipped_entity[:, i, :, :] = self.text_encoder(**entity_dict_i)["last_hidden_state"]  # type: ignore
        entity_text = EntityEncoder.unzip_entities(zipped_entity, entity_sep_idx, self.entity_final_pooling)


class EdgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forword(self, )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vertex_encoder = VertexEncoder()

    def forward(self, batch):
        mention_token_dict, mention_start_pos, mention_end_pos, mention_image, entity_token_dict, entity_sep_idx, entity_image, miet_similarity, mtei_similarity, answer = batch



def main():
    pass


if __name__ == "__main__":
    main()
