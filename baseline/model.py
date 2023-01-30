# -*- coding: utf-8 -*-
# @Date    : 2023-01-03 10:57:44
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""implementation of the baseline model (https://github.com/seukgcode/MEL-GHMFC)"""

from __future__ import annotations
import torch
from torch import nn
from transformers import BertModel
from common.args import *


def bert_model():
    model = BertModel.from_pretrained("bert-base-cased")
    if not finetune_bert:
        for param in model.parameters():  # type: ignore
            param.requires_grad = False
    return model


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        return args[0]


class MaxPool(nn.Module):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, seq, *args):
        return torch.max(seq, dim=self.dim)[0]


class AvgPool(nn.Module):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, seq, *args):
        return torch.mean(seq, dim=self.dim)


class Avg(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, seq, begin, end, *args):
        return self.avg(seq, begin, end)

    @staticmethod
    def avg(seq, begin, end):
        bs = seq.shape[0]
        res = torch.empty(bs, seq.shape[-1], device="cuda")
        for i in range(bs):
            res[i] = torch.mean(seq[i, begin[i] : end[i]], dim=0)
        return res


class AvgLinear(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, seq, begin, end, *args):
        return self.linear(Avg.avg(seq, begin, end))


class MultilayerTransformer(nn.Module):
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

    def forward(self, seq, mask, *args):
        encoded = self.transformer(seq, src_key_padding_mask=(mask == 0))
        return encoded


class CrossAttention(nn.Module):
    def __init__(self, dim_a, dim_b) -> None:
        super().__init__()
        self.a2b_attention = nn.MultiheadAttention(
            dim_a,
            transformer_num_heads,
            transformer_dropout,
            kdim=dim_b,
            vdim=dim_b,
            batch_first=True,
        )
        self.a2b_ffn = nn.Linear(dim_a, dim_a)
        self.b2a_attention = nn.MultiheadAttention(
            dim_a,
            transformer_num_heads,
            transformer_dropout,
            batch_first=True,
        )
        self.b2a_ffn = nn.Linear(dim_a, dim_a)
        self.layernorms = nn.ModuleList([nn.LayerNorm(dim_a) for _ in range(4)])

    def forward(self, seq_a, mask_a, seq_b, mask_b=None, *args):
        mask_a = mask_a == 0  # [batch_size, a_seqlen]
        if mask_b is not None:
            mask_b = mask_b == 0
        else:
            mask_b = torch.zeros(seq_b.shape[:2], dtype=torch.bool, device="cuda")
        attended_b = self.a2b_attention(seq_a, seq_b, seq_b, key_padding_mask=mask_b, need_weights=False)[0]
        attended_b = self.layernorms[0](attended_b)
        attended_b = self.a2b_ffn(attended_b) + attended_b
        attended_b = self.layernorms[1](attended_b)
        attended_a = self.b2a_attention(attended_b, seq_a, seq_a, key_padding_mask=mask_a, need_weights=False)[0]
        attended_a = self.layernorms[2](attended_a)
        attended_a = self.b2a_ffn(attended_a) + attended_a
        attended_a = self.layernorms[3](attended_a)
        return attended_a


class MultimodalFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t2v_attention = CrossAttention(bert_embed_dim, resnet_embed_dim)
        self.v2t_attention = CrossAttention(resnet_embed_dim, bert_embed_dim)
        self.text_linear = nn.Linear(bert_embed_dim, mention_final_output_dim)
        self.image_linear = nn.Linear(resnet_embed_dim, mention_final_output_dim)
        self.subspace_activation = getattr(nn.functional, multimodal_subspace_activation)
        self.score_linear = nn.Linear(mention_final_output_dim * 2, 2)

    def forward(self, text_seq, text_mask, image_seq, *args):
        image_mask = torch.ones(image_seq.shape[:2], dtype=torch.bool, device="cuda")
        attended_text = torch.max(self.t2v_attention(text_seq, text_mask, image_seq, image_mask), dim=1)[0]
        attended_text = self.subspace_activation(self.text_linear(attended_text))
        attended_image = torch.max(self.v2t_attention(image_seq, image_mask, text_seq, text_mask), dim=1)[0]
        attended_image = self.subspace_activation(self.image_linear(attended_image))
        score = nn.functional.softmax(self.score_linear(torch.cat([attended_text, attended_image], dim=1)))
        # [batch_size, 1, 2] @ [batch_size, 2, output_dim]
        return torch.matmul(score.unsqueeze(1), torch.stack([attended_text, attended_image], dim=1)).squeeze(1)


class MentionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if online_bert:
            self.text_encoder = bert_model()
        # intermediate layer: further encode text feature & do multimodal fusion
        # final layer: extract final mention representations according to its position
        if mention_final_representation == "max pool":
            self.final_layer = MaxPool()
        elif mention_final_representation == "avg extract":
            self.final_layer = Avg()
        if mention_final_layer_name == "linear":
            self.intermediate_layer = Identity()
            self.final_layer = AvgLinear(bert_embed_dim, mention_final_output_dim)
        elif mention_final_layer_name == "transformer":
            self.intermediate_layer = MultilayerTransformer()
        elif mention_final_layer_name == "multimodal":
            if mention_multimodal_attention == "bi":
                self.intermediate_layer = MultimodalFusion()
                self.final_layer = Identity()
            elif mention_multimodal_attention == "text":
                self.intermediate_layer = CrossAttention(bert_embed_dim, resnet_embed_dim)
        elif mention_final_layer_name == "none":
            self.intermediate_layer = Identity()

    def forward(self, batch):
        if online_bert:
            mention_dict, mention_begin, mention_end, mention_image = batch
            # [batch_size, max_bert_len, bert_embed_dim]
            mention_sentence = self.text_encoder(**mention_dict)["last_hidden_state"]  # type: ignore
            attention_mask = mention_dict["attention_mask"]
            # clip text max_len according to max_mention_sentence_len to reduce memory usage
            mention_sentence = mention_sentence[:, :max_mention_sentence_len, :]
            attention_mask = attention_mask[:, :max_mention_sentence_len]
        else:
            mention_sentence, attention_mask, mention_begin, mention_end, mention_image = batch
        mention_feature = self.intermediate_layer(mention_sentence, attention_mask, mention_image)
        encoded_mention = self.final_layer(mention_feature, mention_begin, mention_end)
        return encoded_mention


class EntityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if online_bert:
            self.text_encoder = bert_model()
        if entity_final_layer_name == "linear":
            self.final_layer = nn.Linear(bert_embed_dim, entity_final_output_dim)
        elif entity_final_layer_name == "none":
            self.final_layer = nn.Identity()
        if entity_final_pooling == "max":
            self.final_pooling = MaxPool(dim=0)
        elif entity_final_pooling == "avg":
            self.final_pooling = AvgPool(dim=0)

    def forward(self, batch):
        if online_bert:
            bs = batch[1].shape[0]
            encoded_entity = torch.empty([bs, num_candidates, bert_embed_dim], device="cuda")
            entity_dict, entity_token_sep_idx, entity_image = batch
            if num_entity_sentence:
                # entity_dict = {k: v.reshape((bs * num_entity_sentence, max_bert_len)) for k, v in entity_dict.items()}
                # zipped_entity = self.text_encoder(**entity_dict)["last_hidden_state"]  # type: ignore
                # zipped_entity = zipped_entity.reshape([bs, num_entity_sentence, max_bert_len, bert_embed_dim])
                zipped_entity = torch.empty([bs, num_entity_sentence, max_bert_len, bert_embed_dim])
                for i in range(num_entity_sentence):
                    entity_dict_i = {k: v[:, i, :] for k, v in entity_dict.items()}
                    zipped_entity[:, i, :, :] = self.text_encoder(**entity_dict_i)["last_hidden_state"]  # type: ignore
                num_entity_per_sentence = entity_token_sep_idx.shape[-1]
                for i in range(bs):
                    for j in range(num_entity_sentence):
                        last_idx = 1
                        for k in range(num_entity_per_sentence):
                            entity_idx = k + j * num_entity_per_sentence
                            current_idx = entity_token_sep_idx[i, j, k]
                            if entity_idx < num_candidates:
                                entity_feature = self.final_pooling(zipped_entity[i, j, last_idx:current_idx, :])
                                encoded_entity[i, entity_idx, :] = entity_feature
                            last_idx = current_idx + 1
            else:
                for i in range(num_candidates):
                    entity_i = {k: v[:, i, :] for k, v in entity_dict.items()}
                    if entity_final_pooling != "bert default":
                        seq = self.text_encoder(**entity_i)["last_hidden_state"]  # type: ignore
                        for j in range(bs):
                            num_tokens = torch.sum(entity_i["attention_mask"], dim=-1)
                            encoded_entity[j, i, :] = self.final_pooling(seq[j, 1 : num_tokens[j] - 1, :])
                    else:
                        encoded_entity[:, i, :] = self.text_encoder(**entity_i)["pooler_output"]  # type: ignore
        else:
            encoded_entity, entity_image = batch
        encoded_entity = self.final_layer(encoded_entity)
        return encoded_entity


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mention_encoder = MentionEncoder()
        self.entity_encoder = EntityEncoder()
        self.similarity_function = nn.CosineSimilarity(dim=-1)

    def forward(self, batch):
        """
        mention_dict: dict of [batch_size, max_bert_len]
        mention_begin/end: begin/end pos of mention name tokens in sentence (including CLS) [batch_size]
        entity_dict: dict of [batch_size, num_entity_sentence, max_bert_len]
        entity_token_sep_idx: index of all SEP tokens [batch_size, num_entity_sentence, num_entity_pre_sentence]
        """
        mention_entity_sep = 4 if online_bert else 5
        encoded_mention = self.mention_encoder(batch[:mention_entity_sep])
        encoded_entity = self.entity_encoder(batch[mention_entity_sep:])
        encoded_mention = torch.tile(torch.unsqueeze(encoded_mention, 1), [1, num_candidates, 1])
        return self.similarity_function(encoded_mention, encoded_entity)


def main():
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # bert = BertEncoder(False)
    # tokens = tokenizer("It was the best of times, it was the worst of times.", return_tensors="pt")
    # output = bert(tokens)
    # print(output)
    pass


if __name__ == "__main__":
    main()
