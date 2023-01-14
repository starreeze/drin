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


def avg_with_mask(seq, mask):
    mask_expanded = torch.tile(mask.unsqueeze(-1), [1, 1, bert_embed_dim])
    num_valid_tokens = torch.tile(torch.sum(mask, dim=1).unsqueeze(-1), [1, bert_embed_dim])
    return torch.sum(seq * mask_expanded, dim=1) / num_valid_tokens


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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, seq, *args):
        return torch.max(seq, dim=1)[0]


class Avg(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, seq, begin, end, *args):
        return self.avg(seq, begin, end)

    @staticmethod
    def avg(seq, begin, end):
        bs = seq.shape[0]
        res = torch.empty(bs, seq.shape[-1])
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

    def forward(self, seq_a, mask_a, seq_b, mask_b, *args):
        mask_a = mask_a == 0  # [batch_size, a_seqlen]
        mask_b = mask_b == 0
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
            self.intermediate_layer = MultimodalFusion()
            self.final_layer = Identity()
        elif mention_final_layer_name == "none":
            self.intermediate_layer = Identity()

    def forward(self, mention_dict, mention_begin, mention_end, mention_image):
        # [batch_size, max_bert_len, bert_embed_dim]
        mention_sentence = self.text_encoder(**mention_dict)["last_hidden_state"]  # type: ignore
        # clip text max_len according to max_mention_sentence_len to reduce memory usage
        mention_sentence = mention_sentence[:, :max_mention_sentence_len, :]
        attention_mask = mention_dict["attention_mask"][:, :max_mention_sentence_len]
        mention_feature = self.intermediate_layer(mention_sentence, attention_mask, mention_image)
        encoded_mention = self.final_layer(mention_feature, mention_begin, mention_end)
        return encoded_mention


class EntityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = bert_model()
        if entity_final_layer_name == "linear":
            self.final_layer = nn.Linear(bert_embed_dim, entity_final_output_dim)
        elif entity_final_layer_name == "none":
            self.final_layer = nn.Identity()

    def forward(self, entity_dict, entity_token_sep_idx, entity_image):
        bs = entity_token_sep_idx.shape[0]
        encoded_entity = torch.empty([bs, num_candidates, bert_embed_dim], device="cuda")
        if entity_text_type == "name":
            if num_entity_sentence:
                entity_dict = {k: v.reshape((bs * num_entity_sentence, max_bert_len)) for k, v in entity_dict.items()}
                zipped_entity = self.text_encoder(**entity_dict)["last_hidden_state"]  # type: ignore
                zipped_entity = zipped_entity.reshape([bs, num_entity_sentence, max_bert_len, bert_embed_dim])
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
            else:
                for i in range(num_candidates):
                    entity_i = {k: v[:, i, :] for k, v in entity_dict.items()}
                    seq = self.text_encoder(**entity_i)["last_hidden_state"]  # type: ignore
                    for j in range(bs):
                        num_tokens = torch.sum(entity_i["attention_mask"], dim=-1)
                        encoded_entity[j, i, :] = torch.mean(seq[j, 1 : num_tokens[j] - 1, :], dim=0)
        else:
            for i in range(num_candidates):
                entity_i = {k: v[:, i, :] for k, v in entity_dict.items()}
                encoded_entity[:, i, :] = self.text_encoder(**entity_i)["pooler_output"]  # type: ignore
        encoded_entity = self.final_layer(encoded_entity)  # TODO: image
        return encoded_entity


class MainModel(pl.LightningModule):
    metrics_topk = [1, 5, 10, 20, 50]

    def __init__(self) -> None:
        super().__init__()
        self.mention_encoder = MentionEncoder()
        self.entity_encoder = EntityEncoder()
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        self.metrics = nn.ModuleList([TopkAccuracy(k) for k in self.metrics_topk])
        self.loss = TripletLoss(triplet_margin)

    def forward(self, batch):
        """
        mention_dict: dict of [batch_size, max_bert_len]
        mention_begin/end: begin/end pos of mention name tokens in sentence (including CLS) [batch_size]
        entity_dict: dict of [batch_size, num_entity_sentence, max_bert_len]
        entity_token_sep_idx: index of all SEP tokens [batch_size, num_entity_sentence, num_entity_pre_sentence]
        """
        mention_dict, mention_begin, mention_end, mention_image, entity_dict, entity_token_sep_idx, entity_image = batch
        encoded_mention = self.mention_encoder(mention_dict, mention_begin, mention_end, mention_image)
        encoded_mention = torch.tile(torch.unsqueeze(encoded_mention, 1), [1, num_candidates, 1])
        encoded_entity = self.entity_encoder(entity_dict, entity_token_sep_idx, entity_image)
        return self.similarity_function(encoded_mention, encoded_entity)

    def _forward_step(self, batch, batch_idx):
        log_str = f" {batch_idx}\t"
        y = batch[-1]
        y_hat = self(batch[:-1])
        loss = self.loss(y, y_hat)
        log_str += f"loss: {float(loss):.5f}\t"
        for k, metric in zip(self.metrics_topk, self.metrics):
            metric.update(y_hat, y)  # type: ignore
            log_str += f"top-{k}: {float(metric.compute()):.5f}\t"  # type: ignore
        if batch_idx % stdout_freq == 0:
            print(log_str, end="\r")
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
