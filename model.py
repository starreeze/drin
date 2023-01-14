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


identity_fn = lambda *args: args[0]


class Avg(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, seq, mask, *args):
        return avg_with_mask(seq, mask)


class AvgLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(bert_embed_dim, linear_output_dim)

    def forward(self, seq, mask, *args):
        avgs = avg_with_mask(seq, mask)
        return self.linear(avgs)


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


class MultimodalFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t2v_attention = nn.MultiheadAttention(
            bert_embed_dim,
            transformer_num_heads,
            transformer_dropout,
            kdim=resnet_embed_dim,
            vdim=resnet_embed_dim,
            batch_first=True,
        )
        self.t2v_ffn = nn.Linear(bert_embed_dim, bert_embed_dim)
        self.v2t_attention = nn.MultiheadAttention(
            bert_embed_dim,
            transformer_num_heads,
            transformer_dropout,
            batch_first=True,
        )
        self.v2t_ffn = nn.Linear(bert_embed_dim, bert_embed_dim)
        self.layernorms = nn.ModuleList([nn.LayerNorm(bert_embed_dim) for _ in range(4)])

    def forward(self, text_seq, text_mask, image_seq, *args):
        text_mask = text_mask == 0  # [batch_size, text_seqlen]
        # [batch_size * num_head, text_seqlen, image_seqlen]
        # t2v_attention_mask = torch.tile(text_mask.unsqueeze(-1), [transformer_num_heads, 1, resnet_num_region])
        attended_image = self.t2v_attention(text_seq, image_seq, image_seq, need_weights=False)[0]
        attended_image = self.layernorms[0](attended_image)
        attended_image = self.t2v_ffn(attended_image) + attended_image
        attended_image = self.layernorms[1](attended_image)

        attended_text = self.v2t_attention(
            attended_image, text_seq, text_seq, key_padding_mask=text_mask, need_weights=False
        )[0]
        attended_text = self.layernorms[2](attended_text)
        attended_text = self.v2t_ffn(attended_text) + attended_text
        attended_text = self.layernorms[3](attended_text)
        return attended_text


class MentionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = bert_model()
        # for mentions, we feed token representations and padding mask into final encoder
        if mention_final_layer_name == "linear":
            self.intermidiate_layer = identity_fn
            self.final_layer = AvgLinear()
        elif mention_final_layer_name == "transformer":
            self.intermidiate_layer = MultilayerTransformer()
            self.final_layer = Avg()
        elif mention_final_layer_name == "multimodal":
            self.intermidiate_layer = MultimodalFusion()
            self.final_layer = Avg()
        elif mention_final_layer_name == "none":
            self.intermidiate_layer = identity_fn
            self.final_layer = Avg()

    def forward(self, mention_dict, mention_begin, mention_end, mention_image):
        bs = mention_begin.shape[0]
        # [batch_size, max_bert_len, bert_embed_dim]
        mention_sentence = self.text_encoder(**mention_dict)["last_hidden_state"]  # type: ignore
        # clip text max_len according to max_mention_sentence_len to reduce memory usage
        mention_sentence = mention_sentence[:, :max_mention_sentence_len, :]
        attention_mask = mention_dict["attention_mask"][:, :max_mention_sentence_len]
        mention_sentence = self.intermidiate_layer(mention_sentence, attention_mask, mention_image)
        encoded_mention = torch.zeros([bs, max_mention_name_len, bert_embed_dim], device="cuda")
        mention_pad_mask = torch.zeros([bs, max_mention_name_len], dtype=torch.bool, device="cuda")
        for i in range(bs):
            b, e = mention_begin[i], mention_end[i]
            mention_pad_mask[i, : e - b] = 1
            encoded_mention[i, : e - b, :] = mention_sentence[i, b:e]
        # [batch_size, max_token_len, bert_embed_dim] -> [batch_size, encoder_output_dim]
        encoded_mention = self.final_layer(encoded_mention, mention_pad_mask)
        return encoded_mention


class EntityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = bert_model()
        # for entities, we 'hardly' calculate mean on their tokens to obtain a vector, and then feed this into encoder
        if entity_final_layer_name == "linear":
            self.final_layer = nn.Linear(bert_embed_dim, linear_output_dim)
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
            # entity_dict = {k: v.reshape((bs * num_candidates, max_bert_len)) for k, v in entity_dict.items()}
            # encoded_entity = self.entity_base_encoder(**entity_dict)["pooler_output"]  # type: ignore
            # encoded_entity = encoded_entity.reshape([bs, num_candidates, bert_embed_dim])
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
        mention_begin/end: begin/end pos of mention name tokens insentence (including CLS) [batch_size]
        entity_dict: dict of [batch_size, num_entity_sentence, max_bert_len]
        entity_token_sep_idx: index of all SEP tokens [batch_size, num_entity_sentence, num_entity_pre_sentense]
        """
        mention_dict, mention_begin, mention_end, mention_image, entity_dict, entity_token_sep_idx, entity_image = batch
        encoded_mention = self.mention_encoder(mention_dict, mention_begin, mention_end, mention_image)
        encoded_mention = torch.tile(torch.unsqueeze(encoded_mention, 1), [1, num_candidates, 1])
        encoded_entity = self.entity_encoder(entity_dict, entity_token_sep_idx, entity_image)
        return self.similarity_function(encoded_mention, encoded_entity)

    def _forward_step(self, batch, batch_idx):
        log_str = f"{batch_idx}\t"
        y = batch[-1]
        y_hat = self(batch[:-1])
        loss = self.loss(y, y_hat)
        log_str += f"loss: {float(loss):.5f}\t"
        for k, metric in zip(self.metrics_topk, self.metrics):
            metric.update(y_hat, y)  # type: ignore
            log_str += f"top-{k}: {float(metric.compute()):.5f}\t"  # type: ignore
        if batch_idx % stdout_freq == 0:
            print(log_str)
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
