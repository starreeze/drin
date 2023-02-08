# -*- coding: utf-8 -*-
# @Date    : 2023-01-28 19:39:31
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""implementation of the DRGCN model (ours)"""

from __future__ import annotations
import torch
from torch import nn
from common.args import *
from baseline.model import MentionEncoder, EntityEncoder


class VertexEncoder(nn.Module):
    """
    Encode the 4 types of vertex:
    text vertex is controlled by the args same as baseline; for image vertex we just apply avg pooling
    """

    def __init__(self):
        super().__init__()
        self.mention_text_encoder = MentionEncoder(inline_bert=False)
        self.entity_text_encoder = EntityEncoder(inline_bert=False)

    def forward(
        self,
        mention_text_feature,
        mention_text_mask,
        mention_start_pos,
        mention_end_pos,
        mention_image_feature,
        entity_text_feature,
        entity_text_mask,
        entity_image_feature,
    ):
        encoded_mention_text = self.mention_text_encoder(
            (mention_text_feature, mention_text_mask, mention_start_pos, mention_end_pos, None)
        )
        encoded_entity_text = self.entity_text_encoder((entity_text_feature, entity_text_mask, None))
        encoded_mention_image = torch.mean(mention_image_feature, dim=-2)
        encoded_entity_image = torch.mean(entity_image_feature, dim=-2)
        return [encoded_mention_text, encoded_mention_image, encoded_entity_text, encoded_entity_image]


class EdgeEncoder(nn.Module):
    """
    Encode the 2 types of edge (of the same modality): mtet, miei
    miei is obtained by calculating the pair-wise similarity and then apply weighted average
    """

    def __init__(self):
        super().__init__()
        self.similarity_fn = nn.CosineSimilarity(dim=-1)
        self.mention_text_encoder = MentionEncoder.get_mention_final_repr_fn()

    def forward(
        self,
        mention_text_feature,
        mention_start_pos,
        mention_end_pos,
        mention_object_feature,
        mention_object_score,
        entity_text_feature,
        entity_object_feature,
        entity_object_score,
    ):
        mention_text_encoded = self.mention_text_encoder(mention_text_feature, mention_start_pos, mention_end_pos)
        mention_text_encoded = mention_text_encoded.unsqueeze(1).expand(-1, num_candidates, -1)
        mtet = self.similarity_fn(mention_text_encoded, entity_text_feature[:, :, 0])  # entity CLS

        mention_object_feature = torch.mean(mention_object_feature, dim=-2)
        mention_object_feature = mention_object_feature.unsqueeze(1).expand(-1, num_candidates, -1, -1)
        mention_object_score = mention_object_score.unsqueeze(1).expand(-1, num_candidates, -1)
        entity_object_feature = torch.mean(entity_object_feature, dim=-2)
        similarity = torch.zeros(mention_object_feature.shape[0], num_candidates, device="cuda")
        scores = torch.zeros(mention_object_feature.shape[0], num_candidates, device="cuda")
        for i in range(mention_object_feature.shape[2]):
            for j in range(entity_object_feature.shape[2]):
                sim = self.similarity_fn(mention_object_feature[:, :, i], entity_object_feature[:, :, j])
                score = mention_object_score[:, :, i] * entity_object_score[:, :, j]
                similarity += sim * score
                scores += score
        miei = similarity / scores

        return mtet, miei


class GCNLayer(nn.Module):
    """
    vertexes: mention[batch_size, gcn_embed_dim]; entity[batch_size, num_candidates, gcn_embed_dim]
    edges: [batch_size, num_candidates, gcn_embed_dim]
    output same shape as input
    """
    # [u --- [e=N(u) --- v=N(e)]]
    vertex_graph = [[[0,2],[1,3]],[[2,2],[3,3]],[[0,0],[2,1]],[[1,0],[3,1]]]
    # [u=N(e) --- e --- v=N(e)]
    edge_graph = [[0,2],[0,3],[1,2],[1,3]]

    def __init__(self):
        super().__init__()

    def forward(self, vertexes: list[torch.Tensor], edges: list[torch.Tensor]):
        new_vertexes, new_edges = [], []
        for u, neighbors in zip(vertexes, self.vertex_graph):
            new_u = torch.zeros_like(u)
            for ei, vi in neighbors:
                new_u = new_u + self.convolute_vertex(edges[ei], vertexes[vi])
            # TODO: linear and activation, then add u
            new_vertexes.append(new_u)
        for e, (ui, vi) in zip(edges, self.edge_graph):
            new_e = self.convolute_edge(vertexes[ui], vertexes[vi])
            # TODO: linear and activation, then add e
            new_edges.append(new_e)
        return new_vertexes, new_edges
    
    def convolute_vertex(self, edge, neighbor):
        pass

    def convolute_edge(self, u, v):
        pass


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vertex_encoder = VertexEncoder()
        self.edge_encoder = EdgeEncoder()
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(num_gcn_layers)])
        self.similarity_fn = nn.CosineSimilarity(dim=-1)

    def forward(self, batch):
        (
            mention_text_feature,
            mention_text_mask,
            mention_start_pos,
            mention_end_pos,
            mention_image_feature,
            mention_object_feature,
            mention_object_score,
            entity_text_feature,
            entity_text_mask,
            entity_image_feature,
            entity_object_feature,
            entity_object_score,
            miet_similarity,
            mtei_similarity,
        ) = batch
        vertexes = self.vertex_encoder(
            mention_text_feature,
            mention_text_mask,
            mention_start_pos,
            mention_end_pos,
            mention_image_feature,
            entity_text_feature,
            entity_text_mask,
            entity_image_feature,
        )  # mt, mi, et, ei
        edges = self.edge_encoder(
            mention_text_feature,
            mention_start_pos,
            mention_end_pos,
            mention_object_feature,
            mention_object_score,
            entity_text_feature,
            entity_object_feature,
            entity_object_score,
        )  # tt, ti, it, ii
        edges = [
            e.unsqueeze(-1).expand(-1, -1, gcn_embed_dim)
            for e in (edges[0], mtei_similarity, miet_similarity, edges[1])
        ]
        for gcn_layer in self.gcn_layers:
            vertexes, edges = gcn_layer(vertexes, edges)
        mention, entity = vertexes[0], vertexes[2]
        mention = mention.unsqueeze(1).expand(-1, num_candidates, -1)
        return self.similarity_fn(mention, entity)


if __name__ == "__main__":
    pass
