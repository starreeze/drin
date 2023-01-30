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
    def __init__(self):
        super().__init__()
        self.mention_text_encoder = MentionEncoder()
        self.entity_text_encoder = EntityEncoder()

    def forward(self, batch):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
