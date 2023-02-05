# -*- coding: utf-8 -*-
# @Date    : 2023-01-09 19:42:36
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import torch, os
from torch import Tensor
from torchmetrics import Metric
from PIL import Image


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
        self.add_state("correct", default=torch.tensor(0, device="cuda"), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, device="cuda"), dist_reduce_fx="sum")

    def update(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape[1] != y_true.shape[1]:
            y_pred = y_pred[:, :-1]
        top_k_lowerbound = torch.tile(torch.topk(y_pred, self.top_k)[0][:, -1:], (1, y_true.shape[1]))
        top_k_mask = y_pred >= top_k_lowerbound
        self.correct += torch.sum(y_true * top_k_mask)
        self.total += y_true.shape[0]  # type: ignore

    def compute(self):
        return self.correct / self.total  # type: ignore

    def reset(self):
        self.correct = torch.tensor(0, device="cuda")
        self.total = torch.tensor(0, device="cuda")


def pad_tokens(tokens: dict, target_len: int):
    return {
        k: (
            torch.constant_pad_nd(v, [0, target_len - v.shape[-1]])
            if v.dtype in [torch.int32, torch.int64, torch.uint8, torch.bool]
            else v
        )
        for k, v in tokens.items()
    }


def load_image(basename, default_image):
    for suffix in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".tif", ".TIF", ".tiff", ".TIFF"]:
        if not os.path.exists(basename + suffix):
            continue
        try:
            image = Image.open(basename + suffix)
            return image
        except Exception as e:
            print(e, end=" - ")
            break
    print(f"{basename} error")
    return Image.open(default_image)


def main():
    pass


if __name__ == "__main__":
    main()
