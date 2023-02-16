# -*- coding: utf-8 -*-
# @Date    : 2023-01-09 19:42:36
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import torch, os
from torch import Tensor
from torchmetrics import Metric
from PIL import Image
from common.args import *
from ctypes import c_uint8
import numpy as np


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
    for suffix in ["", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".tif", ".TIF", ".tiff", ".TIFF"]:
        if not os.path.exists(basename + suffix):
            continue
        try:
            image = Image.open(basename + suffix)
            if image.size[0] < min_image_size[0] or image.size[1] < min_image_size[1]:
                raise ValueError("Image is too small")
            return image.convert("RGB")
        except Exception as e:
            print(e, end=" - ")
            break
    print(f"{basename} error")
    return Image.open(default_image)


class NpyWriter:
    """
    An object to facilitate writing numerical data to disk, without
    the need for holding the whole data in memory at once at any point
    in time.

    Example usage:

        writer = NpyWriter('bigdata.npy')
        writer.append(np.array([1,2,3]))
        writer.append(np.array([4,5,6]))
        writer.close()

        np.load('bigdata.npy')
        > array([[1, 2, 3],
        >        [4, 5, 6]])

    Notes:

     - THE close() METHOD *MUST* BE CALLED otherwise the .npy file
       will be unreadable
     - the shape and type of the elements that get appended to the
       writer must be consistent, or a RuntimeError will occur
    """

    def __init__(self, output_fpath):
        self.output_fpath = output_fpath
        self.output_file = open(self.output_fpath, "wb")
        for _ in range(128):
            self.output_file.write(c_uint8(10))
        self.item_shape = None
        self.item_dtype = None
        self.n_items = 0

    @staticmethod
    def is_allowed_type(item):
        if type(item) != np.ndarray:
            return False
        if item.dtype.type in [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
            np.float128,
        ]:
            return True
        return False

    def append(self, item):
        # check item type is a scalar o a numeric numpy array
        if not self.is_allowed_type(item):
            raise RuntimeError(
                "invalid type: must be a numeric type, either a scalar, a (nested) list, or a numpy array"
            )
        # is it the first item? this sets the shape ...
        if self.item_dtype is None:
            self.item_shape = item.shape
            self.item_dtype = item.dtype
        # ... otherwise check the shape to make sure it matches the previous one
        else:
            if item.shape != self.item_shape:
                raise RuntimeError(
                    "item shape %s, does not match previous shape %s" % (str(item.shape), str(self.item_shape))
                )
            if item.dtype != self.item_dtype:
                raise RuntimeError("item type %s does not match previous type %s" % (str(self.item_dtype), item.dtype))
        # - write binary blob to output in C order
        self.output_file.write(item.tobytes(order="C"))
        self.n_items += 1

    def extend(self, item):
        for i in item:
            self.append(i)

    @property
    def shape(self) -> tuple:
        return self.item_shape  # type: ignore

    def reshape(self, shape):
        shape = list(shape)
        if shape.count(-1) > 1:
            raise RuntimeError("invalid input shape %s" % (str(shape)))
        try:
            i = shape.index(-1)
            shape[i] = np.prod(self.item_shape) * -self.n_items // np.prod(shape)  # type: ignore
        except ValueError:
            pass
        if np.prod(shape) != np.prod(self.item_shape) * self.n_items:  # type: ignore
            raise RuntimeError("input shape %s does not match previous shape %s" % (str(shape), str(self.item_shape)))
        self.item_shape = tuple(shape[1:])
        self.n_items = shape[0]
        return self

    def close(self):
        # write header
        self.output_file.seek(0)
        self.output_file.write(c_uint8(147))
        self.output_file.write(bytes("NUMPY", "utf-8"))
        self.output_file.write(c_uint8(1))
        self.output_file.write(c_uint8(0))
        self.output_file.write(c_uint8(118))  # uint16(118) in little endian
        self.output_file.write(c_uint8(0))  #
        total_shape = tuple([self.n_items] + list(self.item_shape))  # type: ignore
        header = "{'descr': '%s', 'fortran_order': False, 'shape': %s}" % (
            self.item_dtype.descr[0][1],  # type: ignore
            str(total_shape),
        )
        self.output_file.write(bytes(header, "utf-8"))
        self.output_file.close()


def main():
    pass


if __name__ == "__main__":
    main()
