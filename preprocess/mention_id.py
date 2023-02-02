# -*- coding: utf-8 -*-
# @Date    : 2023-01-30 10:32:16
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import json, os
import numpy as np

mention_path = "/home/data_91_c/xsy/mel-dataset/wikimel/mentions/WIKIMEL_%s.json"
output_path = "."


def main():
    for type in ["train", "valid", "test"]:
        with open(mention_path % type, "r") as f:
            data = json.load(f)
        res = np.array([int(s.split("-")[0]) for s in data.keys()], dtype=np.int32)
        np.save(os.path.join(output_path, "mention-id_%s.npy" % type), res)


if __name__ == "__main__":
    main()
